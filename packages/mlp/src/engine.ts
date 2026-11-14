/**
 * GPU host for the portable MLP arms. Owns weights, optimizer state and scratch; encodes training steps and
 * inference into a caller's command encoder. Record counts come from a GPU buffer (u32 at `countWord`), so a frame
 * needs no readback (CLAUDE.md working rules).
 *
 * Usage:
 *   const mlp = await MlpEngine.create(device, { precision: 'f32', capacity: 16384, trainHooks, inferHooks });
 *   mlp.uploadWeights(initWeights(DEFAULT_SHAPE, 1));
 *   mlp.encodeTrainStep(encoder, countBuffer, 0, trainBindGroup);  // trainBindGroup: group 1 of mlp.trainLayout
 *   mlp.encodeInfer(encoder, countBuffer, 1, inferBindGroup);
 */

import { DEFAULT_SHAPE, DEFAULT_TRAIN, weightCount, type MlpShape, type TrainSettings } from './config.ts';
import { NN_WORKGROUP, adamWgsl, inferWgsl, prepareWgsl, rowStride, trainWgsl, weightGradientWgsl, type Precision } from './wgsl.ts';

export interface MlpEngineOptions {
  shape?: MlpShape;
  precision: Precision;
  /** Max training records per step (activation/delta scratch is sized from it). */
  capacity: number;
  /** Max inference records per call. */
  inferCapacity?: number;
  train?: TrainSettings;
  /** Inference reads the EMA weights. */
  useEma?: boolean;
  /** Group-1 WGSL for training (defines nnTrainLoad and its bindings). */
  trainHooks: string;
  /** Group-1 WGSL for inference (defines nnInferLoad / nnInferStore and their bindings). */
  inferHooks: string;
}

const PARAMS_BYTES = 48;

export class MlpEngine {
  readonly device: GPUDevice;
  readonly shape: MlpShape;
  readonly precision: Precision;
  readonly capacity: number;
  readonly inferCapacity: number;
  readonly weightCount: number;

  /** f32 master weights (Adam). */
  readonly master: GPUBuffer;
  /** `T` weights the training pass reads. */
  readonly trainWeights: GPUBuffer;
  /** `T` weights inference reads (EMA or raw). */
  readonly useWeights: GPUBuffer;
  readonly ema: GPUBuffer;
  readonly moments: GPUBuffer;
  readonly gradient: GPUBuffer;
  readonly activations: GPUBuffer;
  readonly deltas: GPUBuffer;
  /** Per-record loss of the last training step (f32 × capacity). */
  readonly loss: GPUBuffer;

  private readonly trainState: GPUBuffer;
  private readonly inferState: GPUBuffer;
  private readonly trainArgs: GPUBuffer;
  private readonly inferArgs: GPUBuffer;
  private readonly trainParams: GPUBuffer;
  private readonly inferParams: GPUBuffer;

  private readonly pipelines: Record<'prepare' | 'infer' | 'forwardBackward' | 'weightGradient' | 'adam' | 'sync', GPUComputePipeline>;
  private readonly groups: Record<'infer' | 'forwardBackward' | 'weightGradient' | 'adam' | 'sync', GPUBindGroup>;
  private readonly settings: TrainSettings;
  private readonly useEma: boolean;

  private constructor(device: GPUDevice, options: MlpEngineOptions, pipelines: MlpEngine['pipelines']) {
    this.device = device;
    this.shape = options.shape ?? DEFAULT_SHAPE;
    this.precision = options.precision;
    this.capacity = options.capacity;
    this.inferCapacity = options.inferCapacity ?? options.capacity;
    this.settings = options.train ?? DEFAULT_TRAIN;
    this.useEma = options.useEma ?? false;
    this.pipelines = pipelines;
    this.weightCount = weightCount(this.shape);

    const bpe = this.precision === 'f16' ? 2 : 4;
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    const buffer = (label: string, size: number, usage = storage) => device.createBuffer({ label: `nn ${label}`, size: Math.ceil(size / 16) * 16, usage });
    const layers = this.shape.hidden + 1;
    const rowBytes = rowStride(this.shape) * bpe;

    this.master = buffer('master', this.weightCount * 4);
    this.trainWeights = buffer('trainWeights', this.weightCount * bpe);
    this.useWeights = buffer('useWeights', this.weightCount * bpe);
    this.ema = buffer('ema', this.weightCount * 4);
    this.moments = buffer('moments', this.weightCount * 8);
    this.gradient = buffer('gradient', this.weightCount * 4);
    this.activations = buffer('activations', layers * this.capacity * rowBytes);
    this.deltas = buffer('deltas', layers * this.capacity * rowBytes);
    this.loss = buffer('loss', this.capacity * 4);
    this.trainState = buffer('trainState', 16);
    this.inferState = buffer('inferState', 16);
    this.trainArgs = buffer('trainArgs', 16, storage | GPUBufferUsage.INDIRECT);
    this.inferArgs = buffer('inferArgs', 16, storage | GPUBufferUsage.INDIRECT);
    this.trainParams = buffer('trainParams', PARAMS_BYTES, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.inferParams = buffer('inferParams', PARAMS_BYTES, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.resetOptimizer();

    // 'auto' layouts keep only the bindings a kernel uses, so nnSync gets just its own five.
    const bg = (pipeline: GPUComputePipeline, buffers: GPUBuffer[], first = 0) =>
      device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: buffers.map((b, i) => ({ binding: first + i, resource: { buffer: b } })) });
    this.groups = {
      infer: bg(pipelines.infer, [this.useWeights, this.inferState]), // nnInfer reads no params
      forwardBackward: bg(pipelines.forwardBackward, [this.trainWeights, this.trainState, this.trainParams, this.activations, this.deltas, this.loss]),
      weightGradient: bg(pipelines.weightGradient, [this.trainState, this.trainParams, this.activations, this.deltas, this.gradient]),
      adam: bg(pipelines.adam, [this.trainState, this.trainParams, this.gradient, this.master, this.moments, this.ema, this.trainWeights, this.useWeights]),
      sync: bg(pipelines.sync, [this.master, this.moments, this.ema, this.trainWeights, this.useWeights], 3),
    };
  }

  static async create(device: GPUDevice, options: MlpEngineOptions): Promise<MlpEngine> {
    const shape = options.shape ?? DEFAULT_SHAPE;
    const o = { shape, precision: options.precision };
    if (options.precision === 'f16' && !device.features.has('shader-f16')) throw new Error('MlpEngine: f16 needs the shader-f16 feature');
    const make = async (label: string, code: string, entryPoint: string) => {
      const module = device.createShaderModule({ label: `nn ${label}`, code });
      const info = await module.getCompilationInfo();
      const errors = info.messages.filter((m) => m.type === 'error');
      if (errors.length) throw new Error(`MlpEngine: ${label} failed to compile:\n${errors.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join('\n')}`);
      return device.createComputePipelineAsync({ label: `nn ${label}`, layout: 'auto', compute: { module, entryPoint } });
    };
    const adam = adamWgsl(o);
    const [prepare, infer, forwardBackward, weightGradient, adamPipeline, sync] = await Promise.all([
      make('prepare', prepareWgsl(o), 'nnPrepare'),
      make('infer', inferWgsl(o, options.inferHooks), 'nnInfer'),
      make('forwardBackward', trainWgsl(o, options.trainHooks), 'nnForwardBackward'),
      make('weightGradient', weightGradientWgsl(o), 'nnWeightGradient'),
      make('adam', adam, 'nnAdam'),
      make('sync', adam, 'nnSync'),
    ]);
    return new MlpEngine(device, options, { prepare, infer, forwardBackward, weightGradient, adam: adamPipeline, sync });
  }

  /** Layout for the caller's group-1 bind group (training hooks). */
  get trainLayout(): GPUBindGroupLayout {
    return this.pipelines.forwardBackward.getBindGroupLayout(1);
  }

  /** Layout for the caller's group-1 bind group (inference hooks). */
  get inferLayout(): GPUBindGroupLayout {
    return this.pipelines.infer.getBindGroupLayout(1);
  }

  private writeParams(buffer: GPUBuffer, capacity: number, countWord: number, train: boolean): void {
    const s = this.settings;
    const data = new ArrayBuffer(PARAMS_BYTES);
    const u = new Uint32Array(data);
    const f = new Float32Array(data);
    u[0] = capacity;
    u[1] = countWord;
    u[2] = train ? 1 : 0;
    u[3] = this.useEma ? 1 : 0;
    f[4] = s.learningRate;
    f[5] = s.beta1;
    f[6] = s.beta2;
    f[7] = s.epsilon;
    f[8] = s.emaDecay;
    f[9] = s.lossEpsilon;
    this.device.queue.writeBuffer(buffer, 0, data);
  }

  /** Resets Adam (step 0, zero moments on the next sync). */
  resetOptimizer(): void {
    const state = new ArrayBuffer(16);
    new Float32Array(state).set([0, 0, 1, 1]);
    new Uint32Array(state, 0, 2).set([0, 0]);
    this.device.queue.writeBuffer(this.trainState, 0, state);
    this.device.queue.writeBuffer(this.inferState, 0, state);
  }

  /** Uploads f32 weights (master), then derives the T copies, EMA and zero moments on the GPU. */
  uploadWeights(weights: Float32Array): void {
    if (weights.length !== this.weightCount) throw new Error(`uploadWeights: expected ${this.weightCount} weights, got ${weights.length}`);
    this.device.queue.writeBuffer(this.master, 0, weights.buffer, weights.byteOffset, weights.byteLength);
    this.resetOptimizer();
    const encoder = this.device.createCommandEncoder({ label: 'nn sync' });
    const pass = encoder.beginComputePass({ label: 'nn sync' });
    pass.setPipeline(this.pipelines.sync);
    pass.setBindGroup(0, this.groups.sync);
    pass.dispatchWorkgroups(Math.ceil(this.weightCount / NN_WORKGROUP));
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  private prepareGroup(kind: 'train' | 'infer', counts: GPUBuffer): GPUBindGroup {
    const cache = kind === 'train' ? this.trainPrepareGroups : this.inferPrepareGroups;
    let group = cache.get(counts);
    if (!group) {
      const buffers = kind === 'train' ? [this.trainState, this.trainParams, counts, this.trainArgs] : [this.inferState, this.inferParams, counts, this.inferArgs];
      group = this.device.createBindGroup({
        layout: this.pipelines.prepare.getBindGroupLayout(0),
        entries: buffers.map((b, binding) => ({ binding, resource: { buffer: b } })),
      });
      cache.set(counts, group);
    }
    return group;
  }
  private readonly trainPrepareGroups = new WeakMap<GPUBuffer, GPUBindGroup>();
  private readonly inferPrepareGroups = new WeakMap<GPUBuffer, GPUBindGroup>();

  /**
   * One training step: prepare (count, args, Adam step state) → forward/backward (indirect) → weight gradient →
   * Adam. `counts[countWord]` is the number of records; `hooks` is the caller's group-1 bind group.
   */
  encodeTrainStep(encoder: GPUCommandEncoder, counts: GPUBuffer, countWord: number, hooks: GPUBindGroup, timestampWrites?: GPUComputePassTimestampWrites): void {
    this.writeParams(this.trainParams, this.capacity, countWord, true);
    const pass = encoder.beginComputePass({ label: 'nn train', timestampWrites });
    pass.setPipeline(this.pipelines.prepare);
    pass.setBindGroup(0, this.prepareGroup('train', counts));
    pass.dispatchWorkgroups(1);
    pass.setPipeline(this.pipelines.forwardBackward);
    pass.setBindGroup(0, this.groups.forwardBackward);
    pass.setBindGroup(1, hooks);
    pass.dispatchWorkgroupsIndirect(this.trainArgs, 0);
    pass.setPipeline(this.pipelines.weightGradient);
    pass.setBindGroup(0, this.groups.weightGradient);
    pass.dispatchWorkgroups(Math.ceil(this.weightCount / NN_WORKGROUP));
    pass.setPipeline(this.pipelines.adam);
    pass.setBindGroup(0, this.groups.adam);
    pass.dispatchWorkgroups(Math.ceil(this.weightCount / NN_WORKGROUP));
    pass.end();
  }

  /** Inference over `counts[countWord]` records (clamped to inferCapacity). */
  encodeInfer(encoder: GPUCommandEncoder, counts: GPUBuffer, countWord: number, hooks: GPUBindGroup, timestampWrites?: GPUComputePassTimestampWrites): void {
    this.writeParams(this.inferParams, this.inferCapacity, countWord, false);
    const pass = encoder.beginComputePass({ label: 'nn infer', timestampWrites });
    pass.setPipeline(this.pipelines.prepare);
    pass.setBindGroup(0, this.prepareGroup('infer', counts));
    pass.dispatchWorkgroups(1);
    pass.setPipeline(this.pipelines.infer);
    pass.setBindGroup(0, this.groups.infer);
    pass.setBindGroup(1, hooks);
    pass.dispatchWorkgroupsIndirect(this.inferArgs, 0);
    pass.end();
  }

  destroy(): void {
    for (const b of [this.master, this.trainWeights, this.useWeights, this.ema, this.moments, this.gradient, this.activations, this.deltas, this.loss, this.trainState, this.inferState, this.trainArgs, this.inferArgs, this.trainParams, this.inferParams]) b.destroy();
  }
}
