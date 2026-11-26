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
import { DEFAULT_GRADIENT, gradientPartialBytes, gradientReduceWgsl, gradientTileWgsl, type GradientKind, type GradientOptions } from './wgsl-gradient.ts';
import { DEFAULT_FORWARD, defaultForwardRecords, forwardWorkgroupBytes, inferTiledWgsl, inferTiledWorkgroupBytes, trainTiledWgsl, type ForwardKind, type ForwardOptions } from './wgsl-forward.ts';
import { inferSgmWgsl, packSgmWgsl, sgmWeightCount, sgmWorkgroupBytes, type SgmOptions } from './wgsl-sgm.ts';
import { sgmTrainWorkgroupBytes, trainSgmWgsl } from './wgsl-sgm-train.ts';
import { gradientSgmWgsl, type SgmGradientOptions } from './wgsl-gradient-sgm.ts';

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
  /**
   * Inference on subgroup matrices (N3a) instead of the portable kernel. Training stays portable. The operand type
   * must equal `precision` (emulation: f32).
   */
  sgm?: Omit<SgmOptions, 'shape'>;
  /** Portable inference (no `sgm`): the tiled kernel (default; N1.4, `inferTiledWgsl`: 8.6× the per-record kernel in f16 on the M5 Pro) or one thread per record ('perRecord'). */
  infer?: 'perRecord' | 'tiled';
  /** Tiled inference: records per workgroup (default `defaultForwardRecords(precision)`). */
  inferOptions?: Partial<ForwardOptions>;
  /** Weight-gradient arm: one thread per weight looping over all records, or the tiled split-K pass (N1.2). Default 'tiled' (N1.2 A/B: f16 step 6.9 → 2.7 ms on the M5 Pro). */
  gradient?: GradientKind;
  /** Tiled and sgm arms: `splits` record ranges per layer (`chunk`: tiled only, records staged per iteration). */
  gradientOptions?: Partial<GradientOptions>;
  /**
   * Subgroup-matrix weight gradient (`gradient: 'sgm'`, N2r-d): an f32-result config with the MLP's operand type. Needs
   * `forward: 'sgm'` in the 'blocked' layout (it writes the zero deltas the whole K-record steps read) and a `capacity`
   * that is a multiple of the config's K.
   */
  gradientSgm?: SgmGradientOptions;
  /** Forward-backward arm: one thread per record ('perRecord') or the tiled workgroup-per-R-records kernel (N1.2b). Default 'tiled' (N1.2b A/B: forward-backward 2.49 → 0.31 ms f16, 3.65 → 0.50 ms f32, M5 Pro). */
  forward?: ForwardKind;
  /**
   * Subgroup-matrix forward-backward (`forward: 'sgm'`, N3b): config, records per workgroup, subgroups, `emulate`.
   * The operand type must equal `precision` (emulation: f32). The engine repacks the train weights into the padded sgm
   * layout at the start of every step.
   */
  trainSgm?: Omit<SgmOptions, 'shape'>;
  /** Tiled forward arm only: `records` per workgroup (default `defaultForwardRecords(precision)`: 64 f16, 48 f32). */
  forwardOptions?: Partial<ForwardOptions>;
}

/** Per-stage timestamps for one training step (3 passes, 6 slots from `first`: begin/end of prepare+forward-backward, gradient, Adam). */
export interface TrainStageTimestamps {
  querySet: GPUQuerySet;
  first: number;
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

  /** Prepared batch length and optimizer step, available to input-gradient consumers before Adam. */
  readonly trainState: GPUBuffer;
  private readonly inferState: GPUBuffer;
  private readonly trainArgs: GPUBuffer;
  private readonly inferArgs: GPUBuffer;
  private readonly trainParams: GPUBuffer;
  private readonly inferParams: GPUBuffer;

  /** Padded inference weights for the sgm arm (null on the portable arm). */
  readonly sgmWeights: GPUBuffer | null;
  /** Records per workgroup of the inference kernel. */
  readonly inferGroupRecords: number;
  readonly inferBackend: 'portable' | 'portable-tiled' | 'sgm' | 'sgm-emulated';

  private readonly pipelines: Record<'prepare' | 'infer' | 'forwardBackward' | 'weightGradient' | 'adam' | 'sync', GPUComputePipeline> & { pack?: GPUComputePipeline; packTrain?: GPUComputePipeline; gradientReduce?: GPUComputePipeline };
  private readonly groups: Record<'infer' | 'forwardBackward' | 'weightGradient' | 'adam' | 'sync', GPUBindGroup> & { pack?: GPUBindGroup; packTrain?: GPUBindGroup; gradientReduce?: GPUBindGroup };
  /** Padded train weights for the sgm forward-backward (null otherwise). */
  readonly trainSgmWeights: GPUBuffer | null;
  private readonly trainSgmCount: number;
  private readonly trainGroupRecords: number;
  readonly gradientKind: GradientKind;
  readonly gradientOptions: GradientOptions;
  readonly forwardKind: ForwardKind;
  readonly forwardOptions: ForwardOptions;
  /** Per-split partial dW (tiled arm only). */
  readonly gradientPartials: GPUBuffer | null;
  private readonly sgmCount: number;
  private settings: TrainSettings;
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
    this.gradientKind = options.gradient ?? 'tiled';
    this.gradientOptions = { ...DEFAULT_GRADIENT, ...options.gradientOptions };
    this.forwardKind = options.forward ?? 'tiled';
    this.forwardOptions = { ...DEFAULT_FORWARD, records: defaultForwardRecords(options.precision), ...options.forwardOptions };

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
    const sgm = options.sgm;
    const tiledInfer = !sgm && (options.infer ?? 'tiled') === 'tiled';
    this.inferBackend = sgm ? (sgm.emulate ? 'sgm-emulated' : 'sgm') : tiledInfer ? 'portable-tiled' : 'portable';
    this.inferGroupRecords = sgm ? sgm.records : tiledInfer ? (options.inferOptions?.records ?? defaultForwardRecords(options.precision)) : NN_WORKGROUP;
    this.sgmCount = sgm ? sgmWeightCount(this.shape, sgm.config) : 0;
    this.sgmWeights = sgm ? buffer('sgmWeights', this.sgmCount * bpe) : null;
    this.gradientPartials = this.gradientKind === 'tiled' || this.gradientKind === 'sgm' ? buffer('gradientPartials', gradientPartialBytes({ shape: this.shape, precision: this.precision }, this.gradientOptions)) : null;
    const trainSgm = options.trainSgm;
    if (this.forwardKind === 'sgm' && !trainSgm) throw new Error("MlpEngine: forward 'sgm' needs trainSgm");
    this.trainSgmCount = trainSgm && this.forwardKind === 'sgm' ? sgmWeightCount(this.shape, trainSgm.config) : 0;
    this.trainSgmWeights = this.trainSgmCount ? buffer('trainSgmWeights', this.trainSgmCount * bpe) : null;
    this.trainGroupRecords = this.forwardKind === 'sgm' && trainSgm ? trainSgm.records : this.forwardKind === 'tiled' ? this.forwardOptions.records : NN_WORKGROUP;
    this.resetOptimizer();

    // 'auto' layouts keep only the bindings a kernel uses, so nnSync gets just its own five.
    const bg = (pipeline: GPUComputePipeline, buffers: GPUBuffer[], first = 0) =>
      device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: buffers.map((b, i) => ({ binding: first + i, resource: { buffer: b } })) });
    this.groups = {
      infer: bg(pipelines.infer, [this.sgmWeights ?? this.useWeights, this.inferState]), // the infer kernels read no params
      forwardBackward: bg(pipelines.forwardBackward, [this.trainSgmWeights ?? this.trainWeights, this.trainState, this.trainParams, this.activations, this.deltas, this.loss]),
      weightGradient: this.gradientPartials
        ? bg(pipelines.weightGradient, [this.trainState, this.trainParams, this.activations, this.deltas, this.gradientPartials])
        : bg(pipelines.weightGradient, [this.trainState, this.trainParams, this.activations, this.deltas, this.gradient]),
      adam: bg(pipelines.adam, [this.trainState, this.trainParams, this.gradient, this.master, this.moments, this.ema, this.trainWeights, this.useWeights]),
      sync: bg(pipelines.sync, [this.master, this.moments, this.ema, this.trainWeights, this.useWeights], 3),
      ...(pipelines.gradientReduce && this.gradientPartials
        ? {
            gradientReduce: device.createBindGroup({
              layout: pipelines.gradientReduce.getBindGroupLayout(0),
              entries: [{ binding: 0, resource: { buffer: this.trainState } }, { binding: 2, resource: { buffer: this.gradientPartials } }, { binding: 3, resource: { buffer: this.gradient } }],
            }),
          }
        : {}),
      ...(pipelines.pack && this.sgmWeights ? { pack: bg(pipelines.pack, [this.useWeights, this.sgmWeights]) } : {}),
      ...(pipelines.packTrain && this.trainSgmWeights ? { packTrain: bg(pipelines.packTrain, [this.trainWeights, this.trainSgmWeights]) } : {}),
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
    const sgm = options.sgm ? { ...options.sgm, shape } : null;
    if (sgm) {
      const operand = sgm.emulate ? 'f32' : sgm.config.componentType;
      if (operand !== options.precision) throw new Error(`MlpEngine: sgm operand type ${operand} must equal precision ${options.precision}`);
      const need = sgmWorkgroupBytes(sgm);
      if (need > device.limits.maxComputeWorkgroupStorageSize) throw new Error(`MlpEngine: sgm needs ${need} B of workgroup memory, device allows ${device.limits.maxComputeWorkgroupStorageSize}; lower sgm.records`);
    }
    const forwardOptions = { ...DEFAULT_FORWARD, records: defaultForwardRecords(options.precision), ...options.forwardOptions };
    const forwardKind = options.forward ?? 'tiled';
    const tiledForward = forwardKind === 'tiled';
    const trainSgm = forwardKind === 'sgm' && options.trainSgm ? { ...options.trainSgm, shape } : null;
    if (forwardKind === 'sgm' && !trainSgm) throw new Error("MlpEngine: forward 'sgm' needs trainSgm");
    if (trainSgm) {
      const operand = trainSgm.emulate ? 'f32' : trainSgm.config.componentType;
      if (operand !== options.precision) throw new Error(`MlpEngine: trainSgm operand type ${operand} must equal precision ${options.precision}`);
      const need = sgmTrainWorkgroupBytes(trainSgm);
      if (need > device.limits.maxComputeWorkgroupStorageSize) throw new Error(`MlpEngine: sgm training needs ${need} B of workgroup memory, device allows ${device.limits.maxComputeWorkgroupStorageSize}; lower trainSgm.records`);
    }
    const tiledInfer = !sgm && (options.infer ?? 'tiled') === 'tiled';
    const inferOptions = { ...DEFAULT_FORWARD, records: defaultForwardRecords(options.precision), ...options.inferOptions };
    if (tiledInfer) {
      const need = inferTiledWorkgroupBytes(o, inferOptions);
      if (need > device.limits.maxComputeWorkgroupStorageSize) throw new Error(`MlpEngine: tiled inference needs ${need} B of workgroup memory, device allows ${device.limits.maxComputeWorkgroupStorageSize}; lower inferOptions.records`);
    }
    if (tiledForward) {
      const need = forwardWorkgroupBytes(o, forwardOptions);
      if (need > device.limits.maxComputeWorkgroupStorageSize) throw new Error(`MlpEngine: tiled forward needs ${need} B of workgroup memory, device allows ${device.limits.maxComputeWorkgroupStorageSize}; lower forwardOptions.records`);
    }
    const gradientKind = options.gradient ?? 'tiled';
    const tiled = gradientKind === 'tiled';
    const sgmGradient = gradientKind === 'sgm';
    const gradientOptions = { ...DEFAULT_GRADIENT, ...options.gradientOptions };
    if (sgmGradient) {
      if (!options.gradientSgm) throw new Error("MlpEngine: gradient 'sgm' needs gradientSgm");
      if (!trainSgm || trainSgm.layout !== 'blocked' || trainSgm.emulate) throw new Error("MlpEngine: gradient 'sgm' needs forward 'sgm' with layout 'blocked' (zero deltas up to capacity)");
      if (options.capacity % options.gradientSgm.config.K) throw new Error(`MlpEngine: gradient 'sgm' needs capacity to be a multiple of K=${options.gradientSgm.config.K}`);
    }
    const [prepare, infer, forwardBackward, weightGradient, adamPipeline, sync, pack, gradientReduce, packTrain] = await Promise.all([
      make('prepare', prepareWgsl(o), 'nnPrepare'),
      sgm ? make('inferSgm', inferSgmWgsl(sgm, options.inferHooks), 'nnInferSgm') : make('infer', tiledInfer ? inferTiledWgsl(o, options.inferHooks, inferOptions) : inferWgsl(o, options.inferHooks), 'nnInfer'),
      make('forwardBackward', trainSgm ? trainSgmWgsl(trainSgm, options.trainHooks) : tiledForward ? trainTiledWgsl(o, options.trainHooks, forwardOptions) : trainWgsl(o, options.trainHooks), 'nnForwardBackward'),
      sgmGradient ? make('gradientSgm', gradientSgmWgsl(o, gradientOptions, options.gradientSgm as SgmGradientOptions), 'nnGradientSgm') : tiled ? make('gradientTile', gradientTileWgsl(o, gradientOptions), 'nnGradientTile') : make('weightGradient', weightGradientWgsl(o), 'nnWeightGradient'),
      make('adam', adam, 'nnAdam'),
      make('sync', adam, 'nnSync'),
      sgm ? make('packSgm', packSgmWgsl(sgm), 'nnPackSgm') : Promise.resolve(undefined),
      tiled || sgmGradient ? make('gradientReduce', gradientReduceWgsl(o, gradientOptions), 'nnGradientReduce') : Promise.resolve(undefined),
      trainSgm ? make('packTrainSgm', packSgmWgsl(trainSgm), 'nnPackSgm') : Promise.resolve(undefined),
    ]);
    return new MlpEngine(device, options, { prepare, infer, forwardBackward, weightGradient, adam: adamPipeline, sync, ...(pack ? { pack } : {}), ...(packTrain ? { packTrain } : {}), ...(gradientReduce ? { gradientReduce } : {}) });
  }

  /** Layout for the caller's group-1 bind group (training hooks). */
  get trainLayout(): GPUBindGroupLayout {
    return this.pipelines.forwardBackward.getBindGroupLayout(1);
  }

  /** Layout for the caller's group-1 bind group (inference hooks). */
  get inferLayout(): GPUBindGroupLayout {
    return this.pipelines.infer.getBindGroupLayout(1);
  }

  private writeParams(buffer: GPUBuffer, capacity: number, countWord: number, train: boolean, groupRecords: number): void {
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
    u[10] = groupRecords;
    u[11] = s.emaWarmup ? 1 : 0;
    this.device.queue.writeBuffer(buffer, 0, data);
  }

  /** Changes the learning rate from the next encoded training step on (it is part of the per-call params: a submit sees the last value set). */
  setLearningRate(learningRate: number): void {
    this.settings = { ...this.settings, learningRate };
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

  /**
   * Reads the f32 weights back: the trained `master` copy or the `ema` copy inference uses. A readback (copy + `mapAsync`): for saving a
   * network, never inside a timed region.
   */
  async readWeights(which: 'master' | 'ema' = 'master'): Promise<Float32Array> {
    const bytes = this.weightCount * 4;
    const staging = this.device.createBuffer({ label: 'nn weights readback', size: Math.ceil(bytes / 16) * 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = this.device.createCommandEncoder({ label: 'nn weights readback' });
    encoder.copyBufferToBuffer(which === 'ema' ? this.ema : this.master, 0, staging, 0, staging.size);
    this.device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const weights = new Float32Array(staging.getMappedRange(0, bytes).slice(0));
    staging.unmap();
    staging.destroy();
    return weights;
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
   * An optional input optimizer runs after backpropagation while the forward weights still match the deltas.
   */
  encodeTrainStep(encoder: GPUCommandEncoder, counts: GPUBuffer, countWord: number, hooks: GPUBindGroup, timestampWrites?: GPUComputePassTimestampWrites, stages?: TrainStageTimestamps, updateInputs?: (pass: GPUComputePassEncoder) => void): void {
    this.writeParams(this.trainParams, this.capacity, countWord, true, this.trainGroupRecords);
    const stage = (name: string, n: number): GPUComputePassDescriptor =>
      stages ? { label: `nn train ${name}`, timestampWrites: { querySet: stages.querySet, beginningOfPassWriteIndex: stages.first + 2 * n, endOfPassWriteIndex: stages.first + 2 * n + 1 } } : {};
    // Without `stages` everything is one pass (the production shape); with it, three passes so each stage is timed.
    let pass = encoder.beginComputePass(stages ? stage('forward-backward', 0) : { label: 'nn train', timestampWrites });
    if (this.pipelines.packTrain && this.groups.packTrain) {
      // Repack the current train weights into the padded sgm layout (~21k threads, same pass).
      pass.setPipeline(this.pipelines.packTrain);
      pass.setBindGroup(0, this.groups.packTrain);
      pass.dispatchWorkgroups(Math.ceil(this.trainSgmCount / 64));
    }
    pass.setPipeline(this.pipelines.prepare);
    pass.setBindGroup(0, this.prepareGroup('train', counts));
    pass.dispatchWorkgroups(1);
    pass.setPipeline(this.pipelines.forwardBackward);
    pass.setBindGroup(0, this.groups.forwardBackward);
    pass.setBindGroup(1, hooks);
    pass.dispatchWorkgroupsIndirect(this.trainArgs, 0);
    if (stages) {
      pass.end();
      pass = encoder.beginComputePass(stage('gradient', 1));
    }
    pass.setPipeline(this.pipelines.weightGradient);
    pass.setBindGroup(0, this.groups.weightGradient);
    if (this.pipelines.gradientReduce && this.groups.gradientReduce) {
      pass.dispatchWorkgroups(this.gradientOptions.splits, this.shape.hidden + 1);
      pass.setPipeline(this.pipelines.gradientReduce);
      pass.setBindGroup(0, this.groups.gradientReduce);
    }
    pass.dispatchWorkgroups(Math.ceil(this.weightCount / NN_WORKGROUP));
    updateInputs?.(pass);
    if (stages) {
      pass.end();
      pass = encoder.beginComputePass(stage('adam', 2));
    }
    pass.setPipeline(this.pipelines.adam);
    pass.setBindGroup(0, this.groups.adam);
    pass.dispatchWorkgroups(Math.ceil(this.weightCount / NN_WORKGROUP));
    pass.end();
  }

  /** Inference over `counts[countWord]` records (clamped to inferCapacity). */
  encodeInfer(encoder: GPUCommandEncoder, counts: GPUBuffer, countWord: number, hooks: GPUBindGroup, timestampWrites?: GPUComputePassTimestampWrites): void {
    this.writeParams(this.inferParams, this.inferCapacity, countWord, false, this.inferGroupRecords);
    const pass = encoder.beginComputePass({ label: `nn infer (${this.inferBackend})`, timestampWrites });
    if (this.pipelines.pack && this.groups.pack) {
      // Repack the current inference weights into the padded sgm layout (~21k threads; part of the timed pass).
      pass.setPipeline(this.pipelines.pack);
      pass.setBindGroup(0, this.groups.pack);
      pass.dispatchWorkgroups(Math.ceil(this.sgmCount / 64));
    }
    pass.setPipeline(this.pipelines.prepare);
    pass.setBindGroup(0, this.prepareGroup('infer', counts));
    pass.dispatchWorkgroups(1);
    pass.setPipeline(this.pipelines.infer);
    pass.setBindGroup(0, this.groups.infer);
    pass.setBindGroup(1, hooks);
    pass.dispatchWorkgroupsIndirect(this.inferArgs, 0);
    pass.end();
  }

  private get buffers(): (GPUBuffer | null)[] {
    return [this.master, this.trainWeights, this.useWeights, this.ema, this.moments, this.gradient, this.activations, this.deltas, this.loss,
      this.trainState, this.inferState, this.trainArgs, this.inferArgs, this.trainParams, this.inferParams, this.gradientPartials, this.sgmWeights, this.trainSgmWeights];
  }

  /** Allocated GPU buffer sizes, including optimizer and training workspace. */
  get bufferBytes(): number {
    return this.buffers.reduce((sum, buffer) => sum + (buffer?.size ?? 0), 0);
  }

  destroy(): void {
    for (const buffer of this.buffers) buffer?.destroy();
  }
}
