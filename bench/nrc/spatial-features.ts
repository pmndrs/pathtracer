/** Trainable trilinear position features for the 64-input NRC decoder. */
import { header, type MlpEngine } from '@pathtracer/mlp';

const RESOLUTIONS = [4, 8, 16, 32, 64, 128] as const;
const CHANNELS = 2;
const GROUP_SIZE = 64;
const CORNERS = 8;
const GRADIENT_SCALE = 32768;

export interface SpatialFeatureSettings {
  log2Entries: number;
  learningRate: number;
}

export const DEFAULT_SPATIAL_FEATURES: SpatialFeatureSettings = { log2Entries: 12, learningRate: 0.01 };

export function spatialFeatureLayout(settings: SpatialFeatureSettings) {
  if (!Number.isInteger(settings.log2Entries) || settings.log2Entries < 8 || settings.log2Entries > 18) {
    throw new Error('Spatial features require log2Entries from 8 to 18');
  }
  if (!Number.isFinite(settings.learningRate) || settings.learningRate <= 0) {
    throw new Error('Spatial features require a positive finite learning rate');
  }
  const limit = 2 ** settings.log2Entries;
  const sizes = RESOLUTIONS.map((resolution) => Math.min((resolution + 1) ** 3, limit));
  const offsets = [0];
  for (const size of sizes) offsets.push(offsets[offsets.length - 1]! + size);
  return { sizes, offsets, vertices: offsets[RESOLUTIONS.length]! };
}

function gridWgsl(settings: SpatialFeatureSettings): string {
  const { offsets } = spatialFeatureLayout(settings);
  return /* wgsl */ `
const SPATIAL_LEVELS = ${RESOLUTIONS.length}u;
const SPATIAL_OFFSETS = array<u32, ${offsets.length}>(${offsets.map((n) => n + 'u').join(', ')});

fn nnSpatialResolution(level: u32) -> u32 { return 4u << level; }
fn nnSpatialIndex(level: u32, side: u32, xyz: vec3u) -> u32 {
  let begin = SPATIAL_OFFSETS[level];
  let count = SPATIAL_OFFSETS[level + 1u] - begin;
  if (count == side * side * side) { return begin + xyz.x + side * (xyz.y + side * xyz.z); }
  var h = (xyz.x * 73856093u) ^ (xyz.y * 19349663u) ^ (xyz.z * 83492791u);
  h = (h ^ (h >> 16u)) * 0x7feb352du;
  h = (h ^ (h >> 15u)) * 0x846ca68bu;
  return begin + ((h ^ (h >> 16u)) & (count - 1u));
}
fn nnSpatialCorner(c: u32) -> vec3u { return vec3u(c & 1u, (c >> 1u) & 1u, c >> 2u); }
fn nnSpatialWeight(t: vec3f, corner: vec3u) -> f32 {
  let w = select(1.0 - t, t, corner == vec3u(1u));
  return w.x * w.y * w.z;
}
fn nnSpatialPosition(position: vec3f, resolution: u32) -> vec3f {
  return (0.5 * position + 0.5) * f32(resolution);
}
`;
}

export function spatialEncodingWgsl(settings: SpatialFeatureSettings): string {
  return /* wgsl */ `
@group(1) @binding(3) var<storage, read> nnSpatialValues: array<vec2f>;
${gridWgsl(settings)}
fn nnEncodeSpatial(position: vec3f, out: ptr<function, array<f32, NN_IN>>) {
  for (var level = 0u; level < SPATIAL_LEVELS; level++) {
    let resolution = nnSpatialResolution(level);
    let p = nnSpatialPosition(position, resolution);
    let base = vec3u(floor(p));
    let t = fract(p);
    var value = vec2f(0.0);
    for (var c = 0u; c < ${CORNERS}u; c++) {
      let corner = nnSpatialCorner(c);
      let xyz = min(base + corner, vec3u(resolution));
      value += nnSpatialValues[nnSpatialIndex(level, resolution + 1u, xyz)] * nnSpatialWeight(t, corner);
    }
    (*out)[level * 2u] = value.x;
    (*out)[level * 2u + 1u] = value.y;
  }
}
`;
}

function optimizerWgsl(mlp: MlpEngine, settings: SpatialFeatureSettings, recordVec4: number, nrcCoordinates: string, recordIndex: string): string {
  const { vertices } = spatialFeatureLayout(settings);
  const width = mlp.shape.width;
  const spatialVec4s = RESOLUTIONS.length * CHANNELS / 4;
  // A record contributes total corner weight 1 per level, including corners which hash to the same slot.
  const clip = Math.min(32, 2 ** 30 / (mlp.capacity * GRADIENT_SCALE));
  return /* wgsl */ `
${header({ shape: mlp.shape, precision: mlp.precision })}
${gridWgsl(settings)}
${nrcCoordinates}
${recordIndex}
@group(0) @binding(0) var<storage, read> nnState: NnState;
@group(0) @binding(1) var<storage, read_write> gradients: array<atomic<i32>>;
@group(0) @binding(2) var<storage, read> weights: array<T4>;
@group(0) @binding(3) var<storage, read> deltas: array<T4>;
@group(0) @binding(4) var<storage, read> nrcSet: array<vec4f>;
@group(0) @binding(5) var<storage, read> nrcSetCount: array<u32>;
@group(0) @binding(6) var<uniform> nrcBox: NrcBox;
@group(0) @binding(7) var<storage, read_write> values: array<vec2f>;
@group(0) @binding(8) var<storage, read_write> moments: array<vec4f>;
@group(0) @binding(9) var<storage, read_write> steps: array<u32>;
var<workgroup> spatialWeights: array<vec4f, ${width * spatialVec4s}>;

@compute @workgroup_size(${GROUP_SIZE})
fn nnSpatialScatter(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) lid: u32) {
  for (var j = lid; j < ${width * spatialVec4s}u; j += ${GROUP_SIZE}u) {
    spatialWeights[j] = vec4f(weights[(j / ${spatialVec4s}u) * (NN_IN / 4u) + j % ${spatialVec4s}u]);
  }
  workgroupBarrier();
  let r = gid.x;
  if (r >= nnState.count) { return; }

  // The first layer's preactivation deltas already include the first ReLU derivative.
  var dx: array<vec4f, ${spatialVec4s}>;
  for (var j = 0u; j < NN_W; j++) {
    let d = f32(deltas[r * NN_S4 + j / 4u][j % 4u]);
    for (var k = 0u; k < ${spatialVec4s}u; k++) {
      dx[k] += spatialWeights[j * ${spatialVec4s}u + k] * d;
    }
  }
  let position = nrcNormalizePosition(nrcSet[nrcTrainRecordIndex(r) * ${recordVec4}u].xyz);
  for (var level = 0u; level < SPATIAL_LEVELS; level++) {
    let resolution = nnSpatialResolution(level);
    let p = nnSpatialPosition(position, resolution);
    let base = vec3u(floor(p));
    let t = fract(p);
    let offset = level % 2u * 2u;
    let d = clamp(vec2f(dx[level / 2u][offset], dx[level / 2u][offset + 1u]), vec2f(-${clip}), vec2f(${clip}));
    for (var c = 0u; c < ${CORNERS}u; c++) {
      let corner = nnSpatialCorner(c);
      let xyz = min(base + corner, vec3u(resolution));
      let g = vec2i(round(d * (nnSpatialWeight(t, corner) * ${GRADIENT_SCALE}.0)));
      if (any(g != vec2i(0))) {
        let index = nnSpatialIndex(level, resolution + 1u, xyz) * 2u;
        if (g.x != 0) { atomicAdd(&gradients[index], g.x); }
        if (g.y != 0) { atomicAdd(&gradients[index + 1u], g.y); }
      }
    }
  }
}

@compute @workgroup_size(${GROUP_SIZE})
fn nnSpatialAdam(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= ${vertices}u || nnState.count == 0u) { return; }
  let g = vec2i(atomicExchange(&gradients[i * 2u], 0), atomicExchange(&gradients[i * 2u + 1u], 0));
  if (all(g == vec2i(0))) { return; }
  let d = vec2f(g) / (f32(nnState.count) * ${GRADIENT_SCALE}.0);
  let previous = moments[i];
  let m = mix(d, previous.xy, vec2f(0.9));
  let v = mix(d * d, previous.zw, vec2f(0.99));
  moments[i] = vec4f(m, v);
  let step = steps[i] + 1u;
  steps[i] = step;
  let mHat = m / (1.0 - pow(0.9, f32(step)));
  let vHat = v / (1.0 - pow(0.99, f32(step)));
  values[i] -= ${settings.learningRate} * mHat / (sqrt(vHat) + 1e-8);
}
`;
}

export class SpatialFeatures {
  readonly settings: SpatialFeatureSettings;
  readonly values: GPUBuffer;
  private readonly mlp: MlpEngine;
  private readonly scatter: GPUComputePipeline;
  private readonly update: GPUComputePipeline;
  private readonly gradients: GPUBuffer;
  private readonly moments: GPUBuffer;
  private readonly steps: GPUBuffer;
  private readonly vertices: number;
  private readonly scatterGroup: GPUBindGroup;
  private readonly updateGroup: GPUBindGroup;

  private constructor(
    mlp: MlpEngine,
    settings: SpatialFeatureSettings,
    scatter: GPUComputePipeline,
    update: GPUComputePipeline,
    trainSet: GPUBuffer,
    trainCount: GPUBuffer,
    box: GPUBuffer,
  ) {
    this.mlp = mlp;
    this.settings = settings;
    this.scatter = scatter;
    this.update = update;
    const device = mlp.device;
    this.vertices = spatialFeatureLayout(settings).vertices;
    const buffer = (label: string, bytesPerVertex: number) => device.createBuffer({
      label, size: this.vertices * bytesPerVertex, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.values = buffer('NRC position features', 8);
    this.gradients = buffer('NRC position feature gradients', 8);
    this.moments = buffer('NRC position feature moments', 16);
    this.steps = buffer('NRC position feature update counts', 4);
    const bind = (pipeline: GPUComputePipeline, entries: [number, GPUBuffer][]) => device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0), entries: entries.map(([binding, resource]) => ({ binding, resource: { buffer: resource } })),
    });
    this.scatterGroup = bind(scatter, [
      [0, mlp.trainState], [1, this.gradients], [2, mlp.trainWeights], [3, mlp.deltas], [4, trainSet], [5, trainCount], [6, box],
    ]);
    this.updateGroup = bind(update, [[0, mlp.trainState], [1, this.gradients], [7, this.values], [8, this.moments], [9, this.steps]]);
  }

  static async create(
    mlp: MlpEngine, settings: SpatialFeatureSettings, trainSet: GPUBuffer, trainCount: GPUBuffer, box: GPUBuffer,
    recordVec4: number, coordinates: string, recordIndex: string,
  ): Promise<SpatialFeatures> {
    if (mlp.forwardKind === 'sgm') throw new Error('Spatial features require the portable training layout');
    const device = mlp.device;
    const module = device.createShaderModule({ label: 'NRC spatial feature optimizer', code: optimizerWgsl(mlp, settings, recordVec4, coordinates, recordIndex) });
    const make = (entryPoint: string) => device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint } });
    const [scatter, update] = await Promise.all([make('nnSpatialScatter'), make('nnSpatialAdam')]);
    return new SpatialFeatures(mlp, settings, scatter, update, trainSet, trainCount, box);
  }

  /** Run before the decoder's optimizer changes the first layer. */
  encodeUpdate = (pass: GPUComputePassEncoder): void => {
    pass.setPipeline(this.scatter);
    pass.setBindGroup(0, this.scatterGroup);
    pass.dispatchWorkgroups(Math.ceil(this.mlp.capacity / GROUP_SIZE));
    pass.setPipeline(this.update);
    pass.setBindGroup(0, this.updateGroup);
    pass.dispatchWorkgroups(Math.ceil(this.vertices / GROUP_SIZE));
  };

  reset(seed: number): void {
    const values = new Float32Array(this.vertices * CHANNELS);
    let x = (seed | 0) || 1;
    for (let i = 0; i < values.length; i++) {
      x ^= x << 13;
      x ^= x >>> 17;
      x ^= x << 5;
      values[i] = ((x >>> 0) / 2 ** 32 - 0.5) * 0.002;
    }
    const device = this.mlp.device;
    device.queue.writeBuffer(this.values, 0, values);
    const encoder = device.createCommandEncoder({ label: 'NRC reset position optimizer' });
    for (const buffer of [this.gradients, this.moments, this.steps]) encoder.clearBuffer(buffer);
    device.queue.submit([encoder.finish()]);
  }

  get bytes(): number {
    return this.values.size + this.gradients.size + this.moments.size + this.steps.size;
  }

  /** Upper bound on the number of 32-bit atomic operands emitted per training record. */
  get gradientAddsPerRecord(): number { return RESOLUTIONS.length * CORNERS * CHANNELS; }

  async readUpdatedVertices(): Promise<number> {
    const device = this.mlp.device;
    const staging = device.createBuffer({ size: this.steps.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    try {
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(this.steps, 0, staging, 0, this.steps.size);
      device.queue.submit([encoder.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      return new Uint32Array(staging.getMappedRange()).reduce((count, step) => count + Number(step !== 0), 0);
    } finally {
      staging.destroy();
    }
  }

  destroy(): void {
    for (const buffer of [this.values, this.gradients, this.moments, this.steps]) buffer.destroy();
  }
}

/** Distinct sampled lattice corners and their aliases in one frame, measured outside timing. */
export function spatialCollisions(worldPositions: Float32Array, stride: number, box: Float32Array, settings: SpatialFeatureSettings) {
  const { sizes } = spatialFeatureLayout(settings);
  return RESOLUTIONS.map((resolution, level) => {
    const side = resolution + 1;
    const size = sizes[level]!;
    const vertices = new Set<number>();
    const slots = new Set<number>();
    for (let i = 0; i < worldPositions.length; i += stride) {
      const p = [0, 1, 2].map((axis) => Math.fround(resolution * Math.min(1, Math.max(0, Math.fround((worldPositions[i + axis]! - box[axis]!) * box[4 + axis]!) * 0.5))));
      const base = p.map(Math.floor);
      for (let c = 0; c < CORNERS; c++) {
        const xyz = base.map((v, axis) => Math.min(resolution, v + ((c >>> axis) & 1)));
        const [x, y, z] = xyz as [number, number, number];
        const linear = x + side * (y + side * z);
        if (vertices.has(linear)) continue;
        vertices.add(linear);
        if (size === side ** 3) {
          slots.add(linear);
        } else {
          let h = Math.imul(x, 73856093) ^ Math.imul(y, 19349663) ^ Math.imul(z, 83492791);
          h = Math.imul(h ^ (h >>> 16), 0x7feb352d);
          h = Math.imul(h ^ (h >>> 15), 0x846ca68b);
          slots.add((h ^ (h >>> 16)) & (size - 1));
        }
      }
    }
    return { resolution, vertices: vertices.size, aliases: vertices.size - slots.size };
  });
}
