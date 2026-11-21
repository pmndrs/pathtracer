/**
 * Draws one real-time frame of the path tracer (`integrator.frame`, `mode: 'realtime'`): the beauty or any of its frame planes, optionally
 * averaged over the frames since the camera last moved. A compute pass picks the value per pixel and adds it to a running sum; a fullscreen
 * triangle divides, tone maps radiance views and encodes sRGB. No readback.
 *
 * Plane layout (docs: Path tracer > Real time): 7 planes of width * height vec4f, plane p at element p * width * height.
 *   0 diffuse / albedo   1 specular / factor   2 normal, viewZ   3 motion, roughness   4 albedo   5 specular factor   6 emission
 */

/** What to show. Radiance views are tone mapped; data views are drawn as values. */
export const FRAME_VIEWS = {
  Beauty: 0,
  'Remodulated (diffuse x albedo + specular x factor + emission)': 1,
  'Diffuse (demodulated)': 2,
  'Specular (demodulated)': 3,
  Emission: 4,
  Albedo: 5,
  'Specular factor': 6,
  Normal: 7,
  'View depth': 8,
  'Motion (x 20)': 9,
  Roughness: 10,
} as const;
export type FrameViewMode = (typeof FRAME_VIEWS)[keyof typeof FRAME_VIEWS];

const RADIANCE_VIEWS = new Set<number>([0, 1, 2, 3, 4]);

const COMMON = /* wgsl */ `
struct View { width: u32, height: u32, mode: u32, count: u32, exposure: f32, radiance: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> view: View;
`;

const PICK = /* wgsl */ `${COMMON}
@group(0) @binding(1) var<storage, read> beauty: array<vec4f>;
@group(0) @binding(2) var<storage, read> planes: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> sums: array<vec4f>;

fn plane(p: u32, i: u32) -> vec4f { return planes[p * view.width * view.height + i]; }

fn pick(i: u32) -> vec3f {
  switch view.mode {
    case 0u: { return beauty[i].rgb; }
    case 1u: { return plane(0u, i).rgb * plane(4u, i).rgb + plane(1u, i).rgb * plane(5u, i).rgb + plane(6u, i).rgb; }
    case 2u: { return plane(0u, i).rgb; }
    case 3u: { return plane(1u, i).rgb; }
    case 4u: { return plane(6u, i).rgb; }
    case 5u: { return plane(4u, i).rgb; }
    case 6u: { return plane(5u, i).rgb; }
    case 7u: { let n = plane(2u, i).rgb; return select(n * 0.5 + 0.5, vec3f(0.0), dot(n, n) == 0.0); }
    case 8u: { let z = -plane(2u, i).a; return select(vec3f(exp(-0.15 * z)), vec3f(0.0), z <= 0.0); }
    case 9u: { let m = plane(3u, i).rg * 20.0; return vec3f(max(m.x, 0.0) + max(-m.y, 0.0) * 0.5, max(m.y, 0.0) + max(-m.x, 0.0) * 0.5, abs(m.x) * 0.25); }
    default: { return vec3f(plane(3u, i).b); }
  }
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= view.width || id.y >= view.height) { return; }
  let i = id.y * view.width + id.x;
  var v = pick(i);
  // NaN and Inf stay out of the running sum (a real-time frame should have none; they would stick until the next reset).
  if (any(v != v) || any(abs(v) > vec3f(3e38))) { v = vec3f(0.0); }
  let prev = select(sums[i], vec4f(0.0), view.count <= 1u);
  sums[i] = prev + vec4f(v, 1.0);
}
`;

const DRAW = /* wgsl */ `${COMMON}
@group(0) @binding(1) var<storage, read> sums: array<vec4f>;

@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

// Krzysztof Narkowicz's 2016 fit of the ACES RRT + ODT (THIRD_PARTY.md).
fn aces(x: vec3f) -> vec3f {
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3f(0.0), vec3f(1.0));
}
fn srgb(c: vec3f) -> vec3f {
  return select(1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055, 12.92 * c, c <= vec3f(0.0031308));
}

@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let p = vec2u(pos.xy);
  let s = sums[p.y * view.width + p.x];
  var c = max(select(vec3f(0.0), s.rgb / s.w, s.w > 0.0), vec3f(0.0));
  if (view.radiance == 1u) { c = aces(c * view.exposure); } else { c = clamp(c, vec3f(0.0), vec3f(1.0)); }
  return vec4f(srgb(c), 1.0);
}
`;

export class FrameView {
  mode: FrameViewMode = 0;
  exposure = 1;
  /** Average the frames since the last `restart` (a still camera converges; a moving one shows the single frame). */
  accumulate = true;
  /** Frames in the running sum. */
  count = 0;
  /** Frames drawn since the page started (checks wait on this). */
  rendered = 0;
  private readonly device: GPUDevice;
  private readonly pickPipeline: GPUComputePipeline;
  private readonly drawPipeline: GPURenderPipeline;
  private readonly uniform: GPUBuffer;
  private sums: GPUBuffer | null = null;
  private pickGroup: GPUBindGroup | null = null;
  private drawGroup: GPUBindGroup | null = null;
  private key: unknown[] = [];

  constructor(device: GPUDevice, format: GPUTextureFormat) {
    this.device = device;
    this.pickPipeline = device.createComputePipeline({ label: 'frame view pick', layout: 'auto', compute: { module: device.createShaderModule({ label: 'frame view pick', code: PICK }), entryPoint: 'main' } });
    const draw = device.createShaderModule({ label: 'frame view draw', code: DRAW });
    this.drawPipeline = device.createRenderPipeline({ label: 'frame view draw', layout: 'auto', vertex: { module: draw, entryPoint: 'vs' }, fragment: { module: draw, entryPoint: 'fs', targets: [{ format }] } });
    this.uniform = device.createBuffer({ label: 'frame view', size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }

  /** Starts the average again (camera moved, view or kernel changed). */
  restart(): void {
    this.count = 0;
  }

  /** Records the pick pass and the draw into `target` for a frame whose buffers are `frame` (`pt.frameBuffers`). */
  encode(encoder: GPUCommandEncoder, frame: { accum: GPUBuffer; planes: GPUBuffer; width: number; height: number }, target: GPUTextureView): void {
    const { width, height } = frame;
    const key = [frame.accum, frame.planes, width, height];
    if (key.some((k, i) => k !== this.key[i])) {
      this.key = key;
      this.sums?.destroy();
      this.sums = this.device.createBuffer({ label: 'frame view sums', size: width * height * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      this.pickGroup = this.device.createBindGroup({ layout: this.pickPipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: this.uniform } },
        { binding: 1, resource: { buffer: frame.accum } },
        { binding: 2, resource: { buffer: frame.planes } },
        { binding: 3, resource: { buffer: this.sums } },
      ] });
      this.drawGroup = this.device.createBindGroup({ layout: this.drawPipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: this.uniform } },
        { binding: 1, resource: { buffer: this.sums } },
      ] });
      this.count = 0;
    }
    this.count = this.accumulate ? this.count + 1 : 1;
    this.rendered++;
    const u = new ArrayBuffer(32);
    new Uint32Array(u).set([width, height, this.mode, this.count]);
    new Float32Array(u)[4] = this.exposure;
    new Uint32Array(u)[5] = RADIANCE_VIEWS.has(this.mode) ? 1 : 0;
    this.device.queue.writeBuffer(this.uniform, 0, u);
    const cp = encoder.beginComputePass({ label: 'frame view pick' });
    cp.setPipeline(this.pickPipeline);
    cp.setBindGroup(0, this.pickGroup!);
    cp.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
    cp.end();
    const rp = encoder.beginRenderPass({ label: 'frame view draw', colorAttachments: [{ view: target, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }] });
    rp.setPipeline(this.drawPipeline);
    rp.setBindGroup(0, this.drawGroup!);
    rp.draw(3);
    rp.end();
  }

  /** The current average as linear values (a readback, for checks). Needs a copy: the sums buffer is not mappable. */
  async read(width: number, height: number): Promise<Float32Array> {
    const size = width * height * 16;
    const staging = this.device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.sums!, 0, staging, 0, size);
    this.device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const sums = new Float32Array(staging.getMappedRange().slice(0));
    staging.destroy();
    const out = new Float32Array(width * height * 4);
    for (let i = 0; i < width * height; i++) for (let c = 0; c < 4; c++) out[i * 4 + c] = sums[i * 4 + 3]! > 0 ? sums[i * 4 + c]! / sums[i * 4 + 3]! : 0;
    return out;
  }

  destroy(): void {
    this.sums?.destroy();
    this.uniform.destroy();
  }
}
