/**
 * Draws a path tracer's accumulation buffer (`array<vec4f>`: radiance sum, w = samples) to the canvas: divide by the sample count, apply
 * exposure, tone map, encode sRGB. No readback: one fullscreen triangle reads the storage buffer directly.
 */

export type ToneMapping = 'aces' | 'reinhard' | 'clamp';
const TONE_MAPPING: Record<ToneMapping, number> = { aces: 0, reinhard: 1, clamp: 2 };

const WGSL = /* wgsl */ `
struct View { width: u32, height: u32, toneMapping: u32, pad: u32, exposure: f32, pad1: f32, pad2: f32, pad3: f32 }
@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<storage, read> accum: array<vec4f>;

@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

// Krzysztof Narkowicz's 2015 fit of the ACES RRT + ODT ("ACES Filmic Tone Mapping Curve", THIRD_PARTY.md).
fn aces(x: vec3f) -> vec3f {
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3f(0.0), vec3f(1.0));
}

fn srgb(c: vec3f) -> vec3f {
  return select(1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055, 12.92 * c, c <= vec3f(0.0031308));
}

@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let p = vec2u(pos.xy);
  let s = accum[p.y * view.width + p.x];
  var c = select(vec3f(0.0), s.rgb / s.w, s.w > 0.0) * view.exposure;
  c = max(c, vec3f(0.0));
  switch view.toneMapping {
    case 0u: { c = aces(c); }
    case 1u: { c = c / (1.0 + c); }
    default: { c = clamp(c, vec3f(0.0), vec3f(1.0)); }
  }
  return vec4f(srgb(c), 1.0);
}
`;

export class AccumulationView {
  exposure = 1;
  toneMapping: ToneMapping = 'aces';
  private readonly pipeline: GPURenderPipeline;
  private readonly uniform: GPUBuffer;
  private group: GPUBindGroup | null = null;
  private bound: GPUBuffer | null = null;

  private readonly device: GPUDevice;

  constructor(device: GPUDevice, format: GPUTextureFormat) {
    this.device = device;
    const module = device.createShaderModule({ label: 'present', code: WGSL });
    this.pipeline = device.createRenderPipeline({ label: 'present', layout: 'auto', vertex: { module, entryPoint: 'vs' }, fragment: { module, entryPoint: 'fs', targets: [{ format }] } });
    this.uniform = device.createBuffer({ label: 'present view', size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }

  /** Draws `accum` (a film of `width` x `height`) into `target`, which must be the same size. */
  encode(encoder: GPUCommandEncoder, accum: GPUBuffer, width: number, height: number, target: GPUTextureView): void {
    if (this.bound !== accum) {
      this.group = this.device.createBindGroup({ label: 'present', layout: this.pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: this.uniform } }, { binding: 1, resource: { buffer: accum, size: width * height * 16 } }] });
      this.bound = accum;
    }
    const data = new ArrayBuffer(32);
    new Uint32Array(data, 0, 4).set([width, height, TONE_MAPPING[this.toneMapping], 0]);
    new Float32Array(data, 16, 1)[0] = this.exposure;
    this.device.queue.writeBuffer(this.uniform, 0, data);
    const pass = encoder.beginRenderPass({ label: 'present', colorAttachments: [{ view: target, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }] });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.group);
    pass.draw(3);
    pass.end();
  }

  destroy(): void {
    this.uniform.destroy();
  }
}
