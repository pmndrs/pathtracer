/**
 * A-SVGF gradient samples from the path tracer (N11.18), for the Denoiser's `antiLag: 'gradient'` (`inputs.gradient`: one texel per
 * 3 x 3 stratum, r = luminance of a sample as shaded last frame, g = the same sample shaded this frame with the same random numbers,
 * b > 0 = valid; Schied et al. 2018, "Gradient Estimation for Real-Time Adaptive Temporal Filtering", sec. 3.1-3.3).
 *
 * How the samples are made here (the bench's "replay" variant: correct, not cheap):
 *  1. before this frame is traced, the PREVIOUS frame is traced again through the CURRENT scene: same camera, same sub-pixel jitter,
 *     same frame index and seed (so the same random numbers), whatever changed since (lights, the radiance cache's weights) included;
 *     its demodulated diffuse / specular planes are kept (`keepRetrace`);
 *  2. after this frame is traced, each stratum's centre pixel follows its (jitter-free) motion vector to the pixel q it came from, and
 *     the texel is (luminance of last frame's plane at q, luminance of the retrace at q, 1) per signal (`encode`); then this frame's
 *     planes become "last frame's" (`keepFrame`).
 * A static scene gives g = r bit for bit, so neither the noise nor the jitter reads as change; a moving camera needs no forward
 * projection because the whole previous frame is retraced (A-SVGF proper retraces one sample per stratum and forward-projects it:
 * that is the cheap form, which needs a kernel that traces a list of previous pixels; not built).
 * Cost as built: one more path-traced frame per frame, a 5 n vec4 copy, one small pass. 1 sample per pixel per frame only.
 */

const WGSL = /* wgsl */ `
struct P { dims: vec2u, gdims: vec2u, n: u32, valid: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> prevPlanes: array<vec4f>;   // last frame: planes 0 (diffuse), 1 (specular), 2 (normal, viewZ)
@group(0) @binding(2) var<storage, read> retrace: array<vec4f>;      // last frame traced again now: planes 0, 1
@group(0) @binding(3) var<storage, read> curPlanes: array<vec4f>;    // this frame: all planes (2 = normal + viewZ, 3 = motion.xy)
@group(0) @binding(4) var outD: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var outS: texture_storage_2d<rgba16float, write>;
fn luma(c: vec3f) -> f32 { return dot(max(c, vec3f(0.0)), vec3f(0.2126, 0.7152, 0.0722)); }
fn ok(x: f32) -> bool { return x == x && abs(x) < 6.0e4; }
@compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= p.gdims.x || gid.y >= p.gdims.y) { return; }
  var d = vec4f(0.0);
  var s = vec4f(0.0);
  let c = min(gid.xy * 3u + vec2u(1u), p.dims - vec2u(1u));
  let ci = c.y * p.dims.x + c.x;
  let nz = curPlanes[2u * p.n + ci];
  if (p.valid != 0u && dot(nz.xyz, nz.xyz) > 0.01) {
    let mv = curPlanes[3u * p.n + ci].xy;   // NDC, current - previous, +y up
    let uv = (vec2f(c) + 0.5) / vec2f(p.dims) - vec2f(mv.x, -mv.y) * 0.5;
    if (all(uv >= vec2f(0.0)) && all(uv < vec2f(1.0))) {
      let q = vec2u(uv * vec2f(p.dims));
      let qi = q.y * p.dims.x + q.x;
      let pn = prevPlanes[2u * p.n + qi];
      if (dot(pn.xyz, pn.xyz) > 0.01) {
        let rd = luma(prevPlanes[qi].rgb);
        let gd = luma(retrace[qi].rgb);
        let rs = luma(prevPlanes[p.n + qi].rgb);
        let gs = luma(retrace[p.n + qi].rgb);
        if (ok(rd) && ok(gd)) { d = vec4f(rd, gd, 1.0, 0.0); }
        if (ok(rs) && ok(gs)) { s = vec4f(rs, gs, 1.0, 0.0); }
      }
    }
  }
  textureStore(outD, gid.xy, d);
  textureStore(outS, gid.xy, s);
}
`;

let shared: { device: GPUDevice; pipeline: GPUComputePipeline; layout: GPUBindGroupLayout } | null = null;

export class GradientSamples {
  readonly diffuse: GPUTexture;
  readonly specular: GPUTexture;
  /** Last frame's planes are kept (a retrace of them makes sense). */
  have = false;
  /** The retrace of last frame was recorded for this frame. */
  retraced = false;
  private readonly prev: GPUBuffer;
  private readonly re: GPUBuffer;
  private readonly params: GPUBuffer;
  private readonly n: number;
  private readonly gw: number;
  private readonly gh: number;
  private group: { planes: GPUBuffer; group: GPUBindGroup } | null = null;
  private readonly device: GPUDevice;
  private readonly width: number;
  private readonly height: number;

  constructor(device: GPUDevice, width: number, height: number, label: string) {
    this.device = device;
    this.width = width;
    this.height = height;
    this.n = width * height;
    this.gw = Math.ceil(width / 3);
    this.gh = Math.ceil(height / 3);
    const tex = (name: string) => device.createTexture({ label: `${label} gradient ${name}`, size: [this.gw, this.gh], format: 'rgba16float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING });
    this.diffuse = tex('diffuse');
    this.specular = tex('specular');
    this.prev = device.createBuffer({ label: `${label} gradient previous planes`, size: 3 * this.n * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.re = device.createBuffer({ label: `${label} gradient retrace`, size: 2 * this.n * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.params = device.createBuffer({ label: `${label} gradient params`, size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    if (shared?.device !== device) {
      const C = GPUShaderStage.COMPUTE;
      const layout = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: C, buffer: { type: 'uniform' } },
          { binding: 1, visibility: C, buffer: { type: 'read-only-storage' } },
          { binding: 2, visibility: C, buffer: { type: 'read-only-storage' } },
          { binding: 3, visibility: C, buffer: { type: 'read-only-storage' } },
          { binding: 4, visibility: C, storageTexture: { access: 'write-only', format: 'rgba16float' } },
          { binding: 5, visibility: C, storageTexture: { access: 'write-only', format: 'rgba16float' } },
        ],
      });
      shared = { device, layout, pipeline: device.createComputePipeline({ label: 'gradient samples', layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }), compute: { module: device.createShaderModule({ code: WGSL }), entryPoint: 'main' } }) };
    }
  }

  /** After the retrace of last frame was encoded into `planes` (the kernel's frame planes): keep its diffuse / specular planes. */
  keepRetrace(encoder: GPUCommandEncoder, planes: GPUBuffer): void {
    encoder.copyBufferToBuffer(planes, 0, this.re, 0, 2 * this.n * 16);
    this.retraced = true;
  }

  /** After this frame was encoded into `planes`: write the gradient textures (invalid texels unless a retrace was kept), then keep this frame's planes. */
  encode(encoder: GPUCommandEncoder, planes: GPUBuffer): void {
    const s = shared as NonNullable<typeof shared>;
    this.device.queue.writeBuffer(this.params, 0, new Uint32Array([this.width, this.height, this.gw, this.gh, this.n, this.retraced && this.have ? 1 : 0, 0, 0]));
    if (this.group?.planes !== planes) {
      this.group = { planes, group: this.device.createBindGroup({ layout: s.layout, entries: [{ binding: 0, resource: { buffer: this.params } }, { binding: 1, resource: { buffer: this.prev } }, { binding: 2, resource: { buffer: this.re } }, { binding: 3, resource: { buffer: planes } }, { binding: 4, resource: this.diffuse.createView() }, { binding: 5, resource: this.specular.createView() }] }) };
    }
    const pass = encoder.beginComputePass({ label: 'gradient samples' });
    pass.setPipeline(s.pipeline);
    pass.setBindGroup(0, this.group.group);
    pass.dispatchWorkgroups(Math.ceil(this.gw / 8), Math.ceil(this.gh / 8));
    pass.end();
    encoder.copyBufferToBuffer(planes, 0, this.prev, 0, 3 * this.n * 16);
    this.have = true;
    this.retraced = false;
  }

  /** A cut, or a frame the samples cannot describe (several samples per pixel, an accumulated still): nothing to compare with next frame. */
  invalidate(): void {
    this.have = false;
    this.retraced = false;
  }

  destroy(): void {
    for (const t of [this.diffuse, this.specular]) t.destroy();
    for (const b of [this.prev, this.re, this.params]) b.destroy();
  }
}
