// A small WebGPU rasteriser that writes a visibility buffer (P3.6, `packages/pathtracer/src/integrator/vbuffer.ts`) from a scene copy's input: what an engine
// would do in its own G-buffer pass, standing in for it in the bench. Not part of the library.
//
// One draw of every triangle of every visible instance (non-indexed, world-space positions computed on the CPU, flat ids per vertex), depth32float, no culling,
// no alpha test (the integrator's cutout step passes through cutout texels by tracing, so the image is exact without one). The projection is the integrator's own camera:
// a world point P goes to  ndc = (dot(P - o, right) / |right|^2, dot(P - o, up) / |up|^2) / dot(P - o, forward)  (`ptCameraBasis`, the same numbers `ptCameraDirection` uses),
// shifted by the frame's sub-pixel offset: pixel i then samples at i + 0.5 + jitter (x right, y down), as the kernel's `PT_PAIR_JITTER` convention.
import type { PtCameraBasis } from '@pmndrs/pathtracer';
import { PT_VBUFFER_FORMAT, ptVbufferPack } from '@pmndrs/pathtracer';
import type { SceneCopyInput } from '@pmndrs/raytracer';

const NO_MATERIAL = 0xffffffff;
const SLOTS = 1024;
const SLOT_BYTES = 256;

const WGSL = /* wgsl */ `
struct U {
    origin: vec4f,
    right: vec4f,   // right / |right|^2
    up: vec4f,      // up / |up|^2
    forward: vec4f,
    shift: vec4f,   // 2 jitter / size (x, y), near, far
}
@group(0) @binding(0) var<uniform> u: U;

struct VOut {
    @builtin(position) pos: vec4f,
    @location(0) bary: vec2f,
    @location(1) @interpolate(flat) ids: vec3u,
}

@vertex
fn vs(@location(0) p: vec3f, @location(1) corner: u32, @location(2) ids: vec3u) -> VOut {
    let d = p - u.origin.xyz;
    let z = dot(d, u.forward.xyz);
    var o: VOut;
    o.pos = vec4f(dot(d, u.right.xyz) - u.shift.x * z, dot(d, u.up.xyz) + u.shift.y * z, u.shift.w * (z - u.shift.z) / (u.shift.w - u.shift.z), z);
    o.bary = vec2f(select(0.0, 1.0, corner == 1u), select(0.0, 1.0, corner == 2u));
    o.ids = ids;
    return o;
}

@fragment
fn fs(i: VOut) -> @location(0) vec4u {
    let q = vec2u(clamp(round(i.bary * 65535.0), vec2f(0.0), vec2f(65535.0)));
    return vec4u(i.ids.x, i.ids.y, i.ids.z, q.x | (q.y << 16u));
}
`;

export class VbufferRaster {
  readonly texture: GPUTexture;
  readonly triangles: number;
  private readonly depth: GPUTexture;
  private readonly pipeline: GPURenderPipeline;
  private readonly vertices: GPUBuffer;
  private readonly uniforms: GPUBuffer;
  private readonly group: GPUBindGroup;
  private slot = 0;
  private readonly vertexCount: number;

  private readonly device: GPUDevice;
  readonly width: number;
  readonly height: number;

  private constructor(device: GPUDevice, width: number, height: number, vertices: Uint32Array, vertexCount: number) {
    this.device = device;
    this.width = width;
    this.height = height;
    this.vertexCount = vertexCount;
    this.triangles = vertexCount / 3;
    this.texture = device.createTexture({ label: 'vbuffer', size: [width, height], format: PT_VBUFFER_FORMAT, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC });
    this.depth = device.createTexture({ label: 'vbuffer depth', size: [width, height], format: 'depth32float', usage: GPUTextureUsage.RENDER_ATTACHMENT });
    this.vertices = device.createBuffer({ label: 'vbuffer vertices', size: Math.max(16, vertices.byteLength), usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.vertices, 0, vertices.buffer as ArrayBuffer, 0, vertices.byteLength);
    this.uniforms = device.createBuffer({ label: 'vbuffer uniforms', size: SLOTS * SLOT_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const module = device.createShaderModule({ label: 'vbuffer raster', code: WGSL });
    const layout = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 80 } }] });
    this.group = device.createBindGroup({ layout, entries: [{ binding: 0, resource: { buffer: this.uniforms, offset: 0, size: 80 } }] });
    this.pipeline = device.createRenderPipeline({
      label: 'vbuffer raster',
      layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      vertex: {
        module,
        entryPoint: 'vs',
        buffers: [{ arrayStride: 32, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }, { shaderLocation: 1, offset: 12, format: 'uint32' }, { shaderLocation: 2, offset: 16, format: 'uint32x3' }] }],
      },
      fragment: { module, entryPoint: 'fs', targets: [{ format: PT_VBUFFER_FORMAT }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
    });
  }

  /** Builds the vertex data from a scene copy input (`meshes`, `instances`, per-triangle materials) and the target size. */
  static create(device: GPUDevice, input: Pick<SceneCopyInput, 'meshes' | 'instances'>, width: number, height: number): VbufferRaster {
    let triangles = 0;
    for (const instance of input.instances) {
      if (instance.visible === false) continue;
      const mesh = input.meshes[instance.meshIndex]!;
      triangles += Math.floor((mesh.indices ? mesh.indices.length : mesh.positions.length / 3) / 3);
    }
    const words = new Uint32Array(triangles * 3 * 8);
    const floats = new Float32Array(words.buffer);
    let v = 0;
    for (const instance of input.instances) {
      if (instance.visible === false) continue;
      const mesh = input.meshes[instance.meshIndex]!;
      const m = instance.objectToWorld;
      const count = Math.floor((mesh.indices ? mesh.indices.length : mesh.positions.length / 3) / 3);
      for (let t = 0; t < count; t++) {
        const tm = mesh.triangleMaterials?.[t];
        const material = tm !== undefined && tm !== 0xffff ? tm : instance.material ?? NO_MATERIAL;
        for (let k = 0; k < 3; k++) {
          const vi = mesh.indices ? (mesh.indices[t * 3 + k] as number) : t * 3 + k;
          const x = mesh.positions[vi * 3] as number;
          const y = mesh.positions[vi * 3 + 1] as number;
          const z = mesh.positions[vi * 3 + 2] as number;
          const o = v * 8;
          floats[o] = (m[0] as number) * x + (m[4] as number) * y + (m[8] as number) * z + (m[12] as number);
          floats[o + 1] = (m[1] as number) * x + (m[5] as number) * y + (m[9] as number) * z + (m[13] as number);
          floats[o + 2] = (m[2] as number) * x + (m[6] as number) * y + (m[10] as number) * z + (m[14] as number);
          words[o + 3] = k;
          words[o + 4] = instance.instanceIndex + 1;
          words[o + 5] = t;
          words[o + 6] = material;
          v++;
        }
      }
    }
    return new VbufferRaster(device, width, height, words, v);
  }

  /**
   * Rasterises one frame into `texture` with the sub-pixel offset `jitter` (pixels, x right / y down, 0 = pixel centre), recording into `encoder`.
   * `basis` is the integrator's camera (`ptCameraBasis`).
   */
  encode(encoder: GPUCommandEncoder, basis: PtCameraBasis, jitter: readonly [number, number], near = 0.01, far = 10000): void {
    const dot3 = (a: readonly number[], b: readonly number[]): number => (a[0] as number) * (b[0] as number) + (a[1] as number) * (b[1] as number) + (a[2] as number) * (b[2] as number);
    const scale = (a: readonly number[], s: number): number[] => [(a[0] as number) * s, (a[1] as number) * s, (a[2] as number) * s];
    const data = new Float32Array(20);
    data.set([...basis.origin, 0, ...scale(basis.right, 1 / dot3(basis.right, basis.right)), 0, ...scale(basis.up, 1 / dot3(basis.up, basis.up)), 0, ...basis.forward, 0, (2 * jitter[0]) / this.width, (2 * jitter[1]) / this.height, near, far]);
    const offset = (this.slot++ % SLOTS) * SLOT_BYTES;
    this.device.queue.writeBuffer(this.uniforms, offset, data);
    const pass = encoder.beginRenderPass({
      label: 'vbuffer',
      colorAttachments: [{ view: this.texture.createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 0 } }],
      depthStencilAttachment: { view: this.depth.createView(), depthLoadOp: 'clear', depthStoreOp: 'discard', depthClearValue: 1 },
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.group, [offset]);
    pass.setVertexBuffer(0, this.vertices);
    pass.draw(this.vertexCount);
    pass.end();
  }

  /** Copies the V-buffer to the CPU (`width * height * 4` words). Never inside a timed region. */
  async read(): Promise<Uint32Array> {
    const bytesPerRow = Math.ceil((this.width * 16) / 256) * 256;
    const staging = this.device.createBuffer({ size: bytesPerRow * this.height, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = this.device.createCommandEncoder();
    encoder.copyTextureToBuffer({ texture: this.texture }, { buffer: staging, bytesPerRow }, [this.width, this.height]);
    this.device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const raw = new Uint32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    const out = new Uint32Array(this.width * this.height * 4);
    for (let y = 0; y < this.height; y++) out.set(raw.subarray((y * bytesPerRow) / 4, (y * bytesPerRow) / 4 + this.width * 4), y * this.width * 4);
    return out;
  }

  destroy(): void {
    this.texture.destroy();
    this.depth.destroy();
    this.vertices.destroy();
    this.uniforms.destroy();
  }
}

/** The V-buffer texel of a hit as the rasteriser writes it (for tests that build a V-buffer on the CPU). */
export const vbufferTexel = ptVbufferPack;
