/**
 * Fog (experimental): sunlit thin fog with light shafts, and the estimators that make it converge.
 *
 * A dark room with three narrow slits in its back wall, two pillars and a small lamp is filled with a thin homogeneous fog (one closed box with a medium-boundary
 * material, optical depth about 0.4 across the room). A low, bright sun shines through the slits towards the camera, so the fog in the beams scatters light forwards: god rays.
 * The view is split: the left half is one estimator and the right half another, both at the same sample count.
 *
 * In thin fog most camera rays never collide with a particle, so the plain estimator (pbrt's volpath: light is sampled only where a real collision happens)
 * sees a shaft only through the rare rays that do collide: a very noisy image. `forcedNee` adds one light sample per free-flight segment, at a distance drawn
 * on the segment, and `equiangular` draws that distance around the lamp so that a lamp inside the fog stops being a 1/r^2 noise source. Both are unbiased:
 * the three estimators converge to the same image, only the noise differs. The estimator is compiled into the kernel (a change rebuilds it).
 */
import { buildGpuScene, type GpuScene, type SceneCopyInput, type RtMaterialInput } from '@pmndrs/raytracer';
import { ptMediumFrom, type PtIntegratorOptions } from '@pmndrs/pathtracer';
import { box, mesh } from '../../shared/cornell.ts';
import { defineExample, type ExampleContext } from '../../shared/example.ts';
import { sunFrom } from '../../shared/models.ts';
import { OrbitCamera } from '../../shared/orbit.ts';
import { ProgressiveRenderer } from '../../shared/progressive.ts';

type V3 = [number, number, number];
const NOTE = 'Both halves have the same sample count. In a mid-air shaft the plain half is about 3x noisier per pixel than the forced half (about 9x in variance, measured at 64 spp); both converge to the same mean.';
type Estimator = 'plain' | 'forced' | 'equiangular';
const isEstimator = (e: string | null): e is Estimator => e === 'plain' || e === 'forced' || e === 'equiangular';
const NAMES: Record<Estimator, string> = { plain: 'Plain (collisions only)', forced: 'Forced in-scatter NEE', equiangular: 'Forced + equiangular lamp' };

const query = new URLSearchParams(location.search);
const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

/** The room's inside is x, z in [-3, 3], y in [0, 3]; the fog box is inset by 1 cm so no surface is shared with the walls. */
const FOG_MIN: V3 = [-2.99, 0.01, -2.99];
const FOG_MAX: V3 = [2.99, 2.99, 2.99];

/** Boxes (centre, half size) as one mesh. */
const boxes = (list: [V3, V3][]) => mesh(list.flatMap(([c, h]) => box(c, h, 0)));

/** The static room: floor, ceiling, side walls, a back wall with four slits, two pillars. One grey material. */
function room() {
  const back: [V3, V3][] = [];
  const slitX = [-1.7, 0, 1.7], w = 0.2;
  // Columns between the slits (full height), then a sill and a lintel over each slit.
  const edges = [-3.4, ...slitX.flatMap((x) => [x - w, x + w]), 3.4];
  for (let i = 0; i < edges.length; i += 2) back.push([[(edges[i]! + edges[i + 1]!) / 2, 1.5, -3.05], [(edges[i + 1]! - edges[i]!) / 2, 1.6, 0.05]]);
  for (const x of slitX) {
    back.push([[x, 0.15, -3.05], [w, 0.25, 0.05]]);
    back.push([[x, 2.85, -3.05], [w, 0.35, 0.05]]);
  }
  return boxes([
    [[0, -0.2, 0], [3.4, 0.2, 3.4]], // floor
    [[0, 3.2, 0], [3.4, 0.2, 3.4]], // ceiling
    [[-3.2, 1.5, 0], [0.2, 1.6, 3.4]], // left wall
    [[3.2, 1.5, 0], [0.2, 1.6, 3.4]], // right wall
    [[0, 1.5, 3.2], [3.4, 1.6, 0.2]], // front wall (behind the camera)
    ...back,
    [[-0.9, 1.5, -0.9], [0.25, 1.5, 0.25]], // pillars in the beams
    [[1.0, 1.0, 0.3], [0.25, 1.0, 0.25]],
  ]);
}

defineExample(async (ctx) => {
  const { device, pane, canvas } = ctx;
  const scale = Number(query.get('scale') ?? 0);
  if (scale > 0) ctx.setRenderScale(scale);
  const est = query.get('estimator');
  const params = {
    density: 0.07, g: 0.8, albedo: 0.9, sunElevation: 26, sunAzimuth: 140, sun: 200, exposure: 3,
    lamp: true, lampPower: 40, left: (isEstimator(est) ? est : 'plain') as Estimator, right: (isEstimator(est) ? est : 'equiangular') as Estimator,
    /** Where the split is, as a fraction of the width (1 = only the left estimator). */
    split: isEstimator(est) ? 1 : 0.5,
    maxDepth: 6,
  };

  const orbit = new OrbitCamera(canvas, { position: [2.5, 1.5, 2.6], target: [-0.6, 1.1, -1.2], up: [0, 1, 0], yFovDeg: 62 });
  const roomMesh = room();
  // A small lamp hanging in the fog, facing down (it emits from its front face).
  const lampMesh = mesh([[[1.05, 2.1, -1.65], [1.35, 2.1, -1.65], [1.35, 2.1, -1.35], [1.05, 2.1, -1.35]]]);
  const fogMesh = mesh(box([0, 1.5, 0], [(FOG_MAX[0] - FOG_MIN[0]) / 2, (FOG_MAX[1] - FOG_MIN[1]) / 2, (FOG_MAX[2] - FOG_MIN[2]) / 2], 0));
  const input = (): SceneCopyInput => {
    const lamp: V3 = [params.lampPower, params.lampPower * 0.85, params.lampPower * 0.55];
    const materials: RtMaterialInput[] = [
      { baseColor: [0.05, 0.05, 0.05], specularWeight: 0 },
      { baseColor: [0, 0, 0], baseWeight: 0, specularWeight: 0, emissionColor: lamp },
      { baseColor: [0, 0, 0], baseWeight: 0, specularWeight: 0 }, // the fog boundary scatters nothing itself
    ];
    const meshes = [roomMesh, fogMesh, ...(params.lamp ? [lampMesh] : [])];
    return {
      meshes,
      instances: meshes.map((_, i) => ({ meshIndex: i, instanceIndex: i, objectToWorld: IDENTITY, material: [0, 2, 1][i]! })),
      materials,
      sun: sunFrom(params.sunElevation, params.sunAzimuth, params.sun),
      environment: { radiance: [0.002, 0.003, 0.005] },
    };
  };

  let gpuScene: GpuScene | null = null;
  /** Two renderers over the same scene copy, one per half of the view; B is idle while the split is at 1. */
  let A: ProgressiveRenderer | null = null;
  let B: ProgressiveRenderer | null = null;

  // Each renderer draws its own accumulator into a 1x1 scratch target (its present pass is not wanted); the split view below reads both accumulators.
  const scratch = device.createTexture({ label: 'fog scratch', size: [1, 1], format: ctx.format, usage: GPUTextureUsage.RENDER_ATTACHMENT });
  const hidden: ExampleContext = Object.create(ctx, { target: { value: () => scratch.createView() } });
  const splitView = new SplitView(device, ctx.format);

  const volumes = (estimator: Estimator): PtIntegratorOptions['volumes'] => {
    const sigmaS = params.density * params.albedo, sigmaA = params.density * (1 - params.albedo);
    return {
      media: [ptMediumFrom([sigmaA, sigmaA, sigmaA], [sigmaS, sigmaS, sigmaS], params.g)],
      // Scene materials: 0 walls, 1 lamp, 2 the fog boundary (medium 1 inside).
      materials: [{ interior: 0, exterior: 0, boundary: false }, { interior: 0, exterior: 0, boundary: false }, { interior: 1, exterior: 0, boundary: true }],
      // The camera sits inside the fog box, so its rays start in medium 1 (nothing else tells the kernel: boundaries are only crossed, never tested against the camera).
      cameraMedium: 1,
      ...(estimator !== 'plain' ? { forcedNee: true } : {}),
      ...(estimator === 'equiangular' ? { equiangular: true } : {}),
    };
  };
  const note = document.createElement('div');
  note.style.cssText = 'position:fixed;left:50%;top:56px;transform:translateX(-50%);max-width:min(760px,92vw);padding:8px 14px;border-radius:8px;background:rgba(0,0,0,.6);color:#eee;font:13px/1.4 system-ui,sans-serif;text-align:center;pointer-events:none';
  note.textContent = NOTE;
  document.body.append(note);
  const labels = [0, 1].map((i) => {
    const el = document.createElement('div');
    el.style.cssText = `position:fixed;top:12px;${i === 0 ? 'left:14px' : 'right:14px'};padding:4px 10px;border-radius:6px;background:rgba(0,0,0,.6);color:#fff;font:600 13px system-ui,sans-serif;pointer-events:none`;
    document.body.append(el);
    return el;
  });
  const updateLabels = () => {
    labels[0]!.textContent = NAMES[params.left];
    labels[1]!.textContent = NAMES[params.right];
    labels[0]!.style.left = '14px';
    labels[1]!.style.display = params.split < 0.95 ? '' : 'none';
    ctx.stats.set('optical depth across the room', (params.density * 6).toFixed(2));
  };
  let presentDirty = true;

  /** Rebuilds both kernels (the medium and the estimators are compiled in) and restarts both images together. */
  const rebuild = async () => {
    ctx.progress('Compiling…');
    try {
      gpuScene ??= await buildGpuScene(device, input(), { label: 'fog' });
      const make = async (e: Estimator) => {
        const r = await ProgressiveRenderer.create(hidden, gpuScene!, orbit.camera, { integrator: { volumes: volumes(e), sunVisible: false } });
        r.settings.maxDepth = params.maxDepth;
        return r;
      };
      const [a, b] = [await make(params.left), await make(params.right)];
      const old = [A, B];
      A = a;
      B = b;
      (globalThis as { __progressive?: ProgressiveRenderer }).__progressive = a;
      Object.assign(globalThis, { __fog: { get left() { return A; }, get right() { return B; }, params, rebuild: () => rebuild(), redraw: () => (presentDirty = true), rebuildScene: () => rebuildScene() } });
      updateLabels();
      presentDirty = true;
      ctx.progress(null);
      await device.queue.onSubmittedWorkDone();
      for (const r of old) r?.destroy();
    } catch (error) {
      ctx.error('Could not build the fog', error);
    }
  };
  /** The sun and the lamp only change the scene copy: rebuild it (cheap) and rebind. */
  const rebuildScene = async () => {
    const old = gpuScene;
    gpuScene = await buildGpuScene(device, input(), { label: 'fog' });
    A?.setScene(gpuScene);
    B?.setScene(gpuScene);
    if (old) void device.queue.onSubmittedWorkDone().then(() => old.destroy());
  };
  orbit.onChange = () => {
    A?.setCamera(orbit.camera);
    B?.setCamera(orbit.camera);
  };
  const scene = (e: { last: boolean }) => e.last && void rebuildScene();
  const recompile = (e: { last: boolean }) => e.last && void rebuild();

  const estimator = pane.addFolder({ title: 'Estimators (rebuild the kernels)' });
  const options = { [NAMES.plain]: 'plain', [NAMES.forced]: 'forced', [NAMES.equiangular]: 'equiangular' };
  estimator.addBinding(params, 'left', { label: 'left half', options }).on('change', () => void rebuild());
  estimator.addBinding(params, 'right', { label: 'right half', options }).on('change', () => void rebuild());
  estimator.addBinding(params, 'split', { label: 'split (1 = left only)', min: 0.05, max: 1, step: 0.01 }).on('change', () => {
    // Both images must have the same sample count again after B was idle.
    A?.restart();
    B?.restart();
    updateLabels();
    presentDirty = true;
  });

  const fog = pane.addFolder({ title: 'Fog (rebuilds the kernels)' });
  fog.addBinding(params, 'density', { label: 'extinction (1/m)', min: 0.01, max: 0.4, step: 0.005 }).on('change', recompile);
  fog.addBinding(params, 'g', { label: 'anisotropy g', min: 0, max: 0.95, step: 0.01 }).on('change', recompile);
  fog.addBinding(params, 'albedo', { label: 'albedo', min: 0.5, max: 1, step: 0.01 }).on('change', recompile);

  const sun = pane.addFolder({ title: 'Sun' });
  sun.addBinding(params, 'sunElevation', { label: 'elevation (°)', min: 3, max: 80, step: 1 }).on('change', scene);
  sun.addBinding(params, 'sunAzimuth', { label: 'azimuth (°)', min: -180, max: 180, step: 1 }).on('change', scene);
  sun.addBinding(params, 'sun', { label: 'irradiance', min: 0, max: 400, step: 1 }).on('change', scene);

  const lamp = pane.addFolder({ title: 'Lamp' });
  lamp.addBinding(params, 'lamp', { label: 'on' }).on('change', () => void rebuildScene());
  lamp.addBinding(params, 'lampPower', { label: 'radiance', min: 1, max: 400, step: 1 }).on('change', scene);

  const path = pane.addFolder({ title: 'Path' });
  path.addBinding(params, 'exposure', { label: 'exposure', min: 0.1, max: 8, step: 0.05 }).on('change', () => (presentDirty = true));
  path.addBinding(params, 'maxDepth', { label: 'max scatters', min: 1, max: 16, step: 1 }).on('change', (e) => {
    if (!A || !B) return;
    A.settings.maxDepth = B.settings.maxDepth = params.maxDepth;
    if (e.last) {
      A.restart();
      B.restart();
    }
  });

  let lastSize = '';
  await rebuild();
  ctx.loop((encoder) => {
    if (!A || !B) return false;
    const [w, h] = ctx.filmSize();
    const a = A.frame(encoder);
    const b = params.split < 1 ? B.frame(encoder) : false;
    const size = `${w}x${h}`;
    if (!a && !b && !presentDirty && size === lastSize) return false;
    lastSize = size;
    presentDirty = false;
    splitView.encode(encoder, A.kernel.accumulator!, (params.split < 1 ? B : A).kernel.accumulator!, w, h, params.split * w, params.exposure, ctx.target());
    return true;
  });
});

/** Draws two accumulation buffers side by side (A left of `splitX`, B right of it) with a thin divider: divide by the sample count, expose, ACES, sRGB. */
class SplitView {
  private readonly pipeline: GPURenderPipeline;
  private readonly uniform: GPUBuffer;
  constructor(private readonly device: GPUDevice, format: GPUTextureFormat) {
    const module = device.createShaderModule({
      label: 'fog split',
      code: /* wgsl */ `
struct View { width: u32, height: u32, split: f32, exposure: f32 }
@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<storage, read> accumA: array<vec4f>;
@group(0) @binding(2) var<storage, read> accumB: array<vec4f>;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}
fn aces(x: vec3f) -> vec3f { return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3f(0.0), vec3f(1.0)); }
fn srgb(c: vec3f) -> vec3f { return select(1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055, 12.92 * c, c <= vec3f(0.0031308)); }
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let p = vec2u(pos.xy);
  let i = p.y * view.width + p.x;
  let left = pos.x < view.split;
  let s = select(accumB[i], accumA[i], left);
  var c = select(vec3f(0.0), s.rgb / s.w, s.w > 0.0) * view.exposure;
  c = srgb(aces(max(c, vec3f(0.0))));
  if (view.split < f32(view.width) - 1.0 && abs(pos.x - view.split) < 1.0) { c = vec3f(1.0); }
  return vec4f(c, 1.0);
}`,
    });
    this.pipeline = device.createRenderPipeline({ label: 'fog split', layout: 'auto', vertex: { module, entryPoint: 'vs' }, fragment: { module, entryPoint: 'fs', targets: [{ format }] } });
    this.uniform = device.createBuffer({ label: 'fog split view', size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }
  encode(encoder: GPUCommandEncoder, a: GPUBuffer, b: GPUBuffer, width: number, height: number, splitX: number, exposure: number, target: GPUTextureView): void {
    const data = new ArrayBuffer(16);
    new Uint32Array(data, 0, 2).set([width, height]);
    new Float32Array(data, 8, 2).set([splitX, exposure]);
    this.device.queue.writeBuffer(this.uniform, 0, data);
    const size = width * height * 16;
    const group = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.uniform } }, { binding: 1, resource: { buffer: a, size } }, { binding: 2, resource: { buffer: b, size } }],
    });
    const pass = encoder.beginRenderPass({ label: 'fog split', colorAttachments: [{ view: target, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }] });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, group);
    pass.draw(3);
    pass.end();
  }
}
