/**
 * wf-frame-check (P4.1b, P4.2a): the wavefront scheduler's real-time frames against the megakernel's, PER PIXEL. Correctness only, no timing.
 *
 * Both kernels are built from the same adapter settings (`PtMegakernelAdapter`: the megakernel is the adapter's own, the wavefront is `createWavefrontKernel`, plain scheduler)
 * and run lock-step along a short camera path, one 1 spp frame per camera, with the same seed and frame indices. After each frame both are read back (`readFrame`) and every
 * plane is compared: beauty, diffuse, specular, normal + viewZ, motion + roughness, albedo factor, specular factor. The estimator, the sampler dimensions and the ReSTIR passes
 * are the same text / module, so the two differ only by float summation order (the megakernel adds into registers, the wavefront through memory): the gate is a relative
 * difference <= `tol` (default 1e-5) per pixel, the pixel's size being the largest |component| of the two vec4s. The wavefront's planes must also recompose its own radiance
 * (diffuse x albedo factor + specular x specular factor).
 *
 * Cases (`cases=` list, default `cornell,sponza-white,tex-checker,pbr`): each is a scene + settings, `mode: 'realtime'` (frame planes, blue noise, secondary material LOD, packed
 * scene, maxDepth 2). `restir=unbiased|biased` runs every case with ReSTIR DI at the first vertex on both kernels (same reservoirs expected: same passes, same inputs).
 *
 * `timing=1` (P4.1b / P4.2a; BEST EFFORT, the quiet pass owns the numbers): after the agreement, the SAME kernels timed ABBA in this one page: `rounds` rounds of `batch` frames (1 spp, `nextFrame`
 * between two cameras, the real-time frame loop) recorded back to back into ONE command buffer per sample, awaited with `onSubmittedWorkDone` before the next (one batch in flight), the
 * megakernel and the wavefront interleaved (A B B A ...). Reports ms per frame (min / median / max over the rounds) and the paired wavefront / megakernel ratio per round (min..max).
 * `res=960x540` sets the film of both kernels after the load (a file scene cannot be resized by its description; the kernels can).
 *
 * Args: cases=cornell,...  frames=6  step=1 (degrees)  restir=off|unbiased|biased  tol=1e-5  size=128 (Cornell side; file scenes keep their film)  scheduler=plain|  packing=packed|split
 */
import type { PtLookAtCamera, PtMegakernel, PtRestirOptions, PtWavefront, PtWavefrontOptions, PtFrame } from '@pmndrs/pathtracer';
import type { AdapterSummary } from '../runner/contract.ts';
import { getArgs, numberArg, reportError, reportResult, summarizeAdapter } from '../runner/page-client.ts';
import type { SceneDesc } from '../scenes/schema.ts';
import { getScene } from '../scenes/index.ts';
import { PtMegakernelAdapter, type PtAdapterSettings } from '../adapters/pt-megakernel/index.ts';

let adapterSummary: AdapterSummary | null = null;
const deviceErrors: string[] = [];
const problems: string[] = [];
const metrics: Record<string, number> = {};
const note = (m: string): void => {
  if (problems.length < 40) problems.push(m);
};

type V3 = [number, number, number];

const PLANES = ['radiance', 'diffuse', 'specular', 'normalViewZ', 'motionRoughness', 'albedo', 'specularFactor'] as const;

interface Case {
  scene: string;
  /** Pan instead of orbit (a camera inside a closed building must not orbit). */
  pan?: boolean;
  size?: number;
  settings?: PtAdapterSettings;
}

const CASES: Record<string, Case> = {
  cornell: { scene: 'cornell', size: 128 },
  'sponza-white': { scene: 'sponza-white', pan: true },
  'tex-checker': { scene: 'tex-checker' },
  pbr: { scene: 'restir-pbr-lamps' },
  'restir-tex': { scene: 'restir-tex-lamps' },
  'many-lights': { scene: 'many-lights-64' },
  lamps: { scene: 'sponza-lamps', pan: true },
  // alpha MASK with the exact shadow policy (alphaShadow 'policy' with alphaPolicy exact; the weighted policies hash their own per-ray random numbers: not per-pixel comparable)
  // the same without the realtime preset (full materials, level-0 textures, Owen-Sobol): separates a preset effect from a frame-plane effect
  'foliage-plain': { scene: 'foliage-mask', settings: { mode: undefined, frame: true, alphaShadow: 'policy' } },
  'foliage-tex0': { scene: 'foliage-mask', settings: { alphaShadow: 'policy', textureLod: 0 } },
  'foliage-matfull': { scene: 'foliage-mask', settings: { alphaShadow: 'policy', materialLod: 'full' } },
  foliage: { scene: 'foliage-mask', settings: { alphaShadow: 'policy' } },
};

function orbit(scene: SceneDesc, frames: number, stepDeg: number, pan: boolean): PtLookAtCamera[] {
  const c = scene.camera;
  if (pan) {
    const f: V3 = [c.target[0] - c.position[0], c.target[1] - c.position[1], c.target[2] - c.position[2]];
    const right: V3 = [-f[2] / Math.hypot(f[0], f[2]), 0, f[0] / Math.hypot(f[0], f[2])];
    return Array.from({ length: frames }, (_, i): PtLookAtCamera => {
      const a = ((i - (frames - 1) / 2) * stepDeg * Math.PI) / 180;
      const cs = Math.cos(a);
      const sn = Math.sin(a);
      const k = 0.03 * (i - (frames - 1) / 2);
      const pos: V3 = [c.position[0] + right[0] * k, c.position[1], c.position[2] + right[2] * k];
      return { position: pos, target: [pos[0] + f[0] * cs + f[2] * sn, pos[1] + f[1], pos[2] - f[0] * sn + f[2] * cs], up: [c.up[0], c.up[1], c.up[2]], yFovDeg: c.yFovDeg };
    });
  }
  const d: V3 = [c.position[0] - c.target[0], c.position[1] - c.target[1], c.position[2] - c.target[2]];
  return Array.from({ length: frames }, (_, i): PtLookAtCamera => {
    const a = ((i - (frames - 1) / 2) * stepDeg * Math.PI) / 180;
    const cs = Math.cos(a);
    const sn = Math.sin(a);
    return { position: [c.target[0] + d[0] * cs + d[2] * sn, c.position[1], c.target[2] - d[0] * sn + d[2] * cs], target: [c.target[0], c.target[1], c.target[2]], up: [c.up[0], c.up[1], c.up[2]], yFovDeg: c.yFovDeg };
  });
}

interface Agree {
  maxRel: number;
  /** Pixels (over all frames) with a relative difference above the tolerance. */
  over: number;
  /** Pixels whose four floats are bit-identical. */
  same: number;
  total: number;
  /** Per-channel relative difference, floor 1e-6 absolute (a strict view; cancellation in `radiance - direct` shows up here). */
  maxRelChannel: number;
  sumRel: number;
  rels: number[];
}

function compare(a: Float32Array, b: Float32Array, tol: number, into: Agree): void {
  for (let i = 0; i < a.length; i += 4) {
    let diff = 0;
    let size = 0;
    let identical = true;
    for (let k = 0; k < 4; k++) {
      const x = a[i + k] as number;
      const y = b[i + k] as number;
      if (!Object.is(x, y)) identical = false;
      if (Number.isNaN(x) || Number.isNaN(y)) {
        diff = Infinity;
        continue;
      }
      diff = Math.max(diff, Math.abs(x - y));
      size = Math.max(size, Math.abs(x), Math.abs(y));
      into.maxRelChannel = Math.max(into.maxRelChannel, Math.abs(x - y) / Math.max(Math.abs(x), Math.abs(y), 1e-6));
    }
    const rel = size > 0 ? diff / size : diff > 0 ? Infinity : 0;
    into.maxRel = Math.max(into.maxRel, rel);
    into.sumRel += rel;
    if (rel > tol) into.over++;
    if (identical) into.same++;
    into.total++;
    if (rel > 1e-7 && into.rels.length < 4096) into.rels.push(rel);
  }
}

const fresh = (): Agree => ({ maxRel: 0, over: 0, same: 0, total: 0, maxRelChannel: 0, sumRel: 0, rels: [] });

interface Opts {
  frames: number;
  step: number;
  tol: number;
  restir: PtRestirOptions | null;
  packing: 'packed' | 'split';
  size: number;
  res: [number, number] | null;
  timing: boolean;
  rounds: number;
  batch: number;
  minMs: number;
  agree: boolean;
  wavefront: Partial<PtWavefrontOptions>;
}

/** ms per frame of `n` frames recorded back to back into one command buffer (nothing read back; the one wait is the end of the batch). */
async function timeBatch(device: GPUDevice, kernel: PtMegakernel | PtWavefront, cams: PtLookAtCamera[], n: number): Promise<number> {
  const e = device.createCommandEncoder({ label: 'timing batch' });
  for (let i = 0; i < n; i++) {
    kernel.nextFrame(cams[i & 1] as PtLookAtCamera);
    kernel.encode(e, 1);
  }
  await device.queue.onSubmittedWorkDone();
  const t0 = performance.now();
  device.queue.submit([e.finish()]);
  await device.queue.onSubmittedWorkDone();
  return (performance.now() - t0) / n;
}

async function timeCase(tag: string, device: GPUDevice, mk: PtMegakernel, wf: PtWavefront, cams: PtLookAtCamera[], o: Opts): Promise<void> {
  mk.setCamera(cams[0] as PtLookAtCamera);
  wf.setCamera(cams[0] as PtLookAtCamera);
  // warm up (pipelines, bind groups, clocks) and size the batch from a probe: >= minMs per sample
  await timeBatch(device, mk, cams, 4);
  await timeBatch(device, wf, cams, 4);
  const probe = Math.max(await timeBatch(device, mk, cams, 8), await timeBatch(device, wf, cams, 8));
  const n = o.batch > 0 ? o.batch : Math.max(8, Math.min(200, Math.ceil(o.minMs / Math.max(probe, 0.05))));
  const a: number[] = [];
  const b: number[] = [];
  for (let r = 0; r < o.rounds; r++) {
    // ABBA: even rounds mk first, odd rounds wavefront first
    if (r % 2 === 0) {
      a.push(await timeBatch(device, mk, cams, n));
      b.push(await timeBatch(device, wf, cams, n));
    } else {
      b.push(await timeBatch(device, wf, cams, n));
      a.push(await timeBatch(device, mk, cams, n));
    }
  }
  const sorted = (v: number[]) => [...v].sort((x, y) => x - y);
  const med = (v: number[]) => sorted(v)[Math.floor(v.length / 2)] as number;
  metrics[`${tag}_time_batch_frames`] = n;
  metrics[`${tag}_time_megakernel_ms_min`] = Math.min(...a);
  metrics[`${tag}_time_megakernel_ms_median`] = med(a);
  metrics[`${tag}_time_megakernel_ms_max`] = Math.max(...a);
  metrics[`${tag}_time_wavefront_ms_min`] = Math.min(...b);
  metrics[`${tag}_time_wavefront_ms_median`] = med(b);
  metrics[`${tag}_time_wavefront_ms_max`] = Math.max(...b);
  const ratios = b.map((v, i) => v / (a[i] as number));
  metrics[`${tag}_time_ratio_wf_over_mk_min`] = Math.min(...ratios);
  metrics[`${tag}_time_ratio_wf_over_mk_median`] = med(ratios);
  metrics[`${tag}_time_ratio_wf_over_mk_max`] = Math.max(...ratios);
}

async function runCase(name: string, c: Case, o: Opts): Promise<void> {
  const tag = `${name}${o.restir ? '_restir' : ''}`;
  let scene = getScene(c.scene);
  if (c.size ?? (c.scene === 'cornell' ? o.size : 0)) {
    const s = c.size ?? o.size;
    scene = { ...scene, film: { ...scene.film, width: s, height: s } };
  }
  const settings: PtAdapterSettings = { nee: true, mode: 'realtime', scenePacking: o.packing, maxDepth: 2, rrStartDepth: 0, ...(o.restir ? { restir: o.restir } : {}), ...(c.settings ?? {}) };
  const adapter = new PtMegakernelAdapter(settings);
  await adapter.load(scene);
  const mk = adapter.renderer as PtMegakernel;
  const wf: PtWavefront = await adapter.createWavefrontKernel({ wavefront: o.wavefront });
  metrics[`${tag}_wavefront_max_kernel_bytes`] = wf.wgslBytes;
  for (const [k, v] of Object.entries(wf.kernelBytes)) metrics[`${tag}_wavefront_${k}_bytes`] = v as number;
  metrics[`${tag}_megakernel_bytes`] = mk.wgslBytes;
  const device = adapter.gpuContext!.device;
  if (o.res) {
    mk.setFilm(o.res[0], o.res[1]);
    wf.setFilm(o.res[0], o.res[1]);
  }
  const w = o.res ? o.res[0] : scene.film.width;
  const h = o.res ? o.res[1] : scene.film.height;
  const cams = orbit(scene, o.frames, o.step, !!c.pan);
  mk.setFrameIndex(0);
  wf.setFrameIndex(0);
  mk.setPrevCamera(null);
  wf.setPrevCamera(null);
  mk.setCamera(cams[0] as PtLookAtCamera);
  wf.setCamera(cams[0] as PtLookAtCamera);
  if (!o.agree) {
    await timeCase(tag, device, mk, wf, cams, o);
    wf.destroy();
    adapter.dispose();
    return;
  }
  const agree = Object.fromEntries(PLANES.map((p) => [p, fresh()])) as Record<(typeof PLANES)[number], Agree>;
  let identityWorst = 0;
  let nonFinite = 0;
  for (let f = 0; f < cams.length; f++) {
    if (f > 0) {
      mk.nextFrame(cams[f] as PtLookAtCamera);
      wf.nextFrame(cams[f] as PtLookAtCamera);
    }
    const e1 = device.createCommandEncoder({ label: 'mk frame' });
    mk.encode(e1, 1);
    device.queue.submit([e1.finish()]);
    const e2 = device.createCommandEncoder({ label: 'wf frame' });
    wf.encode(e2, 1);
    device.queue.submit([e2.finish()]);
    const a: PtFrame = await mk.readFrame();
    const b: PtFrame = await wf.readFrame();
    for (const p of PLANES) compare(a[p], b[p], o.tol, agree[p]);
    for (let i = 0; i < b.radiance.length; i += 4) {
      for (let k = 0; k < 3; k++) {
        const rec = (b.diffuse[i + k] as number) * (b.albedo[i + k] as number) + (b.specular[i + k] as number) * (b.specularFactor[i + k] as number);
        const want = b.radiance[i + k] as number;
        if (!Number.isFinite(rec) || !Number.isFinite(want)) nonFinite++;
        else identityWorst = Math.max(identityWorst, Math.abs(rec - want) / Math.max(Math.abs(want), 1e-3));
      }
    }
  }
  let worstPlane = 0;
  for (const p of PLANES) {
    const g = agree[p];
    metrics[`${tag}_${p}_rel_max`] = g.maxRel;
    metrics[`${tag}_${p}_rel_mean`] = g.sumRel / Math.max(1, g.total);
    metrics[`${tag}_${p}_pixels_over_tol`] = g.over;
    metrics[`${tag}_${p}_bit_identical_frac`] = g.same / Math.max(1, g.total);
    metrics[`${tag}_${p}_rel_channel_max`] = g.maxRelChannel;
    worstPlane = Math.max(worstPlane, g.maxRel);
    if (g.over > 0) note(`${tag}: plane ${p}: ${g.over} of ${g.total} pixels differ by more than ${o.tol} (max ${g.maxRel})`);
  }
  metrics[`${tag}_plane_rel_max`] = worstPlane;
  metrics[`${tag}_identity_rel_max`] = identityWorst;
  metrics[`${tag}_non_finite`] = nonFinite;
  metrics[`${tag}_pixels`] = w * h;
  metrics[`${tag}_disagree_mismatch`] = worstPlane > o.tol ? 1 : 0;
  metrics[`${tag}_identity_mismatch`] = identityWorst > 1e-5 || nonFinite > 0 ? 1 : 0;
  if (o.timing) await timeCase(tag, device, mk, wf, cams, o);
  wf.destroy();
  adapter.dispose();
}

try {
  const args = getArgs();
  if (!navigator.gpu) throw new Error('navigator.gpu is undefined: no WebGPU in this browser.');
  const gpu = await navigator.gpu.requestAdapter();
  if (!gpu) throw new Error('requestAdapter() returned null: no usable GPU.');
  adapterSummary = summarizeAdapter(gpu);
  const names = (args.cases ?? 'cornell,sponza-white,tex-checker,pbr').split(',');
  const restir: PtRestirOptions | null = args.restir === undefined || args.restir === 'off' ? null : args.restir === 'biased' ? {} : { bias: 'unbiased', initialVisibility: false };
  const wavefront: Partial<PtWavefrontOptions> = { ...(args.appends ? { appends: args.appends as 'atomic' | 'subgroup' | 'auto' } : {}), ...(args.sort ? { sort: args.sort as 'none' | 'material' | 'octant' | 'both' } : {}) };
  const opts: Opts = {
    frames: numberArg(args, 'frames', 6),
    step: numberArg(args, 'step', 1),
    tol: numberArg(args, 'tol', 1e-5),
    restir,
    packing: (args.packing ?? 'packed') as 'packed' | 'split',
    size: numberArg(args, 'size', 128),
    res: args.res ? (args.res.split('x').map(Number) as [number, number]) : null,
    timing: numberArg(args, 'timing', 0) === 1,
    rounds: numberArg(args, 'rounds', 5),
    batch: numberArg(args, 'batch', 0),
    minMs: numberArg(args, 'minms', 250),
    agree: numberArg(args, 'agree', 1) === 1,
    wavefront,
  };
  if (opts.timing) metrics.timing_best_effort = 1;
  for (const n of names) {
    const c = CASES[n];
    if (!c) throw new Error(`unknown case '${n}' (have ${Object.keys(CASES).join(', ')})`);
    await runCase(n, c, opts);
  }
  const total = Object.entries(metrics).filter(([k]) => k.endsWith('_mismatch')).reduce((sum, [, v]) => sum + v, 0);
  reportResult({ scene: 'wf-frame-check', renderer: 'ours', config: { cases: names.join(','), frames: opts.frames, restir: args.restir ?? 'off', packing: opts.packing, res: args.res ?? 'film', timing: opts.timing, wavefront: JSON.stringify(wavefront) }, valid: total === 0 && deviceErrors.length === 0, metrics: { ...metrics, mismatches: total }, notes: { problems } }, { adapter: adapterSummary, deviceErrors });
  if (total > 0) throw new Error(`wf-frame-check: ${total} mismatches. First:\n${problems.join('\n')}`);
} catch (error) {
  reportError(error, { adapter: adapterSummary, deviceErrors });
}
