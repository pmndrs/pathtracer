/**
 * realtime-check (P4.1): the real-time mode's inputs against closed forms. Correctness only, no timing.
 *
 * `frames` (default): a short camera path (an orbit of `step` degrees per frame about the look-at point) rendered one 1 spp frame at a time with `integrator.frame` and the
 * blue-noise sampler, on each scene of `scenes` (default `cornell,sponza-white`). Per frame the outputs are written as PFMs (`write=1`): radiance, diffuse / specular
 * demodulated, albedo, specular factor, normal, viewZ, motion, roughness. Checks:
 *  - motion: the kernel's motion equals the closed form from the two cameras at the primary hit. The hit is rebuilt from the frame's viewZ and the EXACT sub-pixel jitter
 *    (the TS twin of the sampler's pair 0), then projected through the unjittered current and previous cameras: |kernel - closed form| in pixels (p50, p99, max).
 *  - reprojection: frame f's motion maps each (smooth-depth) pixel into frame f-1; the world position rebuilt there equals this pixel's within 1.5 pixel footprints and
 *    the normals agree: fraction of pixels (`reproj_ok_frac`).
 *  - demodulation: diffuse x albedo factor + specular x specular factor equals the radiance (the planes are one exact split), albedo factors are materials of the scene.
 *  - guides: normals unit, viewZ negative on hits and 0 on misses, motion zero on the first frame.
 * `emission` (P4.5a): the emission plane against closed forms on a small scene (a lamp seen directly, a sky strip, a perfect mirror that reflects the lamp or the floor), for each of
 *  `frameEmission` none / primary / mirror: the planes recompose the radiance exactly (diffuse x albedo + specular x specular factor + emission), a directly seen lamp's emission is its radiance
 *  and its diffuse plane is empty (none: emission / albedo floor), the sky is the emission of a missed ray, a mirror's lamp reflection is emission (mirror) or specular-class radiance (primary),
 *  a mirror's floor reflection is never emission.
 * `bluenoise`: Cornell, N 1 spp frames per sampler (owen-sobol, pcg, bluenoise kronecker, bluenoise owen shift) against a progressive 2048 spp reference of the same depth:
 *  - unbiased: relBias of the N-frame mean image (each |.| <= `bias_tol`, default 2%).
 *  - the error spectrum, as a proxy: relMSE of a single frame against the reference raw, and after a Gaussian blur of both (sigma 2 and 4 px); the ratio blur / raw is
 *    the share of the error that survives a low-pass filter (lower = bluer). Averaged over the first `blur_frames` frames.
 *
 * `timing` (not in the default set; BEST EFFORT, the quiet pass owns the numbers): ms per 1 spp dispatch of plain Owen-Sobol, frame outputs, blue noise, and the whole realtime mode,
 *  interleaved ABBA over `rounds` rounds of `batch` dispatches in one command buffer, paired ratios to the plain kernel.
 *
 * `restir=unbiased|biased|shared` (P4.2e): the kernels under test (not the references) run ReSTIR DI at the first vertex, so the same checks apply to it: the planes recompose the
 * radiance (the reservoir's first-vertex light lands in the demodulated plane of the primary BSDF), motion / reprojection through the shared previous camera, and the blue-noise
 * A/B per frame after a blur (`bn_*_blur2_gain_vs_owen_sobol` = Owen-Sobol + ReSTIR over blue noise + ReSTIR). `unbiased` = unbiased combine, no initial visibility;
 * `shared` = unbiased + `sharedHit`.
 *
 * Args: scenes=cornell,sponza-white  frames=16  step=1 (degrees)  write=1  check=frames,emission,bluenoise  n=128 (bluenoise frames)  bias_tol=0.02  size=128 (Cornell side)  restir=unbiased
 */
import { fetchBlueNoise, type BlueNoiseTexture } from '@pmndrs/raytracer';
import { PT_DEMOD_EPS, ptCameraBasis, ptSampleBlueNoise, type PtFrame, type PtLookAtCamera, type PtMegakernel, type PtBlueNoiseShift, type PtRestirOptions } from '@pmndrs/pathtracer';
import type { AdapterSummary } from '../runner/contract.ts';
import { getArgs, numberArg, reportError, reportResult, summarizeAdapter, uploadImage } from '../runner/page-client.ts';
import type { SceneDesc } from '../scenes/schema.ts';
import { getScene } from '../scenes/index.ts';
import { compose, rotationX, translation } from '../scenes/geometry.ts';
import { PtMegakernelAdapter, type PtAdapterSettings } from '../adapters/pt-megakernel/index.ts';

let adapterSummary: AdapterSummary | null = null;
const deviceErrors: string[] = [];
const problems: string[] = [];
const metrics: Record<string, number> = {};
const note = (m: string) => {
  if (problems.length < 40) problems.push(m);
};

type V3 = [number, number, number];
const BN_NAME = 'gf-vec2-64';
const PAIRS = 6;

function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  return sorted[Math.min(sorted.length - 1, Math.floor(p * (sorted.length - 1)))] as number;
}

/** The scene at a smaller film (the checks are about exactness, not resolution). */
function sized(scene: SceneDesc, width: number, height: number): SceneDesc {
  return { ...scene, film: { ...scene.film, width, height } };
}

/** Camera `i` of an orbit about the look-at point: yaw `i * step - span / 2` degrees around +Y through the target. */
function orbit(scene: SceneDesc, frames: number, stepDeg: number, pan = false): PtLookAtCamera[] {
  const c = scene.camera;
  if (pan) {
    // A camera inside a closed building (Sponza) must not orbit (the orbit runs into a wall: viewZ 7e-5 m and f32 has no precision there). Pan instead: yaw about the camera
    // position and strafe 3 cm per frame along the camera's right axis.
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
  const out: PtLookAtCamera[] = [];
  for (let i = 0; i < frames; i++) {
    const a = ((i - (frames - 1) / 2) * stepDeg * Math.PI) / 180;
    const cs = Math.cos(a);
    const sn = Math.sin(a);
    out.push({ position: [c.target[0] + d[0] * cs + d[2] * sn, c.position[1], c.target[2] - d[0] * sn + d[2] * cs], target: [c.target[0], c.target[1], c.target[2]], up: [c.up[0], c.up[1], c.up[2]], yFovDeg: c.yFovDeg });
  }
  return out;
}

/** `restir=` page arg: the ReSTIR settings of the kernels under test (null = plain NEE). */
let restirUnderTest: PtRestirOptions | null = null;
/** `scheduler=wavefront` page arg (P4.1b / P4.2a): the kernels under test are `PtWavefront` (plain scheduler), the same checks apply (it has the same frame surface). */
let schedulerUnderTest: 'megakernel' | 'wavefront' = 'megakernel';

async function loadAdapter(scene: SceneDesc, settings: PtAdapterSettings): Promise<{ adapter: PtMegakernelAdapter; kernel: PtMegakernel; device: GPUDevice }> {
  const adapter = new PtMegakernelAdapter({ nee: true, ...(restirUnderTest ? { restir: restirUnderTest } : {}), ...(schedulerUnderTest === 'wavefront' ? { scheduler: 'wavefront' as const } : {}), ...settings });
  await adapter.load(scene);
  return { adapter, kernel: adapter.renderer as PtMegakernel, device: adapter.gpuContext!.device };
}

/** One 1 spp frame per camera; frame i uses `cameras[i]` with `cameras[i-1]` as the previous one. */
async function renderPath(kernel: PtMegakernel, device: GPUDevice, cameras: PtLookAtCamera[]): Promise<PtFrame[]> {
  kernel.setFrameIndex(0);
  const frames: PtFrame[] = [];
  for (let i = 0; i < cameras.length; i++) {
    if (i === 0) {
      kernel.setPrevCamera(null);
      kernel.setCamera(cameras[0] as PtLookAtCamera);
    } else kernel.nextFrame(cameras[i] as PtLookAtCamera);
    const encoder = device.createCommandEncoder({ label: 'realtime frame' });
    kernel.encode(encoder, 1);
    device.queue.submit([encoder.finish()]);
    frames.push(await kernel.readFrame());
  }
  return frames;
}

const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

interface Cam {
  basis: ReturnType<typeof ptCameraBasis>;
}
const cam = (c: PtLookAtCamera, w: number, h: number): Cam => ({ basis: ptCameraBasis(c, w, h) });

/** (ndc.x, ndc.y, viewZ) of world point p. Closed form of the kernel's `ptProject` in f64. */
function project(c: Cam, p: V3): V3 {
  const b = c.basis;
  const d: V3 = [p[0] - b.origin[0], p[1] - b.origin[1], p[2] - b.origin[2]];
  const z = dot(d, b.forward);
  return [dot(d, b.right) / (dot(b.right, b.right) * z), dot(d, b.up) / (dot(b.up, b.up) * z), -z];
}

/** World hit of the sample at continuous pixel position (sx, sy) with the given viewZ (negative). */
function unproject(c: Cam, w: number, h: number, sx: number, sy: number, viewZ: number): V3 {
  const b = c.basis;
  const nx = (sx / w) * 2 - 1;
  const ny = 1 - (sy / h) * 2;
  const t = -viewZ;
  return [0, 1, 2].map((k) => b.origin[k]! + (b.forward[k]! + b.right[k]! * nx + b.up[k]! * ny) * t) as V3;
}

function exactJitter(tex: BlueNoiseTexture, shift: PtBlueNoiseShift, x: number, y: number, frame: number): [number, number] {
  return ptSampleBlueNoise({ width: tex.meta.width, height: tex.meta.height, data: tex.data }, x, y, frame, 0, { pairs: PAIRS, shift });
}

interface PathStats {
  motionErrPx: number[];
  reprojOk: number;
  reprojTotal: number;
}

function checkPath(label: string, scene: SceneDesc, cameras: PtLookAtCamera[], frames: PtFrame[], tex: BlueNoiseTexture, shift: PtBlueNoiseShift): PathStats {
  const w = scene.film.width;
  const h = scene.film.height;
  const stats: PathStats = { motionErrPx: [], reprojOk: 0, reprojTotal: 0 };
  const diffuseAlbedos: V3[] = [];
  for (const m of Object.values(scene.materials)) {
    const r = (m as { reflectance?: V3 }).reflectance;
    if (r) diffuseAlbedos.push(r);
  }
  // V5.1c: a first-vertex medium scatter is diffuse class with albedo sigma_s / sigma_t
  for (const m of Object.values(scene.media ?? {})) {
    const mm = m as { sigmaA?: V3; sigmaS?: V3 };
    if (mm.sigmaA && mm.sigmaS) diffuseAlbedos.push([0, 1, 2].map((k) => mm.sigmaS![k]! / Math.max(mm.sigmaA![k]! + mm.sigmaS![k]!, 1e-12)) as V3);
  }
  let normalDev = 0;
  let viewZBad = 0;
  let demodWorst = 0;
  let firstFrameMotion = 0;
  let albedoOff = 0;
  let albedoChecked = 0;
  const worst: string[] = [];
  const tanHalf = Math.tan((scene.camera.yFovDeg * Math.PI) / 360);
  // a big film (Sponza at its own size: file scenes cannot be resized, their export is keyed on the film) is checked on a pixel grid
  const stride = Math.max(1, Math.round(Math.sqrt((w * h) / (160 * 160))));
  for (let f = 0; f < frames.length; f++) {
    const fr = frames[f] as PtFrame;
    const c = cam(cameras[f] as PtLookAtCamera, w, h);
  const pc = cam(cameras[Math.max(0, f - 1)] as PtLookAtCamera, w, h);
    const prev = f > 0 ? (frames[f - 1] as PtFrame) : null;
    for (let y = 0; y < h; y += stride) {
      for (let x = 0; x < w; x += stride) {
        const p = y * w + x;
        const i = p * 4;
        const vz = fr.normalViewZ[i + 3] as number;
        const hit = (fr.specularFactor[i + 3] as number) > 0.5;
        if (hit !== vz < 0) viewZBad++;
        // demodulation: the two planes recompose the radiance
        for (let k = 0; k < 3; k++) {
          const rec = (fr.diffuse[i + k] as number) * (fr.albedo[i + k] as number) + (fr.specular[i + k] as number) * (fr.specularFactor[i + k] as number) + (fr.emission[i + k] as number);
          const want = fr.radiance[i + k] as number;
          demodWorst = Math.max(demodWorst, Math.abs(rec - want) / Math.max(Math.abs(want), 1e-3));
        }
        if (!hit) continue;
        const nlen = Math.hypot(fr.normalViewZ[i] as number, fr.normalViewZ[i + 1] as number, fr.normalViewZ[i + 2] as number);
        normalDev = Math.max(normalDev, Math.abs(nlen - 1));
        // albedo factor is one of the scene's diffuse albedos (floored), unless the hit is specular-class
        if ((fr.albedo[i + 3] as number) < 0.5) {
          albedoChecked++;
          const a: V3 = [fr.albedo[i] as number, fr.albedo[i + 1] as number, fr.albedo[i + 2] as number];
          const ok = diffuseAlbedos.some((r) => [0, 1, 2].every((k) => Math.abs(Math.max(r[k]!, PT_DEMOD_EPS) - a[k]!) < 2e-3)) || diffuseAlbedos.length === 0 || a.every((v) => v === Math.fround(PT_DEMOD_EPS));
          if (!ok) {
            albedoOff++;
            if (albedoOff <= 3) note(`${label}: albedo factor ${a.join(',')} at (${x},${y}) frame ${f} is none of ${JSON.stringify(diffuseAlbedos)}`);
          }
        }
        const [jx, jy] = exactJitter(tex, shift, x, y, f);
        const sx = x + jx;
        const sy = y + jy;
        const P = unproject(c, w, h, sx, sy, vz);
        const cur = project(c, P);
        const old = project(pc, P);
        const mx = fr.motionRoughness[i] as number;
        const my = fr.motionRoughness[i + 1] as number;
        const err = Math.hypot(((mx - (cur[0] - old[0])) * w) / 2, ((my - (cur[1] - old[1])) * h) / 2);
        if (f === 0) firstFrameMotion = Math.max(firstFrameMotion, Math.hypot(mx, my));
        else {
          stats.motionErrPx.push(err);
          if (err > 0.05 && worst.length < 6) worst.push(`(${x},${y}) f${f} viewZ ${vz} jitter ${jx.toFixed(3)},${jy.toFixed(3)} motion ${mx},${my} expected ${cur[0] - old[0]},${cur[1] - old[1]} err ${err.toFixed(3)}px normal ${fr.normalViewZ[i]},${fr.normalViewZ[i + 1]},${fr.normalViewZ[i + 2]} P ${P.map((v) => v.toFixed(4))}`);
        }
        // reprojection into the previous frame (smooth-depth pixels only: a neighbour with another depth is an edge)
        if (prev && x > 0 && y > 0 && x < w - 1 && y < h - 1) {
          const smooth = [-1, 1].every((d) => Math.abs((fr.normalViewZ[(p + d) * 4 + 3] as number) - vz) < 0.02 * -vz && Math.abs((fr.normalViewZ[(p + d * w) * 4 + 3] as number) - vz) < 0.02 * -vz);
          if (!smooth) continue;
          const ndcx = (sx / w) * 2 - 1 - mx;
          const ndcy = 1 - (sy / h) * 2 - my;
          const qx = Math.floor((ndcx * 0.5 + 0.5) * w);
          const qy = Math.floor((0.5 - ndcy * 0.5) * h);
          if (qx < 1 || qy < 1 || qx >= w - 1 || qy >= h - 1) continue;
          const j = (qy * w + qx) * 4;
          const pvz = prev.normalViewZ[j + 3] as number;
          if (!(pvz < 0)) continue;
          // The previous frame's depth at the pixel the motion points into must be the depth motionZ predicts, up to the depth change over the <= 1.5 pixels the
          // nearest-pixel lookup can be off along the surface (it grows toward grazing angles), and the normals must agree.
          const footprint = (-vz * 2 * tanHalf) / h;
          const dirLen = Math.hypot(P[0] - c.basis.origin[0], P[1] - c.basis.origin[1], P[2] - c.basis.origin[2]);
          const cosT = Math.abs(((fr.normalViewZ[i] as number) * (P[0] - c.basis.origin[0]) + (fr.normalViewZ[i + 1] as number) * (P[1] - c.basis.origin[1]) + (fr.normalViewZ[i + 2] as number) * (P[2] - c.basis.origin[2])) / dirLen);
          const tol = (1.5 * footprint * Math.sqrt(Math.max(0, 1 - cosT * cosT))) / Math.max(cosT, 0.1) + 0.003 * -vz;
          const dz = Math.abs(pvz - (vz + (fr.motionRoughness[i + 3] as number)));
          const nd = (fr.normalViewZ[i] as number) * (prev.normalViewZ[j] as number) + (fr.normalViewZ[i + 1] as number) * (prev.normalViewZ[j + 1] as number) + (fr.normalViewZ[i + 2] as number) * (prev.normalViewZ[j + 2] as number);
          stats.reprojTotal++;
          if (dz < tol && nd > 0.9) stats.reprojOk++;
        }
      }
    }
  }
  metrics[`${label}_normal_len_dev`] = normalDev;
  metrics[`${label}_viewz_sign_mismatch`] = viewZBad;
  metrics[`${label}_demod_max_rel`] = demodWorst;
  metrics[`${label}_demod_mismatch`] = demodWorst > 1e-4 ? 1 : 0;
  metrics[`${label}_albedo_off`] = albedoOff;
  metrics[`${label}_albedo_checked`] = albedoChecked;
  metrics[`${label}_first_frame_motion_max`] = firstFrameMotion;
  const sorted = [...stats.motionErrPx].sort((a, b) => a - b);
  metrics[`${label}_motion_err_px_p50`] = percentile(sorted, 0.5);
  metrics[`${label}_motion_err_px_p99`] = percentile(sorted, 0.99);
  metrics[`${label}_motion_err_px_max`] = sorted.length ? (sorted[sorted.length - 1] as number) : 0;
  metrics[`${label}_motion_samples`] = sorted.length;
  metrics[`${label}_reproj_ok_frac`] = stats.reprojTotal ? stats.reprojOk / stats.reprojTotal : 1;
  metrics[`${label}_reproj_samples`] = stats.reprojTotal;
  metrics[`${label}_motion_mismatch`] = metrics[`${label}_motion_err_px_max`]! > 0.05 ? 1 : 0;
  metrics[`${label}_reproj_mismatch`] = metrics[`${label}_reproj_ok_frac`]! < 0.95 ? 1 : 0;
  metrics[`${label}_guides_mismatch`] = normalDev > 1e-3 || viewZBad > 0 || firstFrameMotion > 1e-6 || albedoOff > 0 ? 1 : 0;
  for (const line of worst) note(`${label}: ${line}`);
  if (metrics[`${label}_motion_mismatch`]) note(`${label}: motion error max ${metrics[`${label}_motion_err_px_max`]} px (> 0.05)`);
  if (metrics[`${label}_reproj_mismatch`]) note(`${label}: only ${(100 * metrics[`${label}_reproj_ok_frac`]!).toFixed(2)}% of pixels reproject onto the same surface point`);
  if (metrics[`${label}_demod_mismatch`]) note(`${label}: demodulated x factor misses the radiance by ${demodWorst}`);
  if (metrics[`${label}_guides_mismatch`]) note(`${label}: guides: normalDev ${normalDev}, viewZ sign mismatches ${viewZBad}, first frame motion ${firstFrameMotion}, albedo not in the scene ${albedoOff}/${albedoChecked}`);
  return stats;
}

async function writeFrames(label: string, scene: SceneDesc, frames: PtFrame[]): Promise<void> {
  const w = scene.film.width;
  const h = scene.film.height;
  const n = w * h;
  const pick = (plane: Float32Array, channels: number[]): Float32Array => {
    const out = new Float32Array(n * channels.length);
    for (let p = 0; p < n; p++) channels.forEach((c, k) => (out[p * channels.length + k] = plane[p * 4 + c] as number));
    return out;
  };
  for (let f = 0; f < frames.length; f++) {
    const fr = frames[f] as PtFrame;
    const stem = `${label}_f${String(f).padStart(2, '0')}`;
    const planes: Array<[string, Float32Array, 1 | 3]> = [
      ['radiance', pick(fr.radiance, [0, 1, 2]), 3],
      ['diffuse', pick(fr.diffuse, [0, 1, 2]), 3],
      ['specular', pick(fr.specular, [0, 1, 2]), 3],
      ['albedo', pick(fr.albedo, [0, 1, 2]), 3],
      ['normal', pick(fr.normalViewZ, [0, 1, 2]), 3],
      ['viewz', pick(fr.normalViewZ, [3]), 1],
      ['motion', pick(fr.motionRoughness, [0, 1, 3]), 3],
      ['roughness', pick(fr.motionRoughness, [2]), 1],
    ];
    for (const [name, data, channels] of planes) await uploadImage(`${stem}_${name}`, { width: w, height: h, channels, data });
  }
}

async function checkFrames(sceneNames: string[], frameCount: number, stepDeg: number, write: boolean, size: number): Promise<void> {
  const tex = await fetchBlueNoise('/packages/raytracer/assets/bluenoise', BN_NAME);
  for (const name of sceneNames) {
    const base = getScene(name);
    const aspect = base.film.width / base.film.height;
    const hasFile = base.meshes.some((m) => m.geometry.type === 'file');
    const scene = hasFile ? base : sized(base, Math.round(size * (aspect >= 1 ? aspect : 1)), Math.round(size * (aspect >= 1 ? 1 : 1 / aspect)));
    const cameras = orbit(scene, frameCount, stepDeg, hasFile);
    for (const shift of (hasFile ? ['kronecker'] : ['kronecker', 'owen']) as Array<PtBlueNoiseShift>) {
      const label = `${name.replace(/-/g, '_')}_${shift}`;
      const { adapter, kernel, device } = await loadAdapter(scene, { mode: 'realtime', maxDepth: 2, rrStartDepth: 0, sampler: 'bluenoise', blueNoise: BN_NAME, blueNoiseShift: shift, blueNoisePairs: PAIRS });
      const t0 = performance.now();
      const frames = await renderPath(kernel, device, cameras);
      metrics[`${label}_wall_ms_per_frame_with_readback`] = (performance.now() - t0) / frameCount;
      checkPath(label, scene, cameras, frames, tex, shift);
      metrics[`${label}_kernel_bytes`] = kernel.wgslBytes;
      if (write && shift === 'kronecker') await writeFrames(`rt_${name.replace(/-/g, '_')}`, scene, frames);
      adapter.dispose();
    }
  }
}


// ---------------------------------------------------------------------------------------------------------------- emission plane (P4.5a)

const LAMP = 20;
const SKY: V3 = [0.1, 0.2, 0.3];

/** A lamp (radiance 20, facing down) over a floor, a perfect mirror wall behind it that reflects the lamp into the camera, a strip of sky above the wall. */
function emissionScene(size: number): SceneDesc {
  return {
    version: 1,
    name: 'emission-check',
    purpose: 'P4.5a closed forms: emitter seen directly, sky behind a primary miss, emitter and floor seen in a perfect mirror.',
    tier: 0,
    camera: { type: 'perspective', position: [0, 1, 4], target: [0, 1, -2], up: [0, 1, 0], yFovDeg: 40 },
    film: { width: size, height: size, filter: 'box' },
    integrator: { maxDepth: 2 },
    materials: {
      floor: { type: 'diffuse', reflectance: [0.5, 0.5, 0.5] },
      lampBack: { type: 'black' },
      mirror: { type: 'conductor', eta: [0.2, 0.924, 1.102], k: [3.912, 2.452, 2.142], roughness: 0 },
    },
    meshes: [
      { name: 'floor', geometry: { type: 'quad', size: [12, 12] }, material: 'floor' },
      { name: 'mirror', geometry: { type: 'quad', size: [4, 3] }, material: 'mirror', transform: compose(translation(0, 1.5, -2), rotationX(Math.PI / 2)) },
      { name: 'lamp', geometry: { type: 'quad', size: [2, 2] }, material: 'lampBack', transform: compose(translation(0, 2, 0), rotationX(Math.PI)), emission: [LAMP, LAMP, LAMP] },
    ],
    lights: [],
    environment: { type: 'uniform', radiance: SKY },
  };
}

async function checkEmission(size: number): Promise<void> {
  const scene = emissionScene(size);
  const cameras = [scene.camera as PtLookAtCamera, scene.camera as PtLookAtCamera];
  for (const mode of ['none', 'primary', 'mirror'] as const) {
    const label = `emission_${mode}`;
    const { adapter, kernel, device } = await loadAdapter(scene, { mode: 'realtime', maxDepth: 2, rrStartDepth: 0, frameEmission: mode });
    const frames = await renderPath(kernel, device, cameras);
    const fr = frames[1] as PtFrame;
    const n = size * size;
    let identityWorst = 0;
    const count = { lamp: 0, sky: 0, mirrorLamp: 0, mirrorSky: 0, mirrorFloor: 0, other: 0 };
    let lampOff = 0;
    let lampDiffuse = 0;
    let skyOff = 0;
    let skyDiffuse = 0;
    let mirrorLampOff = 0;
    let mirrorFloorOff = 0;
    let zeroPlane = 0;
    for (let p = 0; p < n; p++) {
      const i = p * 4;
      const vz = fr.normalViewZ[i + 3] as number;
      const em = [0, 1, 2].map((k) => fr.emission[i + k] as number) as V3;
      const rad = [0, 1, 2].map((k) => fr.radiance[i + k] as number) as V3;
      const dif = [0, 1, 2].map((k) => (fr.diffuse[i + k] as number) * (fr.albedo[i + k] as number)) as V3;
      const spc = [0, 1, 2].map((k) => (fr.specular[i + k] as number) * (fr.specularFactor[i + k] as number)) as V3;
      for (let k = 0; k < 3; k++) identityWorst = Math.max(identityWorst, Math.abs(dif[k]! + spc[k]! + em[k]! - rad[k]!) / Math.max(Math.abs(rad[k]!), 1e-3));
      if (mode === 'none') zeroPlane += em.reduce((a, b) => a + Math.abs(b), 0);
      const near = (a: V3, b: V3, tol = 1e-5) => [0, 1, 2].every((k) => Math.abs(a[k]! - b[k]!) <= tol * Math.max(1, Math.abs(b[k]!)));
      const onMirror = vz < 0 && Math.abs(vz + 6) < 0.01 && (fr.normalViewZ[i + 2] as number) > 0.99; // the mirror wall: viewZ -6, normal +z
      const lampDirect = vz < 0 && near(rad, [LAMP, LAMP, LAMP]) && Math.abs(vz) < 5.5; // the lamp is nearer than the mirror (z = 0: viewZ -4)
      if (vz === 0) {
        // a primary miss: the sky
        count.sky++;
        if (mode === 'none') {
          if (!near(dif, SKY, 1e-4)) skyOff++;
        } else {
          if (!near(em, SKY, 1e-5)) skyOff++;
          skyDiffuse += dif.reduce((a, b) => a + Math.abs(b), 0) + spc.reduce((a, b) => a + Math.abs(b), 0);
        }
      } else if (lampDirect) {
        count.lamp++;
        if (mode === 'none') {
          if (!near(dif, [LAMP, LAMP, LAMP], 1e-4)) lampOff++;
          lampDiffuse = Math.max(lampDiffuse, (fr.diffuse[i] as number));
        } else {
          if (!near(em, [LAMP, LAMP, LAMP], 1e-6)) lampOff++;
          lampDiffuse += dif.reduce((a, b) => a + Math.abs(b), 0) + spc.reduce((a, b) => a + Math.abs(b), 0);
        }
      } else if (onMirror) {
        // the camera ray hit the mirror; what it reflects is in the radiance: the lamp (radiance = Fresnel x 20, no other term) or the lit floor (diffuse, NEE-less lamp: only the sky and nothing else here)
        const reflectsLamp = rad[0] > 8; // a lamp reflection is 20 x F (F >= 0.7 for this metal: > 14); the lit floor it can show is below 1.5
        if (reflectsLamp) {
          count.mirrorLamp++;
          if (mode === 'mirror') {
            if (!near(em, rad, 1e-6) || dif.some((v) => v !== 0) || spc.some((v) => v !== 0)) mirrorLampOff++;
          } else if (mode === 'primary') {
            if (em.some((v) => v !== 0) || !near(spc, rad, 1e-5)) mirrorLampOff++;
          }
        } else if (mode === 'mirror' && em.some((v) => v > 0)) {
          // the sky in the mirror: the whole radiance is emission (Fresnel x sky, below the sky itself), nothing in the filtered planes
          count.mirrorSky++;
          if (!near(em, rad, 1e-6) || dif.some((v) => v !== 0) || spc.some((v) => v !== 0) || em.some((v, k) => v > SKY[k]! * (1 + 1e-5))) mirrorFloorOff++;
        } else {
          // the lit floor in the mirror: filtered (specular class), never emission
          count.mirrorFloor++;
          if (mode !== 'none' && em.some((v) => v !== 0)) mirrorFloorOff++;
        }
      } else count.other++;
    }
    metrics[`${label}_identity_max_rel`] = identityWorst;
    metrics[`${label}_identity_mismatch`] = identityWorst > 1e-5 ? 1 : 0;
    metrics[`${label}_pixels_lamp`] = count.lamp;
    metrics[`${label}_pixels_sky`] = count.sky;
    metrics[`${label}_pixels_mirror_lamp`] = count.mirrorLamp;
    metrics[`${label}_pixels_mirror_sky`] = count.mirrorSky;
    metrics[`${label}_pixels_mirror_floor`] = count.mirrorFloor;
    metrics[`${label}_lamp_off`] = lampOff;
    metrics[`${label}_sky_off`] = skyOff;
    metrics[`${label}_mirror_lamp_off`] = mirrorLampOff;
    metrics[`${label}_mirror_floor_emission_off`] = mirrorFloorOff;
    metrics[`${label}_lamp_other_planes_sum`] = lampDiffuse;
    metrics[`${label}_sky_other_planes_sum`] = skyDiffuse;
    metrics[`${label}_emission_mismatch`] = identityWorst > 1e-5 || lampOff + skyOff + mirrorLampOff + mirrorFloorOff > 0 || count.lamp === 0 || count.sky === 0 || count.mirrorLamp === 0 || count.mirrorFloor === 0 || (mode === 'mirror' && count.mirrorSky === 0) || (mode !== 'none' && lampDiffuse + skyDiffuse > 0) || (mode === 'none' && zeroPlane !== 0) ? 1 : 0;
    if (metrics[`${label}_emission_mismatch`]) note(`${label}: identity ${identityWorst}, counts ${JSON.stringify(count)}, off: lamp ${lampOff} sky ${skyOff} mirrorLamp ${mirrorLampOff} mirrorFloor ${mirrorFloorOff}, other planes on emitters ${lampDiffuse + skyDiffuse}, none-mode plane sum ${zeroPlane}`);
    adapter.dispose();
  }
}

// ---------------------------------------------------------------------------------------------------------------- blue noise A/B

function relMse(test: Float32Array, ref: Float32Array): number {
  let sum = 0;
  let count = 0;
  for (let i = 0; i < test.length; i += 4) {
    for (let c = 0; c < 3; c++) {
      const r = ref[i + c] as number;
      const d = (test[i + c] as number) - r;
      sum += (d * d) / (r * r + 1e-2);
      count++;
    }
  }
  return sum / count;
}

/** Separable Gaussian blur of an RGBA float image (alpha untouched), clamped edges. */
function blur(img: Float32Array, w: number, h: number, sigma: number): Float32Array {
  const radius = Math.ceil(3 * sigma);
  const k = Array.from({ length: 2 * radius + 1 }, (_, i) => Math.exp(-((i - radius) ** 2) / (2 * sigma * sigma)));
  const sum = k.reduce((a, b) => a + b, 0);
  const kernel = k.map((v) => v / sum);
  const tmp = new Float32Array(img.length);
  const out = new Float32Array(img.length);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++)
      for (let c = 0; c < 3; c++) {
        let acc = 0;
        for (let t = -radius; t <= radius; t++) acc += (kernel[t + radius] as number) * (img[(y * w + Math.min(w - 1, Math.max(0, x + t))) * 4 + c] as number);
        tmp[(y * w + x) * 4 + c] = acc;
      }
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++)
      for (let c = 0; c < 3; c++) {
        let acc = 0;
        for (let t = -radius; t <= radius; t++) acc += (kernel[t + radius] as number) * (tmp[(Math.min(h - 1, Math.max(0, y + t)) * w + x) * 4 + c] as number);
        out[(y * w + x) * 4 + c] = acc;
      }
  return out;
}

async function checkBlueNoise(n: number, size: number, biasTol: number, blurFrames: number): Promise<void> {
  const scene = sized(getScene('cornell'), size, size);
  const w = size;
  const h = size;
  // reference: progressive Owen-Sobol at the same depth
  const refRun = new PtMegakernelAdapter({ nee: true, maxDepth: 2, rrStartDepth: 0, passesPerSubmit: 16 });
  await refRun.load(scene);
  const ref = (await refRun.render({ spp: 2048 })).image.data;
  refRun.dispose();
  const refBlur2 = blur(ref, w, h, 2);
  const refBlur4 = blur(ref, w, h, 4);
  const variants: Array<[string, PtAdapterSettings]> = [
    ['owen_sobol', { sampler: 'owen-sobol' }],
    ['pcg', { sampler: 'pcg' }],
    ['bluenoise_kronecker', { sampler: 'bluenoise', blueNoise: BN_NAME, blueNoiseShift: 'kronecker', blueNoisePairs: PAIRS }],
    ['bluenoise_owen', { sampler: 'bluenoise', blueNoise: BN_NAME, blueNoiseShift: 'owen', blueNoisePairs: PAIRS }],
  ];
  const cameras = Array.from({ length: n }, () => scene.camera as PtLookAtCamera);
  for (const [label, settings] of variants) {
    const { adapter, kernel, device } = await loadAdapter(scene, { frame: true, maxDepth: 2, rrStartDepth: 0, ...settings });
    const frames = await renderPath(kernel, device, cameras);
    const mean = new Float32Array(w * h * 4);
    for (const fr of frames) for (let i = 0; i < mean.length; i++) mean[i]! += (fr.radiance[i] as number) / n;
    let sMean = 0;
    let sRef = 0;
    for (let i = 0; i < mean.length; i += 4) for (let c = 0; c < 3; c++) {
      sMean += mean[i + c] as number;
      sRef += ref[i + c] as number;
    }
    const bias = sMean / sRef - 1;
    let raw = 0;
    let b2 = 0;
    let b4 = 0;
    const m = Math.min(blurFrames, n);
    for (let f = 0; f < m; f++) {
      const img = (frames[f] as PtFrame).radiance;
      raw += relMse(img, ref) / m;
      b2 += relMse(blur(img, w, h, 2), refBlur2) / m;
      b4 += relMse(blur(img, w, h, 4), refBlur4) / m;
    }
    metrics[`bn_${label}_rel_bias`] = bias;
    metrics[`bn_${label}_mean_relmse`] = relMse(mean, ref);
    metrics[`bn_${label}_frame_relmse_raw`] = raw;
    metrics[`bn_${label}_frame_relmse_blur2`] = b2;
    metrics[`bn_${label}_frame_relmse_blur4`] = b4;
    metrics[`bn_${label}_blur2_over_raw`] = b2 / raw;
    metrics[`bn_${label}_blur4_over_raw`] = b4 / raw;
    metrics[`bn_${label}_bias_mismatch`] = Math.abs(bias) > biasTol ? 1 : 0;
    if (Math.abs(bias) > biasTol) note(`bluenoise ${label}: ${n}-frame mean is ${(100 * bias).toFixed(2)}% off the reference (> ${100 * biasTol}%)`);
    adapter.dispose();
  }
  for (const label of ['bluenoise_kronecker', 'bluenoise_owen']) {
    // the point of blue noise: after a low-pass the error is lower than Owen-Sobol's (a 1.15x floor; 1.3-1.5x measured on Cornell)
    metrics[`bn_${label}_gain_mismatch`] = metrics['bn_owen_sobol_frame_relmse_blur2']! / metrics[`bn_${label}_frame_relmse_blur2`]! < 1.15 ? 1 : 0;
    metrics[`bn_${label}_blur2_gain_vs_owen_sobol`] = metrics['bn_owen_sobol_frame_relmse_blur2']! / metrics[`bn_${label}_frame_relmse_blur2`]!;
    metrics[`bn_${label}_blur4_gain_vs_owen_sobol`] = metrics['bn_owen_sobol_frame_relmse_blur4']! / metrics[`bn_${label}_frame_relmse_blur4`]!;
    metrics[`bn_${label}_mean_gain_vs_owen_sobol`] = metrics['bn_owen_sobol_mean_relmse']! / metrics[`bn_${label}_mean_relmse`]!;
  }
}

// ---------------------------------------------------------------------------------------------------------------- timing (best effort; the quiet pass owns the numbers)

/** ms per 1 spp dispatch, `batch` dispatches back to back in one command buffer, nothing read back inside the timed region. */
async function timeVariant(scene: SceneDesc, settings: PtAdapterSettings, batch: number): Promise<number> {
  const { adapter, kernel, device } = await loadAdapter(scene, { maxDepth: 2, rrStartDepth: 0, ...settings });
  const frame = !!settings.frame || settings.mode === 'realtime';
  const cams = orbit(scene, 2, 0.5);
  kernel.setCamera(cams[0] as PtLookAtCamera);
  const submit = (n: number) => {
    const encoder = device.createCommandEncoder();
    if (frame) for (let i = 0; i < n; i++) {
      kernel.nextFrame(cams[i & 1] as PtLookAtCamera);
      kernel.encode(encoder, 1);
    } else kernel.encode(encoder, n);
    device.queue.submit([encoder.finish()]);
  };
  submit(8);
  await device.queue.onSubmittedWorkDone();
  const t0 = performance.now();
  submit(batch);
  await device.queue.onSubmittedWorkDone();
  const ms = (performance.now() - t0) / batch;
  adapter.dispose();
  return ms;
}

async function checkTiming(sceneNames: string[], rounds: number, batch: number): Promise<void> {
  const variants: Array<[string, PtAdapterSettings]> = [
    ['plain_owen', { sampler: 'owen-sobol' }],
    ['frame_owen', { sampler: 'owen-sobol', frame: true }],
    ['frame_bluenoise', { sampler: 'bluenoise', blueNoise: BN_NAME, frame: true }],
    ['plain_bluenoise', { sampler: 'bluenoise', blueNoise: BN_NAME }],
    ['realtime_mode', { mode: 'realtime' }],
  ];
  for (const name of sceneNames) {
    const scene = getScene(name);
    const times = new Map<string, number[]>(variants.map(([k]) => [k, []]));
    for (let r = 0; r < rounds; r++) {
      const order = r % 2 === 0 ? variants : [...variants].reverse();
      for (const [k, st] of order) times.get(k)!.push(await timeVariant(scene, st, batch));
    }
    const key = name.replace(/-/g, '_');
    for (const [k, v] of times) {
      metrics[`timing_${key}_${k}_ms_min`] = Math.min(...v);
      metrics[`timing_${key}_${k}_ms_max`] = Math.max(...v);
    }
    for (const k of ['frame_owen', 'frame_bluenoise', 'plain_bluenoise', 'realtime_mode']) {
      const ratios = (times.get(k) as number[]).map((v, i) => v / (times.get('plain_owen') as number[])[i]!);
      metrics[`timing_${key}_${k}_over_plain_owen_min`] = Math.min(...ratios);
      metrics[`timing_${key}_${k}_over_plain_owen_max`] = Math.max(...ratios);
    }
  }
}

try {
  const args = getArgs();
  if (!navigator.gpu) throw new Error('navigator.gpu is undefined: no WebGPU in this browser.');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('requestAdapter() returned null: no usable GPU.');
  adapterSummary = summarizeAdapter(adapter);
  const only = args.check ? args.check.split(',') : ['frames', 'emission', 'bluenoise'];
  if (args.scheduler === 'wavefront') schedulerUnderTest = 'wavefront';
  if (args.restir) restirUnderTest = args.restir === 'biased' ? {} : { bias: 'unbiased', initialVisibility: false, ...(args.restir === 'shared' ? { sharedHit: true } : {}) };
  if (only.includes('timing')) metrics.timing_best_effort = 1;
  const frames = numberArg(args, 'frames', 16);
  if (only.includes('frames')) await checkFrames((args.scenes ?? 'cornell,sponza-white').split(','), frames, numberArg(args, 'step', 1), numberArg(args, 'write', 1) !== 0, numberArg(args, 'size', 128));
  if (only.includes('emission')) await checkEmission(numberArg(args, 'emission_size', 128));
  if (only.includes('bluenoise')) await checkBlueNoise(numberArg(args, 'n', 128), numberArg(args, 'bn_size', 96), numberArg(args, 'bias_tol', 0.02), numberArg(args, 'blur_frames', 16));
  if (only.includes('timing')) await checkTiming((args.scenes ?? 'cornell,sponza-white').split(','), numberArg(args, 'rounds', 3), numberArg(args, 'batch', 32));
  const total = Object.entries(metrics).filter(([k]) => k.endsWith('_mismatch')).reduce((sum, [, v]) => sum + v, 0);
  reportResult(
    { scene: 'realtime-check', renderer: 'ours', config: { frames, checks: only.join(','), scenes: args.scenes ?? 'cornell,sponza-white', restir: args.restir ?? 'off', scheduler: schedulerUnderTest }, valid: total === 0 && deviceErrors.length === 0, metrics: { ...metrics, mismatches: total }, notes: { problems } },
    { adapter: adapterSummary, deviceErrors },
  );
  if (total > 0) throw new Error(`realtime-check: ${total} mismatches. First:\n${problems.join('\n')}`);
} catch (error) {
  reportError(error, { adapter: adapterSummary, deviceErrors });
}
