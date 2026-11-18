/**
 * realtime-pipeline (P4.5 / P4.6 / P4.7; docs/design/realtime.md section 5): the library's real-time path through the shared chain, NO cache.
 *
 *   node bench/runner/run.ts --page realtime-pipeline --timeout 3600000 --load-wait 0 --args scenes=sponza-white scenarios=static,path,timing
 *
 * Kernel (`mode: 'realtime'`, frame planes, maxDepth 2, optional ReSTIR) at `res` (960x540) -> Denoiser SVGF -> [reactive mask] -> pmndrs/upscaler, 2x
 * (1920x1080) -> tonemap. References are OUR OWN plain path tracer at the same path length (maxDepth 2, no RR), rendered at the display size and box-
 * resampled to the render size, so every stage is compared with the same estimator's expectation (the pbrt / Mitsuba references are 8 vertices).
 *
 *  static   a cut to the scene's camera, `frames` frames, per variant; checkpoints `checkpoints` (frames after the cut): raw / denoised / final against the
 *           reference: relMSE, relBias, LDR-FLIP (flip=1; final and denoised only). Flicker (per-pixel mean |dL| over frames stableFrom..frames / reference luma)
 *           per stage. Lag = first checkpoint whose final relMSE is within 10% of the last.
 *  control  N11.5: the upscaler alone, fed the noise-free reference (resampled to the render size) for `controlFrames` static frames: its mean bias.
 *  path     a `pathFrames`-frame orbit + dolly around the camera target, per variant: boil (Denoiser scorecard calm, bench/realtime/boil.ts) of the final and the
 *           denoised stage, k = 1 and 4; per-pose error against our own references (`pathRefs` poses, `pathRefSpp` spp) of the final stage.
 *  timing   provisional per-stage wall times (batched submissions, `session.timeStages`) at the render / display size.
 *
 * Args: scheduler=megakernel|wavefront (the kernel of the chain; P4.1b / P4.2a)  scenes=sponza-white,sponza-lamps  restir=off,on  scenarios=static,control,path,timing  variants=svgf,none (static)  pathVariants=svgf,svgf-reactive,oidn-fast,none
 *       res=960x540 (display 2x; a scene whose film is not 16:9 uses its film as the render size)  frames=64  checkpoints=1,2,4,8,16,32,64  stableFrom=32
 *       pathFrames=64  pathRefs=3  pathRefSpp=512  refSpp=2048  refFile=<url> (static reference of the FIRST scene)  flip=1  upload=1  maxDepth=2
 *       emitThreshold=0 (>0: luminance of the reference above which a pixel is an emitter: per-region relMSE / flicker, `_emit` / `_rest`)  ad=frameEmission:none,k:v (extra adapter settings of the kernel)  candidates=8 spatial=1 bias=biased|unbiased vis=0|1 (ReSTIR initialVisibility)  reactiveCap=8  timingMin=250  timingVariant=svgf|oidn-fast|none (the chain's denoiser in the timing scenario)
 */
import type { PtLookAtCamera, PtMegakernel, PtRestirOptions } from '@pmndrs/pathtracer';
import { PtMegakernelAdapter } from '../adapters/pt-megakernel/index.ts';
import { compareImages, relMseMap } from '../metrics/metrics.ts';
import { flipLdr } from '../metrics/flip.ts';
import { readPFM } from '../metrics/pfm.ts';
import type { Image } from '../src/adapter.ts';
import type { AdapterSummary } from '../runner/contract.ts';
import { getArgs, numberArg, reportError, reportResult, uploadImage, watchDevice } from '../runner/page-client.ts';
import { getScene } from '../scenes/index.ts';
import type { SceneDesc } from '../scenes/schema.ts';
import { BoilMeter, tonemappedLuma } from '../realtime/boil.ts';
import { readTextureRgba16f, resample, type PipelineStage } from '../realtime/chain.ts';
import { RealtimePipelineSession } from '../realtime/session.ts';

let adapterSummary: AdapterSummary | null = null;
let deviceErrors: string[] = [];

type Img = { width: number; height: number; channels: number; data: Float32Array };
const metrics: Record<string, number> = {};
const notes: Record<string, unknown> = {};
const STAGES = ['raw', 'denoised', 'final'] as const;

/** Chain variants: the denoiser (and the reactive mask) of the arm. */
const VARIANTS: Record<string, { denoise: 'svgf' | 'oidn' | 'none'; reactive: boolean; oidn?: { moving: 'fast'; stillSamples: number; hold: boolean } }> = {
  svgf: { denoise: 'svgf', reactive: false },
  'svgf-reactive': { denoise: 'svgf', reactive: true },
  'oidn-fast': { denoise: 'oidn', reactive: false, oidn: { moving: 'fast', stillSamples: Infinity, hold: false } },
  none: { denoise: 'none', reactive: false },
};

async function fetchPfmOrNull(path: string): Promise<Img | null> {
  const r = await fetch(path);
  if (!r.ok) return null;
  const buf = new Uint8Array(await r.arrayBuffer());
  if (buf[0] !== 0x50 || (buf[1] !== 0x46 && buf[1] !== 0x66)) return null; // Vite answers unknown paths with index.html
  return readPFM(buf) as unknown as Img;
}

/** Our own reference: the accumulating plain kernel (NEE, maxDepth `maxDepth`, no RR) at `spp`, size w x h, camera `camera`. */
async function ownReference(scene: SceneDesc, w: number, h: number, spp: number, camera: PtLookAtCamera, maxDepth: number, seed = 4242): Promise<Img> {
  const adapter = new PtMegakernelAdapter({ nee: true });
  await adapter.load(scene);
  const kernel = adapter.renderer as PtMegakernel;
  const device = adapter.gpuContext!.device;
  kernel.setFilm(w, h);
  kernel.setCamera(camera);
  kernel.setLimits({ maxDepth, rrStartDepth: 0 });
  kernel.setSeed(seed);
  {
    const e = device.createCommandEncoder();
    kernel.reset(e);
    device.queue.submit([e.finish()]);
  }
  for (let n = 0; n < spp; n += 8) {
    const e = device.createCommandEncoder();
    kernel.encode(e, Math.min(8, spp - n));
    device.queue.submit([e.finish()]);
    await device.queue.onSubmittedWorkDone(); // one ~0.5 s chunk in flight
  }
  const img = await kernel.readImage();
  adapter.dispose();
  return { width: img.width, height: img.height, channels: 4, data: img.data };
}

const lum = (d: Float32Array, i: number, c: number) => 0.2126 * (d[i * c] as number) + 0.7152 * (d[i * c + 1] as number) + 0.0722 * (d[i * c + 2] as number);

/** Emitter mask of a reference: luminance above `threshold`, dilated by one pixel (the silhouette). 1 = emitter pixel. */
function emitterMask(ref: Img, threshold: number): Uint8Array {
  const { width: w, height: h } = ref;
  const hot = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) hot[i] = lum(ref.data, i, ref.channels) > threshold ? 1 : 0;
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      if (!hot[y * w + x]) continue;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) if (x + dx >= 0 && y + dy >= 0 && x + dx < w && y + dy < h) out[(y + dy) * w + x + dx] = 1;
    }
  return out;
}

/** Mean of `values` over the mask pixels (`inside` true) or the others; NaN when empty. */
function maskedMean(values: ArrayLike<number>, mask: Uint8Array, inside: boolean): number {
  let s = 0;
  let n = 0;
  for (let i = 0; i < mask.length; i++) if ((mask[i] === 1) === inside) {
    s += values[i] as number;
    n++;
  }
  return n ? s / n : NaN;
}

/** Pixels of an rgba16float texture readback with a channel at the f16 clamp of `pipePack` (65000 rounds to >= 64992), over the three signal textures. */
async function clampHits(device: GPUDevice, textures: GPUTexture[]): Promise<number> {
  let hits = 0;
  for (const t of textures) {
    const img = await readTextureRgba16f(device, t);
    for (let i = 0; i < img.width * img.height; i++) if ((img.data[i * 4] as number) >= 64000 || (img.data[i * 4 + 1] as number) >= 64000 || (img.data[i * 4 + 2] as number) >= 64000) hits++;
  }
  return hits;
}

/** Orbit + dolly around the camera target: yaw +-5 degrees about the target, 12% dolly in, smoothstep ends. */
function pathCamera(base: PtLookAtCamera, s: number): PtLookAtCamera {
  const k = s * s * (3 - 2 * s);
  const yaw = ((10 * Math.PI) / 180) * (k - 0.5);
  const d = [base.position[0] - base.target[0], base.position[1] - base.target[1], base.position[2] - base.target[2]] as const;
  const scale = 1 - 0.12 * k;
  const c = Math.cos(yaw);
  const sn = Math.sin(yaw);
  const pos: [number, number, number] = [base.target[0] + scale * (d[0] * c + d[2] * sn), base.target[1] + scale * d[1], base.target[2] + scale * (-d[0] * sn + d[2] * c)];
  return { ...base, position: pos };
}

function halfBits(v: number): number {
  const f32 = new Float32Array([v]);
  const x = new Uint32Array(f32.buffer)[0] as number;
  const sign = (x >>> 16) & 0x8000;
  let e = ((x >>> 23) & 0xff) - 127 + 15;
  let m = x & 0x7fffff;
  if (e <= 0) return e < -10 ? sign : sign | ((m | 0x800000) >> (14 - e));
  if (e >= 31) return sign | 0x7c00;
  m = m + 0x1000;
  if (m & 0x800000) {
    m = 0;
    e++;
  }
  return sign | (e << 10) | (m >> 13);
}

/** Linear RGBA f32 -> an rgba16float texture (row 0 = top). */
function writeRgba16f(device: GPUDevice, texture: GPUTexture, img: Img): void {
  const { width: w, height: h } = img;
  const half = new Uint16Array(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    for (let c = 0; c < 3; c++) half[i * 4 + c] = halfBits(img.data[i * img.channels + c] as number);
    half[i * 4 + 3] = 0x3c00;
  }
  device.queue.writeTexture({ texture }, half, { bytesPerRow: w * 8 }, { width: w, height: h });
}

async function readR32f(device: GPUDevice, texture: GPUTexture): Promise<Float32Array> {
  const { width: w, height: h } = texture;
  const stride = Math.ceil((w * 4) / 256) * 256;
  const staging = device.createBuffer({ size: stride * h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  enc.copyTextureToBuffer({ texture }, { buffer: staging, bytesPerRow: stride, rowsPerImage: h }, { width: w, height: h });
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const raw = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) out[y * w + x] = raw[(y * stride) / 4 + x] as number;
  return out;
}

try {
  const args = getArgs();
  const sceneNames = (args.scenes ?? 'sponza-white').split(',');
  const restirModes = (args.restir ?? 'off').split(',');
  const scenarios = (args.scenarios ?? 'static,path,timing').split(',');
  const staticVariants = (args.variants ?? 'svgf,none').split(',');
  const pathVariants = (args.pathVariants ?? 'svgf,svgf-reactive,oidn-fast,none').split(',');
  const frames = numberArg(args, 'frames', 64);
  const checkpoints = (args.checkpoints ?? '1,2,4,8,16,32,64').split(',').map(Number).filter((f) => f <= frames);
  const stableFrom = numberArg(args, 'stableFrom', Math.min(32, frames - 1));
  const pathFrames = numberArg(args, 'pathFrames', 64);
  const pathRefs = numberArg(args, 'pathRefs', 3);
  const pathRefSpp = numberArg(args, 'pathRefSpp', 512);
  const refSpp = numberArg(args, 'refSpp', 2048);
  const maxDepth = numberArg(args, 'maxDepth', 2);
  const doFlip = numberArg(args, 'flip', 1) === 1;
  const doUpload = numberArg(args, 'upload', 1) === 1;
  const controlFrames = numberArg(args, 'controlFrames', 32);
  const reactiveCap = numberArg(args, 'reactiveCap', 8);
  const emitThreshold = numberArg(args, 'emitThreshold', 0);
  const adapterExtra = Object.fromEntries((args.ad ?? '').split(',').filter(Boolean).map((kv) => { const [k, v] = kv.split(':'); return [k, v === 'true' ? true : v === 'false' ? false : Number.isFinite(Number(v)) ? Number(v) : v]; }));
  const needReactive = [...staticVariants, ...pathVariants, args.timingVariant ?? ''].includes('svgf-reactive');
  const restirOptions = (): PtRestirOptions => ({ candidates: numberArg(args, 'candidates', 8), spatial: numberArg(args, 'spatial', 1), bias: (args.bias ?? 'biased') as 'biased' | 'unbiased', ...(args.vis !== undefined ? { initialVisibility: args.vis === '1' } : {}) });
  const rows: string[] = [];
  const pathRows: string[] = [];
  let firstConfig = true;

  for (const sceneName of sceneNames) {
    const scene = getScene(sceneName);
    const film = scene.film;
    const wide = Math.abs(film.width / film.height - 16 / 9) < 0.01;
    const [width, height] = wide ? ((args.res ?? '960x540').split('x').map(Number) as [number, number]) : [film.width, film.height];
    const displayWidth = 2 * width;
    const displayHeight = 2 * height;
    const baseCam = scene.camera as PtLookAtCamera;
    // Static reference (display size, box-resampled to the render size). Rendered once per scene.
    let staticRef: Img | null = null;
    let refRender: Img | null = null;
    let refDisplay: Img | null = null;
    let masks: Record<PipelineStage, Uint8Array> | null = null;
    const needStatic = scenarios.some((s) => ['static', 'control'].includes(s));
    if (needStatic) {
      if (args.refFile && sceneName === sceneNames[0]) {
        staticRef = await fetchPfmOrNull(args.refFile);
        if (!staticRef) throw new Error(`refFile ${args.refFile}: not found`);
        notes[`${sceneName}_reference`] = `refFile ${args.refFile}`;
      } else {
        const t0 = performance.now();
        staticRef = await ownReference(scene, displayWidth, displayHeight, refSpp, baseCam, maxDepth);
        notes[`${sceneName}_reference`] = `ours: plain megakernel, NEE, maxDepth ${maxDepth}, no RR, ${refSpp} spp, ${displayWidth}x${displayHeight}`;
        notes[`${sceneName}_reference_render_s`] = (performance.now() - t0) / 1000;
        if (doUpload) await uploadImage(`reference-${sceneName}-${refSpp}spp`, staticRef as never);
      }
      refDisplay = staticRef.width === displayWidth ? staticRef : resample(staticRef, displayWidth, displayHeight);
      refRender = resample(refDisplay, width, height);
      let mean = 0;
      for (let i = 0; i < refDisplay.width * refDisplay.height; i++) mean += lum(refDisplay.data, i, refDisplay.channels);
      metrics[`${sceneName}_reference_mean_luma`] = mean / (refDisplay.width * refDisplay.height);
      if (emitThreshold > 0) {
        masks = { final: emitterMask(refDisplay, emitThreshold), denoised: emitterMask(refRender, emitThreshold), raw: emitterMask(refRender, emitThreshold) };
        metrics[`${sceneName}_emitter_pixel_frac`] = (masks.final as Uint8Array).reduce((a, b) => a + b, 0) / (refDisplay.width * refDisplay.height);
      }
    }
    const refFor = (stage: PipelineStage) => (stage === 'final' ? (refDisplay as Img) : (refRender as Img));

    // Path references (our own, a few poses at the display size).
    const poses = Array.from({ length: pathRefs }, (_, i) => Math.round(((i + 1) * (pathFrames - 1)) / pathRefs));
    const pathRefImgs = new Map<number, Img>();
    if (scenarios.includes('path') && pathRefs > 0) {
      for (const f of poses) {
        const t0 = performance.now();
        const r = await ownReference(scene, displayWidth, displayHeight, pathRefSpp, pathCamera(baseCam, f / (pathFrames - 1)), maxDepth, 777 + f);
        pathRefImgs.set(f, r);
        notes[`${sceneName}_pathref_${f}_s`] = (performance.now() - t0) / 1000;
        if (doUpload) await uploadImage(`pathref-${sceneName}-f${f}-${pathRefSpp}spp`, r as never);
      }
    }

    for (const restirMode of restirModes) {
      const tag = `${sceneName}_${restirMode === 'off' ? 'nee' : 'restir'}`;
      const session = await RealtimePipelineSession.create({ scene, width, height, displayWidth, displayHeight, maxDepth, adapter: { ...adapterExtra, ...(args.scheduler === 'wavefront' ? { scheduler: 'wavefront' } : {}) }, ...(restirMode === 'on' ? { restir: restirOptions() } : {}), ...(needReactive ? { reactive: { cap: reactiveCap } } : {}) });
      const device = session.device;
      deviceErrors = watchDevice(device);
      adapterSummary = session.adapter.gpuContext!.summary;
      notes[`${tag}_sampler`] = 'bluenoise (realtime preset; ReSTIR binds the texture too since P4.2e)';
      notes[`${tag}_scheduler`] = args.scheduler === 'wavefront' ? 'wavefront (plain, P4.1b / P4.2a)' : 'megakernel';
      notes[`${tag}_adapter`] = JSON.stringify(session.adapter.info.config);
      const setVariant = (name: string) => {
        const v = VARIANTS[name];
        if (!v) throw new Error(`unknown variant ${name} (${Object.keys(VARIANTS).join(', ')})`);
        session.setDenoiser(v.denoise, v.oidn ?? {});
        session.arm.reactiveEnabled = v.reactive;
      };
      const measure = async (prefix: string, upload: boolean, stages: readonly PipelineStage[] = STAGES): Promise<void> => {
        await device.queue.onSubmittedWorkDone();
        for (const stage of stages) {
          const img = await readTextureRgba16f(device, session.arm.stage(stage));
          const r = refFor(stage);
          const m = compareImages(img as unknown as Image, r as unknown as Image);
          metrics[`${prefix}_${stage}_relmse`] = m.relMse;
          metrics[`${prefix}_${stage}_relbias`] = m.relBias;
          if (doFlip && stage !== 'raw') metrics[`${prefix}_${stage}_flip`] = flipLdr(img as unknown as Image, r as unknown as Image).mean;
          if (masks) {
            const map = relMseMap(img as unknown as Image, r as unknown as Image);
            metrics[`${prefix}_${stage}_relmse_emit`] = maskedMean(map, masks[stage], true);
            metrics[`${prefix}_${stage}_relmse_rest`] = maskedMean(map, masks[stage], false);
            // energy in the emitter region and outside it: sum(test) / sum(reference) - 1 (the emitters dominate the total of a scene with lamps: a global relBias is mostly theirs)
            for (const inside of [true, false]) {
              let st = 0;
              let sr = 0;
              for (let i = 0; i < masks[stage].length; i++) if ((masks[stage][i] === 1) === inside) for (let c = 0; c < 3; c++) {
                st += img.data[i * img.channels + c] as number;
                sr += r.data[i * r.channels + c] as number;
              }
              metrics[`${prefix}_${stage}_relbias_${inside ? 'emit' : 'rest'}`] = st / sr - 1;
            }
          }
          if (upload && doUpload) await uploadImage(`${prefix}-${stage}`, img);
        }
        metrics[`${prefix}_clamp_hits`] = await clampHits(device, [session.arm.tex.diffuse, session.arm.tex.specular, session.arm.tex.raw]);
      };

      // ---- static
      if (scenarios.includes('static')) {
        for (const variant of staticVariants) {
          setVariant(variant);
          session.cutNext();
          for (let f = 1; f <= frames; f++) {
            await session.frame(baseCam, { tonemap: f === frames });
            if (f === stableFrom) session.arm.startFlicker();
            if (f >= stableFrom) session.arm.stepFlicker();
            if (checkpoints.includes(f)) await measure(`${tag}_static_${variant}_f${f}`, f === 1 || f === frames);
            else await device.queue.onSubmittedWorkDone();
          }
          if (frames > stableFrom) {
            const fl = await session.arm.readFlicker();
            for (const stage of STAGES) {
              const { map, width: fw, height: fh } = fl[stage];
              const r = refFor(stage);
              let s = 0;
              for (let i = 0; i < map.length; i++) s += (map[i] as number) / (lum(r.data, i, r.channels) + 0.01);
              if (r.width !== fw || r.height !== fh) throw new Error('flicker: size mismatch');
              metrics[`${tag}_static_${variant}_${stage}_flicker`] = s / map.length;
              if (masks) {
                const rel = new Float32Array(map.length);
                for (let i = 0; i < map.length; i++) rel[i] = (map[i] as number) / (lum(r.data, i, r.channels) + 0.01);
                metrics[`${tag}_static_${variant}_${stage}_flicker_emit`] = maskedMean(rel, masks[stage], true);
                metrics[`${tag}_static_${variant}_${stage}_flicker_rest`] = maskedMean(rel, masks[stage], false);
                metrics[`${tag}_static_${variant}_${stage}_absflicker_emit`] = maskedMean(map, masks[stage], true);
              }
            }
          }
          // lag: first checkpoint within 10% of the last checkpoint's final relMSE
          const last = metrics[`${tag}_static_${variant}_f${checkpoints[checkpoints.length - 1]}_final_relmse`] as number;
          const lagAt = checkpoints.find((f) => (metrics[`${tag}_static_${variant}_f${f}_final_relmse`] as number) <= 1.1 * last);
          metrics[`${tag}_static_${variant}_lag_frames`] = lagAt ?? -1;
          for (const f of checkpoints) {
            const k = `${tag}_static_${variant}_f${f}`;
            rows.push(`| ${tag} | ${variant} | ${f} | ${(metrics[`${k}_raw_relmse`] as number).toExponential(2)} | ${(metrics[`${k}_denoised_relmse`] as number).toExponential(2)} | ${(metrics[`${k}_final_relmse`] as number).toExponential(2)} | ${(100 * (metrics[`${k}_raw_relbias`] as number)).toFixed(1)}% / ${(100 * (metrics[`${k}_denoised_relbias`] as number)).toFixed(1)}% / ${(100 * (metrics[`${k}_final_relbias`] as number)).toFixed(1)}% | ${doFlip ? (metrics[`${k}_final_flip`] as number).toFixed(4) : '-'} |`);
          }
        }
        // The plane identity of the real-time kernel (diffuse x albedo + specular x specFactor + emission = beauty).
        const fr = await session.kernel.readFrame();
        let worst = 0;
        const n = fr.width * fr.height;
        for (let i = 0; i < n; i++)
          for (let c = 0; c < 3; c++) {
            const j = i * 4 + c;
            const b = fr.radiance[j] as number;
            worst = Math.max(worst, Math.abs((fr.diffuse[j] as number) * (fr.albedo[j] as number) + (fr.specular[j] as number) * (fr.specularFactor[j] as number) + (fr.emission[j] as number) - b) / (Math.abs(b) + 1e-3));
          }
        metrics[`${tag}_plane_identity_max_rel`] = worst;
      }

      // ---- control (N11.5): the upscaler alone on a noise-free input
      if (scenarios.includes('control') && restirMode === 'off') {
        setVariant('none');
        session.cutNext();
        await session.frame(baseCam); // depth / motion guides of the reference view
        for (let f = 0; f < controlFrames; f++) {
          writeRgba16f(device, session.arm.color, refRender as Img);
          session.arm.upscale(baseCam, f === 0);
          await device.queue.onSubmittedWorkDone();
        }
        const img = await readTextureRgba16f(device, session.arm.stage('final'));
        const m = compareImages(img as unknown as Image, refDisplay as unknown as Image);
        metrics[`${tag}_control_final_relbias`] = m.relBias;
        metrics[`${tag}_control_final_relmse`] = m.relMse;
        // Baseline: the same chain without the upscaler's temporal path = the reference through the f16 round trip and the 2x resampling.
        const soft = resample(refRender as Img, displayWidth, displayHeight);
        const ms = compareImages(soft as unknown as Image, refDisplay as unknown as Image);
        metrics[`${tag}_control_resample_relbias`] = ms.relBias;
        metrics[`${tag}_control_resample_relmse`] = ms.relMse;
        if (doUpload) await uploadImage(`${tag}-control-final`, img);
      }

      // ---- path
      if (scenarios.includes('path')) {
        for (const variant of pathVariants) {
          setVariant(variant);
          const meters = { final1: new BoilMeter(1, 2), final4: new BoilMeter(4, 2), den1: new BoilMeter(1, 1), albedo1: new BoilMeter(1, 1) };
          session.cutNext();
          const errs: string[] = [];
          for (let f = 0; f < pathFrames; f++) {
            const cam = pathCamera(baseCam, f / (pathFrames - 1));
            await session.frame(cam, { tonemap: false });
            await device.queue.onSubmittedWorkDone();
            // Readbacks (outside any timing): final + denoised luma, the motion planes of the frame.
            const fin = await readTextureRgba16f(device, session.arm.stage('final'));
            const den = await readTextureRgba16f(device, session.arm.stage('denoised'));
            const mo = await readTextureRgba16f(device, session.arm.tex.motion);
            const vz = await readR32f(device, session.arm.tex.viewZ);
            const mz = await readR32f(device, session.arm.tex.motionZ);
            const alb = await readTextureRgba16f(device, session.arm.tex.albedo);
            const motion = new Float32Array(width * height * 2);
            for (let i = 0; i < width * height; i++) {
              motion[i * 2] = mo.data[i * 4] as number;
              motion[i * 2 + 1] = mo.data[i * 4 + 1] as number;
            }
            const guides = { motion, viewZ: vz, motionZ: mz, mw: width, mh: height };
            const lf = tonemappedLuma(fin.data, displayWidth, displayHeight);
            if (f === 0) for (const m of Object.values(meters)) m.restart();
            meters.final1.push({ luma: lf, width: displayWidth, height: displayHeight, ...guides });
            meters.final4.push({ luma: lf, width: displayWidth, height: displayHeight, ...guides });
            meters.den1.push({ luma: tonemappedLuma(den.data, width, height), width, height, ...guides });
            meters.albedo1.push({ luma: tonemappedLuma(alb.data, width, height), width, height, ...guides });
            const ref = pathRefImgs.get(f);
            if (ref) {
              const m = compareImages(fin as unknown as Image, ref as unknown as Image);
              const fl = doFlip ? flipLdr(fin as unknown as Image, ref as unknown as Image).mean : null;
              const key = `${tag}_path_${variant}_pose${f}`;
              metrics[`${key}_final_relmse`] = m.relMse;
              metrics[`${key}_final_relbias`] = m.relBias;
              if (fl !== null) metrics[`${key}_final_flip`] = fl;
              errs.push(`f${f}: relMSE ${m.relMse.toExponential(2)}, bias ${(100 * m.relBias).toFixed(1)}%${fl !== null ? `, FLIP ${fl.toFixed(4)}` : ''}`);
              if (doUpload) await uploadImage(`${tag}-path-${variant}-f${f}-final`, fin);
            }
          }
          for (const [name, m] of Object.entries(meters)) {
            const r = m.result();
            metrics[`${tag}_path_${variant}_boil_${name}`] = r.boil;
            metrics[`${tag}_path_${variant}_boil_${name}_valid`] = r.validFrac;
          }
          pathRows.push(`| ${tag} | ${variant} | ${(metrics[`${tag}_path_${variant}_boil_final1`] as number).toFixed(4)} | ${(metrics[`${tag}_path_${variant}_boil_final4`] as number).toFixed(4)} | ${(metrics[`${tag}_path_${variant}_boil_den1`] as number).toFixed(4)} | ${(metrics[`${tag}_path_${variant}_boil_albedo1`] as number).toFixed(4)} | ${(100 * (metrics[`${tag}_path_${variant}_boil_final1_valid`] as number)).toFixed(0)}% | ${errs.join('; ')} |`);
        }
      }

      // ---- timing
      if (scenarios.includes('timing')) {
        setVariant(args.timingVariant ?? 'svgf');
        session.cutNext();
        await session.frame(baseCam);
        const t = await session.timeStages({ minMs: numberArg(args, 'timingMin', 250) });
        for (const [k, v] of Object.entries(t)) {
          metrics[`${tag}_time_${k}_ms`] = v.median;
          metrics[`${tag}_time_${k}_min_ms`] = v.min;
          metrics[`${tag}_time_${k}_max_ms`] = v.max;
        }
        metrics[`${tag}_time_denoise_composite_ms`] = Math.max(0, t.post!.median - t.postNoDenoise!.median);
        metrics[`${tag}_time_upscale_ms`] = t.postNoDenoise!.median;
        metrics[`${tag}_time_fps`] = 1000 / t.frame!.median;
        notes[`${tag}_timing`] = 'PROVISIONAL: wall clock of batched submissions (>= 250 ms per round, 5 rounds, median; no GPU timestamps); machine load recorded by the runner';
      }

      session.destroy();
      firstConfig = false;
    }
  }
  void firstConfig;

  notes.staticTable = ['| config | variant | frame | raw relMSE | denoised relMSE | final relMSE | bias raw / den / final | final FLIP |', '|---|---|---|---|---|---|---|---|', ...rows].join('\n');
  notes.pathTable = ['| config | variant | boil final k=1 | k=4 | denoised k=1 | albedo control k=1 | valid | per-pose final error |', '|---|---|---|---|---|---|---|---|', ...pathRows].join('\n');
  if (rows.length) console.log(notes.staticTable);
  if (pathRows.length) console.log(notes.pathTable);

  reportResult(
    { scene: sceneNames.join('+'), renderer: 'realtime-pipeline', config: { scenes: sceneNames, restir: restirModes, scenarios, staticVariants, pathVariants, frames, checkpoints, stableFrom, pathFrames, pathRefs, pathRefSpp, refSpp, maxDepth }, valid: deviceErrors.length === 0, metrics, notes },
    { adapter: adapterSummary, deviceErrors },
  );
} catch (error) {
  reportError(error, { adapter: adapterSummary, deviceErrors });
}
