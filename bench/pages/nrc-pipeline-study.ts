/**
 * nrc-pipeline-study (track N, M9; docs/design/nrc-pipeline.md §6): the headless twin of `nrc-pipeline`.
 *
 *   node bench/runner/run.ts --page nrc-pipeline-study --gpu nvidia --timeout 1800000 [--headed] --args res=960x540 ...
 *
 * Both arms (plain path tracing at N spp per frame, path tracing + cache at 1 spp per frame) through the identical chain (frame planes ->
 * Denoiser SVGF -> composite -> pmndrs/upscaler temporal, 2x). Per checkpoint, every stage (raw and denoised at the render size, final at the
 * display size) of both arms is compared with the reference resampled to that size (relMSE, relBias, LDR-FLIP), and written as linear PFM.
 *
 *  0. Reference: `ref=mitsuba` (default) the Mitsuba 3 render (bench/references/sponza-white/mitsuba3-llvm_ad_rgb-4096spp.pfm, 1280x720, git-ignored:
 *     Mac only); `refFile=<url>` a saved PFM; `ref=ours` our own plain path tracer at refSpp (4096), uploaded as `reference-ours-<spp>spp`.
 *  1. Warm-up: the cache trains `warm` frames at the reference view (a user who has been looking around; nrc-study: steady from frame 1).
 *  2. Equal time: `plainSpp=auto` (default) times one plain sample and one cache frame, each submitted alone (best effort, wall clock), and
 *     sets N = round(cache / plain sample) >= 1; or `plainSpp=<n>`.
 *  3. static: a cut to the reference view, then `frames` frames; checkpoints `checkpoints` (frame numbers after the cut). Stability: per-pixel
 *     mean |L_t - L_(t-1)| over frames `stableFrom`..`frames`, divided by (reference luminance + 0.01), averaged (nrc-study's measure, on the GPU,
 *     read once at the end).
 *  4. path: a cut to a start pose, a `path`-frame dolly + pan to the reference view (histories reproject along it), then a hold; checkpoints
 *     are frames after arrival (`pathCheckpoints`). The mid-path frame is uploaded too (no reference).
 *
 * Args: res=960x540 (display = 2x)  warm=300  frames=64  checkpoints=1,4,16,64  stableFrom=16  path=48  pathCheckpoints=1,4,16  plainSpp=auto|n
 *       infer=tiled|portable|sgm  precision=f16  steps=4  ref=mitsuba|ours  refFile=<url>  refSpp=4096  flip=1  upload=1  scenarios=static,path
 *       sharpness=<0..1>  autoExposure=0|1  maxAccumulation=<n>  (upscaler A/B)
 *       denoiser=svgf|oidn|oidn-fast|oidn-std|none  oidnAux=albedo+normal|albedo  stillSamples=8  hold=1  oidnStill=balanced|high
 *  5. coldstart (N11.6): per warm-up W in `warmups` (0,10,30,100,300): fresh network, W training frames at the view, a cut, then frames `coldFrames` (1,4).
 * Timing rows are wall clock per submit (best effort); the quality numbers do not depend on timing once N is fixed.
 */
import type { PtLookAtCamera, PtMegakernel } from '@pmndrs/pathtracer';
import { PtMegakernelAdapter } from '../adapters/pt-megakernel/index.ts';
import { compareImages } from '../metrics/metrics.ts';
import { flipLdr } from '../metrics/flip.ts';
import { readPFM } from '../metrics/pfm.ts';
import type { Image } from '../src/adapter.ts';
import type { AdapterSummary } from '../runner/contract.ts';
import { getArgs, numberArg, reportError, reportResult, uploadImage, watchDevice } from '../runner/page-client.ts';
import { sponzaWhite } from '../scenes/tier2/sponza-white.ts';
import { NrcPipelineSession, type ArmName } from '../nrc/pipeline-session.ts';
import { readTextureRgba16f, resample, type DenoiserKind, type OidnArmSettings, type OidnQuality, type PipelineStage } from '../nrc/pipeline.ts';

let adapterSummary: AdapterSummary | null = null;

/**
 * `denoiser=svgf` (default) | `oidn` (small net until `stillSamples` still frames, then the standard net once, then held) | `oidn-fast` / `oidn-std`
 * (that net every frame, no hold) | `none`. Every OIDN mode denoises the progressive mean of the frames since the camera last moved (1 frame while
 * it moves), so a still camera's frame 16 is a 16-frame mean; `oidnAux=albedo+normal|albedo`, `stillSamples=8`, `hold=1`, `oidnStill=balanced|high`.
 */
function denoiserOptions(args: Record<string, string>): { denoise: DenoiserKind; oidn?: Partial<OidnArmSettings> } {
  const d = args.denoiser ?? 'svgf';
  const aux = args.oidnAux === 'albedo' ? 'albedo' : 'albedo+normal';
  if (d === 'oidn') return { denoise: 'oidn', oidn: { aux, stillSamples: numberArg(args, 'stillSamples', 8), hold: (args.hold ?? '1') === '1', ...(args.oidnStill ? { still: args.oidnStill as OidnQuality } : {}) } };
  if (d === 'oidn-fast') return { denoise: 'oidn', oidn: { aux, moving: 'fast', stillSamples: Infinity, hold: false } };
  if (d === 'oidn-std') return { denoise: 'oidn', oidn: { aux, moving: 'balanced', stillSamples: Infinity, hold: false } };
  if (d === 'none' || d === 'svgf') return { denoise: d };
  throw new Error(`denoiser must be svgf, oidn, oidn-fast, oidn-std or none, got ${d}`);
}
let deviceErrors: string[] = [];

type Img = { width: number; height: number; channels: number; data: Float32Array };

async function fetchPfmOrNull(path: string): Promise<Img | null> {
  const r = await fetch(path);
  if (r.status === 404) return null;
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  const buf = new Uint8Array(await r.arrayBuffer());
  // Vite answers unknown paths with index.html (200): a PFM starts with "PF" or "Pf".
  if (buf[0] !== 0x50 || (buf[1] !== 0x46 && buf[1] !== 0x66)) return null;
  return readPFM(buf) as unknown as Img;
}

const lum = (d: Float32Array, i: number, c: number) => 0.2126 * (d[i * c] as number) + 0.7152 * (d[i * c + 1] as number) + 0.0722 * (d[i * c + 2] as number);

/** Our own reference: the plain path tracer (accumulating kernel, same scene settings) at `spp`, 1280x720, the reference camera. */
async function renderOwnReference(spp: number): Promise<Img> {
  const adapter = new PtMegakernelAdapter({ nee: true });
  await adapter.load(sponzaWhite);
  const kernel = adapter.renderer as PtMegakernel;
  const device = adapter.gpuContext!.device;
  kernel.setFilm(1280, 720);
  kernel.setCamera(sponzaWhite.camera as PtLookAtCamera);
  kernel.setLimits({ maxDepth: sponzaWhite.integrator.maxDepth, ...(sponzaWhite.integrator.rrStartDepth ? { rrStartDepth: sponzaWhite.integrator.rrStartDepth } : {}) });
  kernel.setSeed(4242);
  {
    const e = device.createCommandEncoder();
    kernel.reset(e);
    device.queue.submit([e.finish()]);
  }
  // One ~0.5 s chunk in flight at a time (many queued batches can time the device out).
  for (let n = 0; n < spp; n += 8) {
    const e = device.createCommandEncoder();
    kernel.encode(e, Math.min(8, spp - n));
    device.queue.submit([e.finish()]);
    await device.queue.onSubmittedWorkDone();
  }
  const img = await kernel.readImage();
  adapter.dispose();
  return { width: img.width, height: img.height, channels: 4, data: img.data };
}

try {
  const args = getArgs();
  const [width, height] = (args.res ?? '960x540').split('x').map(Number) as [number, number];
  const displayWidth = 2 * width;
  const displayHeight = 2 * height;
  const warm = numberArg(args, 'warm', 300);
  const frames = numberArg(args, 'frames', 64);
  const checkpoints = (args.checkpoints ?? '1,4,16,64').split(',').map(Number).filter((f) => f <= frames);
  const stableFrom = numberArg(args, 'stableFrom', 16);
  const pathFrames = numberArg(args, 'path', 48);
  const pathCheckpoints = (args.pathCheckpoints ?? '1,4,16').split(',').map(Number);
  const steps = numberArg(args, 'steps', 4);
  const doFlip = numberArg(args, 'flip', 1) === 1;
  const doUpload = numberArg(args, 'upload', 1) === 1;
  const scenarios = (args.scenarios ?? 'static,path').split(',');
  const notes: Record<string, unknown> = {};
  const metrics: Record<string, number> = {};
  const rows: { scenario: string; frame: number; arm: ArmName; stage: PipelineStage; relMse: number; relBias: number; flip: number | null }[] = [];

  // 0. Reference. `ref=mitsuba` (default) needs the git-ignored Mitsuba PFM in this checkout (a 404 fails the run: the runner fails on
  // console errors); `refFile=<url>` loads a saved one; `ref=ours` renders our own.
  let ref: Img | null = null;
  let refSource = '';
  const mitsuba = '/bench/references/sponza-white/mitsuba3-llvm_ad_rgb-4096spp.pfm';
  const refMode = args.refFile ? 'file' : (args.ref ?? 'mitsuba');
  if (refMode === 'file') {
    ref = await fetchPfmOrNull(args.refFile as string);
    if (!ref) throw new Error(`refFile ${args.refFile}: not found`);
    refSource = `refFile ${args.refFile}`;
  } else if (refMode === 'mitsuba') {
    ref = await fetchPfmOrNull(mitsuba);
    if (!ref) throw new Error(`${mitsuba} is not in this checkout (references are git-ignored): pass ref=ours or refFile=<url>`);
    refSource = `Mitsuba 3 llvm_ad_rgb 4096 spp (${mitsuba})`;
  } else {
    const spp = numberArg(args, 'refSpp', 4096);
    const t0 = performance.now();
    ref = await renderOwnReference(spp);
    refSource = `ours: plain megakernel, ${spp} spp, 1280x720`;
    notes.reference_render_s = (performance.now() - t0) / 1000;
    if (doUpload) await uploadImage(`reference-ours-${spp}spp`, ref as never);
  }
  notes.reference = refSource;
  const refRender = resample(ref, width, height);
  const refDisplay = resample(ref, displayWidth, displayHeight);
  const refFor = (stage: PipelineStage) => (stage === 'final' ? refDisplay : refRender);

  // Session.
  const session = await NrcPipelineSession.create({
    scene: sponzaWhite,
    width,
    height,
    displayWidth,
    displayHeight,
    infer: args.infer ?? 'tiled',
    precision: args.precision === 'f32' ? 'f32' : 'f16',
    cache: { stepsPerFrame: steps },
    ...denoiserOptions(args),
    // Upscaler A/B knobs (defaults: the upscaler's own: sharpness 0.8, auto exposure, 24 frames).
    upscaler: {
      ...(args.sharpness !== undefined ? { sharpness: Number(args.sharpness) } : {}),
      ...(args.autoExposure !== undefined ? { autoExposure: args.autoExposure === '1' } : {}),
      ...(args.maxAccumulation !== undefined ? { maxAccumulation: Number(args.maxAccumulation) } : {}),
    },
  });
  const device = session.device;
  adapterSummary = session.adapter.gpuContext!.summary;
  deviceErrors = watchDevice(device);
  const refCam = sponzaWhite.camera as PtLookAtCamera;
  notes.infer = session.inferDescription;
  notes.denoiser = session.arms.plain.kind === 'oidn' ? `oidn ${JSON.stringify(session.arms.plain.oidn)}` : session.arms.plain.kind === 'svgf' ? session.arms.plain.denoiser.name : 'none';
  await session.arms.plain.ready();

  // 1. Warm-up at the reference view.
  session.cutNext();
  for (let f = 0; f < warm; f += 25) {
    session.trainOnly(refCam, Math.min(25, warm - f));
    await device.queue.onSubmittedWorkDone();
  }

  // 2. Equal time.
  const t = await session.timeStages(20);
  for (const [k, v] of Object.entries(t)) metrics[`time_${k}${k === 'equalSpp' ? '' : '_ms'}`] = v;
  session.plainSpp = args.plainSpp && args.plainSpp !== 'auto' ? Math.max(1, Math.round(Number(args.plainSpp))) : (t.equalSpp as number);
  metrics.plain_spp = session.plainSpp;
  notes.timing = 'wall clock per submit + onSubmittedWorkDone, median of 20 after 3 warm-up: BEST EFFORT (queue for the quiet pass)';

  const measure = async (scenario: string, frame: number, upload: boolean): Promise<void> => {
    await device.queue.onSubmittedWorkDone();
    for (const arm of ['plain', 'nrc'] as const)
      for (const stage of ['raw', 'denoised', 'final'] as const) {
        const img = await readTextureRgba16f(device, session.arms[arm].stage(stage));
        const r = refFor(stage);
        const m = compareImages(img as unknown as Image, r as unknown as Image);
        const flip = doFlip ? flipLdr(img as unknown as Image, r as unknown as Image).mean : null;
        rows.push({ scenario, frame, arm, stage, relMse: m.relMse, relBias: m.relBias, flip });
        const key = `${scenario}_f${frame}_${arm}_${stage}`;
        metrics[`${key}_relmse`] = m.relMse;
        metrics[`${key}_relbias`] = m.relBias;
        if (flip !== null) metrics[`${key}_flip`] = flip;
        if (upload && doUpload) await uploadImage(`${scenario}-f${frame}-${arm}-${stage}`, img);
      }
  };

  const stability = async (scenario: string): Promise<void> => {
    for (const arm of ['plain', 'nrc'] as const) {
      const f = await session.arms[arm].readFlicker();
      for (const stage of ['raw', 'denoised', 'final'] as const) {
        const { map, width: fw, height: fh, changes } = f[stage];
        const r = refFor(stage);
        if (r.width !== fw || r.height !== fh) throw new Error('stability: size mismatch');
        let s = 0;
        for (let i = 0; i < map.length; i++) s += (map[i] as number) / (lum(r.data, i, r.channels) + 0.01);
        metrics[`${scenario}_${arm}_${stage}_flicker`] = s / map.length;
        metrics[`${scenario}_flicker_frames`] = changes;
      }
    }
  };

  // 3. Static view.
  if (scenarios.includes('static')) {
    session.cutNext();
    for (let f = 1; f <= frames; f++) {
      await session.frame(refCam, { train: true });
      if (f === stableFrom) {
        session.arms.plain.startFlicker();
        session.arms.nrc.startFlicker();
      }
      if (f >= stableFrom) {
        session.arms.plain.stepFlicker();
        session.arms.nrc.stepFlicker();
      }
      if (checkpoints.includes(f)) await measure('static', f, f === 1 || f === checkpoints[checkpoints.length - 1]);
      else await device.queue.onSubmittedWorkDone(); // one frame in flight
    }
    if (frames > stableFrom) await stability('static');
    // The cache's light lands in the right plane (nrc-pipeline.md §2): diffuse x albedo + specular x specFactor = beauty, per pixel.
    for (const [name, k] of [['plain', session.plain], ['nrc', session.cache.render]] as const) {
      const fr = await k.readFrame();
      let worst = 0;
      let sum = 0;
      let beauty = 0;
      const n = fr.width * fr.height;
      for (let i = 0; i < n; i++)
        for (let c = 0; c < 3; c++) {
          const j = i * 4 + c;
          const recomposed = (fr.diffuse[j] as number) * (fr.albedo[j] as number) + (fr.specular[j] as number) * (fr.specularFactor[j] as number);
          const b = fr.radiance[j] as number;
          beauty += b;
          const e = Math.abs(recomposed - b) / (Math.abs(b) + 1e-3);
          worst = Math.max(worst, e);
          sum += e;
        }
      metrics[`plane_identity_${name}_max_rel`] = worst;
      metrics[`plane_identity_${name}_mean_rel`] = sum / (3 * n);
      // Non-zero, so an exact 0 above is not an empty read (sponza-white's albedo 0.5 makes divide-then-multiply bit-exact).
      metrics[`plane_identity_${name}_beauty_mean`] = beauty / (3 * n);
    }
  }

  // 4. Camera path: dolly + pan into the reference view, then hold.
  if (scenarios.includes('path')) {
    const startPos: [number, number, number] = [refCam.position[0] + 2.0, refCam.position[1] + 0.4, refCam.position[2] + 1.2];
    const startTarget: [number, number, number] = [refCam.target[0], refCam.target[1] - 0.4, refCam.target[2] + 2.0];
    const lerp = (a: readonly number[], b: readonly number[], s: number) => a.map((v, i) => v + (b[i]! - v) * s) as [number, number, number];
    const camAt = (s: number): PtLookAtCamera => {
      const k = s * s * (3 - 2 * s);
      return { ...refCam, position: lerp(startPos, refCam.position, k), target: lerp(startTarget, refCam.target, k) };
    };
    const hold = Math.max(...pathCheckpoints);
    session.cutNext();
    for (let f = 0; f < pathFrames; f++) {
      await session.frame(camAt(f / (pathFrames - 1)), { train: true });
      if (f === Math.floor(pathFrames / 2) && doUpload) {
        await device.queue.onSubmittedWorkDone();
        for (const arm of ['plain', 'nrc'] as const) for (const stage of ['raw', 'denoised', 'final'] as const) await uploadImage(`path-mid-${arm}-${stage}`, await readTextureRgba16f(device, session.arms[arm].stage(stage)));
      } else await device.queue.onSubmittedWorkDone();
      if (f === pathFrames - 1) {
        session.arms.plain.startFlicker();
        session.arms.nrc.startFlicker();
      }
    }
    // The last path frame is the reference pose: arrival frame 1 is the first frame AFTER the path (the hold).
    for (let h = 1; h <= hold; h++) {
      await session.frame(refCam, { train: true });
      session.arms.plain.stepFlicker();
      session.arms.nrc.stepFlicker();
      if (pathCheckpoints.includes(h)) await measure('path', h, h === pathCheckpoints[0]);
      else await device.queue.onSubmittedWorkDone();
    }
    await stability('path');
  }

  // 5. Cold start (N11.6): a fresh network trained W frames at the view, then a cut and single frames (the cache keeps training). Plain does not
  // depend on W; it is measured each time as the noise control. The global `warm` and the timing pass are irrelevant here (fresh network).
  if (scenarios.includes('coldstart')) {
    const warmups = (args.warmups ?? '0,10,30,100,300').split(',').map(Number);
    const coldFrames = (args.coldFrames ?? '1,4').split(',').map(Number);
    for (const W of warmups) {
      session.cache.resetNetwork();
      session.cutNext();
      for (let f = 0; f < W; f += 25) {
        session.trainOnly(refCam, Math.min(25, W - f));
        await device.queue.onSubmittedWorkDone();
      }
      session.cutNext();
      for (let f = 1; f <= Math.max(...coldFrames); f++) {
        await session.frame(refCam, { train: true });
        if (coldFrames.includes(f)) await measure(`cold-w${W}`, f, false);
        else await device.queue.onSubmittedWorkDone();
      }
    }
  }

  // Equal-time table (markdown) for the write-up.
  const fmt = (v: number | null, d = 3) => (v === null ? '—' : v < 0.01 && v !== 0 ? v.toExponential(2) : v.toFixed(d));
  const lines = ['| scenario | frame | stage | plain relMSE | NRC relMSE | ratio | plain FLIP | NRC FLIP | plain bias | NRC bias |', '|---|---|---|---|---|---|---|---|---|---|'];
  for (const r of rows.filter((x) => x.arm === 'plain')) {
    const n = rows.find((x) => x.arm === 'nrc' && x.scenario === r.scenario && x.frame === r.frame && x.stage === r.stage)!;
    lines.push(`| ${r.scenario} | ${r.frame} | ${r.stage} | ${fmt(r.relMse)} | ${fmt(n.relMse)} | ${(r.relMse / n.relMse).toFixed(2)}× | ${fmt(r.flip)} | ${fmt(n.flip)} | ${(100 * r.relBias).toFixed(1)}% | ${(100 * n.relBias).toFixed(1)}% |`);
  }
  notes.table = lines.join('\n');
  const flick = ['| scenario | stage | plain flicker | NRC flicker | ratio |', '|---|---|---|---|---|'];
  for (const sc of ['static', 'path'])
    for (const st of ['raw', 'denoised', 'final']) {
      const a = metrics[`${sc}_plain_${st}_flicker`];
      const b = metrics[`${sc}_nrc_${st}_flicker`];
      if (a !== undefined && b !== undefined) flick.push(`| ${sc} | ${st} | ${fmt(a, 4)} | ${fmt(b, 4)} | ${(a / b).toFixed(2)}× |`);
    }
  notes.stabilityTable = flick.join('\n');
  console.log(notes.table);
  console.log(notes.stabilityTable);

  reportResult(
    { scene: sponzaWhite.name, renderer: 'nrc-pipeline-study', config: { denoiser: notes.denoiser, width, height, displayWidth, displayHeight, warm, frames, checkpoints, stableFrom, pathFrames, pathCheckpoints, steps, plainSpp: session.plainSpp, infer: session.inferDescription, reference: refSource }, valid: deviceErrors.length === 0, metrics, notes },
    { adapter: adapterSummary, deviceErrors },
  );
} catch (error) {
  reportError(error, { adapter: adapterSummary, deviceErrors });
}
