/**
 * ReSTIR DI in real time (experimental). The point: at the same ONE sample per pixel per frame, ReSTIR DI gives a far less noisy single
 * frame than plain next-event estimation (NEE), and both converge to the same image: it is unbiased, it only removes noise.
 *
 * Two real-time megakernels render the same 64-lamp scene with the same camera every frame: plain NEE (`integrator: { restir: null }`)
 * picks one light per pixel, so with many small lights most pixels pick one that contributes little; ReSTIR DI (`integrator.restir`)
 * draws several candidates per pixel, keeps one in proportion to its contribution, then merges the kept samples of the previous frame
 * (temporal) and of neighbouring pixels (spatial). The split view shows NEE on the left and ReSTIR on the right, one frame each.
 *
 * While the camera is still the page averages the NEE frames into a reference (NEE is unbiased, so its mean converges to the true
 * image) and, about once a second, reads back the current frame of each method and prints its error against that reference (luminance
 * relMSE). "Average while still" averages both halves too: they converge to the same picture. Biased ReSTIR is cheaper and less noisy
 * still, but darker where neighbours disagree: the averaged halves then differ.
 */
import { PtMegakernel, ptSceneOptions, type PtRestirOptions } from '@pmndrs/pathtracer';
import { buildGpuScene } from '@pmndrs/raytracer';
import { defineExample } from '../../shared/example.ts';
import { FrameView } from '../../shared/frame-view.ts';
import { LAMPS_CAMERA, lampsScene } from '../../shared/lamps.ts';
import { OrbitCamera } from '../../shared/orbit.ts';

type ViewMode = 'split' | 'nee' | 'restir';

/** Frames the NEE reference needs before the error numbers mean anything (its own noise is about 1/frames of a single frame's). */
const REFERENCE_MIN = 64;
/** The reference stops growing here (f32 running sums; the error is already dominated by the single frames). */
const REFERENCE_MAX = 4096;

const LABEL_CSS = `
.rs-label { position: fixed; top: 60px; z-index: 50; padding: 4px 10px; border-radius: 8px; background: rgba(14, 14, 12, 0.72);
  border: 1px solid var(--border); font: 12px/1.45 var(--mono); color: var(--text); pointer-events: none; white-space: nowrap; }
.rs-label[hidden] { display: none; }
.rs-label.left { right: calc(50% + 12px); }
.rs-label.right { left: calc(50% + 12px); }
.rs-label.full { left: 12px; }
`;

/** Rec. 709 luminance relMSE of `x` against `ref` (both rgba per pixel): mean of (x - r)^2 / (r^2 + 0.01). */
function relMse(x: Float32Array, ref: Float32Array): number {
  let sum = 0, n = 0;
  for (let i = 0; i < ref.length; i += 4) {
    const a = 0.2126 * x[i]! + 0.7152 * x[i + 1]! + 0.0722 * x[i + 2]!;
    const r = 0.2126 * ref[i]! + 0.7152 * ref[i + 1]! + 0.0722 * ref[i + 2]!;
    if (!Number.isFinite(a) || !Number.isFinite(r)) continue;
    sum += ((a - r) * (a - r)) / (r * r + 0.01);
    n++;
  }
  return n ? sum / n : 0;
}

defineExample(async (ctx) => {
  const { device, caps, pane, stats, canvas } = ctx;
  const search = new URLSearchParams(location.search);
  const params = {
    view: (search.get('view') ?? 'split') as ViewMode,
    bias: (search.get('bias') ?? 'unbiased') as 'unbiased' | 'biased',
    candidates: 8,
    temporal: true,
    spatial: 2,
    accumulate: search.get('accumulate') === '1',
    spin: search.get('spin') === '1',
    maxDepth: 1,
    exposure: 1,
  };
  stats.set('image', 'path traced, 1 sample per pixel per frame · no denoiser');
  ctx.setRenderScale(0.5);
  const orbit = new OrbitCamera(canvas, LAMPS_CAMERA);
  // One view per method (what is drawn: the single frame, or its average while still) and the NEE reference (summed, never drawn).
  const neeView = new FrameView(device, ctx.format);
  const restirView = new FrameView(device, ctx.format);
  const reference = new FrameView(device, ctx.format);
  for (const v of [neeView, restirView]) v.accumulate = params.accumulate;
  const scene = await buildGpuScene(device, lampsScene(), { label: 'lamps' });
  stats.set('lights', '64 lamps, sun, sky');

  const style = document.createElement('style');
  style.textContent = LABEL_CSS;
  document.head.append(style);
  const label = (text: string) => {
    const el = document.createElement('div');
    el.className = 'rs-label';
    el.textContent = text;
    document.body.append(el);
    return el;
  };
  const neeLabel = label('Plain NEE · 1 light per pixel');
  const restirLabel = label('ReSTIR DI · same 1 sample');
  const placeLabels = () => {
    neeLabel.hidden = params.view === 'restir';
    restirLabel.hidden = params.view === 'nee';
    neeLabel.className = `rs-label ${params.view === 'split' ? 'left' : 'full'}`;
    restirLabel.className = `rs-label ${params.view === 'split' ? 'right' : 'full'}`;
  };
  placeLabels();

  interface Arm { pt: PtMegakernel; size: [number, number] }
  let nee: Arm | null = null;
  let restir: Arm | null = null;
  let frames = 0;
  const limits = () => ({ maxDepth: params.maxDepth, rrStartDepth: 0, textureConeScale: 1 });
  const restartAll = () => {
    neeView.restart();
    restirView.restart();
    reference.restart();
    resetReadings();
  };

  const build = async (restirOptions: PtRestirOptions | null): Promise<Arm> => {
    const pt = await PtMegakernel.create(device, caps, ptSceneOptions(scene, { mode: 'realtime', integrator: { restir: restirOptions } }));
    pt.setScene(scene.trace, scene.buffers);
    pt.setLimits(limits());
    return { pt, size: [0, 0] };
  };
  // Only the ReSTIR kernel depends on the ReSTIR controls: the NEE kernel, and with it the reference, survive a change of them.
  const rebuildRestir = async () => {
    ctx.progress('Compiling…');
    try {
      const next = await build({ bias: params.bias, candidates: params.candidates, temporal: params.temporal, spatial: params.spatial });
      stats.set('kernels', `NEE ${((nee?.pt.wgslBytes ?? 0) / 1024).toFixed(1)} KB · ReSTIR ${(next.pt.wgslBytes / 1024).toFixed(1)} KB`);
      const old = restir;
      restir = next;
      resetReadings();
      ctx.progress(null);
      await device.queue.onSubmittedWorkDone();
      old?.pt.destroy();
    } catch (error) {
      ctx.error('This combination does not compile', error);
    }
  };
  orbit.onChange = restartAll;

  const r = pane.addFolder({ title: 'ReSTIR DI (experimental)' });
  r.addBinding(params, 'view', { options: { 'Split: NEE | ReSTIR': 'split', 'Plain NEE only': 'nee', 'ReSTIR only': 'restir' } }).on('change', placeLabels);
  r.addBinding(params, 'bias', { options: { unbiased: 'unbiased', biased: 'biased' } }).on('change', () => void rebuildRestir());
  r.addBinding(params, 'candidates', { min: 1, max: 32, step: 1 }).on('change', (e) => e.last && void rebuildRestir());
  r.addBinding(params, 'temporal', { label: 'temporal reuse' }).on('change', () => void rebuildRestir());
  r.addBinding(params, 'spatial', { label: 'spatial neighbours', min: 0, max: 8, step: 1 }).on('change', (e) => e.last && void rebuildRestir());
  const f = pane.addFolder({ title: 'Frame' });
  f.addBinding(params, 'accumulate', { label: 'average while still' }).on('change', () => {
    for (const v of [neeView, restirView]) {
      v.accumulate = params.accumulate;
      v.restart();
    }
  });
  f.addBinding(params, 'spin', { label: 'orbit by itself' });
  f.addBinding(params, 'maxDepth', { label: 'max bounces', min: 1, max: 4, step: 1 }).on('change', () => {
    for (const arm of [nee, restir]) arm?.pt.setLimits(limits());
    restartAll();
  });
  f.addBinding(params, 'exposure', { min: 0.1, max: 8 }).on('change', () => {
    neeView.exposure = restirView.exposure = params.exposure;
  });

  // Error per frame. A separate readback about once a second: its copies are submitted between two frames (the frame loop records and
  // submits synchronously), so they see one frame of each kernel and the reference after that same frame. Not a timing.
  // One frame's error jumps from reading to reading (ReSTIR's more than NEE's: its reuse varies), so the numbers shown are the mean of
  // the readings since the camera stopped or the ReSTIR settings changed; `last` is the latest reading.
  let lastError: { nee: number; restir: number; readings: number; referenceFrames: number; last: { nee: number; restir: number } } | null = null;
  const readings = { nee: 0, restir: 0, n: 0, epoch: 0, frame: -1 };
  const resetReadings = () => {
    readings.nee = readings.restir = readings.n = 0;
    readings.frame = -1;
    readings.epoch++; // a reading in flight across the reset is dropped
    lastError = null;
  };
  const readBuffer = (buffer: GPUBuffer, bytes: number): Promise<Float32Array> => {
    const staging = device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder({ label: 'restir error readback' });
    encoder.copyBufferToBuffer(buffer, 0, staging, 0, bytes);
    device.queue.submit([encoder.finish()]);
    return staging.mapAsync(GPUMapMode.READ).then(() => {
      const data = new Float32Array(staging.getMappedRange().slice(0));
      staging.destroy();
      return data;
    });
  };
  const measure = async () => {
    if (!nee || !restir || reference.count < REFERENCE_MIN) {
      stats.set('error per frame', 'hold the camera still to measure');
      stats.set('reference', `${reference.count} of ${REFERENCE_MIN} NEE frames`);
      return;
    }
    const [w, h] = nee.size;
    if (w !== restir.size[0] || h !== restir.size[1] || w === 0) return;
    const referenceFrames = reference.count, epoch = readings.epoch, frame = frames;
    if (frame === readings.frame) return; // no new frame since the last reading (a slow GPU)
    // All three copies are submitted before the first await: the same frame for each.
    const pending = [reference.read(w, h), readBuffer(nee.pt.frameBuffers.accum, w * h * 16), readBuffer(restir.pt.frameBuffers.accum, w * h * 16)];
    const [ref, a, b] = await Promise.all(pending);
    if (epoch !== readings.epoch) return; // the camera moved (or the ReSTIR kernel changed) while reading
    const last = { nee: relMse(a!, ref!), restir: relMse(b!, ref!) };
    readings.nee += last.nee;
    readings.restir += last.restir;
    readings.n++;
    readings.frame = frame;
    lastError = { nee: readings.nee / readings.n, restir: readings.restir / readings.n, readings: readings.n, referenceFrames, last };
    const ratio = lastError.nee / Math.max(lastError.restir, 1e-12);
    stats.set('error per frame', `NEE ${lastError.nee.toFixed(2)} · ReSTIR ${lastError.restir.toFixed(2)} (${ratio.toFixed(1)}× lower, ${readings.n} readings)`);
    stats.set('reference', `mean of ${referenceFrames} NEE frames${referenceFrames >= REFERENCE_MAX ? ' (full)' : ''}`);
  };
  const measureLoop = () => {
    measure().catch((error) => console.warn('restir error readback', error)).finally(() => setTimeout(measureLoop, 1000));
  };

  (globalThis as { __realtime?: unknown }).__realtime = {
    get frames() { return frames; },
    /** The image as drawn (split: NEE columns left of the divider, ReSTIR right of it), linear. */
    read: async () => {
      const [w, h] = restir?.size ?? [0, 0];
      const [a, b] = await Promise.all([neeView.read(w, h), restirView.read(w, h)]);
      if (params.view !== 'split') return params.view === 'nee' ? a : b;
      const out = new Float32Array(b);
      for (let y = 0; y < h; y++) out.set(a.subarray(y * w * 4, (y * w + (w >> 1)) * 4), y * w * 4);
      return out;
    },
    size: () => restir?.size ?? [0, 0],
    /** The last error numbers (luminance relMSE of one frame against the NEE reference), or null before the reference has 64 frames. */
    get error() { return lastError; },
  };

  ctx.progress('Compiling…');
  try {
    nee = await build(null);
  } catch (error) {
    ctx.error('The NEE kernel does not compile', error);
    return;
  }
  await rebuildRestir();
  measureLoop();
  ctx.loop((encoder, dt) => {
    if (!nee || !restir) return false;
    if (params.spin) {
      orbit.yaw += Math.min(dt, 50) * 0.0003;
      restartAll();
    }
    const [w, h] = ctx.filmSize();
    if (params.accumulate && restirView.count >= 2048 && neeView.count >= 2048 && w === restir.size[0] && h === restir.size[1]) return false;
    const camera = orbit.camera;
    for (const arm of [nee, restir]) {
      if (w !== arm.size[0] || h !== arm.size[1]) {
        arm.pt.setFilm(w, h);
        arm.pt.setCamera(camera);
        arm.size = [w, h];
      } else arm.pt.nextFrame(camera);
      arm.pt.encode(encoder, 1);
    }
    const target = ctx.target();
    const half = w >> 1;
    if (params.view === 'split') {
      neeView.encode(encoder, nee.pt.frameBuffers, target, { columns: [0, half] });
      restirView.encode(encoder, restir.pt.frameBuffers, target, { columns: [half, w], load: true, divider: true });
    } else {
      // The hidden method still renders: the reference and the error numbers need both.
      neeView.encode(encoder, nee.pt.frameBuffers, params.view === 'nee' ? target : null);
      restirView.encode(encoder, restir.pt.frameBuffers, params.view === 'restir' ? target : null);
    }
    if (reference.count < REFERENCE_MAX) reference.encode(encoder, nee.pt.frameBuffers, null);
    frames++;
    stats.set('film', `${w} × ${h}${params.view === 'split' ? ' (two kernels)' : ''}`);
    stats.set('averaged', params.accumulate ? `${restirView.count} frames` : 'off (single frames)');
    return true;
  });
});
