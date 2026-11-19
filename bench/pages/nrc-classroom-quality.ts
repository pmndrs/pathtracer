import type { PtLookAtCamera } from '@pmndrs/pathtracer';
import { loadInterior } from '../scenes/interiors.ts';
import { phase10Settings } from '../nrc/phase10.ts';
import { NrcPipelineSession } from '../nrc/pipeline-session.ts';
import { readTextureRgba16f, type PipelineStage } from '../realtime/chain.ts';

const width = 640, height = 360, exposure = 2 ** 2.5;
const stages: PipelineStage[] = ['raw', 'denoised', 'final'];
const run = document.querySelector<HTMLButtonElement>('#run')!;
const progress = document.querySelector('#progress')!;
const rows = document.querySelector('tbody')!;
const captures = document.querySelector('#captures')!;
type Image = Awaited<ReturnType<typeof readTextureRgba16f>>;

function tone(linear: number): number {
  const x = Math.max(0, linear * exposure);
  if (!Number.isFinite(x)) throw new Error('Nonfinite radiance in GPU readback.');
  return Math.min(1, x * (2.51 * x + 0.03) / (x * (2.43 * x + 0.59) + 0.14)) ** (1 / 2.2);
}

function luminance(image: Image): Float32Array {
  const result = new Float32Array(width * height);
  const scale = image.width / width;
  for (let y = 0; y < image.height; y++) for (let x = 0; x < image.width; x++) {
    const i = (y * image.width + x) * 4;
    const j = Math.floor(y / scale) * width + Math.floor(x / scale);
    result[j]! += (0.2126 * tone(image.data[i]!) + 0.7152 * tone(image.data[i + 1]!) + 0.0722 * tone(image.data[i + 2]!)) / (scale * scale);
  }
  return result;
}

function patchChanges(a: Float32Array, b: Float32Array, changes: number[]): void {
  for (let y = 0; y < height; y += 8) for (let x = 0; x < width; x += 8) {
    let delta = 0;
    for (let dy = 0; dy < 8; dy++) for (let dx = 0; dx < 8; dx++) {
      const i = (y + dy) * width + x + dx;
      delta += a[i]! - b[i]!;
    }
    changes.push(Math.abs(delta) / 64);
  }
}

function capture(image: Image, caption: string): void {
  const canvas = document.createElement('canvas');
  canvas.width = image.width;
  canvas.height = image.height;
  const ctx = canvas.getContext('2d')!;
  const pixels = ctx.createImageData(canvas.width, canvas.height);
  for (let i = 0; i < pixels.data.length; i += 4) {
    for (let c = 0; c < 3; c++) pixels.data[i + c] = Math.round(255 * tone(image.data[i + c]!));
    pixels.data[i + 3] = 255;
  }
  ctx.putImageData(pixels, 0, 0);
  const figure = document.createElement('figure');
  const label = document.createElement('figcaption');
  label.textContent = caption;
  figure.append(label, canvas);
  captures.prepend(figure);
}

async function compare(): Promise<void> {
  const scene = await loadInterior('hero-classroom');
  const camera = scene.camera as PtLookAtCamera;
  let reference: Float32Array | undefined;
  const cases = [
    ['tile', false, 'svgf'], ['pixel', false, 'svgf'], ['pixel', true, 'svgf'],
    ['tile', true, 'svgf'], ['tile', true, 'oidn'],
  ] as const;
  for (const [layout, accumulate, denoiser] of cases) {
    const name = (layout === 'tile' ? 'Tiles' : 'Pixels') + (accumulate ? ' · still' : ' · fresh') + ' · ' + denoiser.toUpperCase();
    progress.textContent = name + ' · loading scene';
    const s = await NrcPipelineSession.create({ scene, width, height, displayWidth: width * 2, displayHeight: height * 2,
      cache: { ...phase10Settings('mlp'), trainSelect: layout }, denoise: denoiser, oidn: { hold: false }, upscaler: { rcasDenoise: true } });
    let gpuError = '';
    s.device.addEventListener('uncapturederror', event => { gpuError = event.error.message; });
    try {
      if (!reference) {
        progress.textContent = 'Rendering the 256-spp plain reference…';
        s.accumulateStill = true;
        s.plainSpp = 16;
        s.plain.setSeed(907);
        for (let i = 0; i < 16; i++) {
          await s.frame(camera, { arms: ['plain'], train: false, denoise: false });
          await s.device.queue.onSubmittedWorkDone();
        }
        const ref = await readTextureRgba16f(s.device, s.arms.plain.stage('raw'));
        reference = luminance(ref);
        capture(ref, 'Plain path tracing · 256 spp · render resolution');
      }
      s.cutNext();
      s.accumulateStill = accumulate;
      s.setGate(8, 16);
      const previous = new Map<PipelineStage, Float32Array>();
      const changes = { raw: [] as number[], denoised: [] as number[], final: [] as number[] };
      const times: number[] = [];
      for (let frame = 0; frame < 49; frame++) {
        const start = performance.now();
        await s.frame(camera, { arms: ['nrc'], train: true });
        await s.device.queue.onSubmittedWorkDone();
        if (gpuError) throw new Error(gpuError);
        if (frame >= 8) times.push(performance.now() - start);
        progress.textContent = `${name} · frame ${frame + 1}/49`;
        if (frame < 40) continue;
        for (const stage of stages) {
          const image = await readTextureRgba16f(s.device, s.arms.nrc.stage(stage));
          const current = luminance(image);
          const prior = previous.get(stage);
          if (prior) patchChanges(current, prior, changes[stage]);
          previous.set(stage, current);
          if (frame === 48) {
            const flicker = changes[stage].sort((a, b) => a - b);
            const error = current.reduce((sum, v, i) => sum + Math.abs(v - reference![i]!), 0) / current.length;
            const median = times.sort((a, b) => a - b)[Math.floor(times.length / 2)]!;
            const row = rows.insertRow();
            for (const value of [name + ' / ' + stage, String(s.samples.nrc), flicker[Math.floor(flicker.length * 0.99)]!.toFixed(4), error.toFixed(4), median.toFixed(1)]) row.insertCell().textContent = value;
            if (stage === 'final') capture(image, name + ' · temporal 2× · ' + s.samples.nrc + ' spp');
          }
        }
      }
    } finally { s.destroy(); }
  }
}

// A stationary spatial metric catches fixed grain even when adjacent frames agree.
function surfaceGrain({ width: w, height: h, data }: Image): number {
  function displayY(x: number, y: number): number {
    const i = 4 * (y * w + x);
    return tone(0.2126 * data[i]! + 0.7152 * data[i + 1]! + 0.0722 * data[i + 2]!);
  }
  let sum = 0, count = 0;
  for (const [left, right, top, bottom] of [[0.39, 0.49, 0.06, 0.105], [0.48, 0.52, 0.22, 0.27], [0.60, 0.67, 0.19, 0.22]] as const) {
    for (let y = Math.ceil(top * h); y < bottom * h; y++) for (let x = Math.ceil(left * w); x < right * w; x++) {
      sum += Math.abs(displayY(x, y) - 0.25 * (displayY(x - 1, y) + displayY(x + 1, y) + displayY(x, y - 1) + displayY(x, y + 1)));
      count++;
    }
  }
  return sum / count;
}

async function testSettling(): Promise<void> {
  const section = document.querySelector<HTMLElement>('#settling')!;
  const tbody = section.querySelector('tbody')!;
  section.hidden = false;
  tbody.replaceChildren();
  const scene = await loadInterior('hero-classroom');
  const checkpoints = new Set([8, 32, 64, 128, 256, 512, 1024]);
  for (const limit of [0, 16]) {
    const label = limit ? limit + ' spp' : 'Unlimited';
    const session = await NrcPipelineSession.create({ scene, width: 480, height: 270, displayWidth: 960, displayHeight: 540,
      cache: phase10Settings('mlp'), upscaler: { rcasDenoise: true } });
    try {
      session.setGate(8, 16);
      session.accumulateStill = true;
      session.svgfHistoryLimit = limit;
      for (let frame = 1; frame <= 1024; frame++) {
        await session.frame(scene.camera as PtLookAtCamera);
        await session.device.queue.onSubmittedWorkDone();
        if (frame % 8 === 0) progress.textContent = `${label} · frame ${frame}/1024`;
        if (!checkpoints.has(frame)) continue;
        for (const arm of ['plain', 'nrc'] as const) {
          const values = [label, String(frame), arm];
          for (const stage of stages) {
            const image = await readTextureRgba16f(session.device, session.arms[arm].stage(stage));
            values.push(surfaceGrain(image).toFixed(5));
          }
          const row = tbody.insertRow();
          for (const v of values) row.insertCell().textContent = v;
        }
      }
    } finally { session.destroy(); }
  }
}

const hold = document.querySelector<HTMLButtonElement>('#hold')!;
for (const [button, test] of [[run, compare], [hold, testSettling]] as const) {
  button.disabled = false;
  button.addEventListener('click', async () => {
    run.disabled = hold.disabled = true;
    progress.textContent = 'Loading Classroom…';
    if (button === run) { rows.replaceChildren(); captures.replaceChildren(); }
    try { await test(); progress.textContent = 'Comparison complete.'; }
    catch (e) { progress.textContent = 'Failed: ' + (e instanceof Error ? e.message : String(e)); }
    finally { run.disabled = hold.disabled = false; }
  });
}
