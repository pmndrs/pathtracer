import type { PtLookAtCamera } from '@pmndrs/pathtracer';
import { loadHeroScene } from '../hero/showcase.ts';
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
  const scene = await loadHeroScene('hero-classroom');
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

run.addEventListener('click', async () => {
  run.disabled = true;
  progress.textContent = 'Loading Classroom…';
  rows.replaceChildren();
  captures.replaceChildren();
  try { await compare(); progress.textContent = 'Comparison complete.'; }
  catch (e) { progress.textContent = 'Failed: ' + (e instanceof Error ? e.message : String(e)); }
  finally { run.disabled = false; }
});

run.disabled = false;
