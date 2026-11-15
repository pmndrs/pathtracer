/**
 * Research track N live demo: the MLP learns a procedural HDR image while you watch; switch backends (portable
 * f32/f16, subgroup matrices when Chrome exposes them, the emulated sgm kernel) and measure them.
 *
 * Open by hand (not a runner page): `pnpm --filter @pathtracer/bench mlp:live`, in Chrome with
 * `--enable-unsafe-webgpu` for the subgroup-matrix backend. URL args: ?size=256&seed=1&offscreen=0
 *
 * offscreen=1 renders into a texture instead of the canvas and exposes `window.__mlpLive.snapshot()` (pauses, reads the
 * rendered image back once): for headless checks where canvas presentation is unavailable (GPU-less containers).
 *
 * The frame loop never reads back: Benchmark pauses it, runs a timestamped burst, reads once, resumes.
 */

import { DEFAULT_SHAPE, DEFAULT_TRAIN, MlpEngine, initWeights, weightCount, type Precision } from '@pathtracer/mlp';
import { hasSubgroupMatrix, inferArmOptions, readBuffer, requestMlpDevice, storageBuffer } from './mlp-common.ts';
import { IMAGE_HOOKS_INFER, IMAGE_HOOKS_TRAIN, imageBindGroups, targetImage } from './mlp-image.ts';

const RENDER_WGSL = /* wgsl */ `
struct View { size: u32, mode: u32, exposure: f32, _pad: f32 }
@group(0) @binding(0) var<storage, read> tgt: array<f32>;
@group(0) @binding(1) var<storage, read> fit: array<f32>;
@group(0) @binding(2) var<uniform> view: View;

@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

fn rgb(buf: u32, i: u32) -> vec3f {
  if (buf == 0u) { return vec3f(tgt[i], tgt[i + 1u], tgt[i + 2u]); }
  return vec3f(fit[i], fit[i + 1u], fit[i + 2u]);
}
// Narkowicz ACES fit, then sRGB encode.
fn display(c: vec3f) -> vec3f {
  let x = max(c, vec3f(0.0));
  let t = clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3f(0.0), vec3f(1.0));
  return select(1.055 * pow(t, vec3f(1.0 / 2.4)) - 0.055, 12.92 * t, t <= vec3f(0.0031308));
}
fn heat(e: f32) -> vec3f {
  let t = clamp(e, 0.0, 1.0);
  return clamp(vec3f(1.5 * t, 1.5 - abs(3.0 * t - 1.5), 1.5 * (1.0 - t)) * 0.9, vec3f(0.0), vec3f(1.0));
}

@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let x = u32(pos.x);
  let y = u32(pos.y);
  let side = x / view.size;
  let i = (y * view.size + x % view.size) * 3u;
  if (side == 1u && view.mode == 1u) {
    let l = vec3f(0.299, 0.587, 0.114);
    let t = dot(rgb(0u, i), l);
    let e = abs(dot(rgb(1u, i), l) - t) / (abs(t) + 0.01);
    return vec4f(heat(e), 1.0);
  }
  return vec4f(display(rgb(side, i) * view.exposure), 1.0);
}
`;

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const statsEl = $<HTMLDivElement>('stats');
const errorEl = $<HTMLDivElement>('error');

function showError(e: unknown): void {
  errorEl.textContent = e instanceof Error ? (e.stack ?? e.message) : String(e);
}

try {
  const params = new URLSearchParams(location.search);
  const size = Number(params.get('size') ?? 256);
  const seed = Number(params.get('seed') ?? 1);
  const { adapter, device } = await requestMlpDevice();
  device.addEventListener('uncapturederror', (e) => showError((e as GPUUncapturedErrorEvent).error.message));

  // Backends this device can run.
  const backendSelect = $<HTMLSelectElement>('backend');
  const backends: Array<[string, string]> = [['portable:f32', 'portable f32']];
  if (device.features.has('shader-f16')) backends.push(['portable:f16', 'portable f16']);
  if (hasSubgroupMatrix(device)) {
    if (device.features.has('shader-f16')) backends.push(['sgm:f16', 'subgroup matrix f16']);
    backends.push(['sgm:f32', 'subgroup matrix f32']);
  }
  backends.push(['sgm-emu:f32', 'sgm emulated (slow, correctness)']);
  for (const [value, label] of backends) backendSelect.add(new Option(label, value));
  backendSelect.value = backends.find(([v]) => v === 'sgm:f16')?.[0] ?? backends.find(([v]) => v === 'portable:f16')?.[0] ?? 'portable:f32';

  const offscreen = params.get('offscreen') === '1';
  const canvas = $<HTMLCanvasElement>('canvas');
  canvas.width = size * 2;
  canvas.height = size;
  const format: GPUTextureFormat = offscreen ? 'rgba8unorm' : navigator.gpu.getPreferredCanvasFormat();
  const context = offscreen ? null : (canvas.getContext('webgpu') as GPUCanvasContext);
  context?.configure({ device, format, alphaMode: 'opaque' });
  const offscreenTexture = offscreen
    ? device.createTexture({ size: [size * 2, size], format, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC })
    : null;
  const renderTarget = (): GPUTextureView => (offscreenTexture ?? (context as GPUCanvasContext).getCurrentTexture()).createView();

  const target = storageBuffer(device, targetImage(size), 'target');
  const out = storageBuffer(device, size * size * 12, 'out');
  const dims = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(dims, 0, new Uint32Array([size, size, seed, 0]));
  const viewUniform = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const renderModule = device.createShaderModule({ code: RENDER_WGSL });
  const renderPipeline = await device.createRenderPipelineAsync({
    layout: 'auto',
    vertex: { module: renderModule, entryPoint: 'vs' },
    fragment: { module: renderModule, entryPoint: 'fs', targets: [{ format }] },
  });
  const renderGroup = device.createBindGroup({
    layout: renderPipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: target } }, { binding: 1, resource: { buffer: out } }, { binding: 2, resource: { buffer: viewUniform } }],
  });

  let mlp: MlpEngine | null = null;
  let groups: ReturnType<typeof imageBindGroups> | null = null;
  let counts: GPUBuffer | null = null;
  let steps = 0;
  let description = '';
  let paused = false;
  let lastBench = '';
  let frames = 0;
  let fpsTime = performance.now();
  let fps = 0;
  // Frames in flight: rAF does not wait for the GPU (headless, or a slow backend), so cap the queue at 2 frames.
  let inFlight = 0;

  async function build(): Promise<void> {
    paused = true;
    await device.queue.onSubmittedWorkDone();
    mlp?.destroy();
    const [arm, precision] = backendSelect.value.split(':') as [string, Precision];
    const batch = Number($<HTMLSelectElement>('batch').value);
    const options = inferArmOptions(arm, adapter, device, DEFAULT_SHAPE, precision, { records: 64, subgroups: 1, tile: 8 });
    description = options.description;
    mlp = await MlpEngine.create(device, {
      precision: options.precision,
      capacity: batch,
      inferCapacity: size * size,
      useEma: $<HTMLInputElement>('ema').checked,
      train: { ...DEFAULT_TRAIN, learningRate: Number($<HTMLInputElement>('lr').value) },
      trainHooks: IMAGE_HOOKS_TRAIN,
      inferHooks: IMAGE_HOOKS_INFER,
      ...(options.sgm ? { sgm: options.sgm } : {}),
    });
    mlp.uploadWeights(initWeights(DEFAULT_SHAPE, seed));
    counts?.destroy();
    counts = storageBuffer(device, new Uint32Array([batch, size * size, 0, 0]), 'counts');
    groups = imageBindGroups(device, mlp.trainLayout, mlp.inferLayout, target, dims, out);
    steps = 0;
    lastBench = '';
    paused = false;
  }

  async function benchmark(): Promise<void> {
    if (!mlp || !groups || !counts || !device.features.has('timestamp-query')) {
      lastBench = 'benchmark needs timestamp-query';
      return;
    }
    paused = true;
    await device.queue.onSubmittedWorkDone();
    const trainN = 16;
    const inferN = 8;
    const querySet = device.createQuerySet({ type: 'timestamp', count: 2 * (trainN + inferN) });
    for (let i = 0; i < trainN + inferN; i++) {
      const encoder = device.createCommandEncoder();
      const writes = { querySet, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 };
      if (i < trainN) mlp.encodeTrainStep(encoder, counts, 0, groups.train, writes);
      else mlp.encodeInfer(encoder, counts, 1, groups.infer, writes);
      device.queue.submit([encoder.finish()]);
    }
    steps += trainN;
    const resolve = device.createBuffer({ size: querySet.count * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const encoder = device.createCommandEncoder();
    encoder.resolveQuerySet(querySet, 0, querySet.count, resolve, 0);
    device.queue.submit([encoder.finish()]);
    const t = new BigUint64Array(await readBuffer(device, resolve));
    const span = (i: number) => Number((t[2 * i + 1] as bigint) - (t[2 * i] as bigint)) / 1e6;
    const med = (v: number[]) => [...v].sort((a, b) => a - b)[Math.floor(v.length / 2)] as number;
    const train = med(Array.from({ length: trainN - 2 }, (_, i) => span(i + 2)));
    const infer = med(Array.from({ length: inferN - 2 }, (_, i) => span(trainN + i + 2)));
    const records = size * size;
    const tflops = (2 * weightCount(DEFAULT_SHAPE) * records) / (infer * 1e-3) / 1e12;
    lastBench = `benchmark (${description}): train step ${train.toFixed(3)} ms · inference ${infer.toFixed(3)} ms for ${records} records = ${((infer * 1e6) / records).toFixed(2)} ms/Mquery, ${tflops.toFixed(2)} TFLOP/s`;
    querySet.destroy();
    resolve.destroy();
    paused = false;
  }

  function frame(): void {
    requestAnimationFrame(frame);
    if (paused || !mlp || !groups || !counts || inFlight >= 2) return;
    const k = Number($<HTMLSelectElement>('steps').value);
    const view = new ArrayBuffer(16);
    new Uint32Array(view, 0, 2).set([size, Number($<HTMLSelectElement>('view').value)]);
    new Float32Array(view, 8, 1)[0] = 2 ** Number($<HTMLInputElement>('exposure').value);
    device.queue.writeBuffer(viewUniform, 0, view);
    const encoder = device.createCommandEncoder();
    for (let i = 0; i < k; i++) mlp.encodeTrainStep(encoder, counts, 0, groups.train);
    mlp.encodeInfer(encoder, counts, 1, groups.infer);
    const pass = encoder.beginRenderPass({ colorAttachments: [{ view: renderTarget(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }] });
    pass.setPipeline(renderPipeline);
    pass.setBindGroup(0, renderGroup);
    pass.draw(3);
    pass.end();
    device.queue.submit([encoder.finish()]);
    inFlight++;
    void device.queue.onSubmittedWorkDone().then(() => inFlight--);
    steps += k;
    frames++;
    const now = performance.now();
    if (now - fpsTime > 500) {
      fps = (frames * 1000) / (now - fpsTime);
      frames = 0;
      fpsTime = now;
    }
    statsEl.textContent = `${description} · ${adapter.info.vendor} ${adapter.info.architecture}\nsteps ${steps} · ${fps.toFixed(0)} fps (frame rate includes vsync; use Benchmark for GPU time)\n${lastBench}`;
  }

  // Debug hook for headless checks (offscreen=1 only): pause, wait for the GPU, read the rendered image once.
  if (offscreenTexture) {
    (window as unknown as Record<string, unknown>).__mlpLive = {
      steps: () => steps,
      stats: () => statsEl.textContent,
      snapshot: async (): Promise<number[]> => {
        paused = true;
        await device.queue.onSubmittedWorkDone();
        const bytesPerRow = Math.ceil((size * 2 * 4) / 256) * 256;
        const buffer = device.createBuffer({ size: bytesPerRow * size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const encoder = device.createCommandEncoder();
        encoder.copyTextureToBuffer({ texture: offscreenTexture }, { buffer, bytesPerRow }, [size * 2, size]);
        device.queue.submit([encoder.finish()]);
        await buffer.mapAsync(GPUMapMode.READ);
        const raw = new Uint8Array(buffer.getMappedRange());
        const out: number[] = [];
        for (let y = 0; y < size; y++) for (let x = 0; x < size * 2; x++) for (let c = 0; c < 3; c++) out.push(raw[y * bytesPerRow + x * 4 + c] as number);
        buffer.unmap();
        buffer.destroy();
        paused = false;
        return out;
      },
    };
  }

  for (const id of ['backend', 'batch', 'ema', 'lr']) $(id).addEventListener('change', () => void build().catch(showError));
  $('reset').addEventListener('click', () => void build().catch(showError));
  $('bench').addEventListener('click', () => void benchmark().catch(showError));
  await build();
  requestAnimationFrame(frame);
} catch (e) {
  showError(e);
}
