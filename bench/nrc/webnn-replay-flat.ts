/**
 * The one-dispatch WebNN frame, written flat (pmndrs/pathtracer WebNN sync study, bench/results/NRC-WEBNN-SYNC.md).
 *
 * This is the same work as `webnn-replay.ts` with `tensors=chunk fold=8 route=export sync=none`, without the switches and helper
 * layers of that file, so the hand-off sequence can be read and edited in one place: everything one frame does is in `frame()`.
 * A frame = encode the recorded queries on WebGPU, hand weights and queries to WebNN, ONE `dispatch`, hand the answers back, unpack.
 *
 * Run: `--page nrc-study --args webnnReplay=1 webnnFlat=1 ...` (the other args as in NRC-WEBNN-SYNC.md). `wait=1` adds a WebGPU queue
 * wait before `dispatch`, the one optional await. `check=0` skips the correctness check that runs before the timing.
 *
 * The file has three parts: setup (once), `frame()` (the thing being measured), and the measurement harness. The harness calls
 * `frame()` 4 times untimed to check the answers (with GPU read-backs after each, which are not part of a frame), then `repeats` + 2
 * times to time it, one call at a time.
 *
 * Fixed layout: the network is f16, 64 inputs, 4 outputs (RGB + one zero). An exportable tensor is capped at 8,192 rows on macOS, so a
 * frame of up to 32,768 queries crosses as [4096, 512] and the graph reshapes it to [32768, 64], and the result back to [4096, 32].
 */
import { encodingWgsl, forward, layerCount, layerIn, layerOffset, layerOut, nrcEncodingWgsl } from '@pathtracer/mlp';
import { halfToFloat } from '../pages/mlp-common.ts';
import { type NrcCache, nrcInputsWgsl } from './cache.ts';

// The interop API is experimental and has no DOM library declarations.
interface Tensor { destroy(): void }
interface Graph { destroy(): void }
interface Descriptor { dataType: 'float16'; shape: number[] }
interface Context {
  createExportableTensor(d: Descriptor, gpu: GPUDevice): Promise<Tensor>;
  exportToGPU(t: Tensor): Promise<GPUBuffer>;
  dispatch(graph: Graph, inputs: Record<string, Tensor>, outputs: Record<string, Tensor>): void;
  destroy(): void;
}
interface Builder {
  input(name: string, d: Descriptor): object;
  gemm(a: object, b: object, options: { bTranspose: boolean }): object;
  relu(a: object): object;
  slice(a: object, starts: number[], sizes: number[]): object;
  reshape(a: object, shape: number[]): object;
  build(outputs: Record<string, object>): Promise<Graph>;
}

const HEIGHT = 4096, FOLD = 8, ROWS = HEIGHT * FOLD; // One dispatch holds 32,768 queries.

export async function replayWebnnFlat(cache: NrcCache, args: Record<string, string | undefined>): Promise<Record<string, number>> {
  const mlp = cache.mlp;
  if (!mlp || cache.settings.network !== 'compact' || mlp.precision !== 'f16' || cache.settings.materialInputs !== 'evaluated' ||
    cache.settings.factorization !== 'off' || cache.settings.positionBands !== 4) throw new Error('WebNN replay requires the compact four-band evaluated fp16 baseline');
  const repeats = Number(args.repeats ?? 20);
  const waitBeforeDispatch = args.wait === '1';
  const ml = (navigator as unknown as { ml?: { createContext(options: { deviceType: string }): Promise<Context> } }).ml;
  if (!ml) throw new Error('navigator.ml unavailable');
  const gpu = cache.device;
  const records = cache.replayBuffers; // The renderer's recorded queries, its own WGSL inference pass and that pass's predictions.
  const { shape } = mlp;
  if (shape.inputs !== 64) throw new Error('The flat replay is written for 64 inputs');
  const layers = layerCount(shape);
  const weightHalves = mlp.weightCount + shape.width; // The last matrix gets a fourth, zero output row.
  const f16 = (dims: number[]): Descriptor => ({ dataType: 'float16', shape: dims });

  // ---- Setup (once): WebNN context, tensors, graph. ----
  const ctx = await ml.createContext({ deviceType: args.device ?? 'gpu' });
  const x = await ctx.createExportableTensor(f16([HEIGHT, FOLD * 64]), gpu);
  const y = await ctx.createExportableTensor(f16([HEIGHT, FOLD * 4]), gpu);
  const weights = await ctx.createExportableTensor(f16([weightHalves]), gpu);
  const builder = new (globalThis as unknown as { MLGraphBuilder: new (context: Context) => Builder }).MLGraphBuilder(ctx);
  const packet = builder.input('weights', f16([weightHalves]));
  let node = builder.reshape(builder.input('x', f16([HEIGHT, FOLD * 64])), [ROWS, 64]);
  for (let i = 0; i < layers; i++) {
    const dims = [i === layers - 1 ? 4 : layerOut(shape, i), layerIn(shape, i)];
    const weight = builder.reshape(builder.slice(packet, [layerOffset(shape, i)], [dims[0]! * dims[1]!]), dims);
    node = builder.gemm(node, weight, { bTranspose: true });
    if (i + 1 < layers) node = builder.relu(node);
  }
  const buildStart = performance.now();
  const graph = await builder.build({ y: builder.reshape(node, [HEIGHT, FOLD * 4]) });
  const metrics: Record<string, number> = { webnn_graph_build_ms: performance.now() - buildStart };

  // ---- Setup (once): the WebGPU side. `features` = encoded queries (f16), `halfResult` = WebNN's answers (f16), `rgb` = answers as f32. ----
  const { STORAGE, COPY_SRC, COPY_DST } = GPUBufferUsage;
  const features = gpu.createBuffer({ size: ROWS * 128, usage: STORAGE | COPY_SRC });
  const halfResult = gpu.createBuffer({ size: ROWS * 8, usage: STORAGE | COPY_DST });
  const rgb = gpu.createBuffer({ size: ROWS * 16, usage: STORAGE | COPY_SRC });
  const zero = gpu.createBuffer({ size: shape.width * 2, usage: COPY_SRC });
  const shader = gpu.createShaderModule({
    code: `
const NN_IN = 64u;
${encodingWgsl()}
${nrcInputsWgsl(true, nrcEncodingWgsl(4))}
@group(0) @binding(0) var<storage,read> records: array<vec4f>;
@group(0) @binding(1) var<uniform> nrcBox: NrcBox;
@group(0) @binding(2) var<storage,read_write> features: array<u32>;
@group(0) @binding(3) var<storage,read> halfRgb: array<vec2u>;
@group(0) @binding(4) var<storage,read_write> rgb: array<vec4f>;
@compute @workgroup_size(64) fn encode(@builtin(global_invocation_id) id: vec3u) {
  let r = id.x; let b = r * 5u;
  if (b + 4u >= arrayLength(&records)) { return; }
  let v = nrcInputsOf(records[b], records[b+1u], records[b+2u], records[b+4u]);
  for (var j=0u; j<32u; j++) { features[r*32u+j] = pack2x16float(vec2f(v[2u*j],v[2u*j+1u])); }
}
@compute @workgroup_size(64) fn unpack(@builtin(global_invocation_id) id: vec3u) {
  let h = halfRgb[id.x]; rgb[id.x] = vec4f(unpack2x16float(h.x),unpack2x16float(h.y));
}` });
  const encode = await gpu.createComputePipelineAsync({ layout: 'auto', compute: { module: shader, entryPoint: 'encode' } });
  const unpack = await gpu.createComputePipelineAsync({ layout: 'auto', compute: { module: shader, entryPoint: 'unpack' } });
  const encodeGroup = gpu.createBindGroup({
    layout: encode.getBindGroupLayout(0),
    entries: [records.queries, records.box, features].map((buffer, i) => ({ binding: i, resource: { buffer } })),
  });
  const unpackGroup = gpu.createBindGroup({
    layout: unpack.getBindGroupLayout(0),
    entries: [halfResult, rgb].map((buffer, i) => ({ binding: 3 + i, resource: { buffer } })),
  });

  // ---- One frame. `n` queries in `records.queries`, live weights in `mlp.useWeights`; answers land in `rgb`. ----
  let phase: Record<string, number> = {};
  const frame = async (n: number) => {
    if (n > ROWS) throw new Error('More queries than one dispatch holds');
    const t0 = performance.now();

    // Await 1 of 3: borrow the weight and input tensors from WebNN as WebGPU buffers.
    const [weightBuffer, xBuffer] = await Promise.all([ctx.exportToGPU(weights), ctx.exportToGPU(x)]);
    const t1 = performance.now();

    // One command buffer: encode the queries, then copy the weights and the encoded queries into the borrowed buffers.
    let encoder = gpu.createCommandEncoder();
    let pass = encoder.beginComputePass();
    pass.setPipeline(encode);
    pass.setBindGroup(0, encodeGroup);
    pass.dispatchWorkgroups(Math.ceil(n / 64));
    pass.end();
    encoder.copyBufferToBuffer(mlp.useWeights, 0, weightBuffer, 0, mlp.useWeights.size);
    encoder.copyBufferToBuffer(zero, 0, weightBuffer, mlp.useWeights.size, zero.size);
    encoder.copyBufferToBuffer(features, 0, xBuffer, 0, ROWS * 128);
    gpu.queue.submit([encoder.finish()]);
    weightBuffer.destroy(); // destroy() returns the tensor to WebNN.
    xBuffer.destroy();
    if (waitBeforeDispatch) await gpu.queue.onSubmittedWorkDone(); // Optional (`wait=1`).
    const t2 = performance.now();

    // The one dispatch of the frame. It returns at once; WebNN runs the graph behind it.
    ctx.dispatch(graph, { x, weights }, { y });

    // Await 2 of 3: borrow the output tensor. This resolves when the graph has run.
    const yBuffer = await ctx.exportToGPU(y);
    const t3 = performance.now();

    // One command buffer: copy the answers out and unpack f16 -> f32 for the renderer.
    encoder = gpu.createCommandEncoder();
    encoder.copyBufferToBuffer(yBuffer, 0, halfResult, 0, ROWS * 8);
    pass = encoder.beginComputePass();
    pass.setPipeline(unpack);
    pass.setBindGroup(0, unpackGroup);
    pass.dispatchWorkgroups(Math.ceil(n / 64));
    pass.end();
    gpu.queue.submit([encoder.finish()]);
    yBuffer.destroy();

    // Await 3 of 3: the frame ends when the answers are in `rgb` on the GPU. The WGSL timing below ends the same way.
    await gpu.queue.onSubmittedWorkDone();
    const t4 = performance.now();
    phase = { borrow_inputs: t1 - t0, copy_in: t2 - t1, dispatch_and_borrow_output: t3 - t2, copy_out_and_finish: t4 - t3 };
  };

  // ---- Measurement. Nothing below is part of a frame. ----
  const submit = (f: (e: GPUCommandEncoder) => void) => {
    const e = gpu.createCommandEncoder();
    f(e);
    gpu.queue.submit([e.finish()]);
  };
  const read = async (src: GPUBuffer, bytes: number) => {
    const b = gpu.createBuffer({ size: bytes, usage: COPY_DST | GPUBufferUsage.MAP_READ });
    try {
      submit(e => e.copyBufferToBuffer(src, 0, b, 0, bytes));
      await b.mapAsync(GPUMapMode.READ);
      return b.getMappedRange().slice(0);
    } finally { b.destroy(); }
  };
  try {
    // ---- Correctness check. NOT timed, and not part of a frame: `check=0` skips it and the times below do not change. ----
    // Four frames with different live weights, one after the other on purpose (each must see the weights the training just changed). WebNN's answers are compared with the renderer's own WGSL answers on every
    // row, and with a CPU forward pass (current weights, and the previous frame's weights to catch stale answers) on a sample of rows.
    let failures = 0, count = 0;
    let priorWeights: Float32Array | undefined;
    const snapshots = args.check === '0' ? 0 : 4;
    if (!snapshots) {
      for (let i = 0; i < 8; i++) submit(e => cache.encodeFrame(e));
      count = await cache.readQueryCount();
    }
    for (let sample = 0; sample < snapshots; sample++) {
      for (let i = 0; i < 8; i++) submit(e => cache.encodeFrame(e)); // Train, so the weights change.
      count = await cache.readQueryCount();
      if (count < 1) throw new Error('No NRC queries to replay');
      submit(records.infer);
      await frame(count);
      const [actual, native, encoded, live] = await Promise.all([
        read(rgb, count * 16), read(records.predictions, count * 16), read(features, count * 128), read(mlp.useWeights, mlp.useWeights.size),
      ]);
      const a = new Float32Array(actual), b = new Float32Array(native), q = new Uint16Array(encoded);
      const w = Float32Array.from(new Uint16Array(live), halfToFloat);
      let squared = 0, energy = 0, cpuWorst = 0, currentError = 0, staleError = 0;
      for (let j = 0; j < a.length; j++) { squared += (a[j]! - b[j]!) ** 2; energy += b[j]! ** 2; }
      for (let r = 0; r < count; r += Math.max(1, Math.floor(count / 128))) {
        const row = Float32Array.from(q.subarray(r * 64, r * 64 + 64), halfToFloat);
        const expected = forward(shape, w, row).output;
        const stale = priorWeights && forward(shape, priorWeights, row).output;
        for (let c = 0; c < 3; c++) {
          const delta = a[r * 4 + c]! - expected[c]!;
          cpuWorst = Math.max(cpuWorst, Math.abs(delta) / (0.01 + Math.abs(expected[c]!)));
          currentError += delta * delta;
          if (stale) staleError += (a[r * 4 + c]! - stale[c]!) ** 2;
        }
      }
      const relativeRms = Math.sqrt(squared / Math.max(energy, 1e-20));
      metrics[`snapshot${sample}_rrms`] = relativeRms;
      metrics[`snapshot${sample}_cpu_max_relative`] = cpuWorst;
      if (!(relativeRms < 0.01 && cpuWorst < 0.08)) failures++;
      if (priorWeights) {
        const ratio = Math.sqrt(currentError / Math.max(staleError, 1e-30));
        metrics[`snapshot${sample}_current_over_stale_error`] = ratio;
        if (!(ratio < 0.5)) failures++;
      }
      priorWeights = w;
    }
    metrics.parity_failures = failures;
    metrics.parity_snapshots_checked = snapshots;

    // ---- Timing. Each `frame(count)` call is one whole frame (all of its queries, one dispatch) and is timed on its own; the
    // reported number is the median of `repeats` single calls, never a sum. No read-back happens in this loop. ----
    // Our WGSL inference and one WebNN frame, alternating, each timed until the WebGPU queue is idle. Two warm-ups.
    const wgslTimes: number[] = [], nnTimes: number[] = [], phases: Record<string, number>[] = [];
    for (let i = 0; i < repeats + 2; i++) {
      let t = performance.now();
      submit(records.infer);
      await gpu.queue.onSubmittedWorkDone();
      if (i >= 2) wgslTimes.push(performance.now() - t);
      t = performance.now();
      await frame(count);
      if (i >= 2) { nnTimes.push(performance.now() - t); phases.push(phase); }
    }
    const percentile = (v: number[], p: number) => [...v].sort((m, n) => m - n)[Math.ceil(v.length * p) - 1]!;
    for (const name of Object.keys(phases[0]!)) metrics[`webnn_phase_${name}_p50_ms`] = percentile(phases.map(p => p[name]!), .5);
    metrics.webnn_p50_ms = percentile(nnTimes, .5);
    metrics.webnn_p95_ms = percentile(nnTimes, .95);
    metrics.wgsl_p50_ms = percentile(wgslTimes, .5);
    metrics.wgsl_p95_ms = percentile(wgslTimes, .95);
    metrics.webnn_slabs = 1;
    metrics.webnn_queries = count;
    return metrics;
  } finally {
    for (const b of [features, halfResult, rgb, zero]) b.destroy();
    for (const t of [x, y, weights]) t.destroy();
    graph.destroy();
    ctx.destroy();
  }
}
