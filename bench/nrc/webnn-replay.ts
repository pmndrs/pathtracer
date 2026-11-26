import { encodingWgsl, forward, layerCount, layerIn, layerOffset, layerOut, nrcEncodingWgsl } from '@pathtracer/mlp';
import { halfToFloat } from '../pages/mlp-common.ts';
import { type NrcCache, nrcInputsWgsl } from './cache.ts';
import { webnnFoldShapes, webnnReplayOptions } from './webnn-replay-options.ts';

// The interop API is experimental and has no DOM library declarations.
interface Tensor { destroy(): void }
interface Graph { destroy(): void }
interface Descriptor { dataType: 'float16'; shape: number[]; readable?: boolean; writable?: boolean }
interface Context {
  createExportableTensor(d: Descriptor, gpu: GPUDevice): Promise<Tensor>;
  createTensor(d: Descriptor): Promise<Tensor>;
  exportToGPU(t: Tensor): Promise<GPUBuffer>;
  writeTensor(t: Tensor, data: ArrayBufferView): void;
  readTensor(t: Tensor): Promise<ArrayBuffer>;
  dispatch(graph: Graph, inputs: Record<string, Tensor>, outputs: Record<string, Tensor>): void;
  destroy(): void;
}

interface Builder {
  input(name: string, d: Descriptor): object;
  gemm(a: object, b: object, options: { bTranspose: boolean; c?: object }): object;
  constant(d: Descriptor, data: ArrayBufferView): object;
  relu(a: object): object;
  slice(a: object, starts: number[], sizes: number[]): object;
  reshape(a: object, shape: number[]): object;
  build(outputs: Record<string, object>): Promise<Graph>;
}

export async function replayWebnn(cache: NrcCache, args: Record<string, string | undefined>): Promise<Record<string, number>> {
  const mlp = cache.mlp;
  if (!mlp || cache.settings.network !== 'compact' || mlp.precision !== 'f16' || cache.settings.materialInputs !== 'evaluated' ||
    cache.settings.factorization !== 'off' || cache.settings.positionBands !== 4) throw new Error('WebNN replay requires the compact four-band evaluated fp16 baseline');
  const rows = Number(args.rows ?? 4096), repeats = Number(args.repeats ?? 12);
  if (![64, 4096, 8192, 16384, 32768].includes(rows) || !Number.isSafeInteger(repeats) || repeats < 1 || repeats > 100) throw new Error('Invalid replay geometry or repeat count');
  const fold = webnnFoldShapes(rows, Number(args.fold ?? 1), mlp.shape.inputs);
  const slab = fold.slab; // Queries per dispatch. Equal to `rows` unless `fold` widens the tensors.
  if (args.weights && !['layers', 'packed'].includes(args.weights)) throw new Error('weights must be layers or packed');
  if (args.publish && !['frame', 'slab'].includes(args.publish)) throw new Error('publish must be frame or slab');
  if (args.device && !['gpu', 'npu'].includes(args.device)) throw new Error('device must be gpu or npu');
  // Chrome's ONNX Runtime backend on Windows fuses gemm + relu into DirectML's DmlFusedGemm, which requires the bias input C:
  // without one, build() fails with "Failed to create session" (input size 2 not in range [3, 3]). A zero C is exact.
  const gemmC = args.gemmC ?? (/Windows/.test(navigator.userAgent) ? 'zero' : 'none');
  if (!['zero', 'none'].includes(gemmC)) throw new Error('gemmC must be zero or none');
  const packed = (args.weights ?? 'packed') === 'packed';
  const perSlab = args.publish === 'slab';
  const deviceType = args.device ?? 'gpu';
  const opt = webnnReplayOptions(args);
  const ml = (navigator as unknown as { ml?: { createContext(options: { deviceType: string } | GPUDevice): Promise<Context> } }).ml;
  if (!ml) throw new Error('navigator.ml unavailable');
  const gpu = cache.device;
  const records = cache.replayBuffers;
  const ctx = await ml.createContext(opt.ctx === 'device' ? gpu : { deviceType });
  const builder = new (globalThis as unknown as { MLGraphBuilder: new (context: Context) => Builder }).MLGraphBuilder(ctx);
  const buffers: GPUBuffer[] = [];
  const tensors: Tensor[] = [];
  const make = (size: number, usage: GPUBufferUsageFlags) => {
    const b = gpu.createBuffer({ size, usage });
    buffers.push(b);
    return b;
  };
  const desc = (shape: number[]): Descriptor => ({ dataType: 'float16', shape });
  const tensor = async (shape: number[]) => {
    const t = opt.route === 'export' ? await ctx.createExportableTensor(desc(shape), gpu)
      : await ctx.createTensor({ ...desc(shape), readable: true, writable: true });
    tensors.push(t);
    return t;
  };
  const submit = (f: (e: GPUCommandEncoder) => void) => {
    const e = gpu.createCommandEncoder();
    f(e);
    gpu.queue.submit([e.finish()]);
  };
  const read = async (src: GPUBuffer, bytes: number) => {
    const b = gpu.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    try {
      submit(e => e.copyBufferToBuffer(src, 0, b, 0, bytes));
      await b.mapAsync(GPUMapMode.READ);
      return b.getMappedRange().slice(0);
    } finally { b.destroy(); }
  };
  const metrics: Record<string, number> = {};
  let graph: Graph | undefined;
  try {
    const { shape } = mlp;
    const layers = layerCount(shape);
    const paddedWeights = mlp.weightCount + shape.width; // Last matrix has four output rows. RGB are trained and the fourth row stays zero.
    const inputs: Record<string, Tensor> = { x: await tensor(fold.x) };
    let y = builder.input('x', desc(fold.x));
    if (slab !== rows) y = builder.reshape(y, fold.flatX);
    const packet = packed ? builder.input('weights', desc([paddedWeights])) : undefined;
    if (packet) inputs.weights = await tensor([paddedWeights]);
    for (let i = 0; i < layers; i++) {
      const dims = [i === layers - 1 ? 4 : layerOut(shape, i), layerIn(shape, i)];
      let weight;
      if (packet) weight = builder.reshape(builder.slice(packet, [layerOffset(shape, i)], [dims[0]! * dims[1]!]), dims);
      else {
        inputs['w' + i] = await tensor(dims);
        weight = builder.input('w' + i, desc(dims));
      }
      const c = gemmC === 'zero' ? builder.constant(desc([dims[0]!]), new Uint16Array(dims[0]!)) : undefined;
      y = builder.gemm(y, weight, c ? { bTranspose: true, c } : { bTranspose: true });
      if (i + 1 < layers) y = builder.relu(y);
    }
    if (slab !== rows) y = builder.reshape(y, fold.y);
    const start = performance.now();
    graph = await builder.build({ y });
    metrics.webnn_graph_build_ms = performance.now() - start;
    metrics.webnn_gemm_zero_c = gemmC === 'zero' ? 1 : 0;
    const output = await tensor(fold.y);
    const { STORAGE, COPY_SRC, COPY_DST } = GPUBufferUsage;
    const capacity = Math.ceil(records.queries.size / 80 / slab) * slab;
    const features = make(capacity * shape.inputs * 2, STORAGE | COPY_SRC);
    const halfResult = make(capacity * 8, STORAGE | COPY_DST);
    const rgb = make(capacity * 16, STORAGE | COPY_SRC);
    const zero = make(shape.width * 2, COPY_SRC);
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
    const pipeline = (entryPoint: string) => gpu.createComputePipelineAsync({ layout: 'auto', compute: { module: shader, entryPoint } });
    const [encode, unpack] = await Promise.all([pipeline('encode'), pipeline('unpack')]);
    const bind = (p: GPUComputePipeline, list: GPUBuffer[], start: number) => gpu.createBindGroup({
      layout: p.getBindGroupLayout(0), entries: list.map((buffer, i) => ({ binding: start + i, resource: { buffer } })),
    });
    const encGroup = bind(encode, [records.queries, records.box, features], 0);
    const unpackGroup = bind(unpack, [halfResult, rgb], 3);
    const dispatch = (p: GPUComputePipeline, group: GPUBindGroup, n: number) => submit(e => {
      const pass = e.beginComputePass();
      pass.setPipeline(p);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(Math.ceil(n / 64));
      pass.end();
    });
    const publish = async () => {
      if (opt.route === 'cpu') {
        // Through script: read the live weights back and write them into the WebNN tensor(s), padded like the exported copy.
        const live = new Uint16Array(await read(mlp.useWeights, mlp.useWeights.size));
        if (packed) {
          const padded = new Uint16Array(paddedWeights);
          padded.set(live);
          ctx.writeTensor(inputs.weights!, padded);
        } else for (let i = 0; i < layers; i++) {
          const halves = layerIn(shape, i) * layerOut(shape, i);
          const layer = new Uint16Array(i + 1 === layers ? halves + shape.width : halves);
          layer.set(live.subarray(layerOffset(shape, i), layerOffset(shape, i) + halves));
          ctx.writeTensor(inputs['w' + i]!, layer);
        }
        return;
      }
      if (packed) {
        const dst = await ctx.exportToGPU(inputs.weights!);
        submit(e => {
          e.copyBufferToBuffer(mlp.useWeights, 0, dst, 0, mlp.useWeights.size);
          e.copyBufferToBuffer(zero, 0, dst, mlp.useWeights.size, zero.size);
        });
        dst.destroy();
      } else for (let i = 0; i < layers; i++) {
        const dst = await ctx.exportToGPU(inputs['w' + i]!);
        const bytes = layerIn(shape, i) * layerOut(shape, i) * 2;
        submit(e => {
          e.copyBufferToBuffer(mlp.useWeights, layerOffset(shape, i) * 2, dst, 0, bytes);
          if (i + 1 === layers) e.copyBufferToBuffer(zero, 0, dst, bytes, zero.size);
        });
        dst.destroy();
      }
    };
    // Per-phase milliseconds of one replay: timestamps around the awaits the variant already has, never an extra wait.
    let phase: Record<string, number> = {};
    let mark = 0;
    const lap = (name: string) => {
      const now = performance.now();
      phase[name] = (phase[name] ?? 0) + now - mark;
      mark = now;
    };
    // `tensors=shared` (the November 19 sequence): one input / output pair, each chunk lent out, copied, returned and run in turn.
    const sequential = async (n: number) => {
      if (!perSlab) { await publish(); lap('weights'); }
      const host = opt.route === 'cpu' ? new Uint8Array(await read(features, Math.ceil(n / slab) * slab * 128)) : undefined;
      if (host) lap('in');
      for (let offset = 0; offset < n; offset += slab) {
        if (perSlab) { await publish(); lap('weights'); }
        if (host) ctx.writeTensor(inputs.x!, host.subarray(offset * 128, (offset + slab) * 128));
        else {
          const x = await ctx.exportToGPU(inputs.x!);
          submit(e => e.copyBufferToBuffer(features, offset * 128, x, 0, slab * 128));
          x.destroy();
        }
        lap('in');
        if (opt.route === 'export' && opt.sync === 'drain') { await gpu.queue.onSubmittedWorkDone(); lap('wait'); }
        ctx.dispatch(graph!, inputs, { y: output });
        lap('dispatch');
        if (host) gpu.queue.writeBuffer(halfResult, offset * 8, await ctx.readTensor(output));
        else {
          const out = await ctx.exportToGPU(output);
          submit(e => e.copyBufferToBuffer(out, 0, halfResult, offset * 8, slab * 8));
          out.destroy();
        }
        lap('out');
      }
    };
    // `tensors=chunk`: a tensor pair per chunk, so the whole frame crosses in one batch each way (MLTensor explainer, chained inference).
    const chunks: { x: Tensor; y: Tensor }[] = [];
    const weightInputs = Object.fromEntries(Object.entries(inputs).filter(([name]) => name !== 'x'));
    const batched = async (n: number) => {
      while (chunks.length < Math.ceil(n / slab)) chunks.push({ x: await tensor(fold.x), y: await tensor(fold.y) });
      const list = chunks.slice(0, Math.ceil(n / slab));
      await publish();
      lap('weights');
      if (opt.route === 'cpu') {
        const host = new Uint8Array(await read(features, list.length * slab * 128));
        list.forEach((c, k) => ctx.writeTensor(c.x, host.subarray(k * slab * 128, (k + 1) * slab * 128)));
        lap('in');
        for (const c of list) ctx.dispatch(graph!, { ...weightInputs, x: c.x }, { y: c.y });
        lap('dispatch');
        const outs = await Promise.all(list.map(c => ctx.readTensor(c.y)));
        outs.forEach((o, k) => gpu.queue.writeBuffer(halfResult, k * slab * 8, o));
        lap('out');
        return;
      }
      const xs = await Promise.all(list.map(c => ctx.exportToGPU(c.x)));
      submit(e => xs.forEach((b, k) => e.copyBufferToBuffer(features, k * slab * 128, b, 0, slab * 128)));
      for (const b of xs) b.destroy();
      lap('in');
      if (opt.sync === 'drain') { await gpu.queue.onSubmittedWorkDone(); lap('wait'); }
      for (const c of list) ctx.dispatch(graph!, { ...weightInputs, x: c.x }, { y: c.y });
      lap('dispatch');
      const ys = await Promise.all(list.map(c => ctx.exportToGPU(c.y)));
      submit(e => ys.forEach((b, k) => e.copyBufferToBuffer(b, 0, halfResult, k * slab * 8, slab * 8)));
      for (const b of ys) b.destroy();
      lap('out');
    };
    const webnn = async (n: number) => {
      phase = {};
      mark = performance.now();
      dispatch(encode, encGroup, n);
      lap('encode');
      if (opt.tensors === 'chunk') await batched(n);
      else await sequential(n);
      dispatch(unpack, unpackGroup, n);
      await gpu.queue.onSubmittedWorkDone();
      lap('finish');
    };
    let parityFailures = 0;
    const parityFail = (message: string) => {
      if (opt.parity === 'strict') throw new Error(message);
      parityFailures++;
    };
    let count = 0;
    let priorWeights: Float32Array | undefined;
    for (let sample = 0; sample < 4; sample++) {
      // Intentionally change the EMA. The same trained weights and real path records feed both engines.
      for (let i = 0; i < 8; i++) submit(e => cache.encodeFrame(e));
      count = await cache.readQueryCount();
      if (count < 1) throw new Error('No NRC queries to replay');
      submit(records.infer);
      await webnn(count);
      const [actual, native, encoded, weights] = await Promise.all([
        read(rgb, count * 16), read(records.predictions, count * 16), read(features, count * 128), read(mlp.useWeights, mlp.useWeights.size),
      ]);
      const a = new Float32Array(actual), b = new Float32Array(native), x = new Uint16Array(encoded);
      const w = Float32Array.from(new Uint16Array(weights), halfToFloat);
      let squared = 0, energy = 0, worst = 0, cpuWorst = 0, correctError = 0, staleError = 0;
      for (let j = 0; j < a.length; j++) {
        const delta = Math.abs(a[j]! - b[j]!);
        squared += delta * delta; energy += b[j]! ** 2; worst = Math.max(worst, delta / (0.001 + Math.abs(b[j]!)));
      }
      for (let r = 0; r < count; r += Math.max(1, Math.floor(count / 128))) {
        const row = Float32Array.from(x.subarray(r * 64, r * 64 + 64), halfToFloat);
        const expected = forward(shape, w, row).output;
        const stale = priorWeights && forward(shape, priorWeights, row).output;
        for (let c = 0; c < 3; c++) {
          const delta = a[r * 4 + c]! - expected[c]!;
          cpuWorst = Math.max(cpuWorst, Math.abs(delta) / (0.01 + Math.abs(expected[c]!)));
          correctError += delta * delta;
          if (stale) staleError += (a[r * 4 + c]! - stale[c]!) ** 2;
        }
      }
      if (priorWeights) {
        const ratio = Math.sqrt(correctError / Math.max(staleError, 1e-30));
        metrics[`snapshot${sample}_current_over_stale_error`] = ratio;
        if (!(ratio < 0.5)) parityFail('Could not distinguish consecutive live weight snapshots: ' + ratio);
      }
      priorWeights = w;
      const relativeRms = Math.sqrt(squared / Math.max(energy, 1e-20));
      Object.assign(metrics, { [`snapshot${sample}_rows`]: count, [`snapshot${sample}_rrms`]: relativeRms, [`snapshot${sample}_max_relative`]: worst, [`snapshot${sample}_cpu_max_relative`]: cpuWorst });
      if (!(relativeRms < 0.01 && cpuWorst < 0.08)) parityFail('WebNN consecutive snapshot parity failed: ' + JSON.stringify(metrics));
    }
    metrics.parity_failures = parityFailures;
    const wgslTimes: number[] = [], nnTimes: number[] = [];
    const phaseRuns: Record<string, number>[] = [];
    for (let i = 0; i < repeats + 2; i++) {
      let t = performance.now();
      submit(records.infer);
      await gpu.queue.onSubmittedWorkDone();
      if (i >= 2) wgslTimes.push(performance.now() - t);
      t = performance.now();
      await webnn(count);
      if (i >= 2) {
        nnTimes.push(performance.now() - t);
        phaseRuns.push(phase);
      }
    }
    if (opt.breakdown) {
      for (const name of new Set(phaseRuns.flatMap(p => Object.keys(p)))) {
        const v = phaseRuns.map(p => p[name] ?? 0).sort((a, b) => a - b);
        metrics[`webnn_phase_${name}_p50_ms`] = v[Math.ceil(v.length * .5) - 1]!;
      }
    }
    for (const [prefix, times] of [['webnn', nnTimes], ['wgsl', wgslTimes]] as const) {
      times.sort((a, b) => a - b);
      metrics[prefix + '_p50_ms'] = times[Math.ceil(times.length * .5) - 1]!;
      metrics[prefix + '_p95_ms'] = times[Math.ceil(times.length * .95) - 1]!;
    }
    metrics.webnn_slabs = Math.ceil(count / slab);
    metrics.webnn_padded_rows = metrics.webnn_slabs * slab - count;
    return metrics;
  } finally {
    for (const buffer of buffers) buffer.destroy();
    for (const t of tensors) t.destroy();
    graph?.destroy();
    ctx.destroy();
  }
}
