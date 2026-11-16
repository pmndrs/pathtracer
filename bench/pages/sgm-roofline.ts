/**
 * Research track N: how fast can WGSL subgroup matrices go on this device, with none of the MLP around them?
 * Two kernels, both f16 operands, accumulator `acc=f32|f16`:
 *
 *   mode=peak   Each subgroup loads `a` left and `b` right tiles once, then runs `iter` rounds of a × b independent
 *               multiply-accumulates on them (no loads in the loop). The ceiling of the matrix units through Dawn/Tint.
 *   mode=chain  The MLP's data flow without encoding, ReLU or biases: `records` rows per workgroup, `layers` layers of
 *               64 × 64 weights. Activations live in workgroup memory (row-major, stride 64 + pad); for every output
 *               column block a subgroup loads the 4 right tiles of that block (from `weights=global` storage or
 *               `weights=shared` workgroup memory, copied once per layer) and reuses them over its row blocks.
 *               With acc=f16 the result tile is stored straight into the other activation buffer (ping-pong); with
 *               acc=f32 it goes to a result buffer and one pass converts it back to f16, as the MLP's ReLU pass does.
 *               Inputs are generated in the kernel; 4 outputs per record are written to storage.
 *               Correctness: the weights are a cyclic shift (out[n] = in[(n + 1) % 64]), so after L layers record r's
 *               output n equals its input (n + L) % 64; the page checks the first 4096 records exactly.
 *
 * Args: mode=peak|chain  acc=f32|f16  subgroups=4  repeats=20  warmup=3
 *       peak:  a=2 b=2 iter=1024 groups=1024
 *       chain: layers=6 records=64 (per workgroup) size=1024 (records = size²) weights=global|shared pad=0
 *              tiling=row (the MLP's: per output block, one accumulator per row block, a left load per MMA) |
 *                     block (all 16 right tiles of a layer held in registers, 4 left loads feed 4 accumulators)
 * Timestamps around each dispatch's pass; the only readback is after all timed passes.
 */

import { DEFAULT_SHAPE, pickSubgroupMatrixConfig, type SgmConfig, type SgmScalar } from '@pathtracer/mlp';
import { drainDevice, getArgs, numberArg, reportError, reportResult, summarizeAdapter, watchDevice } from '../runner/page-client.ts';
import type { AdapterSummary } from '../runner/contract.ts';
import { hasSubgroupMatrix, readBuffer, requestMlpDevice, storageBuffer } from './mlp-common.ts';

let adapterSummary: AdapterSummary | null = null;
let deviceErrors: string[] = [];

const median = (v: number[]) => [...v].sort((x, y) => x - y)[Math.floor(v.length / 2)] ?? NaN;
const W = 64;

interface Common {
  c: SgmConfig;
  enableName: string;
  subgroups: number;
  lanes: number;
}

function header(o: Common): string {
  const d = [`enable ${o.enableName};`, 'enable f16;'];
  if (o.subgroups > 1) d.push('enable subgroups;');
  return d.join('\n');
}

function types(c: SgmConfig): string {
  return `alias TR = ${c.resultComponentType};
alias L = subgroup_matrix_left<f16, ${c.K}, ${c.M}>;
alias R = subgroup_matrix_right<f16, ${c.N}, ${c.K}>;
alias A = subgroup_matrix_result<TR, ${c.N}, ${c.M}>;`;
}

function sgBuiltin(o: Common): { param: string; expr: string } {
  return o.subgroups > 1 ? { param: ', @builtin(subgroup_id) sgid: u32', expr: 'sgid' } : { param: '', expr: '0u' };
}

export function peakWgsl(o: Common & { a: number; b: number; iter: number }): string {
  const { c } = o;
  const sg = sgBuiltin(o);
  const lefts = Array.from({ length: o.a }, (_, i) => `  let l${i} = subgroupMatrixLoad<L>(&inp, ${i * c.M * c.K}u, false, ${c.K}u);`);
  const rights = Array.from({ length: o.b }, (_, j) => `  let r${j} = subgroupMatrixLoad<R>(&inp, ${(o.a * c.M * c.K + j * c.K * c.N)}u, true, ${c.K}u);`);
  const accs: string[] = [];
  const mmas: string[] = [];
  const stores: string[] = [];
  for (let i = 0; i < o.a; i++)
    for (let j = 0; j < o.b; j++) {
      accs.push(`  var c${i}_${j} = A();`);
      mmas.push(`    c${i}_${j} = subgroupMatrixMultiplyAccumulate(l${i}, r${j}, c${i}_${j});`);
      stores.push(`  subgroupMatrixStore(&out, slot + ${(i * o.b + j) * c.M * c.N}u, c${i}_${j}, false, ${c.N}u);`);
    }
  const tile = o.a * o.b * c.M * c.N;
  return `${header(o)}
${types(c)}
@group(0) @binding(0) var<storage, read> inp: array<f16>;
@group(0) @binding(1) var<storage, read_write> out: array<TR>;

@compute @workgroup_size(${o.subgroups * o.lanes})
fn main(@builtin(workgroup_id) wid: vec3u${sg.param}) {
${lefts.join('\n')}
${rights.join('\n')}
${accs.join('\n')}
  // Constant bound with the test first: Tint adds no loop counter (P-W1).
  for (var it = 0u; it < ${o.iter}u; it++) {
${mmas.join('\n')}
  }
  // 64 slots per subgroup index: writes from different workgroups overlap (timing only, the values are not read).
  let slot = ((wid.x % 64u) * ${o.subgroups}u + ${sg.expr}) * ${tile}u;
${stores.join('\n')}
}
`;
}

export function chainWgsl(o: Common & { layers: number; records: number; weights: 'global' | 'shared'; pad: number; tiling: 'row' | 'block' }): string {
  const { c } = o;
  const sg = sgBuiltin(o);
  const S = W + o.pad;
  const Rr = o.records;
  const wg = o.subgroups * o.lanes;
  const rowBlocks = Rr / c.M / o.subgroups;
  const direct = c.resultComponentType === 'f16';
  const kBlocks = W / c.K;
  const wPtr = o.weights === 'shared' ? '&wsh' : '&wts';
  const layer = (l: number) => {
    const src = `act${l % 2}`;
    const dst = direct ? `act${(l + 1) % 2}` : 'res';
    const copy = o.weights === 'shared'
      ? `  for (var i = lid; i < ${W * W}u; i += ${wg}u) { wsh[i] = wts[${l * W * W}u + i]; }
  workgroupBarrier();
`
      : '';
    const wOff = o.weights === 'shared' ? 0 : l * W * W;
    const convert = direct
      ? ''
      : `  for (var i = lid; i < ${Rr * W}u; i += ${wg}u) {
    let at = (i / ${W}u) * ${S}u + i % ${W}u;
    act${(l + 1) % 2}[at] = f16(res[at]);
  }
  workgroupBarrier();
`;
    const nBlocks = W / c.N;
    if (o.tiling === 'block') {
      // Register blocking: the layer's nBlocks x kBlocks right tiles are loaded once per subgroup and held; each row
      // block loads its kBlocks left tiles once and feeds nBlocks accumulators (kBlocks x nBlocks MMAs per row block).
      const rights: string[] = [];
      for (let nb = 0; nb < nBlocks; nb++)
        for (let kb = 0; kb < kBlocks; kb++) rights.push(`  let r${l}_${nb}_${kb} = subgroupMatrixLoad<R>(${wPtr}, ${wOff + nb * c.N * W + kb * c.K}u, true, ${W}u);`);
      const lefts = Array.from({ length: kBlocks }, (_, kb) => `    let l${kb} = subgroupMatrixLoad<L>(&${src}, rowBase + ${kb * c.K}u, false, ${S}u);`);
      const body: string[] = [];
      for (let nb = 0; nb < nBlocks; nb++) {
        body.push(`    var a${nb} = A();`);
        for (let kb = 0; kb < kBlocks; kb++) body.push(`    a${nb} = subgroupMatrixMultiplyAccumulate(l${kb}, r${l}_${nb}_${kb}, a${nb});`);
        body.push(`    subgroupMatrixStore(&${dst}, rowBase + ${nb * c.N}u, a${nb}, false, ${S}u);`);
      }
      return `  // Layer ${l} (register-blocked)
${copy}${rights.join('\n')}
  for (var mbi = 0u; mbi < ${rowBlocks}u; mbi++) {
    let rowBase = (sg * ${rowBlocks}u + mbi) * ${c.M * S}u;
${lefts.join('\n')}
${body.join('\n')}
  }
  workgroupBarrier();
${convert}`;
    }
    const rights = Array.from({ length: kBlocks }, (_, kb) => `      let r${kb} = subgroupMatrixLoad<R>(${wPtr}, ${wOff}u + nb * ${c.N * W}u + ${kb * c.K}u, true, ${W}u);`);
    const mmas = Array.from({ length: kBlocks }, (_, kb) => `        acc = subgroupMatrixMultiplyAccumulate(subgroupMatrixLoad<L>(&${src}, rowBase + ${kb * c.K}u, false, ${S}u), r${kb}, acc);`);
    return `  // Layer ${l}
${copy}  for (var nb = 0u; nb < ${W / c.N}u; nb++) {
${rights.join('\n')}
    for (var mbi = 0u; mbi < ${rowBlocks}u; mbi++) {
      let rowBase = (sg * ${rowBlocks}u + mbi) * ${c.M * S}u;
      var acc = A();
${mmas.join('\n')}
      subgroupMatrixStore(&${dst}, rowBase + nb * ${c.N}u, acc, false, ${S}u);
    }
  }
  workgroupBarrier();
${convert}`;
  };
  const last = `act${o.layers % 2}`;
  return `${header(o)}
${types(c)}
@group(0) @binding(0) var<storage, read> wts: array<f16>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

var<workgroup> act0: array<f16, ${Rr * S}>;
var<workgroup> act1: array<f16, ${Rr * S}>;
${direct ? '' : `var<workgroup> res: array<TR, ${Rr * S}>;\n`}${o.weights === 'shared' ? `var<workgroup> wsh: array<f16, ${W * W}>;\n` : ''}
@compute @workgroup_size(${wg})
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u, @builtin(local_invocation_index) lid: u32${sg.param}) {
  let base = (wid.x + wid.y * nwg.x) * ${Rr}u;
  let sg = ${sg.expr};
  // Inputs made in the kernel: small multiples of 1/8, exact in f16.
  for (var i = lid; i < ${Rr * W}u; i += ${wg}u) {
    let r = base + i / ${W}u;
    let k = i % ${W}u;
    act0[(i / ${W}u) * ${S}u + k] = f16(f32((r * 7u + k) % 13u) * 0.125);
  }
  workgroupBarrier();
${Array.from({ length: o.layers }, (_, l) => layer(l)).join('')}
  for (var i = lid; i < ${Rr * 4}u; i += ${wg}u) {
    out[base * 4u + i] = f32(${last}[(i / 4u) * ${S}u + i % 4u]);
  }
}
`;
}

/** The chain kernel's weights: a cyclic shift per layer, row-major [out][in] like the MLP (out[n] = in[(n + 1) % W]). */
function shiftWeights(layers: number): Uint16Array {
  const w = new Uint16Array(layers * W * W);
  const one = 0x3c00;
  for (let l = 0; l < layers; l++) for (let n = 0; n < W; n++) w[l * W * W + n * W + ((n + 1) % W)] = one;
  return w;
}

try {
  const args = getArgs();
  const mode = args.mode ?? 'peak';
  const acc = (args.acc ?? 'f32') as SgmScalar;
  const subgroups = numberArg(args, 'subgroups', 4);
  const repeats = numberArg(args, 'repeats', 20);
  const warmup = numberArg(args, 'warmup', 3);

  const { adapter, device } = await requestMlpDevice();
  adapterSummary = summarizeAdapter(adapter);
  deviceErrors = watchDevice(device);
  if (!device.features.has('timestamp-query')) throw new Error('sgm-roofline needs timestamp-query.');
  if (!hasSubgroupMatrix(device)) throw new Error('sgm-roofline needs subgroup matrices (Dawn Vulkan on NVIDIA, Chrome on Apple).');
  if (!device.features.has('shader-f16')) throw new Error('sgm-roofline needs shader-f16 (NVIDIA Vulkan: toggle vulkan_enable_f16_on_nvidia).');
  const c = pickSubgroupMatrixConfig(adapter.info, DEFAULT_SHAPE, true, acc);
  if (!c || c.componentType !== 'f16' || c.resultComponentType !== acc) throw new Error(`no f16->${acc} subgroup matrix config on this adapter`);
  const enableName = device.features.has('subgroup-matrix' as GPUFeatureName) ? 'subgroup_matrix' : 'chromium_experimental_subgroup_matrix';
  const common: Common = { c, enableName, subgroups, lanes: c.maxSubgroupSize };

  let code: string;
  let buffers: GPUBuffer[];
  let dispatch: [number, number];
  let flop: number;
  let records = 0;
  let check: (() => Promise<{ checked: number; wrong: number }>) | null = null;
  let config: Record<string, unknown>;

  if (mode === 'peak') {
    const a = numberArg(args, 'a', 2);
    const b = numberArg(args, 'b', 2);
    const iter = numberArg(args, 'iter', 1024);
    const groups = numberArg(args, 'groups', 1024);
    code = peakWgsl({ ...common, a, b, iter });
    const inp = new Uint16Array((a * c.M * c.K + b * c.K * c.N));
    inp.fill(0x2000); // 2^-7: sums of 1024 x 2^-14 per element stay finite in f16
    const outBytes = 64 * subgroups * a * b * c.M * c.N * (acc === 'f16' ? 2 : 4);
    buffers = [storageBuffer(device, inp, 'inp'), storageBuffer(device, outBytes, 'out')];
    dispatch = [groups, 1];
    flop = groups * subgroups * iter * a * b * 2 * c.M * c.N * c.K;
    config = { mode, a, b, iter, groups };
  } else if (mode === 'chain') {
    const layers = numberArg(args, 'layers', 6);
    const perGroup = numberArg(args, 'records', 64);
    const size = numberArg(args, 'size', 1024);
    const weights = (args.weights ?? 'global') as 'global' | 'shared';
    const pad = numberArg(args, 'pad', 0);
    const tiling = (args.tiling ?? 'row') as 'row' | 'block';
    if (perGroup % (c.M * subgroups)) throw new Error(`records must be a multiple of M x subgroups = ${c.M * subgroups}`);
    records = size * size;
    if (records % perGroup) throw new Error('size² must be a multiple of records');
    code = chainWgsl({ ...common, layers, records: perGroup, weights, pad, tiling });
    const out = storageBuffer(device, records * 16, 'out');
    buffers = [storageBuffer(device, shiftWeights(layers), 'weights'), out];
    const groups = records / perGroup;
    dispatch = [Math.min(groups, 65535), Math.ceil(groups / 65535)];
    if (dispatch[0] * dispatch[1] !== groups) throw new Error(`${groups} workgroups do not tile a ${dispatch[0]} x ${dispatch[1]} grid`);
    flop = records * layers * 2 * W * W;
    const bytes = perGroup * (W + pad) * (2 + 2 + (acc === 'f16' ? 0 : 4)) + (weights === 'shared' ? W * W * 2 : 0);
    config = { mode, layers, recordsPerGroup: perGroup, size, records, weights, pad, tiling, workgroupBytes: bytes };
    check = async () => {
      const n = Math.min(records, 4096);
      const got = new Float32Array(await readBuffer(device, out, n * 16));
      let wrong = 0;
      for (let r = 0; r < n; r++)
        for (let j = 0; j < 4; j++) {
          const k = (j + layers) % W;
          if (got[r * 4 + j] !== ((r * 7 + k) % 13) * 0.125) wrong++;
        }
      return { checked: n * 4, wrong };
    };
  } else throw new Error(`mode must be peak or chain, got ${mode}`);

  device.pushErrorScope('validation');
  const module = device.createShaderModule({ label: `sgm-roofline ${mode}`, code });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((m) => m.type === 'error');
  if (errors.length) throw new Error(`WGSL: ${errors.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join('; ')}\n${code}`);
  const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
  const bind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
  const scopeError = await device.popErrorScope();
  if (scopeError) throw new Error(`pipeline: ${scopeError.message}`);

  const querySet = device.createQuerySet({ type: 'timestamp', count: 2 * (warmup + repeats) });
  for (let i = 0; i < warmup + repeats; i++) {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass({ timestampWrites: { querySet, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 } });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(dispatch[0], dispatch[1]);
    pass.end();
    device.queue.submit([encoder.finish()]);
    // One pass in flight at a time (no queued-up batches, CLAUDE.md Traps).
    await device.queue.onSubmittedWorkDone();
  }
  const resolve = device.createBuffer({ size: querySet.count * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
  const encoder = device.createCommandEncoder();
  encoder.resolveQuerySet(querySet, 0, querySet.count, resolve, 0);
  device.queue.submit([encoder.finish()]);
  const t = new BigUint64Array(await readBuffer(device, resolve));
  const ms = Array.from({ length: repeats }, (_, i) => Number((t[2 * (warmup + i) + 1] as bigint) - (t[2 * (warmup + i)] as bigint)) / 1e6);
  const correctness = check ? await check() : null;
  await drainDevice(device);
  if (correctness?.wrong) throw new Error(`chain kernel wrong: ${correctness.wrong} of ${correctness.checked} outputs differ from the shifted input`);

  const med = median(ms);
  const software = adapterSummary.isFallbackAdapter || /swiftshader/i.test(`${adapterSummary.architecture} ${adapterSummary.description}`);
  reportResult(
    {
      scene: `sgm-roofline-${mode}`,
      renderer: 'wgsl-sgm',
      config: { ...config, acc, subgroups, config: `${c.M}x${c.N}x${c.K} f16->${acc}`, repeats, warmup, wgslBytes: code.length },
      metrics: {
        gpu_ms_median: med,
        gpu_ms_min: Math.min(...ms),
        tflops: flop / (med * 1e-3) / 1e12,
        tflops_best: flop / (Math.min(...ms) * 1e-3) / 1e12,
        ...(records ? { gpu_ms_per_mrecord: (med * 1e6) / records } : {}),
      },
      valid: !software,
      notes: { software, ms, ...(correctness ? { correctness } : {}) },
    },
    { adapter: adapterSummary, deviceErrors },
  );
} catch (error) {
  reportError(error, { adapter: adapterSummary, deviceErrors });
}
