/**
 * bsdf-check: correctness only (no timing). Runs `ptBsdfEval` and `ptBsdfSample` (WGSL, packages/pathtracer/src/bsdf)
 * on the GPU for a fixed seeded set of inputs and compares with the TS twin (f64, inputs rounded to f32 first).
 *
 * Three module configurations: VNDF standard + radiance transport, VNDF spherical cap + radiance, standard + importance.
 * Cases mix diffuse, smooth/rough conductor (gold, isotropic and anisotropic alpha), smooth/rough dielectric (ior 1.5,
 * wo on both sides), black. Tolerance: |gpu - cpu| <= tol * max(|gpu|, |cpu|) + abs for f and pdf, direction 1e-3 abs.
 * A flag mismatch (u at a Fresnel decision boundary, grazing TIR) is counted, not compared, and must stay under 0.5 %.
 *
 * Args: cases=3000 (per configuration), tol=5e-3
 */

import { drainDevice, getArgs, numberArg, reportError, reportResult, summarizeAdapter, watchDevice } from '../runner/page-client.ts';
import type { AdapterSummary } from '../runner/contract.ts';
import { PT_BSDF_BLACK, PT_BSDF_CONDUCTOR, PT_BSDF_DIELECTRIC, PT_BSDF_DIFFUSE, ptBsdfEval, ptBsdfSample, type PtBsdfOptions, type PtBsdfParams, type PtVec3 } from '../../packages/pathtracer/src/bsdf/bsdf.ts';
import { ptBsdfWgsl } from '../../packages/pathtracer/src/bsdf/bsdf-wgsl.ts';

const IN_STRIDE = 24;
const OUT_STRIDE = 16;

function shaderSource(options: PtBsdfOptions): string {
  return `${ptBsdfWgsl(options)}
@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> outp: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= arrayLength(&inp) / ${IN_STRIDE}u) { return; }
  let b = i * ${IN_STRIDE}u;
  var p: PtBsdfParams;
  p.kind = u32(inp[b]);
  p.ior = inp[b + 1u];
  p.alpha = vec2f(inp[b + 2u], inp[b + 3u]);
  p.reflectance = vec3f(inp[b + 4u], inp[b + 5u], inp[b + 6u]);
  p.eta = vec3f(inp[b + 7u], inp[b + 8u], inp[b + 9u]);
  p.k = vec3f(inp[b + 10u], inp[b + 11u], inp[b + 12u]);
  let wo = vec3f(inp[b + 13u], inp[b + 14u], inp[b + 15u]);
  let wi = vec3f(inp[b + 16u], inp[b + 17u], inp[b + 18u]);
  let u = vec3f(inp[b + 19u], inp[b + 20u], inp[b + 21u]);
  let e = ptBsdfEval(p, wo, wi);
  let s = ptBsdfSample(p, wo, u);
  let o = i * ${OUT_STRIDE}u;
  outp[o] = e.f.x; outp[o + 1u] = e.f.y; outp[o + 2u] = e.f.z; outp[o + 3u] = e.pdf;
  outp[o + 4u] = s.wi.x; outp[o + 5u] = s.wi.y; outp[o + 6u] = s.wi.z;
  outp[o + 7u] = s.f.x; outp[o + 8u] = s.f.y; outp[o + 9u] = s.f.z;
  outp[o + 10u] = s.pdf; outp[o + 11u] = f32(s.flags); outp[o + 12u] = s.etap;
}
`;
}

function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const f32 = Math.fround;
const GOLD_ETA: PtVec3 = [0.143, 0.375, 1.442];
const GOLD_K: PtVec3 = [3.983, 2.386, 1.603];
const ROUGH_ALPHA = [0, 0.05, 0.1, 0.25, 0.5, 1].map((r) => r * r);

interface Case {
  p: PtBsdfParams;
  wo: PtVec3;
  wi: PtVec3;
  u: PtVec3;
}

function unit(rand: () => number, upperOnly: boolean): PtVec3 {
  const z0 = 2 * rand() - 1;
  const z = upperOnly ? Math.abs(z0) : z0;
  const s = Math.sqrt(Math.max(0, 1 - z * z));
  const phi = 2 * Math.PI * rand();
  const v: PtVec3 = [f32(s * Math.cos(phi)), f32(s * Math.sin(phi)), f32(z)];
  const l = Math.hypot(v[0], v[1], v[2]);
  return [f32(v[0] / l), f32(v[1] / l), f32(v[2] / l)];
}

function makeCases(n: number, seed: number): Case[] {
  const rand = rng(seed);
  const cases: Case[] = [];
  for (let i = 0; i < n; i++) {
    const kind = [PT_BSDF_DIFFUSE, PT_BSDF_CONDUCTOR, PT_BSDF_CONDUCTOR, PT_BSDF_DIELECTRIC, PT_BSDF_DIELECTRIC, PT_BSDF_BLACK][Math.floor(rand() * 6)] as number;
    const a = ROUGH_ALPHA[Math.floor(rand() * ROUGH_ALPHA.length)] as number;
    const aniso = rand() < 0.2;
    const alpha: [number, number] = aniso ? [f32(a), f32(Math.max(0.05, a * (0.3 + rand())))] : [f32(a), f32(a)];
    const p: PtBsdfParams = {
      kind,
      reflectance: [f32(rand()), f32(rand()), f32(rand())],
      eta: kind === PT_BSDF_CONDUCTOR ? GOLD_ETA.map(f32) as unknown as PtVec3 : [0, 0, 0],
      k: kind === PT_BSDF_CONDUCTOR ? GOLD_K.map(f32) as unknown as PtVec3 : [0, 0, 0],
      alpha,
      ior: f32(rand() < 0.5 ? 1.5 : 1.33),
    };
    // Dielectric: both sides; others: upper hemisphere (Lambert and the conductor also accept the lower one, test some).
    const wo = unit(rand, kind !== PT_BSDF_DIELECTRIC && rand() < 0.9);
    // Half of the eval directions are far from the lobes (uniform), keep that; the sampled direction is checked in the sample path.
    const wi = unit(rand, false);
    cases.push({ p, wo, wi, u: [f32(rand()), f32(rand()), f32(rand())] });
  }
  return cases;
}

function pack(cases: Case[]): Float32Array<ArrayBuffer> {
  const data = new Float32Array(new ArrayBuffer(cases.length * IN_STRIDE * 4));
  cases.forEach((c, i) => {
    data.set([c.p.kind, c.p.ior, c.p.alpha[0], c.p.alpha[1], ...c.p.reflectance, ...c.p.eta, ...c.p.k, ...c.wo, ...c.wi, ...c.u], i * IN_STRIDE);
  });
  return data;
}

interface Stats {
  evalCompared: number;
  evalBad: number;
  evalMaxRel: number;
  sampleCompared: number;
  sampleBad: number;
  sampleMaxRel: number;
  flagMismatch: number;
  nonFinite: number;
  failedSamples: number;
  basisSkips: number;
  firstBad: string[];
}

function rel(a: number, b: number, abs: number): number {
  return Math.abs(a - b) / (Math.max(Math.abs(a), Math.abs(b)) + abs / 1);
}

function compare(cases: Case[], out: Float32Array, options: PtBsdfOptions, tol: number): Stats {
  const st: Stats = { evalCompared: 0, evalBad: 0, evalMaxRel: 0, sampleCompared: 0, sampleBad: 0, sampleMaxRel: 0, flagMismatch: 0, nonFinite: 0, failedSamples: 0, basisSkips: 0, firstBad: [] };
  const note = (m: string) => {
    if (st.firstBad.length < 8) st.firstBad.push(m);
  };
  cases.forEach((c, i) => {
    const o = i * OUT_STRIDE;
    const g = Array.from(out.subarray(o, o + OUT_STRIDE));
    if (g.some((x) => !Number.isFinite(x))) {
      st.nonFinite++;
      note(`case ${i} kind ${c.p.kind} non-finite output`);
      return;
    }
    const e = ptBsdfEval(c.p, c.wo, c.wi, options);
    // eval: f (3) and pdf
    for (const [gv, cv, what] of [
      [g[0], e.f[0], 'f.r'],
      [g[1], e.f[1], 'f.g'],
      [g[2], e.f[2], 'f.b'],
      [g[3], e.pdf, 'pdf'],
    ] as [number, number, string][]) {
      st.evalCompared++;
      const r = rel(gv, cv, 1e-3);
      if (r > st.evalMaxRel) st.evalMaxRel = r;
      if (r > tol) {
        st.evalBad++;
        note(`eval ${what} case ${i} kind ${c.p.kind} alpha ${c.p.alpha} wo ${c.wo} wi ${c.wi}: gpu ${gv} cpu ${cv}`);
      }
    }
    const s = ptBsdfSample(c.p, c.wo, c.u, options);
    const gflags = g[11] as number;
    if (gflags === 0) st.failedSamples++;
    if (gflags !== s.flags) {
      st.flagMismatch++;
      return;
    }
    if (s.flags === 0) return;
    // The standard VNDF sampler (pbrt) picks its tangent t1 by `wh.z < 0.99999`: next to that threshold a 1 ulp-scale
    // difference in the GPU's normalize() flips the basis, which is a different (equally valid) u -> wi mapping.
    if (options.vndf !== 'cap' && c.p.kind !== PT_BSDF_DIFFUSE) {
      const wx = c.p.alpha[0] * c.wo[0];
      const wy = c.p.alpha[1] * c.wo[1];
      const whz = Math.abs(c.wo[2]) / Math.hypot(wx, wy, c.wo[2]);
      if (Math.abs(whz - 0.99999) < 4e-6) {
        st.basisSkips++;
        return;
      }
    }
    const pairs: [number, number, string][] = [
      [g[4] as number, s.wi[0], 'wi.x'],
      [g[5] as number, s.wi[1], 'wi.y'],
      [g[6] as number, s.wi[2], 'wi.z'],
      [g[7] as number, s.f[0], 'f.r'],
      [g[8] as number, s.f[1], 'f.g'],
      [g[9] as number, s.f[2], 'f.b'],
      [g[10] as number, s.pdf, 'pdf'],
      [g[12] as number, s.etap, 'etap'],
    ];
    for (const [gv, cv, what] of pairs) {
      st.sampleCompared++;
      const isDir = what.startsWith('wi');
      const r = isDir ? Math.abs(gv - cv) / 1e-3 * tol : rel(gv, cv, 1e-3);
      if (r > st.sampleMaxRel) st.sampleMaxRel = r;
      if (r > tol) {
        st.sampleBad++;
        note(`sample ${what} case ${i} kind ${c.p.kind} alpha ${c.p.alpha} wo ${c.wo} u ${c.u} flags ${s.flags}: gpu ${gv} cpu ${cv}`);
      }
    }
  });
  return st;
}

let adapterSummary: AdapterSummary | null = null;
let deviceErrors: string[] = [];

try {
  const args = getArgs();
  const count = numberArg(args, 'cases', 3000);
  const tol = numberArg(args, 'tol', 5e-3);

  if (!navigator.gpu) throw new Error('navigator.gpu is undefined: no WebGPU in this browser.');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('requestAdapter() returned null: no usable GPU.');
  adapterSummary = summarizeAdapter(adapter);
  const device = await adapter.requestDevice();
  deviceErrors = watchDevice(device);

  const configs: { name: string; options: PtBsdfOptions }[] = [
    { name: 'standard-radiance', options: { vndf: 'standard', transport: 'radiance' } },
    { name: 'cap-radiance', options: { vndf: 'cap', transport: 'radiance' } },
    { name: 'standard-importance', options: { vndf: 'standard', transport: 'importance' } },
  ];
  const metrics: Record<string, number> = {};
  const problems: string[] = [];
  let seed = 20261114;

  for (const config of configs) {
    const cases = makeCases(count, seed++);
    const input = pack(cases);
    const inBuf = device.createBuffer({ size: input.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(inBuf, 0, input);
    const outBytes = count * OUT_STRIDE * 4;
    const outBuf = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const readback = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    const module = device.createShaderModule({ code: shaderSource(config.options) });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((m) => m.type === 'error');
    if (errors.length > 0) throw new Error(`WGSL (${config.name}) failed to compile:\n${errors.map((m) => `  line ${m.lineNum}: ${m.message}`).join('\n')}`);
    const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inBuf } },
        { binding: 1, resource: { buffer: outBuf } },
      ],
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(count / 64));
    pass.end();
    encoder.copyBufferToBuffer(outBuf, 0, readback, 0, outBytes);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(readback.getMappedRange().slice(0));
    readback.unmap();

    const st = compare(cases, out, config.options, tol);
    const key = config.name.replace('-', '_');
    metrics[`${key}_eval_max_rel`] = st.evalMaxRel;
    metrics[`${key}_eval_bad`] = st.evalBad;
    metrics[`${key}_sample_max_rel`] = st.sampleMaxRel;
    metrics[`${key}_sample_bad`] = st.sampleBad;
    metrics[`${key}_flag_mismatch`] = st.flagMismatch;
    metrics[`${key}_failed_samples`] = st.failedSamples;
    metrics[`${key}_basis_skips`] = st.basisSkips;
    if (st.nonFinite > 0) problems.push(`${config.name}: ${st.nonFinite} non-finite outputs`);
    if (st.evalBad > 0) problems.push(`${config.name}: ${st.evalBad}/${st.evalCompared} eval values off by more than ${tol}`);
    if (st.sampleBad > 0) problems.push(`${config.name}: ${st.sampleBad}/${st.sampleCompared} sample values off by more than ${tol}`);
    if (st.flagMismatch > count * 0.005) problems.push(`${config.name}: ${st.flagMismatch} flag mismatches (limit ${count * 0.005})`);
    for (const m of st.firstBad) problems.push(`  ${m}`);
  }

  await drainDevice(device);
  if (problems.length > 0) throw new Error(`GPU BSDF differs from the TS twin:\n${problems.join('\n')}`);
  reportResult(
    { renderer: 'bsdf-check', config: { cases: count, tol, configs: configs.map((c) => c.name) }, metrics, valid: true, notes: { kind: 'correctness only, no timing' } },
    { adapter: adapterSummary, deviceErrors },
  );
} catch (error) {
  reportError(error, { adapter: adapterSummary, deviceErrors });
}
