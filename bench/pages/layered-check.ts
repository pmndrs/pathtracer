/**
 * layered-check (P2.21): the WGSL layered BSDF (`ptLayeredWgsl`, packages/pathtracer/src/bsdf) against its TS twin.
 *
 *  1. albedo layering (openpbr and internal base): deterministic, so eval f/pdf and sample (wi, f, pdf, flags) are compared per
 *     case like bsdf-check (tol 5e-3, inputs rounded to f32 first).
 *  2. stochastic layering: a walk in f32 and one in f64 diverge at the first branch that flips, so a case is compared on its
 *     mean over N shared uniforms: the sampler's estimate of the directional albedo (mean of f |cos| / pdf) and the eval's
 *     (mean of f pi over cosine-distributed wi), GPU against CPU, tolerance 5 combined standard errors + 2%. Two module
 *     variants (maxDepth 100 / exitMis none, and maxDepth 10 / nSamples 2 / exitMis pbrt). A white base with thickness 0 must read
 *     near 1 on the GPU by itself.
 *  2b. coat over a conductor (P2.23, `conductorBase: true`): the same two comparisons (albedo per case, stochastic per-case means) for
 *      `ptLayeredCoatedConductor` stacks (smooth/rough coat over smooth/rough metals, complex IOR relative to the coat).
 *  3. timing, BEST EFFORT (not a quiet-pass number; the load average is recorded by the runner): ns per `ptLayeredEval` for albedo
 *     and stochastic against the plain rough-dielectric `ptBsdfEval`, p10 of interleaved repeats, no readback inside the timed
 *     region (the fence is `onSubmittedWorkDone`).
 *
 * Args: cases=2000 (albedo cases per variant), n=2048 (stochastic samples per case), tol=5e-3, reps=15
 */

import { drainDevice, getArgs, numberArg, reportError, reportResult, summarizeAdapter, watchDevice } from '../runner/page-client.ts';
import { layerCorrectionGroup } from './layer-corr-bind.ts';
import type { AdapterSummary } from '../runner/contract.ts';
import type { PtVec3 } from '../../packages/pathtracer/src/bsdf/bsdf.ts';
import { ptBsdfWgsl } from '../../packages/pathtracer/src/bsdf/bsdf-wgsl.ts';
import { ptInternalReflectance, ptLayeredCoatedConductor, ptLayeredEval, ptLayeredSample, type PtLayeredOptions, type PtLayeredParams } from '../../packages/pathtracer/src/bsdf/layered.ts';
import { ptLayeredWgsl } from '../../packages/pathtracer/src/bsdf/layered-wgsl.ts';

const f32 = Math.fround;
const COND_STRIDE = 24; // conductor-base cases: the 16 above + cEta(3), cK(3), cAlpha, pad
const PARAM_STRIDE = 16; // ior, alpha(2), sw, base(3), sc(3), thickness, Ri, wo(3) = 15 (+1 pad)

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

const WGSL_PARAMS = `
fn loadParams(b: u32) -> PtLayeredParams {
  var p: PtLayeredParams;
  p.ior = inp[b];
  p.alpha = vec2f(inp[b + 1u], inp[b + 2u]);
  p.specularWeight = inp[b + 3u];
  p.baseColor = vec3f(inp[b + 4u], inp[b + 5u], inp[b + 6u]);
  p.specularColor = vec3f(inp[b + 7u], inp[b + 8u], inp[b + 9u]);
  p.thickness = inp[b + 10u];
  p.internalReflectance = inp[b + 11u];
  return p;
}
fn loadWo(b: u32) -> vec3f { return vec3f(inp[b + 12u], inp[b + 13u], inp[b + 14u]); }
`;

const COND_PARAMS = `
fn loadParams(b: u32) -> PtLayeredParams {
  var p: PtLayeredParams;
  p.ior = inp[b];
  p.alpha = vec2f(inp[b + 1u], inp[b + 2u]);
  p.specularWeight = 1.0;
  p.baseColor = vec3f(0.0);
  p.specularColor = vec3f(1.0);
  p.thickness = inp[b + 10u];
  p.internalReflectance = 0.0;
  p.cEta = vec3f(inp[b + 16u], inp[b + 17u], inp[b + 18u]);
  p.cK = vec3f(inp[b + 19u], inp[b + 20u], inp[b + 21u]);
  p.cAlpha = inp[b + 22u];
  p.isCond = true;
  return p;
}
fn loadWo(b: u32) -> vec3f { return vec3f(inp[b + 12u], inp[b + 13u], inp[b + 14u]); }
`;

/** Per-case eval and sample, compare against the CPU. Input: params(16) wi(3) u(3) pad(2) = 24 floats. */
function deterministicShader(options: PtLayeredOptions, cond = false): string {
  const S = cond ? 32 : 24; // floats per case
  const W = cond ? 24 : 16; // offset of wi, then u
  return `${ptBsdfWgsl(options)}
${ptLayeredWgsl(cond ? { ...options, conductorBase: true } : options)}
@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> outp: array<f32>;
${cond ? COND_PARAMS : WGSL_PARAMS}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= arrayLength(&inp) / ${S}u) { return; }
  let b = i * ${S}u;
  let p = loadParams(b);
  let wo = loadWo(b);
  let wi = vec3f(inp[b + ${W}u], inp[b + ${W + 1}u], inp[b + ${W + 2}u]);
  let u = vec3f(inp[b + ${W + 3}u], inp[b + ${W + 4}u], inp[b + ${W + 5}u]);
  let e = ptLayeredEval(p, wo, wi, 1u);
  let s = ptLayeredSample(p, wo, u, 1u);
  let o = i * 16u;
  outp[o] = e.f.x; outp[o + 1u] = e.f.y; outp[o + 2u] = e.f.z; outp[o + 3u] = e.pdf;
  outp[o + 4u] = s.wi.x; outp[o + 5u] = s.wi.y; outp[o + 6u] = s.wi.z;
  outp[o + 7u] = s.f.x; outp[o + 8u] = s.f.y; outp[o + 9u] = s.f.z;
  outp[o + 10u] = s.pdf; outp[o + 11u] = f32(s.flags);
}
`;
}

/** Per-case mean estimates over N shared uniforms. `us` holds N x 5 floats (sample u0 u1 u2, eval u0 u1). */
function estimatorShader(options: PtLayeredOptions, cond = false): string {
  const P = cond ? COND_STRIDE : PARAM_STRIDE;
  return `${ptBsdfWgsl(options)}
${ptLayeredWgsl(cond ? { ...options, conductorBase: true } : options)}
@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> outp: array<f32>;
@group(0) @binding(2) var<storage, read> us: array<f32>;
${cond ? COND_PARAMS : WGSL_PARAMS}
@compute @workgroup_size(32)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= arrayLength(&inp) / ${P}u) { return; }
  let p = loadParams(i * ${P}u);
  let wo = loadWo(i * ${P}u);
  let n = arrayLength(&us) / 5u;
  var sa = vec3f(0.0);
  var ea = vec3f(0.0);
  var sq = vec2f(0.0);
  for (var k = 0u; k < n; k++) {
    let seed = k + 1u;
    let s = ptLayeredSample(p, wo, vec3f(us[k * 5u], us[k * 5u + 1u], us[k * 5u + 2u]), seed);
    if (s.flags != 0u) {
      let v = s.f * (abs(s.wi.z) / s.pdf);
      sa = sa + v;
      sq.x = sq.x + v.x * v.x;
    }
    let u0 = us[k * 5u + 3u];
    let u1 = us[k * 5u + 4u];
    let r = sqrt(u0);
    let ph = 2.0 * PT_PI * u1;
    let wi = vec3f(r * cos(ph), r * sin(ph), sqrt(1.0 - u0));
    let e = ptLayeredEval(p, wo, wi, seed);
    ea = ea + e.f * PT_PI;
    sq.y = sq.y + e.f.x * PT_PI * e.f.x * PT_PI;
  }
  let o = i * 8u;
  outp[o] = sa.x; outp[o + 1u] = sa.y; outp[o + 2u] = sa.z;
  outp[o + 3u] = ea.x; outp[o + 4u] = ea.y; outp[o + 5u] = ea.z;
  outp[o + 6u] = sq.x; outp[o + 7u] = sq.y;
}
`;
}

/** Timing kernel: 16 evals per thread at hashed directions; `which` 0 = layered, 1 = plain rough dielectric ptBsdfEval. */
function timingShader(options: PtLayeredOptions, which: 0 | 1): string {
  return `${ptBsdfWgsl(options)}
${ptLayeredWgsl(options)}
@group(0) @binding(0) var<storage, read_write> outp: array<f32>;
fn hash(v: u32) -> u32 {
  let state = v * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  var p: PtLayeredParams;
  p.ior = 1.5; p.alpha = vec2f(0.09); p.specularWeight = 1.0; p.baseColor = vec3f(0.8); p.specularColor = vec3f(1.0);
  p.thickness = 0.01; p.internalReflectance = 0.596;
  var d: PtBsdfParams;
  d.kind = PT_BSDF_DIELECTRIC; d.ior = 1.5; d.alpha = vec2f(0.09); d.reflectance = vec3f(0.0); d.eta = vec3f(0.0); d.k = vec3f(0.0);
  var h = hash(id.x);
  let a = f32(h >> 8u) * 5.9604645e-8;
  let wo = normalize(vec3f(sin(a * 1.2), 0.0, cos(a * 1.2)));
  var acc = vec3f(0.0);
  for (var k = 0u; k < 16u; k++) {
    h = hash(h);
    let u0 = f32(h >> 8u) * 5.9604645e-8;
    h = hash(h);
    let u1 = f32(h >> 8u) * 5.9604645e-8;
    let r = sqrt(u0);
    let wi = vec3f(r * cos(6.2831853 * u1), r * sin(6.2831853 * u1), sqrt(1.0 - u0));
    ${which === 0 ? 'acc = acc + ptLayeredEval(p, wo, wi, h).f;' : 'acc = acc + ptBsdfEval(d, wo, wi).f;'}
  }
  outp[id.x] = acc.x + acc.y + acc.z;
}
`;
}

interface Case {
  p: PtLayeredParams;
  wo: PtVec3;
  wi: PtVec3;
  u: PtVec3;
}

function unit(rand: () => number): PtVec3 {
  const z = rand();
  const s = Math.sqrt(Math.max(0, 1 - z * z));
  const phi = 2 * Math.PI * rand();
  const v: PtVec3 = [f32(s * Math.cos(phi)), f32(s * Math.sin(phi)), f32(z)];
  const l = Math.hypot(v[0], v[1], v[2]);
  return [f32(v[0] / l), f32(v[1] / l), f32(v[2] / l)];
}
const f3 = (a: readonly number[]): PtVec3 => [f32(a[0] as number), f32(a[1] as number), f32(a[2] as number)];

function makeParams(rand: () => number, ior: number, roughness: number, extra: boolean): PtLayeredParams {
  const a = f32(roughness * roughness);
  return {
    baseColor: [f32(0.2 + 0.8 * rand()), f32(0.2 + 0.8 * rand()), f32(0.2 + 0.8 * rand())],
    ior: f32(ior),
    alpha: [a, a],
    specularWeight: extra ? f32(0.3 + 0.7 * rand()) : 1,
    specularColor: extra ? [1, f32(0.5 + 0.5 * rand()), f32(0.5 + 0.5 * rand())] : [1, 1, 1],
    thickness: f32(rand() < 0.5 ? 0.01 : 0),
    internalReflectance: f32(ptInternalReflectance(ior)),
  };
}

function pack(params: PtLayeredParams, wo: PtVec3): number[] {
  return [params.ior, params.alpha[0], params.alpha[1], params.specularWeight, ...params.baseColor, ...params.specularColor, params.thickness, params.internalReflectance, ...wo, 0];
}

async function runKernel(device: GPUDevice, code: string, label: string, bindings: Float32Array<ArrayBuffer>[], outFloats: number, groups: number): Promise<Float32Array> {
  const module = device.createShaderModule({ code });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((m) => m.type === 'error');
  if (errors.length > 0) {
    const lines = code.split('\n');
    throw new Error(`WGSL (${label}) failed to compile:\n${errors.map((m) => `  line ${m.lineNum}: ${m.message}\n    ${lines[m.lineNum - 1] ?? ''}`).join('\n')}`);
  }
  const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
  const bufs = bindings.map((data) => {
    const b = device.createBuffer({ size: Math.max(16, data.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(b, 0, data);
    return b;
  });
  const outBytes = outFloats * 4;
  const outBuf = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readback = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const entries = [{ binding: 0, resource: { buffer: bufs[0] as GPUBuffer } }, { binding: 1, resource: { buffer: outBuf } }, ...bufs.slice(1).map((b, i) => ({ binding: 2 + i, resource: { buffer: b } }))];
  const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
  const corr = await layerCorrectionGroup(device, pipeline, code);
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  if (corr) pass.setBindGroup(2, corr);
  pass.dispatchWorkgroups(groups);
  pass.end();
  encoder.copyBufferToBuffer(outBuf, 0, readback, 0, outBytes);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(readback.getMappedRange().slice(0));
  readback.unmap();
  return out;
}

function relErr(a: number, b: number, abs: number): number {
  return Math.abs(a - b) / (Math.max(Math.abs(a), Math.abs(b)) + abs);
}

let adapterSummary: AdapterSummary | null = null;
let deviceErrors: string[] = [];

try {
  const args = getArgs();
  const count = numberArg(args, 'cases', 2000);
  const nSamples = numberArg(args, 'n', 2048);
  const tol = numberArg(args, 'tol', 5e-3);
  const reps = numberArg(args, 'reps', 15);

  if (!navigator.gpu) throw new Error('navigator.gpu is undefined: no WebGPU in this browser.');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('requestAdapter() returned null: no usable GPU.');
  adapterSummary = summarizeAdapter(adapter);
  const device = await adapter.requestDevice();
  deviceErrors = watchDevice(device);

  const metrics: Record<string, number> = {};
  const problems: string[] = [];

  // ---- 1. albedo layering, per case ------------------------------------------------------------------------------
  for (const [name, options] of [
    ['albedo_openpbr', { layering: 'albedo' }],
    ['albedo_internal', { layering: 'albedo', albedoBase: 'internal' }],
    ['albedo_internal_corr_lut', { layering: 'albedo', albedoBase: 'internal', albedoCorrection: 'lut' }],
  ] as [string, PtLayeredOptions][]) {
    const rand = rng(name === 'albedo_openpbr' ? 7001 : name === 'albedo_internal' ? 7002 : 7003);
    const cases: Case[] = [];
    const roughs = [0, 0.05, 0.1, 0.25, 0.5, 1];
    for (let i = 0; i < count; i++) {
      const p = makeParams(rand, rand() < 0.5 ? 1.5 : 1.33, roughs[Math.floor(rand() * roughs.length)] as number, rand() < 0.4);
      const w0 = unit(rand);
      const wo: PtVec3 = rand() < 0.15 ? [w0[0], w0[1], f32(-w0[2])] : w0; // 15% below the surface: two-sided
      cases.push({ p, wo, wi: unit(rand), u: [f32(rand()), f32(rand()), f32(rand())] });
    }
    const input = new Float32Array(new ArrayBuffer(count * 24 * 4));
    cases.forEach((c, i) => {
      input.set([...pack(c.p, c.wo).slice(0, 15), 0, ...c.wi, ...c.u, 0, 0], i * 24);
    });
    const out = await runKernel(device, deterministicShader(options), name, [input], count * 16, Math.ceil(count / 64));
    let bad = 0;
    let maxRel = 0;
    let sBad = 0;
    let sMaxRel = 0;
    let flagMismatch = 0;
    const notes: string[] = [];
    cases.forEach((c, i) => {
      const o = i * 16;
      const g = Array.from(out.subarray(o, o + 16));
      if (g.some((x) => !Number.isFinite(x))) {
        bad++;
        notes.push(`case ${i}: non-finite`);
        return;
      }
      const e = ptLayeredEval(c.p, c.wo, c.wi, options);
      for (const [gv, cv] of [[g[0], e.f[0]], [g[1], e.f[1]], [g[2], e.f[2]], [g[3], e.pdf]] as [number, number][]) {
        const r = relErr(gv, cv, 1e-3);
        maxRel = Math.max(maxRel, r);
        if (r > tol) {
          bad++;
          if (notes.length < 6) notes.push(`eval case ${i} r ${c.p.alpha[0]} wo ${c.wo} wi ${c.wi}: gpu ${gv} cpu ${cv}`);
        }
      }
      const s = ptLayeredSample(c.p, c.wo, c.u, options);
      if ((g[11] as number) !== s.flags) {
        flagMismatch++;
        return;
      }
      if (s.flags === 0) return;
      for (const [gv, cv, dir] of [[g[4], s.wi[0], true], [g[5], s.wi[1], true], [g[6], s.wi[2], true], [g[7], s.f[0], false], [g[8], s.f[1], false], [g[9], s.f[2], false], [g[10], s.pdf, false]] as [number, number, boolean][]) {
        const r = dir ? (Math.abs(gv - cv) / 1e-3) * tol : relErr(gv, cv, 1e-3);
        sMaxRel = Math.max(sMaxRel, r);
        if (r > tol) {
          sBad++;
          if (notes.length < 6) notes.push(`sample case ${i} wo ${c.wo} u ${c.u}: gpu ${gv} cpu ${cv}`);
        }
      }
    });
    metrics[`${name}_eval_max_rel`] = maxRel;
    metrics[`${name}_eval_bad`] = bad;
    metrics[`${name}_sample_max_rel`] = sMaxRel;
    metrics[`${name}_sample_bad`] = sBad;
    metrics[`${name}_flag_mismatch`] = flagMismatch;
    if (bad > 0) problems.push(`${name}: ${bad} eval values off by more than ${tol}`);
    if (sBad > 0) problems.push(`${name}: ${sBad} sample values off by more than ${tol}`);
    if (flagMismatch > count * 0.005) problems.push(`${name}: ${flagMismatch} flag mismatches (limit ${count * 0.005})`);
    for (const n of notes) problems.push(`  ${n}`);
  }

  // ---- 2. stochastic layering, per-case means ------------------------------------------------------------------------
  const base = [
    { name: 'plastic', ior: 1.5, r: 0.3, base: [0.8, 0.8, 0.8], sw: 1, sc: [1, 1, 1], th: 0.01 },
    { name: 'white_smooth', ior: 1.5, r: 0, base: [1, 1, 1], sw: 1, sc: [1, 1, 1], th: 0 },
    { name: 'white_r01', ior: 1.5, r: 0.1, base: [1, 1, 1], sw: 1, sc: [1, 1, 1], th: 0 },
    { name: 'tinted', ior: 1.33, r: 0.5, base: [0.8, 0.6, 0.4], sw: 0.5, sc: [1, 0.8, 0.6], th: 0.01 },
    { name: 'rough', ior: 1.5, r: 1, base: [0.8, 0.8, 0.8], sw: 1, sc: [1, 1, 1], th: 0.01 },
    { name: 'dense', ior: 2, r: 0.2, base: [0.5, 0.5, 0.5], sw: 1, sc: [1, 1, 1], th: 0.01 },
  ];
  const angles = [0, 30, 60, 80];
  const est: { label: string; p: PtLayeredParams; wo: PtVec3 }[] = [];
  for (const b of base) {
    for (const a of angles) {
      const t = (a * Math.PI) / 180;
      const alpha = f32(b.r * b.r);
      est.push({
        label: `${b.name}@${a}`,
        p: { baseColor: f3(b.base), ior: f32(b.ior), alpha: [alpha, alpha], specularWeight: f32(b.sw), specularColor: f3(b.sc), thickness: f32(b.th), internalReflectance: f32(ptInternalReflectance(b.ior)) },
        wo: [f32(Math.sin(t)), 0, f32(Math.cos(t))],
      });
    }
  }
  const ur = rng(9100);
  const us = new Float32Array(new ArrayBuffer(nSamples * 5 * 4));
  for (let i = 0; i < nSamples * 5; i++) us[i] = f32(ur());
  const input = new Float32Array(new ArrayBuffer(est.length * PARAM_STRIDE * 4));
  est.forEach((c, i) => input.set(pack(c.p, c.wo), i * PARAM_STRIDE));
  for (const [name, options] of [
    ['stoch_deep', { layering: 'stochastic', maxDepth: 100, nSamples: 1, exitMis: 'none' }],
    ['stoch_pbrt', { layering: 'stochastic', maxDepth: 10, nSamples: 2, exitMis: 'pbrt' }],
  ] as [string, PtLayeredOptions][]) {
    const out = await runKernel(device, estimatorShader(options), name, [input, us], est.length * 8, Math.ceil(est.length / 32));
    let worst = 0;
    let furnaceWorst = 0;
    est.forEach((c, i) => {
      // CPU twin over the same uniforms and seeds
      let sa = 0;
      let ea = 0;
      let sq1 = 0;
      let sq2 = 0;
      for (let k = 0; k < nSamples; k++) {
        const seed = k + 1;
        const s = ptLayeredSample(c.p, c.wo, [us[k * 5] as number, us[k * 5 + 1] as number, us[k * 5 + 2] as number], { ...options, seed });
        if (s.flags !== 0) {
          const v = ((s.f[0] as number) * Math.abs(s.wi[2])) / s.pdf;
          sa += v;
          sq1 += v * v;
        }
        const u0 = us[k * 5 + 3] as number;
        const u1 = us[k * 5 + 4] as number;
        const r = Math.sqrt(u0);
        const ph = 2 * Math.PI * u1;
        const e = ptLayeredEval(c.p, c.wo, [r * Math.cos(ph), r * Math.sin(ph), Math.sqrt(1 - u0)], { ...options, seed });
        ea += (e.f[0] as number) * Math.PI;
        sq2 += ((e.f[0] as number) * Math.PI) ** 2;
      }
      const cpuS = sa / nSamples;
      const cpuE = ea / nSamples;
      const gpuS = (out[i * 8] as number) / nSamples;
      const gpuE = (out[i * 8 + 3] as number) / nSamples;
      const seS = Math.sqrt(Math.max(0, sq1 / nSamples - cpuS * cpuS) / nSamples);
      const seE = Math.sqrt(Math.max(0, sq2 / nSamples - cpuE * cpuE) / nSamples);
      for (const [g, cpu, se, what] of [[gpuS, cpuS, seS, 'sample'], [gpuE, cpuE, seE, 'eval']] as [number, number, number, string][]) {
        const dev = Math.abs(g - cpu);
        const allow = 5 * Math.hypot(se, se) + 0.02 * Math.abs(cpu) + 1e-3;
        worst = Math.max(worst, dev / allow);
        if (!Number.isFinite(g) || dev > allow) problems.push(`${name} ${c.label} ${what}: gpu ${g.toFixed(4)} cpu ${cpu.toFixed(4)} (allow ${allow.toFixed(4)})`);
      }
      if (c.label.startsWith('white_smooth') && name === 'stoch_deep') furnaceWorst = Math.max(furnaceWorst, Math.abs(gpuS - 1));
    });
    metrics[`${name}_worst_dev_over_allow`] = worst;
    if (name === 'stoch_deep') {
      metrics.stoch_gpu_furnace_smooth_max_dev = furnaceWorst;
      if (furnaceWorst > 0.03) problems.push(`GPU white-furnace (smooth, thickness 0): sample albedo off by ${furnaceWorst}`);
    }
  }

  // ---- 2b. coat over a conductor (P2.23) ---------------------------------------------------------------------------------
  const condStacks: { name: string; eta: PtVec3; k: PtVec3; rb: number; ior: number; rc: number; th: number }[] = [
    { name: 'smooth_smooth', eta: [0.2, 0.9, 1.1], k: [3.9, 2.4, 2.2], rb: 0, ior: 1.5, rc: 0, th: 0 },
    { name: 'smooth_rough', eta: [0.2, 0.9, 1.1], k: [3.9, 2.4, 2.2], rb: 0.45, ior: 1.5, rc: 0, th: 0 },
    { name: 'rough_smooth', eta: [0.2, 0.9, 1.1], k: [3.9, 2.4, 2.2], rb: 0, ior: 1.5, rc: 0.3, th: 0.01 },
    { name: 'rough_rough', eta: [1, 1, 1], k: [5, 5, 5], rb: 0.4, ior: 1.6, rc: 0.15, th: 0.01 },
    { name: 'white_rough_rough', eta: [1, 1, 1], k: [1e4, 1e4, 1e4], rb: 0.3, ior: 1.5, rc: 0.1, th: 0 },
  ];
  const condParams = (c: (typeof condStacks)[number]): PtLayeredParams => {
    const q = ptLayeredCoatedConductor(f3(c.eta), f3(c.k), c.rb, c.ior, c.rc, { thickness: c.th });
    const bc = q.baseConductor as NonNullable<PtLayeredParams['baseConductor']>;
    return { ...q, ior: f32(q.ior), alpha: [f32(q.alpha[0]), f32(q.alpha[1])], thickness: f32(q.thickness), baseConductor: { eta: f3(bc.eta), k: f3(bc.k), alpha: f32(bc.alpha) } };
  };
  const packCond = (q: PtLayeredParams, wo: PtVec3): number[] => {
    const bc = q.baseConductor as NonNullable<PtLayeredParams['baseConductor']>;
    return [...pack(q, wo).slice(0, 15), 0, ...bc.eta, ...bc.k, bc.alpha, 0];
  };
  for (const [tag, options] of [['albedo_conductor', { layering: 'albedo' }], ['albedo_conductor_internal', { layering: 'albedo', albedoBase: 'internal' }]] as [string, PtLayeredOptions][]) {
    const rand = rng(7101);
    const n = Math.min(count, 1500);
    const cases: Case[] = [];
    for (let i = 0; i < n; i++) {
      const w0 = unit(rand);
      cases.push({ p: condParams(condStacks[Math.floor(rand() * condStacks.length)] as (typeof condStacks)[number]), wo: rand() < 0.15 ? [w0[0], w0[1], f32(-w0[2])] : w0, wi: unit(rand), u: [f32(rand()), f32(rand()), f32(rand())] });
    }
    const inputC = new Float32Array(new ArrayBuffer(n * 32 * 4));
    cases.forEach((c, i) => inputC.set([...packCond(c.p, c.wo), ...c.wi, ...c.u, 0, 0], i * 32));
    const out = await runKernel(device, deterministicShader(options, true), tag, [inputC], n * 16, Math.ceil(n / 64));
    let bad = 0;
    let maxRel = 0;
    let sMaxRel = 0;
    let flagMismatch = 0;
    cases.forEach((c, i) => {
      const g = Array.from(out.subarray(i * 16, i * 16 + 16));
      if (g.some((x) => !Number.isFinite(x))) {
        bad++;
        return;
      }
      const e = ptLayeredEval(c.p, c.wo, c.wi, options);
      for (const [gv, cv] of [[g[0], e.f[0]], [g[1], e.f[1]], [g[2], e.f[2]], [g[3], e.pdf]] as [number, number][]) {
        const r = relErr(gv, cv, 1e-3);
        maxRel = Math.max(maxRel, r);
        if (r > tol) {
          bad++;
          if (problems.length < 8) problems.push(`${tag} eval case ${i}: gpu ${gv} cpu ${cv}`);
        }
      }
      const s = ptLayeredSample(c.p, c.wo, c.u, options);
      if ((g[11] as number) !== s.flags) {
        flagMismatch++;
        return;
      }
      if (s.flags === 0) return;
      for (const [gv, cv, dir] of [[g[4], s.wi[0], true], [g[5], s.wi[1], true], [g[6], s.wi[2], true], [g[7], s.f[0], false], [g[8], s.f[1], false], [g[9], s.f[2], false], [g[10], s.pdf, false]] as [number, number, boolean][]) {
        const r = dir ? (Math.abs(gv - cv) / 1e-3) * tol : relErr(gv, cv, 1e-3);
        sMaxRel = Math.max(sMaxRel, r);
        if (r > tol) {
          bad++;
          if (problems.length < 8) problems.push(`${tag} sample case ${i}: gpu ${gv} cpu ${cv}`);
        }
      }
    });
    metrics[`${tag}_eval_max_rel`] = maxRel;
    metrics[`${tag}_sample_max_rel`] = sMaxRel;
    metrics[`${tag}_bad`] = bad;
    metrics[`${tag}_flag_mismatch`] = flagMismatch;
    if (bad > 0) problems.push(`${tag}: ${bad} values off by more than ${tol}`);
    if (flagMismatch > n * 0.005) problems.push(`${tag}: ${flagMismatch} flag mismatches`);
  }
  {
    const estC: { label: string; p: PtLayeredParams; wo: PtVec3 }[] = [];
    for (const c of condStacks) {
      for (const a of [0, 30, 60, 80]) {
        const t = (a * Math.PI) / 180;
        estC.push({ label: `${c.name}@${a}`, p: condParams(c), wo: [f32(Math.sin(t)), 0, f32(Math.cos(t))] });
      }
    }
    const inputE = new Float32Array(new ArrayBuffer(estC.length * COND_STRIDE * 4));
    estC.forEach((c, i) => inputE.set(packCond(c.p, c.wo), i * COND_STRIDE));
    for (const [name, options] of [
      ['stoch_cond_deep', { layering: 'stochastic', maxDepth: 100, nSamples: 1, exitMis: 'none' }],
      ['stoch_cond_pbrt', { layering: 'stochastic', maxDepth: 10, nSamples: 2, exitMis: 'pbrt' }],
    ] as [string, PtLayeredOptions][]) {
      const out = await runKernel(device, estimatorShader(options, true), name, [inputE, us], estC.length * 8, Math.ceil(estC.length / 32));
      let worst = 0;
      let furnaceWorst = 0;
      estC.forEach((c, i) => {
        let sa = 0;
        let ea = 0;
        let sq1 = 0;
        let sq2 = 0;
        for (let k = 0; k < nSamples; k++) {
          const seed = k + 1;
          const s = ptLayeredSample(c.p, c.wo, [us[k * 5] as number, us[k * 5 + 1] as number, us[k * 5 + 2] as number], { ...options, seed });
          if (s.flags !== 0) {
            const v = ((s.f[0] as number) * Math.abs(s.wi[2])) / s.pdf;
            sa += v;
            sq1 += v * v;
          }
          const u0 = us[k * 5 + 3] as number;
          const u1 = us[k * 5 + 4] as number;
          const r = Math.sqrt(u0);
          const ph = 2 * Math.PI * u1;
          const e = ptLayeredEval(c.p, c.wo, [r * Math.cos(ph), r * Math.sin(ph), Math.sqrt(1 - u0)], { ...options, seed });
          ea += (e.f[0] as number) * Math.PI;
          sq2 += ((e.f[0] as number) * Math.PI) ** 2;
        }
        const cpuS = sa / nSamples;
        const cpuE = ea / nSamples;
        const gpuS = (out[i * 8] as number) / nSamples;
        const gpuE = (out[i * 8 + 3] as number) / nSamples;
        const seS = Math.sqrt(Math.max(0, sq1 / nSamples - cpuS * cpuS) / nSamples);
        const seE = Math.sqrt(Math.max(0, sq2 / nSamples - cpuE * cpuE) / nSamples);
        for (const [g, cpu, se, what] of [[gpuS, cpuS, seS, 'sample'], [gpuE, cpuE, seE, 'eval']] as [number, number, number, string][]) {
          const dev = Math.abs(g - cpu);
          const allow = 5 * Math.hypot(se, se) + 0.02 * Math.abs(cpu) + 1e-3;
          worst = Math.max(worst, dev / allow);
          if (!Number.isFinite(g) || dev > allow) problems.push(`${name} ${c.label} ${what}: gpu ${g.toFixed(4)} cpu ${cpu.toFixed(4)} (allow ${allow.toFixed(4)})`);
        }
        if (c.label.startsWith('white_rough_rough') && name === 'stoch_cond_deep') furnaceWorst = Math.max(furnaceWorst, 1 - gpuS);
      });
      metrics[`${name}_worst_dev_over_allow`] = worst;
      if (name === 'stoch_cond_deep') {
        metrics.stoch_cond_gpu_furnace_loss_max = furnaceWorst;
        if (furnaceWorst > 0.1) problems.push(`GPU white-conductor furnace: sample albedo loses ${furnaceWorst}`);
      }
    }
  }

  // ---- 3. timing (best effort) -------------------------------------------------------------------------------------
  const threads = 1 << 20;
  const timingVariants: [string, string][] = [
    ['plain_rough_dielectric', timingShader({ layering: 'albedo' }, 1)],
    ['albedo', timingShader({ layering: 'albedo' }, 0)],
    ['stochastic_d10_n1_pbrt', timingShader({ layering: 'stochastic', maxDepth: 10, nSamples: 1, exitMis: 'pbrt' }, 0)],
    ['stochastic_d10_n1_none', timingShader({ layering: 'stochastic', maxDepth: 10, nSamples: 1, exitMis: 'none' }, 0)],
  ];
  const runners: { name: string; run: () => Promise<void> }[] = [];
  for (const [name, code] of timingVariants) {
    const module = device.createShaderModule({ code });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((m) => m.type === 'error');
    if (errors.length > 0) throw new Error(`WGSL timing (${name}) failed to compile:\n${errors.map((m) => `  line ${m.lineNum}: ${m.message}`).join('\n')}`);
    const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    const buf = device.createBuffer({ size: threads * 4, usage: GPUBufferUsage.STORAGE });
    const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: buf } }] });
    const corr = await layerCorrectionGroup(device, pipeline, code);
    runners.push({
      name,
      run: async () => {
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, group);
        if (corr) pass.setBindGroup(2, corr);
        pass.dispatchWorkgroups(threads / 64);
        pass.end();
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();
      },
    });
  }
  const times: Record<string, number[]> = {};
  for (const r of runners) {
    times[r.name] = [];
    await r.run(); // warm-up
  }
  for (let rep = 0; rep < reps; rep++) {
    for (const r of runners) {
      const t0 = performance.now();
      await r.run();
      (times[r.name] as number[]).push(performance.now() - t0);
    }
  }
  const evals = threads * 16;
  const p10 = (a: number[]): number => [...a].sort((x, y) => x - y)[Math.floor(a.length * 0.1)] as number;
  for (const r of runners) {
    metrics[`ns_per_eval_${r.name}`] = (p10(times[r.name] as number[]) * 1e6) / evals;
    metrics[`ms_min_${r.name}`] = Math.min(...(times[r.name] as number[]));
    metrics[`ms_max_${r.name}`] = Math.max(...(times[r.name] as number[]));
  }

  await drainDevice(device);
  if (problems.length > 0) throw new Error(`GPU layered BSDF differs from the TS twin:\n${problems.slice(0, 40).join('\n')}`);
  reportResult(
    {
      renderer: 'layered-check',
      config: { cases: count, n: nSamples, tol, reps },
      metrics,
      valid: true,
      notes: { kind: 'correctness; the ns_per_eval_* timings are BEST EFFORT (page-clock, p10 of interleaved repeats, not a quiet-pass number)' },
    },
    { adapter: adapterSummary, deviceErrors },
  );
} catch (error) {
  reportError(error, { adapter: adapterSummary, deviceErrors });
}
