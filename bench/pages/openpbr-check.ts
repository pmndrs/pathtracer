/**
 * openpbr-check (P2.21 / P2.22 / P2.30): the GPU's OpenPBR-mode WGSL against its TypeScript twins. Correctness only, no timing.
 *
 *  1. Adapter: `ptBsdfFromMaterial(rtMaterial(id), u)` (all features on, `albedoBase: 'internal'`) against the TS `ptBsdfFromMaterial(material, options, u)`
 *     for a table that hits every branch (metal with and without eta/k, fractional metalness at three u, thick and thin glass, tinted thin, coat,
 *     specular weight/colour, ior 1, black base, black) and the medium's `ptMediumSigma` against `ptMediumSigma`.
 *  2. Thin-walled dielectric: `ptBsdfSample` of `PT_BSDF_THIN` (wi, f, pdf, flags, etap) against the TS twin on random directions and u.
 *  3. Layered dispatch: `ptEval` / `ptSample` through `ptLayeredOf` (albedo layering, deterministic) against `ptLayeredEval/Sample` of `ptLayeredFromBsdf`.
 *
 * Args: cases=4000  tol=2e-3 (layered, f32 table + sampling)  ptol=2e-4 (adapter, thin)
 */
import { buildSceneCopy, rtSceneWgsl, uploadSceneCopy, rtSceneCopyBindGroupEntries, rtSceneCopyLayoutEntries, type RtMaterialInput } from '@pmndrs/raytracer';
import {
  PT_BSDF_LAYERED, PT_BSDF_THIN, ptBsdfAdapterWgsl, ptFrameFromTangent, ptBsdfFromMaterial, ptBsdfSample, ptBsdfWgsl, ptLayeredDispatchWgsl, ptLayeredEval, ptLayeredFromBsdf, ptLayeredSample, ptLayeredWgsl, ptMediumSigma,
  type PtMaterialOptions, type PtBsdfParams,
} from '@pmndrs/pathtracer';
import type { AdapterSummary } from '../runner/contract.ts';
import { drainDevice, getArgs, numberArg, reportError, reportResult, summarizeAdapter, watchDevice } from '../runner/page-client.ts';
import { sceneDescToCopyInput } from '../scenes/to-scene-copy.ts';
import { quadLight } from '../scenes/tier0/index.ts';

let adapterSummary: AdapterSummary | null = null;
let deviceErrors: string[] = [];
const problems: string[] = [];
const metrics: Record<string, number> = {};
const note = (m: string) => {
  if (problems.length < 20) problems.push(m);
};

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

async function compile(device: GPUDevice, code: string, label: string): Promise<GPUShaderModule> {
  const module = device.createShaderModule({ label, code });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((m) => m.type === 'error');
  if (errors.length > 0) throw new Error(`WGSL (${label}) failed to compile:\n${errors.map((m) => `  line ${m.lineNum}: ${m.message}`).join('\n')}`);
  return module;
}

async function readBack(device: GPUDevice, buffer: GPUBuffer, bytes: number): Promise<ArrayBuffer> {
  const staging = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, staging, 0, bytes);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const copy = staging.getMappedRange().slice(0);
  staging.unmap();
  staging.destroy();
  return copy;
}

const storage = (device: GPUDevice, data: ArrayBufferView<ArrayBuffer>, extra = 0): GPUBuffer => {
  const buffer = device.createBuffer({ size: Math.max(16, data.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | extra });
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
};

const rel = (a: number, b: number, abs: number): number => Math.abs(a - b) / (Math.max(Math.abs(a), Math.abs(b)) + abs);

const OPTIONS: PtMaterialOptions = { features: { mix: true, thin: true, layered: true, medium: true, coatedMetal: true, aniso: true }, layered: { layering: 'albedo', albedoBase: 'internal' }, layerThickness: 1e-6, transmissionTint: true };

const TABLE: RtMaterialInput[] = [
  { conductorEtaK: true, conductorEta: [0.2, 0.9, 1.1], conductorK: [3.9, 2.4, 2.1], metalness: 1, specularRoughness: 0.35 },
  { metalness: 1, baseColor: [0.9, 0.5, 0.1], baseWeight: 0.8, specularRoughness: 0.5 },
  { metalness: 0.3, baseColor: [0.6, 0.6, 0.6], specularRoughness: 0.2 },
  { metalness: 0.7, baseColor: [0.2, 0.4, 0.6], specularRoughness: 0.9 },
  { transmissionWeight: 1, ior: 1.33, specularRoughness: 0.2, transmissionColor: [0.2, 0.5, 0.9], transmissionDepth: 0.5 },
  { transmissionWeight: 1, ior: 1.76, specularRoughness: 0, thinWalled: true, transmissionColor: [0.3, 0.6, 0.9] },
  { transmissionWeight: 1, ior: 2.42, specularRoughness: 0 },
  { baseColor: [0.5, 0.02, 0.01], specularRoughness: 0.44, ior: 1.5, specularWeight: 0.25, specularColor: [1, 0.5, 0.8] },
  { baseColor: [0.5, 0.02, 0.01], specularRoughness: 0.44, coatWeight: 1, coatRoughness: 0.03, coatIor: 1.5 },
  { metalness: 1, baseColor: [0.9, 0.6, 0.25], specularRoughness: 0.35, coatWeight: 1, coatRoughness: 0.03, coatIor: 1.5 },
  { conductorEtaK: true, conductorEta: [0.2, 0.9, 1.1], conductorK: [3.9, 2.4, 2.1], metalness: 1, specularRoughness: 0.5, coatWeight: 1, coatRoughness: 0.2, coatIor: 1.7 },
  { metalness: 0.5, baseColor: [0.8, 0.8, 0.8], specularRoughness: 0.3, coatWeight: 0.8, coatRoughness: 0.1, coatIor: 1.5 },
  { baseColor: [0.5, 0.5, 0.5], ior: 1, specularWeight: 1 },
  { baseColor: [0, 0, 0], ior: 1.76, specularRoughness: 0 },
  { baseColor: [0.4, 0.4, 0.4], specularWeight: 0 },
  { baseColor: [0, 0, 0], baseWeight: 0, specularWeight: 0 },
  { metalness: 0.5, baseColor: [0.3, 0.3, 0.3] },
  { baseColor: [0.9, 0.9, 0.9], ior: 2.2, specularRoughness: 0.7, specularWeight: 0.5 },
  // P2.26 anisotropic metals (OpenPBR alpha pair), with and without eta/k, and a mix whose metal end is anisotropic
  { conductorEtaK: true, conductorEta: [0.2, 0.9, 1.1], conductorK: [3.9, 2.4, 2.1], metalness: 1, specularRoughness: 0.35, specularAnisotropy: 0.6 },
  { metalness: 1, baseColor: [1, 1, 1], specularRoughness: 0.84, specularAnisotropy: 1 },
  { metalness: 0.5, baseColor: [0.8, 0.5, 0.2], specularRoughness: 0.2, specularAnisotropy: 0.3 },
];

const U_VALUES = [0, 0.29, 0.5, 0.71, 0.999];

async function checkAdapter(device: GPUDevice, ptol: number): Promise<void> {
  const base = sceneDescToCopyInput(quadLight);
  const input = { meshes: base.meshes, instances: base.instances, materials: [...base.materials, ...TABLE], sun: null, environment: null };
  const copy = buildSceneCopy(input);
  const buffers = uploadSceneCopy(device, copy, { label: 'openpbr-check' });
  const count = copy.materials.length;
  const code = `${rtSceneWgsl({ group: 0, firstBinding: 0, sampling: true })}${ptBsdfWgsl({ thin: true, aniso: true })}${ptLayeredWgsl({ ...OPTIONS.layered, conductorBase: true })}${ptLayeredDispatchWgsl(true)}${ptBsdfAdapterWgsl(OPTIONS)}
@group(1) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn adapter(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= ${count * U_VALUES.length}u) { return; }
  let mat = i / ${U_VALUES.length}u;
  let ui = i % ${U_VALUES.length}u;
  var us = array<f32, ${U_VALUES.length}>(${U_VALUES.map((u) => `${u}`.includes('.') ? `${u}` : `${u}.0`).join(', ')});
  let m = rtMaterial(mat);
  let p = ptBsdfFromMaterial(m, us[ui]);
  let s = ptMediumSigma(m);
  let o = i * 20u;
  out[o] = f32(p.kind); out[o + 1u] = p.ior; out[o + 2u] = p.alpha.x; out[o + 3u] = p.alpha.y;
  out[o + 4u] = p.reflectance.x; out[o + 5u] = p.reflectance.y; out[o + 6u] = p.reflectance.z;
  out[o + 7u] = p.eta.x; out[o + 8u] = p.eta.y; out[o + 9u] = p.eta.z;
  out[o + 10u] = p.k.x; out[o + 11u] = p.k.y; out[o + 12u] = p.k.z;
  out[o + 13u] = s.x; out[o + 14u] = s.y; out[o + 15u] = s.z;
}`;
  const module = await compile(device, code, 'openpbr adapter');
  const sceneLayout = device.createBindGroupLayout({ entries: rtSceneCopyLayoutEntries() });
  const ioLayout = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }] });
  const pipeline = await device.createComputePipelineAsync({ layout: device.createPipelineLayout({ bindGroupLayouts: [sceneLayout, ioLayout] }), compute: { module, entryPoint: 'adapter' } });
  const total = count * U_VALUES.length;
  const outBuf = storage(device, new Float32Array(total * 20), GPUBufferUsage.COPY_SRC);
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, device.createBindGroup({ layout: sceneLayout, entries: rtSceneCopyBindGroupEntries(buffers) }));
  pass.setBindGroup(1, device.createBindGroup({ layout: ioLayout, entries: [{ binding: 0, resource: { buffer: outBuf } }] }));
  pass.dispatchWorkgroups(Math.ceil(total / 64));
  pass.end();
  device.queue.submit([encoder.finish()]);
  const got = new Float32Array(await readBack(device, outBuf, total * 20 * 4));
  let bad = 0;
  const kinds = new Set<number>();
  for (let id = 0; id < count; id++) {
    const m = copy.materials[id]!;
    for (let ui = 0; ui < U_VALUES.length; ui++) {
      const p = ptBsdfFromMaterial(m, OPTIONS, U_VALUES[ui] as number);
      kinds.add(p.kind);
      const g = got.subarray((id * U_VALUES.length + ui) * 20);
      const sigma = ptMediumSigma(m);
      const c = [p.kind, p.ior, p.alpha[0], p.alpha[1], ...p.reflectance, ...p.eta, ...p.k, ...sigma];
      c.forEach((cv, k) => {
        const idx = k; // contiguous: kind, ior, alpha(2), reflectance(3), eta(3), k(3), sigma(3)
        if (rel(g[idx] as number, cv, 1e-4) > ptol) {
          bad++;
          note(`adapter material ${id} u ${U_VALUES[ui]} field ${k}: gpu ${g[idx]} cpu ${cv}`);
        }
      });
    }
  }
  metrics.adapter_mismatch = bad;
  metrics.adapter_cases = total;
  metrics.adapter_kinds = kinds.size;
}

/** P2.26: `ptFrameFromTangent` (WGSL) against the TS twin on random normals, tangents (incl. degenerate) and rotations. */
async function checkFrame(device: GPUDevice, count: number): Promise<void> {
  const rand = rng(2626);
  const n = Math.min(count, 2000);
  const input = new Float32Array(n * 8);
  const cases: { nrm: Vec3; tan: [number, number, number, number]; rot: number }[] = [];
  for (let i = 0; i < n; i++) {
    const nrm = unit([rand() * 2 - 1, rand() * 2 - 1, rand() * 2 - 1]).map(Math.fround) as Vec3;
    const tan: [number, number, number, number] = i % 50 === 7 ? [0, 0, 0, 1] : i % 50 === 9 ? [nrm[0], nrm[1], nrm[2], 1] : [rand() * 2 - 1, rand() * 2 - 1, rand() * 2 - 1, rand() < 0.5 ? -1 : 1];
    const rot = i % 3 === 0 ? 0 : (rand() * 2 - 1) * 2 * Math.PI;
    cases.push({ nrm, tan: tan.map(Math.fround) as [number, number, number, number], rot: Math.fround(rot) });
    input.set([nrm[0], nrm[1], nrm[2], 0, ...cases[i]!.tan.slice(0, 3), 0], i * 8);
    input[i * 8 + 3] = cases[i]!.rot;
    input[i * 8 + 7] = cases[i]!.tan[3];
  }
  const code = `${ptBsdfWgsl({ aniso: true })}
@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= ${n}u) { return; }
  let b = i * 8u;
  let f = ptFrameFromTangent(vec3f(inp[b], inp[b + 1u], inp[b + 2u]), vec4f(inp[b + 4u], inp[b + 5u], inp[b + 6u], inp[b + 7u]), inp[b + 3u]);
  let o = i * 9u;
  out[o] = f.t.x; out[o + 1u] = f.t.y; out[o + 2u] = f.t.z;
  out[o + 3u] = f.b.x; out[o + 4u] = f.b.y; out[o + 5u] = f.b.z;
  out[o + 6u] = f.n.x; out[o + 7u] = f.n.y; out[o + 8u] = f.n.z;
}`;
  const module = await compile(device, code, 'aniso frame');
  const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
  const inBuf = storage(device, input);
  const outBuf = storage(device, new Float32Array(n * 9), GPUBufferUsage.COPY_SRC);
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: inBuf } }, { binding: 1, resource: { buffer: outBuf } }] }));
  pass.dispatchWorkgroups(Math.ceil(n / 64));
  pass.end();
  device.queue.submit([encoder.finish()]);
  const got = new Float32Array(await readBack(device, outBuf, n * 9 * 4));
  let bad = 0;
  cases.forEach((c, i) => {
    const f = ptFrameFromTangent(c.nrm, c.tan, c.rot);
    [...f.t, ...f.b, ...f.n].forEach((cv, k) => {
      if (Math.abs((got[i * 9 + k] as number) - cv) > 2e-5) {
        bad++;
        note(`aniso frame case ${i} component ${k}: gpu ${got[i * 9 + k]} cpu ${cv}`);
      }
    });
  });
  metrics.aniso_frame_mismatch = bad;
  metrics.aniso_frame_cases = n;
}

type Vec3 = [number, number, number];
const unit = (v: Vec3): Vec3 => {
  const l = Math.hypot(...v);
  return [v[0] / l, v[1] / l, v[2] / l];
};

async function checkLobes(device: GPUDevice, count: number, ptol: number, tol: number, layered: NonNullable<PtMaterialOptions['layered']> = OPTIONS.layered!, tag = ''): Promise<void> {
  const stochastic = layered.layering === 'stochastic';
  const rand = rng(4242);
  const cases: { p: PtBsdfParams; wo: Vec3; wi: Vec3; u: Vec3 }[] = [];
  for (let i = 0; i < count; i++) {
    const thin = i % 2 === 0;
    const ior = 1.1 + rand() * 1.5;
    const rough = rand() * 0.9;
    const p: PtBsdfParams = thin
      ? { kind: PT_BSDF_THIN, reflectance: [rand(), rand(), rand()], eta: [0, 0, 0], k: [0, 0, 0], alpha: [0, 0], ior }
      : { kind: PT_BSDF_LAYERED, reflectance: [rand(), rand(), rand()], eta: [0.3 + 0.7 * rand(), 0.3 + 0.7 * rand(), 0.3 + 0.7 * rand()], k: [1e-6, 0.4, 0], alpha: [Math.max(rough * rough, 2e-3), Math.max(rough * rough, 2e-3)], ior };
    const wo = unit([rand() * 2 - 1, rand() * 2 - 1, (rand() * 1.9 + 0.05) * (i % 10 === 3 ? -1 : 1)]);
    const wi = unit([rand() * 2 - 1, rand() * 2 - 1, (rand() * 1.9 + 0.05) * (i % 10 === 3 ? -1 : 1)]);
    const u: Vec3 = [rand(), rand(), rand()];
    cases.push({ p: { ...p, reflectance: p.reflectance.map(Math.fround) as Vec3, eta: p.eta.map(Math.fround) as Vec3, alpha: [Math.fround(p.alpha[0]), Math.fround(p.alpha[1])], ior: Math.fround(ior), k: p.k.map(Math.fround) as Vec3 }, wo: wo.map(Math.fround) as Vec3, wi: wi.map(Math.fround) as Vec3, u: u.map(Math.fround) as Vec3 });
  }
  // per case, 18 floats: kind, ior, alpha(2), reflectance(3), eta(3), k0, k1, wo(3), u(3)
  const IN = 18;
  const inp = new Float32Array(count * IN);
  cases.forEach((c, i) => inp.set([c.p.kind, c.p.ior, c.p.alpha[0], c.p.alpha[1], ...c.p.reflectance, ...c.p.eta, c.p.k[0], c.p.k[1], ...c.wo, ...c.u], i * IN));
  const OUT = 24;
  const code = `${ptBsdfWgsl({ thin: true })}${ptLayeredWgsl(layered)}${ptLayeredDispatchWgsl()}
@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> outp: array<f32>;
@compute @workgroup_size(64)
fn lobes(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= ${count}u) { return; }
  let b = i * ${IN}u;
  var p: PtBsdfParams;
  p.kind = u32(inp[b]); p.ior = inp[b + 1u]; p.alpha = vec2f(inp[b + 2u], inp[b + 3u]);
  p.reflectance = vec3f(inp[b + 4u], inp[b + 5u], inp[b + 6u]);
  p.eta = vec3f(inp[b + 7u], inp[b + 8u], inp[b + 9u]);
  p.k = vec3f(inp[b + 10u], inp[b + 11u], 0.0);
  let wo = vec3f(inp[b + 12u], inp[b + 13u], inp[b + 14u]);
  let u = vec3f(inp[b + 15u], inp[b + 16u], inp[b + 17u]);
  ptLayerSeed = i * 2654435761u + 12345u;
  let s = ptSample(p, wo, u);
  let o = i * ${OUT}u;
  outp[o] = s.wi.x; outp[o + 1u] = s.wi.y; outp[o + 2u] = s.wi.z;
  outp[o + 3u] = s.f.x; outp[o + 4u] = s.f.y; outp[o + 5u] = s.f.z;
  outp[o + 6u] = s.pdf; outp[o + 7u] = f32(s.flags); outp[o + 8u] = s.etap;
  // evaluate towards a second direction: the sampled one's mirror hemisphere partner (the TS side uses the same wi)
  let wi = vec3f(-wo.y, wo.x, abs(wo.z) * 0.9 + 0.1);
  let e = ptEval(p, wo, normalize(wi));
  outp[o + 9u] = e.f.x; outp[o + 10u] = e.f.y; outp[o + 11u] = e.f.z; outp[o + 12u] = e.pdf;
  if (p.kind == PT_BSDF_LAYERED) { outp[o + 13u] = ptLayeredPdf(ptLayeredOf(p), wo, normalize(wi), ptLayerSeed); }
}`;
  const module = await compile(device, code, 'openpbr lobes');
  const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'lobes' } });
  const inBuf = storage(device, inp);
  const outBuf = storage(device, new Float32Array(count * OUT), GPUBufferUsage.COPY_SRC);
  const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: inBuf } }, { binding: 1, resource: { buffer: outBuf } }] });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, group);
  pass.dispatchWorkgroups(Math.ceil(count / 64));
  pass.end();
  device.queue.submit([encoder.finish()]);
  const got = new Float32Array(await readBack(device, outBuf, count * OUT * 4));
  const bad = { thin: 0, layered: 0, evalLayered: 0 };
  // Non-finite outputs of any lobe (a stochastic walk has no CPU twin to compare with on a per-case basis: its f32 and f64 branches diverge; the means are checked by the furnace scenes).
  let nonFinite = 0;
  for (let i = 0; i < count; i++) {
    const g = got.subarray(i * OUT, i * OUT + 14);
    if (g.some((v) => !Number.isFinite(v))) {
      nonFinite++;
      note(`${tag}non-finite output, case ${i} kind ${cases[i]!.p.kind} ior ${cases[i]!.p.ior} alpha ${cases[i]!.p.alpha[0]}: ${Array.from(g).join(', ')}`);
    }
  }
  metrics[`${tag}non_finite_mismatch`] = nonFinite;
  const worst = { thin: 0, layered: 0, evalLayered: 0 };
  let skipped = 0;
  cases.forEach((c, i) => {
    const g = got.subarray(i * OUT, i * OUT + OUT);
    if (stochastic) return;
    const thin = c.p.kind === PT_BSDF_THIN;
    const s = thin ? ptBsdfSample(c.p, c.wo, c.u) : ptLayeredSample(ptLayeredFromBsdf(c.p), c.wo, c.u, OPTIONS.layered);
    const gFlags = g[7] as number;
    if ((gFlags === 0) !== (s.flags === 0) || (gFlags !== 0 && (gFlags & 7) !== (s.flags & 7))) {
      // A branch decided within f32 noise of a threshold (u vs R, the VNDF basis): counted, not compared.
      skipped++;
      return;
    }
    if (s.flags !== 0) {
      const pairs: [number, number][] = [[g[0] as number, s.wi[0]], [g[1] as number, s.wi[1]], [g[2] as number, s.wi[2]], [g[3] as number, s.f[0]], [g[4] as number, s.f[1]], [g[5] as number, s.f[2]], [g[6] as number, s.pdf], [g[8] as number, s.etap]];
      const key = thin ? 'thin' : 'layered';
      const t = thin ? ptol : tol;
      for (const [a, b] of pairs) {
        const r = rel(a, b, 1e-4);
        worst[key] = Math.max(worst[key], r);
        if (r > t) {
          bad[key]++;
          note(`${key} sample case ${i}: gpu ${a} cpu ${b}`);
          break;
        }
      }
    }
    if (!thin) {
      const wi = unit([-c.wo[1], c.wo[0], Math.abs(c.wo[2]) * 0.9 + 0.1]).map(Math.fround) as Vec3;
      const e = ptLayeredEval(ptLayeredFromBsdf(c.p), c.wo, wi, OPTIONS.layered);
      for (const [a, b] of [[g[9] as number, e.f[0]], [g[10] as number, e.f[1]], [g[11] as number, e.f[2]], [g[12] as number, e.pdf]] as [number, number][]) {
        const r = rel(a, b, 1e-4);
        worst.evalLayered = Math.max(worst.evalLayered, r);
        if (r > tol) {
          bad.evalLayered++;
          note(`layered eval case ${i}: gpu ${a} cpu ${b}`);
          break;
        }
      }
    }
  });
  if (stochastic) return;
  metrics.thin_sample_mismatch = bad.thin;
  metrics.layered_sample_mismatch = bad.layered;
  metrics.layered_eval_mismatch = bad.evalLayered;
  metrics.thin_sample_max_rel = worst.thin;
  metrics.layered_sample_max_rel = worst.layered;
  metrics.layered_eval_max_rel = worst.evalLayered;
  metrics.lobe_cases = count;
  metrics.lobe_flag_ties_skipped = skipped;
}

try {
  const args = getArgs();
  const count = numberArg(args, 'cases', 4000);
  const tol = numberArg(args, 'tol', 2e-3);
  const ptol = numberArg(args, 'ptol', 2e-4);
  if (!navigator.gpu) throw new Error('navigator.gpu is undefined: no WebGPU in this browser.');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('requestAdapter() returned null: no usable GPU.');
  adapterSummary = summarizeAdapter(adapter);
  const device = await adapter.requestDevice();
  deviceErrors = watchDevice(device);
  await checkAdapter(device, ptol);
  await checkFrame(device, count);
  await checkLobes(device, count, ptol, tol);
  await checkLobes(device, count, ptol, tol, { layering: 'stochastic', maxDepth: 64, nSamples: 1, exitMis: 'pbrt' }, 'stochastic_');
  await drainDevice(device);
  const total = Object.entries(metrics).filter(([k]) => k.endsWith('_mismatch')).reduce((sum, [, v]) => sum + v, 0);
  reportResult(
    { scene: 'openpbr-check', renderer: 'ours', config: { cases: count, tol, ptol }, valid: total === 0 && deviceErrors.length === 0, metrics: { ...metrics, mismatches: total }, notes: { problems } },
    { adapter: adapterSummary, deviceErrors },
  );
  if (total > 0) throw new Error(`openpbr-check: ${total} mismatches. First:\n${problems.join('\n')}`);
} catch (error) {
  reportError(error, { adapter: adapterSummary, deviceErrors });
}
