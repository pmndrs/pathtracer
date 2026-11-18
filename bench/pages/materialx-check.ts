/**
 * materialx-check (P5): the generated MaterialX WGSL (approach b) against its TypeScript twin. Correctness only, no timing.
 *
 * For each codegen scene (`mtlx-uv-codegen`, `mtlx-proc-codegen`): the scene copy is built as the renderer builds it (textures included), the generated
 * programs are compiled into a compute kernel next to `rtSceneWgsl`, and `ptMtlxApply` / `ptMtlxNormal` run on random shading points
 * (uv in [0, 1]^2, position in [-1, 1]^3, random unit normal and tangent). The TypeScript oracle is `evalMtlxParams` / the normalmap arguments evaluated
 * with `evalIr` over the same synthetic images (`imageSampler`), sampled bilinearly in f64. Compared per point: base colour, metalness, roughness,
 * and the tangent-space normal texel.
 *
 * Tolerances (absolute): procedural graph 2e-3 (f32 `sin` of arguments up to ~25 rad and a smoothstep of slope 10 amplify rounding); image graph 1.5e-2
 * (the hardware filter has 8-bit weights, texture-check uses 1e-2 for a single lookup and the graph chains a place2d rotation and a smoothstep).
 *
 * Args: points=2000  tolProc=2e-3  tolImage=1.5e-2
 */
import { buildSceneCopy, rtSceneCopyBindGroupEntries, rtSceneCopyLayoutEntries, rtSceneWgsl, uploadSceneCopy } from '@pmndrs/raytracer';
import { coerceValue, constantContext, evalIr, evalMtlxParams, imageSampler, type IrOp, type MtlxTsContext } from '@pmndrs/pathtracer';
import type { AdapterSummary } from '../runner/contract.ts';
import { drainDevice, getArgs, numberArg, reportError, reportResult, summarizeAdapter, watchDevice } from '../runner/page-client.ts';
import { collectMaterialXPrograms, getMtlxEntry } from '../scenes/materialx.ts';
import { syntheticImage } from '../scenes/materialx-images.ts';
import { getScene } from '../scenes/index.ts';
import { sceneDescToCopyInput } from '../scenes/to-scene-copy.ts';

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

const L_IN = 16; // uv(2) pad(2) pos(3) pad(1) nrm(3) pad(1) tg(3) pad(1)
const L_OUT = 16; // base colour(3) metalness | roughness, normal valid, pad, pad | normal texel(3) pad | emission(3) luminance

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

async function checkScene(device: GPUDevice, sceneName: string, count: number, tol: number): Promise<void> {
  const scene = getScene(sceneName);
  const input = sceneDescToCopyInput(scene);
  const copy = buildSceneCopy(input);
  const buffers = uploadSceneCopy(device, copy, { label: sceneName });
  const kernel = collectMaterialXPrograms(scene, input.materialNames);
  if (!kernel) throw new Error(`${sceneName}: no MaterialX program`);
  const index = input.materialNames.indexOf('subject');
  const entry = getMtlxEntry((scene.materials.subject as { materialx?: string }).materialx!);
  const mtl = entry.mtl;
  metrics[`${sceneName}_wgsl_bytes`] = kernel.programs.wgsl.length;

  const code = `${rtSceneWgsl({ group: 0, firstBinding: 0, textures: copy.textureLayout })}
${kernel.programs.wgsl}
@group(1) @binding(0) var<storage, read> inp: array<f32>;
@group(1) @binding(1) var<storage, read_write> outp: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= ${count}u) { return; }
  let b = i * ${L_IN}u;
  let uv = vec2f(inp[b], inp[b + 1u]);
  let pos = vec3f(inp[b + 4u], inp[b + 5u], inp[b + 6u]);
  let nrm = vec3f(inp[b + 8u], inp[b + 9u], inp[b + 10u]);
  let tg = vec3f(inp[b + 12u], inp[b + 13u], inp[b + 14u]);
  let m = ptMtlxApply(rtMaterial(${index}u), ${index}u, uv, pos, nrm, tg, 0.0);
  let o = i * ${L_OUT}u;
  outp[o] = m.baseColor.x; outp[o + 1u] = m.baseColor.y; outp[o + 2u] = m.baseColor.z; outp[o + 3u] = m.metalness;
  outp[o + 4u] = m.specularRoughness;
${kernel.programs.normal ? `  let nt = ptMtlxNormal(${index}u, uv, pos, nrm, tg, 0.0);
  outp[o + 5u] = nt.w; outp[o + 8u] = nt.x; outp[o + 9u] = nt.y; outp[o + 10u] = nt.z;` : ''}
  outp[o + 12u] = m.emissionColor.x; outp[o + 13u] = m.emissionColor.y; outp[o + 14u] = m.emissionColor.z; outp[o + 15u] = m.emissionLuminance;
}`;
  const module = device.createShaderModule({ label: `materialx-check ${sceneName}`, code });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((m) => m.type === 'error');
  if (errors.length > 0) throw new Error(`WGSL (${sceneName}) failed to compile:\n${errors.map((m) => `  line ${m.lineNum}: ${m.message}`).join('\n')}\n--- generated ---\n${kernel.programs.wgsl}`);
  const sceneLayout = device.createBindGroupLayout({ entries: rtSceneCopyLayoutEntries(0, copy.textureLayout) });
  const ioLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  });
  const pipeline = await device.createComputePipelineAsync({ layout: device.createPipelineLayout({ bindGroupLayouts: [sceneLayout, ioLayout] }), compute: { module, entryPoint: 'main' } });

  const rand = rng(5150);
  const unit = (): [number, number, number] => {
    const z = 2 * rand() - 1;
    const a = 2 * Math.PI * rand();
    const r = Math.sqrt(1 - z * z);
    return [Math.fround(r * Math.cos(a)), Math.fround(r * Math.sin(a)), Math.fround(z)];
  };
  const input32 = new Float32Array(count * L_IN);
  const points: MtlxTsContext[] = [];
  const sample = imageSampler(syntheticImage);
  for (let i = 0; i < count; i++) {
    const uv: [number, number] = [Math.fround(rand()), Math.fround(rand())];
    const pos: [number, number, number] = [Math.fround(rand() * 2 - 1), Math.fround(rand() * 2 - 1), Math.fround(rand() * 2 - 1)];
    const nrm = unit();
    const tg = unit();
    input32.set([uv[0], uv[1], 0, 0, ...pos, 0, ...nrm, 0, ...tg, 0], i * L_IN);
    points.push({ ...constantContext(sample), uv, position: pos, normal: nrm, tangent: tg });
  }
  const inBuf = device.createBuffer({ size: input32.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(inBuf, 0, input32);
  const outBuf = device.createBuffer({ size: count * L_OUT * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, device.createBindGroup({ layout: sceneLayout, entries: rtSceneCopyBindGroupEntries(buffers) }));
  pass.setBindGroup(1, device.createBindGroup({ layout: ioLayout, entries: [{ binding: 0, resource: { buffer: inBuf } }, { binding: 1, resource: { buffer: outBuf } }] }));
  pass.dispatchWorkgroups(Math.ceil(count / 64));
  pass.end();
  device.queue.submit([encoder.finish()]);
  const got = new Float32Array(await readBack(device, outBuf, count * L_OUT * 4));

  let bad = 0;
  let worst = 0;
  const normal = mtl.normal?.ir.kind === 'op' ? (mtl.normal.ir as IrOp) : undefined;
  points.forEach((ctx, i) => {
    const p = evalMtlxParams(mtl, ctx);
    const want: [string, number, number][] = [];
    const base = p.baseColor as number[] | undefined;
    if (base) base.forEach((v, k) => want.push([`baseColor.${k}`, got[i * L_OUT + k]!, v]));
    if (typeof p.baseMetalness === 'number' && 'baseMetalness' in mtl.varying) want.push(['metalness', got[i * L_OUT + 3]!, p.baseMetalness]);
    if (typeof p.specularRoughness === 'number' && 'specularRoughness' in mtl.varying) want.push(['roughness', got[i * L_OUT + 4]!, p.specularRoughness]);
    if (normal && normal.args.in) {
      const w = coerceValue(evalIr(normal.args.in.ir, ctx), normal.args.in.from, normal.args.in.to);
      const s = normal.args.scale ? coerceValue(evalIr(normal.args.scale.ir, ctx), normal.args.scale.from, normal.args.scale.to)[0] : 1;
      let v: [number, number, number] = w[0] === 0 && w[1] === 0 && w[2] === 0 ? [0, 0, 1] : [w[0] * 2 - 1, w[1] * 2 - 1, w[2] * 2 - 1];
      v = [v[0] * s, v[1] * s, v[2]];
      const l = Math.hypot(v[0], v[1], v[2]);
      [0, 1, 2].forEach((k) => want.push([`normalTexel.${k}`, got[i * L_OUT + 8 + k]!, (v[k]! / l) * 0.5 + 0.5]));
      if (got[i * L_OUT + 5] !== 1) want.push(['normalValid', got[i * L_OUT + 5]!, 1]);
    }
    for (const [what, g, c] of want) {
      const err = Math.abs(g - c);
      worst = Math.max(worst, err);
      if (!(err <= tol)) {
        bad++;
        note(`${sceneName} point ${i} uv (${ctx.uv[0].toFixed(3)}, ${ctx.uv[1].toFixed(3)}) ${what}: gpu ${g} cpu ${c}`);
      }
    }
  });
  metrics[`${sceneName}_mismatch`] = bad;
  metrics[`${sceneName}_max_abs_error`] = worst;
  metrics[`${sceneName}_points`] = count;
}

try {
  const args = getArgs();
  const count = numberArg(args, 'points', 2000);
  if (!navigator.gpu) throw new Error('navigator.gpu is undefined: no WebGPU in this browser.');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('requestAdapter() returned null: no usable GPU.');
  adapterSummary = summarizeAdapter(adapter);
  const device = await adapter.requestDevice();
  deviceErrors = watchDevice(device);
  await checkScene(device, 'mtlx-proc-codegen', count, numberArg(args, 'tolProc', 2e-3));
  await drainDevice(device);
  await checkScene(device, 'mtlx-uv-codegen', count, numberArg(args, 'tolImage', 1.5e-2));
  await drainDevice(device);
  const total = Object.entries(metrics).filter(([k]) => k.endsWith('_mismatch')).reduce((sum, [, v]) => sum + v, 0);
  reportResult(
    { scene: 'materialx-check', renderer: 'ours', config: { points: count }, valid: total === 0 && deviceErrors.length === 0, metrics: { ...metrics, mismatches: total }, notes: { problems } },
    { adapter: adapterSummary, deviceErrors },
  );
  if (total > 0) throw new Error(`materialx-check: ${total} mismatches. First:\n${problems.join('\n')}`);
} catch (error) {
  reportError(error, { adapter: adapterSummary, deviceErrors });
}
