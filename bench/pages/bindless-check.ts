/**
 * P1.4c2 bindless experiment: texture arrays (one binding per size bucket + a `switch`) vs the resource table
 * (`chromium-experimental-sampling-resource-table`, `getResource` on the slot's bucket) behind the same slot API.
 *
 *   node bench/runner/run.ts --page bindless-check --chrome <Chrome with the feature> [--args check=lookup,render timing=1 scenes=tex-checker,tex-mask spp=4]
 *
 * Chrome 160 Canary on macOS exposes the feature but Metal fails a COMPUTE pass with a table (device lost, V296): on a Mac run it on SwiftShader
 * (`PT_CHROME_FLAGS="--use-webgpu-adapter=swiftshader --enable-unsafe-swiftshader"`, `--allow-software`; correctness only) and time it on Windows / Linux.
 *
 * check=lookup: the library's own lookup (`rtTexturesWgsl`) on synthetic texture sets, every lane draws (slot, uv, lod) from a hash:
 *   - `fit` (48 textures, 6 size x colour-space groups): arrays, table and table1 (one texture per table entry, `maxLayers: 1`) must be BIT-identical
 *     (same texels, same samplers); lod-0 lookups are also compared with the CPU twin (`sampleSceneTexture`, 1e-2: 8-bit filter weights).
 *   - `many` (64 textures, 20 groups): arrays must merge to 8 buckets (textures resampled up), the table keeps 20: bytes and the
 *     arrays-vs-table difference are reported (not an error).
 *   - `timing=1`: ABBA rounds of N back-to-back dispatches (timestamps around the batch), `coherent` (one slot per 64 lanes) and `divergent`
 *     (one slot per lane) access; ratios table / arrays. Provisional unless the machine is quiet (CLAUDE.md).
 * check=fragment: the lookups of check=lookup from a fragment shader, the table given in the render pass DESCRIPTOR (render passes have no
 *   `setResourceTable`). Metal takes tables in render passes, so this is the real-GPU check of divergent indices on a Mac:
 *   `--args check=fragment` without SwiftShader.
 * check=render: the `pt-megakernel` adapter renders each scene with `pt_tex_binding` arrays and table; images must be per-pixel identical.
 */
import {
  detectCaps, packTextureSlot, requestDeviceForCaps, rtSceneCopyBindGroupEntries, rtSceneCopyLayoutEntries, rtSetResourceTable,
  rtTextureTableEnable, rtTexturesWgsl, rtUsesResourceTable, sampleSceneTexture, textureLayoutOf, uploadSceneTextures, buildSceneTextures,
  type Caps, type SceneCopyBuffers, type SceneTextures, type TextureInput, type TextureOptions,
} from '@pmndrs/raytracer';
import { getArgs, numberArg, reportError, reportResult, summarizeAdapter, uploadImage, watchDevice } from '../runner/page-client.ts';
import type { AdapterSummary } from '../runner/contract.ts';
import { getScene } from '../scenes/index.ts';
import { PtMegakernelAdapter } from '../adapters/pt-megakernel/index.ts';


// ---------------------------------------------------------------------------------------------------------------
// Synthetic texture sets
// ---------------------------------------------------------------------------------------------------------------

/** Deterministic texel pattern: id-dependent colour ramps plus a checker, so a wrong texture, layer, wrap or level reads differently. */
function pattern(id: number, w: number, h: number, colorSpace: TextureInput['colorSpace'], wrap: TextureInput['wrap']): TextureInput {
  const data = new Uint8Array(w * h * 4);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const o = (y * w + x) * 4;
      const c = ((x >> 2) + (y >> 2)) & 1;
      data[o] = (id * 37 + x * 5) & 255;
      data[o + 1] = (id * 91 + y * 3) & 255;
      data[o + 2] = c ? 230 : (id * 13) & 255;
      data[o + 3] = 255 - ((id * 7) & 63);
    }
  }
  return { width: w, height: h, data, colorSpace, wrap, label: `t${id}` };
}

function fitSet(): TextureInput[] {
  const sizes = [64, 128, 256];
  const out: TextureInput[] = [];
  for (let i = 0; i < 48; i++) out.push(pattern(i, sizes[i % 3]!, sizes[i % 3]!, i % 2 ? 'srgb' : 'linear', 'repeat'));
  return out;
}

function manySet(): TextureInput[] {
  const sizes = [32, 64, 128, 256, 512];
  const wraps: Array<TextureInput['wrap']> = ['repeat', 'clamp'];
  const out: TextureInput[] = [];
  for (let i = 0; i < 64; i++) {
    const g = i % 20;
    const s = sizes[g % 5]!;
    out.push(pattern(i, s, s, Math.floor(g / 5) % 2 ? 'srgb' : 'linear', wraps[Math.floor(g / 10)]));
  }
  return out;
}

// ---------------------------------------------------------------------------------------------------------------
// Lookup kernel: the library's lookup over a scene table that holds only the slot records
// ---------------------------------------------------------------------------------------------------------------

const WG = 64;

/**
 * How a `table` variant indexes the table (diagnostics for divergent indices): `plain` = the library's text; `fixedSampler` = every lookup uses
 * bucket 0's sampler (isolates the texture index; only valid when all samplers are equal); `waterfall` = loop over the distinct buckets of the
 * subgroup (`subgroupBroadcastFirst`, lanes leave the loop when served), so each `getResource` sees an index uniform over the ACTIVE lanes;
 * `waterfallMin` = every lane stays in the loop and the index is `subgroupMin` over the lanes still waiting: uniform over the whole subgroup.
 */
type TableLookup = 'plain' | 'fixedSampler' | 'waterfall' | 'waterfallMin';

function lookupKernel(caps: Caps, textures: SceneTextures, lookups: number, mode: TableLookup = 'plain'): string {
  const layout = textureLayoutOf(textures);
  let lib = rtTexturesWgsl({ group: 1, firstBinding: 0, layout, packing: 'split' });
  const n = layout.buckets.length;
  const plain = `    return textureSampleLevel(getResource<texture_2d_array<f32>>(r.x), getResource<sampler>(${n}u + r.x), uv, i32(r.y), lod);`;
  if (mode !== 'plain') {
    if (!lib.includes(plain)) throw new Error('bindless-check: the library lookup line changed; update the diagnostics');
    lib = lib.replace(plain, mode === 'waterfallMin'
      ? `    var o = vec4f(0.0);
    var pending = true;
    loop {
        let u = subgroupMin(select(0xffffffffu, r.x, pending));
        if (u == 0xffffffffu) {
            break;
        }
        let t = textureSampleLevel(getResource<texture_2d_array<f32>>(u), getResource<sampler>(${n}u + u), uv, i32(r.y), lod);
        if (pending && u == r.x) {
            o = t;
            pending = false;
        }
    }
    return o;`
      : mode === 'fixedSampler'
      ? `    return textureSampleLevel(getResource<texture_2d_array<f32>>(r.x), getResource<sampler>(${n}u), uv, i32(r.y), lod);`
      : `    var o = vec4f(0.0);
    loop {
        let u = subgroupBroadcastFirst(r.x);
        if (u == r.x) {
            o = textureSampleLevel(getResource<texture_2d_array<f32>>(u), getResource<sampler>(${n}u + u), uv, i32(r.y), lod);
            break;
        }
    }
    return o;`);
  }
  // Keep only what the lookup needs (rtApplyMaterialTextures / normal map refer to RtMaterial, not declared here).
  const cut = lib.indexOf('fn rtTexLodOf');
  const enable = rtUsesResourceTable(layout) ? `${rtTextureTableEnable(caps.featureNames.samplingResourceTable)}\n` : '';
  const wf = mode === 'waterfall' || mode === 'waterfallMin' ? 'enable subgroups;\ndiagnostic(off, subgroup_uniformity);\n' : '';
  return `${enable}${wf}
struct Params { count: u32, slots: u32, divergent: u32, levels: u32 }
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> outv: array<vec4f>;
@group(1) @binding(0) var<storage, read> rtSceneTable: array<vec4u>;
${lib.slice(0, cut)}
fn pcg(v: u32) -> u32 {
    let s = v * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}
fn unit(v: u32) -> f32 { return f32(v >> 8u) * (1.0 / 16777216.0); }

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) g: vec3u, @builtin(workgroup_id) wg: vec3u) {
    let i = g.x;
    if (i >= P.count) { return; }
    // Access mode 2 ('holes'): a quarter of the lanes leave first (a megakernel's finished paths), the others are divergent.
    if (P.divergent == 2u && (pcg(i ^ 0x9e3779b9u) & 3u) == 0u) {
        outv[i] = vec4f(-1.0);
        return;
    }
    var acc = vec4f(0.0);
    for (var k = 0u; k < ${lookups}u; k++) {
        let key = select(wg.x * 7919u, i, P.divergent != 0u) + k * 104729u;
        let slot = pcg(key) % P.slots;
        let h1 = pcg(i * 3u + k * 65537u + 1u);
        let h2 = pcg(h1);
        let h3 = pcg(h2);
        let uv = vec2f(unit(h1), unit(h2)) * 3.0 - 1.0;
        let lod = select(unit(h3) * f32(P.levels), 0.0, P.levels == 0u);
        acc += rtTexSample(slot, uv, lod);
    }
    outv[i] = acc;
}
`;
}

/** The slot records at the table base the WGSL reads (`rtSceneTable[3].z`). */
function slotTable(textures: SceneTextures): Uint32Array {
  const base = 4;
  const words = new Uint32Array((base + textures.slots.length) * 4);
  words[3 * 4 + 2] = base;
  textures.slots.forEach((slot, i) => packTextureSlot(slot, words, (base + i) * 4));
  return words;
}

interface LookupVariant {
  name: string;
  mode: TableLookup;
  textures: SceneTextures;
  buffers: SceneCopyBuffers;
  pipelines: Map<number, GPUComputePipeline>;
  layouts: [GPUBindGroupLayout, GPUBindGroupLayout];
  groups: [GPUBindGroup, GPUBindGroup];
  kernelBytes: number;
}

async function makeVariant(device: GPUDevice, caps: Caps, name: string, inputs: TextureInput[], options: TextureOptions, outBuffer: GPUBuffer, params: GPUBuffer, mode: TableLookup = 'plain'): Promise<LookupVariant> {
  const textures = buildSceneTextures(inputs, options);
  const layout = textureLayoutOf(textures);
  const tableWords = slotTable(textures);
  const table = device.createBuffer({ label: `${name} slots`, size: tableWords.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(table, 0, tableWords as Uint32Array<ArrayBuffer>);
  const data = device.createBuffer({ label: `${name} data`, size: 16, usage: GPUBufferUsage.STORAGE });
  const tex = uploadSceneTextures(device, textures, name);
  const buffers = { packing: 'split', table, data, storageBytes: 0, layout: { normals: false, uvs: false, positions: false }, byteLength: textures.byteLength, textures: tex, destroy() {} } as unknown as SceneCopyBuffers;
  const io = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }, { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }] });
  const scene = device.createBindGroupLayout({ entries: rtSceneCopyLayoutEntries(0, layout, 'split') });
  return {
    name,
    mode,
    textures,
    buffers,
    pipelines: new Map(),
    layouts: [io, scene],
    groups: [
      device.createBindGroup({ layout: io, entries: [{ binding: 0, resource: { buffer: params } }, { binding: 1, resource: { buffer: outBuffer } }] }),
      device.createBindGroup({ layout: scene, entries: rtSceneCopyBindGroupEntries(buffers) }),
    ],
    kernelBytes: 0,
  };
}

async function pipelineOf(device: GPUDevice, caps: Caps, v: LookupVariant, lookups: number): Promise<GPUComputePipeline> {
  let p = v.pipelines.get(lookups);
  if (p) return p;
  const code = lookupKernel(caps, v.textures, lookups, v.mode);
  v.kernelBytes = code.length;
  const module = device.createShaderModule({ label: `${v.name} x${lookups}`, code });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((m) => m.type === 'error');
  if (errors.length) throw new Error(`${v.name}: ${errors.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join('; ')}\n${code.split('\n').slice(0, 12).join('\n')}`);
  const layout = device.createPipelineLayout({ bindGroupLayouts: v.layouts, ...(rtUsesResourceTable(textureLayoutOf(v.textures)) ? { usesResourceTable: true } : {}) } as GPUPipelineLayoutDescriptor);
  p = await device.createComputePipelineAsync({ layout, compute: { module, entryPoint: 'main' } });
  v.pipelines.set(lookups, p);
  return p;
}

function encodeRuns(encoder: GPUCommandEncoder, v: LookupVariant, pipeline: GPUComputePipeline, count: number, repeats: number, timestampWrites?: GPUComputePassTimestampWrites): void {
  const pass = encoder.beginComputePass(timestampWrites ? { timestampWrites } : {});
  if (v.buffers.textures?.table) rtSetResourceTable(pass, v.buffers.textures.table);
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, v.groups[0]);
  pass.setBindGroup(1, v.groups[1]);
  for (let r = 0; r < repeats; r++) pass.dispatchWorkgroups(Math.ceil(count / WG));
  pass.end();
}

async function readVec4(device: GPUDevice, src: GPUBuffer, count: number): Promise<Float32Array> {
  const rb = device.createBuffer({ size: count * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const e = device.createCommandEncoder();
  e.copyBufferToBuffer(src, 0, rb, 0, count * 16);
  device.queue.submit([e.finish()]);
  await rb.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(rb.getMappedRange().slice(0));
  rb.unmap();
  rb.destroy();
  return out;
}

/** One lookup per lane with these params, read back. Correctness only: never inside a timed region. */
type Access = 'coh' | 'div' | 'holes';
const ACCESS_CODE: Record<Access, number> = { coh: 0, div: 1, holes: 2 };

async function lookupOnce(device: GPUDevice, caps: Caps, v: LookupVariant, params: GPUBuffer, out: GPUBuffer, count: number, access: Access, levels: number): Promise<Float32Array> {
  device.queue.writeBuffer(params, 0, new Uint32Array([count, v.textures.slots.length, ACCESS_CODE[access], levels]));
  const pipeline = await pipelineOf(device, caps, v, 1);
  const e = device.createCommandEncoder();
  encodeRuns(e, v, pipeline, count, 1);
  device.queue.submit([e.finish()]);
  return readVec4(device, out, count);
}

function compare(a: Float32Array, b: Float32Array): { differing: number; maxAbs: number } {
  let differing = 0;
  let maxAbs = 0;
  for (let i = 0; i < a.length; i += 4) {
    let d = 0;
    for (let c = 0; c < 4; c++) d = Math.max(d, Math.abs((a[i + c] as number) - (b[i + c] as number)));
    if (d > 0) differing++;
    maxAbs = Math.max(maxAbs, d);
  }
  return { differing, maxAbs };
}

/** The CPU twin of lane i's lod-0 lookup (the same hash in TypeScript). */
function pcg(v: number): number {
  const s = (Math.imul(v >>> 0, 747796405) + 2891336453) >>> 0;
  const w = Math.imul(((s >>> ((s >>> 28) + 4)) ^ s) >>> 0, 277803737) >>> 0;
  return ((w >>> 22) ^ w) >>> 0;
}
const unit = (v: number) => (v >>> 8) * (1 / 16777216);
function twinLane(textures: SceneTextures, i: number, access: Access): [number, number, number, number] {
  if (access === 'holes' && (pcg((i ^ 0x9e3779b9) >>> 0) & 3) === 0) return [-1, -1, -1, -1];
  const key = access !== 'coh' ? i : Math.imul(Math.floor(i / WG), 7919) >>> 0;
  const slot = pcg(key) % textures.slots.length;
  const h1 = pcg((Math.imul(i, 3) + 1) >>> 0);
  const h2 = pcg(h1);
  const f = (x: number) => Math.fround(Math.fround(Math.fround(x) * 3) - 1);
  return sampleSceneTexture(textures, slot, f(unit(h1)), f(unit(h2)), 0);
}

async function lookupCheck(device: GPUDevice, caps: Caps, args: Record<string, string>, metrics: Record<string, number>, notes: Record<string, unknown>, issues: string[], findings: string[]): Promise<void> {
  const count = numberArg(args, 'lanes', 1 << 16);
  const out = device.createBuffer({ label: 'lookup out', size: count * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const params = device.createBuffer({ label: 'lookup params', size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const timing = numberArg(args, 'timing', 0) !== 0;
  for (const set of ['fit', 'many'] as const) {
    const inputs = set === 'fit' ? fitSet() : manySet();
    // CPU mips so every level exists on both sides and the twin can be read at level 0 (GPU mips are the same bits on both bindings, but the twin then has level 0 only).
    const base: TextureOptions = { mips: 'cpu', maxSize: 512 };
    const variants = [
      await makeVariant(device, caps, `${set}-arrays`, inputs, base, out, params),
      await makeVariant(device, caps, `${set}-table`, inputs, { ...base, binding: 'table' }, out, params),
      ...(set === 'fit' ? [await makeVariant(device, caps, `${set}-table1`, inputs, { ...base, binding: 'table', maxLayers: 1 }, out, params)] : []),
      ...(set === 'fit' ? [await makeVariant(device, caps, `${set}-tablefixsmp`, inputs, { ...base, binding: 'table' }, out, params, 'fixedSampler')] : []),
      ...(caps.subgroups ? [await makeVariant(device, caps, `${set}-tablewf`, inputs, { ...base, binding: 'table' }, out, params, 'waterfall')] : []),
      ...(caps.subgroups && set === 'fit' ? [await makeVariant(device, caps, `${set}-table1wf`, inputs, { ...base, binding: 'table', maxLayers: 1 }, out, params, 'waterfall')] : []),
      ...(caps.subgroups ? [await makeVariant(device, caps, `${set}-tablewfmin`, inputs, { ...base, binding: 'table' }, out, params, 'waterfallMin')] : []),
      ...(caps.subgroups && set === 'fit' ? [await makeVariant(device, caps, `${set}-table1wfmin`, inputs, { ...base, binding: 'table', maxLayers: 1 }, out, params, 'waterfallMin')] : []),
    ];
    const [arrays] = variants as [LookupVariant, ...LookupVariant[]];
    for (const v of variants) {
      metrics[`${v.name}_buckets`] = v.textures.buckets.length;
      metrics[`${v.name}_mib`] = v.textures.byteLength / 2 ** 20;
    }
    const levels = Math.min(...arrays.textures.buckets.map((b) => b.mipLevels));
    for (const access of ['coh', 'div', 'holes'] as const) {
      const divergent = access !== 'coh';
      // What must hold on any device: arrays always; every table variant under coherent access; the uniform waterfall when every lane takes part.
      // A failure elsewhere is a finding about the implementation's non-uniform indexing (the proposal allows any index; V297).
      const mustHold = (v: LookupVariant) => v === arrays || access === 'coh' || (access === 'div' && v.mode === 'waterfallMin');
      const tag = `${set}_${access}`;
      const ref = await lookupOnce(device, caps, arrays, params, out, count, access, levels);
      let nonDefault = 0;
      for (let i = 0; i < ref.length; i += 4) if (!(ref[i] === 1 && ref[i + 1] === 1 && ref[i + 2] === 1) && !(ref[i] === 0 && ref[i + 1] === 0 && ref[i + 2] === 0)) nonDefault++;
      metrics[`${tag}_arrays_nondefault_frac`] = nonDefault / count;
      for (const v of variants.slice(1)) {
        const got = await lookupOnce(device, caps, v, params, out, count, access, levels);
        const c = compare(ref, got);
        metrics[`${tag}_${v.name.split('-')[1]}_vs_arrays_differing`] = c.differing;
        metrics[`${tag}_${v.name.split('-')[1]}_vs_arrays_max_abs`] = c.maxAbs;
        // Coherent access and the uniform waterfall must equal arrays (`fit`: no merge); a divergent index that breaks is a FINDING about the
        // implementation (the proposal allows non-uniform indices), recorded and not a failure: SwiftShader reads lane 0's index (V297).
        // fixedSampler uses bucket 0's lodMaxClamp for every bucket: it can only be judged at lod 0 (the twin check below).
        if (set === 'fit' && c.differing && v.mode !== 'fixedSampler') (mustHold(v) ? issues : findings).push(`${tag}: ${v.name} differs from arrays in ${c.differing} lanes (max ${c.maxAbs})`);
      }
      // CPU twin at level 0 (each variant's own buckets: arrays merged in `many`, the tables did not).
      for (const v of variants) {
        const lod0 = await lookupOnce(device, caps, v, params, out, count, access, 0);
        let maxTwin = 0;
        let bad = 0;
        const n = Math.min(count, 4096);
        for (let i = 0; i < n; i++) {
          const t = twinLane(v.textures, i, access);
          let d = 0;
          for (let c = 0; c < 4; c++) d = Math.max(d, Math.abs((lod0[i * 4 + c] as number) - (t[c] as number)));
          if (d > 1e-2) bad++;
          maxTwin = Math.max(maxTwin, d);
        }
        const key = v.name.split('-')[1];
        metrics[`${tag}_${key}_vs_twin_max_abs`] = maxTwin;
        metrics[`${tag}_${key}_vs_twin_bad_frac`] = bad / n;
        if (divergent && maxTwin > 1e-2) metrics[`${tag}_${key}_broken`] = 1;
        if (maxTwin > 1e-2) (mustHold(v) ? issues : findings).push(`${tag}: ${v.name} lod-0 lookups differ from the CPU twin by ${maxTwin} in ${bad} of ${n} lanes (> 1e-2)`);
      }
    }
    for (const v of variants) notes[`${v.name}_kernel_bytes`] = lookupKernel(caps, v.textures, 1).length;

    if (timing) {
      const lookups = numberArg(args, 'lookups', 16);
      const repeats = numberArg(args, 'repeats', 64);
      const rounds = numberArg(args, 'rounds', 4);
      const qs = device.createQuerySet({ type: 'timestamp', count: 2 });
      const qb = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
      const time = async (v: LookupVariant, access: Access): Promise<number> => {
        device.queue.writeBuffer(params, 0, new Uint32Array([count, v.textures.slots.length, ACCESS_CODE[access], levels]));
        const pipeline = await pipelineOf(device, caps, v, lookups);
        const e = device.createCommandEncoder();
        encodeRuns(e, v, pipeline, count, repeats, { querySet: qs, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 });
        e.resolveQuerySet(qs, 0, 2, qb, 0);
        device.queue.submit([e.finish()]);
        await device.queue.onSubmittedWorkDone();
        const t = new BigUint64Array((await readVec4(device, qb, 1)).buffer);
        return Number((t[1] as bigint) - (t[0] as bigint)) / 1e6 / repeats;
      };
      for (const access of ['coh', 'div'] as const) {
        const tag = `${set}_${access}`;
        const samples = new Map<string, number[]>(variants.map((v) => [v.name, []]));
        for (const v of variants) await time(v, access); // warm-up (compile, clocks)
        for (let r = 0; r < rounds; r++) {
          const order = r % 2 ? [...variants].reverse() : variants;
          for (const v of order) samples.get(v.name)!.push(await time(v, access));
        }
        const med = (xs: number[]) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)] as number;
        for (const v of variants) {
          const xs = samples.get(v.name)!;
          metrics[`${tag}_${v.name.split('-')[1]}_ms`] = med(xs);
          metrics[`${tag}_${v.name.split('-')[1]}_ms_min`] = Math.min(...xs);
          metrics[`${tag}_${v.name.split('-')[1]}_ms_max`] = Math.max(...xs);
        }
        for (const v of variants.slice(1)) metrics[`${tag}_${v.name.split('-')[1]}_over_arrays`] = med(samples.get(v.name)!) / med(samples.get(arrays.name)!);
      }
      notes[`${set}_timing`] = { lanes: count, lookups, repeats, rounds, note: 'ms per dispatch, median of ABBA rounds; one machine, provisional unless quiet' };
    }
    for (const v of variants) v.buffers.textures?.textures.forEach((t) => t.destroy());
  }
}

// ---------------------------------------------------------------------------------------------------------------
// check=fragment: the same lookups from a fragment shader, the table passed in the RENDER PASS DESCRIPTOR
// (`beginRenderPass({ resourceTable })`, gpuweb proposals/bindless.md; render passes have no `setResourceTable`). Metal takes tables in
// render passes but not in compute passes (V296), so this is the real-GPU check of divergent indices on a Mac.
// ---------------------------------------------------------------------------------------------------------------

const FRAG_W = 128;

function fragmentKernel(caps: Caps, textures: SceneTextures): string {
  const layout = textureLayoutOf(textures);
  const lib = rtTexturesWgsl({ group: 1, firstBinding: 0, layout, packing: 'split' });
  const cut = lib.indexOf('fn rtTexLodOf');
  const enable = rtUsesResourceTable(layout) ? `${rtTextureTableEnable(caps.featureNames.samplingResourceTable)}\n` : '';
  return `${enable}
struct Params { count: u32, slots: u32, divergent: u32, levels: u32 }
@group(0) @binding(0) var<uniform> P: Params;
@group(1) @binding(0) var<storage, read> rtSceneTable: array<vec4u>;
${lib.slice(0, cut)}
fn pcg(v: u32) -> u32 {
    let s = v * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}
fn unit(v: u32) -> f32 { return f32(v >> 8u) * (1.0 / 16777216.0); }

@vertex fn vs(@builtin(vertex_index) v: u32) -> @builtin(position) vec4f {
    let p = vec2f(f32((v << 1u) & 2u), f32(v & 2u)) * 2.0 - 1.0;
    return vec4f(p, 0.0, 1.0);
}

@fragment fn fs(@builtin(position) q: vec4f) -> @location(0) vec4f {
    let i = u32(q.y) * ${FRAG_W}u + u32(q.x);
    if (P.divergent == 2u && (pcg(i ^ 0x9e3779b9u) & 3u) == 0u) {
        return vec4f(-1.0);
    }
    let key = select((i / ${WG}u) * 7919u, i, P.divergent != 0u);
    let slot = pcg(key) % P.slots;
    let h1 = pcg(i * 3u + 1u);
    let h2 = pcg(h1);
    let h3 = pcg(h2);
    let uv = vec2f(unit(h1), unit(h2)) * 3.0 - 1.0;
    let lod = select(unit(h3) * f32(P.levels), 0.0, P.levels == 0u);
    return rtTexSample(slot, uv, lod);
}
`;
}

async function fragmentCheck(device: GPUDevice, caps: Caps, args: Record<string, string>, metrics: Record<string, number>, notes: Record<string, unknown>, issues: string[], findings: string[]): Promise<void> {
  const height = Math.max(1, Math.floor(numberArg(args, 'lanes', 1 << 16) / FRAG_W));
  const count = FRAG_W * height;
  const FRAGMENT = GPUShaderStage.FRAGMENT;
  const params = device.createBuffer({ label: 'fragment params', size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const target = device.createTexture({ size: [FRAG_W, height], format: 'rgba32float', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
  const readback = device.createBuffer({ size: count * 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  const run = async (textures: SceneTextures, tex: ReturnType<typeof uploadSceneTextures>, table: GPUBuffer, access: Access, levels: number): Promise<Float32Array> => {
    const layout = textureLayoutOf(textures);
    const io = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: FRAGMENT, buffer: { type: 'uniform' } }] });
    const scene = device.createBindGroupLayout({ entries: rtSceneCopyLayoutEntries(0, layout, 'split').map((e) => ({ ...e, visibility: FRAGMENT })) });
    const module = device.createShaderModule({ code: fragmentKernel(caps, textures) });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((m) => m.type === 'error');
    if (errors.length) throw new Error(`fragment kernel: ${errors.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join('; ')}`);
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [io, scene], ...(rtUsesResourceTable(layout) ? { usesResourceTable: true } : {}) } as GPUPipelineLayoutDescriptor);
    const pipeline = await device.createRenderPipelineAsync({ layout: pipelineLayout, vertex: { module, entryPoint: 'vs' }, fragment: { module, entryPoint: 'fs', targets: [{ format: 'rgba32float' }] } });
    const data = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE });
    const buffers = { packing: 'split', table, data, textures: tex } as unknown as SceneCopyBuffers;
    device.queue.writeBuffer(params, 0, new Uint32Array([count, textures.slots.length, ACCESS_CODE[access], levels]));
    const e = device.createCommandEncoder();
    const pass = e.beginRenderPass({ colorAttachments: [{ view: target.createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] }], ...(tex.table ? { resourceTable: tex.table } : {}) } as GPURenderPassDescriptor);
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, device.createBindGroup({ layout: io, entries: [{ binding: 0, resource: { buffer: params } }] }));
    pass.setBindGroup(1, device.createBindGroup({ layout: scene, entries: rtSceneCopyBindGroupEntries(buffers) }));
    pass.draw(3);
    pass.end();
    e.copyTextureToBuffer({ texture: target }, { buffer: readback, bytesPerRow: FRAG_W * 16 }, [FRAG_W, height]);
    device.queue.submit([e.finish()]);
    data.destroy();
    return readVec4(device, readback, count);
  };
  for (const set of ['fit', 'many'] as const) {
    const inputs = set === 'fit' ? fitSet() : manySet();
    const base: TextureOptions = { mips: 'cpu', maxSize: 512 };
    const variants = [
      { name: 'arrays', textures: buildSceneTextures(inputs, base) },
      { name: 'table', textures: buildSceneTextures(inputs, { ...base, binding: 'table' }) },
      ...(set === 'fit' ? [{ name: 'table1', textures: buildSceneTextures(inputs, { ...base, binding: 'table', maxLayers: 1 }) }] : []),
    ].map((v) => {
      const words = slotTable(v.textures);
      const table = device.createBuffer({ size: words.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(table, 0, words as Uint32Array<ArrayBuffer>);
      return { ...v, table, tex: uploadSceneTextures(device, v.textures, `frag ${set} ${v.name}`) };
    });
    const levels = Math.min(...variants[0]!.textures.buckets.map((b) => b.mipLevels));
    for (const access of ['coh', 'div', 'holes'] as const) {
      const tag = `frag_${set}_${access}`;
      const ref = await run(variants[0]!.textures, variants[0]!.tex, variants[0]!.table, access, levels);
      for (const v of variants) {
        if (v !== variants[0]) {
          const c = compare(ref, await run(v.textures, v.tex, v.table, access, levels));
          metrics[`${tag}_${v.name}_vs_arrays_differing`] = c.differing;
          if (set === 'fit' && c.differing) (access === 'coh' ? issues : findings).push(`${tag}: ${v.name} differs from arrays in ${c.differing} fragments (max ${c.maxAbs})`);
        }
        const lod0 = await run(v.textures, v.tex, v.table, access, 0);
        let bad = 0;
        let maxTwin = 0;
        const n = Math.min(count, 4096);
        for (let i = 0; i < n; i++) {
          const t = twinLane(v.textures, i, access);
          let d = 0;
          for (let c = 0; c < 4; c++) d = Math.max(d, Math.abs((lod0[i * 4 + c] as number) - (t[c] as number)));
          if (d > 1e-2) bad++;
          maxTwin = Math.max(maxTwin, d);
        }
        metrics[`${tag}_${v.name}_vs_twin_max_abs`] = maxTwin;
        metrics[`${tag}_${v.name}_vs_twin_bad_frac`] = bad / n;
        if (maxTwin > 1e-2) (v.name === 'arrays' || access === 'coh' ? issues : findings).push(`${tag}: ${v.name} lod-0 fragments differ from the CPU twin by ${maxTwin} in ${bad} of ${n}`);
      }
    }
    for (const v of variants) { v.tex.textures.forEach((t) => t.destroy()); v.tex.table?.destroy(); }
  }
  notes['fragment'] = { width: FRAG_W, height, binding: 'beginRenderPass({ resourceTable })' };
}

async function renderCheck(args: Record<string, string>, metrics: Record<string, number>, notes: Record<string, unknown>, issues: string[], findings: string[]): Promise<void> {
  const scenes = (args['scenes'] ?? 'tex-checker,tex-mask').split(',').filter(Boolean);
  const spp = numberArg(args, 'spp', 4);
  const size = numberArg(args, 'size', 0);
  for (const name of scenes) {
    const desc = getScene(name);
    const scene = size > 0 && !desc.meshes.some((m) => m.geometry.type === 'file') ? { ...desc, film: { ...desc.film, width: size, height: Math.round((size * desc.film.height) / desc.film.width) } } : desc;
    const images: Record<string, Float32Array> = {};
    const bindings = (args['bindings'] ?? 'arrays,table').split(',') as Array<'arrays' | 'table'>;
    for (const binding of bindings) {
      const adapter = new PtMegakernelAdapter({ textureBinding: binding, seed: 1 });
      await adapter.load(scene);
      const result = await adapter.render({ spp });
      images[binding] = result.image.data;
      // `upload=1`: PFMs for a CPU comparison across runs (SwiftShader loses its instance on a second megakernel adapter in one page).
      if (args['upload'] === '1') await uploadImage(`${name}-${binding}`, { width: result.image.width, height: result.image.height, channels: result.image.channels, data: result.image.data });
      metrics[`${name}_${binding}_ms_per_spp`] = result.timings.wallMs / spp;
      adapter.dispose();
    }
    if (!images['arrays'] || !images['table']) continue;
    const c = compare(images['arrays']!, images['table']!);
    let sum = 0;
    for (let i = 0; i < images['arrays']!.length; i += 4) sum += images['arrays']![i]! + images['arrays']![i + 1]! + images['arrays']![i + 2]!;
    metrics[`${name}_pixels_differing`] = c.differing;
    metrics[`${name}_max_abs`] = c.maxAbs;
    metrics[`${name}_mean`] = sum / (images['arrays']!.length / 4) / 3;
    notes[`${name}_film`] = `${scene.film.width}x${scene.film.height}, ${spp} spp`;
    // A megakernel's texture index is divergent: a device that breaks divergent indices (lookup findings) breaks this too.
    if (c.differing) (args['renderStrict'] === '0' ? findings : issues).push(`${name}: table image differs from arrays in ${c.differing} pixels (max ${c.maxAbs})`);
  }
}

async function main(): Promise<void> {
  const args = getArgs();
  const checks = (args['check'] ?? 'lookup,render').split(',');
  if (!navigator.gpu) throw new Error('navigator.gpu is undefined');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('requestAdapter() returned null');
  const summary: AdapterSummary = summarizeAdapter(adapter);
  const detected = detectCaps(adapter);
  if (detected.resourceTable === 'none') throw new Error(`no resource table on this adapter (features: ${[...adapter.features].join(', ')}); run Chrome with the feature (Canary 160+, or a flagged Chrome on Windows / Linux)`);
  const { device, caps } = await requestDeviceForCaps(adapter, detected, { label: 'bindless-check' });
  const errors = watchDevice(device);
  const metrics: Record<string, number> = {};
  const notes: Record<string, unknown> = { feature: caps.featureNames.samplingResourceTable ?? caps.featureNames.heterogeneousResourceTable, adapter: `${summary.vendor} ${summary.architecture}` };
  const issues: string[] = [];
  const findings: string[] = [];
  if (checks.includes('lookup')) await lookupCheck(device, caps, args, metrics, notes, issues, findings);
  if (checks.includes('fragment')) await fragmentCheck(device, caps, args, metrics, notes, issues, findings);
  if (checks.includes('render')) await renderCheck(args, metrics, notes, issues, findings);
  notes['findings'] = findings;
  metrics['findings'] = findings.length;
  await device.queue.onSubmittedWorkDone();
  notes['issues'] = issues;
  metrics['issues'] = issues.length;
  if (issues.length) throw new Error(`bindless-check: ${issues.join('; ')}\nmetrics: ${JSON.stringify(metrics)}`);
  reportResult({ metrics, notes }, { adapter: summary, deviceErrors: errors });
}

main().catch((e) => reportError(e));
