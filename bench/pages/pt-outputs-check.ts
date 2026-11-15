/**
 * pt-outputs-check (R10, R4, P2 AOVs): the optional integrator outputs against closed forms, on both schedulers. Correctness only, no timing.
 *
 *  1. Hooks, a "constant cache" on a closed emissive diffuse room (sphere r = 1, camera at the centre, emission E two-sided, albedo rho, NEE off, no RR):
 *     every path is deterministic (a cosine-sampled Lambert bounce has weight exactly rho), so every pixel reads
 *       no hook, maxDepth M                          E (1 - rho^(M+1)) / (1 - rho)
 *       terminate at vertex k, cache 0               E (1 - rho^(k+1)) / (1 - rho)         (= the depth-limited sum)
 *       terminate at vertex k, cache = true indirect radiance rho E / (1 - rho)     E / (1 - rho)   (= the untruncated answer)
 *       update paths (1 per 4x4 block per sample, never terminate): the mean over 16 samples follows from the two values above
 *     A hook with its own storage buffer (group 3) counts its terminations: pixels * spp. The two schedulers agree per pixel.
 *  2. Counters, same room: depth M, extension rays M + 1, shadow rays M with NEE, termination `maxDepth`; with the hook at vertex k: depth k + 1, reason `hook`;
 *     a Lambert sphere under a uniform sky: depth 1, 2 extension rays, reason `escaped`; the reasons always sum to 1; megakernel = wavefront per pixel.
 *  4. The optional radiance callback (`ptOnRadiance`): per-vertex values in the closed room, and the sum of everything a path reports equals its pixel (checkRadiance).
 *  3. AOVs on a plane with a red mirror sphere above it: albedo = the plane's base colour (the mirror's red never shows: the guide follows the specular bounce),
 *     normal = the plane's exactly, viewZ of the plane equals the closed form per pixel, megakernel = wavefront.
 *
 * Args: spp=16 (hooks, counters), tol=1e-4
 */
import { PT_TERM_NAMES, PT_VERTEX_UPDATE, type PtAovs, type PtCounters } from '@pmndrs/pathtracer';
import type { AdapterSummary } from '../runner/contract.ts';
import { drainDevice, getArgs, numberArg, reportError, reportResult, summarizeAdapter, watchDevice } from '../runner/page-client.ts';
import type { SceneDesc } from '../scenes/schema.ts';
import { tessellateSphere } from '../scenes/geometry.ts';
import { cornell } from '../scenes/tier0/index.ts';
import { PtMegakernelAdapter, type PtAdapterSettings } from '../adapters/pt-megakernel/index.ts';

let adapterSummary: AdapterSummary | null = null;
let deviceErrors: string[] = [];
const problems: string[] = [];
const metrics: Record<string, number> = {};
const note = (m: string) => {
  if (problems.length < 30) problems.push(m);
};

const RHO = 0.5;
const E = 1;
const sphere = (() => {
  const m = tessellateSphere(1, 64, 32, { flat: true });
  return { type: 'triangles' as const, positions: Array.from(m.positions), indices: Array.from(m.indices), normals: Array.from(m.normals) };
})();

/** Closed emissive diffuse sphere, camera in the centre. */
function room(maxDepth: number, width = 32): SceneDesc {
  return {
    version: 1,
    name: 'closed-room',
    purpose: 'Closed diffuse emissive sphere: L = E / (1 - rho), depth-limited sums in closed form.',
    tier: 0,
    camera: { type: 'perspective', position: [0, 0, 0], target: [0, 0, -1], up: [0, 1, 0], yFovDeg: 60 },
    film: { width, height: width, filter: 'box' },
    integrator: { maxDepth },
    materials: { wall: { type: 'diffuse', reflectance: [RHO, RHO, RHO] } },
    meshes: [{ name: 'room', geometry: sphere, material: 'wall', emission: [E, E, E], twoSided: true }],
  };
}

/** Convex Lambert sphere under a uniform sky: one bounce, then escape. */
function skySphere(): SceneDesc {
  return {
    version: 1,
    name: 'sky-sphere',
    purpose: 'Lambert sphere under a uniform sky.',
    tier: 0,
    camera: { type: 'perspective', position: [0, 0, 2], target: [0, 0, 0], up: [0, 1, 0], yFovDeg: 40 },
    film: { width: 32, height: 32, filter: 'box' },
    integrator: { maxDepth: 8 },
    materials: { m: { type: 'diffuse', reflectance: [RHO, RHO, RHO] } },
    meshes: [{ name: 'ball', geometry: sphere, material: 'm' }],
    environment: { type: 'uniform', radiance: [1, 1, 1] },
  };
}

const PLANE = [0.2, 0.5, 0.8] as const;
/** A plane (albedo PLANE) seen from above with a red mirror sphere over it, under a uniform sky. */
function planeScene(): SceneDesc {
  return {
    version: 1,
    name: 'plane-mirror',
    purpose: 'AOV check: plane + smooth red conductor sphere under a sky.',
    tier: 0,
    camera: { type: 'perspective', position: [0, 3, 0.5], target: [0, 0, 0], up: [0, 1, 0], yFovDeg: 40 },
    film: { width: 48, height: 48, filter: 'box' },
    integrator: { maxDepth: 3 },
    materials: { floor: { type: 'diffuse', reflectance: PLANE }, mirror: { type: 'conductor', eta: [0.2, 0.2, 0.2], k: [3, 3, 3], roughness: 0 } },
    meshes: [
      { name: 'plane', geometry: { type: 'quad', size: [40, 40] }, material: 'floor' },
      { name: 'ball', geometry: { ...sphere }, material: 'mirror', transform: [0.5, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0.5, 0, 0, 1, 1, 1] },
    ],
    environment: { type: 'uniform', radiance: [1, 1, 1] },
  };
}

type Scheduler = 'megakernel' | 'wavefront';
const SCHEDULERS: Scheduler[] = ['megakernel', 'wavefront'];

interface Run {
  data: Float32Array;
  aovs: PtAovs | null;
  counters: PtCounters | null;
  adapter: PtMegakernelAdapter;
  hookBuffer?: GPUBuffer;
}

async function render(scene: SceneDesc, settings: PtAdapterSettings, spp: number, withHookBuffer = false): Promise<Run> {
  const adapter = new PtMegakernelAdapter({ nee: false, rrStartDepth: 0, passesPerSubmit: 4, ...settings });
  await adapter.load(scene);
  let hookBuffer: GPUBuffer | undefined;
  if (withHookBuffer) {
    const device = adapter.gpuContext!.device;
    const kernel = adapter.renderer!;
    hookBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    kernel.setHookBindGroup(device.createBindGroup({ layout: kernel.hookBindGroupLayout!, entries: [{ binding: 0, resource: { buffer: hookBuffer } }] }));
  }
  const result = await adapter.render({ spp });
  return { data: result.image.data, aovs: adapter.lastAovs, counters: adapter.lastCounters, adapter, hookBuffer };
}

async function readU32(device: GPUDevice, buffer: GPUBuffer): Promise<number> {
  const staging = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(buffer, 0, staging, 0, 16);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const v = new Uint32Array(staging.getMappedRange().slice(0))[0] as number;
  staging.unmap();
  staging.destroy();
  return v;
}

const geometricSum = (n: number) => (E * (1 - RHO ** n)) / (1 - RHO);

/** Max over pixels and channels 0..2 of |v - expected| / expected. */
function maxRelDev(data: Float32Array, expected: number): number {
  let worst = 0;
  for (let i = 0; i < data.length; i += 4) for (let c = 0; c < 3; c++) worst = Math.max(worst, Math.abs((data[i + c] as number) - expected) / expected);
  return worst;
}

function maxAbsDiff(a: Float32Array, b: Float32Array): number {
  let worst = 0;
  for (let i = 0; i < a.length; i++) worst = Math.max(worst, Math.abs((a[i] as number) - (b[i] as number)));
  return worst;
}

const hookWgsl = (k: number, cache: number, update = false) => `
fn ptOnVertex(v: PtPathVertex) -> u32 {
    if (${update ? '(v.flags & PT_VERTEX_UPDATE) != 0u' : 'false'}) {
        return PT_HOOK_CONTINUE;
    }
    return select(PT_HOOK_CONTINUE, PT_HOOK_TERMINATE, v.depth >= ${k}u);
}
fn ptTerminate(v: PtPathVertex) -> vec3f {
    return vec3f(${cache});
}`;

const countingHook = (k: number) => `
@group(3) @binding(0) var<storage, read_write> giCount: array<atomic<u32>>;
fn ptOnVertex(v: PtPathVertex) -> u32 {
    if (v.depth >= ${k}u) {
        atomicAdd(&giCount[0], 1u);
        return PT_HOOK_TERMINATE;
    }
    return PT_HOOK_CONTINUE;
}`;

async function checkHooks(spp: number, tol: number): Promise<void> {
  const K = 3;
  const M = 8;
  const trueIndirect = (RHO * E) / (1 - RHO);
  const results: Record<string, Float32Array> = {};
  for (const scheduler of SCHEDULERS) {
    const cases: { name: string; settings: PtAdapterSettings; expected: number; depth?: number }[] = [
      { name: 'no_hook', settings: {}, expected: geometricSum(M + 1) },
      { name: 'cache0', settings: { hooks: { wgsl: hookWgsl(K, 0) } }, expected: geometricSum(K + 1) },
      { name: 'cache_true', settings: { hooks: { wgsl: hookWgsl(K, trueIndirect) } }, expected: E / (1 - RHO) },
    ];
    for (const c of cases) {
      const run = await render(room(M), { scheduler, ...c.settings }, spp);
      const dev = maxRelDev(run.data, c.expected);
      metrics[`hooks_${scheduler}_${c.name}_max_rel_dev`] = dev;
      results[`${scheduler}_${c.name}`] = run.data;
      if (dev > tol) {
        metrics[`hooks_${scheduler}_${c.name}_mismatch`] = 1;
        note(`hooks ${scheduler} ${c.name}: max rel dev ${dev} (expected ${c.expected}, got ${run.data[0]})`);
      } else metrics[`hooks_${scheduler}_${c.name}_mismatch`] = 0;
      run.adapter.dispose();
    }
    // update paths: 1 of 16 samples per pixel (block 4) does not terminate, and runs to maxDepth M
    {
      const run = await render(room(M), { scheduler, hooks: { wgsl: hookWgsl(K, trueIndirect, true), updateBlock: 4 } }, 16);
      const expected = (15 * (E / (1 - RHO)) + geometricSum(M + 1)) / 16;
      const dev = maxRelDev(run.data, expected);
      metrics[`hooks_${scheduler}_update_max_rel_dev`] = dev;
      if (dev > tol) {
        metrics[`hooks_${scheduler}_update_mismatch`] = 1;
        note(`hooks ${scheduler} update paths: max rel dev ${dev} (expected ${expected}, got ${run.data[0]})`);
      } else metrics[`hooks_${scheduler}_update_mismatch`] = 0;
      run.adapter.dispose();
    }
    // a hook with its own buffer
    {
      const width = 32;
      const run = await render(room(M, width), { scheduler, hooks: { wgsl: countingHook(K), bindings: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }] } }, spp, true);
      const count = await readU32(run.adapter.gpuContext!.device, run.hookBuffer!);
      const expected = width * width * spp;
      metrics[`hooks_${scheduler}_buffer_count`] = count;
      metrics[`hooks_${scheduler}_buffer_mismatch`] = count === expected ? 0 : 1;
      if (count !== expected) note(`hooks ${scheduler} buffer: ${count} terminations, expected ${expected}`);
      run.adapter.dispose();
    }
  }
  for (const name of ['no_hook', 'cache0', 'cache_true']) {
    const d = maxAbsDiff(results[`megakernel_${name}`]!, results[`wavefront_${name}`]!);
    metrics[`hooks_agreement_${name}_max_abs`] = d;
    metrics[`hooks_agreement_${name}_mismatch`] = d < 1e-5 ? 0 : 1;
    if (d >= 1e-5) note(`hooks megakernel vs wavefront ${name}: max abs diff ${d}`);
  }
}

function expectCounters(label: string, c: PtCounters, expect: { depth: number; ext: number; shadow?: number; reason: (typeof PT_TERM_NAMES)[number] }): void {
  const s = c.summary;
  const reasonSum = Object.values(s.reasons).reduce((a, b) => a + b, 0);
  const bad: string[] = [];
  if (Math.abs(s.depth - expect.depth) > 1e-6) bad.push(`depth ${s.depth} != ${expect.depth}`);
  if (Math.abs(s.extensionRays - expect.ext) > 1e-6) bad.push(`extension rays ${s.extensionRays} != ${expect.ext}`);
  // a light sample can fail (invalid / zero pdf) now and then, so the shadow rays traced are a hair below one per vertex: 0.1% of them did not get one here
  if (expect.shadow !== undefined && !(s.shadowRays <= expect.shadow + 1e-9 && s.shadowRays >= expect.shadow * 0.995)) bad.push(`shadow rays ${s.shadowRays} != ${expect.shadow}`);
  if (Math.abs(s.reasons[expect.reason] - 1) > 1e-9) bad.push(`reason ${expect.reason} = ${s.reasons[expect.reason]}`);
  if (Math.abs(reasonSum - 1) > 1e-9) bad.push(`reasons sum to ${reasonSum}`);
  metrics[`counters_${label}_mismatch`] = bad.length ? 1 : 0;
  if (bad.length) note(`counters ${label}: ${bad.join('; ')}`);
}

/**
 * The optional radiance callback (`ptOnRadiance`, track N V84): a hook that sums every reported value per (pixel, vertex index) into its own buffer.
 *  - Closed room, NEE off, no RR, maxDepth M: the emission of vertex j is rho^j E, so slot j holds spp * rho^j E (j = 0..M); with the cache hook
 *    terminating at vertex K with value c, slots 0..K hold spp * rho^j E and slot K + 1 holds spp * rho^K c.
 *  - Bookkeeping, any settings (NEE + MIS, RR): the slots of a pixel sum to its accumulated radiance.
 * Both schedulers: the wavefront reports next-event light from its shadow kernel (the hook's group 3 there: `setHookRadianceBindGroup`).
 */
async function checkRadiance(spp: number, tol: number): Promise<void> {
  const M = 6;
  const K = 2;
  const SLOTS = 16;
  const wgsl = (terminateAt: number, cache: number) => `
@group(3) @binding(0) var<storage, read_write> radSlots: array<f32>;
fn ptOnRadiance(pixel: u32, sampleIndex: u32, index: u32, flags: u32, value: vec3f) {
    let i = (pixel * ${SLOTS}u + min(index, ${SLOTS - 1}u)) * 3u;
    radSlots[i] += value.x;
    radSlots[i + 1u] += value.y;
    radSlots[i + 2u] += value.z;
}
fn ptOnVertex(v: PtPathVertex) -> u32 {
    return select(PT_HOOK_CONTINUE, PT_HOOK_TERMINATE, v.depth >= ${terminateAt}u);
}
fn ptTerminate(v: PtPathVertex) -> vec3f {
    return vec3f(${cache});
}`;
  const bindings: GPUBindGroupLayoutEntry[] = [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }];
  const cases: { name: string; settings: PtAdapterSettings; terminateAt: number; cache: number; expected?: (j: number) => number }[] = [
    { name: 'closed', settings: {}, terminateAt: 1000, cache: 0, expected: (j) => (j <= M ? RHO ** j * E : 0) },
    { name: 'closed_cache', settings: {}, terminateAt: K, cache: 0.75, expected: (j) => (j <= K ? RHO ** j * E : j === K + 1 ? RHO ** K * 0.75 : 0) },
    { name: 'nee_rr_sum', settings: { nee: true, rrStartDepth: 2 }, terminateAt: 1000, cache: 0 },
  ];
  for (const scheduler of SCHEDULERS) for (const c of cases) {
    const adapter = new PtMegakernelAdapter({ nee: false, rrStartDepth: 0, passesPerSubmit: 4, scheduler, ...c.settings, hooks: { wgsl: wgsl(c.terminateAt, c.cache), bindings } });
    await adapter.load(room(M));
    const device = adapter.gpuContext!.device;
    const kernel = adapter.renderer!;
    const pixels = 32 * 32;
    const slots = device.createBuffer({ size: pixels * SLOTS * 12, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    kernel.setHookBindGroup(device.createBindGroup({ layout: kernel.hookBindGroupLayout!, entries: [{ binding: 0, resource: { buffer: slots } }] }));
    if ('setHookRadianceBindGroup' in kernel) kernel.setHookRadianceBindGroup(device.createBindGroup({ layout: kernel.hookRadianceBindGroupLayout!, entries: [{ binding: 0, resource: { buffer: slots } }] }));
    const result = await adapter.render({ spp });
    const staging = device.createBuffer({ size: slots.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(slots, 0, staging, 0, slots.size);
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const v = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    let worst = 0;
    for (let p = 0; p < pixels; p++) {
      for (let ch = 0; ch < 3; ch++) {
        let sum = 0;
        for (let j = 0; j < SLOTS; j++) {
          const got = (v[(p * SLOTS + j) * 3 + ch] as number) / spp;
          sum += got;
          if (c.expected) {
            const want = c.expected(j);
            worst = Math.max(worst, Math.abs(got - want) / Math.max(want, 1e-3));
          }
        }
        const pixel = result.image.data[p * 4 + ch] as number;
        worst = Math.max(worst, Math.abs(sum - pixel) / Math.max(pixel, 1e-3));
      }
    }
    metrics[`radiance_${scheduler}_${c.name}_max_rel_dev`] = worst;
    metrics[`radiance_${scheduler}_${c.name}_mismatch`] = worst > tol ? 1 : 0;
    if (worst > tol) note(`radiance callback ${scheduler} ${c.name}: max rel dev ${worst}`);
    slots.destroy();
    adapter.dispose();
  }
}

async function checkCounters(spp: number): Promise<void> {
  const M = 5;
  const K = 2;
  const raws: Record<string, Uint32Array> = {};
  for (const scheduler of SCHEDULERS) {
    const base = { scheduler, counters: true } as const;
    let run = await render(room(M), base, spp);
    expectCounters(`${scheduler}_room`, run.counters!, { depth: M, ext: M + 1, shadow: 0, reason: 'maxDepth' });
    raws[`${scheduler}_room`] = run.counters!.raw;
    run.adapter.dispose();
    run = await render(room(M), { ...base, nee: true }, spp);
    expectCounters(`${scheduler}_room_nee`, run.counters!, { depth: M, ext: M + 1, shadow: M, reason: 'maxDepth' });
    run.adapter.dispose();
    run = await render(room(M), { ...base, hooks: { wgsl: hookWgsl(K, 0) } }, spp);
    expectCounters(`${scheduler}_hook`, run.counters!, { depth: K + 1, ext: K + 1, reason: 'hook' });
    run.adapter.dispose();
    run = await render(skySphere(), { ...base, nee: false }, spp);
    // pixels outside the sphere do not exist at this field of view? the camera sees the sphere only partly: split by the reasons instead
    const s = run.counters!.summary;
    const ok = s.reasons.escaped > 0.999 && s.depth > 0 && s.depth <= 1 && s.extensionRays >= 1 && s.extensionRays <= 2;
    metrics[`counters_${scheduler}_sky_depth`] = s.depth;
    metrics[`counters_${scheduler}_sky_mismatch`] = ok ? 0 : 1;
    if (!ok) note(`counters ${scheduler} sky sphere: ${JSON.stringify(s)}`);
    raws[`${scheduler}_sky`] = run.counters!.raw;
    run.adapter.dispose();
    run = await render(room(M), { ...base, rrStartDepth: 1 }, spp);
    const rr = run.counters!.summary;
    const rrSum = Object.values(rr.reasons).reduce((a, b) => a + b, 0);
    metrics[`counters_${scheduler}_rr_fraction`] = rr.reasons.rr;
    metrics[`counters_${scheduler}_rr_mismatch`] = rr.reasons.rr > 0 && Math.abs(rrSum - 1) < 1e-9 ? 0 : 1;
    if (metrics[`counters_${scheduler}_rr_mismatch`]) note(`counters ${scheduler} rr: ${JSON.stringify(rr)}`);
    run.adapter.dispose();
  }
  for (const name of ['room', 'sky']) {
    const a = raws[`megakernel_${name}`]!;
    const b = raws[`wavefront_${name}`]!;
    let diff = 0;
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) diff++;
    metrics[`counters_agreement_${name}_differing_words`] = diff;
    metrics[`counters_agreement_${name}_mismatch`] = diff === 0 ? 0 : 1;
    if (diff) note(`counters megakernel vs wavefront ${name}: ${diff} differing words`);
  }
}

async function checkAovs(): Promise<void> {
  const scene = planeScene();
  const spp = 32;
  const runs: Record<string, Run> = {};
  for (const scheduler of SCHEDULERS) {
    const run = await render(scene, { scheduler, aovs: true, nee: true }, spp);
    runs[scheduler] = run;
    const a = run.aovs!;
    const bad: string[] = [];
    // The red mirror must never show up in the albedo: only the plane's base colour (or 0 where a reflected ray escapes) can.
    let maxR = 0;
    let normalBad = 0;
    let covered = 0;
    let zWorst = 0;
    let zChecked = 0;
    const cam = scene.camera;
    const fwd = norm3(sub3(cam.target, cam.position));
    const right = norm3(cross3(fwd, cam.up));
    const up = cross3(right, fwd);
    const tanHalf = Math.tan((cam.yFovDeg * Math.PI) / 360);
    for (let y = 0; y < a.height; y++) {
      for (let x = 0; x < a.width; x++) {
        const p = y * a.width + x;
        maxR = Math.max(maxR, a.albedo[p * 3] as number);
        const nx = a.normal[p * 3] as number;
        const ny = a.normal[p * 3 + 1] as number;
        const nz = a.normal[p * 3 + 2] as number;
        const len = Math.hypot(nx, ny, nz);
        if (len > 0 && (Math.abs(nx) > 1e-4 || Math.abs(ny - 1) > 1e-4 || Math.abs(nz) > 1e-4)) normalBad++;
        if ((a.coverage[p] as number) >= 1) covered++;
        // viewZ against the plane y = 0 along the pixel-centre ray, for pixels whose ray hits the plane without meeting the sphere (|x| big)
        const ndcX = ((x + 0.5) / a.width) * 2 - 1;
        const ndcY = 1 - ((y + 0.5) / a.height) * 2;
        const d = norm3(add3(add3(fwd, scale3(right, ndcX * tanHalf)), scale3(up, ndcY * tanHalf)));
        if (d[1] < -0.05) {
          const t = -cam.position[1] / d[1];
          const hitXZ = [cam.position[0] + t * d[0], cam.position[2] + t * d[2]];
          if (Math.hypot(hitXZ[0]! - 0, hitXZ[1]! - 1) > 1.3 && Math.abs(hitXZ[0]!) < 15) {
            const expected = -t * dot3(d, fwd);
            zWorst = Math.max(zWorst, Math.abs((a.viewZ[p] as number) - expected) / Math.abs(expected));
            zChecked++;
          }
        }
      }
    }
    metrics[`aov_${scheduler}_max_albedo_r`] = maxR;
    metrics[`aov_${scheduler}_normal_bad_pixels`] = normalBad;
    metrics[`aov_${scheduler}_coverage_full_pixels`] = covered;
    metrics[`aov_${scheduler}_viewz_max_rel_dev`] = zWorst;
    metrics[`aov_${scheduler}_viewz_checked`] = zChecked;
    if (maxR > PLANE[0] + 1e-5) bad.push(`albedo.r reaches ${maxR} (> plane ${PLANE[0]}): the mirror shows`);
    if (normalBad > 0) bad.push(`${normalBad} pixels with a normal other than +y`);
    if (zWorst > 3e-3 || zChecked < 100) bad.push(`viewZ rel dev ${zWorst} over ${zChecked} pixels`);
    // a pure plane pixel: exact albedo
    const corner = a.albedo.slice(0, 3);
    if (Math.abs((corner[0] as number) - PLANE[0]) > 1e-5 || Math.abs((corner[1] as number) - PLANE[1]) > 1e-5 || Math.abs((corner[2] as number) - PLANE[2]) > 1e-5) bad.push(`corner albedo ${[...corner]} != ${PLANE}`);
    metrics[`aov_${scheduler}_mismatch`] = bad.length ? 1 : 0;
    if (bad.length) note(`aovs ${scheduler}: ${bad.join('; ')}`);
  }
  const a = runs.megakernel!.aovs!;
  const b = runs.wavefront!.aovs!;
  const d = Math.max(maxAbsDiff(a.albedo, b.albedo), maxAbsDiff(a.normal, b.normal), maxAbsDiff(a.viewZ, b.viewZ));
  metrics.aov_agreement_max_abs = d;
  metrics.aov_agreement_mismatch = d < 1e-4 ? 0 : 1;
  if (d >= 1e-4) note(`aovs megakernel vs wavefront: max abs diff ${d}`);
  // the image itself with AOVs on equals the image without (the AOV code must not change the estimator)
  const plain = await render(scene, { scheduler: 'megakernel', nee: true }, spp);
  const dImg = maxAbsDiff(plain.data, runs.megakernel!.data);
  metrics.aov_image_unchanged_max_abs = dImg;
  metrics.aov_image_unchanged_mismatch = dImg === 0 ? 0 : 1;
  if (dImg !== 0) note(`image with aovs differs from without: ${dImg}`);
  for (const run of [...Object.values(runs), plain]) run.adapter.dispose();
}

/** Cornell: the wall albedos are their base colours and the normals are the walls' (green wall's normal exactly; the skewed red wall has interpolated vertex normals, so within 0.02 of its triangles' normals). */
async function checkCornell(): Promise<void> {
  const scene: SceneDesc = { ...cornell, film: { ...cornell.film, width: 128, height: 128 } };
  const RED = [0.63, 0.065, 0.05];
  const GREEN = [0.14, 0.45, 0.091];
  const WHITE = [0.725, 0.71, 0.68];
  // The red wall's four corners are not coplanar (Goral et al. data): two triangles (diagonal 0-2), two exact normals, both facing the room (-x).
  const corner = (x: number, y: number, z: number) => [x, y, z];
  const [q0, q1, q2, q3] = [corner(552.8, 0, 0), corner(549.6, 0, 559.2), corner(556, 548.8, 559.2), corner(556, 548.8, 0)];
  const tn = (a: number[], b: number[], c: number[]) => {
    const n = norm3(cross3(sub3(b, a), sub3(c, a)));
    return n[0] > 0 ? [-n[0], -n[1], -n[2]] : n;
  };
  const RED_NS = [tn(q0!, q1!, q2!), tn(q0!, q2!, q3!)];
  const near = (a: ArrayLike<number>, o: number, b: number[], eps: number) => [0, 1, 2].every((c) => Math.abs((a[o + c] as number) - b[c]!) < eps);
  for (const scheduler of SCHEDULERS) {
    const run = await render(scene, { scheduler, aovs: true, nee: true }, 8);
    const a = run.aovs!;
    const count = { red: 0, green: 0, white: 0, black: 0, bad: 0, edge: 0 };
    const badPixels: number[] = [];
    for (let p = 0; p < a.width * a.height; p++) {
      if ((a.coverage[p] as number) < 1) continue;
      const nx = a.normal[p * 3] as number;
      const ny = a.normal[p * 3 + 1] as number;
      const nz = a.normal[p * 3 + 2] as number;
      // Judged by the albedo the pixel reports: red must sit on the red wall (normal -x), green on the green wall (+x exactly), white on anything else.
      // The red wall is skewed (Goral et al. data) and has interpolated vertex normals: within 0.02 of its triangles' exact ones, not equal.
      if (near(a.albedo, p * 3, RED, 1e-5)) {
        if (RED_NS.some((r) => Math.hypot(nx - r[0]!, ny - r[1]!, nz - r[2]!) < 0.02)) count.red++;
        else badPixels.push(p);
      } else if (near(a.albedo, p * 3, GREEN, 1e-5)) {
        if (Math.hypot(nx - 1, ny, nz) < 2e-4) count.green++;
        else badPixels.push(p);
      } else if (near(a.albedo, p * 3, WHITE, 1e-5)) {
        if (Math.abs(Math.hypot(nx, ny, nz) - 1) < 1e-5) count.white++;
        else badPixels.push(p);
      } else if (near(a.albedo, p * 3, [0, 0, 0], 1e-5)) count.black++;
      else badPixels.push(p);
    }
    // A pixel that is neither wall colour is fine on an edge (the samples of a pixel land on two surfaces and average): it must differ from a neighbour.
    for (const p of badPixels) {
      const x = p % a.width;
      const y = (p - x) / a.width;
      let onEdge = false;
      for (let dy = -1; dy <= 1 && !onEdge; dy++) for (let dx = -1; dx <= 1; dx++) {
        const qx = x + dx;
        const qy = y + dy;
        if (qx < 0 || qy < 0 || qx >= a.width || qy >= a.height) continue;
        const q = qy * a.width + qx;
        if (Math.abs((a.albedo[q * 3] as number) - (a.albedo[p * 3] as number)) + Math.abs((a.albedo[q * 3 + 1] as number) - (a.albedo[p * 3 + 1] as number)) > 1e-3 || Math.abs((a.coverage[q] as number) - 1) > 1e-6) { onEdge = true; break; }
      }
      if (onEdge) count.edge++;
      else { count.bad++; if (count.bad <= 3) note(`cornell ${scheduler} wrong pixel (${x},${y}): normal ${[...a.normal.slice(p * 3, p * 3 + 3)]} albedo ${[...a.albedo.slice(p * 3, p * 3 + 3)]} coverage ${a.coverage[p]}`); }
    }
    metrics[`aov_cornell_${scheduler}_red`] = count.red;
    metrics[`aov_cornell_${scheduler}_green`] = count.green;
    metrics[`aov_cornell_${scheduler}_white`] = count.white;
    metrics[`aov_cornell_${scheduler}_emitter`] = count.black;
    metrics[`aov_cornell_${scheduler}_wrong`] = count.bad;
    metrics[`aov_cornell_${scheduler}_edge_pixels`] = count.edge;
    const ok = count.bad === 0 && count.red > 500 && count.green > 500 && count.white > 2000;
    metrics[`aov_cornell_${scheduler}_mismatch`] = ok ? 0 : 1;
    if (!ok) note(`cornell aovs ${scheduler}: ${JSON.stringify(count)}`);
    run.adapter.dispose();
  }
}

/** Informational: how many storage buffers a hook can add before the device refuses (the 10 per stage of Apple GPUs; see PtHooksOptions.bindings). */
async function checkBudget(): Promise<void> {
  const adapter = await navigator.gpu.requestAdapter();
  metrics.budget_max_storage_buffers_per_stage = adapter!.limits.maxStorageBuffersPerShaderStage;
  // P3.11: `packed` binds the scene copy as one storage buffer, so each kernel has one more binding for the hooks (keys carry `_packed`; the split keys are unchanged).
  for (const scenePacking of ['split', 'packed'] as const) for (const scheduler of SCHEDULERS) {
    for (const counters of [false, true]) {
      let fits = 0;
      for (let hookBuffers = 0; hookBuffers <= 8; hookBuffers++) {
        const decl = Array.from({ length: hookBuffers }, (_, i) => `@group(3) @binding(${i}) var<storage, read_write> giBuf${i}: array<u32>;`).join('\n');
        const use = Array.from({ length: hookBuffers }, (_, i) => `giBuf${i}[0] = giBuf${i}[0] + 1u;`).join(' ');
        const bindings = Array.from({ length: hookBuffers }, (_, i) => ({ binding: i, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' as const } }));
        const wgsl = `${decl}\nfn ptOnVertex(v: PtPathVertex) -> u32 { ${use} return PT_HOOK_CONTINUE; }`;
        try {
          const a = new PtMegakernelAdapter({ scheduler, counters, nee: false, scenePacking, hooks: { wgsl, bindings } });
          await a.load(room(2, 8));
          a.dispose();
          fits = hookBuffers;
        } catch {
          break;
        }
      }
      metrics[`budget_${scheduler}_counters${counters ? 1 : 0}_hook_storage_buffers_fit${scenePacking === 'packed' ? '_packed' : ''}`] = fits;
    }
  }
}

type V3 = [number, number, number];
const sub3 = (a: readonly number[], b: readonly number[]): V3 => [a[0]! - b[0]!, a[1]! - b[1]!, a[2]! - b[2]!];
const add3 = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const scale3 = (a: V3, s: number): V3 => [a[0] * s, a[1] * s, a[2] * s];
const dot3 = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross3 = (a: readonly number[], b: readonly number[]): V3 => [a[1]! * b[2]! - a[2]! * b[1]!, a[2]! * b[0]! - a[0]! * b[2]!, a[0]! * b[1]! - a[1]! * b[0]!];
const norm3 = (a: readonly number[]): V3 => {
  const l = Math.hypot(a[0]!, a[1]!, a[2]!);
  return [a[0]! / l, a[1]! / l, a[2]! / l];
};
void PT_VERTEX_UPDATE;

try {
  const args = getArgs();
  const spp = numberArg(args, 'spp', 16);
  const tol = numberArg(args, 'tol', 1e-4);
  if (!navigator.gpu) throw new Error('navigator.gpu is undefined: no WebGPU in this browser.');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('requestAdapter() returned null: no usable GPU.');
  adapterSummary = summarizeAdapter(adapter);
  const only = args.check ? args.check.split(',') : ['hooks', 'counters', 'aovs', 'radiance'];
  if (only.includes('hooks')) await checkHooks(spp, tol);
  if (only.includes('counters')) await checkCounters(spp);
  if (only.includes('aovs')) await checkAovs();
  if (only.includes('aovs')) await checkCornell();
  if (only.includes('budget')) await checkBudget();
  if (only.includes('radiance')) await checkRadiance(spp, tol);
  void drainDevice;
  void watchDevice;
  const total = Object.entries(metrics).filter(([k]) => k.endsWith('_mismatch')).reduce((sum, [, v]) => sum + v, 0);
  reportResult(
    { scene: 'pt-outputs-check', renderer: 'ours', config: { spp, tol, checks: only.join(',') }, valid: total === 0 && deviceErrors.length === 0, metrics: { ...metrics, mismatches: total }, notes: { problems } },
    { adapter: adapterSummary, deviceErrors },
  );
  if (total > 0) throw new Error(`pt-outputs-check: ${total} mismatches. First:\n${problems.join('\n')}`);
} catch (error) {
  reportError(error, { adapter: adapterSummary, deviceErrors });
}
