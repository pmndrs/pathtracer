/**
 * WGSL of the participating-media code (P5, prefix `ptVol` / `ptHg`), appended to the integrator only when `integrator.volumes` is set, so every kernel
 * without media keeps its exact text. Twin of `volume.ts` (read its header for the estimators).
 *
 * Tables are `var<private>` initialisers (a few media and one entry per material: tens of vec4, no storage buffer). A dense density grid is a
 * `texture_3d<f32>` (`r32float`, `textureLoad` only) at group 2 binding `PT_VOLUME_BINDING`; at most one grid per kernel (all grid media share it).
 *
 * Hard loop caps (V1.4; a dense medium that reaches one is silently biased; `cap-risk.ts` `ptVolumeCapRisk` warns at create when expected collisions > 10% of the cap):
 *  - free flight `ptVolSample` (global majorant), 4096 tentative collisions: the path is "killed" (darkens).
 *  - brick free flight `ptVolSampleBrick`: 4096 cells (outer) or 4096 collisions in one cell (inner): the rest of the segment / cell is passed without
 *    collisions (brightens; rare).
 *  - ratio-tracking shadow transmittance `ptVolTr*`, 4096 collisions: returns the partial product so far (brightens).
 *  - `ptVoBegin` (volume outputs), 256 boundary crossings: the primary-transmittance walk stops (that output only, never the image).
 *  - `ptVolTrCore`, 64 boundary crossings of a shadow ray: returns 0, the light is treated as occluded (darkens).
 *  - path interface skips, 256: the path ends (darkens).
 *  - telescoping `k` steps per cell, clamp 64: more variance only, no bias.
 *
 * Needs from the surrounding module: `RtRay`, `rtClosest`, `RT_MASK_ALL`, `ptOffsetOrigin`, `ptSideNormal`, `ptMax3`, `PtFrame` helpers, the light sampling.
 */
import type { PtMedium, PtVolumeOptions } from './volume.ts';
import { ptMediumMajorant } from './volume.ts';
import { ptPhaseValidate } from './phase.ts';

export const PT_VOLUME_BINDING = 16;
/** The coarse majorant grid of `majorant: 'brick'` (`r32float` 3D texture, one texel per 8^3 voxels; V2.1: `rg32float` (max, min) with `shadowTracking: 'residual'`). */
export const PT_VOLUME_MAJORANT_BINDING = 17;
/** V1.2 `densityFilter: 'hardware'`: the linear clamp-to-edge sampler of the density lookup (group 2; the grid texture is then bound `float`, filterable). */
export const PT_VOLUME_SAMPLER_BINDING = 20;
/** V3.2 `gridLayout: 'atlas'`: the `r32uint` brick indirection texture (`bx | by << 10 | bz << 20` of the atlas brick, `0xffffffff` empty); binding 16 is then the atlas. */
export const PT_VOLUME_INDIRECTION_BINDING = 21;

const f32Lit = (v: number): string => {
  const f = Math.fround(v);
  if (!Number.isFinite(f)) throw new RangeError(`volume table value ${v} is not finite`);
  return Number.isInteger(f) && Math.abs(f) < 1e9 ? `${f}.0` : String(f);
};

/** True when a medium has a second phase lobe (P5.v14): the media table gets a 7th row per medium and the phase calls go through `ptVolPhase` / `ptVolPhaseSample`. */
export const ptVolumeHasLobes = (v: PtVolumeOptions | null | undefined): boolean => !!v && v.media.some((m) => !!m.lobes && m.lobes.w > 0);

/** True when the options have a grid medium (the kernel needs the 3D texture binding). */
export const ptVolumeHasGrid = (v: PtVolumeOptions | null | undefined): boolean => !!v && v.media.some((m) => m.grid);

/** Validates the options and returns the grid resolution (all grid media must share one grid texture). */
export function ptVolumeValidate(v: PtVolumeOptions): void {
  if (v.media.length === 0) throw new RangeError('integrator.volumes: no media');
  v.media.forEach((m, i) => {
    if (m.sigmaT.some((s) => !(s >= 0) || !Number.isFinite(s))) throw new RangeError(`integrator.volumes: medium ${i} has an invalid sigmaT`);
    if (!(m.g > -1 && m.g < 1)) throw new RangeError(`integrator.volumes: medium ${i} g ${m.g} is outside (-1, 1)`);
    if (m.height) {
      if (m.grid) throw new RangeError(`integrator.volumes: medium ${i} has both a grid and a height profile`);
      if (!(m.height.H > 0) || !Number.isFinite(m.height.H) || !Number.isFinite(m.height.y0)) throw new RangeError(`integrator.volumes: medium ${i} has an invalid height profile (H must be > 0)`);
      if (v.forcedNee || v.equiangular) throw new RangeError('integrator.volumes: a height medium cannot be combined with forcedNee / equiangular (they assume a homogeneous medium)');
    }
    if (m.grid && !(m.grid.densityMax >= 0)) throw new RangeError(`integrator.volumes: medium ${i} has an invalid grid densityMax`);
    if (m.grid && !(Math.max(...m.sigmaT) > 0)) throw new RangeError(`integrator.volumes: grid medium ${i} has no extinction (sigmaT is 0)`);
    if (m.lobes) {
      ptPhaseValidate(m.lobes, `integrator.volumes: medium ${i} phase`);
      if (m.lobes.gA !== m.g) throw new RangeError(`integrator.volumes: medium ${i} has g ${m.g} but its lobe A is ${m.lobes.gA}`);
    }
  });
  if ((v.shadowTracking === 'residual' || v.shadowTracking === 'telescoping') && !(v.majorant === 'brick' && v.media.some((m) => m.grid))) throw new RangeError(`integrator.volumes: shadowTracking '${v.shadowTracking}' needs a grid medium with majorant 'brick'`);
  const grids = v.media.filter((m) => m.grid);
  if (grids.length > 1) {
    const [a] = grids;
    for (const m of grids) if (m.grid!.resolution.join() !== a!.grid!.resolution.join()) throw new RangeError('integrator.volumes: grid media must share one resolution (one 3D texture per kernel)');
  }
  const n = v.media.length;
  for (const mv of v.materials) if (mv.interior > n || mv.exterior > n || mv.interior < 0 || mv.exterior < 0) throw new RangeError('integrator.volumes: a material names a medium that does not exist');
  if ((v.cameraMedium ?? 0) > n) throw new RangeError('integrator.volumes: cameraMedium does not exist');
}

function mediumVec4s(m: PtMedium, lobes: boolean): string[] {
  const mu = ptMediumMajorant(m);
  const grid = m.grid;
  const k = grid?.densityScale ?? 1; // V3.1: r8unorm texels are density / scale; sigma_t x scale is exact, and the brick majorant texture holds texel units
  const w = grid ? grid.worldToIndex : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
  const row = (a: number[]): string => `vec4f(${a.map(f32Lit).join(', ')})`;
  return [
    row([...(k === 1 ? m.sigmaT : m.sigmaT.map((x) => x * k)), mu]),
    row([...m.albedo, m.g]),
    row(grid ? [Math.max(...m.sigmaT) * k, ...grid.resolution] : m.height ? [0, m.height.H, m.height.y0, 0] : [0, 0, 0, 0]),
    row(w.slice(0, 4)),
    row(w.slice(4, 8)),
    row(w.slice(8, 12)),
    ...(lobes ? [row(m.lobes && m.lobes.w > 0 ? [m.lobes.gB, m.lobes.w, m.lobes.alpha, 0] : [0, 0, 0, 0])] : []),
  ];
}

/**
 * The pieces of the volume WGSL, so a scheduler that runs the stages in different kernels (the wavefront, V270) takes only what a kernel needs:
 * `seedOf` the per-path stream seed (a pure function of pixel seed and sample index), `bind` the grid textures, `tables` the media / material tables,
 * the stream state `ptVolState` and `ptVolRand`, `hg` the Henyey-Greenstein phase and its sampling (needs the BSDF frame helpers), `density` the grid lookup and
 * the brick DDA (needs `bind`), `flight` delta tracking (`ptVolSample`), `tr` shadow-ray transmittance (walk, `ptOffsetOrigin`), `nee` next-event estimation at a
 * medium vertex (light sampling, MIS). No argument = all of them, in the order the megakernel appends them.
 */
export type PtVolumePart = 'seedOf' | 'bind' | 'tables' | 'hg' | 'density' | 'flight' | 'tr' | 'nee' | 'outputs' | 'outputsMarch';
/** `outputs` emits text only with `volumes.outputs` (megakernel). `outputsMarch` (not in ALL_PARTS; V5.2) is the first half of it, the privates and `ptVoBegin`, which the wavefront's intersect kernel runs at iteration 0 (needs `tr`). */
export const PT_VOLUME_ALL_PARTS: readonly PtVolumePart[] = ['seedOf', 'bind', 'tables', 'hg', 'density', 'flight', 'tr', 'nee', 'outputs'];

/**
 * True when a shadow ray's transmittance is ESTIMATED (ratio tracking: a grid medium, or `shadowTracking: 'ratio'`), so it draws random numbers. Such a ray then
 * runs on its own stream (`ptVolFork`, forked from the path's stream when the shadow ray is built; the path's stream moves on by one step). That is what lets a
 * scheduler trace the shadow ray later than the path's next free flight and still draw the same numbers as the megakernel (V270). The analytic shadow draws nothing.
 */
export const ptVolumeRatioShadows = (v: PtVolumeOptions): boolean => ptVolumeHasGrid(v) || v.shadowTracking === 'ratio';

/** The volume WGSL: tables, HG phase, delta-tracking free flight, transmittance along a shadow ray through boundary surfaces, NEE at a medium vertex. `parts` picks pieces (default all). */
export function ptVolumeWgsl(v: PtVolumeOptions, parts: readonly PtVolumePart[] = PT_VOLUME_ALL_PARTS): string {
  ptVolumeValidate(v);
  const grid = ptVolumeHasGrid(v);
  const ratio = ptVolumeRatioShadows(v);
  const brick = grid && v.majorant === 'brick';
  const hw = grid && v.densityFilter === 'hardware';
  const atlas = grid && v.gridLayout === 'atlas';
  const roulette = v.trRoulette === true;
  const tele = brick && v.shadowTracking === 'telescoping';
  // telescoping estimates can be negative (a debiasing term): only an exact zero ends a shadow ray, or the clamp would bias the mean
  const alive = v.shadowTracking === 'telescoping' ? 'ptMax3(abs(tr)) > 0.0' : 'ptMax3(tr) > 0.0';
  const residual = brick && (v.shadowTracking === 'residual' || tele);
  const lobes = ptVolumeHasLobes(v);
  const height = v.media.some((m) => m.height); // V4.5: closed-form exponential fog; the code below is emitted only then
  const stride = lobes ? 7 : 6;
  const mats = v.materials.length > 0 ? v.materials : [{ interior: 0, exterior: 0, boundary: false }];
  const matRow = (m: (typeof mats)[number]): string => `vec4u(${m.interior}u, ${m.exterior}u, ${m.interior !== 0 || m.exterior !== 0 || m.boundary ? 1 : 0}u, ${m.boundary ? 1 : 0}u)`;
  const chunks: string[] = [];
  const has = (p: PtVolumePart): boolean => parts.includes(p);
  if (has('seedOf')) {
    chunks.push(/* wgsl */ `
fn ptVolSeedOf(pixelSeed: u32, sampleIndex: u32) -> u32 {
    var h = pixelSeed ^ (sampleIndex * 2654435761u) ^ 0x68e31da4u;
    h = (h ^ (h >> 16u)) * 0x7feb352du;
    h = (h ^ (h >> 15u)) * 0x846ca68bu;
    return h ^ (h >> 16u);
}
`);
  }
  if (has('bind')) {
    chunks.push(`${grid ? `@group(2) @binding(${PT_VOLUME_BINDING}) var ptVolGrid: texture_3d<f32>;
` : ''}${atlas ? `@group(2) @binding(${PT_VOLUME_INDIRECTION_BINDING}) var ptVolInd: texture_3d<u32>;
` : ''}${hw ? `@group(2) @binding(${PT_VOLUME_SAMPLER_BINDING}) var ptVolSampler: sampler;
` : ''}${brick ? `@group(2) @binding(${PT_VOLUME_MAJORANT_BINDING}) var ptVolMaj: texture_3d<f32>;
` : ''}`);
  }
  if (has('tables')) chunks.push(/* wgsl */ `
var<private> ptVolMedia: array<vec4f, ${v.media.length * stride}> = array<vec4f, ${v.media.length * stride}>(${v.media.flatMap((m) => mediumVec4s(m, lobes)).join(', ')});
var<private> ptVolMats: array<vec4u, ${mats.length}> = array<vec4u, ${mats.length}>(${mats.map(matRow).join(', ')});
var<private> ptVolState: u32;${tele ? `
var<private> ptVolSampleIdx: u32;` : ''}

// Free flight starts at the TRUE boundary crossing (V302): an interface skip moves the ray origin q off the surface point p along the side normal n, which would skip
// dot(q - p, n) / |cos| of the medium. The path carries that distance ("back"); the next flight samples from q - d * back over [0, segment + back].
fn ptVolBack(q: vec3f, p: vec3f, n: vec3f, d: vec3f) -> f32 {
    return dot(q - p, n) / max(abs(dot(d, n)), 0.015625);
}

struct PtVolMedium {
    sigmaT: vec3f,
    mu: f32,
    albedo: vec3f,
    g: f32,
    info: vec4f,
    w0: vec4f,
    w1: vec4f,
    w2: vec4f,
}

struct PtVolEvent {
    t: f32,
    kind: u32,
}

fn ptVolMedium(id: u32) -> PtVolMedium {
    let b = (id - 1u) * ${stride}u;
    let a0 = ptVolMedia[b];
    let a1 = ptVolMedia[b + 1u];
    return PtVolMedium(a0.xyz, a0.w, a1.xyz, a1.w, ptVolMedia[b + 2u], ptVolMedia[b + 3u], ptVolMedia[b + 4u], ptVolMedia[b + 5u]);
}

fn ptVolMat(material: u32) -> vec4u {
    return select(vec4u(0u), ptVolMats[min(material, ${mats.length - 1}u)], material < ${mats.length}u);
}

fn ptVolSeed(pixelSeed: u32, sampleIndex: u32) {
    ptVolState = ptVolSeedOf(pixelSeed, sampleIndex);${tele ? `
    ptVolSampleIdx = sampleIndex;` : ''}
}

fn ptVolRand() -> f32 {
    ptVolState = ptVolState * 747796405u + 2891336453u;
    let w = ((ptVolState >> ((ptVolState >> 28u) + 4u)) ^ ptVolState) * 277803737u;
    return f32(((w >> 22u) ^ w) >> 8u) * 5.9604644775390625e-8;
}
${ratio ? `
// Seed of a shadow ray's own random stream (ratio tracking): a hash of the path's stream, which then moves on by one step (V270).
fn ptVolFork() -> u32 {
    var h = ptVolState ^ 0x2545f491u;
    h = (h ^ (h >> 16u)) * 0x7feb352du;
    h = (h ^ (h >> 15u)) * 0x846ca68bu;
    ptVolState = ptVolState * 747796405u + 2891336453u;
    return h ^ (h >> 16u);
}
` : ''}`);
  if (has('hg')) chunks.push(/* wgsl */ `
fn ptHgPhase(cosTheta: f32, g: f32) -> f32 {
    let d = 1.0 + g * g + 2.0 * g * cosTheta;
    return 0.07957747154594767 * (1.0 - g * g) / (d * sqrt(max(d, 1.0e-12)));
}

fn ptHgSample(wo: vec3f, g: f32, u: vec2f) -> vec3f {
    var cosT = 1.0 - 2.0 * u.x;
    if (abs(g) >= 1.0e-3) {
        let s = (1.0 - g * g) / (1.0 + g - 2.0 * g * u.x);
        cosT = -(1.0 + g * g - s * s) / (2.0 * g);
    }
    cosT = clamp(cosT, -1.0, 1.0);
    let sinT = sqrt(max(0.0, 1.0 - cosT * cosT));
    let phi = 6.283185307179586 * u.y;
    return ptFromLocal(ptFrameFromNormal(wo), vec3f(sinT * cos(phi), sinT * sin(phi), cosT));
}
${lobes ? PT_VOL_LOBES_WGSL(stride) : ''}`);
  if (has('density') && height) chunks.push(/* wgsl */ `
// V4.5 height fog (info = (0, H, y0, 0)): sigma_t(p) = sigmaT * exp(-(p.y - y0) / H). Unit optical depth G of o + t d over [0, L], inversion, and free flight (volume.ts heightTrack).
fn ptVolIsHeight(m: PtVolMedium) -> bool {
    return m.info.x == 0.0 && m.info.y > 0.0;
}

fn ptVolHeightPhi(x: f32) -> f32 {
    if (abs(x) < 0.05) {
        return 1.0 - x * 0.5 + x * x * (1.0 / 6.0) - x * x * x * (1.0 / 24.0) + x * x * x * x * (1.0 / 120.0);
    }
    return (1.0 - exp(min(-x, 80.0))) / x;
}

fn ptVolHeightPsi(c: f32) -> f32 {
    if (abs(c) < 0.05) {
        return 1.0 + c * 0.5 + c * c * (1.0 / 3.0) + c * c * c * 0.25 + c * c * c * c * 0.2;
    }
    return -log(1.0 - c) / c;
}

fn ptVolHeightG(m: PtVolMedium, o: vec3f, d: vec3f, len: f32) -> f32 {
    let a = clamp((o.y - m.info.z) / m.info.y, -60.0, 60.0);
    return min(exp(-a) * len * ptVolHeightPhi(len * d.y / m.info.y), 1.0e30);
}
`);
  if (has('density')) chunks.push(/* wgsl */ `
fn ptVolDensity(m: PtVolMedium, p: vec3f) -> f32 {
${grid ? `    if (m.info.x == 0.0) {
        return 1.0;
    }
    let gx = vec3f(dot(m.w0.xyz, p) + m.w0.w, dot(m.w1.xyz, p) + m.w1.w, dot(m.w2.xyz, p) + m.w2.w);
${atlas ? `    let fl = floor(gx);
    let fr = gx - fl;
    let i0 = vec3i(fl);
    let dims = vec3i(i32(m.info.y), i32(m.info.z), i32(m.info.w));
    let c = (i0 + vec3i(1)) >> vec3u(3u);
    if (any(c < vec3i(0)) || any(c >= (dims + vec3i(8)) / vec3i(8))) {
        return 0.0;
    }
    let pk = textureLoad(ptVolInd, vec3u(c), 0).x;
    if (pk == 0xffffffffu) {
        return 0.0;
    }
    let org = vec3i(i32(pk & 1023u), i32((pk >> 10u) & 1023u), i32((pk >> 20u) & 1023u)) * 9 + i0 - (c * 8 - vec3i(1));
${hw ? `    return textureSampleLevel(ptVolGrid, ptVolSampler, (vec3f(org) + fr + vec3f(0.5)) / vec3f(textureDimensions(ptVolGrid)), 0.0).x;
` : `    var acc = 0.0;
    for (var cc = 0u; cc < 8u; cc++) {
        let o = vec3i(i32(cc & 1u), i32((cc >> 1u) & 1u), i32((cc >> 2u) & 1u));
        let v = textureLoad(ptVolGrid, vec3u(org + o), 0).x;
        let w = select(1.0 - fr.x, fr.x, o.x == 1) * select(1.0 - fr.y, fr.y, o.y == 1) * select(1.0 - fr.z, fr.z, o.z == 1);
        acc += v * w;
    }
    return acc;
`}` : hw ? `    let nf = vec3f(m.info.y, m.info.z, m.info.w);
    let bf = clamp(gx + vec3f(1.0), vec3f(0.0), vec3f(1.0)) * clamp(nf - gx, vec3f(0.0), vec3f(1.0));
    return bf.x * bf.y * bf.z * textureSampleLevel(ptVolGrid, ptVolSampler, (gx + vec3f(0.5)) / nf, 0.0).x;
` : `    let dims = vec3i(i32(m.info.y), i32(m.info.z), i32(m.info.w));
    let fl = floor(gx);
    let fr = gx - fl;
    let i0 = vec3i(fl);
    var acc = 0.0;
    for (var c = 0u; c < 8u; c++) {
        let o = vec3i(i32(c & 1u), i32((c >> 1u) & 1u), i32((c >> 2u) & 1u));
        let ii = i0 + o;
        let inside = all(ii >= vec3i(0)) && all(ii < dims);
        let v = select(0.0, textureLoad(ptVolGrid, vec3u(clamp(ii, vec3i(0), dims - vec3i(1))), 0).x, inside);
        let w = select(1.0 - fr.x, fr.x, o.x == 1) * select(1.0 - fr.y, fr.y, o.y == 1) * select(1.0 - fr.z, fr.z, o.z == 1);
        acc += v * w;
    }
    return acc;
`}` : `    return 1.0;
`}}

${brick ? `
struct PtDda {
    ua: vec3f,
    ub: vec3f,
    cell: vec3i,
    tNext: vec3f,
    dt: vec3f,
    step: vec3i,
    cdims: vec3i,
    t: f32,
    tExit: f32,
    valid: bool,
}

// The ray in coarse-cell space u = (index + 1) / 8, clipped to the support box of the grid; the DDA then visits the cells in order.
fn ptDdaInit(m: PtVolMedium, o: vec3f, d: vec3f, tmax: f32) -> PtDda {
    let a = vec3f(dot(m.w0.xyz, o) + m.w0.w, dot(m.w1.xyz, o) + m.w1.w, dot(m.w2.xyz, o) + m.w2.w);
    let b = vec3f(dot(m.w0.xyz, d), dot(m.w1.xyz, d), dot(m.w2.xyz, d));
    let cdims = (vec3i(i32(m.info.y), i32(m.info.z), i32(m.info.w)) + vec3i(8)) / vec3i(8);
    let ua = (a + vec3f(1.0)) * 0.125;
    let ub = b * 0.125;
    var s: PtDda;
    s.ua = ua;
    s.ub = ub;
    s.cdims = cdims;
    s.valid = false;
    var tEnter = 0.0;
    var tExit = tmax;
    for (var k = 0; k < 3; k++) {
        if (abs(ub[k]) < 1.0e-20) {
            if (ua[k] < 0.0 || ua[k] >= f32(cdims[k])) {
                return s;
            }
        } else {
            let t0 = (0.0 - ua[k]) / ub[k];
            let t1 = (f32(cdims[k]) - ua[k]) / ub[k];
            tEnter = max(tEnter, min(t0, t1));
            tExit = min(tExit, max(t0, t1));
        }
    }
    if (!(tEnter < tExit)) {
        return s;
    }
    s.cell = clamp(vec3i(floor(ua + ub * tEnter)), vec3i(0), cdims - vec3i(1));
    s.step = select(vec3i(-1), vec3i(1), ub > vec3f(0.0));
    let inv = select(vec3f(1.0e30), 1.0 / ub, abs(ub) >= vec3f(1.0e-20));
    s.dt = abs(inv);
    s.tNext = select(vec3f(1.0e30), (vec3f(s.cell) + select(vec3f(0.0), vec3f(1.0), ub > vec3f(0.0)) - ua) * inv, abs(ub) >= vec3f(1.0e-20));
    s.t = tEnter;
    s.tExit = tExit;
    s.valid = true;
    return s;
}

// (t0, t1, majorant density${residual ? ', minorant density' : ''}) of the next cell; t1 <= t0 = no more cells.
fn ptDdaNext(s: ptr<function, PtDda>) -> ${residual ? 'vec4f' : 'vec3f'} {
    if (!(*s).valid) {
        return ${residual ? 'vec4f(0.0)' : 'vec3f(0.0)'};
    }
    let t0 = (*s).t;
    let tn = (*s).tNext;
    var axis = 0;
    if (tn.y < tn[axis]) {
        axis = 1;
    }
    if (tn.z < tn[axis]) {
        axis = 2;
    }
    let t1 = min(tn[axis], (*s).tExit);
${residual ? `    let mm = textureLoad(ptVolMaj, vec3u((*s).cell), 0);
    let dens = mm.x;
` : `    let dens = textureLoad(ptVolMaj, vec3u((*s).cell), 0).x;
`}    (*s).t = t1;
    if (t1 >= (*s).tExit) {
        (*s).valid = false;
    } else {
        (*s).cell[axis] += (*s).step[axis];
        (*s).tNext[axis] += (*s).dt[axis];
        if ((*s).cell[axis] < 0 || (*s).cell[axis] >= (*s).cdims[axis]) {
            (*s).valid = false;
        }
    }
    return ${residual ? 'vec4f(t0, t1, dens, mm.y)' : 'vec3f(t0, t1, dens)'};
}

fn ptVolSampleBrick(m: PtVolMedium, o: vec3f, d: vec3f, tmax: f32, beta: ptr<function, vec3f>) -> PtVolEvent {
    var dda = ptDdaInit(m, o, d, tmax);
    for (var c = 0u; c < 4096u; c++) {
        let sg = ptDdaNext(&dda);
        if (!(sg.y > sg.x)) {
            break;
        }
        let mu = sg.z * m.info.x;
        if (!(mu > 0.0)) {
            continue;
        }
        var t = sg.x;
        for (var i = 0u; i < 4096u; i++) {
            t += -log(1.0 - ptVolRand()) / mu;
            if (t >= sg.y) {
                break;
            }
            let st = m.sigmaT * ptVolDensity(m, o + d * t);
            let ss = st * m.albedo;
            let ssMax = ptMax3(ss);
            let ps = ssMax / mu;
            if (ptVolRand() < ps) {
                *beta = *beta * (ss / ssMax);
                return PtVolEvent(t, 1u);
            }
            *beta = *beta * ((vec3f(1.0) - st / mu) / (1.0 - ps));
            if (!(ptMax3(*beta) > 0.0)) {
                return PtVolEvent(t, 2u);
            }
        }
    }
    return PtVolEvent(tmax, 0u);
}

${tele ? `// V2.3 telescoping transmittance of the residual optical depth of one DDA cell (Blender Cycles volume_transmittance, Apache-2.0, after Misso et al. 2022): jittered
// marching with k steps plus a debiasing term of random order n (P(n) = 0.9 0.1^n, tail lumped into n = 4) that takes N = 2^n times the samples. The order's uniform is shared
// by every pixel of one sample (hash of the sample index and the cell number: GPU coherent); the jitter comes from the shadow ray's own stream. The estimate may be < 0 or > 1.
fn ptVolTele(m: PtVolMedium, o: vec3f, d: vec3f, t0: f32, len: f32, mn: f32, sc: f32, cell: u32) -> vec3f {
    let k = u32(clamp(round(sc * len), 1.0, 64.0));
    var h = (ptVolSampleIdx * 2654435761u) ^ (cell * 2246822519u) ^ 0x51ed270bu;
    h = (h ^ (h >> 16u)) * 0x7feb352du;
    h = (h ^ (h >> 15u)) * 0x846ca68bu;
    h = h ^ (h >> 16u);
    let order = min(u32(floor(log(max(f32(h >> 8u) * 5.9604644775390625e-8, 1.0e-7)) / -2.302585092994046)), 4u);
    let nn = 1u << order;
    let pmf = select(0.9 * pow(0.1, f32(order)), 1.0e-4, order == 4u);
    let step = len / f32(nn * k);
    let off = ptVolRand();
    if (nn == 1u) {
        var tk = vec3f(0.0);
        for (var i = 0u; i < 64u; i++) {
            if (i >= k) {
                break;
            }
            tk += m.sigmaT * (ptVolDensity(m, o + d * (t0 + (off + f32(i)) * step)) - mn);
        }
        return exp(-tk * step);
    }
    var tj0 = vec3f(0.0);
    var tj1 = vec3f(0.0);
    var tja = vec3f(0.0);
    var tkSum = vec3f(0.0);
    for (var n = 0u; n < 16u; n++) {
        if (n >= nn) {
            break;
        }
        var tk = vec3f(0.0);
        for (var i = 0u; i < 64u; i++) {
            if (i >= k) {
                break;
            }
            let s = i * nn + n;
            let tau = step * m.sigmaT * (ptVolDensity(m, o + d * (t0 + (off + f32(s)) * step)) - mn);
            tk += tau * f32(nn);
            if ((s & 1u) == 0u) {
                tj0 += tau * 2.0;
            } else {
                tj1 += tau * 2.0;
            }
            tja += tau;
        }
        tkSum += exp(-tk);
    }
    return tkSum / f32(nn) + (exp(-tja) - 0.5 * (exp(-tj0) + exp(-tj1))) / pmf;
}

` : ''}fn ptVolTrBrick(m: PtVolMedium, o: vec3f, d: vec3f, len: f32) -> vec3f {
    var tr = vec3f(1.0);
    var dda = ptDdaInit(m, o, d, len);
    for (var c = 0u; c < 4096u; c++) {
        let sg = ptDdaNext(&dda);
        if (!(sg.y > sg.x)) {
            break;
        }
${residual ? `        // V2.2 residual ratio tracking: the cell's minorant is a control (analytic), the residual (max - min) is ratio tracked
        tr = tr * exp(-m.sigmaT * (sg.w * (sg.y - sg.x)));
        let mu = (sg.z - sg.w) * m.info.x;
` : `        let mu = sg.z * m.info.x;
`}        if (!(mu > 0.0)) {
            continue;
        }
${tele ? `        tr = tr * ptVolTele(m, o, d, sg.x, sg.y - sg.x, sg.w, mu, c);
        continue;
` : ''}        var t = sg.x;
        for (var i = 0u; i < 4096u; i++) {
            t += -log(1.0 - ptVolRand()) / mu;
            if (t >= sg.y) {
                break;
            }
            tr = tr * (vec3f(1.0) - m.sigmaT * ${residual ? '(ptVolDensity(m, o + d * t) - sg.w)' : 'ptVolDensity(m, o + d * t)'} / mu);
            if (!(ptMax3(tr) > 0.0)) {
                return vec3f(0.0);
            }
${roulette ? `            if (ptMax3(tr) < 0.05) {
                if (ptVolRand() < 0.75) {
                    return vec3f(0.0);
                }
                tr = tr * 4.0;
            }
` : ''}        }
    }
    return tr;
}
` : ''}`);
  if (has('flight')) chunks.push(/* wgsl */ `
// One free-flight segment [0, tmax] of a path in medium 'id' (delta tracking). kind 0: reached tmax (beta carries the weight), 1: real scatter at t, 2: killed.
fn ptVolSample(id: u32, o: vec3f, d: vec3f, tmax: f32, beta: ptr<function, vec3f>) -> PtVolEvent {
    let m = ptVolMedium(id);
    if (!(m.mu > 0.0)) {
        return PtVolEvent(tmax, 0u);
    }
${height ? `    if (ptVolIsHeight(m)) {
        let ss = m.sigmaT * m.albedo;
        let r = ptMax3(ss);
        let gm = ptVolHeightG(m, o, d, tmax);
        let xi = select(3.0e38, -log(1.0 - ptVolRand()) / r, r > 0.0);
        if (xi < gm) {
            let a = clamp((o.y - m.info.z) / m.info.y, -60.0, 60.0);
            let ea = exp(a);
            let c = min(xi * d.y * ea / m.info.y, 0.9999999);
            *beta = *beta * (ss * exp(-(m.sigmaT - vec3f(r)) * xi) / r);
            if (!(ptMax3(*beta) > 0.0)) {
                return PtVolEvent(xi * ea * ptVolHeightPsi(c), 2u);
            }
            return PtVolEvent(xi * ea * ptVolHeightPsi(c), 1u);
        }
        *beta = *beta * exp(-(m.sigmaT - vec3f(r)) * gm);
        return PtVolEvent(tmax, 0u);
    }
` : ''}${brick ? `    if (m.info.x != 0.0) {
        return ptVolSampleBrick(m, o, d, tmax, beta);
    }
` : ''}    var t = 0.0;
    for (var i = 0u; i < 4096u; i++) {
        t += -log(1.0 - ptVolRand()) / m.mu;
        if (t >= tmax) {
            return PtVolEvent(tmax, 0u);
        }
        let st = m.sigmaT * ptVolDensity(m, o + d * t);
        let ss = st * m.albedo;
        let ssMax = ptMax3(ss);
        let ps = ssMax / m.mu;
        if (ptVolRand() < ps) {
            *beta = *beta * (ss / ssMax);
            return PtVolEvent(t, 1u);
        }
        *beta = *beta * ((vec3f(1.0) - st / m.mu) / (1.0 - ps));
        if (!(ptMax3(*beta) > 0.0)) {
            return PtVolEvent(t, 2u);
        }
    }
    return PtVolEvent(tmax, 2u);
}
`);
  if (has('tr')) chunks.push(/* wgsl */ `
// Transmittance estimate over a segment of length len: closed form (homogeneous) or ratio tracking (grid).
fn ptVolTrSegment(id: u32, o: vec3f, d: vec3f, len: f32) -> vec3f {
    let m = ptVolMedium(id);
${height ? `    if (ptVolIsHeight(m)) {
        return exp(-m.sigmaT * ptVolHeightG(m, o, d, len));
    }
` : ''}${brick ? `    if (m.info.x != 0.0) {
        return ptVolTrBrick(m, o, d, len);
    }
` : ''}${ratio ? `    if (${grid && v.shadowTracking !== 'ratio' ? 'm.info.x != 0.0' : 'true'}) {
        var tr = vec3f(1.0);
        if (!(m.mu > 0.0)) {
            return tr;
        }
        var t = 0.0;
        for (var i = 0u; i < 4096u; i++) {
            t += -log(1.0 - ptVolRand()) / m.mu;
            if (t >= len) {
                break;
            }
            tr = tr * (vec3f(1.0) - m.sigmaT * ptVolDensity(m, o + d * t) / m.mu);
            if (!(ptMax3(tr) > 0.0)) {
                break;
            }
${roulette ? `            if (ptMax3(tr) < 0.05) {
                if (ptVolRand() < 0.75) {
                    return vec3f(0.0);
                }
                tr = tr * 4.0;
            }
` : ''}        }
        return tr;
    }
` : ''}    return exp(-m.sigmaT * len);
}

// Transmittance of a shadow ray from 'med0': solid surfaces block it, boundary (interface) surfaces pass it and switch the medium.${ratio ? ' Draws from the stream in ptVolState (the caller sets the shadow ray\'s own stream).' : ''}
fn ${ratio ? 'ptVolTrCore' : 'ptVolTr'}(ray0: RtRay, med0: u32) -> vec3f {
    var ray = ray0;
    var med = med0;
    var tr = vec3f(1.0);
    var back = 0.0;
    for (var i = 0u; i < 64u; i++) {
        let hit = rtClosest(ray, RT_MASK_ALL);
        let seg = select(ray.tMax, hit.t, hit.t >= 0.0);
        if (med != 0u) {
            tr = tr * ptVolTrSegment(med, ray.origin - ray.direction * back, ray.direction, seg + back);
            if (!(${alive})) {
                return vec3f(0.0);
            }
        }
        if (hit.t < 0.0) {
            return tr;
        }
        let mv = ptVolMat(hit.material);
        if (mv.w == 0u) {
            return vec3f(0.0);
        }
        med = select(mv.y, mv.x, dot(ray.direction, hit.normal) < 0.0);
        let pt = ray.origin + ray.direction * hit.t;
        let ns = ptSideNormal(hit.normal, ray.direction);
        let p = ptOffsetOrigin(pt, ns);
        back = ptVolBack(p, pt, ns, ray.direction);
        ray = RtRay(p, 0.0, ray.direction, ray.tMax - length(p - ray.origin));
        if (ray.tMax <= 0.0) {
            return tr;
        }
    }
    return vec3f(0.0);
}
${ratio ? `
// The megakernel traces the shadow ray where it builds it: on its own forked stream, so the path's stream is the same whenever the ray is traced.
fn ptVolTr(ray0: RtRay, med0: u32) -> vec3f {
    let seed = ptVolFork();
    let keep = ptVolState;
    ptVolState = seed;
    let tr = ptVolTrCore(ray0, med0);
    ptVolState = keep;
    return tr;
}
` : ''}`);
  if (has('nee')) chunks.push(/* wgsl */ `
// Next-event estimation at a scattering vertex inside medium 'id' (MIS against phase sampling).
fn ptSampleLdMedium(p: vec3f, wo: vec3f, id: u32, pixelSeed: u32, sampleIndex: u32, vertex: u32) -> vec3f {
    let selectU = ptSample2D(pixelSeed, sampleIndex, ptVertexPair(vertex, PT_SLOT_SELECT_COMPONENT)).x;
    let pointU = ptSample2D(pixelSeed, sampleIndex, ptVertexPair(vertex, PT_SLOT_LIGHT_POINT));
    let pick = rtSampleLight(selectU);
    if (pick.index == RT_NONE || !(pick.pmf > 0.0)) {
        return vec3f(0.0);
    }
    if (!PT_ENV_SAMPLING && pick.index == rtSceneTable[2].z) {
        return vec3f(0.0);
    }
    let li = rtSampleLightLi(pick.index, p, pointU);
    if (!li.valid || !(li.pdf > 0.0)) {
        return vec3f(0.0);
    }
    let ph = ${ptVolPhaseEvalExpr(v, 'id', 'dot(wo, li.wi)')};
    if (!(ph > 0.0)) {
        return vec3f(0.0);
    }
    var ray: RtRay;
    if (li.distance < 1.0e30) {
        let end = ptOffsetOrigin(li.point, ptSideNormal(li.normal, -li.wi));
        ray = rtSegment(p, end, 0.0, 0.0);
    } else {
        ray = RtRay(p, 0.0, li.wi, PT_FAR);
    }
    let tr = ptVolTr(ray, id);
    if (!(${alive})) {
        return vec3f(0.0);
    }
    let pl = pick.pmf * li.pdf;
    var w = 1.0;
    if (!li.delta) {
        w = ptMis(pl, ph);
    }
    return li.radiance * ph * tr * (w / pl);
}
`);
  if ((has('outputs') || has('outputsMarch')) && v.outputs) chunks.push(/* wgsl */ `
// P5.v15 / P5.v16 volume outputs (megakernel): per-sample values that ptMain adds to the accumulator tail.
var<private> ptVoRad01: vec3f;
var<private> ptVoCaptured: bool;
var<private> ptVoT: f32;
var<private> ptVoDist: f32;
var<private> ptVoHas: f32;

// Transmittance of the camera ray through the media up to the first SOLID surface (interface surfaces pass, a solid one ends the march with the
// transmittance so far: alpha of the media in front of it). The shadow rays' estimator (closed form or ratio tracking, brick majorant when on) on its own
// stream, so the path's numbers are those of the kernel without outputs.
fn ptVoBegin(ray0: RtRay, med0: u32, pixelSeed: u32, sampleIndex: u32) {
    ptVoRad01 = vec3f(0.0);
    ptVoCaptured = false;
    ptVoDist = 0.0;
    ptVoHas = 0.0;
    let keep = ptVolState;
    ptVolState = ptVolSeedOf(pixelSeed ^ 0x5bd1e995u, sampleIndex ^ 0x27d4eb2fu);
    var ray = ray0;
    var med = med0;
    var tr = vec3f(1.0);
    var back = 0.0;
    for (var i = 0u; i < 256u; i++) {
        let hit = rtClosest(ray, RT_MASK_ALL);
        let seg = select(ray.tMax, hit.t, hit.t >= 0.0);
        if (med != 0u) {
            tr = tr * ptVolTrSegment(med, ray.origin - ray.direction * back, ray.direction, seg + back);
        }
        if (hit.t < 0.0 || !(${alive})) {
            break;
        }
        let mv = ptVolMat(hit.material);
        if (mv.w == 0u) {
            break;
        }
        med = select(mv.y, mv.x, dot(ray.direction, hit.normal) < 0.0);
        let pt = ray.origin + ray.direction * hit.t;
        let ns = ptSideNormal(hit.normal, ray.direction);
        let q = ptOffsetOrigin(pt, ns);
        back = ptVolBack(q, pt, ns, ray.direction);
        ray = RtRay(q, 0.0, ray.direction, PT_FAR);
    }
    ptVoT = dot(max(tr, vec3f(0.0)), vec3f(0.2126, 0.7152, 0.0722));
    ptVolState = keep;
}
`);
  if (has('outputs') && v.outputs) chunks.push(/* wgsl */ `
// Called just before a path's depth goes from 1 to 2: everything added so far is orders 0 and 1.
fn ptVoCapture(depth: u32, radiance: vec3f) {
    if (depth == 1u && !ptVoCaptured) {
        ptVoRad01 = radiance;
        ptVoCaptured = true;
    }
}

fn ptVoEnd(radiance: vec3f) {
    if (!ptVoCaptured) {
        ptVoRad01 = radiance;
    }
}

fn ptVoAccumulate(index: u32, n: u32, l: vec3f) {
    ptAccum[n + index] = ptAccum[n + index] + vec4f(ptVoRad01, ptVoT);
    ptAccum[2u * n + index] = ptAccum[2u * n + index] + vec4f(l * l, ptVoDist);
    ptAccum[3u * n + index] = ptAccum[3u * n + index] + vec4f(ptVoHas, 0.0, 0.0, 0.0);
}
`);
  return chunks.join('');
}

/** The phase density at a medium vertex as a WGSL expression: one HG lobe (the text every kernel had before P5.v14) or the two-lobe mixture. */
export function ptVolPhaseEvalExpr(v: PtVolumeOptions, id: string, cosTheta: string): string {
  return ptVolumeHasLobes(v) ? `ptVolPhase(${id}, ${cosTheta})` : `ptHgPhase(${cosTheta}, ptVolMedium(${id}).g)`;
}

/**
 * The phase sample of a medium vertex (both schedulers): declares `wiName` and sets `pdfTarget` (the MIS pdf of the next hit). Without lobes this is the
 * exact text the kernels had before P5.v14 (`gM`, `ptHgSample`, `ptHgPhase`).
 */
export function ptVolPhaseSampleWgsl(v: PtVolumeOptions, indent: string, med: string, wo: string, u: string, wiName: string, pdfTarget: string): string {
  if (!ptVolumeHasLobes(v)) return `${indent}let gM = ptVolMedium(${med}).g;\n${indent}let ${wiName} = ptHgSample(${wo}, gM, ${u});\n${indent}${pdfTarget} = ptHgPhase(dot(${wo}, ${wiName}), gM);\n`;
  return `${indent}let ${wiName} = ptVolPhaseSample(${med}, ${wo}, ${u});\n${indent}${pdfTarget} = ptVolPhase(${med}, dot(${wo}, ${wiName}));\n`;
}

/**
 * Two-lobe phase (P5.v14, twin of `phase.ts`): `(1 - w) HG(gA) + w Draine(gB, alpha)`, gA in row 1 (`m.g`), (gB, w, alpha) in row 6 of the medium. `u` in the
 * lobe functions is the cosine with the FORWARD direction (-wo). HG is sampled in a form of 1 - u that does not cancel at a sub-degree forward peak (the Mie
 * fit's HG lobe has g 0.995 at 20 um); the Draine lobe by the exact CDF inversion of Jendersie and d'Eon 2023 (supplemental Eq. 6). The g = 0 Draine lobe (Cardano in the TS twin) is not compiled
 * (kernel size: grid + brick + lobes is at the 50 KB class); `ptPhaseValidate` refuses |gB| < 0.01 with alpha > 0.
 */
const PT_VOL_LOBES_WGSL = (stride: number): string => /* wgsl */ `
fn ptVolLobe(id: u32) -> vec4f {
    return ptVolMedia[(id - 1u) * ${stride}u + 6u];
}

fn ptVolDraine(u: f32, g: f32, a: f32) -> f32 {
    let g2 = g * g;
    let fac = 1.0 + g2 - 2.0 * g * u;
    return 0.07957747154594767 * (1.0 - g2) * (1.0 + a * u * u) / ((1.0 + a * (1.0 + 2.0 * g2) / 3.0) * fac * sqrt(max(fac, 1.0e-30)));
}

fn ptVolPhase(id: u32, cosTheta: f32) -> f32 {
    let l = ptVolLobe(id);
    let u = -cosTheta;
    return (1.0 - l.y) * ptVolDraine(u, ptVolMedium(id).g, 0.0) + l.y * ptVolDraine(u, l.x, l.z);
}

fn ptVolCbrt(x: f32) -> f32 {
    return sign(x) * pow(abs(x), 0.3333333333333333);
}

// 1 - u of a Draine sample (alpha > 0, |g| >= 0.01: ptPhaseValidate), u = cosine with the forward direction.
fn ptVolDraineOneMinusU(g: f32, a: f32, xi: f32) -> f32 {
    let g2 = g * g;
    let g4 = g2 * g2;
    let t0 = a - a * g2;
    let t1 = a * g4 - a;
    let t2 = -3.0 * (4.0 * (g4 - g2) + t1 * (1.0 + g2));
    let t3 = g * (2.0 * xi - 1.0);
    let t4 = 3.0 * g2 * (1.0 + t3) + a * (2.0 + g2 * (1.0 + (1.0 + 2.0 * g2) * t3));
    let t5 = t0 * (t1 * t2 + t4 * t4) + t1 * t1 * t1;
    let t6 = t0 * 4.0 * (g4 - g2);
    let t7 = ptVolCbrt(t5 + sqrt(max(0.0, t5 * t5 - t6 * t6 * t6)));
    let t8 = 2.0 * (t1 + t6 / t7 + t7) / t0;
    let t9 = sqrt(max(0.0, 6.0 * (1.0 + g2) + t8));
    let r = sqrt(max(0.0, 6.0 * (1.0 + g2) - t8 + 8.0 * t4 / (t0 * t9))) - t9;
    return 1.0 - clamp(g * 0.5 + 0.5 / g - r * r / (8.0 * g), -1.0, 1.0);
}

fn ptVolHgOneMinusU(g: f32, xi: f32) -> f32 {
    if (abs(g) < 1.0e-3) {
        return 2.0 - 2.0 * xi;
    }
    let den = 1.0 - g + 2.0 * g * xi;
    let t = (1.0 - g * g) / den;
    return (1.0 - g) * (1.0 - xi) * (t + 1.0 - g) / den;
}

fn ptVolPhaseSample(id: u32, wo: vec3f, u2: vec2f) -> vec3f {
    let l = ptVolLobe(id);
    var xi = u2.x;
    var x = 0.0;
    if (xi < l.y) {
        xi = min(xi / l.y, 0.99999994);
        if (l.z < 1.0e-3) {
            x = ptVolHgOneMinusU(l.x, xi);
        } else {
            x = ptVolDraineOneMinusU(l.x, l.z, xi);
        }
    } else {
        xi = min((xi - l.y) / (1.0 - l.y), 0.99999994);
        x = ptVolHgOneMinusU(ptVolMedium(id).g, xi);
    }
    x = clamp(x, 0.0, 2.0);
    let sinT = sqrt(max(0.0, x * (2.0 - x)));
    let phi = 6.283185307179586 * u2.y;
    return ptFromLocal(ptFrameFromNormal(-wo), vec3f(sinT * cos(phi), sinT * sin(phi), 1.0 - x));
}
`;
