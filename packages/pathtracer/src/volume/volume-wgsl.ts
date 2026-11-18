/**
 * WGSL of the participating-media code (P5, prefix `ptVol` / `ptHg`), appended to the integrator only when `integrator.volumes` is set, so every kernel
 * without media keeps its exact text. Twin of `volume.ts` (read its header for the estimators).
 *
 * Tables are `var<private>` initialisers (a few media and one entry per material: tens of vec4, no storage buffer). A dense density grid is a
 * `texture_3d<f32>` (`r32float`, `textureLoad` only) at group 2 binding `PT_VOLUME_BINDING`; at most one grid per kernel (all grid media share it).
 *
 * Needs from the surrounding module: `RtRay`, `rtClosest`, `RT_MASK_ALL`, `ptOffsetOrigin`, `ptSideNormal`, `ptMax3`, `PtFrame` helpers, the light sampling.
 */
import type { PtMedium, PtVolumeOptions } from './volume.ts';
import { ptMediumMajorant } from './volume.ts';

export const PT_VOLUME_BINDING = 16;

const f32Lit = (v: number): string => {
  const f = Math.fround(v);
  if (!Number.isFinite(f)) throw new RangeError(`volume table value ${v} is not finite`);
  return Number.isInteger(f) && Math.abs(f) < 1e9 ? `${f}.0` : String(f);
};

/** True when the options have a grid medium (the kernel needs the 3D texture binding). */
export const ptVolumeHasGrid = (v: PtVolumeOptions | null | undefined): boolean => !!v && v.media.some((m) => m.grid);

/** Validates the options and returns the grid resolution (all grid media must share one grid texture). */
export function ptVolumeValidate(v: PtVolumeOptions): void {
  if (v.media.length === 0) throw new RangeError('integrator.volumes: no media');
  v.media.forEach((m, i) => {
    if (m.sigmaT.some((s) => !(s >= 0) || !Number.isFinite(s))) throw new RangeError(`integrator.volumes: medium ${i} has an invalid sigmaT`);
    if (!(m.g > -1 && m.g < 1)) throw new RangeError(`integrator.volumes: medium ${i} g ${m.g} is outside (-1, 1)`);
    if (m.grid && !(m.grid.densityMax >= 0)) throw new RangeError(`integrator.volumes: medium ${i} has an invalid grid densityMax`);
  });
  const grids = v.media.filter((m) => m.grid);
  if (grids.length > 1) {
    const [a] = grids;
    for (const m of grids) if (m.grid!.resolution.join() !== a!.grid!.resolution.join()) throw new RangeError('integrator.volumes: grid media must share one resolution (one 3D texture per kernel)');
  }
  const n = v.media.length;
  for (const mv of v.materials) if (mv.interior > n || mv.exterior > n || mv.interior < 0 || mv.exterior < 0) throw new RangeError('integrator.volumes: a material names a medium that does not exist');
  if ((v.cameraMedium ?? 0) > n) throw new RangeError('integrator.volumes: cameraMedium does not exist');
  if (v.cameraMedium && v.media[v.cameraMedium - 1]?.grid) throw new RangeError('integrator.volumes: the camera medium must be homogeneous (a grid medium is bounded by its mesh)');
}

function mediumVec4s(m: PtMedium): string[] {
  const mu = ptMediumMajorant(m);
  const grid = m.grid;
  const w = grid ? grid.worldToIndex : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
  const row = (a: number[]): string => `vec4f(${a.map(f32Lit).join(', ')})`;
  return [
    row([...m.sigmaT, mu]),
    row([...m.albedo, m.g]),
    row(grid ? [1, ...grid.resolution] : [0, 0, 0, 0]),
    row(w.slice(0, 4)),
    row(w.slice(4, 8)),
    row(w.slice(8, 12)),
  ];
}

/** The volume WGSL: tables, HG phase, delta-tracking free flight, transmittance along a shadow ray through boundary surfaces, NEE at a medium vertex. */
export function ptVolumeWgsl(v: PtVolumeOptions): string {
  ptVolumeValidate(v);
  const grid = ptVolumeHasGrid(v);
  const mats = v.materials.length > 0 ? v.materials : [{ interior: 0, exterior: 0, boundary: false }];
  const matRow = (m: (typeof mats)[number]): string => `vec4u(${m.interior}u, ${m.exterior}u, ${m.interior !== 0 || m.exterior !== 0 || m.boundary ? 1 : 0}u, ${m.boundary ? 1 : 0}u)`;
  return /* wgsl */ `
${grid ? `@group(2) @binding(${PT_VOLUME_BINDING}) var ptVolGrid: texture_3d<f32>;
` : ''}var<private> ptVolMedia: array<vec4f, ${v.media.length * 6}> = array<vec4f, ${v.media.length * 6}>(${v.media.flatMap(mediumVec4s).join(', ')});
var<private> ptVolMats: array<vec4u, ${mats.length}> = array<vec4u, ${mats.length}>(${mats.map(matRow).join(', ')});
var<private> ptVolState: u32;

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
    let b = (id - 1u) * 6u;
    let a0 = ptVolMedia[b];
    let a1 = ptVolMedia[b + 1u];
    return PtVolMedium(a0.xyz, a0.w, a1.xyz, a1.w, ptVolMedia[b + 2u], ptVolMedia[b + 3u], ptVolMedia[b + 4u], ptVolMedia[b + 5u]);
}

fn ptVolMat(material: u32) -> vec4u {
    return select(vec4u(0u), ptVolMats[min(material, ${mats.length - 1}u)], material < ${mats.length}u);
}

fn ptVolSeed(pixelSeed: u32, sampleIndex: u32) {
    var h = pixelSeed ^ (sampleIndex * 2654435761u) ^ 0x68e31da4u;
    h = (h ^ (h >> 16u)) * 0x7feb352du;
    h = (h ^ (h >> 15u)) * 0x846ca68bu;
    ptVolState = h ^ (h >> 16u);
}

fn ptVolRand() -> f32 {
    ptVolState = ptVolState * 747796405u + 2891336453u;
    let w = ((ptVolState >> ((ptVolState >> 28u) + 4u)) ^ ptVolState) * 277803737u;
    return f32(((w >> 22u) ^ w) >> 8u) * 5.9604644775390625e-8;
}

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

fn ptVolDensity(m: PtVolMedium, p: vec3f) -> f32 {
${grid ? `    if (m.info.x == 0.0) {
        return 1.0;
    }
    let gx = vec3f(dot(m.w0.xyz, p) + m.w0.w, dot(m.w1.xyz, p) + m.w1.w, dot(m.w2.xyz, p) + m.w2.w);
    let dims = vec3i(i32(m.info.y), i32(m.info.z), i32(m.info.w));
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
` : `    return 1.0;
`}}

// One free-flight segment [0, tmax] of a path in medium 'id' (delta tracking). kind 0: reached tmax (beta carries the weight), 1: real scatter at t, 2: killed.
fn ptVolSample(id: u32, o: vec3f, d: vec3f, tmax: f32, beta: ptr<function, vec3f>) -> PtVolEvent {
    let m = ptVolMedium(id);
    if (!(m.mu > 0.0)) {
        return PtVolEvent(tmax, 0u);
    }
    var t = 0.0;
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

// Transmittance estimate over a segment of length len: closed form (homogeneous) or ratio tracking (grid).
fn ptVolTrSegment(id: u32, o: vec3f, d: vec3f, len: f32) -> vec3f {
    let m = ptVolMedium(id);
${grid ? `    if (m.info.x != 0.0) {
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
        }
        return tr;
    }
` : ''}    return exp(-m.sigmaT * len);
}

// Transmittance of a shadow ray from 'med0': solid surfaces block it, boundary (interface) surfaces pass it and switch the medium.
fn ptVolTr(ray0: RtRay, med0: u32) -> vec3f {
    var ray = ray0;
    var med = med0;
    var tr = vec3f(1.0);
    for (var i = 0u; i < 64u; i++) {
        let hit = rtClosest(ray, RT_MASK_ALL);
        let seg = select(ray.tMax, hit.t, hit.t >= 0.0);
        if (med != 0u) {
            tr = tr * ptVolTrSegment(med, ray.origin, ray.direction, seg);
            if (!(ptMax3(tr) > 0.0)) {
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
        let p = ptOffsetOrigin(ray.origin + ray.direction * hit.t, ptSideNormal(hit.normal, ray.direction));
        ray = RtRay(p, 0.0, ray.direction, ray.tMax - length(p - ray.origin));
        if (ray.tMax <= 0.0) {
            return tr;
        }
    }
    return vec3f(0.0);
}

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
    let ph = ptHgPhase(dot(wo, li.wi), ptVolMedium(id).g);
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
    if (!(ptMax3(tr) > 0.0)) {
        return vec3f(0.0);
    }
    let pl = pick.pmf * li.pdf;
    var w = 1.0;
    if (!li.delta) {
        w = ptMis(pl, ph);
    }
    return li.radiance * ph * tr * (w / pl);
}
`;
}
