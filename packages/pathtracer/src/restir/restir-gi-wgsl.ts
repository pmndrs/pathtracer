/**
 * ReSTIR GI (P5, real-time mode, megakernel): WGSL of the three extra passes (`ptRestirGiInit`, `ptRestirGiTemporal`, `ptRestirGiSpatial`) that run after the ReSTIR DI
 * passes (they share the DI module, the G-buffer and the params record) and the shade-side read (`ptRestirGiLi`). Written from Ouyang et al. 2021 "ReSTIR GI: Path Resampling for
 * Real-Time Path Tracing" (the secondary vertex as the resampled sample, the reconnection shift) and Lin et al. 2022 "Generalized resampled importance sampling" (the
 * contribution-weight form and the balance-heuristic MIS of the unbiased combine); no code from RTXGI or any other renderer. Design: docs/internal/design/restir-gi.md.
 *
 * **Sample.** A visible point x1 (the primary hit) shoots one BSDF-sampled ray to x2; the sample is the reconnection vertex `(x2, n2, Lo)` with `Lo` the radiance x2 reflects
 * back toward x1: next-event estimation at x2 (light sampling, no MIS), so the sample stands for the 2-bounce path x1 -> x2 -> light. Emission seen directly from x1
 * (including through the BSDF sample) belongs to ReSTIR DI, which covers every light at the first vertex without MIS: the GI term and the shade add up to a path tracer of depth 2.
 *
 * **Measure.** Samples live in the AREA measure at x2, which makes the reconnection Jacobian implicit: the target of a pixel `d` for the sample y is
 * `F_d(y) = f_d(wo_d, wi_d) |cos1_d| Lo G_d V_d(y)`, `G_d = |cos phi2_d| / |x2 - x1_d|^2`, `phat = luminance(F)`, and the contribution weight W is "per unit area at x2"
 * (initial candidate: `W = 1 / (pdf_omega G)`). `F W` is the plain path-tracing estimate `beta Lo` for a fresh candidate, and the ratio `G_d / G_s` between two pixels is exactly the
 * Jacobian of the solid-angle <-> area change of variables of Ouyang et al. (their `|cos phi2'| |x2 - x1|^2 / (|cos phi2| |x2 - x1'|^2)`).
 *
 * **Shift validity.** `Lo` is fixed (computed once toward the original x1), so it is exact for a Lambert x2 only; the sample records whether x2 is Lambert (`diffuse`).
 * A sample can be re-evaluated at another pixel only if x2 is Lambert, both pixels see x2 from its front side, and both distances are at least `minDistance`
 * (a tiny distance makes G explode); otherwise its target at any other pixel is 0 and it only lives at its own pixel. The rule is symmetric in the two pixels, so the
 * balance-heuristic MIS of the unbiased mode stays a valid partition of unity.
 *
 * **Reservoir storage** (2 vec4u per pixel per region): a = x2 (3 f32), n2 octahedral; b = Lo.rg as f16 / s, W, Lo.b as f16 / s + M (8 bit) + flags, s = max(Lo) (f32).
 */

export interface PtRestirGiOptions {
  /** Merge the previous frame's final GI reservoir (reprojected through the previous camera; static scene). Default true. */
  temporal?: boolean;
  /** Spatial neighbours merged per pixel (0 = off, at most 4: the unbiased mode costs one shadow ray per ordered pair of domains). Default 1. */
  spatial?: number;
  /** Radius of the spatial disc in pixels. Default 16. */
  spatialRadius?: number;
  /** `'biased'` (`W = wsum / (phat sum M)`, one shadow ray per reused sample) or `'unbiased'` (balance-heuristic MIS over the domains, one shadow ray per ordered pair; D11, both built). Default `'biased'`. */
  bias?: 'biased' | 'unbiased';
  /** Cap on the stored confidence M. Default 20. */
  historyCap?: number;
  /** Neighbours and history must have a shading normal within this cosine of ours. Default 0.9. */
  normalCos?: number;
  /** ... and a view depth within this fraction of ours. Default 0.1. */
  depthRel?: number;
  /** A sample whose x2 is closer than this (metres) to its pixel's x1 is not reused at other pixels (G = cos / r^2 explodes). Default 0.05. */
  minDistance?: number;
  /** Spatial neighbour offsets `'random'` (per pixel and frame) or `'fixed'` (one spiral). Default `'random'`. */
  neighbours?: 'random' | 'fixed';
  /** The reprojected pixel is `'random'` (stochastic bilinear) or `'nearest'`. Default `'random'`. */
  temporalPick?: 'random' | 'nearest';
}

export interface ResolvedRestirGiOptions {
  temporal: boolean;
  spatial: number;
  spatialRadius: number;
  bias: 'biased' | 'unbiased';
  historyCap: number;
  normalCos: number;
  depthRel: number;
  minDistance: number;
  neighbours: 'random' | 'fixed';
  temporalPick: 'random' | 'nearest';
}

export const PT_RESTIR_GI_MAX_SPATIAL = 4;
/** Domains of one merge: the pixel itself and its reused reservoirs. */
export const PT_RESTIR_GI_DOMAINS = PT_RESTIR_GI_MAX_SPATIAL + 1;
/** Streams of the GI passes' PCG chain, disjoint from ReSTIR DI's (`PT_RESTIR_KEY`: 0..168). */
export const PT_RESTIR_GI_KEY = { dir: 176, comp: 177, light: 178, point: 179, mix: 180, shift: 181, dir2: 182, comp2: 183, temporalSelect: 184, spatialOffset: 192, spatialSelect: 208 } as const;
/** Group 2 binding of the GI reservoirs in the ReSTIR passes' io group (DI uses 0-3). */
export const PT_RESTIR_GI_BINDING = 4;
/** Binding of the final GI region in the megakernel's io group (`restir.gi`). */
export const PT_RESTIR_GI_SHADE_BINDING = 19;
export const PT_RESTIR_GI_ENTRIES = ['ptRestirGiInit', 'ptRestirGiTemporal', 'ptRestirGiSpatial'] as const;
/** vec4u per pixel per region. */
export const PT_RESTIR_GI_VEC4 = 2;
export const PT_RESTIR_GI_FLAG_DIFFUSE = 1;
export const PT_RESTIR_GI_MAX_M = 255;

export function resolveRestirGiOptions(o: PtRestirGiOptions = {}): ResolvedRestirGiOptions {
  return {
    temporal: o.temporal ?? true,
    spatial: Math.max(0, Math.min(PT_RESTIR_GI_MAX_SPATIAL, Math.floor(o.spatial ?? 1))),
    spatialRadius: o.spatialRadius ?? 16,
    bias: o.bias ?? 'biased',
    historyCap: Math.max(1, Math.min(PT_RESTIR_GI_MAX_M, o.historyCap ?? 20)),
    normalCos: o.normalCos ?? 0.9,
    depthRel: o.depthRel ?? 0.1,
    minDistance: o.minDistance ?? 0.05,
    neighbours: o.neighbours ?? 'random',
    temporalPick: o.temporalPick ?? 'random',
  };
}

/** Why GI cannot be compiled for these resolved integrator options (null = it can). `o` is a `ResolvedIntegratorOptions` (structural, no import cycle). */
export function ptRestirGiUnsupported(o: { restir: boolean; restirGi: boolean; nee: boolean; textures: boolean; alphaMask: boolean; lod: string; envSampling: string; hooks: unknown; materials: { features: { aniso?: boolean; fuzz?: boolean; thinFilm?: boolean } } | null }): string | null {
  if (!o.restirGi) return 'integrator.restir.gi is not set';
  if (!o.restir) return 'ReSTIR GI needs the ReSTIR DI passes (integrator.restir)';
  if (o.textures) return 'ReSTIR GI does not support textured scenes yet (the secondary vertex needs the texture footprint of its cone)';
  if (o.alphaMask) return 'ReSTIR GI does not support alpha MASK yet (the secondary ray would need the cutout loop)';
  if (o.lod !== 'off' && o.materials) return 'ReSTIR GI does not support materialLod with OpenPBR materials (the secondary vertex would be the reduced one)';
  if (o.envSampling === 'none') return "ReSTIR GI ends the path at the first vertex: the environment must be light-sampled (envSampling 'uniform' or 'importance')";
  return null;
}

/** What the GI passes need from the DI module they are appended to. */
export interface PtRestirGiContext {
  /** `ptBsdfEval` or `ptEval` (layered albedo). */
  evalFn: string;
  /** `ptBsdfSample` or `ptSample`. */
  sampleFn: string;
  /** `PT_MIS_WGSL` + `PT_ESCAPED_WGSL` (ptMis, ptMax3, ptEscaped): the secondary vertex's own MIS between its light sample and its BSDF sample, as the shade's. */
  prelude: string;
  /** The second argument of `ptBsdfFromMaterial` at the secondary vertex: '' (legacy materials), ', 0.0' (OpenPBR without `mix`) or a random draw (OpenPBR with `mix`). */
  mixArg: string;
}

/** Reservoir decode shared by the GI passes and the shade pass. Needs `ptRsOctDecode`. */
export const PT_RESTIR_GI_DECODE_WGSL = /* wgsl */ `struct PtGiRes {
    x2: vec3f,
    n2: vec3f,
    lo: vec3f,
    w: f32,
    m: f32,
    diffuse: bool,
}

fn ptGiDecode(a: vec4u, b: vec4u) -> PtGiRes {
    var r: PtGiRes;
    r.x2 = vec3f(bitcast<f32>(a.x), bitcast<f32>(a.y), bitcast<f32>(a.z));
    r.n2 = ptRsOctDecode(a.w);
    r.lo = vec3f(unpack2x16float(b.x), unpack2x16float(b.z & 0xffffu).x) * bitcast<f32>(b.w);
    r.w = bitcast<f32>(b.y);
    r.m = f32((b.z >> 16u) & 0xffu);
    r.diffuse = ((b.z >> 24u) & ${PT_RESTIR_GI_FLAG_DIFFUSE}u) != 0u;
    return r;
}`;

/** The passes (appended to the DI module's text after `ptRestirSpatial`). */
export function ptRestirGiWgsl(ctx: PtRestirGiContext): string {
  const { evalFn, sampleFn, mixArg, prelude } = ctx;
  const D = PT_RESTIR_GI_DOMAINS;
  const K = PT_RESTIR_GI_KEY;
  return /* wgsl */ `
@group(2) @binding(${PT_RESTIR_GI_BINDING}) var<storage, read_write> ptGiResv: array<vec4u>;

const PT_GI_KEY_DIR: u32 = ${K.dir}u;
const PT_GI_KEY_COMP: u32 = ${K.comp}u;
const PT_GI_KEY_LIGHT: u32 = ${K.light}u;
const PT_GI_KEY_POINT: u32 = ${K.point}u;
const PT_GI_KEY_MIX: u32 = ${K.mix}u;
const PT_GI_KEY_DIR2: u32 = ${K.dir2}u;
const PT_GI_KEY_COMP2: u32 = ${K.comp2}u;
const PT_GI_KEY_SHIFT: u32 = ${K.shift}u;
const PT_GI_KEY_TEMPORAL: u32 = ${K.temporalSelect}u;
const PT_GI_KEY_OFFSET: u32 = ${K.spatialOffset}u;
const PT_GI_KEY_SPATIAL: u32 = ${K.spatialSelect}u;

${prelude}

${PT_RESTIR_GI_DECODE_WGSL}

fn ptGiEmpty(m: f32) -> PtGiRes {
    return PtGiRes(vec3f(0.0), vec3f(0.0, 0.0, 1.0), vec3f(0.0), 0.0, m, false);
}

fn ptGiLoad(region: u32, pix: u32) -> PtGiRes {
    let npx = ptRsParams.dims.x * ptRsParams.dims.y;
    let i = (region * npx + pix) * ${PT_RESTIR_GI_VEC4}u;
    return ptGiDecode(ptGiResv[i], ptGiResv[i + 1u]);
}

fn ptGiStore(region: u32, pix: u32, r: PtGiRes) {
    let npx = ptRsParams.dims.x * ptRsParams.dims.y;
    let i = (region * npx + pix) * ${PT_RESTIR_GI_VEC4}u;
    let s = max(r.lo.x, max(r.lo.y, r.lo.z));
    let sc = select(1.0, s, s > 0.0);
    let q = r.lo / sc;
    let m = min(u32(r.m), ${PT_RESTIR_GI_MAX_M}u);
    let flags = select(0u, ${PT_RESTIR_GI_FLAG_DIFFUSE}u, r.diffuse);
    ptGiResv[i] = vec4u(bitcast<u32>(r.x2.x), bitcast<u32>(r.x2.y), bitcast<u32>(r.x2.z), ptRsOctEncode(r.n2));
    ptGiResv[i + 1u] = vec4u(pack2x16float(q.xy), bitcast<u32>(r.w), (pack2x16float(vec2f(q.z, 0.0)) & 0xffffu) | (m << 16u) | (flags << 24u), bitcast<u32>(sc));
}

/** F_d(y) = f cos Lo G for the pixel (g, bp), RGB, times visibility when \`vis\`. */
fn ptGiTarget(g: PtRsGeom, bp: PtBsdfParams, y: PtGiRes, vis: bool) -> vec3f {
    let d = y.x2 - g.p;
    let r2 = dot(d, d);
    if (!(r2 > 0.0)) {
        return vec3f(0.0);
    }
    let w = d * inverseSqrt(r2);
    let c2 = dot(y.n2, -w);
    if (!(c2 > 0.0)) {
        return vec3f(0.0);
    }
    let frame = ptFrameFromNormal(g.n);
    let wo = ptToLocal(frame, normalize(g.eye - g.p));
    let wl = ptToLocal(frame, w);
    let f = ${evalFn}(bp, wo, wl).f * abs(wl.z);
    if (!(max(f.x, max(f.y, f.z)) > 0.0)) {
        return vec3f(0.0);
    }
    if (vis) {
        let origin = ptOffsetOrigin(g.p, ptSideNormal(g.ng, w));
        let end = ptOffsetOrigin(y.x2, ptSideNormal(y.n2, -w));
        if (!rtVisible(rtSegment(origin, end, 0.0, 0.0), RT_MASK_ALL)) {
            return vec3f(0.0);
        }
    }
    return f * y.lo * (c2 / r2);
}

/**
 * phat of the sample of domain \`own\` evaluated at domain \`at\`. Its own domain: no visibility ray (visible by construction). Another domain: only a Lambert x2 seen from its front side
 * from two points at least \`rmin\` away is shiftable; the rest has no support there.
 */
fn ptGiPhat(gAt: PtRsGeom, bpAt: PtBsdfParams, y: PtGiRes, gOwn: PtRsGeom, same: bool, rmin: f32) -> f32 {
    if (!(y.w > 0.0)) {
        return 0.0;
    }
    if (same) {
        return ptRsLum(ptGiTarget(gAt, bpAt, y, false));
    }
    if (!y.diffuse || distance(y.x2, gOwn.p) < rmin || distance(y.x2, gAt.p) < rmin) {
        return 0.0;
    }
    return ptRsLum(ptGiTarget(gAt, bpAt, y, true));
}

fn ptGiMerge(n: u32, gs: ptr<function, array<PtRsGeom, ${D}>>, bps: ptr<function, array<PtBsdfParams, ${D}>>, ys: ptr<function, array<PtGiRes, ${D}>>, unbiased: bool, seed: u32, frame: u32, cap: f32, key: u32, rmin: f32) -> PtGiRes {
    var ph: array<f32, ${D * D}>;
    for (var i = 0u; i < n; i++) {
        let y = (*ys)[i];
        ph[i * ${D}u] = ptGiPhat((*gs)[0], (*bps)[0], y, (*gs)[i], i == 0u, rmin);
    }
    if (unbiased) {
        for (var i = 0u; i < n; i++) {
            let y = (*ys)[i];
            for (var j = 1u; j < n; j++) {
                ph[i * ${D}u + j] = ptGiPhat((*gs)[j], (*bps)[j], y, (*gs)[i], i == j, rmin);
            }
        }
    }
    var out = ptGiEmpty(0.0);
    var wsum = 0.0;
    var mtot = 0.0;
    var selPh = 0.0;
    for (var i = 0u; i < n; i++) {
        let y = (*ys)[i];
        mtot += y.m;
        let p0 = ph[i * ${D}u];
        if (!(p0 > 0.0)) {
            continue;
        }
        var w = p0 * y.w * y.m;
        if (unbiased) {
            var den = 0.0;
            for (var j = 0u; j < n; j++) {
                let mj = (*ys)[j].m;
                if (mj > 0.0) {
                    den += mj * ph[i * ${D}u + j];
                }
            }
            w = select(0.0, (y.m * ph[i * ${D}u + i] / den) * p0 * y.w, den > 0.0);
        }
        wsum += w;
        if (ptRsRand2(seed, frame, key + i).x * wsum < w) {
            out = y;
            selPh = p0;
        }
    }
    if (selPh > 0.0 && wsum > 0.0) {
        out.w = select(wsum / (mtot * selPh), wsum / selPh, unbiased);
    } else {
        out.w = 0.0;
    }
    out.m = min(mtot, cap);
    return out;
}

@compute @workgroup_size(${8}, ${8})
fn ptRestirGiInit(@builtin(global_invocation_id) id: vec3u) {
    let P = ptRsParams;
    if (id.x >= P.dims.x || id.y >= P.dims.y) {
        return;
    }
    let pix = id.y * P.dims.x + id.x;
    let g = ptRsLoadGeom(P.reg.x, pix, false);
    if (!(g.depth > 0.0)) {
        ptGiStore(P.reg.w, pix, ptGiEmpty(0.0));
        return;
    }
    let bp = ptRsBsdf(g);
    if (ptBsdfIsDelta(bp) || bp.kind == PT_BSDF_BLACK) {
        ptGiStore(P.reg.w, pix, ptGiEmpty(0.0));
        return;
    }
    let seed = ptPixelSeed(id.x, id.y, P.dims.w);
    let frame1 = ptFrameFromNormal(g.n);
    let wo = ptToLocal(frame1, normalize(g.eye - g.p));
    let ud = ptRsRand2(seed, P.dims.z, PT_GI_KEY_DIR);
    let uc = ptRsRand2(seed, P.dims.z, PT_GI_KEY_COMP).x;
    let s = ${sampleFn}(bp, wo, vec3f(uc, ud.x, ud.y));
    if (s.flags == 0u || !(s.pdf > 0.0)) {
        ptGiStore(P.reg.w, pix, ptGiEmpty(1.0));
        return;
    }
    let wi = ptFromLocal(frame1, s.wi);
    let ray = RtRay(ptOffsetOrigin(g.p, ptSideNormal(g.ng, wi)), 0.0, wi, PT_FAR);
    let hit = rtClosest(ray, RT_MASK_ALL);
    if (hit.t < 0.0) {
        ptGiStore(P.reg.w, pix, ptGiEmpty(1.0));
        return;
    }
    let surf = rtSurface(hit, ray, RT_SURF_POSITION | RT_SURF_SHADING_NORMAL);
    let mat = rtMaterial(hit.material);
    let bp2 = ptBsdfFromMaterial(mat${mixArg});
    if (bp2.kind == PT_BSDF_BLACK || ptBsdfIsDelta(bp2)) {
        ptGiStore(P.reg.w, pix, ptGiEmpty(1.0));
        return;
    }
    // x2 sees x1 from its front side: the geometric normal facing the ray.
    let ng = select(-hit.normal, hit.normal, dot(hit.normal, ray.direction) < 0.0);
    let c2 = dot(ng, -ray.direction);
    let r2 = dot(surf.position - g.p, surf.position - g.p);
    // Lo: the direct light x2 reflects toward x1, as the path tracer's depth-2 vertex does it: one light sample (MIS-weighted against the BSDF) plus one BSDF sample
    // whose emission / sky is MIS-weighted against the light sampler. Emission seen at x2 itself is DI's (the first-vertex lights).
    var lo = vec3f(0.0);
    let frame2 = ptFrameFromNormal(surf.shadingNormal);
    let wo2 = ptToLocal(frame2, -ray.direction);
    let a = ptRsRand2(seed, P.dims.z, PT_GI_KEY_LIGHT);
    let pick = rtSampleLight(a.x);
    if (pick.index != RT_NONE && pick.pmf > 0.0 && (PT_ENV_SAMPLING || pick.index != rtSceneTable[2].z)) {
        let u = ptRsRand2(seed, P.dims.z, PT_GI_KEY_POINT);
        let li = rtSampleLightLi(pick.index, surf.position, u);
        if (li.valid && li.pdf > 0.0) {
            let wl = ptToLocal(frame2, li.wi);
            let e2 = ${evalFn}(bp2, wo2, wl);
            let f2 = e2.f * abs(wl.z);
            if (max(f2.x, max(f2.y, f2.z)) > 0.0) {
                let origin = ptOffsetOriginErr(surf.position, ptSideNormal(hit.normal, li.wi), surf.positionError);
                var lray: RtRay;
                if (li.distance < 1.0e30) {
                    let end = ptOffsetOrigin(li.point, ptSideNormal(li.normal, -li.wi));
                    lray = rtSegment(origin, end, 0.0, 0.0);
                } else {
                    lray = RtRay(origin, 0.0, li.wi, PT_FAR);
                }
                if (rtVisible(lray, RT_MASK_ALL)) {
                    let pl = pick.pmf * li.pdf;
                    lo = li.radiance * f2 * (select(ptMis(pl, e2.pdf), 1.0, li.delta) / pl);
                }
            }
        }
    }
    let ud2 = ptRsRand2(seed, P.dims.z, PT_GI_KEY_DIR2);
    let uc2 = ptRsRand2(seed, P.dims.z, PT_GI_KEY_COMP2).x;
    let s2 = ${sampleFn}(bp2, wo2, vec3f(uc2, ud2.x, ud2.y));
    if (s2.flags != 0u && s2.pdf > 0.0) {
        let wi2 = ptFromLocal(frame2, s2.wi);
        let ray2 = RtRay(ptOffsetOriginErr(surf.position, ptSideNormal(hit.normal, wi2), surf.positionError), 0.0, wi2, PT_FAR);
        let hit2 = rtClosest(ray2, RT_MASK_ALL);
        let beta2 = s2.f * (abs(s2.wi.z) / s2.pdf);
        if (hit2.t < 0.0) {
            lo += beta2 * ptEscaped(wi2, false, s2.pdf);
        } else {
            let m3 = rtMaterial(hit2.material);
            if (m3.emissionLuminance > 0.0 && (hit2.frontFace || m3.emissionTwoSided)) {
                var wm = 1.0;
                if (PT_NEE) {
                    let lidx = rtLightIndexOfHit(hit2.instance, hit2.triangle);
                    if (lidx != RT_NONE) {
                        let light = rtLight(lidx);
                        let p3 = rtSurface(hit2, ray2, RT_SURF_POSITION).position;
                        wm = ptMis(s2.pdf, light.pmf * rtTriangleSolidAnglePdf(light.area, length(p3 - surf.position), abs(dot(hit2.normal, wi2))));
                    }
                }
                lo += beta2 * m3.emissionColor * wm;
            }
        }
    }
    var out = ptGiEmpty(1.0);
    out.x2 = surf.position;
    out.n2 = ng;
    out.lo = lo;
    out.diffuse = bp2.kind == PT_BSDF_DIFFUSE;
    // phat of the candidate at its own pixel, W = 1 / (pdf_omega G): F W = beta Lo, the path-tracing estimate.
    let beta = s.f * (abs(s.wi.z) / s.pdf);
    let ph = ptRsLum(beta * lo);
    if (ph > 0.0 && c2 > 1.0e-6) {
        out.w = r2 / (s.pdf * c2);
    }
    ptGiStore(P.reg.w, pix, out);
}

@compute @workgroup_size(${8}, ${8})
fn ptRestirGiTemporal(@builtin(global_invocation_id) id: vec3u) {
    let P = ptRsParams;
    if (id.x >= P.dims.x || id.y >= P.dims.y) {
        return;
    }
    let pix = id.y * P.dims.x + id.x;
    var gs: array<PtRsGeom, ${D}>;
    var bps: array<PtBsdfParams, ${D}>;
    var ys: array<PtGiRes, ${D}>;
    gs[0] = ptRsLoadGeom(P.reg.x, pix, false);
    ys[0] = ptGiLoad(P.reg.z, pix);
    if (!(gs[0].depth > 0.0) || !(ys[0].m > 0.0)) {
        ptGiStore(P.reg.w, pix, ptGiEmpty(0.0));
        return;
    }
    bps[0] = ptRsBsdf(gs[0]);
    let seed = ptPixelSeed(id.x, id.y, P.dims.w);
    let cap = P.thr.w;
    let rmin = bitcast<f32>(P.cfg.w);
    var n = 1u;
    if ((P.cfg.z & PT_RS_FLAG_HISTORY) != 0u) {
        let d = gs[0].p - P.prevOrigin.xyz;
        let a = dot(d, P.prevForward.xyz);
        if (a > 1.0e-4) {
            let nx = dot(d, P.prevRight.xyz) / dot(P.prevRight.xyz, P.prevRight.xyz) / a;
            let ny = dot(d, P.prevUp.xyz) / dot(P.prevUp.xyz, P.prevUp.xyz) / a;
            var pf = vec2f((nx * 0.5 + 0.5) * f32(P.dims.x), (0.5 - ny * 0.5) * f32(P.dims.y));
            if ((P.cfg.z & PT_RS_FLAG_RAND_TEMP) != 0u) {
                pf += ptRsRand2(seed, P.dims.z, PT_GI_KEY_SHIFT) - vec2f(0.5);
            }
            let q = vec2i(floor(pf));
            if (all(q >= vec2i(0)) && all(q < vec2i(P.dims.xy))) {
                let qpix = u32(q.y) * P.dims.x + u32(q.x);
                let gp = ptRsLoadGeom(P.reg.y, qpix, true);
                if (gp.depth > 0.0 && dot(gs[0].n, gp.n) >= P.thr.y && abs(a - gp.depth) <= P.thr.z * a) {
                    var rp = ptGiLoad(P.reg2.x, qpix);
                    rp.m = min(rp.m, cap);
                    if (rp.m > 0.0) {
                        gs[1] = gp;
                        bps[1] = ptRsBsdf(gp);
                        ys[1] = rp;
                        n = 2u;
                    }
                }
            }
        }
    }
    ptGiStore(P.reg.w, pix, ptGiMerge(n, &gs, &bps, &ys, (P.cfg.z & PT_RS_FLAG_UNBIASED) != 0u, seed, P.dims.z, cap, PT_GI_KEY_TEMPORAL, rmin));
}

@compute @workgroup_size(${8}, ${8})
fn ptRestirGiSpatial(@builtin(global_invocation_id) id: vec3u) {
    let P = ptRsParams;
    if (id.x >= P.dims.x || id.y >= P.dims.y) {
        return;
    }
    let pix = id.y * P.dims.x + id.x;
    var gs: array<PtRsGeom, ${D}>;
    var bps: array<PtBsdfParams, ${D}>;
    var ys: array<PtGiRes, ${D}>;
    gs[0] = ptRsLoadGeom(P.reg.x, pix, false);
    ys[0] = ptGiLoad(P.reg.z, pix);
    if (!(gs[0].depth > 0.0) || !(ys[0].m > 0.0)) {
        ptGiStore(P.reg.w, pix, ptGiEmpty(0.0));
        return;
    }
    bps[0] = ptRsBsdf(gs[0]);
    let seed = ptPixelSeed(id.x, id.y, P.dims.w);
    let cap = P.thr.w;
    let rmin = bitcast<f32>(P.cfg.w);
    let k = min(P.cfg.y, ${PT_RESTIR_GI_MAX_SPATIAL}u);
    var n = 1u;
    for (var j = 0u; j < k; j++) {
        var off: vec2f;
        if ((P.cfg.z & PT_RS_FLAG_RAND_NB) != 0u) {
            let o = ptRsRand2(seed, P.dims.z, PT_GI_KEY_OFFSET + j);
            let r = P.thr.x * sqrt(o.x);
            let ang = 6.283185307179586 * o.y;
            off = vec2f(cos(ang), sin(ang)) * r;
        } else {
            let r = P.thr.x * sqrt((f32(j) + 0.5) / f32(k));
            let ang = f32(j) * 2.399963229728653;
            off = vec2f(cos(ang), sin(ang)) * r;
        }
        let q = vec2i(id.xy) + vec2i(round(off));
        if (all(q == vec2i(id.xy)) || any(q < vec2i(0)) || any(q >= vec2i(P.dims.xy))) {
            continue;
        }
        let qpix = u32(q.y) * P.dims.x + u32(q.x);
        let gq = ptRsLoadGeom(P.reg.x, qpix, false);
        if (!ptRsSurfaceOk(gs[0], gq)) {
            continue;
        }
        var yq = ptGiLoad(P.reg.z, qpix);
        yq.m = min(yq.m, cap);
        if (!(yq.m > 0.0)) {
            continue;
        }
        gs[n] = gq;
        bps[n] = ptRsBsdf(gq);
        ys[n] = yq;
        n += 1u;
    }
    ptGiStore(P.reg.w, pix, ptGiMerge(n, &gs, &bps, &ys, (P.cfg.z & PT_RS_FLAG_UNBIASED) != 0u, seed, P.dims.z, cap, PT_GI_KEY_SPATIAL, rmin));
}
`;
}

/**
 * Shade side (the megakernel, `integrator.restir.gi`): the final GI reservoir of the pixel at the first (non-delta) vertex. Needs `ptRsOctDecode`.
 * `ptGiDist` is the distance to x2 (the frame planes' bounce hit distance).
 */
/** `base`: the WGSL expression of the first final-reservoir index (the megakernel's `ptParams.limits.w`; the wavefront carries it in `up.w`, P5 wavefront GI). */
export function ptRestirGiShadeWgsl(evalFn: string, base = 'ptParams.limits.w'): string {
  return /* wgsl */ `
${PT_RESTIR_GI_DECODE_WGSL}

var<private> ptGiDist: f32 = 0.0;

fn ptRestirGiLi(p: vec3f, frame: PtFrame, wo: vec3f, bp: PtBsdfParams, pxIndex: u32) -> vec3f {
    let i = (${base} + pxIndex) * ${PT_RESTIR_GI_VEC4}u;
    let y = ptGiDecode(ptRestirGiRes[i], ptRestirGiRes[i + 1u]);
    if (!(y.w > 0.0)) {
        return vec3f(0.0);
    }
    let d = y.x2 - p;
    let r2 = dot(d, d);
    let w = d * inverseSqrt(r2);
    let c2 = dot(y.n2, -w);
    if (!(c2 > 0.0)) {
        return vec3f(0.0);
    }
    let wl = ptToLocal(frame, w);
    let f = ${evalFn}(bp, wo, wl).f * abs(wl.z);
    ptGiDist = sqrt(r2);
    return f * y.lo * (y.w * c2 / r2);
}
`;
}

