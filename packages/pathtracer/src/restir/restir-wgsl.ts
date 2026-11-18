/**
 * ReSTIR DI (P4, real-time mode; reference mode stays plain NEE + MIS): WGSL of the four passes that run before the megakernel's shade
 * dispatch. Written from the papers (Bitterli et al. 2020 "Spatiotemporal reservoir resampling"; Lin et al. 2022 "Generalized resampled
 * importance sampling" for the unbiased Talbot-MIS combine); no code from RTXDI or any other renderer.
 *
 *     ptRestirGbuf      camera ray (the same jittered ray the shade pass will trace) -> position, view depth, shading normal, material
 *     ptRestirInit      RIS: M candidates from the light sampler -> one reservoir; optional visibility of the chosen sample
 *     ptRestirTemporal  reproject the primary hit into the previous frame, merge its final reservoir
 *     ptRestirSpatial   merge the reservoirs of k neighbours (random offsets inside a disc, R6)
 *
 * The shade pass is the megakernel itself (`integrator.restir`, integrator-wgsl.ts): at the first scattering vertex it reads the final reservoir,
 * regenerates the light sample, traces one shadow ray and adds `f cos Le / pdf * W`; the BSDF-sampled continuation gets no MIS weight on
 * emission at depth 1 (the reservoir already covers the light), deeper vertices keep NEE + MIS.
 *
 * **Sample space.** A reservoir sample is (light index, u) with u in [0, 1)^2 quantised to 16 bits per axis: the light point / direction is a
 * function of (light, u) and of the shading point only through the solid-angle density, so the same sample can be re-evaluated at any other
 * pixel (identity shift, Jacobian 1 in u-space). The target function is the unshadowed integrand per unit of u:
 * `F = f(wo, wi) |cos| Le / pdf_omega(wi)` and `phat = luminance(F)`; the source density in u-space is the light-selection pmf. For M = 1 the
 * estimate `F W` with `W = 1 / pmf` is exactly plain NEE without MIS.
 *
 * **Bias correction** (runtime flag): `biased` = Bitterli Alg. 5 (`W = wsum / (phat * sum M)`); `unbiased` = balance-heuristic ("Talbot") MIS over
 * the contributing domains, `m_i(y) = M_i phat_i(y) / sum_j M_j phat_j(y)` (every phat evaluated on the surface of its own pixel), `W = wsum / phat`.
 * Unbiased needs `initialVisibility` off (a zeroed occluded sample changes the target).
 *
 * Storage: G-buffer = 2 vec4f per pixel per frame (position + view depth, shading normal + material), two frames (cur / prev); reservoirs =
 * one vec4u per pixel, four regions (A init, B temporal, F0 / F1 final, ping-pong by frame parity). Reservoir words: light index, u (2 x unorm16),
 * bits(W), M. Needs `integrator.nee`, no textures, legacy materials (`materials: null`), no alpha mask.
 */
import { PT_VISIBLE_MASKED_WGSL, ptCommonHeadWgsl, resolveIntegratorOptions, type PtIntegratorOptions } from '../integrator/integrator-wgsl.ts';

export interface PtRestirOptions {
  /** RIS candidates per pixel (M). Default 8. 1 with temporal and spatial off equals plain NEE (no MIS) in distribution. */
  candidates?: number;
  /** Merge the previous frame's final reservoir (reprojected through the previous camera; static scene). Default true. */
  temporal?: boolean;
  /** Spatial neighbours merged per pixel (0 = off, at most 8). Default 2 (D32; was 1). */
  spatial?: number;
  /** Radius of the spatial disc in pixels. Default 16. */
  spatialRadius?: number;
  /** `'biased'` (1 / sum M) or `'unbiased'` (balance-heuristic MIS; build both, D11). Default `'unbiased'` (D32; was `'biased'`: it darkens 4-17% on the A/B scenes, REALTIME-PIPELINE.md). */
  bias?: 'biased' | 'unbiased';
  /** Trace a shadow ray for the chosen candidate and zero its weight when occluded (visibility reuse; biased in `unbiased` mode). Default true for `biased`, false for `unbiased` (the D32 default). */
  initialVisibility?: boolean;
  /** Cap on the stored confidence M, as a multiple of `candidates` (Bitterli: 20). Default 20. */
  historyCap?: number;
  /** R6: spatial neighbour offsets `'random'` (per pixel and frame) or `'fixed'` (one spiral pattern for every pixel and frame: the structured baseline). Default `'random'`. */
  neighbours?: 'random' | 'fixed';
  /** R6: the reprojected pixel is `'random'` (stochastic bilinear: nearest of the position plus a +-0.5 px random shift) or `'nearest'`. Default `'random'`. */
  temporalPick?: 'random' | 'nearest';
  /** Neighbours and history must have a shading normal within this cosine of ours. Default 0.9. */
  normalCos?: number;
  /** ... and a view depth within this fraction of ours. Default 0.1. */
  depthRel?: number;
  /**
   * Trace-mode primaries only (P4.2c): the G-buffer pass traces the camera ray and writes the hit as a V-buffer texel (`vbuffer.ts` format, region 4 of the reservoir buffer) that the shade
   * kernel resolves instead of tracing its own primary ray (one camera-ray trace per frame instead of two; needs `layout.positions`; the hit position then comes from unorm16 barycentrics,
   * DIVERGES V171). Ignored with `primary: 'vbuffer'` (the engine's V-buffer is already shared). Default false.
   */
  sharedHit?: boolean;
}

export interface ResolvedRestirOptions {
  candidates: number;
  temporal: boolean;
  spatial: number;
  spatialRadius: number;
  bias: 'biased' | 'unbiased';
  initialVisibility: boolean;
  historyCap: number;
  neighbours: 'random' | 'fixed';
  temporalPick: 'random' | 'nearest';
  normalCos: number;
  depthRel: number;
  sharedHit: boolean;
}

export const PT_RESTIR_MAX_CANDIDATES = 64;
/**
 * Streams of the passes' PCG hash chain (`ptRsRand2(seed, frame, key)`): every random number a pixel draws in one frame has its OWN key. Candidates use `2c`, `2c + 1` (c < 64: keys 0..127).
 * The temporal and the spatial merge once shared the key base 40 (and the sub-pixel shift key 7 collided with candidate 3's u), so the selection of the spatial merge was correlated with the
 * temporal merge's: with samples of very different weights (a stochastic material pick that changes every frame) the selection probabilities were not proportional to the weights, a darkening bias.
 */
export const PT_RESTIR_KEY = { shift: 128, temporalSelect: 132, spatialOffset: 144, spatialSelect: 160 } as const;
export const PT_RESTIR_MAX_SPATIAL = 8;

export function resolveRestirOptions(o: PtRestirOptions = {}): ResolvedRestirOptions {
  const bias = o.bias ?? 'unbiased';
  return {
    candidates: Math.max(1, Math.min(PT_RESTIR_MAX_CANDIDATES, Math.floor(o.candidates ?? 8))),
    temporal: o.temporal ?? true,
    spatial: Math.max(0, Math.min(PT_RESTIR_MAX_SPATIAL, Math.floor(o.spatial ?? 2))),
    spatialRadius: o.spatialRadius ?? 16,
    bias,
    initialVisibility: o.initialVisibility ?? bias === 'biased',
    historyCap: o.historyCap ?? 20,
    neighbours: o.neighbours ?? 'random',
    temporalPick: o.temporalPick ?? 'random',
    normalCos: o.normalCos ?? 0.9,
    depthRel: o.depthRel ?? 0.1,
    sharedHit: o.sharedHit ?? false,
  };
}


/** Bytes of one `PtRsParams` record (13 vec4). The ring slot is `minUniformBufferOffsetAlignment`. */
export const PT_RESTIR_PARAMS_BYTES = 208;
/** Flag bits of `PtRsParams.cfg.z`. */
export const PT_RESTIR_FLAG = { history: 1, unbiased: 2, initialVisibility: 4, randomNeighbours: 8, randomTemporal: 16 } as const;
/** Reservoir regions (vec4u per pixel each): init, temporal, final parity 0, final parity 1. */
export const PT_RESTIR_REGION = { init: 0, temporal: 1, final0: 2, final1: 3 } as const;
export const PT_RESTIR_REGIONS = 4;
/**
 * Group 2 bindings of the ReSTIR passes (`hit`: with `sharedHit`, the primary hit of the frame as one V-buffer texel per pixel, `vbuffer.ts` format; the V-buffer texture and the blue-noise
 * texture keep the numbers the megakernel gives them: `PT_VBUFFER_BINDING` 9, `PT_BLUENOISE_BINDING` 12).
 */
export const PT_RESTIR_BINDING = { params: 0, gbuf: 1, reservoirs: 2, hit: 3 } as const;
/** Binding of the final reservoir region in the megakernel's io group (`integrator.restir`). */
export const PT_RESTIR_SHADE_BINDING = 11;
/** Binding of the shared primary hit buffer in the megakernel's io group (`restir.sharedHit`). */
export const PT_RESTIR_HIT_BINDING = 15;
export const PT_RESTIR_ENTRIES = ['ptRestirGbuf', 'ptRestirInit', 'ptRestirTemporal', 'ptRestirSpatial'] as const;
/** G-buffer `material` word of a pixel whose camera ray hit nothing (a real material is a bvh index below 2^16, or `RT_NO_MATERIAL` = 0xffffffff). */
export const PT_RESTIR_GBUF_MISS = 0xfffffffe;

/** The words of a G-buffer record for these integrator options (P4.2b/c): position, octahedral normal, material, then the texture footprint and a per-vertex random for stochastic material picks when the scene needs them. */
export function ptRestirGbufLayout(integrator: PtIntegratorOptions = {}): { words: number; uv: number | null; mix: number | null } {
  const o = resolveIntegratorOptions(integrator);
  const mats = o.materials;
  const needMix = !!(mats?.features.mix || (o.lod !== 'off' && mats));
  let words = 5; // position x y z (f32 bits), octahedral shading normal (2 x unorm16), material
  let uv: number | null = null;
  let mix: number | null = null;
  if (o.textures) {
    uv = words; // uv.x, uv.y, footprint (f32 bits)
    words += 3;
  }
  if (needMix) {
    mix = words;
    words += 1;
  }
  return { words, uv, mix };
}

/** Bytes of the two G-buffer frames at `pixels` pixels (before P4.2c this was 2 x 32 B per pixel). */
export const ptRestirGbufBytes = (integrator: PtIntegratorOptions, pixels: number): number => 2 * pixels * ptRestirGbufLayout(integrator).words * 4;

const f32 = (v: number): string => {
  const s = Math.fround(v).toString();
  return /[.e]/.test(s) ? s : `${s}.0`;
};

/** Reservoir decode shared by the ReSTIR module and the megakernel's shade (unorm16 pair). */
export const PT_RESTIR_UNPACK_WGSL = /* wgsl */ `fn ptRsUnpackU(w: u32) -> vec2f {
    return vec2f(f32(w & 0xffffu) + 0.5, f32(w >> 16u) + 0.5) * 0.0000152587890625;
}`;

/** Octahedral unit-vector codec (2 x unorm16 in one word; Cigolle et al. 2014, "A survey of efficient representations for independent unit vectors"). TS twin: `ptRsOctEncode` / `ptRsOctDecode` in restir-oracle.ts. */
export const PT_RESTIR_OCT_WGSL = /* wgsl */ `fn ptRsOctEncode(n: vec3f) -> u32 {
    let a = n / max(abs(n.x) + abs(n.y) + abs(n.z), 1.0e-20);
    var e = a.xy;
    if (a.z < 0.0) {
        e = (vec2f(1.0) - abs(e.yx)) * select(vec2f(-1.0), vec2f(1.0), e >= vec2f(0.0));
    }
    let q = vec2u(clamp(round((e * 0.5 + vec2f(0.5)) * 65535.0), vec2f(0.0), vec2f(65535.0)));
    return q.x | (q.y << 16u);
}

fn ptRsOctDecode(w: u32) -> vec3f {
    let e = vec2f(f32(w & 0xffffu), f32(w >> 16u)) * (2.0 / 65535.0) - vec2f(1.0);
    var n = vec3f(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    if (n.z < 0.0) {
        let xy = (vec2f(1.0) - abs(n.yx)) * select(vec2f(-1.0), vec2f(1.0), n.xy >= vec2f(0.0));
        n = vec3f(xy, n.z);
    }
    return normalize(n);
}`;

/** The ReSTIR passes, to append after the shading modules (light sampling, BSDF, adapter, sampler) of a build whose raytracer module has the walk at group 0 and the scene at group 1. */
export function ptRestirWgsl(integrator: PtIntegratorOptions = {}): string {
  const o = resolveIntegratorOptions(integrator);
  const head = ptCommonHeadWgsl(o);
  const tex = o.textures;
  const mask = o.alphaMask;
  // Reservoir samples are tested with the exact cutout test whatever the shade's alphaShadow policy (V173): the weighted policies need a per-ray weight, a reservoir sample needs a visibility.
  const maskShadow = mask;
  const vb = o.primary === 'vbuffer';
  const bn = o.sampler === 'bluenoise';
  const shared = o.restirShared;
  const mats = o.materials;
  const layered = !!mats?.features.layered;
  const evalFn = layered ? 'ptEval' : 'ptBsdfEval';
  const gl = ptRestirGbufLayout(integrator);
  const mixExpr = 'ptSample2D(pixelSeed, P.dims.z, ptVertexPair(0u, PT_SLOT_RR)).y';
  const visible = maskShadow ? 'ptVisibleMasked' : 'rtVisible';
  return /* wgsl */ `
${head}

struct PtRsParams {
    dims: vec4u,
    cfg: vec4u,
    thr: vec4f,
    reg: vec4u,
    reg2: vec4u,
    curOrigin: vec4f,
    curForward: vec4f,
    curRight: vec4f,
    curUp: vec4f,
    prevOrigin: vec4f,
    prevForward: vec4f,
    prevRight: vec4f,
    prevUp: vec4f,
}

@group(2) @binding(${PT_RESTIR_BINDING.params}) var<uniform> ptRsParams: PtRsParams;
@group(2) @binding(${PT_RESTIR_BINDING.gbuf}) var<storage, read_write> ptRsGbuf: array<u32>;
@group(2) @binding(${PT_RESTIR_BINDING.reservoirs}) var<storage, read_write> ptRsResv: array<vec4u>;

const PT_RS_FLAG_HISTORY: u32 = ${PT_RESTIR_FLAG.history}u;
const PT_RS_FLAG_UNBIASED: u32 = ${PT_RESTIR_FLAG.unbiased}u;
const PT_RS_FLAG_VIS: u32 = ${PT_RESTIR_FLAG.initialVisibility}u;
const PT_RS_FLAG_RAND_NB: u32 = ${PT_RESTIR_FLAG.randomNeighbours}u;
const PT_RS_FLAG_RAND_TEMP: u32 = ${PT_RESTIR_FLAG.randomTemporal}u;
const PT_RS_DOMAINS: u32 = ${PT_RESTIR_MAX_SPATIAL + 1}u;
const PT_RS_LUMA: vec3f = vec3f(${f32(0.2126)}, ${f32(0.7152)}, ${f32(0.0722)});
const PT_RS_KEY_SHIFT: u32 = ${PT_RESTIR_KEY.shift}u;
const PT_RS_KEY_TEMPORAL: u32 = ${PT_RESTIR_KEY.temporalSelect}u;
const PT_RS_KEY_OFFSET: u32 = ${PT_RESTIR_KEY.spatialOffset}u;
const PT_RS_KEY_SPATIAL: u32 = ${PT_RESTIR_KEY.spatialSelect}u;
const PT_RS_WORDS: u32 = ${gl.words}u;
const PT_RS_MISS: u32 = ${PT_RESTIR_GBUF_MISS}u;
${shared ? `@group(2) @binding(${PT_RESTIR_BINDING.hit}) var<storage, read_write> ptRsHit: array<vec4u>;
` : ''}
struct PtRsGeom {
    p: vec3f,
    depth: f32,
    n: vec3f,
    mat: u32,
    eye: vec3f,${tex ? `
    uv: vec2f,
    fp: f32,` : ''}${gl.mix !== null ? `
    mixU: f32,` : ''}
}

struct PtRsRes {
    light: u32,
    u: vec2f,
    w: f32,
    m: f32,
}

${PT_RESTIR_UNPACK_WGSL}

${PT_RESTIR_OCT_WGSL}

fn ptRsPackU(u: vec2f) -> u32 {
    let q = vec2u(min(floor(u * 65536.0), vec2f(65535.0)));
    return q.x | (q.y << 16u);
}

fn ptRsRand2(seed: u32, frame: u32, k: u32) -> vec2f {
    let a = ptPcgHash(seed ^ ptPcgHash(frame * 0x9e3779b1u + k * 0x85ebca6bu + 0x7f4a7c15u));
    let b = ptPcgHash(a ^ 0xb5297a4du);
    return vec2f(rtUnit24(a), rtUnit24(b));
}

fn ptRsEmpty() -> PtRsRes {
    return PtRsRes(RT_NONE, vec2f(0.5), 0.0, 0.0);
}

fn ptRsLoadRes(region: u32, pix: u32) -> PtRsRes {
    let npx = ptRsParams.dims.x * ptRsParams.dims.y;
    let r = ptRsResv[region * npx + pix];
    return PtRsRes(r.x, ptRsUnpackU(r.y), bitcast<f32>(r.z), f32(r.w));
}

fn ptRsStoreRes(region: u32, pix: u32, r: PtRsRes) {
    let npx = ptRsParams.dims.x * ptRsParams.dims.y;
    ptRsResv[region * npx + pix] = vec4u(r.light, ptRsPackU(r.u), bitcast<u32>(r.w), u32(r.m));
}

fn ptRsLoadGeom(fsel: u32, pix: u32, prev: bool) -> PtRsGeom {
    let npx = ptRsParams.dims.x * ptRsParams.dims.y;
    let i = (fsel * npx + pix) * PT_RS_WORDS;
    var g: PtRsGeom;
    g.mat = ptRsGbuf[i + 4u];
    if (g.mat == PT_RS_MISS) {
        g.depth = -1.0;
        return g;
    }
    g.p = vec3f(bitcast<f32>(ptRsGbuf[i]), bitcast<f32>(ptRsGbuf[i + 1u]), bitcast<f32>(ptRsGbuf[i + 2u]));
    g.n = ptRsOctDecode(ptRsGbuf[i + 3u]);
    g.eye = select(ptRsParams.curOrigin.xyz, ptRsParams.prevOrigin.xyz, prev);
    g.depth = dot(g.p - g.eye, select(ptRsParams.curForward.xyz, ptRsParams.prevForward.xyz, prev));${tex ? `
    g.uv = vec2f(bitcast<f32>(ptRsGbuf[i + ${gl.uv}u]), bitcast<f32>(ptRsGbuf[i + ${(gl.uv ?? 0) + 1}u]));
    g.fp = bitcast<f32>(ptRsGbuf[i + ${(gl.uv ?? 0) + 2}u]);` : ''}${gl.mix !== null ? `
    g.mixU = bitcast<f32>(ptRsGbuf[i + ${gl.mix}u]);` : ''}
    return g;
}

fn ptRsBsdf(g: PtRsGeom) -> PtBsdfParams {
${tex ? `    let mat = rtApplyMaterialTextures(rtMaterial(g.mat), g.uv, g.fp);
` : `    let mat = rtMaterial(g.mat);
`}    return ptBsdfFromMaterial(mat${mats ? `, ${gl.mix !== null ? 'g.mixU' : '0.0'}` : ''});
}

fn ptRsTarget(g: PtRsGeom, bp: PtBsdfParams, light: u32, u: vec2f) -> vec3f {
    if (light == RT_NONE) {
        return vec3f(0.0);
    }
    if (!PT_ENV_SAMPLING && light == rtSceneTable[2].z) {
        return vec3f(0.0);
    }
    let li = rtSampleLightLi(light, g.p, u);
    if (!li.valid || !(li.pdf > 0.0)) {
        return vec3f(0.0);
    }
    let frame = ptFrameFromNormal(g.n);
    let wo = ptToLocal(frame, normalize(g.eye - g.p));
    let wi = ptToLocal(frame, li.wi);
    let e = ${evalFn}(bp, wo, wi);
    return li.radiance * e.f * (abs(wi.z) / li.pdf);
}

fn ptRsLum(v: vec3f) -> f32 {
    let l = dot(v, PT_RS_LUMA);
    return select(0.0, l, l > 0.0 && l < 3.0e38);
}

${maskShadow ? PT_VISIBLE_MASKED_WGSL : ''}fn ptRsVisible(g: PtRsGeom, light: u32, u: vec2f) -> bool {
    let li = rtSampleLightLi(light, g.p, u);
    if (!li.valid) {
        return false;
    }
    let origin = ptOffsetOrigin(g.p, ptSideNormal(g.n, li.wi));
    var ray: RtRay;
    if (li.distance < 1.0e30) {
        let end = ptOffsetOrigin(li.point, ptSideNormal(li.normal, -li.wi));
        ray = rtSegment(origin, end, 0.0, 0.0);
    } else {
        ray = RtRay(origin, 0.0, li.wi, PT_FAR);
    }
    return ${visible}(ray, RT_MASK_ALL);
}

fn ptRsMerge(n: u32, gs: ptr<function, array<PtRsGeom, ${PT_RESTIR_MAX_SPATIAL + 1}>>, bps: ptr<function, array<PtBsdfParams, ${PT_RESTIR_MAX_SPATIAL + 1}>>, rs: ptr<function, array<PtRsRes, ${PT_RESTIR_MAX_SPATIAL + 1}>>, unbiased: bool, seed: u32, frame: u32, cap: f32, key: u32) -> PtRsRes {
    var out = ptRsEmpty();
    var wsum = 0.0;
    var mtot = 0.0;
    var selPh = 0.0;
    for (var i = 0u; i < n; i++) {
        let r = (*rs)[i];
        mtot += r.m;
        if (!(r.w > 0.0) || r.light == RT_NONE) {
            continue;
        }
        let ph = ptRsLum(ptRsTarget((*gs)[0], (*bps)[0], r.light, r.u));
        if (!(ph > 0.0)) {
            continue;
        }
        var w = ph * r.w * r.m;
        if (unbiased) {
            var den = 0.0;
            var own = 0.0;
            for (var j = 0u; j < n; j++) {
                let rj = (*rs)[j];
                if (!(rj.m > 0.0)) {
                    continue;
                }
                var pj = ph;
                if (j != 0u) {
                    pj = ptRsLum(ptRsTarget((*gs)[j], (*bps)[j], r.light, r.u));
                }
                den += rj.m * pj;
                if (j == i) {
                    own = pj;
                }
            }
            w = select(0.0, (r.m * own / den) * ph * r.w, den > 0.0);
        }
        wsum += w;
        if (ptRsRand2(seed, frame, key + i).x * wsum < w) {
            out.light = r.light;
            out.u = r.u;
            selPh = ph;
        }
    }
    if (selPh > 0.0 && wsum > 0.0) {
        out.w = select(wsum / (mtot * selPh), wsum / selPh, unbiased);
    } else {
        out.light = RT_NONE;
    }
    out.m = min(mtot, cap);
    return out;
}

fn ptRsSurfaceOk(a: PtRsGeom, b: PtRsGeom) -> bool {
    return b.depth > 0.0 && dot(a.n, b.n) >= ptRsParams.thr.y && abs(a.depth - b.depth) <= ptRsParams.thr.z * a.depth;
}

@compute @workgroup_size(${8}, ${8})
fn ptRestirGbuf(@builtin(global_invocation_id) id: vec3u) {
    let P = ptRsParams;
    if (id.x >= P.dims.x || id.y >= P.dims.y) {
        return;
    }
    let npx = P.dims.x * P.dims.y;
    let pix = id.y * P.dims.x + id.x;
    let i = (P.reg.x * npx + pix) * PT_RS_WORDS;
${bn ? `    ptBnPixel = id.xy;
` : ''}    let pixelSeed = ptPixelSeed(id.x, id.y, P.dims.w);
    let jitter = ${vb ? 'vec2f(P.curOrigin.w, P.curRight.w)' : 'ptSample2D(pixelSeed, P.dims.z, PT_PAIR_JITTER)'};
    let ndc = vec2f(((f32(id.x) + jitter.x) / f32(P.dims.x)) * 2.0 - 1.0, 1.0 - ((f32(id.y) + jitter.y) / f32(P.dims.y)) * 2.0);
    let origin = P.curOrigin.xyz;
    var ray = RtRay(origin, 0.0, normalize(P.curForward.xyz + P.curRight.xyz * ndc.x + P.curUp.xyz * ndc.y), PT_FAR);
${tex ? `    var cone = RtCone(0.0, bitcast<f32>(P.reg2.y) * 2.0 * length(P.curUp.xyz) / (length(P.curForward.xyz) * f32(P.dims.y)));
` : ''}${mask ? `    var skips = 0u;
` : ''}${vb ? `    var primary = true;
` : ''}    loop {
${vb ? `        var hit: RtHit;
        if (primary) {
            primary = false;
            hit = ptVbufferHit(id.xy, ray);
        } else {
            hit = rtClosest(ray, RT_MASK_ALL);
        }
` : `        let hit = rtClosest(ray, RT_MASK_ALL);
`}        if (hit.t < 0.0) {
            ptRsGbuf[i + 4u] = PT_RS_MISS;${shared ? `
            ptRsHit[pix] = vec4u(0u);` : ''}
            return;
        }
${tex ? `        let surf = rtSurface(hit, ray, RT_SURF_POSITION | RT_SURF_SHADING_NORMAL | RT_SURF_UV | RT_SURF_TEXFRAME);
        let coneW = rtConeAt(cone, hit.t);
        let fp = select(0.0, coneW * surf.uvDensity / max(abs(dot(hit.normal, ray.direction)), 0.001), PT_TEX_CONE);
        let mat = rtApplyMaterialTextures(rtMaterial(hit.material), surf.uv, fp);
        var shadingN = surf.shadingNormal;
        if (mat.normalTexture != RT_TEXTURE_NONE) {
            let nr = rtTexRecord(mat.normalTexture);
            shadingN = rtApplyNormalMap(surf.shadingNormal, surf.tangent, rtTexSampleRecord(nr, surf.uv, rtTexLodOf(nr, fp)).xyz);
        }
` : `        let surf = rtSurface(hit, ray, RT_SURF_POSITION | RT_SURF_SHADING_NORMAL);
`}${mask ? `        if (mat.alphaMode == RT_ALPHA_MASK && mat.opacity < mat.alphaCutoff) {
            skips += 1u;
            if (skips > PT_MASK_SKIPS) {
                ptRsGbuf[i + 4u] = PT_RS_MISS;${shared ? `
                ptRsHit[pix] = vec4u(0u);` : ''}
                return;
            }
            cone = RtCone(coneW, cone.spread);
            ray = RtRay(ptOffsetOrigin(surf.position, ptSideNormal(hit.normal, ray.direction)), 0.0, ray.direction, PT_FAR);
            continue;
        }
` : ''}        ptRsGbuf[i] = bitcast<u32>(surf.position.x);
        ptRsGbuf[i + 1u] = bitcast<u32>(surf.position.y);
        ptRsGbuf[i + 2u] = bitcast<u32>(surf.position.z);
        ptRsGbuf[i + 3u] = ptRsOctEncode(${tex ? 'shadingN' : 'surf.shadingNormal'});
        ptRsGbuf[i + 4u] = hit.material;${tex ? `
        ptRsGbuf[i + ${gl.uv}u] = bitcast<u32>(surf.uv.x);
        ptRsGbuf[i + ${(gl.uv ?? 0) + 1}u] = bitcast<u32>(surf.uv.y);
        ptRsGbuf[i + ${(gl.uv ?? 0) + 2}u] = bitcast<u32>(fp);` : ''}${gl.mix !== null ? `
        ptRsGbuf[i + ${gl.mix}u] = bitcast<u32>(${mixExpr});` : ''}${shared ? `
        let bq = vec2u(clamp(round(hit.bary * 65535.0), vec2f(0.0), vec2f(65535.0)));
        ptRsHit[pix] = vec4u(hit.instance + 1u, hit.triangle, hit.material, bq.x | (bq.y << 16u));` : ''}
        return;
    }
}

@compute @workgroup_size(${8}, ${8})
fn ptRestirInit(@builtin(global_invocation_id) id: vec3u) {
    let P = ptRsParams;
    if (id.x >= P.dims.x || id.y >= P.dims.y) {
        return;
    }
    let pix = id.y * P.dims.x + id.x;
    let g = ptRsLoadGeom(P.reg.x, pix, false);
    if (!(g.depth > 0.0)) {
        ptRsStoreRes(P.reg.w, pix, ptRsEmpty());
        return;
    }
    let bp = ptRsBsdf(g);
    if (ptBsdfIsDelta(bp) || bp.kind == PT_BSDF_BLACK) {
        ptRsStoreRes(P.reg.w, pix, ptRsEmpty());
        return;
    }
    let seed = ptPixelSeed(id.x, id.y, P.dims.w);
    let count = P.cfg.x;
    var out = ptRsEmpty();
    var wsum = 0.0;
    var selPh = 0.0;
    for (var c = 0u; c < count; c++) {
        let a = ptRsRand2(seed, P.dims.z, 2u * c);
        let pick = rtSampleLight(a.x);
        if (pick.index == RT_NONE || !(pick.pmf > 0.0)) {
            continue;
        }
        let u = ptRsUnpackU(ptRsPackU(ptRsRand2(seed, P.dims.z, 2u * c + 1u)));
        let ph = ptRsLum(ptRsTarget(g, bp, pick.index, u));
        if (!(ph > 0.0)) {
            continue;
        }
        let w = ph / pick.pmf;
        wsum += w;
        if (a.y * wsum < w) {
            out.light = pick.index;
            out.u = u;
            selPh = ph;
        }
    }
    out.m = f32(count);
    if (selPh > 0.0) {
        out.w = wsum / (f32(count) * selPh);
        if ((P.cfg.z & PT_RS_FLAG_VIS) != 0u && !ptRsVisible(g, out.light, out.u)) {
            out.w = 0.0;
        }
    } else {
        out.light = RT_NONE;
    }
    ptRsStoreRes(P.reg.w, pix, out);
}

@compute @workgroup_size(${8}, ${8})
fn ptRestirTemporal(@builtin(global_invocation_id) id: vec3u) {
    let P = ptRsParams;
    if (id.x >= P.dims.x || id.y >= P.dims.y) {
        return;
    }
    let pix = id.y * P.dims.x + id.x;
    var gs: array<PtRsGeom, ${PT_RESTIR_MAX_SPATIAL + 1}>;
    var bps: array<PtBsdfParams, ${PT_RESTIR_MAX_SPATIAL + 1}>;
    var rs: array<PtRsRes, ${PT_RESTIR_MAX_SPATIAL + 1}>;
    gs[0] = ptRsLoadGeom(P.reg.x, pix, false);
    rs[0] = ptRsLoadRes(P.reg.z, pix);
    if (!(gs[0].depth > 0.0)) {
        ptRsStoreRes(P.reg.w, pix, ptRsEmpty());
        return;
    }
    bps[0] = ptRsBsdf(gs[0]);
    let seed = ptPixelSeed(id.x, id.y, P.dims.w);
    let cap = P.thr.w;
    var n = 1u;
    if ((P.cfg.z & PT_RS_FLAG_HISTORY) != 0u) {
        let d = gs[0].p - P.prevOrigin.xyz;
        let a = dot(d, P.prevForward.xyz);
        if (a > 1.0e-4) {
            let nx = dot(d, P.prevRight.xyz) / dot(P.prevRight.xyz, P.prevRight.xyz) / a;
            let ny = dot(d, P.prevUp.xyz) / dot(P.prevUp.xyz, P.prevUp.xyz) / a;
            var pf = vec2f((nx * 0.5 + 0.5) * f32(P.dims.x), (0.5 - ny * 0.5) * f32(P.dims.y));
            if ((P.cfg.z & PT_RS_FLAG_RAND_TEMP) != 0u) {
                pf += ptRsRand2(seed, P.dims.z, PT_RS_KEY_SHIFT) - vec2f(0.5);
            }
            let q = vec2i(floor(pf));
            if (all(q >= vec2i(0)) && all(q < vec2i(P.dims.xy))) {
                let qpix = u32(q.y) * P.dims.x + u32(q.x);
                let gp = ptRsLoadGeom(P.reg.y, qpix, true);
                if (gp.depth > 0.0 && dot(gs[0].n, gp.n) >= P.thr.y && abs(a - gp.depth) <= P.thr.z * a) {
                    var rp = ptRsLoadRes(P.reg2.x, qpix);
                    rp.m = min(rp.m, cap);
                    if (rp.m > 0.0) {
                        gs[1] = gp;
                        bps[1] = ptRsBsdf(gp);
                        rs[1] = rp;
                        n = 2u;
                    }
                }
            }
        }
    }
    ptRsStoreRes(P.reg.w, pix, ptRsMerge(n, &gs, &bps, &rs, (P.cfg.z & PT_RS_FLAG_UNBIASED) != 0u, seed, P.dims.z, cap, PT_RS_KEY_TEMPORAL));
}

@compute @workgroup_size(${8}, ${8})
fn ptRestirSpatial(@builtin(global_invocation_id) id: vec3u) {
    let P = ptRsParams;
    if (id.x >= P.dims.x || id.y >= P.dims.y) {
        return;
    }
    let pix = id.y * P.dims.x + id.x;
    var gs: array<PtRsGeom, ${PT_RESTIR_MAX_SPATIAL + 1}>;
    var bps: array<PtBsdfParams, ${PT_RESTIR_MAX_SPATIAL + 1}>;
    var rs: array<PtRsRes, ${PT_RESTIR_MAX_SPATIAL + 1}>;
    gs[0] = ptRsLoadGeom(P.reg.x, pix, false);
    rs[0] = ptRsLoadRes(P.reg.z, pix);
    if (!(gs[0].depth > 0.0)) {
        ptRsStoreRes(P.reg.w, pix, ptRsEmpty());
        return;
    }
    bps[0] = ptRsBsdf(gs[0]);
    let seed = ptPixelSeed(id.x, id.y, P.dims.w);
    let cap = P.thr.w;
    let k = min(P.cfg.y, ${PT_RESTIR_MAX_SPATIAL}u);
    var n = 1u;
    for (var j = 0u; j < k; j++) {
        var off: vec2f;
        if ((P.cfg.z & PT_RS_FLAG_RAND_NB) != 0u) {
            let o = ptRsRand2(seed, P.dims.z, PT_RS_KEY_OFFSET + j);
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
        var rq = ptRsLoadRes(P.reg.z, qpix);
        rq.m = min(rq.m, cap);
        if (!(rq.m > 0.0)) {
            continue;
        }
        gs[n] = gq;
        bps[n] = ptRsBsdf(gq);
        rs[n] = rq;
        n += 1u;
    }
    ptRsStoreRes(P.reg.w, pix, ptRsMerge(n, &gs, &bps, &rs, (P.cfg.z & PT_RS_FLAG_UNBIASED) != 0u, seed, P.dims.z, cap, PT_RS_KEY_SPATIAL));
}
`;
}
