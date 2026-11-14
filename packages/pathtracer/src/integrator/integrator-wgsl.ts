/**
 * The megakernel path tracer's WGSL (prefix `pt`), one thread per pixel, one sample per dispatch. The integrator is pbrt-v4's
 * `PathIntegrator` (Apache-2.0, THIRD_PARTY 1.3) in structure: emission at the hit, depth check, next-event estimation with
 * the power heuristic against BSDF sampling, BSDF sampling, Russian roulette; Mitsuba's `max_depth` is ours + 1.
 * See docs/design/integrator.md for the semantics line by line and DIVERGES.md for where we leave pbrt.
 *
 * Bindings: bvh walk at group 0, scene copy at group 1 (two buffers), group 2 = `ptParams` (uniform, dynamic offset) and
 * `ptAccum` (read-write `array<vec4f>`, one f32 sum per pixel). Radiance is `PtRadiance` = vec4f (rgb + a reserved lane,
 * spec section 6: spectral-ready); the accumulator adds `(L.rgb, 1)`, so `accum.w` is the sample count.
 *
 * Specialised per pipeline (module constants): sampler, NEE on/off, environment sampling, sun visibility, VNDF and transport
 * of the BSDF. Per dispatch (uniform): image size, sample index, seed, maxDepth, rrStartDepth, camera.
 */
import type { PtBsdfOptions } from '../bsdf/bsdf.ts';
import type { PtSamplerKind } from './dimensions.ts';

export interface PtIntegratorOptions extends PtBsdfOptions {
  /** `'owen-sobol'` (reference) or `'pcg'` (debug). */
  sampler?: PtSamplerKind;
  /** Next-event estimation + MIS. Default true. False = BSDF sampling only (debug; delta suns contribute nothing). */
  nee?: boolean;
  /**
   * `'uniform'` (sphere; NEE with MIS), `'importance'` (an environment map sampled by its luminance; needs `envMap`, otherwise the
   * same as uniform) or `'none'` (pbrt's `allowIncompletePDF` for a uniform sky: BSDF sampling only for it). Default 'uniform'.
   */
  envSampling?: 'uniform' | 'importance' | 'none';
  /** The scene has an equirect environment map: compile its lookup, pdf and sampling (kernel size +). Default false (constant radiance only). */
  envMap?: boolean;
  /** A sun with an angular radius is visible to camera and specular paths (default false: the references' sun is a delta light and invisible). */
  sunVisible?: boolean;
}

export interface ResolvedIntegratorOptions {
  sampler: PtSamplerKind;
  nee: boolean;
  envSampling: 'uniform' | 'importance' | 'none';
  envMap: boolean;
  sunVisible: boolean;
}

export function resolveIntegratorOptions(o: PtIntegratorOptions = {}): ResolvedIntegratorOptions {
  return { sampler: o.sampler ?? 'owen-sobol', nee: o.nee ?? true, envSampling: o.envSampling ?? 'uniform', envMap: o.envMap ?? false, sunVisible: o.sunVisible ?? false };
}

/** Bytes of one `ptParams` record (5 vec4 + 1 vec4u pair). */
export const PT_PARAMS_BYTES = 96;
export const PT_WORKGROUP = [8, 8] as const;

export const PT_PARAMS_WGSL = /* wgsl */ `
struct PtParams {
    dims: vec4u,
    limits: vec4u,
    origin: vec4f,
    right: vec4f,
    up: vec4f,
    forward: vec4f,
}
`;

/** The integrator module text, to append after the raytracer module, the light sampling, the Owen-Sobol functions, the BSDF, its adapter and the sampler. */
export function ptIntegratorWgsl(options: PtIntegratorOptions = {}): string {
  const o = resolveIntegratorOptions(options);
  return /* wgsl */ `
alias PtRadiance = vec4f;
${PT_PARAMS_WGSL}
@group(2) @binding(0) var<uniform> ptParams: PtParams;
@group(2) @binding(1) var<storage, read_write> ptAccum: array<vec4f>;

const PT_FAR: f32 = 1.0e30;
const PT_NEE: bool = ${o.nee};
const PT_ENV_SAMPLING: bool = ${o.envSampling !== 'none'};
const PT_SUN_VISIBLE: bool = ${o.sunVisible};
const PT_ORIGIN: f32 = 0.03125;
const PT_FLOAT_SCALE: f32 = 0.0000152587890625;
const PT_INT_SCALE: f32 = 256.0;

fn ptOffsetOrigin(p: vec3f, n: vec3f) -> vec3f {
    let ofs = vec3i(PT_INT_SCALE * n);
    let pi = bitcast<vec3f>(bitcast<vec3i>(p) + select(ofs, -ofs, p < vec3f(0.0)));
    return select(pi, p + PT_FLOAT_SCALE * n, abs(p) < vec3f(PT_ORIGIN));
}

fn ptSideNormal(n: vec3f, w: vec3f) -> vec3f {
    return select(-n, n, dot(n, w) > 0.0);
}

fn ptMis(pThis: f32, pOther: f32) -> f32 {
    if (!(pThis > 0.0)) {
        return 0.0;
    }
    let r = min(pOther / pThis, 1.0e18);
    return 1.0 / (1.0 + r * r);
}

fn ptMax3(v: vec3f) -> f32 {
    return max(v.x, max(v.y, v.z));
}

fn ptSampleLd(p: vec3f, ng: vec3f, frame: PtFrame, wo: vec3f, bp: PtBsdfParams, pixelSeed: u32, sampleIndex: u32, vertex: u32) -> vec3f {
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
    let wiLocal = ptToLocal(frame, li.wi);
    let e = ptBsdfEval(bp, wo, wiLocal);
    let f = e.f * abs(wiLocal.z);
    if (!(ptMax3(f) > 0.0)) {
        return vec3f(0.0);
    }
    let origin = ptOffsetOrigin(p, ptSideNormal(ng, li.wi));
    var ray: RtRay;
    if (li.distance < 1.0e30) {
        let end = ptOffsetOrigin(li.point, ptSideNormal(li.normal, -li.wi));
        ray = rtSegment(origin, end, 0.0, 0.0);
    } else {
        ray = RtRay(origin, 0.0, li.wi, PT_FAR);
    }
    if (!rtVisible(ray, RT_MASK_ALL)) {
        return vec3f(0.0);
    }
    let pl = pick.pmf * li.pdf;
    var w = 1.0;
    if (!li.delta) {
        w = ptMis(pl, e.pdf);
    }
    return li.radiance * f * (w / pl);
}

fn ptEscaped(dir: vec3f, direct: bool, prevPdf: f32) -> vec3f {
    var le = vec3f(0.0);
    if (rtHasEnvironment()) {
        let e = rtEnvironment();
        var w = 1.0;
        if (!direct && PT_NEE && PT_ENV_SAMPLING) {
            w = ptMis(prevPdf, e.pmf * rtEnvironmentPdf(e, dir));
        }
        le += rtEnvironmentLe(e, dir) * w;
    }
    if (rtHasSun()) {
        let s = rtSun();
        if (rtSunContains(s, dir) && (PT_SUN_VISIBLE || !direct)) {
            var w = 1.0;
            if (!direct && PT_NEE) {
                w = ptMis(prevPdf, s.pmf / s.solidAngle);
            }
            le += rtSunRadiance(s) * w;
        }
    }
    return le;
}

fn ptPath(px: vec2u, sampleIndex: u32) -> PtRadiance {
    let dims = ptParams.dims;
    let maxDepth = ptParams.limits.x;
    let rrStart = ptParams.limits.y;
    let pixelSeed = ptPixelSeed(px.x, px.y, dims.w);
    let jitter = ptSample2D(pixelSeed, sampleIndex, PT_PAIR_JITTER);
    let ndc = vec2f(((f32(px.x) + jitter.x) / f32(dims.x)) * 2.0 - 1.0, 1.0 - ((f32(px.y) + jitter.y) / f32(dims.y)) * 2.0);
    let origin = ptParams.origin.xyz;
    var ray = RtRay(origin, 0.0, normalize(ptParams.forward.xyz + ptParams.right.xyz * ndc.x + ptParams.up.xyz * ndc.y), PT_FAR);

    var radiance = vec3f(0.0);
    var beta = vec3f(1.0);
    var specularBounce = false;
    var depth = 0u;
    var etaScale = 1.0;
    var prevP = origin;
    var prevPdf = 0.0;

    loop {
        let hit = rtClosest(ray, RT_MASK_ALL);
        if (hit.t < 0.0) {
            radiance += beta * ptEscaped(ray.direction, depth == 0u || specularBounce, prevPdf);
            break;
        }
        let surf = rtSurface(hit, ray, RT_SURF_POSITION | RT_SURF_SHADING_NORMAL);
        let mat = rtMaterial(hit.material);

        if (mat.emissionLuminance > 0.0 && (hit.frontFace || mat.emissionTwoSided)) {
            var w = 1.0;
            if (!(depth == 0u || specularBounce) && PT_NEE) {
                let lightIndex = rtLightIndexOfHit(hit.instance, hit.triangle);
                if (lightIndex != RT_NONE) {
                    let light = rtLight(lightIndex);
                    let dist = length(surf.position - prevP);
                    let cosLight = abs(dot(hit.normal, ray.direction));
                    w = ptMis(prevPdf, light.pmf * rtTriangleSolidAnglePdf(light.area, dist, cosLight));
                }
            }
            radiance += beta * mat.emissionColor * w;
        }

        let bp = ptBsdfFromMaterial(mat);
        if (bp.kind == PT_BSDF_BLACK || depth >= maxDepth) {
            break;
        }
        let vertex = depth;
        depth += 1u;

        let frame = ptFrameFromNormal(surf.shadingNormal);
        let wo = ptToLocal(frame, -ray.direction);
        if (PT_NEE && !ptBsdfIsDelta(bp)) {
            radiance += beta * ptSampleLd(surf.position, hit.normal, frame, wo, bp, pixelSeed, sampleIndex, vertex);
        }

        let dirU = ptSample2D(pixelSeed, sampleIndex, ptVertexPair(vertex, PT_SLOT_BSDF_DIRECTION));
        let compU = ptSample2D(pixelSeed, sampleIndex, ptVertexPair(vertex, PT_SLOT_SELECT_COMPONENT)).y;
        let s = ptBsdfSample(bp, wo, vec3f(compU, dirU.x, dirU.y));
        if (s.flags == 0u || !(s.pdf > 0.0)) {
            break;
        }
        beta = beta * s.f * (abs(s.wi.z) / s.pdf);
        specularBounce = (s.flags & PT_FLAG_DELTA) != 0u;
        if ((s.flags & PT_FLAG_TRANSMISSION) != 0u) {
            etaScale = etaScale * s.etap * s.etap;
        }
        prevPdf = s.pdf;
        prevP = surf.position;
        let wiWorld = ptFromLocal(frame, s.wi);
        ray = RtRay(ptOffsetOrigin(surf.position, ptSideNormal(hit.normal, wiWorld)), 0.0, wiWorld, PT_FAR);

        if (rrStart > 0u && depth >= rrStart) {
            let rrBeta = beta * etaScale;
            let m = ptMax3(rrBeta);
            if (m < 1.0) {
                let q = max(0.0, 1.0 - m);
                if (ptSample2D(pixelSeed, sampleIndex, ptVertexPair(vertex, PT_SLOT_RR)).x < q) {
                    break;
                }
                beta = beta / (1.0 - q);
            }
        }
    }
    return PtRadiance(radiance, 0.0);
}

@compute @workgroup_size(${PT_WORKGROUP[0]}, ${PT_WORKGROUP[1]})
fn ptMain(@builtin(global_invocation_id) id: vec3u) {
    let dims = ptParams.dims;
    if (id.x >= dims.x || id.y >= dims.y) {
        return;
    }
    let l = ptPath(id.xy, dims.z);
    let index = id.y * dims.x + id.x;
    ptAccum[index] = ptAccum[index] + vec4f(l.xyz, 1.0);
}
`;
}
