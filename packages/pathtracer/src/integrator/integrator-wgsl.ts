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
import type { PtMaterialOptions } from '../bsdf/adapter.ts';
import type { PtBsdfOptions } from '../bsdf/bsdf.ts';
import type { PtSamplerKind } from './dimensions.ts';
import { ptHookVertexExpr, type PtHooksOptions } from './outputs-wgsl.ts';

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
  /**
   * The scene has material textures (P1.4c; set by `PtMegakernel.create` from `textures`): the kernel fetches them in the material
   * decode and tracks a ray cone. Default false: no texture code at all (the kernel text is unchanged).
   */
  textures?: boolean;
  /**
   * Level of detail of texture lookups: `0` (default) always the base level, the reference setting (pbrt `bilinear` and Mitsuba `bilinear`
   * do not use mips); `'cone'` ray cones (Akenine-Moller et al. 2021): the width starts at the pixel angle, grows with distance, a
   * specular bounce keeps its spread, a non-delta bounce adds `PT_CONE_DIFFUSE_SPREAD` radians. The starting spread is the pixel angle times
   * `PtLimits.textureConeScale` (default 1; pbrt-v4 uses max(1/8, 1/sqrt(spp)): with many samples per pixel the pixel box already
   * averages the texture, so a full-pixel filter on top blurs it twice).
   */
  textureLod?: 'cone' | 0;
  /**
   * The scene has `alphaMode` MASK materials (P1.4c5; set by the adapter from the scene copy; needs `textures`): a hit whose base colour alpha
   * times `opacity` is below `alphaCutoff` is passed through without costing a bounce (a hard cutoff, as pbrt's `alpha` texture and Mitsuba's `mask`
   * BSDF), and shadow rays pass through such hits too (`ptVisibleMasked`). Default false: no mask code at all (the kernel text is unchanged).
   */
  alphaMask?: boolean;
  /**
   * OpenPBR mode (P2.21, P2.22, P2.30): the scene has `openpbr` materials. Selects the OpenPBR material adapter (`ptBsdfFromOpenPbr`) and compiles
   * only the lobes `features` names: stochastic metal/dielectric mix, thin-walled dielectric, layered dielectric-over-diffuse
   * (`layered.layering` albedo | stochastic) and the absorbing interior medium (path state: Beer's law over the distance to the exit
   * hit; inside/outside from the hit side, no nesting). Default null: the legacy adapter and the unchanged kernel text.
   */
  materials?: PtMaterialOptions | null;
  /**
   * L3 path hooks (R10, spec section 8): user WGSL `ptOnVertex(v: PtPathVertex) -> u32` and `ptTerminate(v: PtPathVertex) -> vec3f`, called by both
   * schedulers at every scattering vertex after emission and next-event estimation (`outputs-wgsl.ts`; `docs/design/integrator.md`). A hook that returns
   * `PT_HOOK_TERMINATE` ends the path and adds `throughput * ptTerminate(v)`. Default null: no hook code at all (the kernel text is unchanged).
   */
  hooks?: PtHooksOptions | null;
  /** Per-pixel debug counters (R4): depth, rays, termination reason, in one storage buffer (`readCounters`). Never in a timed run. Default false: the kernel text is unchanged. */
  counters?: boolean;
  /** First non-specular albedo + normal and the primary viewZ, accumulated like radiance in the tail of the accumulation buffer (`readAovs`). Default false: the kernel text is unchanged. */
  aovs?: boolean;
}

export interface ResolvedIntegratorOptions {
  sampler: PtSamplerKind;
  nee: boolean;
  envSampling: 'uniform' | 'importance' | 'none';
  envMap: boolean;
  sunVisible: boolean;
  textures: boolean;
  textureLod: 'cone' | 0;
  alphaMask: boolean;
  materials: PtMaterialOptions | null;
  hooks: PtHooksOptions | null;
  counters: boolean;
  aovs: boolean;
}

export function resolveIntegratorOptions(o: PtIntegratorOptions = {}): ResolvedIntegratorOptions {
  return { sampler: o.sampler ?? 'owen-sobol', nee: o.nee ?? true, envSampling: o.envSampling ?? 'uniform', envMap: o.envMap ?? false, sunVisible: o.sunVisible ?? false, textures: o.textures ?? false, textureLod: o.textureLod ?? 0, alphaMask: (o.alphaMask ?? false) && (o.textures ?? false), materials: o.materials ?? null, hooks: o.hooks ?? null, counters: o.counters ?? false, aovs: o.aovs ?? false };
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

/**
 * The pieces of the integrator the wavefront scheduler (`../wavefront/`) shares with the megakernel, so both compile the same text:
 * constants, ray-origin offset, MIS weight and the escaped-ray (environment / sun) term. Moving them here left the megakernel's text byte-identical.
 */
export function ptCommonHeadWgsl(o: ResolvedIntegratorOptions): string {
  const tex = o.textures;
  const mask = o.alphaMask;
  return /* wgsl */ `const PT_FAR: f32 = 1.0e30;
const PT_NEE: bool = ${o.nee};
const PT_ENV_SAMPLING: bool = ${o.envSampling !== 'none'};
const PT_SUN_VISIBLE: bool = ${o.sunVisible};
const PT_ORIGIN: f32 = 0.03125;
const PT_FLOAT_SCALE: f32 = 0.0000152587890625;
const PT_INT_SCALE: f32 = 256.0;
${mask ? `const PT_MASK_SKIPS: u32 = 64u;
` : ''}${tex ? `const PT_TEX_CONE: bool = ${o.textureLod === 'cone'};
const PT_CONE_DIFFUSE_SPREAD: f32 = 0.5;
` : ''}
fn ptOffsetOrigin(p: vec3f, n: vec3f) -> vec3f {
    let ofs = vec3i(PT_INT_SCALE * n);
    let pi = bitcast<vec3f>(bitcast<vec3i>(p) + select(ofs, -ofs, p < vec3f(0.0)));
    return select(pi, p + PT_FLOAT_SCALE * n, abs(p) < vec3f(PT_ORIGIN));
}

fn ptSideNormal(n: vec3f, w: vec3f) -> vec3f {
    return select(-n, n, dot(n, w) > 0.0);
}`;
}

export const PT_MIS_WGSL = /* wgsl */ `fn ptMis(pThis: f32, pOther: f32) -> f32 {
    if (!(pThis > 0.0)) {
        return 0.0;
    }
    let r = min(pOther / pThis, 1.0e18);
    return 1.0 / (1.0 + r * r);
}

fn ptMax3(v: vec3f) -> f32 {
    return max(v.x, max(v.y, v.z));
}`;

export const PT_ESCAPED_WGSL = /* wgsl */ `fn ptEscaped(dir: vec3f, direct: bool, prevPdf: f32) -> vec3f {
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
}`;

/** The integrator module text, to append after the raytracer module, the light sampling, the Owen-Sobol functions, the BSDF, its adapter and the sampler. */
export function ptIntegratorWgsl(options: PtIntegratorOptions = {}): string {
  const o = resolveIntegratorOptions(options);
  const tex = o.textures;
  const mask = o.alphaMask;
  const shadingNormal = tex ? 'shadingN' : 'surf.shadingNormal';
  const mats = o.materials;
  const layered = !!mats?.features.layered;
  const stochastic = layered && (mats?.layered?.layering ?? 'albedo') === 'stochastic';
  const medium = !!mats?.features.medium;
  const aniso = !!mats?.features.aniso;
  const fuzz = !!mats?.features.fuzz;
  const film = !!mats?.features.thinFilm;
  const evalFn = fuzz ? 'ptFuzzedEval' : layered ? 'ptEval' : 'ptBsdfEval';
  const sampleFn = fuzz ? 'ptFuzzedSample' : layered ? 'ptSample' : 'ptBsdfSample';
  const mixU = mats?.features.mix ? 'ptSample2D(pixelSeed, sampleIndex, ptVertexPair(depth, PT_SLOT_RR)).y' : '0.0';
  const hooks = !!o.hooks;
  const counters = o.counters;
  const aovs = o.aovs;
  const black = fuzz ? '(bp.kind == PT_BSDF_BLACK && !ptFuzzActive(1.0))' : 'bp.kind == PT_BSDF_BLACK';
  const term = (reason: number): string => (counters ? `            termReason = ${reason}u;\n` : '');
  return /* wgsl */ `
alias PtRadiance = vec4f;
${PT_PARAMS_WGSL}
@group(2) @binding(0) var<uniform> ptParams: PtParams;
@group(2) @binding(1) var<storage, read_write> ptAccum: array<vec4f>;

${ptCommonHeadWgsl(o)}

${mask ? `fn ptMaskedOut(hit: RtHit, ray: RtRay) -> bool {
    let m = rtMaterial(hit.material);
    if (m.alphaMode != RT_ALPHA_MASK) {
        return false;
    }
    let s = rtSurface(hit, ray, RT_SURF_UV);
    let a = rtApplyMaterialTextures(m, s.uv, 0.0);
    return a.opacity < a.alphaCutoff;
}

fn ptVisibleMasked(ray0: RtRay, mask: u32) -> bool {
    var ray = ray0;
    for (var i = 0u; i < PT_MASK_SKIPS; i++) {
        let hit = rtClosest(ray, mask);
        if (hit.t < 0.0) {
            return true;
        }
        if (!ptMaskedOut(hit, ray)) {
            return false;
        }
        let p = ptOffsetOrigin(ray.origin + ray.direction * hit.t, ptSideNormal(hit.normal, ray.direction));
        ray = RtRay(p, 0.0, ray.direction, ray.tMax - length(p - ray.origin));
        if (ray.tMax <= 0.0) {
            return true;
        }
    }
    return false;
}

` : ''}${PT_MIS_WGSL}

fn ptSampleLd(p: vec3f, ng: vec3f, frame: PtFrame, wo: vec3f, bp: PtBsdfParams, pixelSeed: u32, sampleIndex: u32, vertex: u32${counters ? ', pxIndex: u32' : ''}) -> vec3f {
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
    let e = ${evalFn}(bp, wo, wiLocal);
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
${counters ? `    ptCountRay(pxIndex, 0u, 1u);
` : ''}    if (!${mask ? 'ptVisibleMasked' : 'rtVisible'}(ray, RT_MASK_ALL)) {
        return vec3f(0.0);
    }
    let pl = pick.pmf * li.pdf;
    var w = 1.0;
    if (!li.delta) {
        w = ptMis(pl, e.pdf);
    }
    return li.radiance * f * (w / pl);
}

${PT_ESCAPED_WGSL}

fn ptPath(px: vec2u, sampleIndex: u32) -> PtRadiance {
    let dims = ptParams.dims;
    let maxDepth = ptParams.limits.x;
    let rrStart = ptParams.limits.y;
    let pixelSeed = ptPixelSeed(px.x, px.y, dims.w);
    let jitter = ptSample2D(pixelSeed, sampleIndex, PT_PAIR_JITTER);
    let ndc = vec2f(((f32(px.x) + jitter.x) / f32(dims.x)) * 2.0 - 1.0, 1.0 - ((f32(px.y) + jitter.y) / f32(dims.y)) * 2.0);
    let origin = ptParams.origin.xyz;
    var ray = RtRay(origin, 0.0, normalize(ptParams.forward.xyz + ptParams.right.xyz * ndc.x + ptParams.up.xyz * ndc.y), PT_FAR);

${counters || aovs || hooks ? `    let pxIndex = px.y * dims.x + px.x;
` : ''}${counters ? `    var termReason = 0u;
` : ''}${aovs ? `    var aovDone = false;
` : ''}    var radiance = vec3f(0.0);
    var beta = vec3f(1.0);
    var specularBounce = false;
    var depth = 0u;
    var etaScale = 1.0;
${mask ? `    var skips = 0u;
` : ''}    var prevP = origin;
    var prevPdf = 0.0;
${medium ? `    var sigma = vec3f(0.0);
` : ''}${tex ? `    var cone = RtCone(0.0, bitcast<f32>(ptParams.limits.z) * 2.0 * length(ptParams.up.xyz) / (length(ptParams.forward.xyz) * f32(dims.y)));
` : ''}
    loop {
        let hit = rtClosest(ray, RT_MASK_ALL);
${counters ? `        ptCountRay(pxIndex, 1u, 0u);
` : ''}        if (hit.t < 0.0) {
            radiance += beta * ptEscaped(ray.direction, depth == 0u || specularBounce, prevPdf);
            break;
        }
${tex ? `        let surf = rtSurface(hit, ray, RT_SURF_POSITION | RT_SURF_SHADING_NORMAL | RT_SURF_UV | RT_SURF_TEXFRAME${aniso ? ' | RT_SURF_TANGENT' : ''});
        let coneW = rtConeAt(cone, hit.t);
        let fp = select(0.0, coneW * surf.uvDensity / max(abs(dot(hit.normal, ray.direction)), 0.001), PT_TEX_CONE);
        let mat = rtApplyMaterialTextures(rtMaterial(hit.material), surf.uv, fp);
        var shadingN = surf.shadingNormal;
        if (mat.normalTexture != RT_TEXTURE_NONE) {
            let nr = rtTexRecord(mat.normalTexture);
            shadingN = rtApplyNormalMap(surf.shadingNormal, surf.tangent, rtTexSampleRecord(nr, surf.uv, rtTexLodOf(nr, fp)).xyz);
        }` : `        let surf = rtSurface(hit, ray, RT_SURF_POSITION | RT_SURF_SHADING_NORMAL${aniso ? ' | RT_SURF_TANGENT' : ''});
        let mat = rtMaterial(hit.material);`}
${medium ? `        beta = beta * exp(-sigma * hit.t);
` : ''}${mask ? `        if (mat.alphaMode == RT_ALPHA_MASK && mat.opacity < mat.alphaCutoff) {
            skips += 1u;
            if (skips > PT_MASK_SKIPS) {
${counters ? `                termReason = 6u;
` : ''}                break;
            }
            cone = RtCone(coneW, cone.spread);
            ray = RtRay(ptOffsetOrigin(surf.position, ptSideNormal(hit.normal, ray.direction)), 0.0, ray.direction, PT_FAR);
            continue;
        }
` : ''}
${hooks ? `        var hookEmitted = vec3f(0.0);
` : ''}        if (mat.emissionLuminance > 0.0 && (hit.frontFace || mat.emissionTwoSided)) {
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
${hooks ? `            hookEmitted = beta * mat.emissionColor * w;
` : ''}        }

        let bp = ptBsdfFromMaterial(mat${mats ? `, ${mixU}` : ''});
${film ? `        ptFilm = select(vec3f(0.0), ptMaterialFilm(hit.material), mat.coatWeight <= 0.5 && mat.transmissionWeight <= 0.5);
` : ''}${fuzz ? `        ptFuzzW = mat.sheenWeight;
        ptFuzzC = vec4f(mat.sheenColor, mat.sheenRoughness);
` : ''}${aovs ? `        if (depth == 0u) {
            ptAovPrimary(pxIndex, dims.x * dims.y, -dot(surf.position - origin, normalize(ptParams.forward.xyz)));
        }
        if (!aovDone && !ptBsdfIsDelta(bp)) {
            aovDone = true;
            ptAovGuide(pxIndex, dims.x * dims.y, ptAovAlbedo(mat), ${shadingNormal});
        }
` : ''}        if (${black} || depth >= maxDepth) {
${counters ? `            termReason = select(2u, 1u, ${black});
` : ''}            break;
        }
        let vertex = depth;
        depth += 1u;
${layered ? `        ptLayerSeed = pixelSeed ^ (sampleIndex * 2654435761u) ^ (vertex * 2246822519u);
` : ''}
${aniso ? `        var frame = ptFrameFromNormal(${shadingNormal});
        if (bp.kind == PT_BSDF_CONDUCTOR && mat.specularAnisotropy > 0.0) {
            frame = ptFrameFromTangent(${shadingNormal}, surf.vertexTangent, mat.anisotropyRotation * 6.283185307179586);
        }
` : `        let frame = ptFrameFromNormal(${shadingNormal});
`}        let wo = ptToLocal(frame, -ray.direction);
        if (PT_NEE && ${fuzz ? '(!ptBsdfIsDelta(bp) || ptFuzzActive(wo.z))' : '!ptBsdfIsDelta(bp)'}) {
            radiance += beta * ptSampleLd(surf.position, hit.normal, frame, wo, bp, pixelSeed, sampleIndex, vertex${counters ? ', pxIndex' : ''});
        }

${hooks ? `        let hv = ${ptHookVertexExpr({ x: 'px.x', y: 'px.y', pixel: 'pxIndex', sampleIndex: 'sampleIndex' })};
        if (ptOnVertex(hv) == PT_HOOK_TERMINATE) {
            radiance += beta * ptTerminate(hv);
${term(4)}            break;
        }

` : ''}        let dirU = ptSample2D(pixelSeed, sampleIndex, ptVertexPair(vertex, PT_SLOT_BSDF_DIRECTION));
        let compU = ptSample2D(pixelSeed, sampleIndex, ptVertexPair(vertex, PT_SLOT_SELECT_COMPONENT)).y;
        let s = ${sampleFn}(bp, wo, vec3f(compU, dirU.x, dirU.y));
        if (s.flags == 0u || !(s.pdf > 0.0)) {
${term(5)}            break;
        }
        beta = beta * s.f * (abs(s.wi.z) / s.pdf);
        specularBounce = (s.flags & PT_FLAG_DELTA) != 0u;
${tex ? `        cone = RtCone(coneW, cone.spread + select(PT_CONE_DIFFUSE_SPREAD, 0.0, specularBounce));
` : ''}        if ((s.flags & PT_FLAG_TRANSMISSION) != 0u) {
            etaScale = etaScale * s.etap * s.etap;
${medium ? `            sigma = select(vec3f(0.0), ptMediumSigma(mat), wo.z > 0.0);
` : ''}        }
        prevPdf = s.pdf;${stochastic ? `
        if ((s.flags & PT_FLAG_PROPORTIONAL) != 0u) {
            prevPdf = ptLayeredPdf(ptLayeredOf(bp), wo, s.wi, ptLayerSeed);
        }` : ''}
        prevP = surf.position;
        let wiWorld = ptFromLocal(frame, s.wi);
        ray = RtRay(ptOffsetOrigin(surf.position, ptSideNormal(hit.normal, wiWorld)), 0.0, wiWorld, PT_FAR);

        if (rrStart > 0u && depth >= rrStart) {
            let rrBeta = beta * etaScale;
            let m = ptMax3(rrBeta);
            if (m < 1.0) {
                let q = max(0.0, 1.0 - m);
                if (ptSample2D(pixelSeed, sampleIndex, ptVertexPair(vertex, PT_SLOT_RR)).x < q) {
${counters ? `                    termReason = 3u;
` : ''}                    break;
                }
                beta = beta / (1.0 - q);
            }
        }
    }
${counters ? `    ptCountEnd(pxIndex, depth, termReason);
` : ''}    return PtRadiance(radiance, 0.0);
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
