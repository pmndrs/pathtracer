/**
 * WGSL of the wavefront path tracer (P3.1): five kernels over SoA path state (layout in `layout.ts`). Same estimator as the
 * megakernel (`integrator-wgsl.ts`): same sampler dimensions, BSDF / light code, NEE + MIS, Russian roulette. What changes is the
 * control flow, which becomes the host's loop over depths:
 *
 *     camera                       one thread per pixel: jitter, primary ray, path state, accumulator sample count, Q0 = identity
 *     args (INIT)                  counters, dispatch args for N paths
 *     for d = 0 .. maxDepth:
 *         intersect  (indirect)    rtClosest for every queued ray, writes the hit record
 *         shade      (indirect)    escape / emission, BSDF, NEE sample -> shadow ray + shadow queue, BSDF sample, RR, next queue
 *         args       (1 thread)    queue counts -> indirect args, zeroes consumed counters
 *         shadow     (indirect)    rtVisible per shadow ray, adds beta * Ld into the accumulator
 *
 * Bindings: group 0 bvh walk and group 1 scene copy as the megakernel (a kernel that does not use a group declares none of it),
 * group 2 params + accumulator (+ fuzz table at 2) + the wavefront buffers, group 3 (args kernel only) the indirect args.
 *
 * Not in slice 1 (P3.2+): alpha MASK materials (the megakernel passes a masked hit without a bounce; here it would need extra iterations
 * or a loop in intersect), refill, compaction, octant binning, material sorting, several samples in flight, subgroup-aggregated appends.
 */
import { PT_ESCAPED_WGSL, PT_MIS_WGSL, PT_PARAMS_WGSL, ptCommonHeadWgsl, resolveIntegratorOptions, type PtIntegratorOptions, type ResolvedIntegratorOptions } from '../integrator/integrator-wgsl.ts';
import { ptHookVertexExpr } from '../integrator/outputs-wgsl.ts';
import { PTWF_ARGS_MAIN, PTWF_ARGS_SHADOW, PTWF_INIT, PTWF_STRIDE, PTWF_WORKGROUP } from './layout.ts';

/** Why a scene or option cannot run on the wavefront scheduler yet (slice 1), or null. */
export function ptwfUnsupported(options: PtIntegratorOptions): string | null {
  const o = resolveIntegratorOptions(options);
  if (o.alphaMask) return 'alpha MASK materials (P3.x: the megakernel passes a masked hit without a bounce; the wavefront loop has no place for that yet)';
  return null;
}

/** Binding numbers in group 2. 2 is the fuzz table's (`ptFuzzWgsl`), so the wavefront buffers start at 3. */
export const PTWF_BINDING = { params: 0, accum: 1, fuzz: 2, state: 3, hit: 4, count: 5, queue: 6, shadow: 7, counters: 8 /* `integrator.counters`, shade kernel only */, args: 0 /* group 3 of the args kernel; group 3 of the shade kernel is the hooks' */ } as const;

const bindings = (names: readonly (keyof typeof PTWF_BINDING)[]): string => {
  const decl: Record<string, string> = {
    params: '@group(2) @binding(0) var<uniform> ptParams: PtParams;',
    accum: '@group(2) @binding(1) var<storage, read_write> ptAccum: array<vec4f>;',
    state: '@group(2) @binding(3) var<storage, read_write> ptwfState: array<vec4u>;',
    hit: '@group(2) @binding(4) var<storage, read_write> ptwfHit: array<vec4u>;',
    count: '@group(2) @binding(5) var<storage, read_write> ptwfCount: array<atomic<u32>>;',
    queue: '@group(2) @binding(6) var<storage, read_write> ptwfQueue: array<u32>;',
    shadow: '@group(2) @binding(7) var<storage, read_write> ptwfShadowRay: array<vec4u>;',
  };
  return names.map((n) => decl[n]).join('\n');
};

const COMMON = /* wgsl */ `
const PTWF_WG: u32 = ${PTWF_WORKGROUP}u;
const PTWF_STRIDE: u32 = ${PTWF_STRIDE}u;
const PTWF_INIT: u32 = ${PTWF_INIT}u;

fn ptwfSlot(wid: vec3u, li: u32) -> u32 {
    return (wid.y * PTWF_STRIDE + wid.x) * PTWF_WG + li;
}

fn ptwfF3(v: vec4u) -> vec3f {
    return bitcast<vec3f>(v.xyz);
}
`;

/** State access: sections as in `layout.ts`. `extra` = medium or textures (section 4 exists). */
function stateWgsl(o: ResolvedIntegratorOptions): string {
  const medium = !!o.materials?.features.medium;
  const tex = o.textures;
  const extra = medium || tex;
  const aovs = o.aovs;
  return /* wgsl */ `
struct PtwfPath {
    origin: vec3f,
    prevPdf: f32,
    dir: vec3f,
    etaScale: f32,
    beta: vec3f,
    depth: u32,
    specular: bool,
    prevP: vec3f,
    sigma: vec3f,
    coneWidth: f32,
    coneSpread: f32,
${aovs ? `    aovDone: bool,
` : ''}}

fn ptwfLoadPath(path: u32, n: u32) -> PtwfPath {
    let s0 = ptwfState[path];
    let s1 = ptwfState[n + path];
    let s2 = ptwfState[2u * n + path];
    let s3 = ptwfState[3u * n + path];
${extra ? `    let s4 = ptwfState[4u * n + path];
` : ''}    var p: PtwfPath;
    p.origin = ptwfF3(s0);
    p.prevPdf = bitcast<f32>(s0.w);
    p.dir = ptwfF3(s1);
    p.etaScale = bitcast<f32>(s1.w);
    p.beta = ptwfF3(s2);
    p.depth = s2.w & ${aovs ? '0x3fffffff' : '0x7fffffff'}u;
    p.specular = (s2.w >> 31u) == 1u;${aovs ? `
    p.aovDone = ((s2.w >> 30u) & 1u) == 1u;` : ''}
    p.prevP = ptwfF3(s3);
    p.coneSpread = ${tex ? 'bitcast<f32>(s3.w)' : '0.0'};
    p.sigma = ${medium ? 'ptwfF3(s4)' : 'vec3f(0.0)'};
    p.coneWidth = ${tex ? 'bitcast<f32>(s4.w)' : '0.0'};
    return p;
}

fn ptwfStorePath(path: u32, n: u32, p: PtwfPath) {
    ptwfState[path] = vec4u(bitcast<vec3u>(p.origin), bitcast<u32>(p.prevPdf));
    ptwfState[n + path] = vec4u(bitcast<vec3u>(p.dir), bitcast<u32>(p.etaScale));
    ptwfState[2u * n + path] = vec4u(bitcast<vec3u>(p.beta), p.depth | select(0u, 0x80000000u, p.specular)${aovs ? ' | select(0u, 0x40000000u, p.aovDone)' : ''});
    ptwfState[3u * n + path] = vec4u(bitcast<vec3u>(p.prevP), ${tex ? 'bitcast<u32>(p.coneSpread)' : '0u'});
${extra ? `    ptwfState[4u * n + path] = vec4u(${medium ? 'bitcast<vec3u>(p.sigma)' : 'vec3u(0u)'}, ${tex ? 'bitcast<u32>(p.coneWidth)' : '0u'});
` : ''}}
`;
}

/** Camera kernel: the start of `ptPath` of the megakernel, writing path state instead of tracing. Needs the sampler. */
export function ptwfCameraWgsl(options: PtIntegratorOptions = {}): string {
  const o = resolveIntegratorOptions(options);
  const tex = o.textures;
  return /* wgsl */ `
${PT_PARAMS_WGSL}
${bindings(['params', 'accum', 'state', 'queue'])}
const PT_FAR: f32 = 1.0e30;
${COMMON}${stateWgsl(o)}
@compute @workgroup_size(8, 8)
fn ptwfCamera(@builtin(global_invocation_id) id: vec3u) {
    let dims = ptParams.dims;
    if (id.x >= dims.x || id.y >= dims.y) {
        return;
    }
    let path = id.y * dims.x + id.x;
    let n = dims.x * dims.y;
    let pixelSeed = ptPixelSeed(id.x, id.y, dims.w);
    let jitter = ptSample2D(pixelSeed, dims.z, PT_PAIR_JITTER);
    let ndc = vec2f(((f32(id.x) + jitter.x) / f32(dims.x)) * 2.0 - 1.0, 1.0 - ((f32(id.y) + jitter.y) / f32(dims.y)) * 2.0);
    var p: PtwfPath;
    p.origin = ptParams.origin.xyz;
    p.dir = normalize(ptParams.forward.xyz + ptParams.right.xyz * ndc.x + ptParams.up.xyz * ndc.y);
    p.prevPdf = 0.0;
    p.etaScale = 1.0;
    p.beta = vec3f(1.0);
    p.depth = 0u;
    p.specular = false;
    p.prevP = p.origin;
    p.sigma = vec3f(0.0);
    p.coneWidth = 0.0;
${o.aovs ? `    p.aovDone = false;
` : ''}    p.coneSpread = ${tex ? 'bitcast<f32>(ptParams.limits.z) * 2.0 * length(ptParams.up.xyz) / (length(ptParams.forward.xyz) * f32(dims.y))' : '0.0'};
    ptwfStorePath(path, n, p);
    ptAccum[path] = ptAccum[path] + vec4f(0.0, 0.0, 0.0, 1.0);
    ptwfQueue[path] = path;
}
`;
}

/** Intersect kernel: closest hit of every ray in `Q[iter & 1]`. Needs the raytracer module (walk, no scene). */
export function ptwfIntersectWgsl(options: PtIntegratorOptions = {}): string {
  const o = resolveIntegratorOptions(options);
  return /* wgsl */ `
${PT_PARAMS_WGSL}
${bindings(['params', 'state', 'hit', 'count', 'queue'])}
const PT_FAR: f32 = 1.0e30;
${COMMON}${stateWgsl(o)}
@compute @workgroup_size(${PTWF_WORKGROUP})
fn ptwfIntersect(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32) {
    let iter = ptParams.limits.w;
    let q = iter & 1u;
    let slot = ptwfSlot(wid, li);
    if (slot >= atomicLoad(&ptwfCount[q])) {
        return;
    }
    let n = ptParams.dims.x * ptParams.dims.y;
    let path = ptwfQueue[q * n + slot];
    let o = ptwfState[path];
    let d = ptwfState[n + path];
    let hit = rtClosest(RtRay(ptwfF3(o), 0.0, ptwfF3(d), PT_FAR), RT_MASK_ALL);
    ptwfHit[path] = vec4u(bitcast<u32>(hit.t), hit.instance, hit.triangle, hit.material);
    ptwfHit[n + path] = vec4u(bitcast<vec2u>(hit.bary), select(0u, 1u, hit.frontFace), 0u);
    ptwfHit[2u * n + path] = vec4u(bitcast<vec3u>(hit.normal), 0u);
}
`;
}

/** Shadow kernel: visibility of every shadow ray; a visible one adds its contribution to the pixel. Needs the raytracer module (walk, no scene). */
export function ptwfShadowWgsl(options: PtIntegratorOptions = {}): string {
  void options;
  return /* wgsl */ `
${PT_PARAMS_WGSL}
${bindings(['params', 'accum', 'count', 'queue', 'shadow'])}
${COMMON}
@compute @workgroup_size(${PTWF_WORKGROUP})
fn ptwfShadow(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32) {
    let iter = ptParams.limits.w;
    let slot = ptwfSlot(wid, li);
    if (slot >= atomicLoad(&ptwfCount[2u + (iter & 1u)])) {
        return;
    }
    let n = ptParams.dims.x * ptParams.dims.y;
    let path = ptwfQueue[2u * n + slot];
    let r0 = ptwfShadowRay[path];
    let r1 = ptwfShadowRay[n + path];
    let r2 = ptwfShadowRay[2u * n + path];
    if (rtVisible(RtRay(ptwfF3(r0), bitcast<f32>(r0.w), ptwfF3(r1), bitcast<f32>(r1.w)), RT_MASK_ALL)) {
        ptAccum[path] = ptAccum[path] + vec4f(ptwfF3(r2), 0.0);
    }
}
`;
}

/** Args kernel (one thread): queue counts -> indirect dispatch args; zeroes the counters nobody reads any more (`ptwfArgsStep` is its twin). Standalone module. */
export function ptwfArgsWgsl(): string {
  return /* wgsl */ `
${PT_PARAMS_WGSL}
${bindings(['params', 'count'])}
@group(3) @binding(0) var<storage, read_write> ptwfArgs: array<u32>;
const PTWF_WG: u32 = ${PTWF_WORKGROUP}u;
const PTWF_STRIDE: u32 = ${PTWF_STRIDE}u;
const PTWF_INIT: u32 = ${PTWF_INIT}u;

fn ptwfGroups(count: u32) -> vec3u {
    let g = (count + PTWF_WG - 1u) / PTWF_WG;
    return vec3u(min(g, PTWF_STRIDE), (g + PTWF_STRIDE - 1u) / PTWF_STRIDE, 1u);
}

@compute @workgroup_size(1)
fn ptwfArgsMain() {
    let iter = ptParams.limits.w;
    if (iter == PTWF_INIT) {
        atomicStore(&ptwfCount[0], ptParams.dims.x * ptParams.dims.y);
        atomicStore(&ptwfCount[1], 0u);
        atomicStore(&ptwfCount[2], 0u);
        atomicStore(&ptwfCount[3], 0u);
        let g = ptwfGroups(ptParams.dims.x * ptParams.dims.y);
        ptwfArgs[${PTWF_ARGS_MAIN / 4}] = g.x;
        ptwfArgs[${PTWF_ARGS_MAIN / 4 + 1}] = g.y;
        ptwfArgs[${PTWF_ARGS_MAIN / 4 + 2}] = g.z;
        return;
    }
    let cur = iter & 1u;
    let nxt = cur ^ 1u;
    let gm = ptwfGroups(atomicLoad(&ptwfCount[nxt]));
    let gs = ptwfGroups(atomicLoad(&ptwfCount[2u + cur]));
    ptwfArgs[${PTWF_ARGS_MAIN / 4}] = gm.x;
    ptwfArgs[${PTWF_ARGS_MAIN / 4 + 1}] = gm.y;
    ptwfArgs[${PTWF_ARGS_MAIN / 4 + 2}] = gm.z;
    ptwfArgs[${PTWF_ARGS_SHADOW / 4}] = gs.x;
    ptwfArgs[${PTWF_ARGS_SHADOW / 4 + 1}] = gs.y;
    ptwfArgs[${PTWF_ARGS_SHADOW / 4 + 2}] = gs.z;
    atomicStore(&ptwfCount[cur], 0u);
    atomicStore(&ptwfCount[2u + nxt], 0u);
}
`;
}

/**
 * Shade kernel: the body of the megakernel's bounce loop for one queued path (`ptPath`, integrator-wgsl.ts), with the shadow ray queued
 * instead of traced. Needs the scene copy (surface, materials, lights), the BSDF modules, the sampler and the shared integrator head.
 */
export function ptwfShadeWgsl(options: PtIntegratorOptions = {}): string {
  const o = resolveIntegratorOptions(options);
  const tex = o.textures;
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
  // Counters: the path's end (reason) is recorded where it returns; each queued extension / shadow ray where the shade kernel sees it.
  const end = (reason: number): string => (counters ? `        ptCountEnd(path, depth, ${reason}u);\n` : '');
  return /* wgsl */ `
${PT_PARAMS_WGSL}
${bindings(['params', 'accum', 'state', 'hit', 'count', 'queue', 'shadow'])}
${ptCommonHeadWgsl(o)}
${PT_MIS_WGSL}

${PT_ESCAPED_WGSL}
${COMMON}${stateWgsl(o)}
struct PtwfLd {
    valid: bool,
    contrib: vec3f,
    ray: RtRay,
}

// ptSampleLd of the megakernel up to the visibility test: returns the unoccluded contribution and the shadow ray.
fn ptwfSampleLd(p: vec3f, ng: vec3f, frame: PtFrame, wo: vec3f, bp: PtBsdfParams, pixelSeed: u32, sampleIndex: u32, vertex: u32) -> PtwfLd {
    let none = PtwfLd(false, vec3f(0.0), RtRay(vec3f(0.0), 0.0, vec3f(0.0, 0.0, 1.0), 0.0));
    let selectU = ptSample2D(pixelSeed, sampleIndex, ptVertexPair(vertex, PT_SLOT_SELECT_COMPONENT)).x;
    let pointU = ptSample2D(pixelSeed, sampleIndex, ptVertexPair(vertex, PT_SLOT_LIGHT_POINT));
    let pick = rtSampleLight(selectU);
    if (pick.index == RT_NONE || !(pick.pmf > 0.0)) {
        return none;
    }
    if (!PT_ENV_SAMPLING && pick.index == rtSceneTable[2].z) {
        return none;
    }
    let li = rtSampleLightLi(pick.index, p, pointU);
    if (!li.valid || !(li.pdf > 0.0)) {
        return none;
    }
    let wiLocal = ptToLocal(frame, li.wi);
    let e = ${evalFn}(bp, wo, wiLocal);
    let f = e.f * abs(wiLocal.z);
    if (!(ptMax3(f) > 0.0)) {
        return none;
    }
    let origin = ptOffsetOrigin(p, ptSideNormal(ng, li.wi));
    var ray: RtRay;
    if (li.distance < 1.0e30) {
        let end = ptOffsetOrigin(li.point, ptSideNormal(li.normal, -li.wi));
        ray = rtSegment(origin, end, 0.0, 0.0);
    } else {
        ray = RtRay(origin, 0.0, li.wi, PT_FAR);
    }
    let pl = pick.pmf * li.pdf;
    var w = 1.0;
    if (!li.delta) {
        w = ptMis(pl, e.pdf);
    }
    return PtwfLd(true, li.radiance * f * (w / pl), ray);
}

fn ptwfAddRadiance(path: u32, l: vec3f) {
    if (!(l.x == 0.0 && l.y == 0.0 && l.z == 0.0)) {
        ptAccum[path] = ptAccum[path] + vec4f(l, 0.0);
    }
}

fn ptwfShadePath(path: u32) {
    let dims = ptParams.dims;
    let n = dims.x * dims.y;
    let maxDepth = ptParams.limits.x;
    let rrStart = ptParams.limits.y;
    let iter = ptParams.limits.w;
    let sampleIndex = dims.z;
    let pixelSeed = ptPixelSeed(path % dims.x, path / dims.x, dims.w);

    let st = ptwfLoadPath(path, n);
    let ray = RtRay(st.origin, 0.0, st.dir, PT_FAR);
    var beta = st.beta;
    var specularBounce = st.specular;
    var depth = st.depth;
    var etaScale = st.etaScale;
    var prevP = st.prevP;
    var prevPdf = st.prevPdf;
${medium ? `    var sigma = st.sigma;
` : ''}${tex ? `    var cone = RtCone(st.coneWidth, st.coneSpread);
` : ''}    var radiance = vec3f(0.0);

    let h0 = ptwfHit[path];
    let h1 = ptwfHit[n + path];
    let h2 = ptwfHit[2u * n + path];
    let hit = RtHit(bitcast<f32>(h0.x), h0.y, h0.z, h1.z != 0u, bitcast<vec2f>(h1.xy), h0.w, ptwfF3(h2), 0.0);
    if (hit.t < 0.0) {
        radiance += beta * ptEscaped(ray.direction, depth == 0u || specularBounce, prevPdf);
        ptwfAddRadiance(path, radiance);
${counters ? `        ptCountRay(path, 1u, 0u);
        ptCountEnd(path, depth, 0u);
` : ''}        return;
    }${counters ? `
    ptCountRay(path, 1u, 0u);` : ''}${aovs ? `
    var aovDone = st.aovDone;` : ''}
${tex ? `    let surf = rtSurface(hit, ray, RT_SURF_POSITION | RT_SURF_SHADING_NORMAL | RT_SURF_UV | RT_SURF_TEXFRAME${aniso ? ' | RT_SURF_TANGENT' : ''});
    let coneW = rtConeAt(cone, hit.t);
    let fp = select(0.0, coneW * surf.uvDensity / max(abs(dot(hit.normal, ray.direction)), 0.001), PT_TEX_CONE);
    let mat = rtApplyMaterialTextures(rtMaterial(hit.material), surf.uv, fp);
    var shadingN = surf.shadingNormal;
    if (mat.normalTexture != RT_TEXTURE_NONE) {
        let nr = rtTexRecord(mat.normalTexture);
        shadingN = rtApplyNormalMap(surf.shadingNormal, surf.tangent, rtTexSampleRecord(nr, surf.uv, rtTexLodOf(nr, fp)).xyz);
    }` : `    let surf = rtSurface(hit, ray, RT_SURF_POSITION | RT_SURF_SHADING_NORMAL${aniso ? ' | RT_SURF_TANGENT' : ''});
    let mat = rtMaterial(hit.material);`}
${medium ? `    beta = beta * exp(-sigma * hit.t);
` : ''}
${hooks ? `    var hookEmitted = vec3f(0.0);
` : ''}    if (mat.emissionLuminance > 0.0 && (hit.frontFace || mat.emissionTwoSided)) {
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
${hooks ? `        hookEmitted = beta * mat.emissionColor * w;
` : ''}    }

    let bp = ptBsdfFromMaterial(mat${mats ? `, ${mixU}` : ''});
${film ? `    ptFilm = select(vec3f(0.0), ptMaterialFilm(hit.material), mat.coatWeight <= 0.5 && mat.transmissionWeight <= 0.5);
` : ''}${fuzz ? `    ptFuzzW = mat.sheenWeight;
    ptFuzzC = vec4f(mat.sheenColor, mat.sheenRoughness);
` : ''}${aovs ? `    if (depth == 0u) {
        ptAovPrimary(path, n, -dot(surf.position - ptParams.origin.xyz, normalize(ptParams.forward.xyz)));
    }
    if (!aovDone && !ptBsdfIsDelta(bp)) {
        aovDone = true;
        ptAovGuide(path, n, ptAovAlbedo(mat), ${shadingNormal});
    }
` : ''}    if (${black} || depth >= maxDepth) {
        ptwfAddRadiance(path, radiance);
${counters ? `        ptCountEnd(path, depth, select(2u, 1u, ${black}));
` : ''}        return;
    }
    let vertex = depth;
    depth += 1u;
${layered ? `    ptLayerSeed = pixelSeed ^ (sampleIndex * 2654435761u) ^ (vertex * 2246822519u);
` : ''}
${aniso ? `    var frame = ptFrameFromNormal(${shadingNormal});
    if (bp.kind == PT_BSDF_CONDUCTOR && mat.specularAnisotropy > 0.0) {
        frame = ptFrameFromTangent(${shadingNormal}, surf.vertexTangent, mat.anisotropyRotation * 6.283185307179586);
    }
` : `    let frame = ptFrameFromNormal(${shadingNormal});
`}    let wo = ptToLocal(frame, -ray.direction);
    if (PT_NEE && ${fuzz ? '(!ptBsdfIsDelta(bp) || ptFuzzActive(wo.z))' : '!ptBsdfIsDelta(bp)'}) {
        let ld = ptwfSampleLd(surf.position, hit.normal, frame, wo, bp, pixelSeed, sampleIndex, vertex);
        if (ld.valid) {
            ptwfShadowRay[path] = vec4u(bitcast<vec3u>(ld.ray.origin), bitcast<u32>(ld.ray.tMin));
            ptwfShadowRay[n + path] = vec4u(bitcast<vec3u>(ld.ray.direction), bitcast<u32>(ld.ray.tMax));
            ptwfShadowRay[2u * n + path] = vec4u(bitcast<vec3u>(beta * ld.contrib), 0u);
            let ss = atomicAdd(&ptwfCount[2u + (iter & 1u)], 1u);
            ptwfQueue[2u * n + ss] = path;
${counters ? `            ptCountRay(path, 0u, 1u);
` : ''}        }
    }
${hooks ? `    let hv = ${ptHookVertexExpr({ x: 'path % dims.x', y: 'path / dims.x', pixel: 'path', sampleIndex: 'sampleIndex' })};
    if (ptOnVertex(hv) == PT_HOOK_TERMINATE) {
        radiance += beta * ptTerminate(hv);
        ptwfAddRadiance(path, radiance);
${counters ? `        ptCountEnd(path, depth, 4u);
` : ''}        return;
    }
` : ''}
    let dirU = ptSample2D(pixelSeed, sampleIndex, ptVertexPair(vertex, PT_SLOT_BSDF_DIRECTION));
    let compU = ptSample2D(pixelSeed, sampleIndex, ptVertexPair(vertex, PT_SLOT_SELECT_COMPONENT)).y;
    let s = ${sampleFn}(bp, wo, vec3f(compU, dirU.x, dirU.y));
    if (s.flags == 0u || !(s.pdf > 0.0)) {
        ptwfAddRadiance(path, radiance);
${end(5)}        return;
    }
    beta = beta * s.f * (abs(s.wi.z) / s.pdf);
    specularBounce = (s.flags & PT_FLAG_DELTA) != 0u;
${tex ? `    cone = RtCone(coneW, cone.spread + select(PT_CONE_DIFFUSE_SPREAD, 0.0, specularBounce));
` : ''}    if ((s.flags & PT_FLAG_TRANSMISSION) != 0u) {
        etaScale = etaScale * s.etap * s.etap;
${medium ? `        sigma = select(vec3f(0.0), ptMediumSigma(mat), wo.z > 0.0);
` : ''}    }
    prevPdf = s.pdf;${stochastic ? `
    if ((s.flags & PT_FLAG_PROPORTIONAL) != 0u) {
        prevPdf = ptLayeredPdf(ptLayeredOf(bp), wo, s.wi, ptLayerSeed);
    }` : ''}
    prevP = surf.position;
    let wiWorld = ptFromLocal(frame, s.wi);
    let nextOrigin = ptOffsetOrigin(surf.position, ptSideNormal(hit.normal, wiWorld));

    if (rrStart > 0u && depth >= rrStart) {
        let rrBeta = beta * etaScale;
        let m = ptMax3(rrBeta);
        if (m < 1.0) {
            let q = max(0.0, 1.0 - m);
            if (ptSample2D(pixelSeed, sampleIndex, ptVertexPair(vertex, PT_SLOT_RR)).x < q) {
                ptwfAddRadiance(path, radiance);
${counters ? `                ptCountEnd(path, depth, 3u);
` : ''}                return;
            }
            beta = beta / (1.0 - q);
        }
    }

    ptwfAddRadiance(path, radiance);
    var next: PtwfPath;
    next.origin = nextOrigin;
    next.prevPdf = prevPdf;
    next.dir = wiWorld;
    next.etaScale = etaScale;
    next.beta = beta;
    next.depth = depth;
    next.specular = specularBounce;
    next.prevP = prevP;
    next.sigma = ${medium ? 'sigma' : 'vec3f(0.0)'};
    next.coneWidth = ${tex ? 'cone.width' : '0.0'};
    next.coneSpread = ${tex ? 'cone.spread' : '0.0'};
${aovs ? `    next.aovDone = aovDone;
` : ''}    ptwfStorePath(path, n, next);
    let nq = (iter + 1u) & 1u;
    let slot = atomicAdd(&ptwfCount[nq], 1u);
    ptwfQueue[nq * n + slot] = path;
}

@compute @workgroup_size(${PTWF_WORKGROUP})
fn ptwfShade(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32) {
    let iter = ptParams.limits.w;
    let q = iter & 1u;
    let slot = ptwfSlot(wid, li);
    if (slot >= atomicLoad(&ptwfCount[q])) {
        return;
    }
    let n = ptParams.dims.x * ptParams.dims.y;
    ptwfShadePath(ptwfQueue[q * n + slot]);
}
`;
}
