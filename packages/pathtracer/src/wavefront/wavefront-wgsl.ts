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
 * Alpha MASK (P3.7): the intersect kernel loops over cutout hits like the megakernel's bounce loop does (a masked-out hit moves the ray origin past it, updates
 * the cone and the medium attenuation in the path state, costs no depth), so shade only ever sees a solid hit, a miss or a path that overran `PT_MASK_SKIPS`
 * (`hit.w` = 1: the path ends without radiance); the shadow kernel is `ptVisibleMasked`. Both then bind the scene copy (walk 3 + scene 2 + 4 buffers = 9 <= 10).
 *
 * Not yet (P3.3+): compaction, octant binning, material sorting, subgroup-aggregated appends. Refill and layers (P3.2) are the `PtwfConfig` variants of the
 * kernels: inline refill restarts a finished slot in the shade kernel, batch refill dispatches the camera kernel over the film from the args kernel's decision.
 */
import { PT_ESCAPED_WGSL, PT_LOD_TEX_BIAS, PT_MIS_WGSL, PT_PARAMS_WGSL, PT_VISIBLE_MASKED_WGSL, ptCommonHeadWgsl, ptLodSnippets, ptMaskIndirect, ptMaskShadow, ptShadowNeedsScene, ptShadowWeighted, ptShadowWeightWgsl, resolveIntegratorOptions, type PtIntegratorOptions, type ResolvedIntegratorOptions } from '../integrator/integrator-wgsl.ts';
import { ptFrameAddWgsl, ptHookVertexExpr, ptHooksWgsl, ptHookWantsRadiance } from '../integrator/outputs-wgsl.ts';
import { PT_RESTIR_SHADE_BINDING, PT_RESTIR_UNPACK_WGSL } from '../restir/restir-wgsl.ts';
import { PTWF_ARGS_MAIN, PTWF_ARGS_REFILL, PTWF_ARGS_SHADOW, PTWF_COUNT_IDLE, PTWF_INIT, PTWF_START_ITER, PTWF_STRIDE, PTWF_WORKGROUP } from './layout.ts';

/** Why a scene or option cannot run on the wavefront scheduler yet (slice 1), or null. */
export function ptwfUnsupported(options: PtIntegratorOptions): string | null {
  const o = resolveIntegratorOptions(options);
  if (o.volumes) return 'integrator.volumes (participating media, P5: the megakernel only; the wavefront needs a medium stage between intersect and shade, see docs/design/volumes.md)';
  // P4.1b: `sampler: 'bluenoise'` and `integrator.frame` run on the wavefront (the frame planes live in the accumulation buffer's tail, like the AOVs: the two cannot be combined).
  if (o.frame && o.aovs) return 'integrator.frame with integrator.aovs (both live in the tail of the accumulation buffer; the real-time mode has no AOVs)';
  if (o.restir && o.alphaMask) return "integrator.restir with alpha MASK (the megakernel tests reservoir visibility with the exact cutout walk; the wavefront's shadow kernel follows the shadow policy)";
  if (o.restir && o.restirShared) return 'integrator.restir.sharedHit (the shared primary hit joins the wavefront camera kernel: TASKS P4.2a follow-up)';
  if (o.restir && o.hooks) return 'integrator.restir with hooks';
  if (o.alphaMask && o.counters) return 'integrator.counters with alpha MASK (the cutout re-traces happen in the intersect kernel, which has no counter buffer)';
  return null;
}

/** Scheduling variants of the kernels (P3.2). `refill`: a finished path starts the slot's next sample inline; `layers`: slots per pixel (needs `refill`). */
export interface PtwfConfig {
  refill: boolean;
  /** Batch refill (needs `refill`): finished slots wait in an idle queue until the live queue drops below `threshold` x slots, then one dispatch starts them all. */
  batch: boolean;
  threshold: number;
  layers: number;
  /** P3.3: queue appends of the shade kernel are aggregated per subgroup (one `atomicAdd` per subgroup, lane order kept); needs the `subgroups` feature. */
  subgroup?: boolean;
  /** P3.10: `refill: 'ordered'`: the shade kernel starts the next sample inline but only flags the slot; the `primary` kernel appends the flagged slots in slot (pixel) order. Needs `refill`. */
  ordered?: boolean;
  /** P3.4: the `sortHit` kernel (counting sort of each 64-entry block of the queue by material, between intersect and shade) is built and dispatched. */
  sortHit?: boolean;
  /** P3.4: the `sortRay` kernel (block sort of the next queue by ray-direction octant, before the next intersect) is built and dispatched. */
  sortRay?: boolean;
  /** R1b: with refill, a queue mixes depths, so the two shade kernels of `materialLod: 'secondary'` each skip the entries of the other class (`first`: depth < `depth`, `second`: depth >= `depth`). */
  lodGate?: { role: 'first' | 'second'; depth: number };
  /**
   * R1 x alpha MASK x cone LOD: the INTERSECT kernel decides the cutouts, and the megakernel decides them with the same widened footprint (`PT_LOD_TEX_BIAS`) it shades with at
   * depth >= `materialLodDepth` (its `mat` is one texture read per hit). The intersect kernel is built with `materialLod: 'full'`, so the depth is passed here (set only when
   * `materialLod: 'secondary'`, textures, cone LOD and alpha MASK are all on; absent = the intersect text is unchanged).
   */
  lodTexDepth?: number;
}
export const PTWF_PLAIN: PtwfConfig = { refill: false, batch: false, threshold: 0.25, layers: 1 };

const f32Literal = (v: number): string => (Number.isInteger(v) ? `${v}.0` : String(v));

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

const commonWgsl = (wf: PtwfConfig): string => /* wgsl */ `
const PTWF_WG: u32 = ${PTWF_WORKGROUP}u;
const PTWF_STRIDE: u32 = ${PTWF_STRIDE}u;
const PTWF_INIT: u32 = ${PTWF_INIT}u;
${wf.refill ? `const PTWF_LAYERS: u32 = ${wf.layers}u;
` : ''}${wf.lodGate ? `const PTWF_CONSUMED: u32 = 0xffffffffu;
` : ''}
// path slots: the stride of every SoA section (one per pixel, or layers per pixel in refill mode)
fn ptwfSlots(dims: vec4u) -> u32 {
    return dims.x * dims.y${wf.refill ? ' * PTWF_LAYERS' : ''};
}

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
  // P4.5a: `frameEmission: 'mirror'` carries one bit per path: has it left the chain of perfect-mirror bounces from the primary hit? Bit 28 of the depth word (set = impure, so a fresh path's word
  // is unchanged and `(word & 0x1fffffff) == 0` still means depth 0 at the primary), which takes the depth down to 28 bits in such kernels.
  const impure = o.frame && o.frameEmission === 'mirror';
  const depthMask = impure ? '0x0fffffffu' : '0x1fffffffu';
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
` : ''}${impure ? `    impure: bool,
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
    p.depth = s2.w & ${depthMask};
    p.specular = (s2.w >> 31u) == 1u;${aovs ? `
    p.aovDone = ((s2.w >> 30u) & 1u) == 1u;` : ''}${impure ? `
    p.impure = ((s2.w >> 28u) & 1u) == 1u;` : ''}
    p.prevP = ptwfF3(s3);
    p.coneSpread = ${tex ? 'bitcast<f32>(s3.w)' : '0.0'};
    p.sigma = ${medium ? 'ptwfF3(s4)' : 'vec3f(0.0)'};
    p.coneWidth = ${tex ? 'bitcast<f32>(s4.w)' : '0.0'};
    return p;
}

fn ptwfStorePath(path: u32, n: u32, p: PtwfPath) {
    ptwfState[path] = vec4u(bitcast<vec3u>(p.origin), bitcast<u32>(p.prevPdf));
    ptwfState[n + path] = vec4u(bitcast<vec3u>(p.dir), bitcast<u32>(p.etaScale));
    ptwfState[2u * n + path] = vec4u(bitcast<vec3u>(p.beta), p.depth | 0x20000000u | select(0u, 0x80000000u, p.specular)${aovs ? ' | select(0u, 0x40000000u, p.aovDone)' : ''}${impure ? ' | select(0u, 0x10000000u, p.impure)' : ''});
    ptwfState[3u * n + path] = vec4u(bitcast<vec3u>(p.prevP), ${tex ? 'bitcast<u32>(p.coneSpread)' : '0u'});
${extra ? `    ptwfState[4u * n + path] = vec4u(${medium ? 'bitcast<vec3u>(p.sigma)' : 'vec3u(0u)'}, ${tex ? 'bitcast<u32>(p.coneWidth)' : '0u'});
` : ''}}
`;
}

/** Starts sample `sampleIndex` in slot `path`: jitter, primary ray, path state (the camera kernel's body, shared by the start pass and the shade kernel's inline refill). */
function ptwfStartPathWgsl(o: ResolvedIntegratorOptions): string {
  const tex = o.textures;
  return /* wgsl */ `
fn ptwfStartPath(path: u32, n: u32, sampleIndex: u32) {
    let dims = ptParams.dims;
    let pix = path % (dims.x * dims.y);
    let px = pix % dims.x;
    let py = pix / dims.x;
    let pixelSeed = ptPixelSeed(px, py, dims.w);
${o.sampler === 'bluenoise' ? `    ptBnPixel = vec2u(px, py);
` : ''}    let jitter = ptSample2D(pixelSeed, sampleIndex, PT_PAIR_JITTER);
    let ndc = vec2f(((f32(px) + jitter.x) / f32(dims.x)) * 2.0 - 1.0, 1.0 - ((f32(py) + jitter.y) / f32(dims.y)) * 2.0);
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
    p.coneSpread = ${tex ? 'bitcast<f32>(ptParams.limits.z) * 2.0 * length(ptParams.up.xyz) / (length(ptParams.forward.xyz) * f32(dims.y))' : '0.0'};
    ptwfStorePath(path, n, p);
}
`;
}

/** Camera kernel: the start of `ptPath` of the megakernel, writing path state instead of tracing. Needs the sampler. */
export function ptwfCameraWgsl(options: PtIntegratorOptions = {}, wf: PtwfConfig = PTWF_PLAIN): string {
  const o = resolveIntegratorOptions(options);
  const tex = o.textures;
  if (wf.refill) {
    return /* wgsl */ `
${PT_PARAMS_WGSL}
${bindings(['params', 'accum', 'state', 'count', 'queue'])}
const PT_FAR: f32 = 1.0e30;
${commonWgsl(wf)}${stateWgsl(o)}${ptwfStartPathWgsl(o)}
// start pass of an encode: every dormant slot that still has a sample to start (target dims.z) starts it and joins the queue of the first iteration
@compute @workgroup_size(8, 8)
fn ptwfCamera(@builtin(global_invocation_id) id: vec3u) {
    let dims = ptParams.dims;
    if (id.x >= dims.x || id.y >= dims.y) {
        return;
    }
    let n = ptwfSlots(dims);
    let path = id.z * dims.x * dims.y + id.y * dims.x + id.x;
    if ((ptwfState[2u * n + path].w & 0x20000000u) != 0u) {
        return;
    }
    let started = u32(ptAccum[path].w);
    let sampleIndex = started * PTWF_LAYERS + id.z;
    if (sampleIndex >= dims.z) {
        return;
    }
    ptwfStartPath(path, n, sampleIndex);
    ptAccum[path].w = f32(started + 1u);
    let cur = ptParams.limits.w & 1u;
    let slot = atomicAdd(&ptwfCount[cur], 1u);
    ptwfQueue[cur * n + slot] = path;
}
`;
  }
  return /* wgsl */ `
${PT_PARAMS_WGSL}
${bindings(['params', 'accum', 'state', 'queue'])}
const PT_FAR: f32 = 1.0e30;
${commonWgsl(wf)}${stateWgsl(o)}
@compute @workgroup_size(8, 8)
fn ptwfCamera(@builtin(global_invocation_id) id: vec3u) {
    let dims = ptParams.dims;
    if (id.x >= dims.x || id.y >= dims.y) {
        return;
    }
    let path = id.y * dims.x + id.x;
    let n = ptwfSlots(dims);
    let pixelSeed = ptPixelSeed(id.x, id.y, dims.w);
${o.sampler === 'bluenoise' ? `    ptBnPixel = id.xy;
` : ''}    let jitter = ${o.primary === 'vbuffer' ? 'vec2f(ptParams.origin.w, ptParams.right.w)' : 'ptSample2D(pixelSeed, dims.z, PT_PAIR_JITTER)'};
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
    ptAccum[path] = ${o.accumulate ? 'ptAccum[path] + ' : ''}vec4f(0.0, 0.0, 0.0, 1.0);
    ptwfQueue[path] = path;
}
`;
}

/**
 * Intersect kernel: closest hit of every ray in `Q[iter & 1]`. Needs the raytracer module (walk, no scene), or with alpha MASK the module with the scene copy
 * (cutouts are decided here, so shade never sees a masked-out hit).
 */
export function ptwfIntersectWgsl(options: PtIntegratorOptions = {}, wf: PtwfConfig = PTWF_PLAIN): string {
  const o = resolveIntegratorOptions(options);
  if (o.alphaMask) return ptwfIntersectMaskWgsl(o, wf);
  if (o.primary === 'vbuffer') {
    return /* wgsl */ `
${PT_PARAMS_WGSL}
${bindings(['params', 'state', 'hit', 'count', 'queue'])}
const PT_FAR: f32 = 1.0e30;
${commonWgsl(wf)}${stateWgsl(o)}
@compute @workgroup_size(${PTWF_WORKGROUP})
fn ptwfIntersect(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32) {
    let iter = ptParams.limits.w;
    let q = iter & 1u;
    let slot = ptwfSlot(wid, li);
    if (slot >= atomicLoad(&ptwfCount[q])) {
        return;
    }
    let n = ptwfSlots(ptParams.dims);
    let path = ptwfQueue[q * n + slot];
    let o = ptwfState[path];
    let d = ptwfState[n + path];
    let ray = RtRay(ptwfF3(o), 0.0, ptwfF3(d), PT_FAR);
    var hit: RtHit;
    if ((ptwfState[2u * n + path].w & 0x1fffffffu) == 0u) {
        hit = ptVbufferHit(vec2u(path % ptParams.dims.x, path / ptParams.dims.x), ray); // the primary hit: from the V-buffer, not traced
    } else {
        hit = rtClosest(ray, RT_MASK_ALL);
    }
    ptwfHit[path] = vec4u(bitcast<u32>(hit.t), hit.instance, hit.triangle, hit.material);
    ptwfHit[n + path] = vec4u(bitcast<vec2u>(hit.bary), select(0u, 1u, hit.frontFace), 0u);
    ptwfHit[2u * n + path] = vec4u(bitcast<vec3u>(hit.normal), 0u);
}
`;
  }
  return /* wgsl */ `
${PT_PARAMS_WGSL}
${bindings(['params', 'state', 'hit', 'count', 'queue'])}
const PT_FAR: f32 = 1.0e30;
${commonWgsl(wf)}${stateWgsl(o)}
@compute @workgroup_size(${PTWF_WORKGROUP})
fn ptwfIntersect(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32) {
    let iter = ptParams.limits.w;
    let q = iter & 1u;
    let slot = ptwfSlot(wid, li);
    if (slot >= atomicLoad(&ptwfCount[q])) {
        return;
    }
    let n = ptwfSlots(ptParams.dims);
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

/**
 * Intersect with alpha MASK: the megakernel's skip step (`ptPath`: a hit whose base-colour alpha times opacity is below the cutoff is passed through) as a loop
 * around the closest-hit query. The path state (origin, ray cone, medium attenuation of the skipped segment) is updated like the megakernel's locals; the
 * skip count is per ray here (per path there), the one visible difference, and only past 64 skips. Depth 0 always tests the cutout; deeper rays only when
 * the alpha policy says so (`ptMaskIndirect`). A path over the skip limit is marked in `hit.w` (1) and ends in shade.
 */
function ptwfIntersectMaskWgsl(o: ResolvedIntegratorOptions, wf: PtwfConfig): string {
  const medium = !!o.materials?.features.medium;
  const indirect = ptMaskIndirect(o);
  return /* wgsl */ `
${PT_PARAMS_WGSL}
${bindings(['params', 'state', 'hit', 'count', 'queue'])}
${ptCommonHeadWgsl(o)}
${commonWgsl(wf)}${stateWgsl(o)}
@compute @workgroup_size(${PTWF_WORKGROUP})
fn ptwfIntersect(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32) {
    let iter = ptParams.limits.w;
    let q = iter & 1u;
    let slot = ptwfSlot(wid, li);
    if (slot >= atomicLoad(&ptwfCount[q])) {
        return;
    }
    let n = ptwfSlots(ptParams.dims);
    let path = ptwfQueue[q * n + slot];
    var st = ptwfLoadPath(path, n);
    var ray = RtRay(st.origin, 0.0, st.dir, PT_FAR);
${o.primary === 'vbuffer' ? `    var hit: RtHit;
    if (st.depth == 0u) {
        hit = ptVbufferHit(vec2u(path % ptParams.dims.x, path / ptParams.dims.x), ray); // the primary hit: from the V-buffer; a cutout texel continues by tracing below
    } else {
        hit = rtClosest(ray, RT_MASK_ALL);
    }
` : `    var hit = rtClosest(ray, RT_MASK_ALL);
`}    var skips = 0u;
    var over = 0u;
    var moved = false;
    loop {
        if (hit.t < 0.0${indirect ? '' : ' || st.depth > 0u'}) {
            break;
        }
        let m = rtMaterial(hit.material);
        if (m.alphaMode != RT_ALPHA_MASK) {
            break;
        }
        let surf = rtSurface(hit, ray, RT_SURF_POSITION | RT_SURF_UV | RT_SURF_TEXFRAME);
        let coneW = rtConeAt(RtCone(st.coneWidth, st.coneSpread), hit.t);
        let fp = select(0.0, coneW * surf.uvDensity / max(abs(dot(hit.normal, ray.direction)), 0.001)${wf.lodTexDepth !== undefined ? ` * select(1.0, ${PT_LOD_TEX_BIAS}, st.depth >= ${wf.lodTexDepth}u)` : ''}, PT_TEX_CONE);
        let a = rtApplyMaterialTextures(m, surf.uv, fp);
        if (!(a.opacity < a.alphaCutoff)) {
            break;
        }
${medium ? `        st.beta = st.beta * exp(-st.sigma * hit.t);
` : ''}        skips += 1u;
        if (skips > PT_MASK_SKIPS) {
            over = 1u;
            break;
        }
        st.coneWidth = coneW;
        st.origin = ptOffsetOrigin(surf.position, ptSideNormal(hit.normal, ray.direction));
        moved = true;
        ray = RtRay(st.origin, 0.0, ray.direction, PT_FAR);
        hit = rtClosest(ray, RT_MASK_ALL);
    }
    if (moved) {
        ptwfStorePath(path, n, st);
    }
    ptwfHit[path] = vec4u(bitcast<u32>(hit.t), hit.instance, hit.triangle, hit.material);
    ptwfHit[n + path] = vec4u(bitcast<vec2u>(hit.bary), select(0u, 1u, hit.frontFace), over);
    ptwfHit[2u * n + path] = vec4u(bitcast<vec3u>(hit.normal), 0u);
}
`;
}

/** Shadow kernel: visibility of every shadow ray; a visible one adds its contribution to the pixel. Needs the raytracer module (walk; the scene copy with `ptVisibleMasked`, alpha MASK with an exact shadow policy). */
export function ptwfShadowWgsl(options: PtIntegratorOptions = {}, wf: PtwfConfig = PTWF_PLAIN): string {
  const o = resolveIntegratorOptions(options);
  const masked = ptMaskShadow(o);
  // The optional radiance callback: the shade kernel stored (vertex, flags, sample, pixel) in the 4th shadow-ray section.
  const onRadiance = ptHookWantsRadiance(options.hooks);
  const weighted = ptShadowWeighted(o);
  return /* wgsl */ `
${PT_PARAMS_WGSL}
${bindings(['params', 'accum', 'count', 'queue', 'shadow'])}
${masked || weighted ? `${ptCommonHeadWgsl(o)}
${masked || ptShadowNeedsScene(o) ? PT_VISIBLE_MASKED_WGSL : ''}${ptShadowWeightWgsl(o)}` : ''}${onRadiance ? `${ptHooksWgsl(options.hooks ?? null)}\n` : ''}${o.frame ? `${ptFrameAddWgsl()}\n` : ''}${commonWgsl(wf)}
@compute @workgroup_size(${PTWF_WORKGROUP})
fn ptwfShadow(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32) {
    let iter = ptParams.limits.w;
    let slot = ptwfSlot(wid, li);
    if (slot >= atomicLoad(&ptwfCount[2u + (iter & 1u)])) {
        return;
    }
    let n = ptwfSlots(ptParams.dims);
    let path = ptwfQueue[2u * n + slot];
    let r0 = ptwfShadowRay[path];
    let r1 = ptwfShadowRay[n + path];
    let r2 = ptwfShadowRay[2u * n + path];
${weighted ? `    let vis = ptShadowWeight(RtRay(ptwfF3(r0), bitcast<f32>(r0.w), ptwfF3(r1), bitcast<f32>(r1.w)), vec3u(path ^ r0.x, r1.x, r0.z ^ r1.y));
    if (vis > 0.0) {
        ptAccum[path] = ptAccum[path] + vec4f(ptwfF3(r2) * vis, 0.0);
${o.frame ? `        ptFrameAdd(path, n, ptwfF3(r2) * vis);
` : ''}${onRadiance ? `        let info = ptwfShadowRay[3u * n + path];
        ptOnRadiance(info.w, info.z, info.x, info.y, ptwfF3(r2) * vis);
` : ''}    }
` : `    if (${masked ? 'ptVisibleMasked' : 'rtVisible'}(RtRay(ptwfF3(r0), bitcast<f32>(r0.w), ptwfF3(r1), bitcast<f32>(r1.w)), RT_MASK_ALL)) {
        ptAccum[path] = ptAccum[path] + vec4f(ptwfF3(r2), 0.0);
${o.frame ? `        ptFrameAdd(path, n, ptwfF3(r2));
` : ''}${onRadiance ? `        let info = ptwfShadowRay[3u * n + path];
        ptOnRadiance(info.w, info.z, info.x, info.y, ptwfF3(r2));
` : ''}    }
`}}
`;
}

/** Args kernel (one thread): queue counts -> indirect dispatch args; zeroes the counters nobody reads any more (`ptwfArgsStep` is its twin). Standalone module. */
export function ptwfArgsWgsl(wf: PtwfConfig = PTWF_PLAIN): string {
  return /* wgsl */ `
${PT_PARAMS_WGSL}
${bindings(['params', 'count'])}
@group(3) @binding(0) var<storage, read_write> ptwfArgs: array<u32>;
${wf.batch ? `const PTWF_LAYERS: u32 = ${wf.layers}u;
` : ''}const PTWF_WG: u32 = ${PTWF_WORKGROUP}u;
const PTWF_STRIDE: u32 = ${PTWF_STRIDE}u;
const PTWF_INIT: u32 = ${PTWF_INIT}u;

fn ptwfGroups(count: u32) -> vec3u {
    let g = (count + PTWF_WG - 1u) / PTWF_WG;
    return vec3u(min(g, PTWF_STRIDE), (g + PTWF_STRIDE - 1u) / PTWF_STRIDE, 1u);
}

@compute @workgroup_size(1)
fn ptwfArgsMain() {
    let iter = ptParams.limits.w;
${wf.refill ? `    if (iter >= ${PTWF_START_ITER}u) {
${wf.batch ? `        atomicStore(&ptwfCount[${PTWF_COUNT_IDLE}u], 0u); // the start pass restarts every idle slot
` : ''}        let g = ptwfGroups(atomicLoad(&ptwfCount[iter & 1u]));
        ptwfArgs[${PTWF_ARGS_MAIN / 4}] = g.x;
        ptwfArgs[${PTWF_ARGS_MAIN / 4 + 1}] = g.y;
        ptwfArgs[${PTWF_ARGS_MAIN / 4 + 2}] = g.z;
        return;
    }
` : `    if (iter == PTWF_INIT) {
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
`}
    let cur = iter & 1u;
    let nxt = cur ^ 1u;
${wf.batch ? `    let live = atomicLoad(&ptwfCount[nxt]);
    let idle = atomicLoad(&ptwfCount[${PTWF_COUNT_IDLE}u]);
    var refill = 0u;
    if (idle > 0u && (live == 0u || live < u32(f32(ptParams.dims.x * ptParams.dims.y * PTWF_LAYERS) * ${f32Literal(wf.threshold)}))) {
        refill = idle;
        atomicStore(&ptwfCount[${PTWF_COUNT_IDLE}u], 0u);
    }
    // the refill dispatch is the camera kernel over the whole film (it scans the slots in pixel order, so the restarted primary rays stay coherent)
    ptwfArgs[${PTWF_ARGS_REFILL / 4}] = select(0u, (ptParams.dims.x + 7u) / 8u, refill > 0u);
    ptwfArgs[${PTWF_ARGS_REFILL / 4 + 1}] = select(0u, (ptParams.dims.y + 7u) / 8u, refill > 0u);
    ptwfArgs[${PTWF_ARGS_REFILL / 4 + 2}] = select(0u, PTWF_LAYERS, refill > 0u);
    let gm = ptwfGroups(live + refill);
` : `    let gm = ptwfGroups(atomicLoad(&ptwfCount[nxt]));
`}    let gs = ptwfGroups(atomicLoad(&ptwfCount[2u + cur]));
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
export function ptwfShadeWgsl(options: PtIntegratorOptions = {}, wf: PtwfConfig = PTWF_PLAIN): string {
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
  const lodS = ptLodSnippets(o, '    ');
  const mixU = mats?.features.mix || (o.lod !== 'off' && mats) ? 'ptSample2D(pixelSeed, sampleIndex, ptVertexPair(depth, PT_SLOT_RR)).y' : '0.0';
  const hooks = !!o.hooks;
  // The optional radiance callback (`ptOnRadiance`): call sites only when the hook text defines it (every other kernel keeps its exact text).
  const onRadiance = ptHookWantsRadiance(o.hooks);
  const addRadiance = (value: string, index: string, indent: string): string =>
    onRadiance ? `${indent}{\n${indent}    let ptAdded = ${value};\n${indent}    radiance += ptAdded;\n${indent}    ptOnRadiance(path, sampleIndex, ${index}, ptHookFlags, ptAdded);\n${indent}}\n` : `${indent}radiance += ${value};\n`;
  const counters = o.counters;
  const aovs = o.aovs;
  const black = fuzz ? '(bp.kind == PT_BSDF_BLACK && !ptFuzzActive(1.0))' : 'bp.kind == PT_BSDF_BLACK';
  // Counters: the path's end (reason) is recorded where it returns; each queued extension / shadow ray where the shade kernel sees it.
  const end = (reason: number): string => (counters ? `        ptCountEnd(path, depth, ${reason}u);\n` : '');
  // P4.1b / P4.2a: per-frame outputs (planes in the accumulator tail, shared text `ptFrameWgsl('tail')` in the outputs), blue noise (`ptBnPixel`), ReSTIR DI at vertex 0.
  const frameOut = o.frame;
  // P4.5a: with `frameEmission` (not 'none') the primary-visible emission / sky (and, 'mirror', what perfect-mirror chains reach) goes to plane 6 instead of the demodulated planes
  const emitMode = frameOut && o.frameEmission !== 'none';
  const pureCond = o.frameEmission === 'mirror' ? '!st.impure' : 'depth == 0u';
  const addPure = (value: string, index: string, indent: string): string =>
    !emitMode
      ? addRadiance(value, index, indent)
      : onRadiance
        ? `${indent}{\n${indent}    let ptAdded = ${value};\n${indent}    if (${pureCond}) {\n${indent}        fEmit += ptAdded;\n${indent}    } else {\n${indent}        radiance += ptAdded;\n${indent}    }\n${indent}    ptOnRadiance(path, sampleIndex, ${index}, ptHookFlags, ptAdded);\n${indent}}\n`
        : `${indent}if (${pureCond}) {\n${indent}    fEmit += ${value};\n${indent}} else {\n${indent}    radiance += ${value};\n${indent}}\n`;
  const bn = o.sampler === 'bluenoise';
  const restir = o.restir;
  const addOut = emitMode ? 'ptwfAddRadianceE(path, n, radiance, fEmit, fEmitW)' : frameOut ? 'ptwfAddRadianceF(path, n, radiance, fDirect)' : 'ptwfAddRadiance(path, radiance)';
  const pixelOf = wf.refill ? 'pixel' : 'path';
  return /* wgsl */ `
${PT_PARAMS_WGSL}
${bindings(['params', 'accum', 'state', 'hit', 'count', 'queue', 'shadow'])}
${restir ? `@group(2) @binding(${PT_RESTIR_SHADE_BINDING}) var<storage, read> ptRestirRes: array<vec4u>;
` : ''}${ptCommonHeadWgsl(o)}
${PT_MIS_WGSL}

${PT_ESCAPED_WGSL}
${commonWgsl(wf)}${stateWgsl(o)}${wf.refill ? ptwfStartPathWgsl(o) : ''}
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
${emitMode ? `
// Frame mode with an emission plane (P4.5a): \`l\` = light the primary BSDF modulates (class plane, divided by its factor), \`e\` = emission that bypasses it (plane 6); \`ew\` is the part of \`e\`
// the depth-0 write already put in plane 6. The beauty gets both.
fn ptwfAddRadianceE(path: u32, n: u32, l: vec3f, e: vec3f, ew: vec3f) {
    let t = l + e;
    if (!(t.x == 0.0 && t.y == 0.0 && t.z == 0.0)) {
        ptAccum[path] = ptAccum[path] + vec4f(t, 0.0);
    }
    if (!(l.x == 0.0 && l.y == 0.0 && l.z == 0.0)) {
        ptFrameAdd(path, n, l);
    }
    let pe = e - ew;
    if (!(pe.x == 0.0 && pe.y == 0.0 && pe.z == 0.0)) {
        ptFrameAddEmit(path, n, pe);
    }
}
` : frameOut ? `
// Frame mode: \`l\` is everything this invocation gathered (it goes to the beauty); \`direct\` is the part already in the diffuse plane (the primary hit's emission, a primary
// miss's sky). The rest lands in the plane of the primary lobe's class, divided by its demodulation factor.
fn ptwfAddRadianceF(path: u32, n: u32, l: vec3f, direct: vec3f) {
    if (!(l.x == 0.0 && l.y == 0.0 && l.z == 0.0)) {
        ptAccum[path] = ptAccum[path] + vec4f(l, 0.0);
        let c = l - direct;
        if (!(c.x == 0.0 && c.y == 0.0 && c.z == 0.0)) {
            ptFrameAdd(path, n, c);
        }
    }
}
` : ''}${restir ? `
${PT_RESTIR_UNPACK_WGSL}

// ReSTIR DI (P4.2a): the first vertex's light is the final reservoir of its pixel (ptRestirLd of the megakernel up to the visibility test); the shadow kernel tests it. The region of
// the final reservoirs is in the (otherwise unused) \`up.w\` lane of the params record.
fn ptwfRestirLd(p: vec3f, ng: vec3f, frame: PtFrame, wo: vec3f, bp: PtBsdfParams, pxIndex: u32) -> PtwfLd {
    let none = PtwfLd(false, vec3f(0.0), RtRay(vec3f(0.0), 0.0, vec3f(0.0, 0.0, 1.0), 0.0));
    let r = ptRestirRes[bitcast<u32>(ptParams.up.w) + pxIndex];
    let rw = bitcast<f32>(r.z);
    if (r.x == RT_NONE || !(rw > 0.0)) {
        return none;
    }
    let li = rtSampleLightLi(r.x, p, ptRsUnpackU(r.y));
    if (!li.valid || !(li.pdf > 0.0)) {
        return none;
    }
    let wiLocal = ptToLocal(frame, li.wi);
    let f = ${evalFn}(bp, wo, wiLocal).f * abs(wiLocal.z);
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
    return PtwfLd(true, li.radiance * f * (rw / li.pdf), ray);
}
` : ''}
${wf.subgroup ? `// P3.3: the shade body only says whether the path queued a shadow ray / continues; the kernel appends once per subgroup in uniform control flow
var<private> ptwfWantShadow: bool;
var<private> ptwfWantNext: bool;

// one atomicAdd per subgroup; the lanes keep their order inside the reserved range. Returns the queue slot (valid when \`want\`).
fn ptwfAppend(want: bool, counter: u32) -> u32 {
    let one = select(0u, 1u, want);
    let rank = subgroupExclusiveAdd(one);
    let total = subgroupAdd(one);
    var base = 0u;
    if (subgroupElect()) {
        if (total > 0u) {
            base = atomicAdd(&ptwfCount[counter], total);
        }
    }
    return subgroupBroadcastFirst(base) + rank;
}

` : ''}${wf.refill ? `// the path of this thread did not continue (escaped, absorbed, depth limit, roulette): the slot starts its next sample or goes dormant
var<private> ptwfLive: bool;

fn ptwfSampleIndex(path: u32) -> u32 {
    return (u32(ptAccum[path].w) - 1u) * PTWF_LAYERS + path / (ptParams.dims.x * ptParams.dims.y);
}

fn ptwfRestart(path: u32) {
    let dims = ptParams.dims;
    let n = ptwfSlots(dims);
    let started = u32(ptAccum[path].w);
    let sampleIndex = started * PTWF_LAYERS + path / (dims.x * dims.y);
    if (sampleIndex >= dims.z) {
        ptwfState[2u * n + path].w = 0u;
        return;
    }
${wf.batch ? `    ptwfState[2u * n + path].w = 0u;
    atomicAdd(&ptwfCount[${PTWF_COUNT_IDLE}u], 1u);
` : wf.ordered ? `    ptwfStartPath(path, n, sampleIndex);
    ptAccum[path].w = f32(started + 1u);
    ptwfState[path].w = 0xffffffffu; // pending: the primary kernel appends it to the next queue in slot order
` : wf.subgroup ? `    ptwfStartPath(path, n, sampleIndex);
    ptAccum[path].w = f32(started + 1u);
    ptwfWantNext = true;
` : `    ptwfStartPath(path, n, sampleIndex);
    ptAccum[path].w = f32(started + 1u);
    let nq = (ptParams.limits.w + 1u) & 1u;
    let slot = atomicAdd(&ptwfCount[nq], 1u);
    ptwfQueue[nq * n + slot] = path;
`}}

` : ''}fn ptwfShadePath(path: u32) {
    let dims = ptParams.dims;
    let n = ptwfSlots(dims);
    let maxDepth = ptParams.limits.x;
    let rrStart = ptParams.limits.y;
    let iter = ptParams.limits.w;
${wf.refill ? `    let sampleIndex = ptwfSampleIndex(path);
    let pixel = path % (dims.x * dims.y);
    let pixelSeed = ptPixelSeed(pixel % dims.x, pixel / dims.x, dims.w);
` : `    let sampleIndex = dims.z;
    let pixelSeed = ptPixelSeed(path % dims.x, path / dims.x, dims.w);
`}${bn ? `    ptBnPixel = vec2u(${pixelOf} % dims.x, ${pixelOf} / dims.x);
` : ''}
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
${emitMode ? `    var fEmit = vec3f(0.0);
    var fEmitW = vec3f(0.0);
` : frameOut ? `    var fDirect = vec3f(0.0);
` : ''}${onRadiance ? `    let ptHookFlags = ptHookUpdateFlag(path % dims.x, path / dims.x, sampleIndex);
` : ''}
    let h0 = ptwfHit[path];
    let h1 = ptwfHit[n + path];
    let h2 = ptwfHit[2u * n + path];
    let hit = RtHit(bitcast<f32>(h0.x), h0.y, h0.z, h1.z != 0u, bitcast<vec2f>(h1.xy), h0.w, ptwfF3(h2), 0.0);
${o.alphaMask ? `    if (h1.w != 0u) {
        return; // the intersect kernel ran out of cutout skips (PT_MASK_SKIPS): the path ends, as in the megakernel
    }
` : ''}    if (hit.t < 0.0) {
${emitMode ? addPure('beta * ptEscaped(ray.direction, depth == 0u || specularBounce, prevPdf)', 'depth', '        ') : addRadiance('beta * ptEscaped(ray.direction, depth == 0u || specularBounce, prevPdf)', 'depth', '        ')}${emitMode ? `        if (depth == 0u) {
            ptFramePlanes(path, n, vec3f(0.0), vec3f(0.0), fEmit, 0.0, false, false, vec3f(0.0), 0.0, vec3f(0.0), 0.0, vec3f(1.0), vec3f(1.0));
            fEmitW = fEmit;
        }
` : frameOut ? `        if (depth == 0u) {
            ptFramePlanes(path, n, radiance, vec3f(0.0), vec3f(0.0), 0.0, false, false, vec3f(0.0), 0.0, vec3f(0.0), 0.0, vec3f(1.0), vec3f(1.0));
            fDirect = radiance;
        }
` : ''}        ${addOut};
${counters ? `        ptCountRay(path, 1u, 0u);
        ptCountEnd(path, depth, 0u);
` : ''}        return;
    }${counters ? `
    ptCountRay(path, 1u, 0u);` : ''}${aovs ? `
    var aovDone = st.aovDone;` : ''}${frameOut ? `
    if (depth == 1u) {
        ptFrameBounce(path, n, hit.t);
    }` : ''}
${lodS.decl}${tex ? `    let surf = rtSurface(hit, ray, RT_SURF_POSITION | RT_SURF_SHADING_NORMAL | RT_SURF_UV | RT_SURF_TEXFRAME${aniso ? ' | RT_SURF_TANGENT' : ''});
    let coneW = rtConeAt(cone, hit.t);
    let fp = select(0.0, coneW * surf.uvDensity / max(abs(dot(hit.normal, ray.direction)), 0.001)${lodS.texBias}, PT_TEX_CONE);
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
${emitMode ? addPure('beta * mat.emissionColor * w', 'depth', '        ') : addRadiance('beta * mat.emissionColor * w', 'depth', '        ')}${frameOut && !emitMode ? `        if (depth == 0u) {
            fDirect = beta * mat.emissionColor * w;
        }
` : ''}${hooks ? `        hookEmitted = beta * mat.emissionColor * w;
` : ''}    }

${lodS.bsdf(mixU, shadingNormal, true) ?? `    let bp = ptBsdfFromMaterial(mat${mats ? `, ${mixU}` : ''});
`}${film ? `    ptFilm = select(vec3f(0.0), ptMaterialFilm(hit.material), mat.coatWeight <= 0.5 && mat.transmissionWeight <= 0.5${lodS.and});
` : ''}${fuzz ? `    ptFuzzW = ${lodS.fuzzW};
    ptFuzzC = vec4f(mat.sheenColor, mat.sheenRoughness);
` : ''}${aovs ? `    if (depth == 0u) {
        ptAovPrimary(path, n, -dot(surf.position - ptParams.origin.xyz, normalize(ptParams.forward.xyz)));
    }
    if (!aovDone && !ptBsdfIsDelta(bp)) {
        aovDone = true;
        ptAovGuide(path, n, ptAovAlbedo(mat), ${shadingNormal});
    }
` : ''}${emitMode ? `    if (depth == 0u) {
        ptFramePlanes(path, n, vec3f(0.0), vec3f(0.0), fEmit, 0.0, true, bp.kind == PT_BSDF_CONDUCTOR || bp.kind == PT_BSDF_DIELECTRIC, ${shadingNormal}, -dot(surf.position - ptParams.origin.xyz, normalize(ptParams.forward.xyz)), ptFrameMotion(surf.position), mat.specularRoughness, ptAovAlbedo(mat), ptAovSpecular(mat));
        fEmitW = fEmit;
    }
` : frameOut ? `    if (depth == 0u) {
        ptFramePlanes(path, n, fDirect, fDirect, vec3f(0.0), 0.0, true, bp.kind == PT_BSDF_CONDUCTOR || bp.kind == PT_BSDF_DIELECTRIC, ${shadingNormal}, -dot(surf.position - ptParams.origin.xyz, normalize(ptParams.forward.xyz)), ptFrameMotion(surf.position), mat.specularRoughness, ptAovAlbedo(mat), ptAovSpecular(mat));
    }
` : ''}    if (${black} || depth >= maxDepth) {
        ${addOut};
${counters ? `        ptCountEnd(path, depth, select(2u, 1u, ${black}));
` : ''}        return;
    }
    let vertex = depth;
    depth += 1u;
${layered ? `    ptLayerSeed = pixelSeed ^ (sampleIndex * 2654435761u) ^ (vertex * 2246822519u);
` : ''}
${aniso ? `    var frame = ptFrameFromNormal(${shadingNormal});
    if (bp.kind == PT_BSDF_CONDUCTOR && mat.specularAnisotropy > 0.0${lodS.and}) {
        frame = ptFrameFromTangent(${shadingNormal}, surf.vertexTangent, mat.anisotropyRotation * 6.283185307179586);
    }
` : `    let frame = ptFrameFromNormal(${shadingNormal});
`}    let wo = ptToLocal(frame, -ray.direction);
    if (PT_NEE && ${fuzz ? '(!ptBsdfIsDelta(bp) || ptFuzzActive(wo.z))' : '!ptBsdfIsDelta(bp)'}) {
${restir ? `        var ld: PtwfLd;
        if (vertex == 0u) {
            ld = ptwfRestirLd(surf.position, hit.normal, frame, wo, bp, ${pixelOf});
        } else {
            ld = ptwfSampleLd(surf.position, hit.normal, frame, wo, bp, pixelSeed, sampleIndex, vertex);
        }
` : `        let ld = ptwfSampleLd(surf.position, hit.normal, frame, wo, bp, pixelSeed, sampleIndex, vertex);
`}        if (ld.valid) {
            ptwfShadowRay[path] = vec4u(bitcast<vec3u>(ld.ray.origin), bitcast<u32>(ld.ray.tMin));
            ptwfShadowRay[n + path] = vec4u(bitcast<vec3u>(ld.ray.direction), bitcast<u32>(ld.ray.tMax));
            ptwfShadowRay[2u * n + path] = vec4u(bitcast<vec3u>(beta * ld.contrib), 0u);
${onRadiance ? `            ptwfShadowRay[3u * n + path] = vec4u(vertex, ptHookFlags, sampleIndex, path);
` : ''}${wf.subgroup ? `            ptwfWantShadow = true;
` : `            let ss = atomicAdd(&ptwfCount[2u + (iter & 1u)], 1u);
            ptwfQueue[2u * n + ss] = path;
`}${counters ? `            ptCountRay(path, 0u, 1u);
` : ''}        }
    }
${hooks ? `    let hv = ${ptHookVertexExpr({ x: 'path % dims.x', y: 'path / dims.x', pixel: 'path', sampleIndex: 'sampleIndex' })};
    if (ptOnVertex(hv) == PT_HOOK_TERMINATE) {
${addRadiance('beta * ptTerminate(hv)', 'vertex + 1u', '        ')}        ${addOut};
${counters ? `        ptCountEnd(path, depth, 4u);
` : ''}        return;
    }
` : ''}
    let dirU = ptSample2D(pixelSeed, sampleIndex, ptVertexPair(vertex, PT_SLOT_BSDF_DIRECTION));
    let compU = ptSample2D(pixelSeed, sampleIndex, ptVertexPair(vertex, PT_SLOT_SELECT_COMPONENT)).y;
    let s = ${sampleFn}(bp, wo, vec3f(compU, dirU.x, dirU.y));
    if (s.flags == 0u || !(s.pdf > 0.0)) {
        ${addOut};
${end(5)}        return;
    }
    beta = beta * s.f * (abs(s.wi.z) / s.pdf);
    specularBounce = (s.flags & PT_FLAG_DELTA) != 0u;
${emitMode && o.frameEmission === 'mirror' ? `    let nextImpure = !ptFramePureNext(!st.impure, specularBounce, bp.kind);
` : ''}${tex ? `    cone = RtCone(coneW, cone.spread + select(PT_CONE_DIFFUSE_SPREAD, 0.0, specularBounce));
` : ''}    if ((s.flags & PT_FLAG_TRANSMISSION) != 0u) {
        etaScale = etaScale * s.etap * s.etap;
${medium ? `        sigma = select(vec3f(0.0), ptMediumSigma(mat), wo.z > 0.0);
` : ''}    }
    prevPdf = ${restir ? 'select(s.pdf, 0.0, vertex == 0u)' : 's.pdf'};${stochastic ? `
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
                ${addOut};
${counters ? `                ptCountEnd(path, depth, 3u);
` : ''}                return;
            }
            beta = beta / (1.0 - q);
        }
    }

    ${addOut};
${wf.refill ? `    ptwfLive = true;
` : ''}    var next: PtwfPath;
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
` : ''}${emitMode && o.frameEmission === 'mirror' ? `    next.impure = nextImpure;
` : ''}    ptwfStorePath(path, n, next);
${wf.subgroup ? `    ptwfWantNext = true;
` : `    let nq = (iter + 1u) & 1u;
    let slot = atomicAdd(&ptwfCount[nq], 1u);
    ptwfQueue[nq * n + slot] = path;
`}}

@compute @workgroup_size(${PTWF_WORKGROUP})
fn ptwfShade(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32) {
    let iter = ptParams.limits.w;
    let q = iter & 1u;
    let slot = ptwfSlot(wid, li);
${wf.subgroup ? ptwfShadeEntrySubgroup(wf) : `    if (slot >= atomicLoad(&ptwfCount[q])) {
        return;
    }
    let n = ptwfSlots(ptParams.dims);
${wf.refill ? `    ptwfLive = false;
    let path = ptwfQueue[q * n + slot];
${wf.lodGate ? `    if (${ptwfGateTest(wf, `(ptwfState[2u * n + path].w & 0x1fffffffu)`, true)}) {
        return;
    }
` : ''}    ptwfShadePath(path);
    if (!ptwfLive) {
        ptwfRestart(path);
    }
${wf.lodGate?.role === 'first' ? `    ptwfQueue[q * n + slot] = PTWF_CONSUMED;
` : ''}` : `    ptwfShadePath(ptwfQueue[q * n + slot]);
`}`}}
`;
}

/**
 * R1b gate: the test that is true for a queue entry this shade kernel must skip (\`skip\`), or must process. The class is the path's depth when the
 * kernel starts: the FIRST kernel (depth < N) runs before the SECOND (depth >= N) over the same queue and may raise a depth past N (or restart a slot at 0),
 * so it overwrites each entry it consumed with \`PTWF_CONSUMED\` and the second kernel skips those, instead of reading a changed state.
 */
function ptwfGateTest(wf: PtwfConfig, depthExpr: string, skip: boolean): string {
  const g = wf.lodGate as { role: 'first' | 'second'; depth: number };
  const first = g.role === 'first';
  // keep: first -> depth < D, second -> depth >= D (and not consumed)
  if (skip) return `path == PTWF_CONSUMED || ${depthExpr} ${first ? '>=' : '<'} ${g.depth}u`;
  return `path != PTWF_CONSUMED && ${depthExpr} ${first ? '<' : '>='} ${g.depth}u`;
}

/** The shade entry point of the subgroup-append variant (P3.3): no early return before the ballots (they need uniform control flow), the appends sit after the shade body. */
function ptwfShadeEntrySubgroup(wf: PtwfConfig): string {
  return `    let n = ptwfSlots(ptParams.dims);
    var path = 0u;
    var go = false;
    ptwfWantShadow = false;
    ptwfWantNext = false;
${wf.refill ? `    ptwfLive = false;
` : ''}    if (slot < atomicLoad(&ptwfCount[q])) {
        path = ptwfQueue[q * n + slot];
        go = ${wf.lodGate ? ptwfGateTest(wf, '(ptwfState[2u * n + path].w & 0x1fffffffu)', false) : 'true'};
    }
    if (go) {
        ptwfShadePath(path);
${wf.refill ? `        if (!ptwfLive) {
            ptwfRestart(path);
        }
${wf.lodGate?.role === 'first' ? `        ptwfQueue[q * n + slot] = PTWF_CONSUMED;
` : ''}` : ''}    }
    let ss = ptwfAppend(ptwfWantShadow, 2u + (iter & 1u));
    if (ptwfWantShadow) {
        ptwfQueue[2u * n + ss] = path;
    }
    let nq = (iter + 1u) & 1u;
    let ns = ptwfAppend(ptwfWantNext, nq);
    if (ptwfWantNext) {
        ptwfQueue[nq * n + ns] = path;
    }
`;
}

/** Buckets of the block sort keys (the key is `< PTWF_SORT_KEYS`; the sentinel of an empty lane is `PTWF_SORT_KEYS`). */
export const PTWF_SORT_KEYS = 16;

/**
 * Block-wise counting sort of the queue (P3.4, no global payload sort: spec section 5). One workgroup takes one 64-entry block of a queue, computes a small key per
 * entry, and writes the entries back into the SAME block ordered by key (stable inside a key: an entry's new position = the entries with a smaller key + the earlier
 * entries of its own key, found by one pass over the 64 keys in shared memory). Coherence it buys: the 32 lanes of a subgroup mostly hold one key, so shade runs
 * one material per subgroup (`kind: 'hit'`, key = miss / material id mod 15, from the hit record) and the next intersect traces one direction octant per
 * subgroup (`kind: 'ray'`, key = sign bits of the ray direction, from the path state). Each queue entry is read once before the barrier and written once after it,
 * so the in-place rewrite is safe, and no counter changes: only the order inside a block.
 *
 * 'hit' sorts `Q[iter & 1]` (dispatched between intersect and shade of an iteration with that iteration's args); 'ray' sorts `Q[(iter + 1) & 1]`, the queue shade just filled
 * (dispatched after the args kernel, whose main args are that queue's count).
 */
export function ptwfSortWgsl(kind: 'hit' | 'ray', wf: PtwfConfig = PTWF_PLAIN): string {
  const hit = kind === 'hit';
  return /* wgsl */ `
${PT_PARAMS_WGSL}
${bindings(hit ? ['params', 'hit', 'count', 'queue'] : ['params', 'state', 'count', 'queue'])}
${commonWgsl(wf)}
var<workgroup> ptwfKeys: array<u32, ${PTWF_WORKGROUP}>;

@compute @workgroup_size(${PTWF_WORKGROUP})
fn ptwfSort(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32) {
    let iter = ptParams.limits.w;
    let q = ${hit ? 'iter & 1u' : '(iter + 1u) & 1u'};
    let count = atomicLoad(&ptwfCount[q]);
    let first = (wid.y * PTWF_STRIDE + wid.x) * PTWF_WG;
    let n = ptwfSlots(ptParams.dims);
    let slot = first + li;
    let valid = slot < count;
    var path = 0u;
    var key = ${PTWF_SORT_KEYS}u;
    if (valid) {
        path = ptwfQueue[q * n + slot];
${hit ? `        let h = ptwfHit[path];
        key = select(1u + h.w % 15u, 0u, bitcast<f32>(h.x) < 0.0);
` : `        let d = ptwfF3(ptwfState[n + path]);
        key = select(0u, 1u, d.x < 0.0) | select(0u, 2u, d.y < 0.0) | select(0u, 4u, d.z < 0.0);
`}    }
    ptwfKeys[li] = key;
    workgroupBarrier();
    if (valid) {
        var pos = 0u;
        for (var j = 0u; j < PTWF_WG; j++) {
            let k = ptwfKeys[j];
            pos += select(0u, 1u, k < key || (k == key && j < li));
        }
        ptwfQueue[q * n + first + pos] = path;
    }
}
`;
}

/**
 * Primary kernel of 'ordered' refill (P3.10): appends the slots the shade kernel restarted this iteration (state word 0.w == 0xffffffff, a NaN pattern a real pdf never
 * has) to the next queue IN SLOT ORDER: one workgroup per 64 slots, a serial exclusive scan of the 64 flags in shared memory, one `atomicAdd` per workgroup for the range,
 * so a block of restarted primary rays stays contiguous and in pixel order (inline refill's per-thread `atomicAdd` scatters them: V87). Direct dispatch over all slots.
 */
export function ptwfPrimaryWgsl(wf: PtwfConfig = PTWF_PLAIN): string {
  return /* wgsl */ `
${PT_PARAMS_WGSL}
${bindings(['params', 'state', 'count', 'queue'])}
${commonWgsl(wf)}
var<workgroup> ptwfFlag: array<u32, ${PTWF_WORKGROUP}>;
var<workgroup> ptwfRank: array<u32, ${PTWF_WORKGROUP}>;
var<workgroup> ptwfBase: u32;

@compute @workgroup_size(${PTWF_WORKGROUP})
fn ptwfPrimary(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32) {
    let n = ptwfSlots(ptParams.dims);
    let slot = ptwfSlot(wid, li);
    var flag = 0u;
    if (slot < n && ptwfState[slot].w == 0xffffffffu) {
        flag = 1u;
        ptwfState[slot].w = 0u; // prevPdf of a primary ray
    }
    ptwfFlag[li] = flag;
    workgroupBarrier();
    if (li == 0u) {
        var total = 0u;
        for (var i = 0u; i < PTWF_WG; i++) {
            ptwfRank[i] = total;
            total += ptwfFlag[i];
        }
        ptwfBase = 0u;
        if (total > 0u) {
            ptwfBase = atomicAdd(&ptwfCount[(ptParams.limits.w + 1u) & 1u], total);
        }
    }
    workgroupBarrier();
    if (flag == 1u) {
        let nq = (ptParams.limits.w + 1u) & 1u;
        ptwfQueue[nq * n + ptwfBase + ptwfRank[li]] = slot;
    }
}
`;
}
