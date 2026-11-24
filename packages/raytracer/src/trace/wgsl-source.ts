/**
 * WGSL for the trace core (prefix `rt`). Sources are templates: `#if NAME` blocks are resolved by the caps preprocessor
 * (`buildWgsl`), and numbers that come from the packers are interpolated here, so a shader cannot disagree with
 * `surface-table.ts` or `constants.ts` about an offset.
 *
 * Public WGSL API (stable within slice 1):
 *
 *     struct RtRay  { origin: vec3f, tMin: f32, direction: vec3f, tMax: f32 }      // 32 B, == bvh BvhRay == contract Ray
 *     struct RtCone { width: f32, spread: f32 }                                      // 8 B
 *     struct RtHit  { t, instance, triangle, frontFace, bary, material, normal, coneWidth }
 *     fn rtVisible(ray, mask) -> bool                        // L0: true = nothing blocks [tMin, tMax]
 *     fn rtVisibleSegment(a, b, mask, eps) -> bool           // L0 over a world-space segment
 *     fn rtClosest(ray, mask) -> RtHit                       // L1: t < 0 (RT_MISS_T) is a miss
 *     fn rtClosestCone(ray, mask, cone) -> RtHit             // L1 with a ray cone carried through (coneWidth at the hit)
 *     fn rtSurface(hit, ray, want) -> RtSurface              // lazy decode (needs the surface bindings)
 *
 * Everything else starts with `rt`/`Rt`/`RT_`. The bvh walk it sits on keeps its own `bvh*` names.
 */
import { MESH_FLAG, SURFACE, SURFACE_TABLE, RT_MASK_ALL, RT_NO_MATERIAL, HIT_BYTES, HIT_COMPACT_BYTES } from './constants.ts';

const hex = (value: number): string => `0x${value.toString(16)}u`;

/** pbrt-v4 `gamma(n)` = n eps / (1 - n eps) with eps = 2^-24 (f32 unit roundoff): the relative error bound of n chained float operations. */
export const rtGamma = (n: number): number => (n * 2 ** -24) / (1 - n * 2 ** -24);
const RT_GAMMA3 = Math.fround(rtGamma(3)).toString();
const RT_GAMMA7 = Math.fround(rtGamma(7)).toString();
/** V1.7: scene-wide position error bound = 4 gamma(7) x (bounding sphere radius): a triangle is at most 2 R long, and the next ray's test against it errs by about gamma(7) x its extent. For paths with no per-hit vertex decode. */
export const RT_SCENE_ERR_SCALE = Math.fround(4 * rtGamma(7)).toString();

/** L0 and L1 on top of the bvh walk. No bindings of its own. */
export const RT_CORE_WGSL = /* wgsl */ `
// ---- @pmndrs/raytracer core: L0 visibility, L1 closest hit -------------------------------------------------------

struct RtRay {
    origin: vec3f,
    tMin: f32,
    direction: vec3f,   // normalized
    tMax: f32,
}

// Ray cone (Akenine-Moller et al.): footprint width at the origin and spread angle. Carried through, not yet consumed.
struct RtCone {
    width: f32,
    spread: f32,
}

struct RtHit {
    t: f32,              // RT_MISS_T (negative) on a miss
    instance: u32,       // bvh instance index (a merged static hit reports its source instance)
    triangle: u32,       // original triangle index within the instance's mesh
    frontFace: bool,     // ray hit the authored front side
    bary: vec2f,         // weights of vertices 1 and 2 (vertex 0 = 1 - x - y)
    material: u32,       // index into our material table, RT_NO_MATERIAL for none
    normal: vec3f,       // world-space geometric normal of the authored front side, unit length; 0 on a miss
    coneWidth: f32,      // ray-cone footprint at the hit (cone.width + cone.spread * t); 0 without a cone or on a miss
}

const RT_MISS_T: f32 = -1.0;
const RT_NO_MATERIAL: u32 = ${hex(RT_NO_MATERIAL)};
const RT_MASK_ALL: u32 = ${hex(RT_MASK_ALL)};

fn rtRay(origin: vec3f, direction: vec3f, tMin: f32, tMax: f32) -> RtRay {
    return RtRay(origin, tMin, direction, tMax);
}

// A world-space segment a -> b as a ray: tMin and tMax are pulled in by the given amounts (metres) so neither end point's
// own surface blocks it. Zero-length or fully trimmed segments give tMax < tMin, which never hits.
fn rtSegment(a: vec3f, b: vec3f, trimStart: f32, trimEnd: f32) -> RtRay {
    let d = b - a;
    let len = length(d);
    let inv = select(0.0, 1.0 / len, len > 0.0);
    return RtRay(a, trimStart, d * inv, len - trimEnd);
}

fn rtBvhRay(ray: RtRay) -> BvhRay {
    return BvhRay(ray.origin, ray.tMin, ray.direction, ray.tMax);
}

fn rtHitFrom(hit: BvhHit, coneWidth: f32) -> RtHit {
    return RtHit(hit.t, hit.instance, hit.triangle, hit.frontFace, hit.barycentrics, hit.material, hit.normal, coneWidth);
}

fn rtIsHit(hit: RtHit) -> bool {
    return hit.t >= 0.0;
}

// L0. any-hit over [tMin, tMax] (inclusive), far child first (bvh's default order). True when NOTHING blocks.
// bvh's bvhTraceAny returns the opposite (true = blocked); this is the shadow-ray reading: visible.
fn rtVisible(ray: RtRay, mask: u32) -> bool {
#if RT_RAY_MASKS
    return !bvhTraceAnyMasked(rtBvhRay(ray), mask);
#else
    return !bvhTraceAny(rtBvhRay(ray));
#endif
}

fn rtVisibleSegment(a: vec3f, b: vec3f, mask: u32, eps: f32) -> bool {
    let segment = rtSegment(a, b, eps, eps);
    if (segment.tMax < segment.tMin) {
        return true;
    }
    return rtVisible(segment, mask);
}

// L1. closest hit over [tMin, tMax] (inclusive). A miss has t == RT_MISS_T.
fn rtClosest(ray: RtRay, mask: u32) -> RtHit {
#if RT_RAY_MASKS
    let hit = bvhTraceClosestMasked(rtBvhRay(ray), mask);
#else
    let hit = bvhTraceClosest(rtBvhRay(ray));
#endif
    return rtHitFrom(hit, 0.0);
}

fn rtConeAt(cone: RtCone, t: f32) -> f32 {
    return cone.width + cone.spread * t;
}

// The cone for the continuation ray after a hit at distance t; extraSpread is the surface's contribution (curvature, roughness).
fn rtConePropagate(cone: RtCone, t: f32, extraSpread: f32) -> RtCone {
    return RtCone(rtConeAt(cone, t), cone.spread + extraSpread);
}

fn rtClosestCone(ray: RtRay, mask: u32, cone: RtCone) -> RtHit {
    var hit = rtClosest(ray, mask);
    hit.coneWidth = select(0.0, rtConeAt(cone, hit.t), hit.t >= 0.0);
    return hit;
}
`;

export interface SurfaceWgslOptions {
  vertexStride: number;
  normalOffset: number;
  uvOffset: number;
  positionOffset: number;
  /** Float offset of the vertex tangent (xyzw); only read with `tangents` (P2.26). */
  tangentOffset?: number;
  /** Compile the vertex tangent decode (`RT_SURF_TANGENT`, `RtSurface.vertexTangent`); the layout stores tangents. Default false: the text is unchanged. */
  tangents?: boolean;
  /** The layout stores no vertex positions (`positions: 'ray'`): `rtSurface` then reports the scene-wide `positionError` (V1.7, see `rtSceneErr`). Default false: the text is unchanged. */
  rayPositionError?: boolean;
  /** vec4u units before the first instance record: the scene header (`SCENE_HEADER_VEC4S`). Default `SURFACE_TABLE.HEADER_VEC4S`. */
  headerVec4s?: number;
}

/**
 * Lazy surface decode, laid out by surface-table.ts and read out of the scene copy's `rtSceneTable` / `rtSceneData` (declared by
 * `rtSceneBindingsWgsl`, so no bindings here). `want` is a SurfaceMask
 * (`RT_SURF_*`); anything not requested is neither loaded nor computed. Geometric normal, front face and material come free
 * from the hit.
 */
export function rtSurfaceWgsl(o: SurfaceWgslOptions): string {
  const bindings = /* wgsl */ `
fn rtSTable(i: u32) -> vec4u {
    return rtSceneTable[i];
}
fn rtSIndex(i: u32) -> u32 {
    return rtSceneData[i];
}
fn rtSVertex(base: u32, i: u32) -> f32 {
    return bitcast<f32>(rtSceneData[base + i]);
}
`;
  return /* wgsl */ `
// ---- rtSurface: lazy decode from a hit -------------------------------------------------------------------------

const RT_SURF_POSITION: u32 = ${SURFACE.POSITION}u;
const RT_SURF_GEOMETRIC_NORMAL: u32 = ${SURFACE.GEOMETRIC_NORMAL}u;
const RT_SURF_SHADING_NORMAL: u32 = ${SURFACE.SHADING_NORMAL}u;
const RT_SURF_UV: u32 = ${SURFACE.UV}u;
const RT_SURF_MATERIAL: u32 = ${SURFACE.MATERIAL}u;
const RT_SURF_ALL: u32 = ${SURFACE.ALL}u;
#if RT_SURFACE_TEXFRAME
const RT_SURF_TEXFRAME: u32 = ${SURFACE.TEXTURE_FRAME}u;
#endif
${o.tangents ? `const RT_SURF_TANGENT: u32 = ${SURFACE.TANGENT}u;
const RT_MESH_HAS_TANGENTS: u32 = ${MESH_FLAG.HAS_TANGENTS}u;
const RT_SV_TANGENT: u32 = ${o.tangentOffset ?? 0}u;
` : ''}
const RT_MESH_HAS_NORMALS: u32 = ${MESH_FLAG.HAS_NORMALS}u;
const RT_MESH_HAS_UVS: u32 = ${MESH_FLAG.HAS_UVS}u;
const RT_GAMMA3: f32 = ${RT_GAMMA3};
const RT_GAMMA7: f32 = ${RT_GAMMA7};
${o.rayPositionError ? `const RT_SCENE_ERR_SCALE: f32 = ${RT_SCENE_ERR_SCALE};
` : ''}const RT_SV_STRIDE: u32 = ${o.vertexStride}u;
const RT_SV_NORMAL: u32 = ${o.normalOffset}u;
const RT_SV_UV: u32 = ${o.uvOffset}u;
const RT_SV_POSITION: u32 = ${o.positionOffset}u;
const RT_INSTANCE_VEC4S: u32 = ${SURFACE_TABLE.INSTANCE_VEC4S}u;
const RT_TABLE_HEADER_VEC4S: u32 = ${o.headerVec4s ?? SURFACE_TABLE.HEADER_VEC4S}u;

struct RtSurface {
    position: vec3f,       // world space
    positionError: vec3f,  // conservative absolute error bound of position per axis (pbrt-v4 style); zero unless the position was decoded from vertices (RT_SURFACE_POSITIONS)
    geometricNormal: vec3f,
    frontFace: bool,
    shadingNormal: vec3f,  // interpolated, world space, on the geometric normal's side; the geometric normal without data
    uv: vec2f,             // zero without data
    material: u32,
#if RT_SURFACE_TEXFRAME
    uvDensity: f32,        // sqrt(uv area / world area) of the triangle (uv units per metre), 0 without uvs; RT_SURF_TEXFRAME
    tangent: vec4f,        // world dp/du (unit) and the bitangent handedness; RT_SURF_TEXFRAME
#endif
${o.tangents ? `    vertexTangent: vec4f,  // interpolated vertex tangent, world space, unit (zero without data), w = bitangent sign; RT_SURF_TANGENT
` : ''}}
${bindings}
fn rtTableRow(index: u32) -> vec4f {
    return bitcast<vec4f>(rtSTable(index));
}

fn rtVertex3(vbase: u32, base: u32) -> vec3f {
    return vec3f(rtSVertex(vbase, base), rtSVertex(vbase, base + 1u), rtSVertex(vbase, base + 2u));
}

${o.tangents ? `fn rtSurfaceTangent(hit: RtHit) -> vec4f {
    let header = rtSTable(0u);
    if (hit.instance >= header.x) {
        return vec4f(0.0, 0.0, 0.0, 1.0);
    }
    let vbase = header.w;
    let record = RT_TABLE_HEADER_VEC4S + hit.instance * RT_INSTANCE_VEC4S;
    let meshIndex = rtSTable(record + ${SURFACE_TABLE.INSTANCE_META}u).x;
    let mesh = rtSTable(header.y + meshIndex);
    if ((mesh.z & RT_MESH_HAS_TANGENTS) == 0u) {
        return vec4f(0.0, 0.0, 0.0, 1.0);
    }
    let first = mesh.x + hit.triangle * 3u;
    let v0 = (rtSIndex(first) + mesh.y) * RT_SV_STRIDE + RT_SV_TANGENT;
    let v1 = (rtSIndex(first + 1u) + mesh.y) * RT_SV_STRIDE + RT_SV_TANGENT;
    let v2 = (rtSIndex(first + 2u) + mesh.y) * RT_SV_STRIDE + RT_SV_TANGENT;
    let w1 = hit.bary.x;
    let w2 = hit.bary.y;
    let w0 = 1.0 - w1 - w2;
    let t = vec4f(rtSVertex(vbase, v0), rtSVertex(vbase, v0 + 1u), rtSVertex(vbase, v0 + 2u), rtSVertex(vbase, v0 + 3u)) * w0
          + vec4f(rtSVertex(vbase, v1), rtSVertex(vbase, v1 + 1u), rtSVertex(vbase, v1 + 2u), rtSVertex(vbase, v1 + 3u)) * w1
          + vec4f(rtSVertex(vbase, v2), rtSVertex(vbase, v2 + 1u), rtSVertex(vbase, v2 + 2u), rtSVertex(vbase, v2 + 3u)) * w2;
    let q0 = rtTableRow(record + ${SURFACE_TABLE.INSTANCE_O2W}u).xyz;
    let q1 = rtTableRow(record + ${SURFACE_TABLE.INSTANCE_O2W + 1}u).xyz;
    let q2 = rtTableRow(record + ${SURFACE_TABLE.INSTANCE_O2W + 2}u).xyz;
    let world = vec3f(dot(q0, t.xyz), dot(q1, t.xyz), dot(q2, t.xyz));
    let len2 = dot(world, world);
    if (len2 > 1.0e-20) {
        return vec4f(world * inverseSqrt(len2), select(1.0, -1.0, t.w < 0.0));
    }
    return vec4f(0.0, 0.0, 0.0, 1.0);
}

` : ''}fn rtSurface(hit: RtHit, ray: RtRay, want: u32) -> RtSurface {
    var s: RtSurface;
    s.geometricNormal = hit.normal;
    s.frontFace = hit.frontFace;
    s.material = hit.material;
    s.shadingNormal = hit.normal;
    s.uv = vec2f(0.0);
    s.position = vec3f(0.0);
    s.positionError = vec3f(0.0);
#if RT_SURFACE_TEXFRAME
    s.uvDensity = 0.0;
    s.tangent = vec4f(1.0, 0.0, 0.0, 1.0);
#endif
${o.tangents ? `    s.vertexTangent = vec4f(0.0, 0.0, 0.0, 1.0);
` : ''}    if (hit.t < 0.0) {
        return s;
    }
    if ((want & RT_SURF_POSITION) != 0u) {
        s.position = ray.origin + ray.direction * hit.t;
${o.rayPositionError ? `        s.positionError = vec3f(RT_SCENE_ERR_SCALE * bitcast<f32>(rtSTable(3u).y));
` : ''}    }
${o.tangents ? `    if ((want & RT_SURF_TANGENT) != 0u) {
        s.vertexTangent = rtSurfaceTangent(hit);
    }
` : ''}#if RT_SURFACE_TEXFRAME
    let needVertices = (want & (RT_SURF_SHADING_NORMAL | RT_SURF_UV | RT_SURF_POSITION | RT_SURF_TEXFRAME)) != 0u;
#elif RT_SURFACE_POSITIONS
    let needVertices = (want & (RT_SURF_SHADING_NORMAL | RT_SURF_UV | RT_SURF_POSITION)) != 0u;
#else
    let needVertices = (want & (RT_SURF_SHADING_NORMAL | RT_SURF_UV)) != 0u;
#endif
    if (!needVertices) {
        return s;
    }
    let header = rtSTable(0u);
    if (hit.instance >= header.x) {
        return s;
    }
    let vbase = header.w;
    let record = RT_TABLE_HEADER_VEC4S + hit.instance * RT_INSTANCE_VEC4S;
    let meshIndex = rtSTable(record + ${SURFACE_TABLE.INSTANCE_META}u).x;
    let mesh = rtSTable(header.y + meshIndex);   // (indexOffset, vertexOffset, flags, triangleCount)

    let first = mesh.x + hit.triangle * 3u;
    let v0 = (rtSIndex(first) + mesh.y) * RT_SV_STRIDE;
    let v1 = (rtSIndex(first + 1u) + mesh.y) * RT_SV_STRIDE;
    let v2 = (rtSIndex(first + 2u) + mesh.y) * RT_SV_STRIDE;
    let w1 = hit.bary.x;
    let w2 = hit.bary.y;
    let w0 = 1.0 - w1 - w2;

#if RT_SURFACE_POSITIONS
    if ((want & RT_SURF_POSITION) != 0u) {
        let c0 = rtVertex3(vbase, v0 + RT_SV_POSITION) * w0;
        let c1 = rtVertex3(vbase, v1 + RT_SV_POSITION) * w1;
        let c2 = rtVertex3(vbase, v2 + RT_SV_POSITION) * w2;
        let p = c0 + c1 + c2;
        let r0 = rtTableRow(record + ${SURFACE_TABLE.INSTANCE_O2W}u);
        let r1 = rtTableRow(record + ${SURFACE_TABLE.INSTANCE_O2W + 1}u);
        let r2 = rtTableRow(record + ${SURFACE_TABLE.INSTANCE_O2W + 2}u);
        s.position = vec3f(dot(r0.xyz, p) + r0.w, dot(r1.xyz, p) + r1.w, dot(r2.xyz, p) + r2.w);
        // Error bound (pbrt-v4 Triangle interaction + Transform::operator()(Point3fi)): object space gamma(7) * sum |w_i v_i|, then |M| eo + gamma(3) (|M| |p| + |t|).
        let eo = RT_GAMMA7 * (abs(c0) + abs(c1) + abs(c2));
        let ap = abs(p);
        let a0 = abs(r0.xyz);
        let a1 = abs(r1.xyz);
        let a2 = abs(r2.xyz);
        let pe = vec3f(dot(a0, eo), dot(a1, eo), dot(a2, eo)) + RT_GAMMA3 * vec3f(dot(a0, ap) + abs(r0.w), dot(a1, ap) + abs(r1.w), dot(a2, ap) + abs(r2.w));
        // Isotropic (the largest axis): the NEXT ray's test against this same triangle errs by about its extent along ANY axis, so a flat axis-aligned plane (error 0 along its normal) still needs the offset.
        s.positionError = vec3f(max(pe.x, max(pe.y, pe.z)));
    }
#endif
#if RT_SURFACE_UVS
    if ((want & RT_SURF_UV) != 0u && (mesh.z & RT_MESH_HAS_UVS) != 0u) {
        let a = vec2f(rtSVertex(vbase, v0 + RT_SV_UV), rtSVertex(vbase, v0 + RT_SV_UV + 1u));
        let b = vec2f(rtSVertex(vbase, v1 + RT_SV_UV), rtSVertex(vbase, v1 + RT_SV_UV + 1u));
        let c = vec2f(rtSVertex(vbase, v2 + RT_SV_UV), rtSVertex(vbase, v2 + RT_SV_UV + 1u));
        s.uv = a * w0 + b * w1 + c * w2;
    }
#endif
#if RT_SURFACE_TEXFRAME
    if ((want & RT_SURF_TEXFRAME) != 0u && (mesh.z & RT_MESH_HAS_UVS) != 0u) {
        let p0 = rtVertex3(vbase, v0 + RT_SV_POSITION);
        let e1 = rtVertex3(vbase, v1 + RT_SV_POSITION) - p0;
        let e2 = rtVertex3(vbase, v2 + RT_SV_POSITION) - p0;
        let a = vec2f(rtSVertex(vbase, v0 + RT_SV_UV), rtSVertex(vbase, v0 + RT_SV_UV + 1u));
        let d1 = vec2f(rtSVertex(vbase, v1 + RT_SV_UV), rtSVertex(vbase, v1 + RT_SV_UV + 1u)) - a;
        let d2 = vec2f(rtSVertex(vbase, v2 + RT_SV_UV), rtSVertex(vbase, v2 + RT_SV_UV + 1u)) - a;
        let det = d1.x * d2.y - d1.y * d2.x;
        let q0 = rtTableRow(record + ${SURFACE_TABLE.INSTANCE_O2W}u).xyz;
        let q1 = rtTableRow(record + ${SURFACE_TABLE.INSTANCE_O2W + 1}u).xyz;
        let q2 = rtTableRow(record + ${SURFACE_TABLE.INSTANCE_O2W + 2}u).xyz;
        let f1 = vec3f(dot(q0, e1), dot(q1, e1), dot(q2, e1));
        let f2 = vec3f(dot(q0, e2), dot(q1, e2), dot(q2, e2));
        let area2 = length(cross(f1, f2));
        if (abs(det) > 1.0e-30 && area2 > 0.0) {
            s.uvDensity = sqrt(abs(det) / area2);
            let inv = 1.0 / det;
            let tu = (f1 * d2.y - f2 * d1.y) * inv;
            let bv = (f1 * d2.x - f2 * d1.x) * inv;
            s.tangent = vec4f(normalize(tu), select(1.0, -1.0, dot(cross(hit.normal, tu), bv) < 0.0));
        }
    }
#endif
#if RT_SURFACE_NORMALS
    if ((want & RT_SURF_SHADING_NORMAL) != 0u && (mesh.z & RT_MESH_HAS_NORMALS) != 0u) {
        let n = rtVertex3(vbase, v0 + RT_SV_NORMAL) * w0 + rtVertex3(vbase, v1 + RT_SV_NORMAL) * w1 + rtVertex3(vbase, v2 + RT_SV_NORMAL) * w2;
        let m0 = rtTableRow(record + ${SURFACE_TABLE.INSTANCE_NORMAL}u).xyz;
        let m1 = rtTableRow(record + ${SURFACE_TABLE.INSTANCE_NORMAL + 1}u).xyz;
        let m2 = rtTableRow(record + ${SURFACE_TABLE.INSTANCE_NORMAL + 2}u).xyz;
        let world = vec3f(dot(m0, n), dot(m1, n), dot(m2, n));
        let len2 = dot(world, world);
        if (len2 > 1.0e-20) {
            let unit = world * inverseSqrt(len2);
            s.shadingNormal = select(unit, -unit, dot(unit, hit.normal) < 0.0);
        }
    }
#endif
    return s;
}
`;
}

export interface BatchWgslOptions {
  workgroupSize: number;
  /** Largest dimension of a dispatch (the device limit); used by the indirect-args kernel. */
  maxWorkgroupsPerDimension: number;
}

/**
 * Queue ABI v0 batch kernels. Group 1 (group 0 is the scene walk):
 *   b0 rays (AoS `array<RtRay>`, or SoA `array<vec4f>` origin+tMin), b1 rays B (SoA: direction+tMax),
 *   b2 results, b3 params uniform, b4 counts (indirect), b5 cones in, b6 cone widths out.
 */
export function rtBatchWgsl(o: BatchWgslOptions): string {
  const wg = o.workgroupSize;
  return /* wgsl */ `
// ---- Queue ABI v0 batch kernel ---------------------------------------------------------------------------------

// Uniform per dispatch. Direct: count is the ray count. Indirect: countWord indexes the counts buffer; both clamp to capacity.
struct RtBatchParams {
    count: u32,
    mask: u32,
    capacity: u32,
    countWord: u32,
}

#if RT_RAYS_SOA
@group(1) @binding(0) var<storage, read> rtBatchRayA: array<vec4f>;   // origin.xyz, tMin
@group(1) @binding(1) var<storage, read> rtBatchRayB: array<vec4f>;   // direction.xyz, tMax
#else
@group(1) @binding(0) var<storage, read> rtBatchRays: array<RtRay>;
#endif
@group(1) @binding(3) var<uniform> rtBatchParams: RtBatchParams;
#if RT_INDIRECT
@group(1) @binding(4) var<storage, read> rtBatchCounts: array<u32>;
#endif
#if RT_CONES
@group(1) @binding(5) var<storage, read> rtBatchCones: array<RtCone>;
@group(1) @binding(6) var<storage, read_write> rtBatchConeWidths: array<f32>;
#endif

#if RT_MODE_CLOSEST
#if RT_HIT_FULL
// ${HIT_BYTES} bytes: the contract Hit (bvh layout v2). Three scalars for the normal so the record has no padding.
struct RtHitRecord {
    t: f32,
    instance: u32,
    triangle: u32,
    frontFace: u32,
    bary: vec2f,
    material: u32,
    normalX: f32,
    normalY: f32,
    normalZ: f32,
}
#else
// ${HIT_COMPACT_BYTES} bytes: no material, no normal.
struct RtHitRecord {
    t: f32,
    instance: u32,
    triangle: u32,
    frontFace: u32,
    bary: vec2f,
}
#endif
@group(1) @binding(2) var<storage, read_write> rtBatchResults: array<RtHitRecord>;
#else
// 1 = visible (nothing blocks the segment), 0 = blocked. bvh's any-hit result is the opposite; see DIVERGES.
@group(1) @binding(2) var<storage, read_write> rtBatchResults: array<u32>;
#endif

fn rtBatchCount() -> u32 {
#if RT_INDIRECT
    return min(rtBatchCounts[rtBatchParams.countWord], rtBatchParams.capacity);
#else
    return min(rtBatchParams.count, rtBatchParams.capacity);
#endif
}

@compute @workgroup_size(${wg})
fn rtBatch(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) workgroupCount: vec3u,
    @builtin(local_invocation_index) localIndex: u32,
) {
    // 2D dispatch so a huge batch does not exceed the per-dimension limit.
    let rayIndex = (workgroup.y * workgroupCount.x + workgroup.x) * ${wg}u + localIndex;
    if (rayIndex >= rtBatchCount()) {
        return;
    }
#if RT_RAYS_SOA
    let a = rtBatchRayA[rayIndex];
    let b = rtBatchRayB[rayIndex];
    let ray = RtRay(a.xyz, a.w, b.xyz, b.w);
#else
    let ray = rtBatchRays[rayIndex];
#endif
#if RT_MODE_CLOSEST
#if RT_CONES
    let hit = rtClosestCone(ray, rtBatchParams.mask, rtBatchCones[rayIndex]);
    rtBatchConeWidths[rayIndex] = hit.coneWidth;
#else
    let hit = rtClosest(ray, rtBatchParams.mask);
#endif
#if RT_HIT_FULL
    rtBatchResults[rayIndex] = RtHitRecord(hit.t, hit.instance, hit.triangle, select(0u, 1u, hit.frontFace), hit.bary, hit.material, hit.normal.x, hit.normal.y, hit.normal.z);
#else
    rtBatchResults[rayIndex] = RtHitRecord(hit.t, hit.instance, hit.triangle, select(0u, 1u, hit.frontFace), hit.bary);
#endif
#else
    rtBatchResults[rayIndex] = select(0u, 1u, rtVisible(ray, rtBatchParams.mask));
#endif
}
`;
}

/** The indirect-args kernel: one thread turns a GPU-side ray count into DispatchIndirect args (x, y, 1) for the batch kernel. */
export function rtBatchArgsWgsl(o: BatchWgslOptions): string {
  return /* wgsl */ `
struct RtBatchParams {
    count: u32,
    mask: u32,
    capacity: u32,
    countWord: u32,
}
@group(0) @binding(0) var<storage, read> rtArgsCounts: array<u32>;
@group(0) @binding(1) var<storage, read_write> rtArgsOut: array<u32>;
@group(0) @binding(2) var<uniform> rtArgsParams: RtBatchParams;

@compute @workgroup_size(1)
fn rtBatchArgs() {
    let count = min(rtArgsCounts[rtArgsParams.countWord], rtArgsParams.capacity);
    let groups = (count + ${o.workgroupSize - 1}u) / ${o.workgroupSize}u;
    let x = min(groups, ${o.maxWorkgroupsPerDimension}u);
    // groups == 0: x == 0, so the dispatch is a no-op whatever y is.
    rtArgsOut[0] = x;
    rtArgsOut[1] = select(1u, (groups + x - 1u) / x, groups > 0u);
    rtArgsOut[2] = 1u;
}
`;
}
