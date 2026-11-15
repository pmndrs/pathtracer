/**
 * Visibility-buffer primaries (P3.6): `integrator.primary: 'vbuffer'`. The integrator does not trace camera rays; it reads one texel per pixel
 * that an engine's rasteriser wrote, resolves the primary hit from it (the same `RtHit` a traced hit gives, so `rtSurface`, the materials and
 * everything after the first hit are the code of the traced path) and continues the path as usual.
 *
 * Format: `rgba32uint`, one texel per pixel, image row 0 = top (texel (x, y) is pixel (x, y); a rasteriser that writes clip space +y up puts row 0 there).
 *
 *     x   instance + 1      bvh instance index + 1 (the index `rtSurface` keys on; a merged static hit's source instance). 0 = nothing was drawn: the pixel is
 *                           a miss and sees the environment / sun. Clear the target to (0, 0, 0, 0).
 *     y   triangle          triangle index within the instance's mesh, in the mesh's ORIGINAL order (what bvh reports, what the scene copy indexes by)
 *     z   material          the hit's material (the triangle's bvh material, else the instance's); 0xffffffff = none. bvh keeps per-triangle materials inside its
 *                           tree, the scene copy does not store them, so the engine (which owns its draw calls) writes it
 *     w   barycentrics      weight of vertex 1 in bits 0-15, of vertex 2 in bits 16-31, unorm16 (vertex 0 = 1 - b1 - b2); `ptVbufferPack` / `ptVbufferUnpack`
 *
 * What the engine must do, and what the integrator does:
 *  - One jitter owner (CLAUDE.md): with a V-buffer the ENGINE owns the sub-pixel jitter. It rasterises frame `s` with an offset `(jx, jy)` (pixels, x right,
 *    y down, [-0.5, 0.5], 0 = the pixel centre) and tells the integrator the same numbers (`setVbufferJitter`; the params record carries them). The kernel
 *    reads no sampler number for the camera ray: `PT_PAIR_JITTER` (pair 0) is not consumed; every later dimension is `ptVertexPair(v, slot)`, a hash of the
 *    pair index, so none of them shifts (dimensions.ts is unchanged). A box filter built from one shared offset per frame is the same estimator as a per-pixel
 *    jitter. Use a low-discrepancy sequence for the frame offsets (the bench uses the Owen-Sobol pair 0 of a fixed seed).
 *  - The jitter is still needed: a miss pixel needs its view direction for the environment, and the camera ray `normalize(forward + right ndc.x + up ndc.y)` of
 *    the jittered pixel is the ray every hit of this pixel is shaded along (`wo`), exactly as for a traced primary. A hit's `t` is the distance from the camera
 *    to the barycentric position (the cone width and the medium attenuation use it).
 *  - Needs a surface layout with `positions` (the hit position comes from the barycentrics) and one V-buffer per sample: `encode(encoder, 1)` per frame
 *    (the wavefront scheduler's `refill` / `layers` start samples of one pixel at different times and are not supported).
 *  - Alpha MASK: the rasteriser need not alpha-test. A V-buffer hit on a cutout texel is passed through by the integrator's own cutout step (it continues
 *    from that point by TRACING, exactly like a traced primary), so the image is exact whatever the engine did; an engine that does alpha-test just saves those
 *    re-traces. Geometry the rasteriser draws but the BVH does not (or the other way round) is the engine's to avoid: the V-buffer is the truth for primaries.
 *  - Silhouette pixels: the rasteriser's coverage rule and a ray's hit decide differently on the exact edge pixel positions; sub-pixel jitter and many
 *    samples average that away. Primary hits equal triangle for triangle everywhere else (DIVERGES V95).
 *
 * The WGSL needs the scene copy's surface table (`rtSTable`, `rtSIndex`, `rtVertex3`), so the kernels that resolve it bind the scene copy.
 */
import { SURFACE_TABLE, RT_NO_MATERIAL, SCENE_HEADER_VEC4S } from '@pmndrs/raytracer';
import type { SceneCopy } from '@pmndrs/raytracer';

/** Group 2 binding of the V-buffer texture (`texture_2d<u32>`), in both schedulers. */
export const PT_VBUFFER_BINDING = 9;
/** `GPUTextureFormat` of the V-buffer. */
export const PT_VBUFFER_FORMAT = 'rgba32uint';
/** Bytes per texel. */
export const PT_VBUFFER_BYTES_PER_TEXEL = 16;
/** `material` word of a hit without a material. */
export const PT_VBUFFER_NO_MATERIAL = RT_NO_MATERIAL;

export interface PtVbufferTexel {
  /** False: nothing was drawn (word x = 0). */
  hit: boolean;
  /** bvh instance index. */
  instance: number;
  /** Triangle index within the instance's mesh. */
  triangle: number;
  /** `PT_VBUFFER_NO_MATERIAL` for none. */
  material: number;
  /** Weights of vertices 1 and 2. */
  bary: [number, number];
}

const UNORM16 = Math.fround(1 / 65535);

/** The 4 words of a texel. A miss is `[0, 0, 0, 0]`. TS twin of what the engine's fragment shader writes. */
export function ptVbufferPack(texel: PtVbufferTexel): [number, number, number, number] {
  if (!texel.hit) return [0, 0, 0, 0];
  const q = (v: number): number => Math.min(65535, Math.max(0, Math.round(v * 65535)));
  return [(texel.instance + 1) >>> 0, texel.triangle >>> 0, texel.material >>> 0, (q(texel.bary[0]) | (q(texel.bary[1]) << 16)) >>> 0];
}

/** Inverse of `ptVbufferPack` (barycentrics decoded as the WGSL does: `f32(q) * fround(1 / 65535)`). */
export function ptVbufferUnpack(words: ArrayLike<number>, offset = 0): PtVbufferTexel {
  const x = words[offset] as number;
  const w = words[offset + 3] as number;
  return {
    hit: x !== 0,
    instance: x === 0 ? 0 : x - 1,
    triangle: words[offset + 1] as number,
    material: words[offset + 2] as number,
    bary: [Math.fround((w & 0xffff) * UNORM16), Math.fround((w >>> 16) * UNORM16)],
  };
}

export interface PtVbufferHit {
  /** World-space position from the barycentrics (the scene copy's vertex positions and the instance transform). */
  position: [number, number, number];
  /** Unit geometric normal of the authored front side (bvh's `hit.normal`); zero for a degenerate triangle. */
  normal: [number, number, number];
}

/**
 * CPU twin of `ptVbufferHit` for the geometry (position, normal): reads the scene copy's tables like the shader. Needs `copy.layout.positions`.
 * Throws for an instance or triangle that does not exist.
 */
export function ptVbufferResolve(copy: Pick<SceneCopy, 'table' | 'data' | 'layout' | 'surface'>, texel: Pick<PtVbufferTexel, 'instance' | 'triangle' | 'bary'>): PtVbufferHit {
  if (!copy.layout.positions) throw new Error('ptVbufferResolve: the surface layout has no positions');
  const { table, data } = copy;
  const f32 = new Float32Array(1);
  const u32 = new Uint32Array(f32.buffer);
  const word = (i: number): number => table[i] as number;
  const float = (u: number): number => {
    u32[0] = u;
    return f32[0] as number;
  };
  const instances = word(0);
  if (texel.instance >= instances) throw new RangeError(`ptVbufferResolve: instance ${texel.instance} of ${instances}`);
  const meshBase = word(1) * 4;
  const stride = copy.surface.vertexStride;
  const posOffset = copy.surface.offsets.position;
  const record = (SCENE_HEADER_VEC4S + texel.instance * SURFACE_TABLE.INSTANCE_VEC4S) * 4;
  const meshIndex = word(record + SURFACE_TABLE.INSTANCE_META * 4);
  const mesh = meshBase + meshIndex * 4;
  const [indexOffset, vertexOffset, , triangles] = [word(mesh), word(mesh + 1), word(mesh + 2), word(mesh + 3)];
  if (texel.triangle >= (triangles as number)) throw new RangeError(`ptVbufferResolve: triangle ${texel.triangle} of ${triangles}`);
  const vbase = word(3);
  const first = (indexOffset as number) + texel.triangle * 3;
  const corner = (k: number): [number, number, number] => {
    const base = vbase + ((data[first + k] as number) + (vertexOffset as number)) * stride + posOffset;
    return [float(data[base] as number), float(data[base + 1] as number), float(data[base + 2] as number)];
  };
  const [p0, p1, p2] = [corner(0), corner(1), corner(2)];
  const w1 = texel.bary[0];
  const w2 = texel.bary[1];
  const w0 = 1 - w1 - w2;
  const object: [number, number, number] = [p0[0] * w0 + p1[0] * w1 + p2[0] * w2, p0[1] * w0 + p1[1] * w1 + p2[1] * w2, p0[2] * w0 + p1[2] * w1 + p2[2] * w2];
  const row = (k: number): number[] => [0, 1, 2, 3].map((c) => float(word(record + k * 4 + c)));
  const o2w = [row(SURFACE_TABLE.INSTANCE_O2W), row(SURFACE_TABLE.INSTANCE_O2W + 1), row(SURFACE_TABLE.INSTANCE_O2W + 2)] as [number[], number[], number[]];
  const position = o2w.map((r) => (r[0] as number) * object[0] + (r[1] as number) * object[1] + (r[2] as number) * object[2] + (r[3] as number)) as [number, number, number];
  const e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]] as const;
  const e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]] as const;
  const cross = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]] as const;
  const nrow = (k: number): number[] => [0, 1, 2].map((c) => float(word(record + (SURFACE_TABLE.INSTANCE_NORMAL + k) * 4 + c)));
  const n = [0, 1, 2].map((k) => {
    const r = nrow(k);
    return (r[0] as number) * cross[0] + (r[1] as number) * cross[1] + (r[2] as number) * cross[2];
  }) as [number, number, number];
  const len = Math.hypot(n[0], n[1], n[2]);
  return { position, normal: len > 0 ? [n[0] / len, n[1] / len, n[2] / len] : [0, 0, 0] };
}

/**
 * WGSL: the V-buffer binding and `ptVbufferHit(px, ray) -> RtHit`: the hit of pixel `px` (a miss when nothing was drawn or the texel names a triangle that is not in
 * the scene copy), with `t` = distance from the ray origin to the barycentric position, `frontFace = dot(normal, ray.direction) < 0`, `normal` the authored front
 * side (bvh's rule: the object-space cross product through the inverse transpose of the instance matrix). Needs `rtModuleWgsl` with the scene copy (the surface
 * table) and `positions` in the layout. Append it before the kernel that calls it.
 */
export function ptVbufferWgsl(): string {
  return /* wgsl */ `
@group(2) @binding(${PT_VBUFFER_BINDING}) var ptVbuffer: texture_2d<u32>;
const PT_VBUFFER_UNORM16: f32 = ${String(UNORM16)};

fn ptVbufferHit(px: vec2u, ray: RtRay) -> RtHit {
    let miss = RtHit(RT_MISS_T, 0u, 0u, false, vec2f(0.0), RT_NO_MATERIAL, vec3f(0.0), 0.0);
    let word = textureLoad(ptVbuffer, vec2i(px), 0);
    if (word.x == 0u) {
        return miss;
    }
    let instance = word.x - 1u;
    let header = rtSTable(0u);
    if (instance >= header.x) {
        return miss;
    }
    let vbase = header.w;
    let record = RT_TABLE_HEADER_VEC4S + instance * RT_INSTANCE_VEC4S;
    let meshIndex = rtSTable(record + ${SURFACE_TABLE.INSTANCE_META}u).x;
    let mesh = rtSTable(header.y + meshIndex);
    if (word.y >= mesh.w) {
        return miss;
    }
    let first = mesh.x + word.y * 3u;
    let v0 = (rtSIndex(first) + mesh.y) * RT_SV_STRIDE + RT_SV_POSITION;
    let v1 = (rtSIndex(first + 1u) + mesh.y) * RT_SV_STRIDE + RT_SV_POSITION;
    let v2 = (rtSIndex(first + 2u) + mesh.y) * RT_SV_STRIDE + RT_SV_POSITION;
    let p0 = rtVertex3(vbase, v0);
    let p1 = rtVertex3(vbase, v1);
    let p2 = rtVertex3(vbase, v2);
    let w1 = f32(word.w & 0xffffu) * PT_VBUFFER_UNORM16;
    let w2 = f32(word.w >> 16u) * PT_VBUFFER_UNORM16;
    let w0 = 1.0 - w1 - w2;
    let po = p0 * w0 + p1 * w1 + p2 * w2;
    let r0 = rtTableRow(record + ${SURFACE_TABLE.INSTANCE_O2W}u);
    let r1 = rtTableRow(record + ${SURFACE_TABLE.INSTANCE_O2W + 1}u);
    let r2 = rtTableRow(record + ${SURFACE_TABLE.INSTANCE_O2W + 2}u);
    let position = vec3f(dot(r0.xyz, po) + r0.w, dot(r1.xyz, po) + r1.w, dot(r2.xyz, po) + r2.w);
    let c = cross(p1 - p0, p2 - p0);
    let m0 = rtTableRow(record + ${SURFACE_TABLE.INSTANCE_NORMAL}u).xyz;
    let m1 = rtTableRow(record + ${SURFACE_TABLE.INSTANCE_NORMAL + 1}u).xyz;
    let m2 = rtTableRow(record + ${SURFACE_TABLE.INSTANCE_NORMAL + 2}u).xyz;
    let nw = vec3f(dot(m0, c), dot(m1, c), dot(m2, c));
    let len = length(nw);
    let normal = select(vec3f(0.0), nw / len, len > 0.0);
    return RtHit(length(position - ray.origin), instance, word.y, dot(normal, ray.direction) < 0.0, vec2f(w1, w2), word.z, normal, 0.0);
}
`;
}
