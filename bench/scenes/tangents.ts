/**
 * Tangents from the uv frame (P2.26): what pbrt-v4 and Mitsuba 3 derive on their own (the GGX axis of an anisotropic conductor is
 * dp/du of the mesh uvs, Gram-Schmidt'd against the shading normal), as per-vertex `x, y, z, w` tangents in the layout of glTF `TANGENT`.
 * For un-welded meshes (every triangle owns its vertices, as the glTF scene loader emits) the vertex tangent is exactly the triangle's
 * dp/du; vertices shared between triangles get the area-weighted mean of their triangles' directions.
 */
import type { TriangleMesh } from './geometry.ts';

/** `w` follows glTF: the bitangent `cross(n, t) * w` points toward increasing "up" (decreasing v, the bench uvs are glTF's, origin top-left). */
export function uvTangents(mesh: Pick<TriangleMesh, 'positions' | 'indices' | 'uvs'>): Float32Array {
  const { positions: p, indices, uvs } = mesh;
  if (!uvs) throw new Error('uvTangents: the mesh has no uvs');
  const nv = p.length / 3;
  const sum = new Float64Array(nv * 3);
  const sign = new Float64Array(nv);
  for (let t = 0; t + 2 < indices.length; t += 3) {
    const i = [indices[t]!, indices[t + 1]!, indices[t + 2]!];
    const e1 = [0, 1, 2].map((c) => p[i[1]! * 3 + c]! - p[i[0]! * 3 + c]!);
    const e2 = [0, 1, 2].map((c) => p[i[2]! * 3 + c]! - p[i[0]! * 3 + c]!);
    const d1u = uvs[i[1]! * 2]! - uvs[i[0]! * 2]!, d1v = uvs[i[1]! * 2 + 1]! - uvs[i[0]! * 2 + 1]!;
    const d2u = uvs[i[2]! * 2]! - uvs[i[0]! * 2]!, d2v = uvs[i[2]! * 2 + 1]! - uvs[i[0]! * 2 + 1]!;
    const det = d1u * d2v - d1v * d2u;
    if (!(Math.abs(det) > 1e-30)) continue;
    const dpdu = [0, 1, 2].map((c) => (e1[c]! * d2v - e2[c]! * d1v) / det);
    const up = [0, 1, 2].map((c) => (e1[c]! * d2u - e2[c]! * d1u) / det); // -dp/dv
    const n = [e1[1]! * e2[2]! - e1[2]! * e2[1]!, e1[2]! * e2[0]! - e1[0]! * e2[2]!, e1[0]! * e2[1]! - e1[1]! * e2[0]!];
    const len = Math.hypot(dpdu[0]!, dpdu[1]!, dpdu[2]!);
    if (!(len > 0)) continue;
    const area = Math.hypot(n[0]!, n[1]!, n[2]!);
    const nn = n.map((x) => x / (area || 1));
    const b = [nn[1]! * dpdu[2]! - nn[2]! * dpdu[1]!, nn[2]! * dpdu[0]! - nn[0]! * dpdu[2]!, nn[0]! * dpdu[1]! - nn[1]! * dpdu[0]!];
    const w = b[0]! * up[0]! + b[1]! * up[1]! + b[2]! * up[2]! < 0 ? -1 : 1;
    for (const v of i) {
      for (let c = 0; c < 3; c++) sum[v * 3 + c]! += (dpdu[c]! / len) * area;
      sign[v]! += w * area;
    }
  }
  const out = new Float32Array(nv * 4);
  for (let v = 0; v < nv; v++) {
    const l = Math.hypot(sum[v * 3]!, sum[v * 3 + 1]!, sum[v * 3 + 2]!);
    if (l > 0) out.set([sum[v * 3]! / l, sum[v * 3 + 1]! / l, sum[v * 3 + 2]! / l, sign[v]! < 0 ? -1 : 1], v * 4);
    else out.set([1, 0, 0, 1], v * 4);
  }
  return out;
}

/** Angle in degrees between two tangents after both are made orthogonal to `n` (Gram-Schmidt), the way every renderer builds its frame. */
export function tangentAngleDeg(a: ArrayLike<number>, ia: number, b: ArrayLike<number>, ib: number, n: readonly [number, number, number]): number {
  const gs = (t: ArrayLike<number>, i: number): number[] => {
    const x = t[i * 4]!, y = t[i * 4 + 1]!, z = t[i * 4 + 2]!;
    const d = x * n[0] + y * n[1] + z * n[2];
    const v = [x - n[0] * d, y - n[1] * d, z - n[2] * d];
    const l = Math.hypot(v[0]!, v[1]!, v[2]!) || 1;
    return v.map((c) => c / l);
  };
  const u = gs(a, ia), v = gs(b, ib);
  return (Math.acos(Math.max(-1, Math.min(1, u[0]! * v[0]! + u[1]! * v[1]! + u[2]! * v[2]!))) * 180) / Math.PI;
}
