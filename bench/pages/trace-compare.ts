/**
 * Shared by the trace correctness pages (trace-core, rc-interval-check): the edge classification of V27 and the comparison of a GPU closest hit
 * against bvh's CPU oracle. Moved out of trace-core.ts unchanged (P5.rc1); not a page itself.
 */
import type { CpuHit } from '@pmndrs/bvh';
import { normalMatrixRows, type Hit, type Ray } from '@pmndrs/raytracer';
import type { TraceTestScene } from './trace-scenes.ts';

export const T_RELATIVE = 1e-5;
/** t is compared relative to max(t, T_FLOOR * scene diagonal): f32 error in t scales with the coordinates, not with t. */
export const T_FLOOR = 1e-3;
export const NORMAL_COSINE = 0.9999;

/**
 * Edge classification. bvh's CPU walk runs the f32-stored scene in double arithmetic, the GPU in f32, so a ray within
 * rounding distance of an edge or vertex may get a different answer. What is allowed depends on the geometry there:
 *  - `crossing`: the ray passes THROUGH the surface at a shared edge or vertex (every triangle around it faces the ray
 *    the same way). One of the triangles must be hit: a miss is a watertightness slip and fails. A different triangle
 *    of the same edge at the same t is a tie (both answers are right): counted, not failed.
 *  - `tangent`: the ray only touches the surface there (a convex edge or corner of a solid seen from outside, or a
 *    boundary edge of an open mesh). Hit or miss are both right within rounding: counted as grazing, not failed.
 * Everything is counted and reported; nothing passes silently.
 */
export const EDGE_BARY = 1e-4;

export type EdgeKind = 'interior' | 'crossing' | 'tangent';

export interface EdgeIndex {
  kind(instance: number, triangle: number, u: number, v: number, direction: readonly number[]): EdgeKind;
  /** The two triangles of one instance share a corner (by position bits). */
  adjacent(instance: number, a: number, b: number): boolean;
  /** How two triangles of one instance relate: corners they share (by position bits), and whether they lie in one plane. */
  relation(instance: number, a: number, b: number): { sharedCorners: number; coplanar: boolean };
  /** How many object-space coordinates of the point lie on the mesh's bounding box (2 = on a box edge, 3 = a corner). */
  onBox(instance: number, triangle: number, u: number, v: number): number;
}

export function makeEdgeIndex(scene: TraceTestScene): EdgeIndex {
  const bits = new Uint32Array(1);
  const floats = new Float32Array(bits.buffer);
  const perMesh = scene.meshes.map((mesh) => {
    const key = (vertex: number) => {
      const parts: number[] = [];
      for (let k = 0; k < 3; k++) {
        floats[0] = mesh.positions[vertex * 3 + k] as number;
        parts.push(bits[0] as number);
      }
      return parts.join(',');
    };
    const corner = (t: number, k: number) => key(mesh.indices[t * 3 + k] as number);
    const edgeKey = (x: string, y: string) => (x < y ? `${x}|${y}` : `${y}|${x}`);
    const byEdge = new Map<string, number[]>();
    const byVertex = new Map<string, number[]>();
    const triangles = mesh.indices.length / 3;
    const normals = new Float64Array(triangles * 3);
    for (let t = 0; t < triangles; t++) {
      const p = [0, 1, 2].map((k) => [0, 1, 2].map((c) => mesh.positions[(mesh.indices[t * 3 + k] as number) * 3 + c] as number));
      const e1 = [0, 1, 2].map((c) => (p[1]![c] as number) - (p[0]![c] as number));
      const e2 = [0, 1, 2].map((c) => (p[2]![c] as number) - (p[0]![c] as number));
      normals.set([e1[1]! * e2[2]! - e1[2]! * e2[1]!, e1[2]! * e2[0]! - e1[0]! * e2[2]!, e1[0]! * e2[1]! - e1[1]! * e2[0]!], t * 3);
      for (let k = 0; k < 3; k++) {
        const e = edgeKey(corner(t, k), corner(t, (k + 1) % 3));
        byEdge.set(e, [...(byEdge.get(e) ?? []), t]);
        const v = corner(t, k);
        byVertex.set(v, [...(byVertex.get(v) ?? []), t]);
      }
    }
    const lo = [Infinity, Infinity, Infinity];
    const hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < mesh.positions.length; i++) {
      lo[i % 3] = Math.min(lo[i % 3] as number, mesh.positions[i] as number);
      hi[i % 3] = Math.max(hi[i % 3] as number, mesh.positions[i] as number);
    }
    return { corner, edgeKey, byEdge, byVertex, normals, lo, hi, mesh };
  });
  const recordOf = (instance: number) => scene.instances.find((candidate) => candidate.instanceIndex === instance);
  return {
    kind(instance, triangle, u, v, direction) {
      const record = recordOf(instance);
      const m = record ? perMesh[record.meshIndex] : undefined;
      if (!record || !m) return 'interior';
      const w = [1 - u - v, u, v];
      const small = [0, 1, 2].filter((k) => (w[k] as number) < EDGE_BARY);
      if (small.length === 0) return 'interior';
      // Triangles around the edge (one small weight) or the vertex (two).
      let around: number[];
      if (small.length === 1) {
        const k = small[0] as number;
        around = m.byEdge.get(m.edgeKey(m.corner(triangle, (k + 1) % 3), m.corner(triangle, (k + 2) % 3))) ?? [triangle];
        if (around.length < 2) return 'tangent'; // boundary edge of an open mesh
      } else {
        const vertex = [0, 1, 2].find((k) => !small.includes(k)) ?? 0;
        around = m.byVertex.get(m.corner(triangle, vertex)) ?? [triangle];
      }
      // Ray direction in object space (inverse of the linear part), then which way each triangle faces it.
      const rows = normalMatrixRows(record.objectToWorld);
      const d = [0, 1, 2].map((r) => (rows[r] as number) * (direction[0] as number) + (rows[3 + r] as number) * (direction[1] as number) + (rows[6 + r] as number) * (direction[2] as number));
      const signs = new Set(around.map((t) => Math.sign(m.normals[t * 3]! * d[0]! + m.normals[t * 3 + 1]! * d[1]! + m.normals[t * 3 + 2]! * d[2]!)));
      if (small.length === 2) {
        // A vertex: crossing only when it is interior to a closed fan facing one way.
        const openFan = [...new Set(around.flatMap((t) => [0, 1, 2].map((k) => m.edgeKey(m.corner(t, k), m.corner(t, (k + 1) % 3)))))].some((e) => (m.byEdge.get(e)?.length ?? 0) < 2 && e.includes(m.corner(triangle, [0, 1, 2].find((k) => !small.includes(k)) ?? 0)));
        if (openFan) return 'tangent';
      }
      return signs.size === 1 && !signs.has(0) ? 'crossing' : 'tangent';
    },
    adjacent(instance, a, b) {
      const record = recordOf(instance);
      const m = record ? perMesh[record.meshIndex] : undefined;
      if (!m) return false;
      const corners = new Set([0, 1, 2].map((k) => m.corner(a, k)));
      return [0, 1, 2].some((k) => corners.has(m.corner(b, k)));
    },
    relation(instance, a, b) {
      const record = recordOf(instance);
      const m = record ? perMesh[record.meshIndex] : undefined;
      if (!m) return { sharedCorners: 0, coplanar: false };
      const corners = new Set([0, 1, 2].map((k) => m.corner(a, k)));
      const sharedCorners = [0, 1, 2].filter((k) => corners.has(m.corner(b, k))).length;
      // Coplanar: b's three vertices lie on a's plane, to 1e-6 of the mesh's extent (f32 positions, doubles here).
      const pos = (t: number, k: number) => [0, 1, 2].map((c) => m.mesh.positions[(m.mesh.indices[t * 3 + k] as number) * 3 + c] as number);
      const n = [m.normals[a * 3] as number, m.normals[a * 3 + 1] as number, m.normals[a * 3 + 2] as number];
      const length = Math.hypot(n[0]!, n[1]!, n[2]!);
      const p0 = pos(a, 0);
      const extent = Math.max(...[0, 1, 2].map((c) => (m.hi[c] as number) - (m.lo[c] as number)), 1e-9);
      const coplanar = length > 0 && [0, 1, 2].every((k) => {
        const q = pos(b, k);
        return Math.abs(((q[0]! - p0[0]!) * n[0]! + (q[1]! - p0[1]!) * n[1]! + (q[2]! - p0[2]!) * n[2]!) / length) <= 1e-6 * extent;
      });
      return { sharedCorners, coplanar };
    },
    onBox(instance, triangle, u, v) {
      const record = recordOf(instance);
      const m = record ? perMesh[record.meshIndex] : undefined;
      if (!m) return 0;
      const w = [1 - u - v, u, v];
      let count = 0;
      for (let k = 0; k < 3; k++) {
        let p = 0;
        for (let c = 0; c < 3; c++) p += (w[c] as number) * (m.mesh.positions[(m.mesh.indices[triangle * 3 + c] as number) * 3 + k] as number);
        const extent = Math.max((m.hi[k] as number) - (m.lo[k] as number), 1e-9);
        if (Math.abs(p - (m.lo[k] as number)) < 1e-5 * extent || Math.abs(p - (m.hi[k] as number)) < 1e-5 * extent) count++;
      }
      return count;
    },
  };
}

export type Verdict = { verdict: 'match' | 'tie' | 'primaryTie' | 'grazing' | 'tPosition' | 'mismatch'; why?: string; tRelative?: number; slip?: 'boxEdge' | 'other' };

export function tClose(a: number, b: number, floor: number): number {
  return Math.abs(a - b) / Math.max(Math.abs(b), floor);
}

/** The GPU's closest hit against the oracle's. t is compared relative to max(t, T_FLOOR * scene scale). */
export function compareClosest(gpu: Hit, cpu: CpuHit | null, edges: EdgeIndex, tFloor: number, ray: Ray, allHits: () => ReturnType<AllHits>, strict = false): Verdict {
  const gpuHit = gpu.t >= 0;
  const gpuEdge = gpuHit ? edges.kind(gpu.instance, gpu.triangle, gpu.barycentrics[0], gpu.barycentrics[1], ray.direction) : 'interior';
  const cpuEdge = cpu ? edges.kind(cpu.instanceIndex, cpu.triangleIndex, cpu.u, cpu.v, ray.direction) : 'interior';
  if (!cpu) {
    if (!gpuHit) return { verdict: 'match' };
    return gpuEdge === 'tangent' && !strict ? { verdict: 'grazing' } : { verdict: 'mismatch', why: `gpu hit t=${gpu.t} inst=${gpu.instance} tri=${gpu.triangle} (${gpuEdge}), cpu missed` };
  }
  if (!gpuHit) {
    if (cpuEdge === 'tangent' && !strict) return { verdict: 'grazing' };
    const box = edges.onBox(cpu.instanceIndex, cpu.triangleIndex, cpu.u, cpu.v);
    return { verdict: 'mismatch', slip: cpuEdge === 'crossing' ? (box >= 2 ? 'boxEdge' : 'other') : undefined, why: `gpu missed, cpu hit t=${cpu.t} inst=${cpu.instanceIndex} tri=${cpu.triangleIndex} bary ${cpu.u.toExponential(3)},${cpu.v.toExponential(3)} (${cpuEdge}${cpuEdge === 'crossing' ? `: watertight slip, ${box} coords on the mesh box` : ''})` };
  }
  const tRelative = tClose(gpu.t, cpu.t, tFloor);
  const sameTriangle = gpu.instance === cpu.instanceIndex && gpu.triangle === cpu.triangleIndex;
  // t further than 1e-5 relative is still the same point when the gap, measured along the surface normal, is within
  // 1e-5 of the coordinates' magnitude: f32 rounding of the (object-space) ray scales with the coordinates, and at
  // grazing angles it moves the hit along the ray. Counted as 'tPosition', never as a plain match.
  const cos = Math.abs(ray.direction[0] * cpu.normal[0] + ray.direction[1] * cpu.normal[1] + ray.direction[2] * cpu.normal[2]);
  const point = Math.hypot(ray.origin[0] + ray.direction[0] * cpu.t, ray.origin[1] + ray.direction[1] * cpu.t, ray.origin[2] + ray.direction[2] * cpu.t);
  const positional = tRelative > T_RELATIVE && Math.abs(gpu.t - cpu.t) * cos <= T_RELATIVE * Math.max(cpu.t, point, tFloor);
  if (tRelative > T_RELATIVE && !positional) {
    // Grazing past a tangent edge onto something behind it, on either side.
    const nearer = gpu.t < cpu.t ? gpuEdge : cpuEdge;
    if (!sameTriangle && nearer === 'tangent' && !strict) return { verdict: 'grazing', tRelative };
    return { verdict: 'mismatch', why: `t gpu ${gpu.t} cpu ${cpu.t} (rel ${tRelative.toExponential(2)}, cos ${cos.toFixed(3)}, tri gpu ${gpu.instance}/${gpu.triangle} cpu ${cpu.instanceIndex}/${cpu.triangleIndex})`, tRelative };
  }
  if (!sameTriangle && strict) {
    // Primary rays: the instance and triangle must match (the f64 walk names the GPU's on bvh's own Sponza primary rays).
    // The one allowed difference is a geometric tie: the same t (checked above) on two triangles of one instance that
    // share a corner or lie in one plane at the hit (duplicate / overlapping faces: both answers are right).
    if (gpu.instance === cpu.instanceIndex) {
      const relation = edges.relation(gpu.instance, gpu.triangle, cpu.triangleIndex);
      if (relation.sharedCorners > 0 || relation.coplanar) return { verdict: 'primaryTie', tRelative, why: `shared corners ${relation.sharedCorners}, coplanar ${relation.coplanar}` };
    }
    return { verdict: 'mismatch', why: `primary inst/tri gpu ${gpu.instance}/${gpu.triangle} bary ${gpu.barycentrics.map((x) => x.toExponential(3))} (${gpuEdge}) cpu ${cpu.instanceIndex}/${cpu.triangleIndex} bary ${cpu.u.toExponential(3)},${cpu.v.toExponential(3)} (${cpuEdge}) at t=${cpu.t}`, tRelative };
  }
  if (!sameTriangle) {
    if (cpuEdge !== 'interior' || gpuEdge !== 'interior') return { verdict: 'tie', tRelative };
    // Neighbours at the same t: thin triangles make barycentrics a poor distance to the edge.
    if (gpu.instance === cpu.instanceIndex && edges.adjacent(gpu.instance, gpu.triangle, cpu.triangleIndex)) return { verdict: 'tie', tRelative };
    // Coplanar surfaces (a block standing on the floor): the GPU's triangle really is hit at the same t.
    const twin = allHits().find((h) => h.instance === gpu.instance && h.triangle === gpu.triangle && Math.abs(h.t - gpu.t) * cos <= T_RELATIVE * Math.max(h.t, point, tFloor));
    if (twin) return { verdict: 'tie', tRelative };
    return { verdict: 'mismatch', why: `inst/tri gpu ${gpu.instance}/${gpu.triangle} bary ${gpu.barycentrics.map((x) => x.toExponential(3))} (${gpuEdge}) cpu ${cpu.instanceIndex}/${cpu.triangleIndex} bary ${cpu.u.toExponential(3)},${cpu.v.toExponential(3)} (${cpuEdge}) at t=${cpu.t}`, tRelative };
  }
  if (gpu.frontFace !== cpu.frontFace) return { verdict: 'mismatch', why: `frontFace gpu ${gpu.frontFace} cpu ${cpu.frontFace}`, tRelative };
  if (gpu.material !== cpu.material) return { verdict: 'mismatch', why: `material gpu ${gpu.material} cpu ${cpu.material}`, tRelative };
  const normalCos = gpu.normal[0] * cpu.normal[0] + gpu.normal[1] * cpu.normal[1] + gpu.normal[2] * cpu.normal[2];
  if (!(normalCos >= NORMAL_COSINE)) return { verdict: 'mismatch', why: `normal cos ${normalCos}`, tRelative };
  return { verdict: positional ? 'tPosition' : 'match', tRelative };
}

export type AllHits = (ray: Ray, mask: number) => { instance: number; triangle: number; t: number; u: number; v: number; frontFace: boolean }[];
