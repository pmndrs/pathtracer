/**
 * Deterministic tessellation of the schema's analytic shapes into triangles, plus column-major 4×4 helpers.
 *
 * Every renderer (ours, pbrt-v4, Mitsuba, glTF consumers) receives exactly the triangles produced here, so shape
 * discretisation can never be a source of disagreement. Output is float32 positions/normals (what every exporter
 * writes to disk) and uint32 indices, counter-clockwise front faces in the right-handed world of
 * packages/raytracer/src/contract/conventions.ts.
 */
import type { GeometryDesc, Mat4, MeshDesc, Vec3 } from './schema.ts';

export interface TriangleMesh {
  /** xyz per vertex, float32. */
  positions: Float32Array;
  /** Unit normal per vertex (shading normals), float32. */
  normals: Float32Array;
  /** 3 indices per triangle, CCW seen from outside. */
  indices: Uint32Array;
}

export interface TessellateOptions {
  /**
   * Sphere only: use flat (geometric) normals by un-welding every triangle. Needed when a smooth-shaded facet
   * boundary would perturb a quantitative test (e.g. dielectric furnace). Default false (smooth analytic normals).
   */
  flat?: boolean;
}

// ---------------------------------------------------------------------------------------------------------------
// Mat4 (column-major: element (row r, column c) lives at m[c * 4 + r])
// ---------------------------------------------------------------------------------------------------------------

export const identity = (): number[] => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

export function translation(x: number, y: number, z: number): number[] {
  return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1];
}

export function scaling(x: number, y: number = x, z: number = x): number[] {
  return [x, 0, 0, 0, 0, y, 0, 0, 0, 0, z, 0, 0, 0, 0, 1];
}

export function rotationX(rad: number): number[] {
  const c = Math.cos(rad), s = Math.sin(rad);
  return [1, 0, 0, 0, 0, c, s, 0, 0, -s, c, 0, 0, 0, 0, 1];
}

export function rotationY(rad: number): number[] {
  const c = Math.cos(rad), s = Math.sin(rad);
  return [c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, 0, 0, 0, 1];
}

export function rotationZ(rad: number): number[] {
  const c = Math.cos(rad), s = Math.sin(rad);
  return [c, s, 0, 0, -s, c, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
}

/** a · b (apply b first, then a). */
export function multiply(a: Mat4, b: Mat4): number[] {
  const out = new Array<number>(16).fill(0);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      let sum = 0;
      for (let k = 0; k < 4; k++) sum += a[k * 4 + r]! * b[c * 4 + k]!;
      out[c * 4 + r] = sum;
    }
  }
  return out;
}

/** Compose left-to-right as written in code order: `compose(T, R, S)` = T·R·S (S applied first). */
export function compose(...ms: Mat4[]): number[] {
  return ms.reduce<number[]>((acc, m) => multiply(acc, m), identity());
}

export function transformPoint(m: Mat4, p: Vec3): [number, number, number] {
  const [x, y, z] = p;
  const w = m[3]! * x + m[7]! * y + m[11]! * z + m[15]!;
  const iw = w !== 0 && w !== 1 ? 1 / w : 1;
  return [
    (m[0]! * x + m[4]! * y + m[8]! * z + m[12]!) * iw,
    (m[1]! * x + m[5]! * y + m[9]! * z + m[13]!) * iw,
    (m[2]! * x + m[6]! * y + m[10]! * z + m[14]!) * iw,
  ];
}

export function transformDirection(m: Mat4, d: Vec3): [number, number, number] {
  const [x, y, z] = d;
  return [
    m[0]! * x + m[4]! * y + m[8]! * z,
    m[1]! * x + m[5]! * y + m[9]! * z,
    m[2]! * x + m[6]! * y + m[10]! * z,
  ];
}

/** Determinant of the upper-left 3×3: negative means the transform mirrors (flips triangle winding). */
export function determinant3(m: Mat4): number {
  return (
    m[0]! * (m[5]! * m[10]! - m[9]! * m[6]!) -
    m[4]! * (m[1]! * m[10]! - m[9]! * m[2]!) +
    m[8]! * (m[1]! * m[6]! - m[5]! * m[2]!)
  );
}

/** Inverse-transpose of the upper-left 3×3, applied to a normal (not renormalised). */
export function transformNormal(m: Mat4, n: Vec3): [number, number, number] {
  // cofactor matrix of the 3x3 == det · inverse-transpose; sign handled by caller through determinant3.
  const a = m[0]!, b = m[4]!, c = m[8]!;
  const d = m[1]!, e = m[5]!, f = m[9]!;
  const g = m[2]!, h = m[6]!, i = m[10]!;
  const c00 = e * i - f * h, c01 = f * g - d * i, c02 = d * h - e * g;
  const c10 = c * h - b * i, c11 = a * i - c * g, c12 = b * g - a * h;
  const c20 = b * f - c * e, c21 = c * d - a * f, c22 = a * e - b * d;
  const det = a * c00 + b * c01 + c * c02;
  const s = det < 0 ? -1 : 1; // keep normals on the "outside" of mirrored geometry after the winding flip
  const [x, y, z] = n;
  return [
    s * (c00 * x + c01 * y + c02 * z),
    s * (c10 * x + c11 * y + c12 * z),
    s * (c20 * x + c21 * y + c22 * z),
  ];
}

/** Right-handed view-style look-at returning the camera-to-world matrix (camera looks down its local −Z). */
export function lookAtCameraToWorld(eye: Vec3, target: Vec3, up: Vec3): number[] {
  const f = normalize3([target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]]);
  const s = normalize3(cross3(f, up));
  const u = cross3(s, f);
  return [s[0], s[1], s[2], 0, u[0], u[1], u[2], 0, -f[0], -f[1], -f[2], 0, eye[0], eye[1], eye[2], 1];
}

export function cross3(a: Vec3, b: Vec3): [number, number, number] {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

export function normalize3(v: Vec3): [number, number, number] {
  const l = Math.hypot(v[0], v[1], v[2]);
  return l > 0 ? [v[0] / l, v[1] / l, v[2] / l] : [0, 0, 0];
}

// ---------------------------------------------------------------------------------------------------------------
// Tessellation
// ---------------------------------------------------------------------------------------------------------------

/** Quad in the local XZ plane, centred on the origin, normal +Y. 4 vertices, 2 triangles. */
export function tessellateQuad(size: readonly [number, number]): TriangleMesh {
  const hx = size[0] / 2, hz = size[1] / 2;
  return {
    positions: new Float32Array([-hx, 0, hz, hx, 0, hz, hx, 0, -hz, -hx, 0, -hz]),
    normals: new Float32Array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]),
    indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
  };
}

/** Axis-aligned box centred on the origin, flat normals (24 vertices, 12 triangles). */
export function tessellateBox(size: Vec3): TriangleMesh {
  const hx = size[0] / 2, hy = size[1] / 2, hz = size[2] / 2;
  const positions: number[] = [];
  const normals: number[] = [];
  const indices: number[] = [];
  // Each face: outward normal n, then 4 corners CCW seen from outside.
  const faces: Array<[Vec3, Vec3[]]> = [
    [[1, 0, 0], [[hx, -hy, hz], [hx, -hy, -hz], [hx, hy, -hz], [hx, hy, hz]]],
    [[-1, 0, 0], [[-hx, -hy, -hz], [-hx, -hy, hz], [-hx, hy, hz], [-hx, hy, -hz]]],
    [[0, 1, 0], [[-hx, hy, hz], [hx, hy, hz], [hx, hy, -hz], [-hx, hy, -hz]]],
    [[0, -1, 0], [[-hx, -hy, -hz], [hx, -hy, -hz], [hx, -hy, hz], [-hx, -hy, hz]]],
    [[0, 0, 1], [[-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz]]],
    [[0, 0, -1], [[hx, -hy, -hz], [-hx, -hy, -hz], [-hx, hy, -hz], [hx, hy, -hz]]],
  ];
  for (const [n, corners] of faces) {
    const base = positions.length / 3;
    for (const c of corners) {
      positions.push(c[0], c[1], c[2]);
      normals.push(n[0], n[1], n[2]);
    }
    indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
  }
  return { positions: new Float32Array(positions), normals: new Float32Array(normals), indices: new Uint32Array(indices) };
}

/**
 * UV sphere, poles on ±Y. Vertex layout: north pole, (rings − 1) latitude rings of `segments` vertices (no duplicated
 * seam), south pole. Pole fans are single triangles (no degenerate slivers). `rings` ≥ 2, `segments` ≥ 3.
 * Triangle count = 2 · segments · (rings − 1).
 */
export function tessellateSphere(radius: number, segments: number, rings: number, opts: TessellateOptions = {}): TriangleMesh {
  if (segments < 3 || rings < 2) throw new Error('sphere needs segments >= 3 and rings >= 2');
  const pos: number[] = [];
  const nor: number[] = [];
  const idx: number[] = [];
  const push = (x: number, y: number, z: number) => {
    pos.push(x * radius, y * radius, z * radius);
    nor.push(x, y, z);
  };
  push(0, 1, 0);
  for (let r = 1; r < rings; r++) {
    const theta = (Math.PI * r) / rings; // 0 at the north pole
    const sy = Math.cos(theta), sr = Math.sin(theta);
    for (let s = 0; s < segments; s++) {
      const phi = (2 * Math.PI * s) / segments;
      push(sr * Math.cos(phi), sy, sr * Math.sin(phi));
    }
  }
  push(0, -1, 0);
  const south = pos.length / 3 - 1;
  const ring = (r: number, s: number) => 1 + (r - 1) * segments + (s % segments);
  // Winding: with phi increasing from +X toward +Z (counter-clockwise seen from +Y... which is clockwise seen from
  // outside at the top), triangles below are ordered so the geometric normal points outward; verified by the tests.
  for (let s = 0; s < segments; s++) idx.push(0, ring(1, s + 1), ring(1, s));
  for (let r = 1; r < rings - 1; r++) {
    for (let s = 0; s < segments; s++) {
      const a = ring(r, s), b = ring(r, s + 1), c = ring(r + 1, s + 1), d = ring(r + 1, s);
      idx.push(a, b, c, a, c, d);
    }
  }
  for (let s = 0; s < segments; s++) idx.push(south, ring(rings - 1, s), ring(rings - 1, s + 1));
  const mesh: TriangleMesh = { positions: new Float32Array(pos), normals: new Float32Array(nor), indices: new Uint32Array(idx) };
  return opts.flat ? unweld(mesh) : mesh;
}

/** Duplicate vertices so every triangle has its own, with the geometric normal as the vertex normal. */
export function unweld(mesh: TriangleMesh): TriangleMesh {
  const n = mesh.indices.length;
  const positions = new Float32Array(n * 3);
  const normals = new Float32Array(n * 3);
  const indices = new Uint32Array(n);
  for (let t = 0; t < n; t += 3) {
    const p: number[][] = [];
    for (let k = 0; k < 3; k++) {
      const i = mesh.indices[t + k]!;
      p.push([mesh.positions[i * 3]!, mesh.positions[i * 3 + 1]!, mesh.positions[i * 3 + 2]!]);
    }
    const gn = normalize3(cross3(sub3(p[1] as unknown as Vec3, p[0] as unknown as Vec3), sub3(p[2] as unknown as Vec3, p[0] as unknown as Vec3)));
    for (let k = 0; k < 3; k++) {
      positions.set(p[k]!, (t + k) * 3);
      normals.set(gn, (t + k) * 3);
      indices[t + k] = t + k;
    }
  }
  return { positions, normals, indices };
}

function sub3(a: Vec3, b: Vec3): [number, number, number] {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

/** Tessellate any schema geometry (except `file`, which exporters/loaders handle) in local space. */
export function tessellate(geometry: GeometryDesc, opts: TessellateOptions = {}): TriangleMesh {
  switch (geometry.type) {
    case 'quad':
      return tessellateQuad(geometry.size);
    case 'box':
      return tessellateBox(geometry.size);
    case 'sphere':
      return tessellateSphere(geometry.radius, geometry.segments, geometry.rings, opts);
    case 'triangles': {
      const positions = Float32Array.from(geometry.positions);
      const indices = Uint32Array.from(geometry.indices);
      const normals = geometry.normals ? Float32Array.from(geometry.normals) : computeVertexNormals(positions, indices);
      return { positions, normals, indices };
    }
    case 'file':
      throw new Error(`tessellate: file geometry '${geometry.path}' must be loaded by the caller`);
  }
}

/** Area-weighted smooth vertex normals (used only when a `triangles` mesh carries none). */
export function computeVertexNormals(positions: Float32Array, indices: Uint32Array): Float32Array {
  const acc = new Float64Array(positions.length);
  for (let t = 0; t < indices.length; t += 3) {
    const i0 = indices[t]!, i1 = indices[t + 1]!, i2 = indices[t + 2]!;
    const p0: Vec3 = [positions[i0 * 3]!, positions[i0 * 3 + 1]!, positions[i0 * 3 + 2]!];
    const p1: Vec3 = [positions[i1 * 3]!, positions[i1 * 3 + 1]!, positions[i1 * 3 + 2]!];
    const p2: Vec3 = [positions[i2 * 3]!, positions[i2 * 3 + 1]!, positions[i2 * 3 + 2]!];
    const n = cross3(sub3(p1, p0), sub3(p2, p0));
    for (const i of [i0, i1, i2]) for (let k = 0; k < 3; k++) acc[i * 3 + k]! += n[k]!;
  }
  const out = new Float32Array(positions.length);
  for (let i = 0; i < out.length; i += 3) {
    const n = normalize3([acc[i]!, acc[i + 1]!, acc[i + 2]!]);
    out[i] = n[0]; out[i + 1] = n[1]; out[i + 2] = n[2];
  }
  return out;
}

/**
 * World-space triangles for a mesh: tessellate, apply the mesh transform, and flip winding when the transform mirrors.
 * Exporters write exactly this (pbrt/Mitsuba/glTF get identity object transforms; no per-renderer transform maths).
 */
export function meshToWorld(mesh: MeshDesc, opts: TessellateOptions = {}): TriangleMesh {
  const local = tessellate(mesh.geometry, opts);
  const m = mesh.transform;
  if (!m) return local;
  const n = local.positions.length / 3;
  const positions = new Float32Array(n * 3);
  const normals = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const p = transformPoint(m, [local.positions[i * 3]!, local.positions[i * 3 + 1]!, local.positions[i * 3 + 2]!]);
    const nn = normalize3(transformNormal(m, [local.normals[i * 3]!, local.normals[i * 3 + 1]!, local.normals[i * 3 + 2]!]));
    positions.set(p, i * 3);
    normals.set(nn, i * 3);
  }
  let indices = local.indices;
  if (determinant3(m) < 0) {
    indices = new Uint32Array(local.indices);
    for (let t = 0; t < indices.length; t += 3) {
      const tmp = indices[t + 1]!;
      indices[t + 1] = indices[t + 2]!;
      indices[t + 2] = tmp;
    }
  }
  return { positions, normals, indices };
}

/** Total surface area of a triangle mesh. */
export function surfaceArea(mesh: TriangleMesh): number {
  let a = 0;
  for (let t = 0; t < mesh.indices.length; t += 3) {
    const p = [0, 1, 2].map((k) => {
      const i = mesh.indices[t + k]!;
      return [mesh.positions[i * 3]!, mesh.positions[i * 3 + 1]!, mesh.positions[i * 3 + 2]!] as const;
    }) as [Vec3, Vec3, Vec3];
    const c = cross3(sub3(p[1], p[0]), sub3(p[2], p[0]));
    a += 0.5 * Math.hypot(c[0], c[1], c[2]);
  }
  return a;
}
