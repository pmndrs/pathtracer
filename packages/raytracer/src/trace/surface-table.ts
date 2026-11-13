/**
 * The data `rtSurface` needs beyond what bvh's hit carries: per-instance transforms, and per-mesh indices and vertex
 * attributes (shading normals, UVs, optionally positions), packed into three storage buffers.
 *
 * This is a deliberately small stand-in for the material/light scene copy (D4, later): only what lazy surface decode
 * needs, keyed by the instance index bvh reports. Everything here runs in Node, so packing is unit-tested without a GPU.
 *
 * Buffers (bindings in order):
 *  - table `array<vec4u>`: header, instance records (128 B), mesh records (16 B). Floats are stored as bit patterns and
 *    read with `bitcast<f32>`, so no integer passes through a float register and no float is canonicalised.
 *  - indices `array<u32>`: three vertex indices per triangle, in the triangle's ORIGINAL order (the order bvh reports),
 *    mesh-local; unindexed meshes get 0..n-1 synthesised.
 *  - vertices `array<f32>`: `vertexStride` floats per vertex, interleaved `[normal xyz][uv][position xyz]` for the
 *    attributes the table was built with.
 */
import { MESH_FLAG, SURFACE, SURFACE_TABLE } from './constants.ts';

export interface SurfaceMeshInput {
  /** `x, y, z` per vertex (the same array given to bvh's `MeshGeometry`). */
  positions: Float32Array;
  /** Three indices per triangle; omit for an unindexed triangle list (the same indices given to bvh). */
  indices?: ArrayLike<number> | null;
  /** `x, y, z` per vertex, object space. Optional. */
  normals?: ArrayLike<number> | null;
  /** `u, v` per vertex. Optional. */
  uvs?: ArrayLike<number> | null;
}

/** The part of bvh's `TraceSceneInstance` the table needs. `objectToWorld` is column-major 4x4. */
export interface SurfaceInstanceInput {
  instanceIndex: number;
  meshIndex: number;
  objectToWorld: ArrayLike<number>;
}

export interface SurfaceLayout {
  normals: boolean;
  uvs: boolean;
  /** Vertex positions are in the table, so `rtSurface` computes the world position from barycentrics instead of from the ray. */
  positions: boolean;
}

export interface SurfaceTableOptions {
  /** Store shading normals. Default: true when any mesh has them. */
  normals?: boolean;
  /** Store UVs. Default: true when any mesh has them. */
  uvs?: boolean;
  /** Store vertex positions (12 B per vertex more) for barycentric world positions. Default false: position = origin + t * direction. */
  positions?: boolean;
}

export interface SurfaceTable {
  layout: SurfaceLayout;
  /** Floats per vertex in `vertices`. */
  vertexStride: number;
  /** Float offsets inside a vertex. */
  offsets: { normal: number; uv: number; position: number };
  /** `array<vec4u>` words. */
  table: Uint32Array;
  indices: Uint32Array;
  vertices: Float32Array;
  /** Instance slots in the table (largest instance index + 1). */
  instanceSlots: number;
  meshCount: number;
  byteLength: number;
}

export function surfaceVertexOffsets(layout: SurfaceLayout): { stride: number; normal: number; uv: number; position: number } {
  const normal = 0;
  const uv = layout.normals ? 3 : 0;
  const position = uv + (layout.uvs ? 2 : 0);
  return { stride: position + (layout.positions ? 3 : 0), normal, uv, position };
}

const f32Bits = new Float32Array(1);
const u32Bits = new Uint32Array(f32Bits.buffer);
function bits(value: number): number {
  f32Bits[0] = value;
  return u32Bits[0] as number;
}

/**
 * Inverse-transpose of the upper 3x3 of a column-major 4x4, as rows. Normals transform with it. A singular matrix
 * returns the identity (the instance cannot be traced anyway: bvh drops it).
 */
export function normalMatrixRows(m: ArrayLike<number>): number[] {
  // Row-major 3x3 of M: a[r][c] = m[c * 4 + r].
  const a = (r: number, c: number) => m[c * 4 + r] as number;
  const c00 = a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1);
  const c01 = a(1, 2) * a(2, 0) - a(1, 0) * a(2, 2);
  const c02 = a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0);
  const det = a(0, 0) * c00 + a(0, 1) * c01 + a(0, 2) * c02;
  if (!Number.isFinite(det) || Math.abs(det) < 1e-30) return [1, 0, 0, 0, 1, 0, 0, 0, 1];
  const c10 = a(0, 2) * a(2, 1) - a(0, 1) * a(2, 2);
  const c11 = a(0, 0) * a(2, 2) - a(0, 2) * a(2, 0);
  const c12 = a(0, 1) * a(2, 0) - a(0, 0) * a(2, 1);
  const c20 = a(0, 1) * a(1, 2) - a(0, 2) * a(1, 1);
  const c21 = a(0, 2) * a(1, 0) - a(0, 0) * a(1, 2);
  const c22 = a(0, 0) * a(1, 1) - a(0, 1) * a(1, 0);
  // inverse-transpose = cofactor matrix / det.
  return [c00, c01, c02, c10, c11, c12, c20, c21, c22].map((v) => v / det);
}

export function triangleCountOf(mesh: SurfaceMeshInput): number {
  return Math.floor((mesh.indices ? mesh.indices.length : mesh.positions.length / 3) / 3);
}

export function buildSurfaceTable(
  meshes: readonly SurfaceMeshInput[],
  instances: readonly SurfaceInstanceInput[],
  options: SurfaceTableOptions = {},
): SurfaceTable {
  const layout: SurfaceLayout = {
    normals: options.normals ?? meshes.some((mesh) => !!mesh.normals),
    uvs: options.uvs ?? meshes.some((mesh) => !!mesh.uvs),
    positions: options.positions ?? false,
  };
  const offsets = surfaceVertexOffsets(layout);
  const stride = offsets.stride;

  let indexCount = 0;
  let vertexCount = 0;
  for (const mesh of meshes) {
    indexCount += triangleCountOf(mesh) * 3;
    vertexCount += mesh.positions.length / 3;
  }
  const indices = new Uint32Array(Math.max(indexCount, 1));
  const vertices = new Float32Array(Math.max(vertexCount * stride, 1));

  const meshRecords: number[][] = [];
  let indexCursor = 0;
  let vertexCursor = 0;
  for (const mesh of meshes) {
    const triangles = triangleCountOf(mesh);
    const vertexTotal = mesh.positions.length / 3;
    for (let i = 0; i < triangles * 3; i++) indices[indexCursor + i] = mesh.indices ? (mesh.indices[i] as number) : i;
    let flags = 0;
    if (layout.normals && mesh.normals) flags |= MESH_FLAG.HAS_NORMALS;
    if (layout.uvs && mesh.uvs) flags |= MESH_FLAG.HAS_UVS;
    for (let v = 0; v < vertexTotal; v++) {
      const base = (vertexCursor + v) * stride;
      if (layout.normals && mesh.normals) for (let k = 0; k < 3; k++) vertices[base + offsets.normal + k] = mesh.normals[v * 3 + k] as number;
      if (layout.uvs && mesh.uvs) for (let k = 0; k < 2; k++) vertices[base + offsets.uv + k] = mesh.uvs[v * 2 + k] as number;
      if (layout.positions) for (let k = 0; k < 3; k++) vertices[base + offsets.position + k] = mesh.positions[v * 3 + k] as number;
    }
    meshRecords.push([indexCursor, vertexCursor, flags, triangles]);
    indexCursor += triangles * 3;
    vertexCursor += vertexTotal;
  }

  let instanceSlots = 0;
  for (const instance of instances) {
    if (!Number.isInteger(instance.instanceIndex) || instance.instanceIndex < 0) throw new RangeError(`buildSurfaceTable: bad instanceIndex ${instance.instanceIndex}`);
    if (!meshes[instance.meshIndex]) throw new RangeError(`buildSurfaceTable: instance ${instance.instanceIndex} uses mesh ${instance.meshIndex}, which does not exist`);
    instanceSlots = Math.max(instanceSlots, instance.instanceIndex + 1);
  }
  const meshBase = SURFACE_TABLE.HEADER_VEC4S + instanceSlots * SURFACE_TABLE.INSTANCE_VEC4S;
  const totalVec4s = meshBase + Math.max(meshes.length, 1) * SURFACE_TABLE.MESH_VEC4S;
  const table = new Uint32Array(totalVec4s * SURFACE_TABLE.VEC4_WORDS);
  table[0] = instanceSlots;
  table[1] = meshBase;
  table[2] = stride;

  for (const instance of instances) {
    const m = instance.objectToWorld;
    const normal = normalMatrixRows(m);
    const record = (SURFACE_TABLE.HEADER_VEC4S + instance.instanceIndex * SURFACE_TABLE.INSTANCE_VEC4S) * SURFACE_TABLE.VEC4_WORDS;
    for (let r = 0; r < 3; r++) {
      const o2w = record + (SURFACE_TABLE.INSTANCE_O2W + r) * 4;
      table[o2w] = bits(m[r] as number);
      table[o2w + 1] = bits(m[4 + r] as number);
      table[o2w + 2] = bits(m[8 + r] as number);
      table[o2w + 3] = bits(m[12 + r] as number);
      const nrm = record + (SURFACE_TABLE.INSTANCE_NORMAL + r) * 4;
      table[nrm] = bits(normal[r * 3] as number);
      table[nrm + 1] = bits(normal[r * 3 + 1] as number);
      table[nrm + 2] = bits(normal[r * 3 + 2] as number);
    }
    const meta = record + SURFACE_TABLE.INSTANCE_META * 4;
    table[meta] = instance.meshIndex;
    table[meta + 1] = 1; // valid
  }
  meshRecords.forEach((words, i) => table.set(words, (meshBase + i) * SURFACE_TABLE.VEC4_WORDS));

  return {
    layout,
    vertexStride: stride,
    offsets: { normal: offsets.normal, uv: offsets.uv, position: offsets.position },
    table,
    indices,
    vertices,
    instanceSlots,
    meshCount: meshes.length,
    byteLength: table.byteLength + indices.byteLength + vertices.byteLength,
  };
}

// ---------------------------------------------------------------------------------------------------------------
// CPU decode. `readSurfaceFromTable` mirrors the WGSL `rtSurface` word for word (used to test the packing in Node);
// `referenceSurface` works from the raw meshes in doubles and shares nothing with the packer (the oracle).
// ---------------------------------------------------------------------------------------------------------------

export type Vec3 = [number, number, number];

/** What a hit carries into `rtSurface` (`Hit` of the contract, plus the ray for the ray-based position). */
export interface SurfaceHitInput {
  t: number;
  instance: number;
  triangle: number;
  /** Weights of vertices 1 and 2. */
  barycentrics: readonly [number, number];
  frontFace: boolean;
  material: number;
  normal: readonly [number, number, number];
}
export interface SurfaceRayInput {
  origin: readonly [number, number, number];
  direction: readonly [number, number, number];
}

export interface DecodedSurface {
  position: Vec3;
  geometricNormal: Vec3;
  frontFace: boolean;
  shadingNormal: Vec3;
  uv: [number, number];
  material: number;
}

const f32Read = new Float32Array(1);
const u32Read = new Uint32Array(f32Read.buffer);
function floatFrom(word: number): number {
  u32Read[0] = word;
  return f32Read[0] as number;
}

function normalized(v: Vec3): Vec3 | null {
  const l2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
  if (!(l2 > 1e-20)) return null;
  const inv = 1 / Math.sqrt(l2);
  return [v[0] * inv, v[1] * inv, v[2] * inv];
}

export function readSurfaceFromTable(table: SurfaceTable, hit: SurfaceHitInput, ray: SurfaceRayInput, want: number = SURFACE.ALL): DecodedSurface {
  const out: DecodedSurface = {
    position: [0, 0, 0],
    geometricNormal: [hit.normal[0], hit.normal[1], hit.normal[2]],
    frontFace: hit.frontFace,
    shadingNormal: [hit.normal[0], hit.normal[1], hit.normal[2]],
    uv: [0, 0],
    material: hit.material,
  };
  if (hit.t < 0) return out;
  const rayPosition: Vec3 = [ray.origin[0] + ray.direction[0] * hit.t, ray.origin[1] + ray.direction[1] * hit.t, ray.origin[2] + ray.direction[2] * hit.t];
  if (want & SURFACE.POSITION && !table.layout.positions) out.position = rayPosition;
  const needsVertices = (want & (SURFACE.SHADING_NORMAL | SURFACE.UV)) !== 0 || (!!(want & SURFACE.POSITION) && table.layout.positions);
  if (!needsVertices || hit.instance >= table.instanceSlots) {
    if (want & SURFACE.POSITION && table.layout.positions) out.position = rayPosition;
    return out;
  }
  const words = table.table;
  const record = (SURFACE_TABLE.HEADER_VEC4S + hit.instance * SURFACE_TABLE.INSTANCE_VEC4S) * 4;
  const f = (index: number) => floatFrom(words[index] as number);
  const meshIndex = words[record + SURFACE_TABLE.INSTANCE_META * 4] as number;
  const meshRecord = ((words[1] as number) + meshIndex) * 4;
  const indexOffset = words[meshRecord] as number;
  const vertexOffset = words[meshRecord + 1] as number;
  const flags = words[meshRecord + 2] as number;
  const ids = [0, 1, 2].map((k) => (table.indices[indexOffset + hit.triangle * 3 + k] as number) + vertexOffset);
  const [u, v] = hit.barycentrics;
  const w: [number, number, number] = [1 - u - v, u, v];
  const attr = (vertex: number, offset: number, n: number) => Array.from({ length: n }, (_, k) => table.vertices[vertex * table.vertexStride + offset + k] as number);
  const mix = (offset: number, n: number) => {
    const a = ids.map((id) => attr(id, offset, n));
    return Array.from({ length: n }, (_, k) => w[0] * (a[0]![k] as number) + w[1] * (a[1]![k] as number) + w[2] * (a[2]![k] as number));
  };

  if (want & SURFACE.POSITION && table.layout.positions) {
    const p = mix(table.offsets.position, 3);
    const row = (r: number) => record + (SURFACE_TABLE.INSTANCE_O2W + r) * 4;
    out.position = [0, 1, 2].map((r) => f(row(r)) * (p[0] as number) + f(row(r) + 1) * (p[1] as number) + f(row(r) + 2) * (p[2] as number) + f(row(r) + 3)) as Vec3;
  }
  if (want & SURFACE.UV && table.layout.uvs && flags & MESH_FLAG.HAS_UVS) {
    const uv = mix(table.offsets.uv, 2);
    out.uv = [uv[0] as number, uv[1] as number];
  }
  if (want & SURFACE.SHADING_NORMAL && table.layout.normals && flags & MESH_FLAG.HAS_NORMALS) {
    const n = mix(table.offsets.normal, 3);
    const row = (r: number) => record + (SURFACE_TABLE.INSTANCE_NORMAL + r) * 4;
    const world = [0, 1, 2].map((r) => f(row(r)) * (n[0] as number) + f(row(r) + 1) * (n[1] as number) + f(row(r) + 2) * (n[2] as number)) as Vec3;
    const unit = normalized(world);
    if (unit) {
      const side = unit[0] * hit.normal[0] + unit[1] * hit.normal[1] + unit[2] * hit.normal[2] < 0 ? -1 : 1;
      out.shadingNormal = [unit[0] * side, unit[1] * side, unit[2] * side];
    }
  }
  return out;
}

/** Independent oracle: decodes from the raw meshes and instance transforms, in doubles. */
export function referenceSurface(
  meshes: readonly SurfaceMeshInput[],
  instances: readonly SurfaceInstanceInput[],
  hit: SurfaceHitInput,
  ray: SurfaceRayInput,
  options: { positionFromVertices?: boolean } = {},
): DecodedSurface {
  const instance = instances.find((candidate) => candidate.instanceIndex === hit.instance);
  const out: DecodedSurface = {
    position: [ray.origin[0] + ray.direction[0] * hit.t, ray.origin[1] + ray.direction[1] * hit.t, ray.origin[2] + ray.direction[2] * hit.t],
    geometricNormal: [hit.normal[0], hit.normal[1], hit.normal[2]],
    frontFace: hit.frontFace,
    shadingNormal: [hit.normal[0], hit.normal[1], hit.normal[2]],
    uv: [0, 0],
    material: hit.material,
  };
  if (!instance || hit.t < 0) return out;
  const mesh = meshes[instance.meshIndex] as SurfaceMeshInput;
  const m = instance.objectToWorld;
  const id = (k: number) => (mesh.indices ? (mesh.indices[hit.triangle * 3 + k] as number) : hit.triangle * 3 + k);
  const [u, v] = hit.barycentrics;
  const w = [1 - u - v, u, v];
  const mixed = (array: ArrayLike<number>, n: number) => Array.from({ length: n }, (_, c) => w[0]! * (array[id(0) * n + c] as number) + w[1]! * (array[id(1) * n + c] as number) + w[2]! * (array[id(2) * n + c] as number));
  if (options.positionFromVertices) {
    const p = mixed(mesh.positions, 3);
    out.position = [0, 1, 2].map((r) => (m[r] as number) * (p[0] as number) + (m[4 + r] as number) * (p[1] as number) + (m[8 + r] as number) * (p[2] as number) + (m[12 + r] as number)) as Vec3;
  }
  if (mesh.uvs) {
    const uv = mixed(mesh.uvs, 2);
    out.uv = [uv[0] as number, uv[1] as number];
  }
  if (mesh.normals) {
    const n = mixed(mesh.normals, 3);
    const rows = normalMatrixRows(m);
    const world = [0, 1, 2].map((r) => (rows[r * 3] as number) * (n[0] as number) + (rows[r * 3 + 1] as number) * (n[1] as number) + (rows[r * 3 + 2] as number) * (n[2] as number)) as Vec3;
    const unit = normalized(world);
    if (unit) {
      const side = unit[0] * hit.normal[0] + unit[1] * hit.normal[1] + unit[2] * hit.normal[2] < 0 ? -1 : 1;
      out.shadingNormal = [unit[0] * side, unit[1] * side, unit[2] * side];
    }
  }
  return out;
}
