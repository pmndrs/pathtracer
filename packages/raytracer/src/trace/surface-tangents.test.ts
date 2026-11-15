// P2.26: per-vertex tangents in the surface table (RT_SURF_TANGENT): packing, decode vs the double-precision oracle, and gating (a layout
// without tangents keeps its stride, table and module text).
import { describe, expect, it } from 'vitest';
import { buildCaps } from '../caps/index.ts';
import { SURFACE, SURFACE_TABLE } from './constants.ts';
import { rtModuleWgsl } from './modules.ts';
import { buildSurfaceTable, readSurfaceFromTable, referenceSurface, surfaceVertexOffsets, type SurfaceInstanceInput, type SurfaceMeshInput } from './surface-table.ts';

const caps = buildCaps({ features: new Set(['subgroups', 'shader-f16', 'timestamp-query']), limits: {} }, new Set());

function seeded(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const meshes: SurfaceMeshInput[] = [
  { positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0]), indices: [0, 1, 2, 2, 1, 3], normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]), tangents: new Float32Array([1, 0, 0, 1, 1, 0.1, 0, 1, 0.9, 0.2, 0, -1, 1, 0, 0.1, -1]) },
  // no tangents on this one
  { positions: new Float32Array([0, 0, 0, 0, 2, 0, 0, 0, 2]), normals: new Float32Array([1, 0, 0, 1, 0, 0, 1, 0, 0]) },
];
const c = Math.cos(0.7), s = Math.sin(0.7);
const instances: SurfaceInstanceInput[] = [
  { instanceIndex: 0, meshIndex: 0, objectToWorld: [c * 2, 0, -s * 2, 0, 0, 3, 0, 0, s, 0, c, 0, 1, 2, 3, 1] },
  { instanceIndex: 1, meshIndex: 1, objectToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] },
];

describe('vertex tangents in the surface table', () => {
  it('stride grows by 4 only when the layout stores tangents (default: when any mesh has them)', () => {
    const on = buildSurfaceTable(meshes, instances);
    expect(on.layout.tangents).toBe(true);
    expect(on.vertexStride).toBe(3 + 0 + 4);
    expect(on.offsets.tangent).toBe(3);
    const off = buildSurfaceTable(meshes, instances, { tangents: false });
    expect('tangents' in off.layout).toBe(false);
    expect(off.vertexStride).toBe(3);
    const none = buildSurfaceTable(meshes.map(({ positions, indices, normals }) => ({ positions, indices, normals })), instances);
    expect(none.layout).toEqual({ normals: true, uvs: false, positions: false });
    expect(none.vertexStride).toBe(3);
    expect(surfaceVertexOffsets({ normals: true, uvs: true, positions: true, tangents: true })).toEqual({ stride: 12, normal: 0, uv: 3, position: 5, tangent: 8 });
    expect(surfaceVertexOffsets({ normals: true, uvs: true, positions: true })).toEqual({ stride: 8, normal: 0, uv: 3, position: 5, tangent: 8 });
  });

  it('the packed decode equals the oracle: o2w direction transform (not the normal matrix), interpolated, unit, sign of w', () => {
    const rand = seeded(4);
    const table = buildSurfaceTable(meshes, instances, { positions: true });
    for (let k = 0; k < 300; k++) {
      const instance = instances[k % 2] as SurfaceInstanceInput;
      const mesh = meshes[instance.meshIndex] as SurfaceMeshInput;
      const triangles = (mesh.indices ? mesh.indices.length : mesh.positions.length / 3) / 3;
      const u = rand();
      const v = rand() * (1 - u);
      const hit = { t: 1, instance: instance.instanceIndex, triangle: Math.floor(rand() * triangles), barycentrics: [u, v] as const, frontFace: true, material: 0, normal: [0, 0, 1] as const };
      const ray = { origin: [0, 0, 0] as const, direction: [0, 0, 1] as const };
      const packed = readSurfaceFromTable(table, hit, ray, SURFACE.TANGENT);
      const oracle = referenceSurface(meshes, instances, hit, ray);
      if (instance.meshIndex === 1) {
        expect(packed.tangent).toEqual([0, 0, 0, 1]);
        expect(oracle.tangent).toBeUndefined();
        continue;
      }
      expect(oracle.tangent).toBeDefined();
      for (let i = 0; i < 4; i++) expect(packed.tangent![i]).toBeCloseTo(oracle.tangent![i] as number, 5);
      expect(Math.hypot(packed.tangent![0], packed.tangent![1], packed.tangent![2])).toBeCloseTo(1, 6);
    }
    // not requested: no tangent in the result
    expect(readSurfaceFromTable(table, { t: 1, instance: 0, triangle: 0, barycentrics: [0.2, 0.2], frontFace: true, material: 0, normal: [0, 0, 1] }, { origin: [0, 0, 0], direction: [0, 0, 1] }, SURFACE.ALL).tangent).toBeUndefined();
  });

  it('modules: rtSurfaceTangent and RT_SURF_TANGENT exist only for a layout with tangents; the plain layout text is unchanged', () => {
    const slot = { group: 2, firstBinding: 0 };
    const plain = rtModuleWgsl(caps, { scene: { slot, layout: { normals: true, uvs: false, positions: true } } });
    const same = rtModuleWgsl(caps, { scene: { slot, layout: { normals: true, uvs: false, positions: true, tangents: false } } });
    const tan = rtModuleWgsl(caps, { scene: { slot, layout: { normals: true, uvs: false, positions: true, tangents: true } } });
    expect(same).toBe(plain);
    expect(plain).not.toMatch(/RT_SURF_TANGENT|vertexTangent|rtSurfaceTangent/);
    expect(tan).toContain('fn rtSurfaceTangent(');
    expect(tan).toContain('vertexTangent');
    expect(tan).toContain(`RT_SURF_TANGENT: u32 = ${SURFACE.TANGENT}u`);
    expect(tan).toContain(`RT_SV_TANGENT: u32 = ${surfaceVertexOffsets({ normals: true, uvs: false, positions: true, tangents: true }).tangent}u`);
    expect(tan.length - plain.length).toBeLessThan(2400);
    expect(SURFACE.TANGENT & SURFACE.ALL).toBe(0);
    expect(SURFACE_TABLE.INSTANCE_VEC4S).toBe(8);
  });
});
