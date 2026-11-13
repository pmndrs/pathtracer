import { describe, expect, it } from 'vitest';
import { triangulateGltf, parseGlbContainer } from './gltf-loader.ts';

/** One quad (two triangles, one degenerate extra) under a node with translation + non-uniform scale + a child. */
function fixture(mirror = false) {
  const pos = new Float32Array([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0]);
  const idx = new Uint16Array([0, 1, 2, 0, 2, 3, 0, 0, 0]); // last triangle degenerate
  const bin = Buffer.concat([Buffer.from(pos.buffer), Buffer.from(idx.buffer), Buffer.alloc(2)]);
  const json = {
    buffers: [{ byteLength: bin.length }],
    bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: 48 }, { buffer: 0, byteOffset: 48, byteLength: 18 }],
    accessors: [{ bufferView: 0, componentType: 5126, count: 4, type: 'VEC3' }, { bufferView: 1, componentType: 5123, count: 9, type: 'SCALAR' }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0 }, indices: 1 }] }],
    nodes: [{ translation: [10, 0, 0], scale: [2, mirror ? -2 : 2, 2], children: [1] }, { mesh: 0, translation: [0, 0, 5] }],
    scenes: [{ nodes: [0] }],
  };
  return { json, buffers: [bin] };
}

describe('glTF triangle loader', () => {
  it('bakes the node hierarchy, un-welds, uses flat normals and drops degenerate triangles', () => {
    const { json, buffers } = fixture();
    const r = triangulateGltf(json, buffers);
    expect(r.stats).toMatchObject({ fileTriangles: 3, degenerateDropped: 1, triangles: 2, primitives: 1 });
    expect(r.mesh.positions.length).toBe(18);
    // child translation (0,0,5) is scaled by the parent (x2) then translated: z = 10, x = 10 + 2*vx
    expect(Array.from(r.mesh.positions.slice(0, 6))).toEqual([10, 0, 10, 12, 0, 10]);
    expect(Array.from(r.mesh.normals.slice(0, 3))).toEqual([0, 0, 1]); // CCW in +Z
    expect(r.stats.bounds.max).toEqual([12, 2, 10]);
  });
  it('flips winding for mirrored transforms so normals stay consistent', () => {
    const r = triangulateGltf(fixture(true).json, fixture(true).buffers);
    // y mirrored: the quad now spans y in [-2, 0]; flipped winding keeps the normal pointing +Z
    expect(Array.from(r.mesh.normals.slice(0, 3), (v) => v + 0)).toEqual([0, 0, 1]); // + 0 turns -0 into 0
    expect(r.stats.bounds.min[1]).toBe(-2);
  });
  it('rejects non-triangle primitives and non-GLB containers', () => {
    const { json, buffers } = fixture();
    (json.meshes[0]!.primitives[0] as { mode?: number }).mode = 1;
    expect(() => triangulateGltf(json, buffers)).toThrow(/mode 1/);
    expect(() => parseGlbContainer(Buffer.from('not a glb file at all, really'))).toThrow(/bad magic/);
  });
});
