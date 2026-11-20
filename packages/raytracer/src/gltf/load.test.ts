import { describe, expect, it } from 'vitest';
import { buildSceneCopy } from '../scene/scene-copy.ts';
import { ALPHA_MODE } from '../scene/material.ts';
import { isGlb, parseGlb } from './glb.ts';
import { loadGltf } from './load.ts';
import type { GltfJson } from './read.ts';

/** One triangle (z = 0, facing +z) with uvs, a textured clearcoat material, a camera at z = 2 and a downward sun; buffers as a data: URI. */
function triangleGltf(): { json: GltfJson; bin: Uint8Array } {
  const positions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
  const uvs = new Float32Array([0, 1, 1, 1, 0, 0]);
  const indices = new Uint32Array([0, 1, 2]);
  const image = new Uint8Array([0x89, 0x50, 0x4e, 0x47]); // never decoded: the test passes its own decoder
  const parts = [positions, uvs, indices, image].map((a) => new Uint8Array(a.buffer));
  const bin = new Uint8Array(parts.reduce((n, p) => n + p.byteLength, 0));
  const views: { buffer: number; byteOffset: number; byteLength: number }[] = [];
  let offset = 0;
  for (const p of parts) {
    bin.set(p, offset);
    views.push({ buffer: 0, byteOffset: offset, byteLength: p.byteLength });
    offset += p.byteLength;
  }
  const json: GltfJson = {
    asset: { version: '2.0' },
    buffers: [{ byteLength: bin.byteLength }],
    bufferViews: views,
    accessors: [
      { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' },
      { bufferView: 1, componentType: 5126, count: 3, type: 'VEC2' },
      { bufferView: 2, componentType: 5125, count: 3, type: 'SCALAR' },
    ],
    images: [{ bufferView: 3, mimeType: 'image/png' }],
    textures: [{ source: 0 }],
    materials: [{
      name: 'coated', pbrMetallicRoughness: { baseColorFactor: [0.5, 0.25, 1, 1], metallicFactor: 0, roughnessFactor: 0.4, baseColorTexture: { index: 0 } },
      alphaMode: 'MASK', extensions: { KHR_materials_clearcoat: { clearcoatFactor: 1, clearcoatRoughnessFactor: 0.1 } },
    }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0, TEXCOORD_0: 1 }, indices: 2, material: 0 }] }],
    cameras: [{ type: 'perspective', perspective: { yfov: Math.PI / 4, znear: 0.1 } }],
    extensions: { KHR_lights_punctual: { lights: [{ type: 'directional', intensity: 3 }, { type: 'point', name: 'bulb' }] } },
    nodes: [
      { mesh: 0 },
      { camera: 0, translation: [0, 0, 2] },
      // Rotated -90 degrees about x: the light's -z points down.
      { rotation: [-Math.SQRT1_2, 0, 0, Math.SQRT1_2], extensions: { KHR_lights_punctual: { light: 0 } } },
      { extensions: { KHR_lights_punctual: { light: 1 } } },
    ],
    scenes: [{ nodes: [0, 1, 2, 3] }],
  } as GltfJson;
  return { json, bin };
}

function glb(json: GltfJson, bin: Uint8Array): Uint8Array {
  const text = new TextEncoder().encode(JSON.stringify(json));
  const pad = (n: number) => (n + 3) & ~3;
  const jsonLength = pad(text.byteLength), binLength = pad(bin.byteLength);
  const out = new Uint8Array(12 + 8 + jsonLength + 8 + binLength);
  const view = new DataView(out.buffer);
  view.setUint32(0, 0x46546c67, true);
  view.setUint32(4, 2, true);
  view.setUint32(8, out.byteLength, true);
  view.setUint32(12, jsonLength, true);
  view.setUint32(16, 0x4e4f534a, true);
  out.fill(0x20, 20, 20 + jsonLength);
  out.set(text, 20);
  view.setUint32(20 + jsonLength, binLength, true);
  view.setUint32(24 + jsonLength, 0x004e4942, true);
  out.set(bin, 28 + jsonLength);
  return out;
}

const decodeImage = async () => ({ width: 2, height: 2, data: new Uint8Array(16).fill(200) });

describe('loadGltf', () => {
  it('parses a GLB container', () => {
    const { json, bin } = triangleGltf();
    const bytes = glb(json, bin);
    expect(isGlb(bytes)).toBe(true);
    const parsed = parseGlb(bytes);
    expect(parsed.json.meshes?.length).toBe(1);
    expect(Array.from(parsed.bin!.subarray(0, bin.byteLength))).toEqual(Array.from(bin));
  });

  it('turns a GLB into a scene-copy input with materials, textures, camera and sun', async () => {
    const { json, bin } = triangleGltf();
    const input = await loadGltf(glb(json, bin), { decodeImage });
    expect(input.meshes).toHaveLength(1);
    expect(input.instances).toEqual([{ meshIndex: 0, instanceIndex: 0, objectToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], material: 0, mask: 1 }]);
    expect(input.materialNames).toEqual(['coated']);
    const m = input.materials[0]!;
    expect(m.baseColor).toEqual([0.5, 0.25, 1]);
    expect(m.coatWeight).toBe(1);
    expect(m.coatIor).toBe(1.5);
    expect(m.alphaMode).toBe(ALPHA_MODE.mask);
    expect(m.baseColorTexture).toBe(0);
    expect(input.textures).toHaveLength(1);
    expect(input.textures![0]).toMatchObject({ width: 2, height: 2, colorSpace: 'srgb', wrap: 'repeat' });
    expect(Array.from(input.meshes[0]!.uvs!)).toEqual([0, 1, 1, 1, 0, 0]);
    expect(input.cameras[0]!.camera.position).toEqual([0, 0, 2]);
    expect(input.cameras[0]!.camera.target[2]).toBeCloseTo(1, 12);
    expect(input.cameras[0]!.camera.yFovDeg).toBeCloseTo(45, 12);
    expect(input.sun!.direction[1]).toBeCloseTo(-1, 6);
    expect(input.sun!.irradiance).toEqual([3, 3, 3]);
    expect(input.notes.some((n) => n.includes('point bulb'))).toBe(true);
    expect(input.bounds).toEqual({ min: [0, 0, 0], max: [1, 1, 0] });
    // The result is a valid scene-copy input.
    const copy = buildSceneCopy(input);
    expect(copy.textureLayout).not.toBeNull();
  });

  it('reads a .gltf with an embedded buffer (data: URI) and keeps going without textures', async () => {
    const { json, bin } = triangleGltf();
    const base64 = Buffer.from(bin).toString('base64');
    const gltf = { ...json, buffers: [{ byteLength: bin.byteLength, uri: `data:application/octet-stream;base64,${base64}` }] };
    const input = await loadGltf(new TextEncoder().encode(JSON.stringify(gltf)), { textures: false });
    expect(input.textures).toBeUndefined();
    expect(input.materials[0]!.baseColorTexture).toBeUndefined();
    expect(input.meshes[0]!.uvs).toBeUndefined();
    expect(input.notes.some((n) => n.includes('cannot be read'))).toBe(true);
  });

  it('reports an image the decoder fails on instead of failing the load', async () => {
    const { json, bin } = triangleGltf();
    const input = await loadGltf(glb(json, bin), { decodeImage: async () => { throw new Error('bad png'); } });
    expect(input.textures).toBeUndefined();
    expect(input.notes[0]).toBe('image 0: bad png');
  });

  it('refuses compressed geometry it cannot decode', async () => {
    const { json, bin } = triangleGltf();
    await expect(loadGltf(glb({ ...json, extensionsRequired: ['KHR_draco_mesh_compression'] }, bin), { decodeImage })).rejects.toThrow(/KHR_draco_mesh_compression/);
  });
});
