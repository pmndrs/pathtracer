import { describe, expect, it } from 'vitest';
import { triangulateGltf } from './gltf-loader.ts';
import { readAccessor } from './gltf-read.ts';
import type { GltfJson } from './gltf-read.ts';
import { applyUvTransform, gltfSceneFromJson } from './gltf-scene.ts';
import { encodePng } from '../exporters/png.ts';

/** Builds a glTF in memory: typed arrays go into one buffer, each in its own bufferView (optionally strided). */
class Builder {
  json: GltfJson & { [k: string]: unknown } = { buffers: [], bufferViews: [], accessors: [], meshes: [], nodes: [], scenes: [{ nodes: [] }], scene: 0 };
  private chunks: Buffer[] = [];
  private size = 0;
  view(data: ArrayBufferView, byteStride?: number): number {
    const b = Buffer.from(data.buffer, data.byteOffset, data.byteLength);
    const pad = (4 - (this.size % 4)) % 4;
    this.chunks.push(Buffer.alloc(pad), b);
    this.size += pad;
    this.json.bufferViews!.push({ buffer: 0, byteOffset: this.size, byteLength: b.length, ...(byteStride ? { byteStride } : {}) });
    this.size += b.length;
    return this.json.bufferViews!.length - 1;
  }
  accessor(a: { bufferView?: number; byteOffset?: number; componentType: number; count: number; type: string; normalized?: boolean; sparse?: unknown }): number {
    this.json.accessors!.push(a as never);
    return this.json.accessors!.length - 1;
  }
  f32(values: number[], type: string, count: number): number {
    return this.accessor({ bufferView: this.view(new Float32Array(values)), componentType: 5126, count, type });
  }
  u16(values: number[]): number {
    return this.accessor({ bufferView: this.view(new Uint16Array(values)), componentType: 5123, count: values.length, type: 'SCALAR' });
  }
  buffers(): Buffer[] {
    const bin = Buffer.concat(this.chunks);
    this.json.buffers = [{ byteLength: bin.length }];
    return [bin];
  }
}

const QUAD_POS = [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0];
const QUAD_IDX = [0, 1, 2, 0, 2, 3];

/** One quad primitive in node 0 (extra node fields in `node`), returning the builder. */
function quad(node: Record<string, unknown> = {}, prim: Record<string, unknown> = {}, extra?: (b: Builder, attrs: Record<string, number>) => void): Builder {
  const b = new Builder();
  const attrs: Record<string, number> = { POSITION: b.f32(QUAD_POS, 'VEC3', 4) };
  extra?.(b, attrs);
  b.json.meshes!.push({ primitives: [{ attributes: attrs, indices: b.u16(QUAD_IDX), ...prim }] });
  b.json.nodes!.push({ mesh: 0, ...node });
  b.json.scenes![0]!.nodes = [0];
  return b;
}
const load = (b: Builder, opts = {}) => { const buffers = b.buffers(); return gltfSceneFromJson(b.json, buffers, '/nowhere/x.gltf', opts); };

describe('accessors', () => {
  it('reads interleaved attributes through byteStride and byteOffset', () => {
    const b = new Builder();
    // two vertices: position (3 floats) + normal (3 floats) + 4 bytes of padding, stride 28
    const raw = new Float32Array(14);
    raw.set([1, 2, 3, 0, 0, 1, 0, 4, 5, 6, 0, 1, 0, 0]);
    const view = b.view(raw, 28);
    const pos = b.accessor({ bufferView: view, componentType: 5126, count: 2, type: 'VEC3' });
    const nor = b.accessor({ bufferView: view, byteOffset: 12, componentType: 5126, count: 2, type: 'VEC3' });
    const bufs = b.buffers();
    expect(Array.from(readAccessor(b.json, bufs, pos))).toEqual([1, 2, 3, 4, 5, 6]);
    expect(Array.from(readAccessor(b.json, bufs, nor))).toEqual([0, 0, 1, 0, 1, 0]);
  });
  it('normalizes integer accessors (unsigned: /max, signed: clamped /max)', () => {
    const b = new Builder();
    const u8 = b.accessor({ bufferView: b.view(new Uint8Array([0, 255, 51, 0])), componentType: 5121, count: 2, type: 'VEC2', normalized: true });
    const i16 = b.accessor({ bufferView: b.view(new Int16Array([32767, -32768, 0, 16384])), componentType: 5122, count: 2, type: 'VEC2', normalized: true });
    const raw = b.accessor({ bufferView: b.view(new Uint8Array([0, 255])), componentType: 5121, count: 2, type: 'SCALAR' });
    const bufs = b.buffers();
    expect(Array.from(readAccessor(b.json, bufs, u8))).toEqual([0, 1, 0.2, 0]);
    const s = readAccessor(b.json, bufs, i16);
    expect(s[0]).toBe(1); expect(s[1]).toBe(-1); expect(s[3]).toBeCloseTo(16384 / 32767, 12);
    expect(Array.from(readAccessor(b.json, bufs, raw))).toEqual([0, 255]);
  });
  it('applies sparse substitution over a base accessor and over the implicit zero accessor', () => {
    const b = new Builder();
    const base = b.view(new Float32Array([1, 1, 1, 2, 2, 2, 3, 3, 3]));
    const sparse = (count: number) => ({ count, indices: { bufferView: b.view(new Uint16Array([0, 2])), componentType: 5123 }, values: { bufferView: b.view(new Float32Array([9, 9, 9, 8, 8, 8])) } });
    const over = b.accessor({ bufferView: base, componentType: 5126, count: 3, type: 'VEC3', sparse: sparse(2) });
    const zero = b.accessor({ componentType: 5126, count: 3, type: 'VEC3', sparse: sparse(2) });
    const bufs = b.buffers();
    expect(Array.from(readAccessor(b.json, bufs, over))).toEqual([9, 9, 9, 2, 2, 2, 8, 8, 8]);
    expect(Array.from(readAccessor(b.json, bufs, zero))).toEqual([9, 9, 9, 0, 0, 0, 8, 8, 8]);
  });
  it('rejects a sparse index past the accessor count', () => {
    const b = new Builder();
    const a = b.accessor({ componentType: 5126, count: 1, type: 'SCALAR', sparse: { count: 1, indices: { bufferView: b.view(new Uint16Array([5])), componentType: 5123 }, values: { bufferView: b.view(new Float32Array([1])) } } });
    expect(() => readAccessor(b.json, b.buffers(), a)).toThrow(/out of range/);
  });
});

describe('loadGltfScene geometry', () => {
  it('without a material: one group (-1), the same triangles as the flat loader, flat normals, no extras', () => {
    const b = quad({ translation: [10, 0, 0], scale: [2, 2, 2], children: [1] });
    b.json.nodes!.push({ mesh: 0, translation: [0, 0, 5] });
    const scene = load(b);
    const flat = triangulateGltf(b.json, b.buffers());
    expect(scene.groups).toHaveLength(1);
    const g = scene.groups[0]!;
    expect(g.materialIndex).toBe(-1);
    expect(scene.defaultMaterial?.converted.material.type).toBe('openpbr');
    expect(Array.from(g.mesh.positions)).toEqual(Array.from(flat.mesh.positions));
    expect(Array.from(g.mesh.normals)).toEqual(Array.from(flat.mesh.normals));
    expect(Array.from(g.mesh.indices)).toEqual(Array.from(flat.mesh.indices));
    expect(g.mesh.uvs).toBeUndefined(); expect(g.mesh.tangents).toBeUndefined(); expect(g.mesh.colors).toBeUndefined();
    expect(scene.stats).toMatchObject({ fileTriangles: 4, triangles: 4, degenerateDropped: 0, primitives: 2 });
  });
  it('TRS composes as T * R * S and a node matrix is used as is; child transforms stack on the parent', () => {
    const half = Math.SQRT1_2;
    const b = quad({ translation: [1, 2, 3], rotation: [0, 0, half, half], scale: [2, 1, 1] }); // S then 90 deg about Z then T
    expect(Array.from(load(b).groups[0]!.mesh.positions.slice(3, 6), (v) => Number(v.toFixed(5)))).toEqual([1, 4, 3]); // (1,0,0) -> (2,0,0) -> (0,2,0) -> + T
    const m = quad({ matrix: [0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 5, 6, 7, 1] }); // 90 deg about Z, translation (5,6,7)
    expect(Array.from(load(m).groups[0]!.mesh.positions.slice(3, 6), (v) => Number(v.toFixed(5)))).toEqual([5, 7, 7]);
    const c = quad({ translation: [0, 0, 10], children: [1] });
    c.json.nodes!.push({ mesh: 0, matrix: [2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 1, 0, 0, 1] });
    const p = load(c).groups[0]!.mesh.positions; // second instance: scale 2, translation (1, 0, 10)
    expect(Array.from(p.slice(18 + 3, 18 + 6))).toEqual([3, 0, 10]);
  });
  it('splits triangles by glTF material, in material order, default material last', () => {
    const b = new Builder();
    const pos = b.f32(QUAD_POS, 'VEC3', 4), idx = b.u16(QUAD_IDX);
    b.json.meshes!.push({ primitives: [{ attributes: { POSITION: pos }, indices: idx, material: 1 }, { attributes: { POSITION: pos }, indices: idx }, { attributes: { POSITION: pos }, indices: idx, material: 0 }] });
    b.json.nodes!.push({ mesh: 0, name: 'n' });
    b.json.scenes![0]!.nodes = [0];
    b.json.materials = [{ name: 'a' }, { name: 'b', pbrMetallicRoughness: { metallicFactor: 0 } }];
    const s = load(b);
    expect(s.groups.map((g) => [g.materialIndex, g.name, g.triangles])).toEqual([[0, 'a', 2], [1, 'b', 2], [-1, 'glTF default material', 2]]);
    expect(s.materials.map((m) => m.converted.material.baseMetalness)).toEqual([1, 0]);
    expect(() => load(Object.assign(quad({}, { material: 4 }), {}))).toThrow(/material 4/);
  });
  it('shading normals: inverse transpose under non-uniform scale, kept outward under a mirror', () => {
    const attrs = (b: Builder, a: Record<string, number>) => { a['NORMAL'] = b.f32([0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1].map((v) => v * Math.SQRT1_2), 'VEC3', 4); };
    const sc = load(quad({ scale: [1, 1, 2] }, {}, attrs), { normals: 'shading' }).groups[0]!.mesh;
    const l = Math.hypot(1, 0.5);
    expect(sc.normals[0]).toBeCloseTo(0, 6); expect(sc.normals[1]).toBeCloseTo(1 / l, 6); expect(sc.normals[2]).toBeCloseTo(0.5 / l, 6);
    // default stays flat: the quad is in the XY plane, so +Z
    expect(Array.from(load(quad({ scale: [1, 1, 2] }, {}, attrs)).groups[0]!.mesh.normals.slice(0, 3))).toEqual([0, 0, 1]);
    const mir = load(quad({ scale: [1, -1, 1] }, {}, attrs), { normals: 'shading' }).groups[0]!.mesh;
    expect(mir.normals[1]).toBeLessThan(0); // (0, 1, 1) mirrored in y is (0, -1, 1): inverse transpose of diag(1,-1,1) is itself
    expect(mir.normals[2]).toBeGreaterThan(0);
    expect(Array.from(mir.positions.slice(0, 9)).length).toBe(9);
  });
  it('tangents are transformed and their handedness flips under a mirror; COLOR_0 comes out as rgba', () => {
    const attrs = (b: Builder, a: Record<string, number>) => {
      a['TANGENT'] = b.f32([1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, -1, 1, 0, 0, 1], 'VEC4', 4);
      a['COLOR_0'] = b.accessor({ bufferView: b.view(new Uint8Array([255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255])), componentType: 5121, count: 4, type: 'VEC3', normalized: true });
    };
    const m = load(quad({ scale: [3, 1, 1] }, {}, attrs), { tangents: true, colors: true }).groups[0]!.mesh;
    expect(Array.from(m.tangents!.slice(0, 4))).toEqual([1, 0, 0, 1]);
    expect(Array.from(m.colors!.slice(0, 8))).toEqual([1, 0, 0, 1, 0, 1, 0, 1]);
    const mir = load(quad({ scale: [-1, 1, 1] }, {}, attrs), { tangents: true }).groups[0]!.mesh;
    expect(Array.from(mir.tangents!.slice(0, 4))).toEqual([-1, 0, 0, -1]); // x flipped by the transform, w by the mirror
  });
  it('drops degenerate triangles and rejects non-triangle primitives', () => {
    const b = quad({}, {});
    b.json.accessors![1] = { bufferView: b.view(new Uint16Array([0, 1, 2, 0, 0, 0])), componentType: 5123, count: 6, type: 'SCALAR' };
    expect(load(b).stats).toMatchObject({ fileTriangles: 2, degenerateDropped: 1, triangles: 1 });
    expect(() => load(quad({}, { mode: 5 }))).toThrow(/mode 5/);
  });
});

describe('loadGltfScene materials, textures, uvs', () => {
  const png = (w = 4, h = 4, a = 255) => encodePng(w, h, Uint8Array.from({ length: w * h * 4 }, (_, i) => (i % 4 === 3 ? a : (i * 7) & 255)));
  /** A quad with TEXCOORD_0 (and optionally _1) and the given material JSON; the images are PNGs in the buffer. */
  function textured(material: Record<string, unknown>, opts: { uv1?: boolean; samplers?: unknown[]; textures?: unknown[]; images?: number } = {}): Builder {
    const b = quad({}, { material: 0 }, (bb, a) => {
      a['TEXCOORD_0'] = bb.f32([0, 0, 1, 0, 1, 1, 0, 1], 'VEC2', 4);
      if (opts.uv1) a['TEXCOORD_1'] = bb.f32([0, 0, 0.5, 0, 0.5, 0.5, 0, 0.5], 'VEC2', 4);
    });
    const n = opts.images ?? 1;
    b.json.images = Array.from({ length: n }, (_, i) => ({ bufferView: b.view(png(4 + i, 4)), mimeType: 'image/png' }));
    b.json.samplers = (opts.samplers as never) ?? [{ magFilter: 9728, wrapS: 33071, wrapT: 33071 }];
    b.json.textures = (opts.textures as never) ?? [{ source: 0, sampler: 0 }, { source: n - 1 }];
    b.json.materials = [material as never];
    return b;
  }

  it('maps the four texture slots to texture entries with colour space, wrap, filter and the decoded size', () => {
    const s = load(textured({ pbrMetallicRoughness: { baseColorTexture: { index: 0 }, metallicRoughnessTexture: { index: 1 } }, normalTexture: { index: 1 }, emissiveFactor: [1, 1, 1], emissiveTexture: { index: 0 } }));
    const m = s.materials[0]!;
    expect(m.material).toMatchObject({ baseColorTexture: 't0', metallicRoughnessTexture: 't1_lin', normalTexture: 't1_lin', emissiveTexture: 't0' });
    expect(m.converted.material.baseColorTexture).toBeUndefined(); // the golden-equal table has no texture keys
    expect(s.textures.map((t) => [t.key, t.colorSpace, t.wrap, t.filter, t.width, t.height])).toEqual([['t0', 'srgb', 'clamp', 'nearest', 4, 4], ['t1_lin', 'linear', 'repeat', 'linear', 4, 4]]);
    expect(s.groups[0]!.mesh.uvs).toBeDefined();
    expect(Array.from(s.groups[0]!.mesh.uvs!.slice(0, 6))).toEqual([0, 0, 1, 0, 1, 1]);
  });
  it('one image read as colour and as data makes two entries; unmapped slots are noted, not dropped', () => {
    const s = load(textured({ pbrMetallicRoughness: { baseColorTexture: { index: 0 } }, normalTexture: { index: 0 }, occlusionTexture: { index: 0 }, extensions: { KHR_materials_clearcoat: { clearcoatFactor: 1, clearcoatTexture: { index: 0 } } } }));
    expect(s.textures.map((t) => t.key)).toEqual(['t0', 't0_lin']);
    expect(s.materials[0]!.notes.join(' ')).toMatch(/not mapped.*occlusionTexture.*clearcoatTexture/);
    expect(s.notes.join(' ')).toMatch(/not mapped/);
  });
  it('wrapS/wrapT mismatch is recorded', () => {
    const s = load(textured({ pbrMetallicRoughness: { baseColorTexture: { index: 0 } } }, { samplers: [{ wrapS: 33648, wrapT: 10497 }] }));
    expect(s.textures[0]!.wrap).toBe('mirror');
    expect(s.notes.join(' ')).toMatch(/wraps mirror\/repeat/);
  });
  it('KHR_texture_transform is applied to the uvs exactly: uv\' = T R S uv (offset, rotation, scale)', () => {
    const tt = { offset: [0.25, 0.5], rotation: Math.PI / 2, scale: [2, 1] };
    const s = load(textured({ pbrMetallicRoughness: { baseColorTexture: { index: 0, extensions: { KHR_texture_transform: tt } } } }));
    const uv = s.groups[0]!.mesh.uvs!;
    const want = applyUvTransform({ offset: [0.25, 0.5], rotation: Math.PI / 2, scale: [2, 1] }, 1, 0); // second vertex has uv (1, 0)
    expect(want[0]).toBeCloseTo(0.25, 12); expect(want[1]).toBeCloseTo(-1.5, 12);
    expect(uv[2]).toBeCloseTo(want[0], 6); expect(uv[3]).toBeCloseTo(want[1], 6);
    expect(uv[0]).toBeCloseTo(0.25, 6); expect(uv[1]).toBeCloseTo(0.5, 6); // (0, 0) -> offset
    // identity check against the spec matrices: rotation 0 is a plain scale + offset
    expect(applyUvTransform({ offset: [1, 2], rotation: 0, scale: [3, 4] }, 0.5, 0.25)).toEqual([2.5, 3]);
  });
  it('KHR_texture_transform texCoord override picks TEXCOORD_1; conflicting slots are noted and the first wins', () => {
    const over = load(textured({ pbrMetallicRoughness: { baseColorTexture: { index: 0, extensions: { KHR_texture_transform: { texCoord: 1 } } } } }, { uv1: true }));
    expect(Array.from(over.groups[0]!.mesh.uvs!.slice(2, 4))).toEqual([0.5, 0]);
    const conflict = load(textured({ pbrMetallicRoughness: { baseColorTexture: { index: 0 } }, normalTexture: { index: 1, texCoord: 1 } }, { uv1: true }));
    expect(conflict.materials[0]!.notes.join(' ')).toMatch(/different uv sets/);
    expect(conflict.materials[0]!.texCoord).toBe(0);
  });
  it('uvs: false carries none; a missing TEXCOORD is noted with zero uvs', () => {
    expect(load(textured({ pbrMetallicRoughness: { baseColorTexture: { index: 0 } } }), { uvs: false }).groups[0]!.mesh.uvs).toBeUndefined();
    const b = textured({ pbrMetallicRoughness: { baseColorTexture: { index: 0 } } });
    delete (b.json.meshes![0]!.primitives[0]!.attributes as Record<string, number>)['TEXCOORD_0'];
    const s = load(b);
    expect(s.notes.join(' ')).toMatch(/lacks TEXCOORD_0/);
    expect(Array.from(s.groups[0]!.mesh.uvs!.slice(0, 4))).toEqual([0, 0, 0, 0]);
  });
  it('alpha MASK and BLEND, doubleSided, unlit and emissive strength reach the material', () => {
    const b = textured({});
    b.json.materials = [
      { alphaMode: 'MASK', alphaCutoff: 0.3, pbrMetallicRoughness: { baseColorFactor: [1, 0.5, 0.25, 0.6], baseColorTexture: { index: 0 } }, doubleSided: true },
      { alphaMode: 'BLEND', pbrMetallicRoughness: { baseColorFactor: [1, 1, 1, 0.4] } },
      { extensions: { KHR_materials_unlit: {} }, pbrMetallicRoughness: { baseColorFactor: [0.2, 0.4, 0.8, 1], baseColorTexture: { index: 1 } } },
      { emissiveFactor: [0.1, 0.5, 0.9], extensions: { KHR_materials_emissive_strength: { emissiveStrength: 8 } } },
    ];
    const s = load(b);
    expect(s.materials[0]!.material).toMatchObject({ alphaMode: 'mask', alphaCutoff: 0.3, geometryOpacity: 0.6, doubleSided: true, baseColor: [1, 0.5, 0.25], baseColorTexture: 't0' });
    expect(s.materials[1]!.material).toMatchObject({ alphaMode: 'blend', geometryOpacity: 0.4 });
    // unlit: a non-scattering emitter of the base colour; the base colour texture becomes the emissive texture (srgb)
    expect(s.materials[2]!.material).toMatchObject({ emissionColor: [0.2, 0.4, 0.8], emissionLuminance: 1, baseWeight: 0, specularWeight: 0, emissiveTexture: 't1' });
    expect(s.materials[2]!.material.baseColorTexture).toBeUndefined();
    expect(s.materials[3]!.material).toMatchObject({ emissionColor: [0.1, 0.5, 0.9], emissionLuminance: 8 });
  });
  it('textures without a decodable source are noted', () => {
    const s = load(textured({ pbrMetallicRoughness: { baseColorTexture: { index: 0 } } }, { textures: [{ extensions: { EXT_texture_webp: { source: 0 } } }] }));
    expect(s.textures).toHaveLength(0);
    expect(s.notes.join(' ')).toMatch(/no image source/);
  });
});

describe('loadGltfScene cameras and lights', () => {
  it('a perspective camera node gives position, target one unit ahead, up and yFov in degrees', () => {
    const b = quad();
    b.json.cameras = [{ type: 'perspective', name: 'cam', perspective: { yfov: Math.PI / 3, aspectRatio: 2, znear: 0.1 } }];
    b.json.nodes!.push({ camera: 0, translation: [1, 2, 3], rotation: [0, Math.SQRT1_2, 0, Math.SQRT1_2] }); // 90 deg about Y: looks down -X
    b.json.scenes![0]!.nodes = [0, 1];
    const c = load(b).cameras[0]!;
    expect(c.name).toBe('cam'); expect(c.aspectRatio).toBe(2); expect(c.camera.yFovDeg).toBeCloseTo(60, 9);
    expect(c.camera.position).toEqual([1, 2, 3]);
    expect(c.camera.target.map((v) => Number(v.toFixed(6)))).toEqual([0, 2, 3]);
    expect(c.camera.up.map((v) => Number(v.toFixed(6)))).toEqual([0, 1, 0]);
  });
  it('KHR_lights_punctual: a directional light travels along the node\'s -Z; point and spot lights are listed as unsupported', () => {
    const b = quad();
    b.json.extensions = { KHR_lights_punctual: { lights: [{ type: 'directional', color: [1, 0.5, 0.25], intensity: 4 }, { type: 'point', name: 'bulb' }, { type: 'spot' }] } };
    const q = -Math.SQRT1_2;
    b.json.nodes!.push({ rotation: [q, 0, 0, Math.SQRT1_2], extensions: { KHR_lights_punctual: { light: 0 } } }, { extensions: { KHR_lights_punctual: { light: 1 } } }, { name: 's', extensions: { KHR_lights_punctual: { light: 2 } } });
    b.json.scenes![0]!.nodes = [0, 1, 2, 3];
    const s = load(b);
    expect(s.lights).toHaveLength(1);
    const l = s.lights[0]!;
    expect(l.type).toBe('directional');
    if (l.type === 'directional') {
      expect(l.direction.map((v) => Number(v.toFixed(6)) + 0)).toEqual([0, -1, 0]);
      expect(l.irradiance).toEqual([4, 2, 1]);
    }
    expect(s.unsupportedLights).toEqual(['point bulb', 'spot s']);
  });
  it('extensions the loader does not read are listed', () => {
    const b = quad();
    b.json.extensionsUsed = ['KHR_materials_clearcoat', 'KHR_draco_mesh_compression', 'KHR_texture_transform'];
    expect(load(b).notes.join(' ')).toMatch(/extensions not read: KHR_draco_mesh_compression/);
  });
});
