/**
 * glTF 2.0 JSON shapes and accessor reading shared by the triangle loader (`gltf-loader.ts`, Sponza-style flat geometry) and the scene
 * loader (`gltf-scene.ts`, materials/uvs/tangents/colours/cameras/lights). Pure: no I/O.
 *
 * Accessors: byte strides, every component type, `normalized` (integers -> floats per the spec), sparse substitution, and the
 * no-bufferView "all zeros" accessor. Matrix accessor types are read column by column without the 4-byte column padding rule
 * (no Tier 1 model has one; skins are not supported anyway).
 */
import type { Mat4 } from './schema.ts';

export interface GltfAccessor {
  bufferView?: number; byteOffset?: number; componentType: number; count: number; type: string; normalized?: boolean;
  sparse?: { count: number; indices: { bufferView: number; byteOffset?: number; componentType: number }; values: { bufferView: number; byteOffset?: number } };
}
export interface GltfBufferView { buffer: number; byteOffset?: number; byteLength: number; byteStride?: number }
export interface GltfPrimitive { attributes: Record<string, number>; indices?: number; mode?: number; material?: number }
export interface GltfNode {
  mesh?: number; children?: number[]; matrix?: number[]; translation?: number[]; rotation?: number[]; scale?: number[]; name?: string; camera?: number;
  extensions?: { KHR_lights_punctual?: { light: number }; [k: string]: unknown };
}
export interface GltfImage { uri?: string; mimeType?: string; bufferView?: number; name?: string }
export interface GltfSampler { magFilter?: number; minFilter?: number; wrapS?: number; wrapT?: number }
export interface GltfTexture { source?: number; sampler?: number; name?: string; extensions?: Record<string, unknown> }
export interface GltfCameraJson { type: 'perspective' | 'orthographic'; name?: string; perspective?: { yfov: number; aspectRatio?: number; znear: number; zfar?: number }; orthographic?: unknown }
export interface GltfLightJson { type: 'directional' | 'point' | 'spot'; name?: string; color?: number[]; intensity?: number; range?: number; spot?: unknown }
export interface GltfJson {
  accessors?: GltfAccessor[]; bufferViews?: GltfBufferView[]; buffers?: { uri?: string; byteLength: number }[];
  meshes?: { primitives: GltfPrimitive[]; name?: string }[]; nodes?: GltfNode[]; scenes?: { nodes?: number[] }[]; scene?: number;
  materials?: import('./gltf-openpbr.ts').GltfMaterialJson[]; textures?: GltfTexture[]; images?: GltfImage[]; samplers?: GltfSampler[];
  cameras?: GltfCameraJson[]; extensionsUsed?: string[]; extensionsRequired?: string[];
  extensions?: { KHR_lights_punctual?: { lights?: GltfLightJson[] }; [k: string]: unknown };
}

export const COMPONENT: Record<number, { bytes: number; read: (dv: DataView, o: number) => number; /** Divisor of the `normalized` mapping (0: not an integer type). */ norm: number; signed: boolean }> = {
  5120: { bytes: 1, read: (dv, o) => dv.getInt8(o), norm: 127, signed: true },
  5121: { bytes: 1, read: (dv, o) => dv.getUint8(o), norm: 255, signed: false },
  5122: { bytes: 2, read: (dv, o) => dv.getInt16(o, true), norm: 32767, signed: true },
  5123: { bytes: 2, read: (dv, o) => dv.getUint16(o, true), norm: 65535, signed: false },
  5125: { bytes: 4, read: (dv, o) => dv.getUint32(o, true), norm: 0, signed: false },
  5126: { bytes: 4, read: (dv, o) => dv.getFloat32(o, true), norm: 0, signed: false },
};
export const NCOMP: Record<string, number> = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT2: 4, MAT3: 9, MAT4: 16 };

/** Float64 values of accessor `index`, `count * components` of them. `normalized` integer accessors come back as floats in [-1, 1] / [0, 1]. */
export function readAccessor(json: GltfJson, buffers: readonly Uint8Array[], index: number): Float64Array {
  const a = json.accessors?.[index];
  if (!a) throw new Error(`accessor ${index} missing`);
  const comp = COMPONENT[a.componentType];
  if (!comp) throw new Error(`unsupported componentType ${a.componentType}`);
  const n = NCOMP[a.type];
  if (!n) throw new Error(`unsupported accessor type ${a.type}`);
  const out = new Float64Array(a.count * n);
  const views = json.bufferViews ?? [];
  const viewData = (bufferView: number): { dv: DataView; base: number; stride?: number } => {
    const v = views[bufferView];
    if (!v) throw new Error(`bufferView ${bufferView} missing`);
    const buf = buffers[v.buffer];
    if (!buf) throw new Error(`buffer ${v.buffer} missing`);
    return { dv: new DataView(buf.buffer, buf.byteOffset, buf.byteLength), base: v.byteOffset ?? 0, ...(v.byteStride !== undefined ? { stride: v.byteStride } : {}) };
  };
  if (a.bufferView !== undefined) {
    const { dv, base, stride } = viewData(a.bufferView);
    const step = stride ?? n * comp.bytes;
    const start = base + (a.byteOffset ?? 0);
    for (let i = 0; i < a.count; i++) for (let k = 0; k < n; k++) out[i * n + k] = comp.read(dv, start + i * step + k * comp.bytes);
  }
  const sp = a.sparse;
  if (sp) {
    const ic = COMPONENT[sp.indices.componentType];
    if (!ic) throw new Error(`unsupported sparse index componentType ${sp.indices.componentType}`);
    const idx = viewData(sp.indices.bufferView), val = viewData(sp.values.bufferView);
    const iStart = idx.base + (sp.indices.byteOffset ?? 0), vStart = val.base + (sp.values.byteOffset ?? 0);
    for (let s = 0; s < sp.count; s++) {
      const target = ic.read(idx.dv, iStart + s * ic.bytes);
      if (target >= a.count) throw new Error(`sparse index ${target} out of range (count ${a.count})`);
      for (let k = 0; k < n; k++) out[target * n + k] = comp.read(val.dv, vStart + (s * n + k) * comp.bytes);
    }
  }
  if (a.normalized && comp.norm > 0) {
    for (let i = 0; i < out.length; i++) out[i] = comp.signed ? Math.max(out[i]! / comp.norm, -1) : out[i]! / comp.norm;
  }
  return out;
}

/** Local matrix of a node (column-major): `matrix`, or T * R * S from the TRS properties. */
export function localMatrix(n: GltfNode): Mat4 {
  if (n.matrix) return n.matrix;
  const t = n.translation ?? [0, 0, 0], q = n.rotation ?? [0, 0, 0, 1], s = n.scale ?? [1, 1, 1];
  const [x, y, z, w] = q as [number, number, number, number];
  const r = [
    1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w), 0,
    2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w), 0,
    2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y), 0,
    0, 0, 0, 1,
  ];
  const trs = r.slice();
  for (let c = 0; c < 3; c++) for (let k = 0; k < 3; k++) trs[c * 4 + k]! *= s[c]!;
  trs[12] = t[0]!; trs[13] = t[1]!; trs[14] = t[2]!;
  return trs;
}

/** Scene root nodes: the default scene's, else every node nobody lists as a child. */
export function rootNodes(json: GltfJson): number[] {
  const nodes = json.nodes ?? [];
  return json.scenes?.[json.scene ?? 0]?.nodes ?? (nodes.length ? nodes.map((_, i) => i).filter((i) => !nodes.some((n) => n.children?.includes(i))) : []);
}
