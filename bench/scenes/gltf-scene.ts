/**
 * glTF 2.0 scene loader v2 (Node only; P2.19): geometry with materials, texture coordinates, tangents and colours, plus the textures,
 * cameras and punctual lights the file declares. `gltf-loader.ts` stays the Sponza-style "everything as one flat mesh" reader and is
 * unchanged in output; this module groups the same triangles by glTF material.
 *
 * Geometry policy (the same as `gltf-loader.ts`): world space (node hierarchy baked, TRS and `matrix`), **un-welded** (3 vertices per
 * triangle), zero-area triangles dropped, mirrored transforms flip winding. Vertex normals are the flat **geometric** normal by default;
 * `normals: 'shading'` keeps the file's `NORMAL` (transformed by the inverse transpose, still un-welded; a primitive without `NORMAL`
 * falls back to geometric). Instancing is baked: no Tier 1 model shares a mesh between nodes except two tiny ones, and
 * `MetalRoughSpheresNoTextures` (the one that matters) shares *accessors*, not meshes (102 meshes, 98 materials); `SceneDesc` has no
 * shared-mesh instances either, so a keep-instances path would not reach a renderer (docs/design/tier1.md section 1.4).
 *
 * Materials: `gltfMaterialToOpenPbr` per entry of `json.materials` (the P2.17 golden tables are exactly `materials[i].material`), plus the
 * texture slots the scene copy has (baseColor, metallic-roughness, normal, emissive; `KHR_materials_unlit` base colour textures become the
 * emissive texture). Every other texture slot is listed in `notes`, never silently dropped.
 * `KHR_texture_transform` is applied to the uvs at load time (exact: uv' = T R S uv with the spec's matrices) when the material's texture
 * slots agree on one transform and uv set; when they do not, the first slot (base colour, metallic-roughness, normal, emissive) wins and a
 * note says so.
 */
import { compose, cross3, determinant3, multiply, transformPoint } from './geometry.ts';
import type { TriangleMesh } from './geometry.ts';
import { readGltfFile } from './gltf-loader.ts';
import { gltfImageSizes } from './gltf-images.ts';
import { gltfMaterialToOpenPbr } from './gltf-openpbr.ts';
import type { ConvertedMaterial, GltfMaterialJson, GltfTextureInfo } from './gltf-openpbr.ts';
import { gltfMaterialsToOpenPbr } from './gltf-openpbr.ts';
import { localMatrix, readAccessor, rootNodes } from './gltf-read.ts';
import type { GltfJson } from './gltf-read.ts';
import type { CameraDesc, LightDesc, MaterialDesc, TextureDesc, Vec3 } from './schema.ts';

export type NormalMode = 'geometric' | 'shading';

export interface GltfSceneOptions {
  /** Hex sha256 of the main file, verified. */
  sha256?: string;
  /** Vertex normals: flat geometric (default, as `gltf-loader.ts`) or the file's. */
  normals?: NormalMode;
  /** Carry the material's texture coordinates (default true; only primitives whose material uses a mapped texture get any). */
  uvs?: boolean;
  /** Carry `TANGENT` (xyzw, transformed) when present. Default false. */
  tangents?: boolean;
  /** Carry `COLOR_0` as rgba when present. Default false. */
  colors?: boolean;
}

/** The four texture slots the bench materials have, in the order a material's uv set is chosen. */
export const TEXTURE_SLOTS = ['baseColorTexture', 'metallicRoughnessTexture', 'normalTexture', 'emissiveTexture'] as const;
export type TextureSlot = (typeof TEXTURE_SLOTS)[number];
const SRGB_SLOTS: ReadonlySet<TextureSlot> = new Set(['baseColorTexture', 'emissiveTexture']);

export interface UvTransform { offset: readonly [number, number]; rotation: number; scale: readonly [number, number] }

export interface GltfTextureEntry {
  /** Key in `SceneDesc.textures`. */
  key: string;
  /** glTF texture index and its image index. */
  texture: number;
  image: number;
  colorSpace: 'srgb' | 'linear';
  width: number;
  height: number;
  wrap: 'repeat' | 'clamp' | 'mirror';
  filter: 'linear' | 'nearest';
}

export interface GltfSceneMaterial {
  index: number;
  name?: string;
  /** `gltfMaterialToOpenPbr` output, no texture references: equals the P2.17 golden entry. */
  converted: ConvertedMaterial;
  /** The `openpbr` material with texture keys filled in (what a `SceneDesc` holds). */
  material: Extract<MaterialDesc, { type: 'openpbr' }>;
  /** Slots that carry a texture, with the glTF info they came from. */
  slots: Partial<Record<TextureSlot, { key: string; texCoord: number }>>;
  /** The uv set the material's textures read (undefined: no mapped texture) and the transform applied to it. */
  texCoord?: number;
  uvTransform?: UvTransform;
  notes: string[];
}

export interface GltfMaterialGroup {
  /** glTF material index; -1 = primitives without a material (the glTF default material). */
  materialIndex: number;
  name: string;
  mesh: TriangleMesh;
  triangles: number;
  /** Node names that contributed. */
  nodes: string[];
}

export interface GltfCamera { name: string; camera: CameraDesc; aspectRatio?: number; znear: number }

export interface GltfScene {
  /** One per `json.materials` entry, in order. */
  materials: GltfSceneMaterial[];
  /** The glTF default material (used by primitives with no `material`), when any primitive has none. */
  defaultMaterial?: GltfSceneMaterial;
  /** Triangles by material, sorted by material index (default material last). */
  groups: GltfMaterialGroup[];
  textures: GltfTextureEntry[];
  cameras: GltfCamera[];
  /** `KHR_lights_punctual` directional lights (travel direction, lux taken 1:1 as W/m^2). */
  lights: LightDesc[];
  /** Point and spot lights and anything else not mapped: `"<type> <name>"`. */
  unsupportedLights: string[];
  extensionsUsed: string[];
  /** Whole-file notes: dropped textures per material, uv conflicts, unsupported extensions. */
  notes: string[];
  stats: { fileTriangles: number; degenerateDropped: number; triangles: number; primitives: number; bounds: { min: [number, number, number]; max: [number, number, number] } };
}

// ---------------------------------------------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------------------------------------------

/** Growable Float32 array (3M un-welded vertices would be tens of millions of boxed numbers as `number[]`). */
class FloatBuf {
  data = new Float32Array(1024);
  length = 0;
  push(...v: number[]): void {
    if (this.length + v.length > this.data.length) {
      const next = new Float32Array(Math.max(this.data.length * 2, this.length + v.length));
      next.set(this.data.subarray(0, this.length));
      this.data = next;
    }
    for (let i = 0; i < v.length; i++) this.data[this.length++] = v[i]!;
  }
  done(): Float32Array { return this.data.slice(0, this.length); }
}

/** KHR_texture_transform: uv' = T * R * S * uv. */
export function applyUvTransform(t: UvTransform, u: number, v: number): [number, number] {
  const c = Math.cos(t.rotation), s = Math.sin(t.rotation);
  return [c * t.scale[0] * u + s * t.scale[1] * v + t.offset[0], -s * t.scale[0] * u + c * t.scale[1] * v + t.offset[1]];
}

const IDENTITY_UV: UvTransform = { offset: [0, 0], rotation: 0, scale: [1, 1] };
const sameUv = (a: UvTransform, b: UvTransform): boolean => a.rotation === b.rotation && a.offset[0] === b.offset[0] && a.offset[1] === b.offset[1] && a.scale[0] === b.scale[0] && a.scale[1] === b.scale[1];

function textureUv(info: GltfTextureInfo): { texCoord: number; transform: UvTransform } {
  const t = info.extensions?.['KHR_texture_transform'] as { offset?: number[]; rotation?: number; scale?: number[]; texCoord?: number } | undefined;
  return {
    texCoord: t?.texCoord ?? info.texCoord ?? 0,
    transform: t ? { offset: [t.offset?.[0] ?? 0, t.offset?.[1] ?? 0], rotation: t.rotation ?? 0, scale: [t.scale?.[0] ?? 1, t.scale?.[1] ?? 1] } : IDENTITY_UV,
  };
}

const WRAP: Record<number, 'repeat' | 'clamp' | 'mirror'> = { 10497: 'repeat', 33071: 'clamp', 33648: 'mirror' };

/** Cofactor matrix (3x3 upper-left of a column-major 4x4), sign-corrected so it equals the inverse transpose up to a positive scale. */
function normalMatrix(m: readonly number[]): number[] {
  const a = m[0]!, b = m[4]!, c = m[8]!, d = m[1]!, e = m[5]!, f = m[9]!, g = m[2]!, h = m[6]!, i = m[10]!; // rows (a b c; d e f; g h i)
  const sign = determinant3(m) < 0 ? -1 : 1;
  // row-major cofactors C(r, c)
  return [
    (e * i - f * h) * sign, (f * g - d * i) * sign, (d * h - e * g) * sign,
    (c * h - b * i) * sign, (a * i - c * g) * sign, (b * g - a * h) * sign,
    (b * f - c * e) * sign, (c * d - a * f) * sign, (a * e - b * d) * sign,
  ];
}

const slug = (s: string): string => s.replace(/[^A-Za-z0-9._-]+/g, '_').replace(/^_+|_+$/g, '');

// ---------------------------------------------------------------------------------------------------------------
// materials and textures
// ---------------------------------------------------------------------------------------------------------------

function buildMaterials(json: GltfJson, gltfPath: string, buffers: readonly Uint8Array[], notes: string[]): { materials: GltfSceneMaterial[]; defaultMaterial: GltfSceneMaterial; textures: GltfTextureEntry[] } {
  const textures: GltfTextureEntry[] = [];
  const textureKey = new Map<string, GltfTextureEntry>();
  const imageSizeCache = new Map<number, { width: number; height: number }>();
  const entry = (texture: number, colorSpace: 'srgb' | 'linear', owner: string): GltfTextureEntry | undefined => {
    const k = `${texture}:${colorSpace}`;
    const have = textureKey.get(k);
    if (have) return have;
    const tex = json.textures?.[texture];
    if (!tex || tex.source === undefined) { notes.push(`${owner}: texture ${texture} has no image source (extension-only formats such as KTX2/WebP are not decoded)`); return undefined; }
    if (!imageSizeCache.has(tex.source)) {
      try {
        imageSizeCache.set(tex.source, gltfImageSizes(json, buffers, gltfPath, [tex.source]).get(tex.source)!);
      } catch (error) {
        // An image file that was not downloaded (Sponza's textures are not in MANIFEST.json yet) costs the texture, not the geometry.
        notes.push(`${owner}: image ${tex.source} cannot be read (${(error as Error).message}); the texture slot is dropped`);
        return undefined;
      }
    }
    const size = imageSizeCache.get(tex.source)!;
    const sampler = tex.sampler !== undefined ? json.samplers?.[tex.sampler] : undefined;
    const wrapS = WRAP[sampler?.wrapS ?? 10497] ?? 'repeat', wrapT = WRAP[sampler?.wrapT ?? 10497] ?? 'repeat';
    if (wrapS !== wrapT) notes.push(`${owner}: texture ${texture} wraps ${wrapS}/${wrapT} (S/T); the scene description has one wrap, S is used`);
    const e: GltfTextureEntry = {
      key: `t${texture}${colorSpace === 'linear' ? '_lin' : ''}`, texture, image: tex.source, colorSpace, width: size.width, height: size.height,
      wrap: wrapS, filter: sampler?.magFilter === 9728 ? 'nearest' : 'linear',
    };
    textureKey.set(k, e);
    textures.push(e);
    return e;
  };

  const convert = (m: GltfMaterialJson, index: number, converted: ConvertedMaterial): GltfSceneMaterial => {
    const owner = `material ${index}${m.name ? ` '${m.name}'` : ''}`;
    const mNotes = [...converted.notes];
    const pmr = m.pbrMetallicRoughness ?? {};
    const unlit = !!m.extensions?.['KHR_materials_unlit'];
    const infos: [TextureSlot, GltfTextureInfo | undefined][] = [
      ['baseColorTexture', pmr.baseColorTexture], ['metallicRoughnessTexture', pmr.metallicRoughnessTexture], ['normalTexture', m.normalTexture], ['emissiveTexture', m.emissiveTexture],
    ];
    const slots: GltfSceneMaterial['slots'] = {};
    const used: { slot: TextureSlot; texCoord: number; transform: UvTransform }[] = [];
    for (const [slot, info] of infos) {
      if (!info) continue;
      // Unlit: the base colour texture is the displayed colour, i.e. the emission (gltf-openpbr.ts maps the factor the same way).
      const target: TextureSlot = unlit && slot === 'baseColorTexture' ? 'emissiveTexture' : slot;
      const e = entry(info.index, SRGB_SLOTS.has(target) ? 'srgb' : 'linear', owner);
      if (!e) continue;
      const uv = textureUv(info);
      slots[target] = { key: e.key, texCoord: uv.texCoord };
      used.push({ slot: target, ...uv });
    }
    const out: GltfSceneMaterial = { index, ...(m.name !== undefined ? { name: m.name } : {}), converted, material: { ...converted.material }, slots, notes: mNotes };
    for (const slot of TEXTURE_SLOTS) if (slots[slot]) out.material[slot] = slots[slot]!.key;
    if (used.length > 0) {
      const first = used[0]!;
      out.texCoord = first.texCoord;
      out.uvTransform = first.transform;
      const conflict = used.find((u) => u.texCoord !== first.texCoord || !sameUv(u.transform, first.transform));
      if (conflict) mNotes.push(`textures read different uv sets or transforms (${first.slot} vs ${conflict.slot}): the scene copy has one uv set per mesh, ${first.slot}'s is used`);
    }
    // Texture slots that have no bench slot: report, never silently drop.
    const mapped = new Set(['baseColorTexture', 'metallicRoughnessTexture', 'normalTexture', 'emissiveTexture']);
    const dropped = converted.textures.filter((t) => !mapped.has(t));
    if (dropped.length) mNotes.push(`texture slots not mapped (no scene-copy slot): ${dropped.join(', ')}`);
    return out;
  };

  const materials = (json.materials ?? []).map((m, i) => convert(m, i, { index: i, ...(m.name !== undefined ? { name: m.name } : {}), ...gltfMaterialToOpenPbr(m) }));
  const defaultJson: GltfMaterialJson = { name: 'glTF default material' };
  const defaultMaterial = convert(defaultJson, -1, { index: -1, name: 'glTF default material', ...gltfMaterialToOpenPbr(defaultJson) });
  return { materials, defaultMaterial, textures };
}

// ---------------------------------------------------------------------------------------------------------------
// the loader
// ---------------------------------------------------------------------------------------------------------------

interface Group { bufs: { pos: FloatBuf; nor: FloatBuf; uv: FloatBuf; tan: FloatBuf; col: FloatBuf }; triangles: number; nodes: Set<string>; hasUv: boolean; hasTan: boolean; hasCol: boolean }

/** Load a .glb / .gltf into groups of triangles by material plus materials, textures, cameras and lights. */
export function loadGltfScene(path: string, opts: GltfSceneOptions = {}): GltfScene {
  const { json, buffers } = readGltfFile(path, opts.sha256 ? { sha256: opts.sha256 } : {});
  return gltfSceneFromJson(json, buffers, path, opts);
}

/** `loadGltfScene` on already-split content (tests build glTFs in memory; `gltfPath` is only used to find external images). */
export function gltfSceneFromJson(json: GltfJson, buffers: readonly Uint8Array[], gltfPath: string, opts: GltfSceneOptions = {}): GltfScene {
  const normalsMode = opts.normals ?? 'geometric', wantUv = opts.uvs ?? true;
  const meshes = json.meshes ?? [], nodes = json.nodes ?? [];
  const notes: string[] = [];
  const { materials, defaultMaterial, textures } = buildMaterials(json, gltfPath, buffers, notes);
  const lightDefs = json.extensions?.KHR_lights_punctual?.lights ?? [];
  const groups = new Map<number, Group>();
  const group = (m: number): Group => {
    let g = groups.get(m);
    if (!g) groups.set(m, g = { bufs: { pos: new FloatBuf(), nor: new FloatBuf(), uv: new FloatBuf(), tan: new FloatBuf(), col: new FloatBuf() }, triangles: 0, nodes: new Set(), hasUv: false, hasTan: false, hasCol: false });
    return g;
  };
  const cameras: GltfCamera[] = [], lights: LightDesc[] = [], unsupportedLights: string[] = [];
  let fileTriangles = 0, degenerate = 0, primitives = 0;
  const min: [number, number, number] = [Infinity, Infinity, Infinity], max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  const cache = new Map<string, Float64Array>();
  const read = (a: number): Float64Array => {
    let v = cache.get(String(a));
    if (!v) { v = readAccessor(json, buffers, a); if (cache.size > 64) cache.clear(); cache.set(String(a), v); }
    return v;
  };

  const visit = (ni: number, parent: readonly number[], depth: number): void => {
    if (depth > 64) throw new Error('glTF node hierarchy too deep (cycle?)');
    const node = nodes[ni];
    if (!node) throw new Error(`node ${ni} missing`);
    const world = multiply(parent, localMatrix(node));
    const nodeName = node.name ?? `node${ni}`;
    if (node.camera !== undefined) {
      const c = json.cameras?.[node.camera];
      if (c?.type === 'perspective' && c.perspective) {
        const fwd = [-world[8]!, -world[9]!, -world[10]!], len = Math.hypot(fwd[0]!, fwd[1]!, fwd[2]!) || 1;
        const upv = [world[4]!, world[5]!, world[6]!], ul = Math.hypot(upv[0]!, upv[1]!, upv[2]!) || 1;
        const pos: Vec3 = [world[12]!, world[13]!, world[14]!];
        cameras.push({
          name: c.name ?? nodeName,
          camera: { type: 'perspective', position: pos, target: [pos[0] + fwd[0]! / len, pos[1] + fwd[1]! / len, pos[2] + fwd[2]! / len], up: [upv[0]! / ul, upv[1]! / ul, upv[2]! / ul], yFovDeg: (c.perspective.yfov * 180) / Math.PI },
          ...(c.perspective.aspectRatio !== undefined ? { aspectRatio: c.perspective.aspectRatio } : {}), znear: c.perspective.znear,
        });
      } else if (c) notes.push(`camera '${c.name ?? nodeName}' is ${c.type}: only perspective cameras are read`);
    }
    const lightRef = node.extensions?.KHR_lights_punctual;
    if (lightRef) {
      const l = lightDefs[lightRef.light];
      if (l?.type === 'directional') {
        const d = [-world[8]!, -world[9]!, -world[10]!], len = Math.hypot(d[0]!, d[1]!, d[2]!) || 1;
        const c = l.color ?? [1, 1, 1], lux = l.intensity ?? 1;
        lights.push({ type: 'directional', direction: [d[0]! / len, d[1]! / len, d[2]! / len], irradiance: [c[0]! * lux, c[1]! * lux, c[2]! * lux] });
      } else unsupportedLights.push(`${l?.type ?? 'unknown'} ${l?.name ?? nodeName}`);
    }
    if (node.mesh !== undefined) {
      const mirror = determinant3(world) < 0;
      const nm = normalMatrix(world);
      for (const prim of meshes[node.mesh]!.primitives) {
        const mode = prim.mode ?? 4;
        if (mode !== 4) throw new Error(`primitive mode ${mode} is not triangles (only mode 4 is supported)`);
        primitives++;
        const mi = prim.material ?? -1;
        const sm = mi >= 0 ? materials[mi] : defaultMaterial;
        if (!sm) throw new Error(`primitive uses material ${mi}, which does not exist`);
        const g = group(mi);
        g.nodes.add(nodeName);
        const p = read(prim.attributes['POSITION']!);
        const idx = prim.indices !== undefined ? read(prim.indices) : Array.from({ length: p.length / 3 }, (_, i) => i);
        const useShading = normalsMode === 'shading' && prim.attributes['NORMAL'] !== undefined;
        const nIn = useShading ? read(prim.attributes['NORMAL']!) : undefined;
        const uvAttr = wantUv && sm.texCoord !== undefined ? prim.attributes[`TEXCOORD_${sm.texCoord}`] : undefined;
        if (wantUv && sm.texCoord !== undefined && uvAttr === undefined) notes.push(`material ${mi}: primitive of '${nodeName}' lacks TEXCOORD_${sm.texCoord}; uvs are zero`);
        const uvIn = uvAttr !== undefined ? read(uvAttr) : undefined;
        const emitUv = wantUv && sm.texCoord !== undefined;
        const tanIn = opts.tangents && prim.attributes['TANGENT'] !== undefined ? read(prim.attributes['TANGENT']!) : undefined;
        const colIn = opts.colors && prim.attributes['COLOR_0'] !== undefined ? read(prim.attributes['COLOR_0']!) : undefined;
        const colN = colIn ? (json.accessors![prim.attributes['COLOR_0']!]!.type === 'VEC3' ? 3 : 4) : 0;
        const handed = mirror ? -1 : 1;
        for (let t = 0; t + 2 < idx.length; t += 3) {
          fileTriangles++;
          const order = mirror ? [0, 2, 1] : [0, 1, 2];
          const vi = order.map((k) => idx[t + k]!);
          const v = vi.map((i) => transformPoint(world, [p[i * 3]!, p[i * 3 + 1]!, p[i * 3 + 2]!]));
          const f = v.map((q) => q.map(Math.fround) as unknown as Vec3);
          const e1: Vec3 = [f[1]![0] - f[0]![0], f[1]![1] - f[0]![1], f[1]![2] - f[0]![2]];
          const e2: Vec3 = [f[2]![0] - f[0]![0], f[2]![1] - f[0]![1], f[2]![2] - f[0]![2]];
          const c = cross3(e1, e2);
          const len = Math.hypot(c[0], c[1], c[2]);
          if (!(len > 0)) { degenerate++; continue; }
          g.triangles++;
          for (let k = 0; k < 3; k++) {
            const q = f[k]!, i = vi[k]!;
            g.bufs.pos.push(q[0], q[1], q[2]);
            if (nIn) {
              const nx = nIn[i * 3]!, ny = nIn[i * 3 + 1]!, nz = nIn[i * 3 + 2]!;
              const x = nm[0]! * nx + nm[1]! * ny + nm[2]! * nz, y = nm[3]! * nx + nm[4]! * ny + nm[5]! * nz, z = nm[6]! * nx + nm[7]! * ny + nm[8]! * nz;
              const l = Math.hypot(x, y, z);
              if (l > 0) g.bufs.nor.push(x / l, y / l, z / l); else g.bufs.nor.push(c[0] / len, c[1] / len, c[2] / len);
            } else g.bufs.nor.push(c[0] / len, c[1] / len, c[2] / len);
            for (let a = 0; a < 3; a++) { min[a] = Math.min(min[a]!, q[a]!); max[a] = Math.max(max[a]!, q[a]!); }
            if (emitUv) {
              g.hasUv = true;
              if (uvIn) {
                let u = uvIn[i * 2]!, w = uvIn[i * 2 + 1]!;
                if (sm.uvTransform && sm.uvTransform !== IDENTITY_UV) [u, w] = applyUvTransform(sm.uvTransform, u, w);
                g.bufs.uv.push(u, w);
              } else g.bufs.uv.push(0, 0);
            }
            if (tanIn) {
              g.hasTan = true;
              const tx = tanIn[i * 4]!, ty = tanIn[i * 4 + 1]!, tz = tanIn[i * 4 + 2]!, tw = tanIn[i * 4 + 3]!;
              const x = world[0]! * tx + world[4]! * ty + world[8]! * tz, y = world[1]! * tx + world[5]! * ty + world[9]! * tz, z = world[2]! * tx + world[6]! * ty + world[10]! * tz;
              const l = Math.hypot(x, y, z) || 1;
              g.bufs.tan.push(x / l, y / l, z / l, tw * handed);
            }
            if (colIn) {
              g.hasCol = true;
              g.bufs.col.push(colIn[i * colN]!, colIn[i * colN + 1]!, colIn[i * colN + 2]!, colN === 4 ? colIn[i * 4 + 3]! : 1);
            }
          }
        }
      }
    }
    for (const c of node.children ?? []) visit(c, world, depth + 1);
  };
  for (const r of rootNodes(json)) visit(r, compose(), 0);

  const out: GltfMaterialGroup[] = [...groups.entries()]
    .sort(([a], [b]) => (a < 0 ? Infinity : a) - (b < 0 ? Infinity : b))
    .map(([mi, g]) => {
      const nv = g.bufs.pos.length / 3;
      const indices = new Uint32Array(nv);
      for (let i = 0; i < nv; i++) indices[i] = i;
      const mesh: TriangleMesh = { positions: g.bufs.pos.done(), normals: g.bufs.nor.done(), indices };
      if (g.hasUv && g.bufs.uv.length === nv * 2) mesh.uvs = g.bufs.uv.done();
      if (g.hasTan && g.bufs.tan.length === nv * 4) mesh.tangents = g.bufs.tan.done();
      if (g.hasCol && g.bufs.col.length === nv * 4) mesh.colors = g.bufs.col.done();
      return { materialIndex: mi, name: mi >= 0 ? (materials[mi]?.name ?? `material${mi}`) : 'glTF default material', mesh, triangles: g.triangles, nodes: [...g.nodes] };
    });

  for (const m of materials) for (const n of m.notes) if (/not mapped|different uv/.test(n)) notes.push(`material ${m.index}${m.name ? ` '${m.name}'` : ''}: ${n}`);
  const usedDefault = groups.has(-1);
  const unsupportedExt = (json.extensionsUsed ?? []).filter((e) => !/^KHR_(materials_(ior|specular|transmission|volume|clearcoat|sheen|iridescence|anisotropy|emissive_strength|dispersion|diffuse_transmission|unlit)|lights_punctual|texture_transform)$/.test(e));
  if (unsupportedExt.length) notes.push(`extensions not read: ${unsupportedExt.join(', ')}`);
  return {
    materials, ...(usedDefault ? { defaultMaterial } : {}), groups: out, textures, cameras, lights, unsupportedLights, extensionsUsed: json.extensionsUsed ?? [], notes,
    stats: { fileTriangles, degenerateDropped: degenerate, triangles: out.reduce((s, g) => s + g.triangles, 0), primitives, bounds: { min, max } },
  };
}

/** The texture table of a loaded scene as `SceneDesc.textures` entries (only the keys in `keep`, default all). */
export function gltfTextureDescs(scene: GltfScene, path: string, sha256?: string, keep?: ReadonlySet<string>): Record<string, TextureDesc> {
  const out: Record<string, TextureDesc> = {};
  for (const t of scene.textures) {
    if (keep && !keep.has(t.key)) continue;
    out[t.key] = { generator: 'image', path, image: t.image, ...(sha256 ? { sha256 } : {}), width: t.width, height: t.height, colorSpace: t.colorSpace, wrap: t.wrap, filter: t.filter };
  }
  return out;
}

export { gltfMaterialsToOpenPbr, slug as gltfSlug };
