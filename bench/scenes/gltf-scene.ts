/**
 * glTF 2.0 scene loader for the bench (Node; P2.19): reads a file, then groups its triangles by glTF material with the public reader
 * (`gltfSceneFromJson` of `@pmndrs/raytracer`, packages/raytracer/src/gltf/scene.ts, where the geometry, texture, camera and light rules
 * are documented). This module adds what only the bench needs: file reading with a sha256 check, image sizes from the files, the bench
 * `openpbr` `MaterialDesc` with its reference class, and `SceneDesc.textures` entries.
 *
 * `gltf-loader.ts` stays the Sponza-style "everything as one flat mesh" reader and is unchanged in output.
 */
import { applyGltfUvTransform as applyUvTransform, gltfSceneFromJson as rtGltfSceneFromJson, GLTF_TEXTURE_SLOTS } from '@pmndrs/raytracer';
import type { GltfMesh, GltfSceneMaterial as RtGltfSceneMaterial, GltfTextureEntry, GltfUvTransform, GltfTextureSlot } from '@pmndrs/raytracer';
import { readGltfFile } from './gltf-loader.ts';
import { gltfImageSizes } from './gltf-images.ts';
import { gltfMaterialToOpenPbr, gltfMaterialsToOpenPbr } from './gltf-openpbr.ts';
import type { GltfOpenPbrResult } from './gltf-openpbr.ts';
import type { GltfJson } from './gltf-read.ts';
import type { CameraDesc, LightDesc, MaterialDesc, TextureDesc } from './schema.ts';
import type { TriangleMesh } from './geometry.ts';

export type NormalMode = 'geometric' | 'shading';
export type TextureSlot = GltfTextureSlot;
export type UvTransform = GltfUvTransform;
export type { GltfTextureEntry };
export { applyUvTransform };
/** The four texture slots the bench materials have, in the order a material's uv set is chosen. */
export const TEXTURE_SLOTS = GLTF_TEXTURE_SLOTS;

export interface GltfSceneOptions {
  /** Hex sha256 of the main file, verified. */
  sha256?: string;
  /** Vertex normals: flat geometric (default, as `gltf-loader.ts`) or the file's. */
  normals?: NormalMode;
  /** Carry the material's texture coordinates (default true; only primitives whose material uses a mapped texture get any). */
  uvs?: boolean;
  /** Carry `TANGENT` (xyzw, transformed) when present, and TEXCOORD_0 on the primitives of anisotropic materials (the uv frame the references derive the tangent from). Default false. */
  tangents?: boolean;
  /** Carry `COLOR_0` as rgba when present. Default false. */
  colors?: boolean;
}

export interface ConvertedMaterial extends GltfOpenPbrResult { index: number; name?: string }

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

const slug = (s: string): string => s.replace(/[^A-Za-z0-9._-]+/g, '_').replace(/^_+|_+$/g, '');

/** Load a .glb / .gltf into groups of triangles by material plus materials, textures, cameras and lights. */
export function loadGltfScene(path: string, opts: GltfSceneOptions = {}): GltfScene {
  const { json, buffers } = readGltfFile(path, opts.sha256 ? { sha256: opts.sha256 } : {});
  return gltfSceneFromJson(json, buffers, path, opts);
}

/** `loadGltfScene` on already-split content (tests build glTFs in memory; `gltfPath` is only used to find external images). */
export function gltfSceneFromJson(json: GltfJson, buffers: readonly Uint8Array[], gltfPath: string, opts: GltfSceneOptions = {}): GltfScene {
  const { sha256: _sha256, ...sceneOptions } = opts;
  void _sha256;
  const scene = rtGltfSceneFromJson(json, buffers, { ...sceneOptions, imageSize: (image) => gltfImageSizes(json, buffers, gltfPath, [image]).get(image)! });
  // The bench's material entries: the reference class on top of the package's mapping (recomputed: the mapping is pure).
  const bench = (m: RtGltfSceneMaterial, source: Parameters<typeof gltfMaterialToOpenPbr>[0]): GltfSceneMaterial => {
    const converted: ConvertedMaterial = { index: m.index, ...(m.name !== undefined ? { name: m.name } : {}), ...gltfMaterialToOpenPbr(source) };
    return {
      index: m.index, ...(m.name !== undefined ? { name: m.name } : {}), converted, material: { type: 'openpbr', ...m.params }, slots: m.slots, notes: m.notes,
      ...(m.texCoord !== undefined ? { texCoord: m.texCoord } : {}), ...(m.uvTransform ? { uvTransform: m.uvTransform } : {}),
    };
  };
  const materials = scene.materials.map((m, i) => bench(m, json.materials![i]!));
  const defaultMaterial = scene.defaultMaterial ? bench(scene.defaultMaterial, { name: 'glTF default material' }) : undefined;
  const groups: GltfMaterialGroup[] = scene.groups.map((g) => ({ ...g, mesh: g.mesh as GltfMesh as TriangleMesh }));
  const { defaultMaterial: _rtDefault, ...common } = scene;
  void _rtDefault;
  return { ...common, materials, ...(defaultMaterial ? { defaultMaterial } : {}), groups };
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
