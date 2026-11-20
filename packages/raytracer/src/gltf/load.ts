/**
 * `loadGltf`: a .glb / .gltf file (URL or bytes) -> the input of `buildSceneCopy` / `buildGpuScene`, with the file's cameras, notes and bounds.
 *
 * One mesh and one instance per glTF material (the triangles of every node that uses it, baked to world space, `gltfSceneFromJson`), one
 * scene-copy material per glTF material (`gltfMaterialToOpenPbr` -> `openPbrToRtMaterial`), plus the glTF default material when a primitive
 * has none. Textures: base colour, metallic-roughness, normal and emissive images, decoded to RGBA8 (row 0 = top). The first
 * `KHR_lights_punctual` directional light becomes the sun; point and spot lights are listed in `notes`. No environment: set
 * `input.environment` yourself.
 *
 * Images are decoded with `createImageBitmap` + `OffscreenCanvas` by default (browsers and workers): 8 bits per channel, no colour-space
 * conversion, no premultiplication on the bitmap. A 2D canvas may still premultiply internally, so the colour of nearly transparent texels
 * can lose precision; pass `decodeImage` for an exact decoder (and in Node, where there is no default).
 */
import type { SunInput } from '../scene/lights.ts';
import type { RtMaterialInput } from '../scene/material.ts';
import type { SceneCopyInput, SceneCopyInstance, SceneCopyMesh } from '../scene/scene-copy.ts';
import type { TextureInput } from '../scene/texture.ts';
import { isGlb, parseGlb } from './glb.ts';
import { openPbrToRtMaterial } from './material.ts';
import type { GltfJson } from './read.ts';
import { gltfSceneFromJson } from './scene.ts';
import type { GltfCamera, GltfNormalMode, GltfScene } from './scene.ts';

/** RGBA8 pixels, row 0 = top. */
export interface GltfDecodedImage { width: number; height: number; data: Uint8Array }

/** Decodes one image (PNG, JPEG, WebP, ...): `bytes` as stored in the file, `mimeType` when the glTF names one. */
export type GltfImageDecoder = (bytes: Uint8Array, mimeType?: string) => Promise<GltfDecodedImage>;

export interface LoadGltfOptions {
  /** Vertex normals: the file's `NORMAL` (`'shading'`, default) or the flat geometric normal (`'geometric'`, what the reference scenes use). */
  normals?: GltfNormalMode;
  /** Read textures (default true). Without them materials keep their factors only. */
  textures?: boolean;
  /** Carry `TANGENT` for anisotropic materials. Default: when the file uses `KHR_materials_anisotropy`. */
  tangents?: boolean;
  /** Image decoder. Default: `createImageBitmap` + `OffscreenCanvas` (see the module comment); required where those do not exist (Node). */
  decodeImage?: GltfImageDecoder;
  /** Base URL of relative `uri`s when `source` is bytes. Default: the source URL, else the page location. */
  baseUrl?: string | URL;
  /** `fetch` to use (default the global one): add credentials, caching or progress reporting. */
  fetch?: (url: string) => Promise<Response>;
}

/** What `loadGltf` returns: a complete `SceneCopyInput` plus what the file says about cameras, and how the conversion went. */
export interface GltfSceneInput extends SceneCopyInput {
  meshes: SceneCopyMesh[];
  instances: SceneCopyInstance[];
  materials: RtMaterialInput[];
  textures?: TextureInput[];
  sun: SunInput | null;
  /** Name of each entry of `materials` (the glTF material name, `material<i>`, or `glTF default material`). */
  materialNames: string[];
  /** Perspective cameras of the file, in node order (look-at form: hand one to the path tracer's `setCamera`). */
  cameras: GltfCamera[];
  /** World-space bounds of the triangles. */
  bounds: { min: [number, number, number]; max: [number, number, number] };
  /** Everything dropped or approximated: unsupported extensions, texture slots, lights, images that could not be decoded. */
  notes: string[];
  /** The intermediate scene (groups, converted materials, texture entries, statistics). */
  gltf: GltfScene;
}

/** The default image decoder (browsers and workers). */
export const decodeImageWithCanvas: GltfImageDecoder = async (bytes, mimeType) => {
  if (typeof createImageBitmap !== 'function' || typeof OffscreenCanvas !== 'function') {
    throw new Error('loadGltf: no createImageBitmap / OffscreenCanvas here; pass options.decodeImage (or textures: false)');
  }
  const blob = new Blob([bytes as Uint8Array<ArrayBuffer>], mimeType ? { type: mimeType } : {});
  const bitmap = await createImageBitmap(blob, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
  const { width, height } = bitmap;
  const canvas = new OffscreenCanvas(width, height);
  const context = canvas.getContext('2d', { willReadFrequently: true });
  if (!context) throw new Error('loadGltf: no 2D context on OffscreenCanvas');
  context.drawImage(bitmap, 0, 0);
  bitmap.close();
  const pixels = context.getImageData(0, 0, width, height);
  return { width, height, data: new Uint8Array(pixels.data.buffer, pixels.data.byteOffset, pixels.data.byteLength) };
};

const SLOT_INFOS = (m: NonNullable<GltfJson['materials']>[number]) => [m.pbrMetallicRoughness?.baseColorTexture, m.pbrMetallicRoughness?.metallicRoughnessTexture, m.normalTexture, m.emissiveTexture];

/** Images that a mapped texture slot uses (the ones worth decoding). */
function usedImages(json: GltfJson): number[] {
  const out = new Set<number>();
  for (const m of json.materials ?? []) {
    for (const info of SLOT_INFOS(m)) {
      const source = info ? json.textures?.[info.index]?.source : undefined;
      if (source !== undefined) out.add(source);
    }
  }
  return [...out].sort((a, b) => a - b);
}

async function imageBytes(json: GltfJson, buffers: readonly Uint8Array[], index: number, resolve: (uri: string) => Promise<Uint8Array>): Promise<{ bytes: Uint8Array; mimeType?: string }> {
  const image = json.images?.[index];
  if (!image) throw new Error(`image ${index} missing`);
  if (image.bufferView !== undefined) {
    const view = json.bufferViews?.[image.bufferView];
    const buffer = view && buffers[view.buffer];
    if (!view || !buffer) throw new Error(`image ${index}: bufferView ${image.bufferView} missing`);
    return { bytes: buffer.subarray(view.byteOffset ?? 0, (view.byteOffset ?? 0) + view.byteLength), ...(image.mimeType ? { mimeType: image.mimeType } : {}) };
  }
  if (image.uri === undefined) throw new Error(`image ${index} has neither bufferView nor uri`);
  return { bytes: await resolve(image.uri), ...(image.mimeType ? { mimeType: image.mimeType } : {}) };
}

/** Loads a glTF 2.0 file (.glb, or .gltf with external or embedded buffers and images) as a scene-copy input. */
export async function loadGltf(source: string | URL | ArrayBuffer | Uint8Array, options: LoadGltfOptions = {}): Promise<GltfSceneInput> {
  const doFetch = options.fetch ?? ((url: string) => fetch(url));
  const pageBase = typeof location !== 'undefined' ? location.href : undefined;
  let bytes: Uint8Array;
  let base: string | undefined;
  if (typeof source === 'string' || source instanceof URL) {
    const url = new URL(String(source), pageBase).href;
    const response = await doFetch(url);
    if (!response.ok) throw new Error(`loadGltf: ${url}: HTTP ${response.status}`);
    bytes = new Uint8Array(await response.arrayBuffer());
    base = options.baseUrl !== undefined ? String(options.baseUrl) : url;
  } else {
    bytes = source instanceof Uint8Array ? source : new Uint8Array(source);
    base = options.baseUrl !== undefined ? String(options.baseUrl) : pageBase;
  }
  const resolve = async (uri: string): Promise<Uint8Array> => {
    const url = uri.startsWith('data:') ? uri : base ? new URL(uri, base).href : uri;
    const response = await doFetch(url);
    if (!response.ok) throw new Error(`loadGltf: ${uri.startsWith('data:') ? 'data: URI' : url}: HTTP ${response.status}`);
    return new Uint8Array(await response.arrayBuffer());
  };

  let json: GltfJson;
  let bin: Uint8Array | undefined;
  if (isGlb(bytes)) ({ json, bin } = parseGlb(bytes));
  else json = JSON.parse(new TextDecoder().decode(bytes)) as GltfJson;
  for (const ext of json.extensionsRequired ?? []) {
    if (ext === 'KHR_draco_mesh_compression' || ext === 'EXT_meshopt_compression' || ext === 'KHR_mesh_quantization') throw new Error(`loadGltf: the file requires ${ext}, which this loader does not decode`);
  }
  const buffers = await Promise.all((json.buffers ?? []).map(async (b, i) => {
    if (b.uri === undefined) {
      if (i !== 0 || !bin) throw new Error(`loadGltf: buffer ${i} has no uri and no GLB BIN chunk`);
      return bin;
    }
    return resolve(b.uri);
  }));

  const notes: string[] = [];
  const images = new Map<number, GltfDecodedImage>();
  if (options.textures ?? true) {
    const decode = options.decodeImage ?? decodeImageWithCanvas;
    await Promise.all(usedImages(json).map(async (i) => {
      try {
        const { bytes: data, mimeType } = await imageBytes(json, buffers, i, resolve);
        images.set(i, await decode(data, mimeType));
      } catch (error) {
        notes.push(`image ${i}: ${(error as Error).message}`);
      }
    }));
  }
  const tangents = options.tangents ?? (json.extensionsUsed ?? []).includes('KHR_materials_anisotropy');
  const scene = gltfSceneFromJson(json, buffers, {
    normals: options.normals ?? 'shading',
    uvs: options.textures ?? true,
    tangents,
    imageSize: (i) => {
      const image = images.get(i);
      if (!image) throw new Error(options.textures === false ? 'textures are off' : 'not decoded');
      return { width: image.width, height: image.height };
    },
  });
  const input = gltfSceneToInput(scene, images);
  input.notes.unshift(...notes);
  return input;
}

/** `gltfSceneFromJson` output + decoded images (by glTF image index) -> a scene-copy input. `loadGltf` is this after reading the file. */
export function gltfSceneToInput(scene: GltfScene, images: ReadonlyMap<number, GltfDecodedImage>): GltfSceneInput {
  const textures: TextureInput[] = [];
  const textureIndex = new Map<string, number>();
  for (const t of scene.textures) {
    const image = images.get(t.image);
    if (!image) continue;
    textureIndex.set(t.key, textures.length);
    textures.push({ width: image.width, height: image.height, data: image.data, colorSpace: t.colorSpace, wrap: t.wrap, label: t.key });
  }
  const texture = (key: string): number => {
    const i = textureIndex.get(key);
    if (i === undefined) throw new RangeError(`texture '${key}' was not decoded`);
    return i;
  };
  const entries = [...scene.materials, ...(scene.defaultMaterial ? [scene.defaultMaterial] : [])];
  const materials = entries.map((m) => openPbrToRtMaterial(m.params, texture));
  const materialNames = entries.map((m) => m.name ?? (m.index < 0 ? 'glTF default material' : `material${m.index}`));
  const materialOf = (gltfIndex: number): number => (gltfIndex < 0 ? scene.materials.length : gltfIndex);

  const meshes: SceneCopyMesh[] = [];
  const instances: SceneCopyInstance[] = [];
  const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  for (const group of scene.groups) {
    const { positions, normals, indices, uvs, tangents } = group.mesh;
    meshes.push({ positions, normals, indices, ...(uvs ? { uvs } : {}), ...(tangents ? { tangents } : {}) });
    instances.push({ meshIndex: meshes.length - 1, instanceIndex: meshes.length - 1, objectToWorld: identity, material: materialOf(group.materialIndex), mask: 0x01 });
  }

  const light = scene.lights[0];
  const sun: SunInput | null = light ? { direction: light.direction, irradiance: light.irradiance } : null;
  const notes = [...scene.notes];
  if (scene.lights.length > 1) notes.push(`${scene.lights.length} directional lights: the scene copy holds one sun, the first is used`);
  if (scene.unsupportedLights.length) notes.push(`lights not mapped: ${scene.unsupportedLights.join(', ')}`);
  return {
    meshes, instances, materials, materialNames, ...(textures.length ? { textures } : {}), sun, environment: null,
    cameras: scene.cameras, bounds: scene.stats.bounds, notes, gltf: scene,
  };
}
