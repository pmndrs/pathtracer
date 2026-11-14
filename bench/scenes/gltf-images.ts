/**
 * glTF image bytes and the registered decoder for `generator: 'image'` textures (Node only).
 * Importing this module registers the decoder with `textures.ts` (`setImageTextureLoader`), so `generateTexture` / `sceneDescToCopyInput`
 * read glTF images without the scene description carrying any pixels.
 */
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { ASSETS_DIR } from './assets-dir.ts';
import { readGltfFile } from './gltf-loader.ts';
import type { GltfJson } from './gltf-read.ts';
import { decodeImage, imageSize } from './image-decode.ts';
import type { DecodedImage } from './image-decode.ts';
import { setImageTextureLoader } from './textures.ts';
import type { TextureDesc } from './schema.ts';

/** Bytes (and mime type) of `images[index]`: a GLB bufferView, a `data:` URI or a file next to the glTF. */
export function readGltfImageBytes(json: GltfJson, buffers: readonly Uint8Array[], gltfPath: string, index: number): { bytes: Uint8Array; mimeType?: string } {
  const im = json.images?.[index];
  if (!im) throw new Error(`${gltfPath}: image ${index} missing`);
  if (im.bufferView !== undefined) {
    const v = json.bufferViews?.[im.bufferView];
    const buf = v && buffers[v.buffer];
    if (!v || !buf) throw new Error(`${gltfPath}: image ${index} bufferView ${im.bufferView} missing`);
    return { bytes: buf.subarray(v.byteOffset ?? 0, (v.byteOffset ?? 0) + v.byteLength), ...(im.mimeType ? { mimeType: im.mimeType } : {}) };
  }
  if (im.uri === undefined) throw new Error(`${gltfPath}: image ${index} has neither bufferView nor uri`);
  if (im.uri.startsWith('data:')) return { bytes: Buffer.from(im.uri.slice(im.uri.indexOf(',') + 1), 'base64'), ...(im.mimeType ? { mimeType: im.mimeType } : {}) };
  return { bytes: readFileSync(resolve(dirname(gltfPath), decodeURIComponent(im.uri))), ...(im.mimeType ? { mimeType: im.mimeType } : {}) };
}

/** Size of every listed image without decoding (one file read). */
export function gltfImageSizes(json: GltfJson, buffers: readonly Uint8Array[], gltfPath: string, indices: readonly number[]): Map<number, { width: number; height: number }> {
  const out = new Map<number, { width: number; height: number }>();
  for (const i of indices) out.set(i, imageSize(readGltfImageBytes(json, buffers, gltfPath, i).bytes));
  return out;
}

// One parsed file and a small decode cache: a scene's textures all come from one glTF, and several textures can share an image.
let lastFile: { path: string; json: GltfJson; buffers: Buffer[] } | undefined;
const decoded = new Map<string, DecodedImage>();

export function decodeGltfImage(path: string, image: number, sha256?: string): DecodedImage {
  const key = `${path}#${image}`;
  let img = decoded.get(key);
  if (!img) {
    if (lastFile?.path !== path) lastFile = { path, ...readGltfFile(path, sha256 ? { sha256 } : {}) };
    const { bytes, mimeType } = readGltfImageBytes(lastFile.json, lastFile.buffers, path, image);
    img = decodeImage(bytes, mimeType);
    decoded.set(key, img);
  }
  return img;
}

setImageTextureLoader((desc: Extract<TextureDesc, { generator: 'image' }>) => decodeGltfImage(resolve(ASSETS_DIR, desc.path), desc.image, desc.sha256));
