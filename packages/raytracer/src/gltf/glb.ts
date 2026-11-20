/** GLB container reading (pure): the JSON chunk and the first BIN chunk of a binary glTF. */
import type { GltfJson } from './read.ts';

const GLB_MAGIC = 0x46546c67; // 'glTF'
const CHUNK_JSON = 0x4e4f534a;
const CHUNK_BIN = 0x004e4942;

/** Is `bytes` a GLB (binary glTF) container? */
export function isGlb(bytes: Uint8Array): boolean {
  return bytes.byteLength >= 12 && new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getUint32(0, true) === GLB_MAGIC;
}

/** Splits a GLB into its JSON and (optional) BIN chunk. The BIN chunk is a view into `bytes`, not a copy. */
export function parseGlb(bytes: Uint8Array): { json: GltfJson; bin?: Uint8Array } {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (bytes.byteLength < 20 || view.getUint32(0, true) !== GLB_MAGIC) throw new Error('not a GLB file (bad magic)');
  const version = view.getUint32(4, true);
  if (version !== 2) throw new Error(`unsupported GLB version ${version}`);
  const end = Math.min(bytes.byteLength, view.getUint32(8, true));
  let offset = 12;
  let json: GltfJson | undefined;
  let bin: Uint8Array | undefined;
  while (offset + 8 <= end) {
    const length = view.getUint32(offset, true);
    const type = view.getUint32(offset + 4, true);
    const data = bytes.subarray(offset + 8, offset + 8 + length);
    if (type === CHUNK_JSON) json = JSON.parse(new TextDecoder().decode(data)) as GltfJson;
    else if (type === CHUNK_BIN && !bin) bin = data;
    offset += 8 + length;
  }
  if (!json) throw new Error('GLB has no JSON chunk');
  return { json, ...(bin ? { bin } : {}) };
}
