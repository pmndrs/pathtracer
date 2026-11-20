/**
 * glTF 2.0 material -> OpenPBR-subset `MaterialDesc` with the reference class (docs/internal/design/tier1.md section 2, DIVERGES V53).
 * The mapping itself lives in `@pmndrs/raytracer` (gltf/material.ts, the public glTF loader uses it); this module adds the bench's
 * reference class (`classifyOpenPbr`) and keeps the old result shape. `readGltfJson` reads a file (Node).
 */
import { readFileSync } from 'node:fs';
import { gltfMaterialToOpenPbr as rtGltfMaterialToOpenPbr } from '@pmndrs/raytracer';
import type { GltfMaterialExtensions, GltfMaterialJson } from '@pmndrs/raytracer';
import { parseGlbContainer } from './gltf-loader.ts';
import { classifyOpenPbr } from './openpbr.ts';
import type { ReferenceClass } from './openpbr.ts';
import type { MaterialDesc } from './schema.ts';

export { GLTF_DEFAULTS, MAPPED_EXTENSIONS, GLTF_CLEARCOAT_IOR } from '@pmndrs/raytracer';
export type { GltfTextureInfo, GltfMaterialExtensions, GltfMaterialJson } from '@pmndrs/raytracer';

export interface GltfOpenPbrResult {
  /** The `openpbr` material, sparse: set fields only (`resolveOpenPbr` fills the rest). */
  material: Extract<MaterialDesc, { type: 'openpbr' }>;
  class: ReferenceClass;
  classReasons: string[];
  /** Things dropped, approximated or lossy, one line each. */
  notes: string[];
  /** glTF texture slots the material has (not mapped here: textures are the loader's job). */
  textures: string[];
  /** Extensions on the material this function does not map. */
  unsupported: string[];
}

/** Maps one glTF material. `extensions` defaults to `material.extensions`; pass another object to override (tests, sidecar overrides). */
export function gltfMaterialToOpenPbr(material: GltfMaterialJson, extensions: GltfMaterialExtensions = material.extensions ?? {}): GltfOpenPbrResult {
  const r = rtGltfMaterialToOpenPbr(material, extensions);
  const cls = classifyOpenPbr(r.params, { lostTransmissionTint: r.lostTransmissionTint });
  return { material: { type: 'openpbr', ...r.params }, class: cls.class, classReasons: cls.reasons, notes: r.notes, textures: r.textures, unsupported: r.unsupported };
}

// ---------------------------------------------------------------------------------------------------------------
// Whole-document helpers
// ---------------------------------------------------------------------------------------------------------------

export interface ConvertedMaterial extends GltfOpenPbrResult { index: number; name?: string }

/** Converts every entry of `json.materials` (a glTF with no `materials` gives []). */
export function gltfMaterialsToOpenPbr(json: { materials?: GltfMaterialJson[] }): ConvertedMaterial[] {
  return (json.materials ?? []).map((m, index) => ({ index, ...(m.name !== undefined ? { name: m.name } : {}), ...gltfMaterialToOpenPbr(m) }));
}

/** The JSON of a `.gltf` or the JSON chunk of a `.glb` (no buffers are read). */
export function readGltfJson(path: string): { materials?: GltfMaterialJson[]; extensionsUsed?: string[] } {
  const buf = readFileSync(path);
  if (buf.length >= 4 && buf.readUInt32LE(0) === 0x46546c67) return parseGlbContainer(buf).json as unknown as { materials?: GltfMaterialJson[]; extensionsUsed?: string[] };
  return JSON.parse(buf.toString('utf8')) as { materials?: GltfMaterialJson[]; extensionsUsed?: string[] };
}
