/**
 * `SceneDesc` of a glTF model: one `file` mesh per glTF material (`select.material`), the model's material table as `openpbr`
 * materials, and the sidecar's camera, lights, film, integrator and overrides (docs/design/tier1.md section 4).
 *
 * Browser-safe on purpose (no node imports): the bench pages import every scene through `scenes/index.ts`. The material table comes
 * either from the committed golden tables (`tier1/golden/*.json`, static scenes) or from `loadGltfScene` (`gltfSceneDescFromFile` in
 * `gltf-scene-desc.ts`, reads the downloaded model); a test asserts the two agree.
 */
import { scaling } from './geometry.ts';
import type { CameraDesc, EnvironmentDesc, FilmDesc, IntegratorDesc, LightDesc, MaterialDesc, MeshDesc, SceneDesc, TextureDesc } from './schema.ts';

export interface GltfModelRef { /** Relative to bench/assets. */ path: string; format: 'glb' | 'gltf'; sha256: string }

export interface GltfDescMaterial { index: number; name?: string; material: MaterialDesc }

export interface GltfSceneBuild {
  name: string;
  purpose: string;
  tier?: 0 | 1 | 2 | 3;
  model: GltfModelRef;
  /** The model's `materials` in order (index = glTF material index). */
  materials: readonly GltfDescMaterial[];
  /** Add a mesh and a material for primitives that have no glTF material (`select.material` -1). */
  defaultMaterial?: MaterialDesc;
  /** Textures the materials may reference (kept only when a kept material uses them). */
  textures?: Readonly<Record<string, TextureDesc>>;
  normals?: 'geometric' | 'shading';
  /** Uniform scale applied to every mesh (a model authored in millimetres: `MetalRoughSpheresNoTextures` is 7 mm wide; scale 1000 makes it 7 m, and the camera is given in scaled units). Default 1. */
  scale?: number;
  camera: CameraDesc;
  film: FilmDesc;
  integrator: IntegratorDesc;
  lights?: LightDesc[];
  environment?: EnvironmentDesc;
  /** Materials (by glTF name or index) whose meshes are left out (labels). */
  dropMaterials?: readonly (string | number)[];
  /** Materials (by glTF name or index) replaced by another description (a textured backdrop -> factor-only grey). */
  replace?: Readonly<Record<string | number, MaterialDesc>>;
}

const slug = (s: string): string => s.replace(/[^A-Za-z0-9._-]+/g, '_').replace(/^_+|_+$/g, '');

/** Scene-material key of glTF material `index`. */
export const gltfMaterialKey = (index: number, name?: string): string => (index < 0 ? 'default' : `m${index}${name ? `_${slug(name)}` : ''}`);

export function materialUsesTexture(m: MaterialDesc): boolean {
  return (m.type === 'openpbr' || m.type === 'pbr') && !!(m.baseColorTexture || m.metallicRoughnessTexture || m.normalTexture || m.emissiveTexture);
}

/** An `openpbr` material with `specularRoughnessAnisotropy > 0` (needs the mesh tangent frame, P2.26). */
export function isAnisotropic(m: MaterialDesc): boolean {
  return m.type === 'openpbr' && (m.specularRoughnessAnisotropy ?? 0) > 0;
}

export function buildGltfSceneDesc(b: GltfSceneBuild): SceneDesc {
  const matches = (key: string | number, m: GltfDescMaterial): boolean => String(key) === String(m.index) || (m.name !== undefined && key === m.name);
  const dropped = (m: GltfDescMaterial): boolean => !!b.dropMaterials?.some((k) => matches(k, m));
  for (const k of [...(b.dropMaterials ?? []), ...Object.keys(b.replace ?? {})]) {
    if (!b.materials.some((m) => matches(k, m))) throw new Error(`scene '${b.name}': override names '${k}', which is no material of ${b.model.path}`);
  }
  const anyAniso = b.materials.some((m) => {
    if (dropped(m)) return false;
    const replacement = Object.entries(b.replace ?? {}).find(([k]) => matches(k, m));
    return isAnisotropic(replacement ? replacement[1] : m.material);
  });
  const materials: Record<string, MaterialDesc> = {};
  const meshes: MeshDesc[] = [];
  const mesh = (index: number, name: string | undefined, material: MaterialDesc): void => {
    const key = gltfMaterialKey(index, name);
    materials[key] = material;
    const uvs = materialUsesTexture(material) || isAnisotropic(material);
    meshes.push({
      name: key,
      // Anisotropic scenes (P2.26): every mesh carries TANGENT (one file load for the scene) and the anisotropic ones their uvs (the references' tangent is dp/du).
      geometry: { type: 'file', path: b.model.path, format: b.model.format, sha256: b.model.sha256, select: { material: index }, attributes: { ...(b.normals ? { normals: b.normals } : {}), ...(uvs ? { uvs: true } : {}), ...(anyAniso ? { tangents: true } : {}) } },
      material: key,
      ...(materialUsesTexture(material) ? { uv: {} } : {}),
      ...(b.scale !== undefined && b.scale !== 1 ? { transform: scaling(b.scale) } : {}),
    });
  };
  for (const m of b.materials) {
    if (dropped(m)) continue;
    const replacement = Object.entries(b.replace ?? {}).find(([k]) => matches(k, m));
    mesh(m.index, m.name, replacement ? replacement[1] : m.material);
  }
  if (b.defaultMaterial) mesh(-1, undefined, b.defaultMaterial);

  const used = new Set<string>();
  for (const m of Object.values(materials)) if (m.type === 'openpbr' || m.type === 'pbr') for (const k of [m.baseColorTexture, m.metallicRoughnessTexture, m.normalTexture, m.emissiveTexture]) if (k) used.add(k);
  const textures: Record<string, TextureDesc> = {};
  for (const k of [...used].sort()) {
    const t = b.textures?.[k];
    if (!t) throw new Error(`scene '${b.name}': a material uses texture '${k}' which is not in the texture table`);
    textures[k] = t;
  }
  return {
    version: 1, name: b.name, purpose: b.purpose, tier: b.tier ?? 1, camera: b.camera, film: b.film, integrator: b.integrator, materials,
    ...(Object.keys(textures).length ? { textures } : {}), meshes, ...(b.lights ? { lights: b.lights } : {}), ...(b.environment ? { environment: b.environment } : {}),
  };
}
