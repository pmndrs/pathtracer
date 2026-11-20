/**
 * Bench `SceneDesc` -> the inputs of the trace core's scene copy (D4), so every bench scene can be traced, shaded and lit
 * by `@pmndrs/raytracer` and compared with pbrt-v4 / Mitsuba.
 *
 * Mapping (OpenPBR subset, see packages/raytracer/src/scene/material.ts):
 *  - `diffuse`: baseColor = reflectance, `specularWeight` 0 (a pure Lambertian: no dielectric specular lobe);
 *  - `conductor`: physical conductor (`conductorEtaK`, eta + k), metalness 1, roughness as given;
 *  - `dielectric`: transmissionWeight 1, specularWeight 1, ior, roughness, no base layer (baseWeight 0) and no absorption
 *    (`transmissionDepth` 0);
 *  - `black`: baseWeight 0 and specularWeight 0: scatters nothing.
 *  - `pbr`: glTF metallic-roughness factors (baseColor, perceptual roughness, metalness, emissive) plus texture slots; a `diffuse` with a
 *    `reflectanceTexture` takes the slot as its base colour texture.
 * `SceneDesc.textures` become the scene copy's textures (procedural, `textures.ts`, slot = key order); `MeshDesc.uv` meshes carry uvs.
 * A mesh's `emission` is not a material property in the bench schema, so each distinct (material, emission, twoSided) triple
 * becomes its own scene-copy material; `materialNames` says which is which.
 *
 * One mesh per `MeshDesc`, one instance per mesh (object-space triangles + the mesh's transform), instance material = the
 * triple's index, instance mask 0x02 for emitters and 0x01 otherwise (the trace-scenes convention). `file` geometry must be
 * resolved by the caller (`resolveFile`); the bench's async glb loaders live elsewhere.
 */
import type { SceneCopyInput, SceneCopyInstance, SceneCopyMesh, RtMaterialInput, SceneCopyOptions } from '@pmndrs/raytracer';
import { buildSceneCopy, envMapMean, openPbrToRtMaterial, rotationTurns, type SceneCopy } from '@pmndrs/raytracer';
import { compose, tessellate, type TriangleMesh } from './geometry.ts';
import type { EnvironmentDesc, GeometryDesc, MaterialDesc, MeshDesc, Rgb, SceneDesc } from './schema.ts';
import { generateTexture } from './textures.ts';

/** Instance masks (same values as bench/pages/trace-scenes.ts): rays with `MASK_NO_EMITTERS` skip emitters. */
export const INSTANCE_MASK_EMITTER = 0x02;
export const INSTANCE_MASK_PLAIN = 0x01;

export interface DescToSceneCopyOptions {
  /** Loads `file` geometry (glb/gltf/ply/obj) as object-space triangles. Without it, a `file` mesh throws. */
  resolveFile?: (geometry: Extract<GeometryDesc, { type: 'file' }>, mesh: MeshDesc) => TriangleMesh;
  /**
   * The decoded `hdri` image (row 0 = +Y, see schema.ts): copied into the scene copy as the environment map, and its solid-angle
   * mean fills the constant-radiance entry. Without it an `hdri` environment is constant `hdriRadiance` (default black).
   */
  hdriMap?: { width: number; height: number; channels: number; data: ArrayLike<number> };
  /** Constant radiance of an `hdri` environment when there is no `hdriMap` (or to override its mean). Default: black. */
  hdriRadiance?: Rgb;
  /** Passed to `buildSceneCopy`. */
  copy?: SceneCopyOptions;
}

export interface DescSceneCopyInput extends SceneCopyInput {
  meshes: SceneCopyMesh[];
  instances: SceneCopyInstance[];
  materials: RtMaterialInput[];
  /** One name per material: `<bench material>` or `<bench material>+emission(r,g,b)`. */
  materialNames: string[];
  /** `MeshDesc.name` per mesh, in order. */
  meshNames: string[];
  /** P5: per material, the media on its two sides (`SceneDesc.media` order + 1, 0 = none) and whether it is a pure medium boundary (`interface`). Empty when the scene has no media. */
  materialVolumes: Array<{ interior: number; exterior: number; boundary: boolean }>;
}

/** glTF `anisotropyRotation` (radians, any sign) as the scene copy's unorm field: turns in [0, 1). */
export { rotationTurns };

/** The scene-copy material of one bench material (no emission). Exported so tests can check the mapping. */
export function materialFromDesc(desc: MaterialDesc, texture: (name: string) => number = (name) => {
  throw new RangeError(`texture '${name}' is not defined`);
}): RtMaterialInput {
  switch (desc.type) {
    case 'diffuse':
      return { baseColor: [...desc.reflectance], baseWeight: 1, metalness: 0, specularWeight: 0, specularRoughness: 1, ...(desc.reflectanceTexture ? { baseColorTexture: texture(desc.reflectanceTexture) } : {}) };
    case 'pbr':
      return {
        baseColor: [...desc.baseColor],
        baseWeight: 1,
        metalness: desc.metalness,
        specularRoughness: desc.roughness,
        specularWeight: 1,
        emissionColor: [...(desc.emissive ?? [0, 0, 0])],
        ...(desc.baseColorTexture ? { baseColorTexture: texture(desc.baseColorTexture) } : {}),
        ...(desc.metallicRoughnessTexture ? { roughnessMetalTexture: texture(desc.metallicRoughnessTexture) } : {}),
        ...(desc.normalTexture ? { normalTexture: texture(desc.normalTexture) } : {}),
        ...(desc.emissiveTexture ? { emissionTexture: texture(desc.emissiveTexture) } : {}),
      };
    case 'openpbr':
      // The mapping is the public loader's (`openPbrToRtMaterial`, @pmndrs/raytracer gltf/material.ts).
      return openPbrToRtMaterial(desc, texture);
    case 'conductor':
      return { conductorEtaK: true, conductorEta: [...desc.eta], conductorK: [...desc.k], metalness: 1, baseWeight: 1, specularRoughness: desc.roughness, specularWeight: 1 };
    case 'dielectric':
      return { baseWeight: 0, baseColor: [0, 0, 0], transmissionWeight: 1, transmissionColor: [1, 1, 1], transmissionDepth: 0, specularWeight: 1, ior: desc.ior, specularRoughness: desc.roughness };
    case 'black':
    case 'interface':
      return { baseColor: [0, 0, 0], baseWeight: 0, specularWeight: 0 };
  }
}

/** Does the material read any texture (so its meshes need uvs)? */
export function materialUsesTexture(desc: MaterialDesc): boolean {
  if (desc.type === 'diffuse') return !!desc.reflectanceTexture;
  if (desc.type === 'pbr' || desc.type === 'openpbr') return !!((desc.type === 'openpbr' && desc.materialx) || desc.baseColorTexture || desc.metallicRoughnessTexture || desc.normalTexture || desc.emissiveTexture);
  return false;
}

function environmentFromDesc(env: EnvironmentDesc | undefined, options: DescToSceneCopyOptions): SceneCopyInput['environment'] {
  if (!env) return null;
  if (env.type === 'uniform') return { radiance: [...env.radiance] };
  const rotationY = (env.rotationYDeg * Math.PI) / 180;
  if (options.hdriMap) {
    const mean = envMapMean(options.hdriMap);
    return { radiance: [...(options.hdriRadiance ?? mean)] as Rgb, scale: env.scale, rotationY, map: options.hdriMap };
  }
  return { radiance: [...(options.hdriRadiance ?? [0, 0, 0])], scale: env.scale, rotationY };
}

export function sceneDescToCopyInput(scene: SceneDesc, options: DescToSceneCopyOptions = {}): DescSceneCopyInput {
  const meshes: SceneCopyMesh[] = [];
  const instances: SceneCopyInstance[] = [];
  const materials: RtMaterialInput[] = [];
  const materialNames: string[] = [];
  const meshNames: string[] = [];
  const materialVolumes: Array<{ interior: number; exterior: number; boundary: boolean }> = [];
  const mediumNames = Object.keys(scene.media ?? {});
  const mediumId = (name: string | undefined): number => {
    if (name === undefined) return 0;
    const i = mediumNames.indexOf(name);
    if (i < 0) throw new RangeError(`scene '${scene.name}': unknown medium '${name}'`);
    return i + 1;
  };
  const known = new Map<string, number>();
  const textureNames = Object.keys(scene.textures ?? {});
  const textureSlot = (name: string): number => {
    const slot = textureNames.indexOf(name);
    if (slot < 0) throw new RangeError(`scene '${scene.name}': unknown texture '${name}'`);
    return slot;
  };

  const materialIndex = (mesh: MeshDesc): number => {
    const base = scene.materials[mesh.material];
    if (!base) throw new RangeError(`scene '${scene.name}': mesh '${mesh.name}' uses unknown material '${mesh.material}'`);
    if (materialUsesTexture(base) && !mesh.uv) throw new RangeError(`scene '${scene.name}': mesh '${mesh.name}' uses textured material '${mesh.material}' but has no \`uv\``);
    const emission = mesh.emission;
    const interior = mediumId(mesh.medium?.interior), exterior = mediumId(mesh.medium?.exterior);
    const boundary = base.type === 'interface';
    if (boundary && !mesh.medium) throw new RangeError(`scene '${scene.name}': mesh '${mesh.name}' has an interface material but no \`medium\``);
    const key = (emission ? `${mesh.material}+emission(${emission.join(',')})${mesh.twoSided ? '+twoSided' : ''}` : mesh.material) + (mesh.medium ? `+medium(${interior}|${exterior})` : '');
    let index = known.get(key);
    if (index === undefined) {
      index = materials.length;
      known.set(key, index);
      materials.push(emission ? { ...materialFromDesc(base, textureSlot), emissionColor: [...emission], emissionTwoSided: !!mesh.twoSided } : materialFromDesc(base, textureSlot));
      materialNames.push(key);
      materialVolumes.push({ interior, exterior, boundary });
    }
    return index;
  };

  scene.meshes.forEach((desc, i) => {
    const geometry = desc.geometry;
    const local =
      geometry.type === 'file'
        ? options.resolveFile
          ? options.resolveFile(geometry, desc)
          : (() => {
              throw new Error(`scene '${scene.name}': mesh '${desc.name}' is file geometry '${geometry.path}'; pass resolveFile`);
            })()
        : tessellate(geometry, desc.uv ? { uv: desc.uv } : {});
    meshes.push({ positions: local.positions, indices: local.indices, normals: local.normals, ...(local.uvs ? { uvs: local.uvs } : {}), ...(local.tangents ? { tangents: local.tangents } : {}) });
    meshNames.push(desc.name);
    instances.push({
      meshIndex: i,
      instanceIndex: i,
      objectToWorld: desc.transform ? [...desc.transform] : compose(),
      material: materialIndex(desc),
      mask: desc.emission ? INSTANCE_MASK_EMITTER : INSTANCE_MASK_PLAIN,
    });
  });

  const directional = (scene.lights ?? []).filter((light) => light.type === 'directional');
  if (directional.length > 1) throw new RangeError(`scene '${scene.name}': ${directional.length} directional lights; the scene copy holds one sun`);
  const sun = directional[0] ? { direction: [...directional[0].direction] as [number, number, number], irradiance: [...directional[0].irradiance] as [number, number, number], angularRadius: directional[0].angularRadius ?? 0 } : null;

  const textures = textureNames.map((name) => ({ ...generateTexture((scene.textures as NonNullable<SceneDesc['textures']>)[name] as NonNullable<SceneDesc['textures']>[string]), label: name }));
  return { meshes, instances, materials, materialNames, meshNames, materialVolumes: mediumNames.length > 0 ? materialVolumes : [], ...(textures.length > 0 ? { textures } : {}), sun, environment: environmentFromDesc(scene.environment, options), options: options.copy };
}

/** `sceneDescToCopyInput` + `buildSceneCopy`. */
export function buildSceneCopyFromDesc(scene: SceneDesc, options: DescToSceneCopyOptions = {}): { copy: SceneCopy; input: DescSceneCopyInput } {
  const input = sceneDescToCopyInput(scene, options);
  return { copy: buildSceneCopy(input), input };
}
