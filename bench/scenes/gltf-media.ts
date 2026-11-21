/**
 * glTF volume absorption as a real medium (V6.1). `KHR_materials_volume` (attenuationColor + attenuationDistance, thickness > 0) reaches the scene as an OpenPBR
 * material with `transmissionColor` / `transmissionDepth` (a Beer absorber inside the dielectric, P2.22). `gltfVolumesAsMedia` expresses the same physics for
 * `integrator.volumes`: the material keeps its dielectric part (white transmission, depth 0), and every mesh that uses it becomes a medium boundary whose
 * interior is a homogeneous absorber with sigma_a = -ln(attenuationColor) / attenuationDistance per channel, sigma_s = 0.
 *
 * Skipped (left as they are): thin-walled materials (glTF thickness 0 = no volume), transmission 0, no finite depth, white colour (no absorption).
 * The meshes must be closed, outward-wound solids (a medium boundary); the caller checks that.
 */
import type { MaterialDesc, MediumDesc, Rgb, SceneDesc } from './schema.ts';

/** Smallest colour channel used for the log (a 0 channel would give an infinite coefficient). */
export const MIN_ATTENUATION = 1e-6;

/** sigma_a (per metre) of a glTF attenuation: `-ln(colour) / distance`, channels clamped to [MIN_ATTENUATION, 1]. `undefined` when there is no absorption. */
export function attenuationToSigmaA(color: Rgb, distance: number): Rgb | undefined {
  if (!(distance > 0) || !Number.isFinite(distance)) return undefined;
  const s = color.map((c) => Math.max(0, -Math.log(Math.min(1, Math.max(MIN_ATTENUATION, c))) / distance)) as unknown as Rgb;
  return s.every((v) => v === 0) ? undefined : s;
}

/** The sigma_a of a scene material's Beer absorber, or `undefined` when the material has none this conversion handles. */
export function absorberOf(m: MaterialDesc): Rgb | undefined {
  if (m.type !== 'openpbr') return undefined;
  if (!((m.transmissionWeight ?? 0) > 0) || m.geometryThinWalled) return undefined;
  const depth = m.transmissionDepth ?? 0;
  if (!(depth > 0)) return undefined;
  return attenuationToSigmaA((m.transmissionColor ?? [1, 1, 1]) as Rgb, depth);
}

/** The scene with every glTF Beer absorber turned into a homogeneous medium inside the meshes that use it. Returns the scene unchanged when there is none. */
export function gltfVolumesAsMedia(scene: SceneDesc, name = `${scene.name}-media`): SceneDesc {
  const media: Record<string, MediumDesc> = { ...(scene.media ?? {}) };
  const materials: Record<string, MaterialDesc> = { ...scene.materials };
  const mediumOf = new Map<string, string>(); // material name -> medium name
  for (const [key, m] of Object.entries(scene.materials)) {
    const sigmaA = absorberOf(m);
    if (!sigmaA) continue;
    const mediumName = `absorb-${key}`;
    media[mediumName] = { type: 'homogeneous', sigmaA, sigmaS: [0, 0, 0] };
    mediumOf.set(key, mediumName);
    const { transmissionDepth: _depth, ...rest } = m as MaterialDesc & { transmissionDepth?: number };
    materials[key] = { ...rest, transmissionColor: [1, 1, 1] } as MaterialDesc;
  }
  if (mediumOf.size === 0) return scene;
  const meshes = scene.meshes.map((mesh) => {
    const med = typeof mesh.material === 'string' ? mediumOf.get(mesh.material) : undefined;
    return med && !mesh.medium ? { ...mesh, medium: { interior: med } } : mesh;
  });
  return { ...scene, name, purpose: `${scene.purpose} Variant: the Beer absorbers are homogeneous absorbing media inside the closed meshes (V6.1).`, materials, media, meshes };
}
