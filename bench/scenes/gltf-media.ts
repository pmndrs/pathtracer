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

export interface VolumeScatter { strength: number; multiscatterColor: Rgb; anisotropy: number }

/**
 * Single-scatter albedo from the multi-scatter albedo and HG g, the inversion KHR_materials_scatter specifies (draft, glTF PR 2579, after Kulla and Conty 2017) and
 * three-gpu-pathtracer's `getSingleScatterAlbedo` (src/webgpu/nodes/medium.wgsl.js) implements: s = clamp(4.09712 + 4.20863 a - sqrt(9.59217 + 41.6808 a + 17.7126 a^2), 0, 1),
 * rho_ss = (1 - s^2) / (1 - g s^2); a channel with a = 0 gives s = 1 and rho_ss = 0. `multiscatter` is already multiplied by scatterStrength.
 */
export function singleScatterAlbedo(multiscatter: Rgb, g: number): Rgb {
  return multiscatter.map((a) => {
    if (!(a > 0)) return 0;
    const s = Math.min(1, Math.max(0, 4.09712 + 4.20863 * a - Math.sqrt(9.59217 + 41.6808 * a + 17.7126 * a * a)));
    return (1 - s * s) / (1 - g * s * s);
  }) as unknown as Rgb;
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
    // sigmaA from absorberOf is the attenuation sigma_t = -ln(colour) / distance. KHR_materials_scatter splits it: sigma_s = sigma_t rho_ss, sigma_a = sigma_t (1 - rho_ss).
    const vs = (m as { volumeScatter?: VolumeScatter }).volumeScatter;
    const rho = vs ? singleScatterAlbedo(vs.multiscatterColor.map((c) => c * vs.strength) as unknown as Rgb, vs.anisotropy) : ([0, 0, 0] as Rgb);
    const sigmaS = sigmaA.map((t, i) => t * rho[i]!) as unknown as Rgb;
    media[mediumName] = { type: 'homogeneous', sigmaA: sigmaA.map((t, i) => t - sigmaS[i]!) as unknown as Rgb, sigmaS, ...(vs && vs.anisotropy !== 0 ? { g: vs.anisotropy } : {}) };
    mediumOf.set(key, mediumName);
    const { transmissionDepth: _depth, volumeScatter: _vs, ...rest } = m as MaterialDesc & { transmissionDepth?: number; volumeScatter?: VolumeScatter };
    materials[key] = { ...rest, transmissionColor: [1, 1, 1] } as MaterialDesc;
  }
  if (mediumOf.size === 0) return scene;
  const meshes = scene.meshes.map((mesh) => {
    const med = typeof mesh.material === 'string' ? mediumOf.get(mesh.material) : undefined;
    return med && !mesh.medium ? { ...mesh, medium: { interior: med } } : mesh;
  });
  return { ...scene, name, purpose: `${scene.purpose} Variant: the Beer absorbers are homogeneous absorbing media inside the closed meshes (V6.1).`, materials, media, meshes };
}
