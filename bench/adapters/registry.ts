/**
 * Adapter registry for the competitor page. Each adapter is imported lazily so a page for one renderer never loads the
 * others' libraries (they each pull their own copy of three/WebGPU code).
 */
import type { FeatureSupport, RendererAdapter } from '../src/adapter.ts';
import type { SceneDesc } from '../scenes/schema.ts';

export const ADAPTER_IDS = ['tgpt-wavefront', 'tgpt-megakernel', 'mutracer'] as const;
export type AdapterId = (typeof ADAPTER_IDS)[number];

export interface AdapterOptions {
  /** three-gpu-pathtracer frame budget (path steps per renderSample call). */
  frameBudget?: number;
  /** three-gpu-pathtracer: how emissive quads are lit, `rect` (default) or `mesh`. */
  emitters?: 'rect' | 'mesh';
  /** maxBounces override for the renderer (calibration). */
  bounces?: number;
  /** mutracer experiment switches (page args `mt_candidates`, `mt_sampler`, `mt_bevel`, `mt_dispersion`, `mt_offset`). */
  mutracer?: import('./mutracer/index.ts').MutracerSettings;
}

export async function createAdapter(id: string, options: AdapterOptions = {}): Promise<RendererAdapter> {
  switch (id) {
    case 'tgpt-wavefront':
      return (await import('./tgpt-wavefront/index.ts')).createTgptWavefront(options);
    case 'tgpt-megakernel':
      return (await import('./tgpt-megakernel/index.ts')).createTgptMegakernel(options);
    case 'mutracer':
      return (await import('./mutracer/index.ts')).createMutracer(options.bounces, options.mutracer);
    default:
      throw new Error(`unknown renderer "${id}" (have: ${ADAPTER_IDS.join(', ')})`);
  }
}

/** Why this renderer cannot render the scene exactly, or null when it can. Unsupported scenes are skipped, not faked. */
export function unsupportedReason(features: FeatureSupport, scene: SceneDesc): string | null {
  const reasons: string[] = [];
  for (const [name, material] of Object.entries(scene.materials)) {
    if (!features.materials.includes(material.type)) reasons.push(`material "${name}" (${material.type}) is not exact in this renderer`);
  }
  if (scene.meshes.some((mesh) => mesh.emission) && !features.emissiveMeshes) reasons.push('emissive meshes unsupported');
  if ((scene.lights?.length ?? 0) > 0 && !features.directionalLight) reasons.push('directional lights unsupported');
  const env = scene.environment;
  if (env && !features.environment.includes(env.type)) reasons.push(`${env.type} environment unsupported`);
  return reasons.length > 0 ? reasons.join('; ') : null;
}
