/**
 * Adapter registry for the competitor page. Each adapter is imported lazily so a page for one renderer never loads the
 * others' libraries (they each pull their own copy of three/WebGPU code).
 */
import type { FeatureSupport, RendererAdapter } from '../src/adapter.ts';
import type { SceneDesc } from '../scenes/schema.ts';

export const ADAPTER_IDS = ['tgpt-wavefront', 'tgpt-megakernel', 'mutracer'] as const;
export type AdapterId = (typeof ADAPTER_IDS)[number];
/** Our own renderers: valid for `--renderers`, but not in the default competitor set. */
export const OWN_ADAPTER_IDS = ['pt-megakernel', 'pt-wavefront', 'pt-wf-refill', 'pt-wf-batch', 'pt-wf-refill-l2', 'pt-wf-refill-l4', 'pt-wf-batch-l2', 'pt-mk-shadow', 'pt-mk-indirect', 'pt-mk-secondary', 'pt-wf-shadow', 'pt-wf-indirect', 'pt-wf-secondary', 'pt-mk-vbuffer', 'pt-wf-vbuffer', 'pt-mk-packed', 'pt-wf-packed', 'pt-mk-lay-stoch', 'pt-mk-lay-int', 'pt-mk-lay-lut', 'pt-mk-lay-tex', 'pt-wf-lay-int', 'pt-wf-lay-lut', 'pt-wf-lay-tex', 'pt-mk-lod', 'pt-wf-lod', 'pt-wf-sg', 'pt-wf-sort-mat', 'pt-wf-sort-oct', 'pt-wf-sort-both', 'pt-wf-refill-sg', 'pt-wf-ordered', 'pt-wf-ordered-sg', 'pt-wf-ordered-l2', 'pt-wf-ordered-l4', 'pt-wf-refill-lod', 'pt-wf-ordered-lod', 'pt-auto'] as const;
/** Wavefront scheduling presets (P3.2): the same adapter with `refill` / `layers` set, so `matrix.ts --interleave` can A/B them in one run. `--extra pt_refill=...` still overrides. */
const WAVEFRONT_PRESETS = {
  'pt-wf-refill': { refill: 'inline' },
  'pt-wf-batch': { refill: 'batch' },
  'pt-wf-refill-l2': { refill: 'inline', layers: 2 },
  'pt-wf-refill-l4': { refill: 'inline', layers: 4 },
  'pt-wf-batch-l2': { refill: 'batch', layers: 2 },
  // P3.3 / P3.4 / P3.10 / R1b (D11 A/B switches): subgroup-aggregated appends, block sorts, ordered restart, material LOD with refill
  'pt-wf-sg': { appends: 'subgroup' },
  'pt-wf-sort-mat': { sort: 'material' },
  'pt-wf-sort-oct': { sort: 'octant' },
  'pt-wf-sort-both': { sort: 'both' },
  'pt-wf-refill-sg': { refill: 'inline', appends: 'subgroup' },
  'pt-wf-ordered': { refill: 'ordered' },
  'pt-wf-ordered-sg': { refill: 'ordered', appends: 'subgroup' },
  'pt-wf-ordered-l2': { refill: 'ordered', layers: 2 },
  'pt-wf-ordered-l4': { refill: 'ordered', layers: 4 },
  'pt-wf-refill-lod': { refill: 'inline', materialLod: 'secondary' },
  'pt-wf-ordered-lod': { refill: 'ordered', materialLod: 'secondary' },
  // P3.9: scheduler + settings from `ptwfAutoPolicy` (film, maxDepth, subgroups); may resolve to the megakernel
  'pt-auto': { auto: true },
} as const;
/** R1 presets (`integrator.materialLod: 'secondary'`, depth 1): the megakernel / wavefront with the reduced secondary material, so `--renderers pt-megakernel,pt-mk-lod --interleave` pairs them. */
const LOD_PRESETS = { 'pt-mk-lod': { materialLod: 'secondary' }, 'pt-wf-lod': { materialLod: 'secondary', scheduler: 'wavefront' } } as const;

/** Alpha-policy presets (R2): the same adapter with `alphaPolicy` set, so `matrix.ts --interleave` can A/B the policies in one run. BIASED: never a reference. */
const ALPHA_PRESETS = {
  'pt-mk-shadow': { alphaPolicy: 'opaqueShadow' },
  'pt-mk-indirect': { alphaPolicy: 'opaqueIndirect' },
  'pt-mk-secondary': { alphaPolicy: 'opaqueSecondary' },
  'pt-wf-shadow': { alphaPolicy: 'opaqueShadow', scheduler: 'wavefront' },
  'pt-wf-indirect': { alphaPolicy: 'opaqueIndirect', scheduler: 'wavefront' },
  'pt-wf-secondary': { alphaPolicy: 'opaqueSecondary', scheduler: 'wavefront' },
} as const;

/** V-buffer primaries (P3.6): the same adapter with `primary: 'vbuffer'` (a bench rasteriser writes the V-buffer each sample), so `matrix.ts --interleave` can A/B it against traced primaries. */
const VBUFFER_PRESETS = {
  'pt-mk-vbuffer': { primary: 'vbuffer' },
  'pt-wf-vbuffer': { primary: 'vbuffer', scheduler: 'wavefront' },
} as const;

/** Scene copy packing (P3.11): the same adapter with the scene copy bound as ONE storage buffer, so `matrix.ts --interleave` can A/B it against `pt-megakernel` / `pt-wavefront` (split). */
const PACKED_PRESETS = {
  'pt-mk-packed': { scenePacking: 'packed' },
  'pt-wf-packed': { scenePacking: 'packed', scheduler: 'wavefront' },
} as const;

/** Layered-BSDF presets (P2.32b): the stochastic walk and the albedo layering (internal base) without / with the fitted correction, so `matrix.ts --interleave` can A/B them in one run. */
const LAYERING_PRESETS = {
  'pt-mk-lay-stoch': { layering: 'stochastic' },
  'pt-mk-lay-int': { layering: 'albedo', albedoBase: 'internal' },
  'pt-mk-lay-lut': { layering: 'albedo', albedoBase: 'internal', albedoCorrection: 'lut' },
  'pt-mk-lay-tex': { layering: 'albedo', albedoBase: 'internal', albedoCorrection: 'texture' },
  'pt-wf-lay-int': { layering: 'albedo', albedoBase: 'internal', scheduler: 'wavefront' },
  'pt-wf-lay-lut': { layering: 'albedo', albedoBase: 'internal', albedoCorrection: 'lut', scheduler: 'wavefront' },
  'pt-wf-lay-tex': { layering: 'albedo', albedoBase: 'internal', albedoCorrection: 'texture', scheduler: 'wavefront' },
} as const;

export interface AdapterOptions {
  /** three-gpu-pathtracer frame budget (path steps per renderSample call). */
  frameBudget?: number;
  /** three-gpu-pathtracer: how emissive quads are lit, `rect` (default) or `mesh`. */
  emitters?: 'rect' | 'mesh';
  /** maxBounces override for the renderer (calibration). */
  bounces?: number;
  /** mutracer experiment switches (page args `mt_candidates`, `mt_sampler`, `mt_bevel`, `mt_dispersion`, `mt_offset`). */
  mutracer?: import('./mutracer/index.ts').MutracerSettings;
  /** pt-megakernel settings (page args `pt_*`). */
  pt?: import('./pt-megakernel/index.ts').PtAdapterSettings;
}

export async function createAdapter(id: string, options: AdapterOptions = {}): Promise<RendererAdapter> {
  switch (id) {
    case 'tgpt-wavefront':
      return (await import('./tgpt-wavefront/index.ts')).createTgptWavefront(options);
    case 'tgpt-megakernel':
      return (await import('./tgpt-megakernel/index.ts')).createTgptMegakernel(options);
    case 'mutracer':
      return (await import('./mutracer/index.ts')).createMutracer(options.bounces, options.mutracer);
    case 'pt-megakernel':
      return (await import('./pt-megakernel/index.ts')).createPtMegakernel(options.pt);
    case 'pt-wavefront':
      return (await import('./pt-megakernel/index.ts')).createPtMegakernel({ ...options.pt, scheduler: 'wavefront' });
    case 'pt-wf-refill':
    case 'pt-wf-batch':
    case 'pt-wf-refill-l2':
    case 'pt-wf-refill-l4':
    case 'pt-wf-batch-l2':
    case 'pt-wf-sg':
    case 'pt-wf-sort-mat':
    case 'pt-wf-sort-oct':
    case 'pt-wf-sort-both':
    case 'pt-wf-refill-sg':
    case 'pt-wf-ordered':
    case 'pt-wf-ordered-sg':
    case 'pt-wf-ordered-l2':
    case 'pt-wf-ordered-l4':
    case 'pt-wf-refill-lod':
    case 'pt-wf-ordered-lod':
    case 'pt-auto':
      return (await import('./pt-megakernel/index.ts')).createPtMegakernel({ ...WAVEFRONT_PRESETS[id], ...options.pt, scheduler: 'wavefront' }, id);
    case 'pt-mk-shadow':
    case 'pt-mk-indirect':
    case 'pt-mk-secondary':
    case 'pt-wf-shadow':
    case 'pt-wf-indirect':
    case 'pt-wf-secondary':
      return (await import('./pt-megakernel/index.ts')).createPtMegakernel({ ...ALPHA_PRESETS[id], ...options.pt }, id);
    case 'pt-mk-packed':
    case 'pt-wf-packed':
      return (await import('./pt-megakernel/index.ts')).createPtMegakernel({ ...PACKED_PRESETS[id], ...options.pt }, id);
    case 'pt-mk-lay-stoch':
    case 'pt-mk-lay-int':
    case 'pt-mk-lay-lut':
    case 'pt-mk-lay-tex':
    case 'pt-wf-lay-int':
    case 'pt-wf-lay-lut':
    case 'pt-wf-lay-tex':
      return (await import('./pt-megakernel/index.ts')).createPtMegakernel({ ...LAYERING_PRESETS[id], ...options.pt }, id);
    case 'pt-mk-vbuffer':
    case 'pt-wf-vbuffer':
      return (await import('./pt-megakernel/index.ts')).createPtMegakernel({ ...VBUFFER_PRESETS[id], ...options.pt }, id);
    case 'pt-mk-lod':
    case 'pt-wf-lod':
      return (await import('./pt-megakernel/index.ts')).createPtMegakernel({ ...LOD_PRESETS[id], ...options.pt }, id);
    default:
      throw new Error(`unknown renderer "${id}" (have: ${[...ADAPTER_IDS, ...OWN_ADAPTER_IDS].join(', ')})`);
  }
}

/** Why this renderer cannot render the scene exactly, or null when it can. Unsupported scenes are skipped, not faked. */
export function unsupportedReason(features: FeatureSupport, scene: SceneDesc): string | null {
  const reasons: string[] = [];
  for (const [name, material] of Object.entries(scene.materials)) {
    if (!features.materials.includes(material.type)) reasons.push(`material "${name}" (${material.type}) is not exact in this renderer`);
    else {
      const why = features.materialUnsupported?.(material);
      if (why) reasons.push(`material "${name}": ${why}`);
    }
  }
  if (Object.keys(scene.textures ?? {}).length > 0 && !features.textures) reasons.push('image textures unsupported');
  if (scene.meshes.some((mesh) => mesh.emission) && !features.emissiveMeshes) reasons.push('emissive meshes unsupported');
  if ((scene.lights?.length ?? 0) > 0 && !features.directionalLight) reasons.push('directional lights unsupported');
  if (features.fileGeometry === false && scene.meshes.some((mesh) => mesh.geometry.type === 'file')) reasons.push('file geometry (external mesh) is not loaded by this renderer yet');
  const env = scene.environment;
  if (env && !features.environment.includes(env.type)) reasons.push(`${env.type} environment unsupported`);
  return reasons.length > 0 ? reasons.join('; ') : null;
}
