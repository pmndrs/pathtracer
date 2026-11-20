/**
 * Mode presets (P4.1, spec section 7): `mode: 'reference' | 'preview' | 'realtime'` resolves to integrator settings. Explicit options win over a preset:
 * a key the caller set (not `undefined`) is never replaced, so `{ mode: 'realtime', integrator: { materialLod: 'full' } }` keeps full materials.
 *
 * Every preset entry carries its status in `PT_MODE_NOTES`: a measured cost or benefit (with the TASKS / DIVERGES row that holds the numbers) or "pending A/B". A speed setting
 * that is BIASED says so; the `reference` preset is today's defaults: exact everything, Owen-Sobol, accumulation, the scene's own depth.
 *
 *  - `reference`: nothing changes (documented explicitly so a page can print what a mode means).
 *  - `preview`: progressive but cheap: D31 cutouts (exact indirect, `transmit` shadows; `alphaPolicy: 'opaqueShadow'` is the max-speed opt-in), reduced secondary materials, cone texture LOD, AOVs for OIDN (P4.7 builds the pipeline), a shallow depth.
 *  - `realtime`: 1 spp frames for the temporal filter (P4.5): everything of `preview` plus per-frame outputs (`frame`), the blue-noise sampler when the texture is given, 2 bounces.
 *
 * The preset also names what the CALLER must do outside the kernel (`scenePacking` must be the same for `uploadSceneCopy` and the kernel, so `create` never changes it:
 * read `ptModePreset(mode).scenePacking` for both).
 */
import type { PtLayeredOptions } from '../bsdf/layered.ts';
import type { SceneCopyPacking } from '@pmndrs/raytracer';
import type { PtIntegratorOptions } from './integrator-wgsl.ts';

export type PtMode = 'reference' | 'preview' | 'realtime';
export const PT_MODES: readonly PtMode[] = ['reference', 'preview', 'realtime'];

export interface PtModePreset {
  mode: PtMode;
  /** Integrator settings the mode sets (keys the caller left `undefined`). */
  integrator: PtIntegratorOptions;
  /** Layering of OpenPBR layered materials (merged into `integrator.materials.layered` where the caller left a field undefined). */
  layered: Pick<PtLayeredOptions, 'layering' | 'albedoBase' | 'albedoCorrection'> | null;
  /** The path limits the mode suggests (`pt.setLimits`; a scene's own depth stays for `reference`, hence null). */
  limits: { maxDepth: number; rrStartDepth: number; textureConeScale?: number } | null;
  /** The scene copy binding the mode wants. Pass it to `uploadSceneCopy({ packing })` AND to the kernel's `scenePacking`; `create` does not apply it (a mismatch throws in `setScene`). */
  scenePacking: SceneCopyPacking | null;
}

/** Status of each preset entry: what is measured, what is pending. Key = `<mode>.<entry>`. */
export const PT_MODE_NOTES: Readonly<Record<string, string>> = {
  'reference.*': 'today\'s defaults: exact cutouts, full materials, Owen-Sobol, accumulation; the gates run here',
  'preview.alphaPolicy': 'exact + alphaShadow transmit (D31, refines D27; R2b): indirect hits keep the exact cutout loop, cutout shadows cost about the solid walk and read within 0.2% of exact (calibrated beta 0.45); the cutout cards must be in instances with mask `alphaMixedMask` (bench adapter: automatic; library callers set `instance.mask`, else the shadows are the solid walk\'s -12%); provisional: shadow-ray saving kept 70-90% (megakernel foliage-mask 0.81 of exact time vs solid 0.76, wavefront 0.92 vs 0.89), quiet pass to confirm',
  'preview.alphaShadow': 'transmit: see alphaPolicy above (D31); needs the cutout cards in `alphaMixedMask` instances and turns the walk\'s ray masks on (`ptNeedsRayMasks`)',
  'preview.alphaTransmit': '0.45: the calibrated beta of the three foliage scenes (0.446-0.459; foliage-mask-dense 0.323); the mean image is linear in beta',
  'preview.cutoutsMaxSpeed': 'opt in with `integrator: { alphaPolicy: \'opaqueShadow\' }` (the preset then keeps `alphaShadow` at \'policy\'): solid shadow rays, exact indirect; fastest (foliage-mask 0.76 megakernel / 0.89 wavefront of the exact time) but BIASED -5..-17% on card scenes (D27 / D31). `opaqueSecondary` buys nothing over it (V147)',
  'preview.materialLod': 'secondary from depth 1: R1; furnace within about 1%, kernel -6..-19% on the wavefront; timing in the QUIET-PASS queue; pending A/B against full (R9)',
  'preview.textureLod': 'cone, scale 1 (a full-pixel footprint at 1 spp, pbrt-v4 uses max(1/8, 1/sqrt(spp))): pending A/B (R9)',
  'preview.aovs': 'albedo + normal + viewZ for OIDN on the converged-ish frame (P2.34); the OIDN-small pipeline is P4.7',
  'preview.limits': 'maxDepth 4, no Russian roulette: pending A/B (R3)',
  'preview.layered': 'D24 + D32: albedo layering, internal base, albedoCorrection texture (P2.32b: furnace <= 1.003, within 4% of the walk on the Tier 1 class B scenes); these are the library defaults since D32, the preset states them',
  'preview.scenePacking': 'packed: P3.11, one storage buffer less in every kernel, per-pixel identical images; the library default since D32 (provisional: Sponza a wash, Cornell megakernel +2%; the optional quiet pass confirms)',
  'realtime.alphaPolicy': 'as preview (D31: exact + transmit; max speed `alphaPolicy: \'opaqueShadow\'`)',
  'realtime.alphaShadow': 'as preview (D31)',
  'realtime.alphaTransmit': 'as preview (D31)',
  'realtime.materialLod': 'as preview (R1)',
  'realtime.textureLod': 'as preview',
  'realtime.layered': 'as preview (D24 + P2.32b)',
  'realtime.scenePacking': 'as preview',
  'realtime.sampler': 'bluenoise (P4.1, V120): the passed texture, else the one shipped in the package (`loadBlueNoise`, S10). A/B in `realtime-check` (TASKS P4.1); both schedulers since P4.1b (V210)',
  'realtime.frame': 'per-frame outputs for the temporal filter (P4.1, V122); both schedulers since P4.1b (the wavefront: plain scheduler only, planes in the accumulator tail, V211); costs 7 planes of vec4 per pixel of write bandwidth (the 7th is the emission plane, P4.5a, V200); not timed yet (QUIET-PASS)',
  'realtime.restir': 'opt-in (`integrator.restir`), D32: when on, `unbiased` (no initial visibility), M 8, temporal, spatial 2 random are the library defaults, and the realtime preset adds `sharedHit` (megakernel, trace primaries, positions in the layout: -5.8% frame on Sponza 1280x720); pays on many-light scenes only (1.9-2.8x relMSE, -28% FLIP at +13..30% frame time), a sun + sky scene gains nothing',
  'realtime.scheduler': 'D32: `ptwfAutoPolicy({ ..., frame: true, triangles, kernelBytes })` is the recommended scheduler choice: wavefront for big scenes (>= 100k triangles, >= 2^17 pixels) or a megakernel over the ~50 KB DXC class, else megakernel',
  'realtime.limits': 'maxDepth 2, no Russian roulette (a depth cap is the real-time termination; R3 compares RR / throughput / cache termination after the denoiser): pending A/B',
};

const LAYERED = { layering: 'albedo', albedoBase: 'internal', albedoCorrection: 'texture' } as const;

const PRESETS: Record<PtMode, PtModePreset> = {
  reference: { mode: 'reference', integrator: {}, layered: null, limits: null, scenePacking: null },
  preview: {
    mode: 'preview',
    integrator: { alphaPolicy: 'exact', alphaShadow: 'transmit', alphaTransmit: 0.45, materialLod: 'secondary', textureLod: 'cone', aovs: true },
    layered: LAYERED,
    limits: { maxDepth: 4, rrStartDepth: 0, textureConeScale: 1 },
    scenePacking: 'packed',
  },
  realtime: {
    mode: 'realtime',
    integrator: { alphaPolicy: 'exact', alphaShadow: 'transmit', alphaTransmit: 0.45, materialLod: 'secondary', textureLod: 'cone', frame: true },
    layered: LAYERED,
    limits: { maxDepth: 2, rrStartDepth: 0, textureConeScale: 1 },
    scenePacking: 'packed',
  },
};

export function ptModePreset(mode: PtMode): PtModePreset {
  const p = PRESETS[mode];
  if (!p) throw new RangeError(`unknown mode '${String(mode)}' (reference | preview | realtime)`);
  return p;
}

/** The notes of one mode (`reference.*` for the reference mode), in table order. */
export function ptModeNotes(mode: PtMode): Array<[string, string]> {
  return Object.entries(PT_MODE_NOTES).filter(([k]) => k.startsWith(`${mode}.`));
}

export interface PtModeOptions {
  mode?: PtMode;
  integrator?: PtIntegratorOptions;
  /** Blue-noise texture handed to the kernel (default: the package's `gf-vec2-64`, loaded by `create`). */
  blueNoise?: unknown;
}

/**
 * The options with the mode's integrator settings filled in under the caller's own (`undefined` keys only). Both schedulers take the same preset since P4.1b (the wavefront used to drop
 * `bluenoise` and `frame`; `scheduler` is kept for callers and for a future scheduler-specific entry). A scene without OpenPBR layered materials is untouched by `layered`. Returns the
 * same object when there is no mode.
 */
export function ptApplyMode<T extends PtModeOptions>(options: T, scheduler: 'megakernel' | 'wavefront' = 'megakernel'): T & { integrator?: PtIntegratorOptions } {
  if (!options.mode) return options;
  const preset = ptModePreset(options.mode);
  const merged: Record<string, unknown> = { ...(options.integrator ?? {}) };
  const set = (key: string, value: unknown): void => {
    if (merged[key] === undefined) merged[key] = value;
  };
  // D31 max-speed opt-in: a caller who asks for a solid-shadow policy gets that policy's shadows, not the preset's `transmit` override.
  const solidPolicy = merged.alphaPolicy !== undefined && merged.alphaPolicy !== 'exact';
  for (const [key, value] of Object.entries(preset.integrator)) {
    if (solidPolicy && (key === 'alphaShadow' || key === 'alphaTransmit')) continue;
    set(key, value);
  }
  // P4.2e: the ReSTIR passes bind the blue-noise texture too, so `restir` no longer falls back to Owen-Sobol (megakernel only, as before).
  // P4.1b: the wavefront runs the blue-noise sampler and the frame outputs too (plain scheduler: `integrator.frame` throws with refill), so the scheduler no longer changes the preset.
  // S10: the kernels load the blue-noise texture shipped in @pmndrs/raytracer when none is passed (`loadBlueNoise`), so the preset no longer falls back to Owen-Sobol.
  if (options.mode === 'realtime') set('sampler', 'bluenoise');
  // D32: with ReSTIR on, the realtime preset shares the primary hit (one camera-ray trace per frame instead of two: x0.942 on Sponza 1280x720, TASKS P4.2c). Megakernel, trace-mode primaries and a
  // surface layout with positions only (the wavefront and the V-buffer path throw or ignore it); a caller's explicit `sharedHit` (even false) wins.
  const restir = merged.restir as { sharedHit?: boolean } | null | undefined;
  const layout = (options as { layout?: { positions?: boolean } }).layout;
  if (options.mode === 'realtime' && restir && typeof restir === 'object' && restir.sharedHit === undefined && scheduler === 'megakernel' && (merged.primary ?? 'trace') === 'trace' && layout?.positions) merged.restir = { ...restir, sharedHit: true };
  const materials = merged.materials as { features: { layered?: boolean }; layered?: Record<string, unknown> } | null | undefined;
  if (preset.layered && materials?.features.layered) {
    const layered: Record<string, unknown> = { ...(materials.layered ?? {}) };
    for (const [k, v] of Object.entries(preset.layered)) if (layered[k] === undefined) layered[k] = v;
    merged.materials = { ...materials, layered };
  }
  return { ...options, integrator: merged as PtIntegratorOptions } as T & { integrator?: PtIntegratorOptions };
}
