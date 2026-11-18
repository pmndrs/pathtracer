/**
 * Fitted correction of the albedo layering's `'internal'` base (P2.32b, `albedoCorrection: 'lut' | 'texture'`; DIVERGES V97).
 *
 * The closed-form internal base (Weidlich-Wilkie; `layered.ts`) assumes a smooth interface. Under a rough interface the stochastic walk
 * (pbrt's `LayeredBxDF`, the reference) loses energy at every rough crossing (single scattering GGX, no multiple-scattering term), so the
 * closed form reads +11-15% bright on a bright rough base. The correction multiplies the base lobe by
 *     c(r, mu_o, x) = a + b x + c x^2,   x = base * Ri(ior)   (per channel, base = the channel's reflectance),
 * with (a, b, c) bilinear in (sqrt r, sqrt mu_o) from an 8 x 8 grid of nodes. The fit (`bench/layered-correction-fit.ts`, data in `bench/results/layered-correction/`)
 * matches the base energy of the closed form to the walk's (thickness 1e-6, maxDepth 64; grey bases 0.2..1, ior 1.33 / 1.5 / 2, r nodes 0.08..1, mu nodes 0.03..1),
 * so ior enters only through x (no ior axis; the dependence on base and ior collapses onto base * Ri to about 3% rms). The target is 'mid' (DIVERGES V97):
 * the mean of the walk's sampler albedo and of its pbrt-exit-MIS `f` albedo (pbrt's own `f` and `Sample_f` disagree by +4% at r 0.33 to +13% at r 1, V58), from r 0.33
 * up, because a reference-mode render reads between the two (next-event estimates use `f`, the bounce uses the sampler); 'sample' (pure sampler) is a fit option.
 * Smooth interfaces (r below ~0.03) and the r <= 0.02 nodes are the identity.
 * c is capped at 1: the correction only removes energy, so a white furnace stays <= 1.
 *
 * Two encodings of the same numbers: `'lut'` is a WGSL const array (192 numbers, the layered-lut.ts style), `'texture'` an 8 x 8 `rgba16float`
 * texture (a, b, c, 0; x = mu node, y = r node) read with four `textureLoad`s and a manual bilinear (no sampler, one binding;
 * `PT_LAYER_CORR_BINDING` in group 2). The CPU twin of the texture encoding rounds the numbers to f16.
 */

import { f16BitsToF32, f32ToF16Bits } from '@pmndrs/raytracer';
import { PT_LAYER_CORR_DATA } from './layered-correction-data.ts';

export const PT_LAYER_CORR_NR = 8;
export const PT_LAYER_CORR_NM = 8;
/** Group 2 binding of the `'texture'` encoding (megakernel io group; wavefront shade io group). */
export const PT_LAYER_CORR_BINDING = 10;

export type PtAlbedoCorrection = 'none' | 'lut' | 'texture';

/**
 * The library default of `albedoCorrection` (D32, provisional): the fitted `texture` correction on the albedo layering's internal base. P2.32b: it removes the +11-15% overshoot of a bright
 * rough base (AttenuationTest +12.9% -> -3.9% against the walk, SheenTestGrid +11.8% -> +1.9%) for 1-3% of megakernel time (wavefront within noise) and +0.8 KB of kernel text.
 * `'none'` keeps the closed form as it was (the pre-D32 kernel text); `'lut'` is the const-array encoding of the same numbers.
 */
export const PT_DEFAULT_ALBEDO_CORRECTION: PtAlbedoCorrection = 'texture';

/** The correction kind a layered options object resolves to (`undefined` = none: the closed form only). Only the albedo layering's internal base has one. */
export function ptAlbedoCorrectionKind(o: { layering?: 'albedo' | 'stochastic'; albedoBase?: 'openpbr' | 'internal'; albedoCorrection?: PtAlbedoCorrection } | undefined): 'lut' | 'texture' | undefined {
  if ((o?.layering ?? 'albedo') !== 'albedo' || (o?.albedoBase ?? 'internal') !== 'internal') return undefined;
  const kind = o?.albedoCorrection ?? PT_DEFAULT_ALBEDO_CORRECTION;
  return kind === 'none' ? undefined : kind;
}

/** (a, b, c) per node, index ((ir * NM + im) * 3 + k); the numbers the WGSL const array carries. */
export function ptLayerCorrLut(): Float32Array {
  return Float32Array.from(PT_LAYER_CORR_DATA);
}

/** The `'texture'` encoding as rgba16float texels (Uint16, 4 per node, row = r node, column = mu node): (a, b, c, 0) rounded to f16. */
export function ptLayerCorrTextureHalf(): Uint16Array {
  const out = new Uint16Array(PT_LAYER_CORR_NR * PT_LAYER_CORR_NM * 4);
  for (let n = 0; n < PT_LAYER_CORR_NR * PT_LAYER_CORR_NM; n++) for (let k = 0; k < 3; k++) out[n * 4 + k] = f32ToF16Bits(PT_LAYER_CORR_DATA[n * 3 + k] as number);
  return out;
}

let cachedLut: Float32Array | null = null;
let cachedTex: Float32Array | null = null;
function table(kind: 'lut' | 'texture'): Float32Array {
  if (kind === 'lut') return (cachedLut ??= ptLayerCorrLut());
  if (!cachedTex) {
    const half = ptLayerCorrTextureHalf();
    cachedTex = new Float32Array(PT_LAYER_CORR_NR * PT_LAYER_CORR_NM * 3);
    for (let n = 0; n < PT_LAYER_CORR_NR * PT_LAYER_CORR_NM; n++) for (let k = 0; k < 3; k++) cachedTex[n * 3 + k] = f16BitsToF32(half[n * 4 + k] as number);
  }
  return cachedTex;
}

/** Bilinear (a, b, c) at outside cosine `mu` and perceptual roughness `roughness` (the CPU twin of `ptLayerCorrCoef` in WGSL). */
export function ptLayerCorrCoefs(kind: 'lut' | 'texture', mu: number, roughness: number): [number, number, number] {
  const t = table(kind);
  const x = Math.sqrt(Math.min(Math.max(mu, 0), 1)) * (PT_LAYER_CORR_NM - 1);
  const y = Math.sqrt(Math.min(Math.max(roughness, 0), 1)) * (PT_LAYER_CORR_NR - 1);
  const x0 = Math.min(Math.floor(x), PT_LAYER_CORR_NM - 2);
  const y0 = Math.min(Math.floor(y), PT_LAYER_CORR_NR - 2);
  const fx = x - x0;
  const fy = y - y0;
  const at = (iy: number, ix: number, k: number): number => t[(iy * PT_LAYER_CORR_NM + ix) * 3 + k] as number;
  const k3 = (k: number): number => (at(y0, x0, k) * (1 - fx) + at(y0, x0 + 1, k) * fx) * (1 - fy) + (at(y0 + 1, x0, k) * (1 - fx) + at(y0 + 1, x0 + 1, k) * fx) * fy;
  return [k3(0), k3(1), k3(2)];
}

/** The correction factor c for one channel: `base` is the channel's reflectance, `ri` the interface's internal reflectance (`ptInternalReflectance(ior)`). */
export function ptLayerCorrection(kind: 'lut' | 'texture', base: number, ri: number, mu: number, roughness: number): number {
  const [a, b, c] = ptLayerCorrCoefs(kind, mu, roughness);
  const x = Math.min(Math.max(base, 0), 1) * ri;
  return Math.min(1, Math.max(0, a + x * (b + x * c)));
}

/** True when a kernel built with these layered options declares the correction texture (`albedoCorrection: 'texture'` on the albedo layering with the internal base). */
export function ptLayerCorrNeedsTexture(layered: { layering?: 'albedo' | 'stochastic'; albedoBase?: 'openpbr' | 'internal'; albedoCorrection?: PtAlbedoCorrection } | undefined): boolean {
  return ptAlbedoCorrectionKind(layered) === 'texture';
}

/** The bind group layout entry of the texture (group 2, `PT_LAYER_CORR_BINDING`). */
export function ptLayerCorrLayoutEntry(): GPUBindGroupLayoutEntry {
  return { binding: PT_LAYER_CORR_BINDING, visibility: 4 /* COMPUTE */, texture: { sampleType: 'unfilterable-float', viewDimension: '2d' } };
}

/** The 8 x 8 `rgba16float` texture holding `ptLayerCorrTextureHalf()` (TEXTURE_BINDING | COPY_DST, written once). */
export function ptLayerCorrCreateTexture(device: GPUDevice, label: string): GPUTexture {
  const texture = device.createTexture({ label: `${label} layered correction`, size: [PT_LAYER_CORR_NM, PT_LAYER_CORR_NR], format: 'rgba16float', usage: 0x04 | 0x02 /* TEXTURE_BINDING | COPY_DST */ });
  device.queue.writeTexture({ texture }, ptLayerCorrTextureHalf() as Uint16Array<ArrayBuffer>, { bytesPerRow: PT_LAYER_CORR_NM * 8 }, [PT_LAYER_CORR_NM, PT_LAYER_CORR_NR]);
  return texture;
}
