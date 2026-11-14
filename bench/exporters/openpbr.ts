/**
 * Shared analysis for exporting `openpbr` materials to the native references (pbrt-v4, Mitsuba 3), docs/design/tier1.md section 2.
 *
 * Everything renderer-neutral lives here: the class gate (class C throws unless `coverage`), the derived quantities (reflectance ->
 * eta/k, GGX alphas, extinction from `transmission_color`/`transmission_depth`, Cauchy fit of the Abbe number) and the note
 * bookkeeping. The renderer-specific syntax is in `pbrt.ts` / `mitsuba.ts`.
 *
 * Notes go to `ExportResult.approximations` (so the reference JSON sidecar carries them). A note that starts with `coverage:` means
 * the file is NOT a reference for the material (closest approximation, written only with `ExportOptions.coverage`).
 */
import type { OpenPbrParams, Rgb } from '../scenes/schema.ts';
import { classifyOpenPbr, openPbrAnisotropyAlphas, resolveOpenPbr } from '../scenes/openpbr.ts';
import type { OpenPbrResolved } from '../scenes/openpbr.ts';
import { roughnessToAlpha } from './common.ts';
import type { NativeTexture, TextureSet } from './textures.ts';

export type NativeRenderer = 'pbrt' | 'mitsuba';

export interface NativeCtx {
  renderer: NativeRenderer;
  /** Write the closest approximation (flagged `coverage:`) where no reference exists instead of throwing. */
  coverage: boolean;
  /** Layered-BSDF controls (pbrt `maxdepth` / `nsamples` of `coateddiffuse` and `coatedconductor`). */
  layered: { maxDepth: number; nSamples: number };
  notes: Set<string>;
  /**
   * Where the texture files of the scene go (P2.24). Without it an `openpbr` material that names a texture slot throws (or, with `coverage`,
   * drops the slot); with it base colour, metallic-roughness and emissive textures and a base colour alpha mask are written.
   */
  textures?: TextureSet;
}

/**
 * pbrt `coateddiffuse`/`coatedconductor` default `maxdepth` 10 truncates the random walk between the layers; a white diffuse base under a
 * clear coat (ior 1.5) keeps 60% of the light inside per bounce, so 10 loses energy (measured, bench/references/README.md). 32 is the converged value and 64 the default chosen by the P2.20
 * furnace (`bench/references/openpbr-check.ts`), nsamples 1.
 */
export const LAYERED_DEFAULTS = { maxDepth: 64, nSamples: 1 } as const;

/**
 * pbrt `coateddiffuse`/`coatedconductor` `thickness` (default 0.01). With `albedo` 0 the slab is still a unit-extinction absorber, so each
 * pass through it costs exp(-thickness / |cos|): the default loses 8% of a white furnace (measured). 1e-6 makes the layer transparent
 * (furnace 1.0000); the layer has no other effect with albedo 0 and g 0.
 */
export const LAYERED_THICKNESS = 1e-6;

export interface HomogeneousAbsorber {
  /** Extinction (= absorption, no scattering) per metre, per RGB channel. */
  sigma: Rgb;
}

const clamp01 = (v: number): number => Math.min(1, Math.max(0, v));
const mulRgb = (a: Rgb, b: Rgb): Rgb => [a[0] * b[0], a[1] * b[1], a[2] * b[2]];
const isWhite = (c: readonly number[]): boolean => c.every((v) => v === 1);

/** eta = 1, k = 2 sqrt(r) / sqrt(1 - r): the conductor whose normal-incidence reflectance is r (pbrt's `reflectance` parameterisation). */
export function reflectanceToEtaK(r: Rgb): { eta: Rgb; k: Rgb } {
  const k = r.map((v) => {
    const c = Math.min(0.9999, Math.max(0, v));
    return (2 * Math.sqrt(c)) / Math.sqrt(1 - c);
  }) as unknown as Rgb;
  return { eta: [1, 1, 1], k };
}

/** Beer-Lambert: `transmission_color` is the transmittance at `transmission_depth` (metres), so sigma = -ln(T) / depth. Channels of T are clamped to [1e-6, 1]. */
export function extinctionFromTransmission(color: Rgb, depth: number): Rgb {
  return color.map((t) => -Math.log(Math.min(1, Math.max(1e-6, t))) / depth).map((s) => (s === 0 ? 0 : s)) as unknown as Rgb;
}

/** Fraunhofer lines (nm) for the Abbe number V_d = (n_d - 1) / (n_F - n_C). */
export const LAMBDA_D = 587.5618;
export const LAMBDA_F = 486.1327;
export const LAMBDA_C = 656.2725;

/**
 * Cauchy two-term fit n(lambda) = A + B / lambda^2 (lambda in micrometres) of a medium with index `nd` at the d line and Abbe number `abbe`.
 * `abbe` <= 0 or infinite gives B = 0. Returns the coefficients and an evaluator.
 */
export function cauchyFit(nd: number, abbe: number): { a: number; b: number; eta: (lambdaNm: number) => number } {
  const um = (nm: number) => nm / 1000;
  const b = Number.isFinite(abbe) && abbe > 0 ? (nd - 1) / abbe / (1 / um(LAMBDA_F) ** 2 - 1 / um(LAMBDA_C) ** 2) : 0;
  const a = nd - b / um(LAMBDA_D) ** 2;
  return { a, b, eta: (nm) => a + b / um(nm) ** 2 };
}

/** F0 of a dielectric of index n against vacuum. */
const f0 = (n: number): number => ((n - 1) / (n + 1)) ** 2;
/** Inverse of `f0`. */
const iorFromF0 = (f: number): number => (1 + Math.sqrt(f)) / (1 - Math.sqrt(f));

export interface Derived {
  m: OpenPbrResolved;
  /** `base_weight * base_color`. */
  base: Rgb;
  metal: number;
  /** Specular IOR actually used (scaled in coverage mode for specular_weight < 1). */
  ior: number;
  /** GGX alphas of the specular lobe (tangent, bitangent), anisotropy included. */
  alphaU: number;
  alphaV: number;
  anisotropic: boolean;
  coatAlpha: number;
  thin: boolean;
  transmissive: boolean;
  /** Absorber for the interior of a closed mesh (transmission_depth > 0 and a non-white transmission_color). */
  medium?: HomogeneousAbsorber;
  /** Emitted radiance `emission_luminance * emission_color`, or undefined. With an emissive texture it is the factor the texel (linear) multiplies. */
  emission?: Rgb;
  /** Texture slots in use (decoded texels, sRGB base colour and emission, linear metallic-roughness). Only with `ctx.textures`. */
  tex: { base?: NativeTexture; mr?: NativeTexture; emissive?: NativeTexture };
  /** `alpha_mode` mask: the shape is cut out where `opacity * alpha >= cutoff` fails; `alpha` = the base colour texture's alpha channel (absent: constant 1). */
  mask?: { opacity: number; cutoff: number; texture?: NativeTexture };
}

const TEXTURE_FIELDS = ['baseColorTexture', 'metallicRoughnessTexture', 'normalTexture', 'emissiveTexture'] as const;

/**
 * Class gate, shared notes and derived values. Throws for class C (and for features the target cannot express at all) unless
 * `ctx.coverage`; then the closest approximation is derived and every deviation is recorded as a `coverage:` note.
 */
export function deriveOpenPbr(matName: string, p: OpenPbrParams, ctx: NativeCtx): Derived {
  const m = resolveOpenPbr(p);
  const tag = `${ctx.renderer}: material '${matName}'`;
  const note = (s: string): void => void ctx.notes.add(`${tag}: ${s}`);
  const cov = (s: string): void => void ctx.notes.add(`coverage: ${tag}: ${s}`);
  const cls = classifyOpenPbr(p);
  if (cls.class === 'C' && !ctx.coverage) {
    throw new Error(`${tag} is reference class C (${cls.reasons.filter((r) => r.startsWith('C')).join('; ')}); no native renderer expresses it. Pass the 'coverage' option to write the closest approximation (flagged, not a reference).`);
  }
  if (cls.class === 'C') {
    for (const r of cls.reasons.filter((x) => x.startsWith('C'))) cov(`class ${r}; dropped or approximated`);
  }
  const tex: Derived['tex'] = {};
  for (const f of TEXTURE_FIELDS) {
    const key = m[f];
    if (key === undefined) continue;
    if (!ctx.textures || f === 'normalTexture') {
      const why = f === 'normalTexture' ? 'a normal map has no exact native equivalent (pbrt `normalmap` and Mitsuba `normalmap` use their own tangent frames)' : 'is not expressible without a texture set (ctx.textures)';
      if (!ctx.coverage) throw new Error(`${tag}: texture slot '${f}' ${why}; pass 'coverage' to drop it`);
      cov(`texture '${f}' dropped (the factor alone is exported)`);
      continue;
    }
    let t: NativeTexture;
    try {
      t = ctx.textures.get(key);
    } catch (e) {
      throw new Error(`${tag}: texture slot '${f}': ${(e as Error).message}`);
    }
    const want = f === 'metallicRoughnessTexture' ? 'linear' : 'srgb';
    if (t.colorSpace !== want) throw new Error(`${tag}: texture '${key}' in slot '${f}' is ${t.colorSpace}; the slot reads ${want} texels`);
    if (f === 'baseColorTexture') {
      if (m.baseColor.some((c) => c !== 1) || m.baseWeight !== 1) throw new Error(`${tag}: a base colour texture needs baseColor [1, 1, 1] and baseWeight 1 (neither renderer multiplies an image by a colour without baking it into the texels)`);
      tex.base = t;
    } else if (f === 'metallicRoughnessTexture') tex.mr = t;
    else tex.emissive = t;
  }
  let mask: Derived['mask'];
  if (m.alphaMode === 'mask') {
    mask = { opacity: m.geometryOpacity, cutoff: m.alphaCutoff, ...(tex.base ? { texture: tex.base } : {}) };
    note(`alpha_mode mask: cut out where opacity ${m.geometryOpacity} x base colour alpha ${tex.base ? '(texture)' : '(1)'} < cutoff ${m.alphaCutoff}`);
  } else if (m.alphaMode === 'blend' && m.geometryOpacity < 1) {
    cov('alpha blend is approximated as opaque');
  }
  if (m.geometryOpacity < 1 && m.alphaMode === 'opaque') note('geometry_opacity < 1 ignored in opaque alpha mode (as in our renderer)');
  if (m.baseDiffuseRoughness > 0) note('base_diffuse_roughness > 0 (Oren-Nayar) is not expressible; the base is Lambertian');
  if (m.coatWeight > 0 && !isWhite(m.coatColor)) note('coat_color is ignored (the coat is clear)');

  const metal = clamp01(m.baseMetalness);
  const base = mulRgb(m.baseColor, [m.baseWeight, m.baseWeight, m.baseWeight]) as Rgb;
  const anisotropic = m.specularRoughnessAnisotropy > 0;
  let alphaU = roughnessToAlpha(m.specularRoughness);
  let alphaV = alphaU;
  if (anisotropic) {
    const a = openPbrAnisotropyAlphas(m.specularRoughness, m.specularRoughnessAnisotropy);
    alphaU = a.alphaT;
    alphaV = a.alphaB;
    note('anisotropic: the GGX tangent axis is the renderer\'s own dp/du of the mesh uvs (the loader carries TEXCOORD_0 for anisotropic materials); on AnisotropyStrengthTest it is the glTF TANGENT within 1.9 deg per vertex (3.7 deg per flat triangle), V75');
  }
  if (m.anisotropyRotation !== 0) note('anisotropy_rotation is ignored (the axis follows the mesh uv frame)');

  let ior = m.specularIor;
  if (metal < 1 && m.specularWeight < 1 && m.coatWeight === 0 && !m.geometryThinWalled) {
    // class C already gated above: coverage scales F0 through the IOR
    const w = m.specularWeight * (m.specularColor[0] + m.specularColor[1] + m.specularColor[2]) / 3;
    ior = iorFromF0(clamp01(f0(ior) * w));
    cov(`specular_weight/specular_color scale F0 only through an IOR of ${ior.toFixed(4)} (OpenPBR scales F at both ends)`);
  } else if (metal < 1 && m.specularWeight > 0 && !isWhite(m.specularColor)) {
    cov('specular_color != 1 ignored');
  }

  const transmissive = metal < 1 && m.transmissionWeight > 0;
  let medium: HomogeneousAbsorber | undefined;
  if (transmissive) {
    if (m.transmissionDepth > 0 && !isWhite(m.transmissionColor)) {
      medium = { sigma: extinctionFromTransmission(m.transmissionColor, m.transmissionDepth) };
      if (m.geometryThinWalled) note('transmission_depth is ignored for a thin-walled surface (no interior)');
      if (m.geometryThinWalled) medium = undefined;
    } else if (!isWhite(m.transmissionColor)) {
      note('transmission_color tint at depth 0 is not expressible (neither renderer tints a dielectric): exported untinted, compare white-base only');
    }
  }
  const emission = m.emissionLuminance > 0 ? (m.emissionColor.map((c) => c * m.emissionLuminance) as unknown as Rgb) : undefined;
  if (tex.mr && anisotropic) throw new Error(`${tag}: a metallic-roughness texture with anisotropy is not exported (the roughness texture is isotropic)`);
  return { m, base, metal, ior, alphaU, alphaV, anisotropic, coatAlpha: roughnessToAlpha(m.coatRoughness), thin: m.geometryThinWalled, transmissive, medium, emission, tex, ...(mask ? { mask } : {}) };
}
