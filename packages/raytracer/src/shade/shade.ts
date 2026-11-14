/**
 * L2 `rtShade` (P1.3): one-bounce shading of a hit. Radiance leaving a surface point toward the viewer =
 * emission + direct light (ONE next-event ray, one light sample) + indirect (a cache hook, black by default).
 *
 * This file is the options, the constants and the TypeScript twin (f64) of the WGSL in `shade-wgsl.ts`. The twin is what the unit
 * tests and the `shade-check` page compare the GPU against, and it is also usable for CPU-side probes and baking.
 *
 * Modes (`depth` in the spec): FLAT (no rays), LAMBERT (diffuse only, one shadow ray), FULL (diffuse + GGX specular, one shadow ray).
 * The estimator is the plain single-sample one, `L = Le f |cos| / (pmfLight * pdfSample)`: no MIS, because `rtShade` takes no BSDF
 * sample (the consumer's continuation ray is the BSDF sample and it ends at its own `rtShade`). See docs/design/shade.md.
 *
 * Conventions: every vector is world space; `wo` points from the surface toward the viewer; `normal` and `geometricNormal` are on the
 * viewer's side; `u = (selectLight, pointU, pointV)` are three independent numbers in [0, 1) from the caller's sampler.
 */
import { SCENE_NONE } from '../scene/constants.ts';
import { sampleAlias } from '../scene/lights.ts';
import type { AliasTable, RtLight, Vec3Tuple } from '../scene/lights.ts';
import { sampleLightLi, type LightLi, type Rgb, environmentRadiance, orthonormalBasis, sunContains, sunRadiance } from '../scene/light-sample.ts';
import { materialAlpha, type RtMaterial } from '../scene/material.ts';

/** `depth` argument of `rtShade`. */
export const RT_SHADE_MODE = { flat: 0, lambert: 1, full: 2 } as const;
export type RtShadeMode = keyof typeof RT_SHADE_MODE;

export interface RtShadeOptions {
  /**
   * The `Full` mode's BSDF. `'builtin'` (default): diffuse + GGX (height-correlated Smith, Schlick Fresnel), about 40 lines of WGSL,
   * no dependency. `'external'`: you declare `fn rtShadeBsdf(p: RtShadePoint, wi: vec3f) -> vec3f` (the BSDF value without the cosine)
   * anywhere in the final shader, for example over `@pmndrs/pathtracer`'s `ptBsdfEval`.
   */
  bsdf?: 'builtin' | 'external';
  /**
   * The indirect term. `'none'` (default): the module defines `rtShadeIndirect` and it returns black. `'external'`: you declare
   * `fn rtShadeIndirect(p: RtShadePoint) -> vec3f` (outgoing radiance toward `p.wo`, already weighted by the BSDF) anywhere in the final shader.
   */
  cache?: 'none' | 'external';
  /**
   * How an environment light picked by the light selection is sampled: `'uniform'` over the sphere (same as the path tracer's NEE)
   * or `'cosine'` over the shading hemisphere (zero variance for a Lambertian surface under a uniform sky). Default 'uniform'
   * until the A/B in docs/design/shade.md settles it.
   */
  envSampling?: 'uniform' | 'cosine';
  /** FLAT mode content: `'cache'` (default) = emission + indirect only; `'albedo'` = emission + the unlit base colour, in `direct`. */
  flat?: 'cache' | 'albedo';
  /** `rtShadeMiss` also returns the sun's radiance for directions inside its disc (the path tracer's default: no). */
  sunVisible?: boolean;
  /** GGX alpha floor of the builtin BSDF (a near-mirror lobe is a lottery for one light sample). Default 0.01. */
  minAlpha?: number;
}

export interface ResolvedShadeOptions {
  bsdf: 'builtin' | 'external';
  cache: 'none' | 'external';
  envSampling: 'uniform' | 'cosine';
  flat: 'cache' | 'albedo';
  sunVisible: boolean;
  minAlpha: number;
}

export function resolveShadeOptions(o: RtShadeOptions = {}): ResolvedShadeOptions {
  const minAlpha = o.minAlpha ?? 0.01;
  if (!(minAlpha > 0 && minAlpha <= 1)) throw new RangeError(`rtShade: minAlpha ${minAlpha} must be in (0, 1]`);
  return { bsdf: o.bsdf ?? 'builtin', cache: o.cache ?? 'none', envSampling: o.envSampling ?? 'uniform', flat: o.flat ?? 'cache', sunVisible: o.sunVisible ?? false, minAlpha };
}

// ---------------------------------------------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------------------------------------------

/** Twin of WGSL `RtShadePoint`: what the hooks (`rtShadeIndirect`, `rtShadeBsdf`) get. */
export interface ShadePoint {
  position: Vec3Tuple;
  /** Shading normal on the viewer's side. */
  normal: Vec3Tuple;
  /** Geometric normal on the viewer's side (shadow rays offset along it). */
  geometricNormal: Vec3Tuple;
  /** Unit vector toward the viewer. */
  wo: Vec3Tuple;
  /** The hit was on the authored front face (decides one-sided emission). */
  frontFace: boolean;
  material: RtMaterial;
  /** Diffuse reflectance: `baseColor * baseWeight * (1 - metalness) * (1 - transmissionWeight)`. */
  albedo: Rgb;
}

/** Twin of WGSL `RtShaded`. `emission + direct + indirect` is the radiance. */
export interface Shaded {
  /** Emission of the hit surface; the environment (and optionally the sun disc) for a miss. */
  emission: Rgb;
  /** The one-sample direct light estimate (FLAT 'albedo': the unlit base colour). */
  direct: Rgb;
  /** What `rtShadeIndirect` returned. */
  indirect: Rgb;
  /** The light that was sampled (`SCENE_NONE`: none, or a sample with no contribution). */
  light: number;
  /** A shadow ray was traced and something blocked it. */
  occluded: boolean;
}

/** The light data the twin needs: `SceneCopy` has all of it. */
export interface ShadeLights {
  alias: Pick<AliasTable, 'prob' | 'alias' | 'pmf'>;
  lights: readonly RtLight[];
  environmentLight: number;
}

/** What the twin calls where the GPU traces a ray. `distance` is Infinity for the sun and the environment. Return true when nothing blocks. */
export type ShadeVisible = (origin: Vec3Tuple, direction: Vec3Tuple, distance: number, light: number) => boolean;

export interface ShadeHooks {
  /** Twin of `rtShadeIndirect`. Default black. */
  indirect?: (p: ShadePoint) => Rgb;
  /** Twin of `rtShadeBsdf` (value without the cosine), used when `options.bsdf` is 'external'. */
  bsdf?: (p: ShadePoint, wi: Vec3Tuple) => Rgb;
}

// ---------------------------------------------------------------------------------------------------------------
// Vector helpers
// ---------------------------------------------------------------------------------------------------------------

const dot = (a: Vec3Tuple, b: Vec3Tuple): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const neg = (a: Vec3Tuple): Vec3Tuple => [-a[0], -a[1], -a[2]];
const add3 = (a: Vec3Tuple, b: Vec3Tuple): Vec3Tuple => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const mul3 = (a: Vec3Tuple, s: number): Vec3Tuple => [a[0] * s, a[1] * s, a[2] * s];
const norm = (a: Vec3Tuple): Vec3Tuple => mul3(a, 1 / Math.hypot(a[0], a[1], a[2]));
const ZERO: Rgb = [0, 0, 0];

/** Twin of `rtShadeAlbedo`. */
export function shadeAlbedo(m: RtMaterial): Rgb {
  const k = m.baseWeight * (1 - m.metalness) * (1 - m.transmissionWeight);
  return [m.baseColor[0] * k, m.baseColor[1] * k, m.baseColor[2] * k];
}

/** Builds a `ShadePoint` the way WGSL `rtShadePoint` does from a hit: normals flip to the side the ray came from. */
export function makeShadePoint(args: { material: RtMaterial; position: Vec3Tuple; shadingNormal: Vec3Tuple; geometricNormal: Vec3Tuple; frontFace: boolean; rayDirection: Vec3Tuple }): ShadePoint {
  const flip = args.frontFace ? 1 : -1;
  return {
    position: args.position,
    normal: mul3(args.shadingNormal, flip),
    geometricNormal: mul3(args.geometricNormal, flip),
    wo: neg(args.rayDirection),
    frontFace: args.frontFace,
    material: args.material,
    albedo: shadeAlbedo(args.material),
  };
}

/** Twin of `rtShadeEmission`. */
export function shadeEmission(p: ShadePoint): Rgb {
  const m = p.material;
  return m.emissionLuminance > 0 && (p.frontFace || m.emissionTwoSided) ? [...m.emissionColor] : [0, 0, 0];
}

// ---------------------------------------------------------------------------------------------------------------
// Builtin BSDF
// ---------------------------------------------------------------------------------------------------------------

/** Normal-incidence specular reflectance of a material (dielectric from `ior`, specular weight and colour; metal from `baseColor` or eta/k; mixed by metalness). Informational: the BSDF uses `shadeFresnel`, which also scales the dielectric part by `specularWeight` at all angles. */
export function shadeF0(m: RtMaterial): Rgb {
  const d = (m.ior - 1) / (m.ior + 1);
  const out: Rgb = [0, 0, 0];
  for (let c = 0; c < 3; c++) {
    const dielectric = Math.min(1, d * d * m.specularWeight * (m.specularColor[c] as number));
    let metal = (m.baseColor[c] as number) * m.baseWeight;
    if (m.conductorEtaK) {
      const n = m.conductorEta[c] as number;
      const k = m.conductorK[c] as number;
      metal = ((n - 1) * (n - 1) + k * k) / ((n + 1) * (n + 1) + k * k);
    }
    out[c] = dielectric + (metal - dielectric) * m.metalness;
  }
  return out;
}

/** Schlick Fresnel at `voh`: dielectric part scaled by `specularWeight` and `specularColor` (weight 0 = no specular at all), metal part from `shadeF0`'s metal term, mixed by `metalness`. */
export function shadeFresnel(m: RtMaterial, voh: number): Rgb {
  const d = (m.ior - 1) / (m.ior + 1);
  const w = (1 - voh) ** 5;
  const out: Rgb = [0, 0, 0];
  for (let c = 0; c < 3; c++) {
    const dielectric = m.specularWeight * (m.specularColor[c] as number) * (d * d + (1 - d * d) * w);
    let f0m = (m.baseColor[c] as number) * m.baseWeight;
    if (m.conductorEtaK) {
      const n = m.conductorEta[c] as number;
      const k = m.conductorK[c] as number;
      f0m = ((n - 1) * (n - 1) + k * k) / ((n + 1) * (n + 1) + k * k);
    }
    const metal = f0m + (1 - f0m) * w;
    out[c] = dielectric + (metal - dielectric) * m.metalness;
  }
  return out;
}

export function ggxD(noh: number, a2: number): number {
  const d = noh * noh * (a2 - 1) + 1;
  return a2 / (Math.PI * d * d);
}

/** Smith's Lambda for GGX at a direction with cosine `c` to the normal. */
export function ggxLambda(c: number, a2: number): number {
  const c2 = c * c;
  return (Math.sqrt(1 + (a2 * (1 - c2)) / c2) - 1) / 2;
}

/** Builtin BSDF value `f(wo, wi)` (no cosine): Lambert + GGX specular with height-correlated Smith G and Schlick Fresnel. 0 below the shading hemisphere. */
export function builtinBsdf(p: ShadePoint, wi: Vec3Tuple, options: Pick<ResolvedShadeOptions, 'minAlpha'> = { minAlpha: 0.01 }): Rgb {
  const nol = dot(p.normal, wi);
  if (!(nol > 0)) return [0, 0, 0];
  const nov = Math.max(dot(p.normal, p.wo), 1e-4);
  const hRaw = add3(p.wo, wi);
  const hl = Math.hypot(...hRaw);
  const diffuse = mul3(p.albedo, 1 / Math.PI);
  if (!(hl > 0)) return diffuse;
  const h = mul3(hRaw, 1 / hl);
  const noh = Math.min(Math.max(dot(p.normal, h), 0), 1);
  const voh = Math.min(Math.max(dot(p.wo, h), 0), 1);
  const alpha = Math.max(materialAlpha(p.material), options.minAlpha);
  const a2 = alpha * alpha;
  const g = 1 / (1 + ggxLambda(nov, a2) + ggxLambda(nol, a2));
  const scale = (ggxD(noh, a2) * g) / (4 * nov * nol);
  const fr = shadeFresnel(p.material, voh);
  return [0, 1, 2].map((c) => (diffuse[c] as number) + scale * (fr[c] as number)) as Rgb;
}

// ---------------------------------------------------------------------------------------------------------------
// Direct light
// ---------------------------------------------------------------------------------------------------------------

/** Concentric disk (Shirley-Chiu) to the unit disk. */
export function concentricDisk(u0: number, u1: number): [number, number] {
  const a = 2 * u0 - 1;
  const b = 2 * u1 - 1;
  if (a === 0 && b === 0) return [0, 0];
  let r: number;
  let theta: number;
  if (Math.abs(a) > Math.abs(b)) {
    r = a;
    theta = (Math.PI / 4) * (b / a);
  } else {
    r = b;
    theta = Math.PI / 2 - (Math.PI / 4) * (a / b);
  }
  return [r * Math.cos(theta), r * Math.sin(theta)];
}

/** Cosine-weighted direction about unit `n`, with `pdf = cos / pi`. */
export function sampleCosineAbout(n: Vec3Tuple, u0: number, u1: number): { wi: Vec3Tuple; pdf: number } {
  const [x, y] = concentricDisk(u0, u1);
  const z = Math.sqrt(Math.max(0, 1 - x * x - y * y));
  const [a, b] = orthonormalBasis(n);
  const wi = add3(add3(mul3(a, x), mul3(b, y)), mul3(n, z));
  return { wi, pdf: z / Math.PI };
}

/** Radiance of an environment/sun for a direction that left the scene (twin of `rtShadeMiss`). */
export function shadeMiss(lights: ShadeLights, direction: Vec3Tuple, options: Pick<ResolvedShadeOptions, 'sunVisible'> = { sunVisible: false }): Rgb {
  let out: Rgb = [0, 0, 0];
  for (const light of lights.lights) {
    if (light.kind === 'environment') out = add3(out, environmentRadiance(light));
    else if (light.kind === 'sun' && options.sunVisible && sunContains(light, direction)) out = add3(out, sunRadiance(light));
  }
  return out;
}

/**
 * Twin of `rtShade` for a point (`rtShadePoint` is `makeShadePoint`). `u` = (light selection, point u, point v).
 * `visible` answers the one shadow ray (the page replays the GPU's answer, tests use analytic occluders).
 */
export function shade(lights: ShadeLights, p: ShadePoint, mode: RtShadeMode, u: Vec3Tuple, visible: ShadeVisible, hooks: ShadeHooks = {}, options: RtShadeOptions = {}): Shaded {
  const o = resolveShadeOptions(options);
  const out: Shaded = { emission: shadeEmission(p), direct: [0, 0, 0], indirect: hooks.indirect ? hooks.indirect(p) : [...ZERO], light: SCENE_NONE, occluded: false };
  const m = p.material;
  if (mode === 'flat') {
    if (o.flat === 'albedo') out.direct = [m.baseColor[0] * m.baseWeight, m.baseColor[1] * m.baseWeight, m.baseColor[2] * m.baseWeight];
    return out;
  }
  const pick = sampleAlias(lights.alias, u[0]);
  if (pick.index === SCENE_NONE || !(pick.pmf > 0)) return out;
  const light = lights.lights[pick.index] as RtLight;
  let li: LightLi;
  let pdf: number;
  if (light.kind === 'environment' && o.envSampling === 'cosine') {
    const s = sampleCosineAbout(p.normal, u[1], u[2]);
    li = { valid: s.pdf > 0, wi: s.wi, distance: Infinity, point: [NaN, NaN, NaN], normal: [0, 0, 0], radiance: environmentRadiance(light), pdf: s.pdf, delta: false };
    pdf = s.pdf;
  } else {
    li = sampleLightLi(light, p.position, u[1], u[2]);
    pdf = li.pdf;
  }
  if (!li.valid || !(pdf > 0)) return out;
  let f: Rgb;
  if (mode === 'lambert') {
    const nol = dot(p.normal, li.wi);
    f = nol > 0 ? mul3(p.albedo, 1 / Math.PI) : [0, 0, 0];
  } else if (o.bsdf === 'external') {
    if (!hooks.bsdf) throw new Error("rtShade twin: bsdf 'external' needs hooks.bsdf");
    f = hooks.bsdf(p, li.wi);
  } else {
    f = builtinBsdf(p, li.wi, o);
  }
  const cos = Math.abs(dot(p.normal, li.wi));
  const w: Rgb = [f[0] * cos, f[1] * cos, f[2] * cos];
  if (!(Math.max(w[0], w[1], w[2]) > 0)) return out;
  out.light = pick.index;
  if (!visible(p.position, li.wi, li.distance, pick.index)) {
    out.occluded = true;
    return out;
  }
  const k = 1 / (pick.pmf * pdf);
  out.direct = [li.radiance[0] * w[0] * k, li.radiance[1] * w[1] * k, li.radiance[2] * w[2] * k];
  return out;
}

/** Radiance toward the viewer: `emission + direct + indirect`. */
export function shadeTotal(s: Shaded): Rgb {
  return [s.emission[0] + s.direct[0] + s.indirect[0], s.emission[1] + s.direct[1] + s.indirect[1], s.emission[2] + s.direct[2] + s.indirect[2]];
}

/** Twin of `rtShadeDiffuseFromIrradiance`: a cache that stores irradiance E (W/m^2, cosine-convolved about `p.normal`) returns `albedo * E / pi`. */
export function shadeDiffuseFromIrradiance(p: ShadePoint, irradiance: Rgb): Rgb {
  return [(p.albedo[0] * irradiance[0]) / Math.PI, (p.albedo[1] * irradiance[1]) / Math.PI, (p.albedo[2] * irradiance[2]) / Math.PI];
}
