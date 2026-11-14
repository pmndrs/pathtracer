/**
 * P2.2: the adapter from the scene copy's material record (`RtMaterial`, `rtMaterial(id)` in WGSL) to the BSDF's plain
 * parameter set (`PtBsdfParams`). One function in TypeScript and one in WGSL, line for line.
 *
 * Rules, for the bench subset and the corners of OpenPBR the scene copy can already say (anything richer is a later lobe):
 *
 *  1. `transmissionWeight > 0.5`  -> smooth or rough dielectric: `ior`, alpha = specularRoughness^2. (The bench `dielectric`;
 *     the bench maps it to transmissionWeight 1.) Absorption (`transmissionDepth`) and thin walls are not applied.
 *  2. `metalness > 0.5` and `conductorEtaK` -> GGX conductor with the record's complex IOR (the bench `conductor`).
 *  3. `metalness > 0.5` without eta/k -> GGX conductor whose normal-incidence reflectance is `baseColor * baseWeight`
 *     (eta = 1, k = 2 sqrt(r) / sqrt(1 - r): Gulbrandsen's artist-friendly inverse). Approximation, not a Tier 0 case.
 *  4. `baseWeight * max(baseColor) > 0` -> Lambert with reflectance `baseColor * baseWeight`. The dielectric specular lobe
 *     over it (`specularWeight`, roughness) is NOT evaluated yet: the bench `diffuse` has `specularWeight` 0, a general
 *     OpenPBR dielectric on top of a diffuse base needs the layered lobe (a later task).
 *  5. otherwise black (scatters nothing; the bench `black`, emitters' backing).
 *
 * The 0.5 thresholds make the blend weights binary; the bench only produces 0 and 1.
 *
 * OpenPBR mode (P2.21/P2.22/P2.30; `PtMaterialOptions`, a scene with an `openpbr` material): same function with one more argument,
 * a uniform `u` of the vertex for the stochastic metal/dielectric choice, and these rules instead (metal first):
 *
 *  1. metal: with `features.mix`, `u < metalness` (pbrt `mix`: the metal end with probability `metalness`, so the expectation is the
 *     exact linear blend; metalness 0 and 1 stay exact); without it `metalness > 0.5`. Complex-Fresnel conductor as in 2./3. above.
 *  2. `transmissionWeight > 0.5`: thin-walled (and `features.thin`) -> `PT_BSDF_THIN` with `ior` and the transmission colour as tint
 *     (`transmissionTint`); otherwise `PT_BSDF_DIELECTRIC`. The interior medium (`transmissionDepth`) is path state, not BSDF.
 *  3. `features.layered` and a dielectric specular lobe (`specularWeight > 0` and `ior != 1`, or a coat with `coatWeight > 0.5`):
 *     `PT_BSDF_LAYERED`, an interface of (`ior`, `specularRoughness`, Fresnel scale `specularColor * specularWeight`) over a Lambert
 *     base. A coat replaces the interface with (`coatIor`, `coatRoughness`, scale 1) and drops the base's own specular lobe, which is
 *     what the pbrt exporter writes (`coateddiffuse` with the coat as interface; DIVERGES V69). `coatWeight` is thresholded at 0.5.
 *  4. otherwise Lambert / black as above.
 *
 *  P2.23 (`features.coatedMetal`): a metal with `coatWeight > 0.5` and `coatIor != 1` becomes `PT_BSDF_LAYERED` with a conductor bottom layer
 *  (pbrt `coatedconductor`): interface (`coatIor`, `coatRoughness`), conductor eta and k divided by `coatIor`, alpha `specularRoughness^2`.
 *  Packing: `ior`/`alpha` = interface, `reflectance` = conductor eta', `eta` = conductor k', `k` = (thickness, conductor alpha, 1 = conductor).
 */
import type { RtMaterial } from '@pmndrs/raytracer';
import { PT_BSDF_BLACK, PT_BSDF_CONDUCTOR, PT_BSDF_DIELECTRIC, PT_BSDF_DIFFUSE, PT_BSDF_LAYERED, PT_BSDF_THIN, type PtBsdfParams, type PtVec3 } from './bsdf.ts';
import type { PtLayeredOptions, PtLayeredParams } from './layered.ts';

/** k that gives a conductor with eta = 1 the normal-incidence reflectance `r` (clamped just below 1). */
export function conductorKFromReflectance(r: number): number {
  const c = Math.min(Math.max(r, 0), 0.9999);
  return (2 * Math.sqrt(c)) / Math.sqrt(1 - c);
}

/** GGX alpha from a perceptual roughness, as `rtMaterialAlpha` (`roughness^2`). */
const alphaOf = (m: RtMaterial): number => m.specularRoughness * m.specularRoughness;

export function ptBsdfFromMaterial(m: RtMaterial, options?: PtMaterialOptions, u = 0): PtBsdfParams {
  if (options) return ptBsdfFromOpenPbr(m, options, u);
  const a = alphaOf(m);
  if (m.transmissionWeight > 0.5) {
    return { kind: PT_BSDF_DIELECTRIC, reflectance: [0, 0, 0], eta: [0, 0, 0], k: [0, 0, 0], alpha: [a, a], ior: m.ior };
  }
  if (m.metalness > 0.5) {
    if (m.conductorEtaK) {
      return { kind: PT_BSDF_CONDUCTOR, reflectance: [0, 0, 0], eta: [...m.conductorEta] as PtVec3, k: [...m.conductorK] as PtVec3, alpha: [a, a], ior: 1 };
    }
    const k: PtVec3 = [0, 1, 2].map((i) => conductorKFromReflectance((m.baseColor[i] as number) * m.baseWeight)) as unknown as PtVec3;
    return { kind: PT_BSDF_CONDUCTOR, reflectance: [0, 0, 0], eta: [1, 1, 1], k, alpha: [a, a], ior: 1 };
  }
  const reflectance: PtVec3 = [m.baseColor[0] * m.baseWeight, m.baseColor[1] * m.baseWeight, m.baseColor[2] * m.baseWeight];
  if (Math.max(...reflectance) > 0) {
    return { kind: PT_BSDF_DIFFUSE, reflectance, eta: [0, 0, 0], k: [0, 0, 0], alpha: [0, 0], ior: 1 };
  }
  return { kind: PT_BSDF_BLACK, reflectance: [0, 0, 0], eta: [0, 0, 0], k: [0, 0, 0], alpha: [0, 0], ior: 1 };
}

/** The WGSL twin. Needs `ptBsdfWgsl` (PtBsdfParams, PT_BSDF_*) and the scene module (`RtMaterial`) before it. */
export function ptBsdfAdapterWgsl(options?: PtMaterialOptions): string {
  if (options) return ptBsdfOpenPbrWgsl(options);
  return /* wgsl */ `
fn ptConductorKFromReflectance(r: vec3f) -> vec3f {
    let c = clamp(r, vec3f(0.0), vec3f(0.9999));
    return 2.0 * sqrt(c) / sqrt(1.0 - c);
}

fn ptBsdfFromMaterial(m: RtMaterial) -> PtBsdfParams {
    var p: PtBsdfParams;
    p.kind = PT_BSDF_BLACK;
    p.ior = 1.0;
    p.alpha = vec2f(0.0);
    p.reflectance = vec3f(0.0);
    p.eta = vec3f(0.0);
    p.k = vec3f(0.0);
    let a = m.specularRoughness * m.specularRoughness;
    if (m.transmissionWeight > 0.5) {
        p.kind = PT_BSDF_DIELECTRIC;
        p.alpha = vec2f(a);
        p.ior = m.ior;
        return p;
    }
    if (m.metalness > 0.5) {
        p.kind = PT_BSDF_CONDUCTOR;
        p.alpha = vec2f(a);
        if (m.conductorEtaK) {
            p.eta = m.conductorEta;
            p.k = m.conductorK;
        } else {
            p.eta = vec3f(1.0);
            p.k = ptConductorKFromReflectance(m.baseColor * m.baseWeight);
        }
        return p;
    }
    let r = m.baseColor * m.baseWeight;
    if (max(r.x, max(r.y, r.z)) > 0.0) {
        p.kind = PT_BSDF_DIFFUSE;
        p.reflectance = r;
    }
    return p;
}
`;
}

// ---------------------------------------------------------------------------------------------------------------
// OpenPBR mode

/** Which lobes a scene needs; each `false` keeps its code out of the kernel (`ptMaterialFeatures` derives them from the material table). */
export interface PtMaterialFeatures {
  /** Some material has `0 < metalness < 1`: metal or dielectric chosen stochastically per vertex. */
  mix: boolean;
  /** Some transmissive material is thin-walled: `PT_BSDF_THIN`. */
  thin: boolean;
  /** Some opaque dielectric has a specular lobe or a coat: `PT_BSDF_LAYERED` (`ptLayeredWgsl` is appended to the kernel). */
  layered: boolean;
  /** Some thick transmissive material has `transmissionDepth > 0`: Beer's law along the path inside the mesh. */
  medium: boolean;
  /** P2.23: some metal has a coat (`coatWeight > 0.5`, `coatIor != 1`): the layered BSDF gets its conductor bottom layer (implies `layered`). Optional so older literals still type. */
  coatedMetal?: boolean;
  /** P2.26: some metal has `specularAnisotropy > 0`: its GGX alphas split along the tangent (`ptOpenPbrAlpha`) and the integrator builds the frame from the vertex tangent (`ptFrameFromTangent`); needs a surface layout with tangents. Optional like `coatedMetal`. */
  aniso?: boolean;
}

export interface PtMaterialOptions {
  features: PtMaterialFeatures;
  /** Layered BSDF settings (the `ptLayeredWgsl` options; the kernel compiles one variant). Default `layering: 'albedo'`. */
  layered?: PtLayeredOptions;
  /** Layer thickness in pbrt's `coateddiffuse` sense (default 1e-6: transparent, what the reference exporters write; pbrt's own default is 0.01). Stochastic only. */
  layerThickness?: number;
  /** Tint thin-walled transmission by `transmissionColor` (default true). pbrt and Mitsuba `thindielectric` cannot tint. */
  transmissionTint?: boolean;
}

export const PT_NO_FEATURES: PtMaterialFeatures = { mix: false, thin: false, layered: false, medium: false };

/**
 * The features a material table needs in OpenPBR mode (union over the records; a record the scene never hits still counts, which only
 * costs kernel size). Mirrors the branches of `ptBsdfFromMaterial` with options.
 */
export function ptMaterialFeatures(materials: readonly RtMaterial[]): PtMaterialFeatures {
  const f: PtMaterialFeatures = { mix: false, thin: false, layered: false, medium: false };
  for (const m of materials) {
    if (m.metalness > 0 && m.metalness < 1) f.mix = true;
    if (m.metalness > 0 && m.specularAnisotropy > 0) f.aniso = true; // the key exists only when true, as coatedMetal
    if (m.metalness > 0 && m.coatWeight > 0.5 && m.coatIor !== 1) {
      f.coatedMetal = true; // the key exists only when true: scenes without coated metals keep their feature object
      f.layered = true;
    }
    const dielectricEnd = m.metalness < 1;
    if (!dielectricEnd) continue;
    if (m.transmissionWeight > 0.5) {
      if (m.thinWalled) f.thin = true;
      else if (m.transmissionDepth > 0) f.medium = true;
    } else if ((m.coatWeight > 0.5 || m.specularWeight > 0) && (m.coatWeight > 0.5 ? m.coatIor : m.ior) !== 1) f.layered = true;
  }
  return f;
}

/**
 * Cosine-weighted average Fresnel reflectance of a dielectric interface seen from inside (relative IOR `ior`, total internal reflection
 * included): Egan and Hilgeman's polynomial for the diffuse Fresnel reflectance, -1.440/n^2 + 0.710/n + 0.668 + 0.0636 n. The numeric value is
 * `ptInternalReflectance` (layered.ts); the fit is within 0.01 for ior 1.05-2.5 (tested). Only `albedoBase: 'internal'` reads it.
 */
export function ptInternalReflectanceFit(ior: number): number {
  if (ior <= 1) return 0;
  return Math.min(1, Math.max(0, -1.44 / (ior * ior) + 0.71 / ior + 0.668 + 0.0636 * ior));
}

/**
 * OpenPBR 1.1.2 GGX alphas of the specular lobe along the tangent (x) and bitangent (y): `alpha_x = r^2 sqrt(2 / (1 + (1 - a)^2))`,
 * `alpha_y = (1 - a) alpha_x` for roughness `r`, anisotropy `a` (a = 0 gives r^2 on both axes exactly). The same pair the pbrt and Mitsuba exporters write.
 */
export function ptOpenPbrAlpha(roughness: number, anisotropy: number): readonly [number, number] {
  if (!(anisotropy > 0)) return [roughness * roughness, roughness * roughness];
  const ax = roughness * roughness * Math.sqrt(2 / (1 + (1 - anisotropy) * (1 - anisotropy)));
  return [ax, (1 - anisotropy) * ax];
}

/** Default layer thickness of the OpenPBR adapter (transparent layer, see `PtMaterialOptions.layerThickness`). */
export const PT_DEFAULT_LAYER_THICKNESS = 1e-6;

/** Absorption coefficient (per metre, linear RGB) of a thick transmissive material's interior: Beer-Lambert, `-ln(transmissionColor) / transmissionDepth`; 0 = none. */
export function ptMediumSigma(m: RtMaterial): PtVec3 {
  if (m.thinWalled || !(m.transmissionDepth > 0)) return [0, 0, 0];
  return [0, 1, 2].map((i) => -Math.log(Math.max(m.transmissionColor[i] as number, 1e-6)) / m.transmissionDepth) as unknown as PtVec3;
}

/** The interior medium after a transmission event: the material's own absorber when the ray enters (`woZ > 0`, it came from outside), none when it exits. Twin of the integrator's update (no nesting: a ray inside one medium that enters another replaces it). */
export function ptMediumAfterTransmission(m: RtMaterial, woZ: number): PtVec3 {
  return woZ > 0 ? ptMediumSigma(m) : [0, 0, 0];
}

/** TS twin of the WGSL `ptLayeredOf`: the layered parameters packed in a `PT_BSDF_LAYERED` `PtBsdfParams`. */
export function ptLayeredFromBsdf(p: PtBsdfParams): PtLayeredParams {
  if (p.k[2] > 0.5) {
    return { baseColor: [0, 0, 0], ior: p.ior, alpha: p.alpha, specularWeight: 1, specularColor: [1, 1, 1], thickness: p.k[0], internalReflectance: 0, baseConductor: { eta: p.reflectance, k: p.eta, alpha: p.k[1] } };
  }
  return { baseColor: p.reflectance, ior: p.ior, alpha: p.alpha, specularWeight: 1, specularColor: p.eta, thickness: p.k[0], internalReflectance: p.k[1] };
}

const BLACK_PARAMS = (): PtBsdfParams => ({ kind: PT_BSDF_BLACK, reflectance: [0, 0, 0], eta: [0, 0, 0], k: [0, 0, 0], alpha: [0, 0], ior: 1 });

/** OpenPBR-mode adapter (rules in the file comment). `u` in [0, 1) is the vertex's metal/dielectric choice (used with `features.mix`). */
export function ptBsdfFromOpenPbr(m: RtMaterial, options: PtMaterialOptions, u: number): PtBsdfParams {
  const f = options.features;
  const p = BLACK_PARAMS();
  const a = alphaOf(m);
  const metal = f.mix ? u < m.metalness : m.metalness > 0.5;
  if (metal) {
    p.kind = PT_BSDF_CONDUCTOR;
    p.alpha = f.aniso ? ptOpenPbrAlpha(m.specularRoughness, m.specularAnisotropy) : [a, a];
    if (m.conductorEtaK) {
      p.eta = [...m.conductorEta] as PtVec3;
      p.k = [...m.conductorK] as PtVec3;
    } else {
      p.eta = [1, 1, 1];
      p.k = [0, 1, 2].map((i) => conductorKFromReflectance((m.baseColor[i] as number) * m.baseWeight)) as unknown as PtVec3;
    }
    if (f.coatedMetal && m.coatWeight > 0.5 && m.coatIor !== 1) {
      const ci = m.coatIor;
      const cr = m.coatRoughness * m.coatRoughness;
      p.kind = PT_BSDF_LAYERED;
      p.reflectance = [p.eta[0] / ci, p.eta[1] / ci, p.eta[2] / ci];
      p.eta = [p.k[0] / ci, p.k[1] / ci, p.k[2] / ci];
      p.ior = ci;
      p.k = [options.layerThickness ?? PT_DEFAULT_LAYER_THICKNESS, a, 1];
      p.alpha = [cr, cr];
    }
    return p;
  }
  if (m.transmissionWeight > 0.5) {
    if (f.thin && m.thinWalled) {
      p.kind = PT_BSDF_THIN;
      p.ior = m.ior;
      p.reflectance = (options.transmissionTint ?? true) ? ([...m.transmissionColor] as PtVec3) : [1, 1, 1];
      return p;
    }
    p.kind = PT_BSDF_DIELECTRIC;
    p.alpha = [a, a];
    p.ior = m.ior;
    return p;
  }
  const base: PtVec3 = [m.baseColor[0] * m.baseWeight, m.baseColor[1] * m.baseWeight, m.baseColor[2] * m.baseWeight];
  if (f.layered) {
    const coat = m.coatWeight > 0.5;
    const ior = coat ? m.coatIor : m.ior;
    if ((coat || m.specularWeight > 0) && ior !== 1) {
      const r = coat ? m.coatRoughness : m.specularRoughness;
      const s: PtVec3 = coat ? [1, 1, 1] : [m.specularColor[0] * m.specularWeight, m.specularColor[1] * m.specularWeight, m.specularColor[2] * m.specularWeight];
      p.kind = PT_BSDF_LAYERED;
      p.reflectance = base;
      p.ior = ior;
      p.alpha = [r * r, r * r];
      p.eta = s;
      p.k = [options.layerThickness ?? PT_DEFAULT_LAYER_THICKNESS, options.layered?.albedoBase === 'internal' ? ptInternalReflectanceFit(ior) : 0, 0];
      return p;
    }
  }
  if (Math.max(...base) > 0) {
    p.kind = PT_BSDF_DIFFUSE;
    p.reflectance = base;
  }
  return p;
}

/** The OpenPBR-mode WGSL twin: `ptBsdfFromMaterial(m, u)` and, with `features.medium`, `ptMediumSigma(m)`. */
export function ptBsdfOpenPbrWgsl(options: PtMaterialOptions): string {
  const f = options.features;
  const tint = options.transmissionTint ?? true;
  const thickness = options.layerThickness ?? PT_DEFAULT_LAYER_THICKNESS;
  const internal = options.layered?.albedoBase === 'internal';
  const num = (v: number): string => (Number.isInteger(v) ? `${v}.0` : String(v));
  return /* wgsl */ `
fn ptConductorKFromReflectance(r: vec3f) -> vec3f {
    let c = clamp(r, vec3f(0.0), vec3f(0.9999));
    return 2.0 * sqrt(c) / sqrt(1.0 - c);
}
${f.layered && internal ? `
fn ptInternalReflectanceFit(ior: f32) -> f32 {
    return clamp(-1.44 / (ior * ior) + 0.71 / ior + 0.668 + 0.0636 * ior, 0.0, 1.0);
}
` : ''}${f.medium ? `
fn ptMediumSigma(m: RtMaterial) -> vec3f {
    if (m.thinWalled || !(m.transmissionDepth > 0.0)) {
        return vec3f(0.0);
    }
    return -log(max(m.transmissionColor, vec3f(1.0e-6))) / m.transmissionDepth;
}
` : ''}${f.aniso ? `
fn ptOpenPbrAlpha(r: f32, a: f32) -> vec2f {
    if (!(a > 0.0)) {
        return vec2f(r * r);
    }
    let ax = r * r * sqrt(2.0 / (1.0 + (1.0 - a) * (1.0 - a)));
    return vec2f(ax, (1.0 - a) * ax);
}
` : ''}
fn ptBsdfFromMaterial(m: RtMaterial, u: f32) -> PtBsdfParams {
    var p: PtBsdfParams;
    p.kind = PT_BSDF_BLACK;
    p.ior = 1.0;
    p.alpha = vec2f(0.0);
    p.reflectance = vec3f(0.0);
    p.eta = vec3f(0.0);
    p.k = vec3f(0.0);
    let a = m.specularRoughness * m.specularRoughness;
    if (${f.mix ? 'u < m.metalness' : 'm.metalness > 0.5'}) {
        p.kind = PT_BSDF_CONDUCTOR;
        ${f.aniso ? 'p.alpha = ptOpenPbrAlpha(m.specularRoughness, m.specularAnisotropy);' : 'p.alpha = vec2f(a);'}
        if (m.conductorEtaK) {
            p.eta = m.conductorEta;
            p.k = m.conductorK;
        } else {
            p.eta = vec3f(1.0);
            p.k = ptConductorKFromReflectance(m.baseColor * m.baseWeight);
        }
${f.coatedMetal ? `        if (m.coatWeight > 0.5 && m.coatIor != 1.0) {
            let cr = m.coatRoughness * m.coatRoughness;
            p.kind = PT_BSDF_LAYERED;
            p.reflectance = p.eta / m.coatIor;
            p.eta = p.k / m.coatIor;
            p.ior = m.coatIor;
            p.k = vec3f(${num(thickness)}, a, 1.0);
            p.alpha = vec2f(cr);
        }
` : ''}        return p;
    }
    if (m.transmissionWeight > 0.5) {
${f.thin ? `        if (m.thinWalled) {
            p.kind = PT_BSDF_THIN;
            p.ior = m.ior;
            p.reflectance = ${tint ? 'm.transmissionColor' : 'vec3f(1.0)'};
            return p;
        }
` : ''}        p.kind = PT_BSDF_DIELECTRIC;
        p.alpha = vec2f(a);
        p.ior = m.ior;
        return p;
    }
    let r = m.baseColor * m.baseWeight;
${f.layered ? `    let coat = m.coatWeight > 0.5;
    let li = select(m.ior, m.coatIor, coat);
    if ((coat || m.specularWeight > 0.0) && li != 1.0) {
        let lr = select(m.specularRoughness, m.coatRoughness, coat);
        p.kind = PT_BSDF_LAYERED;
        p.reflectance = r;
        p.ior = li;
        p.alpha = vec2f(lr * lr);
        p.eta = select(m.specularColor * m.specularWeight, vec3f(1.0), coat);
        p.k = vec3f(${num(thickness)}, ${internal ? 'ptInternalReflectanceFit(li)' : '0.0'}, 0.0);
        return p;
    }
` : ''}    if (max(r.x, max(r.y, r.z)) > 0.0) {
        p.kind = PT_BSDF_DIFFUSE;
        p.reflectance = r;
    }
    return p;
}
`;
}

/**
 * Kernel glue for `PT_BSDF_LAYERED` (append after `ptLayeredWgsl`): the layered kind is evaluated by `ptLayeredEval/Sample`, every other kind
 * by `ptBsdfEval/Sample`. The stochastic walk needs a seed: the integrator sets the private `ptLayerSeed` once per vertex (a hash of pixel,
 * sample and vertex); the walk mixes the directions in itself. `PtBsdfParams` is reused for the layered parameters (see `PT_BSDF_LAYERED`).
 */
export function ptLayeredDispatchWgsl(coatedMetal = false): string {
  return /* wgsl */ `
const PT_BSDF_LAYERED: u32 = 5u;
var<private> ptLayerSeed: u32;
${coatedMetal ? `fn ptLayeredOf(p: PtBsdfParams) -> PtLayeredParams {
    if (p.k.z > 0.5) {
        return PtLayeredParams(vec3f(0.0), p.ior, p.alpha, 1.0, vec3f(1.0), p.k.x, 0.0, p.reflectance, p.eta, p.k.y, true);
    }
    return PtLayeredParams(p.reflectance, p.ior, p.alpha, 1.0, p.eta, p.k.x, p.k.y, vec3f(0.0), vec3f(0.0), 0.0, false);
}` : `fn ptLayeredOf(p: PtBsdfParams) -> PtLayeredParams {
    return PtLayeredParams(p.reflectance, p.ior, p.alpha, 1.0, p.eta, p.k.x, p.k.y);
}`}
fn ptEval(p: PtBsdfParams, wo: vec3f, wi: vec3f) -> PtBsdfEval {
    if (p.kind == PT_BSDF_LAYERED) {
        return ptLayeredEval(ptLayeredOf(p), wo, wi, ptLayerSeed);
    }
    return ptBsdfEval(p, wo, wi);
}
fn ptSample(p: PtBsdfParams, wo: vec3f, u: vec3f) -> PtBsdfSample {
    if (p.kind == PT_BSDF_LAYERED) {
        return ptLayeredSample(ptLayeredOf(p), wo, u, ptLayerSeed);
    }
    return ptBsdfSample(p, wo, u);
}
`;
}
