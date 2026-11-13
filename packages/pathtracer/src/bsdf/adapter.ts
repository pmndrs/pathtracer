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
 */
import type { RtMaterial } from '@pmndrs/raytracer';
import { PT_BSDF_BLACK, PT_BSDF_CONDUCTOR, PT_BSDF_DIELECTRIC, PT_BSDF_DIFFUSE, type PtBsdfParams, type PtVec3 } from './bsdf.ts';

/** k that gives a conductor with eta = 1 the normal-incidence reflectance `r` (clamped just below 1). */
export function conductorKFromReflectance(r: number): number {
  const c = Math.min(Math.max(r, 0), 0.9999);
  return (2 * Math.sqrt(c)) / Math.sqrt(1 - c);
}

/** GGX alpha from a perceptual roughness, as `rtMaterialAlpha` (`roughness^2`). */
const alphaOf = (m: RtMaterial): number => m.specularRoughness * m.specularRoughness;

export function ptBsdfFromMaterial(m: RtMaterial): PtBsdfParams {
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
export function ptBsdfAdapterWgsl(): string {
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
