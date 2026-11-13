/** Constructors from the bench material subset (`bench/scenes/schema.ts` MaterialDesc) to `PtBsdfParams`. */

import { roughnessToAlpha } from '@pmndrs/raytracer';
import { PT_BSDF_BLACK, PT_BSDF_CONDUCTOR, PT_BSDF_DIELECTRIC, PT_BSDF_DIFFUSE, type PtBsdfParams, type PtVec3 } from './bsdf.ts';

const ZERO: PtVec3 = [0, 0, 0];

export function ptDiffuse(reflectance: PtVec3): PtBsdfParams {
  return { kind: PT_BSDF_DIFFUSE, reflectance, eta: ZERO, k: ZERO, alpha: [0, 0], ior: 1 };
}

/** `roughness` is perceptual; alpha = roughness^2 on both axes (isotropic). */
export function ptConductor(eta: PtVec3, k: PtVec3, roughness: number): PtBsdfParams {
  const a = roughnessToAlpha(roughness);
  return { kind: PT_BSDF_CONDUCTOR, reflectance: ZERO, eta, k, alpha: [a, a], ior: 1 };
}

export function ptDielectric(ior: number, roughness: number): PtBsdfParams {
  const a = roughnessToAlpha(roughness);
  return { kind: PT_BSDF_DIELECTRIC, reflectance: ZERO, eta: ZERO, k: ZERO, alpha: [a, a], ior };
}

export function ptBlack(): PtBsdfParams {
  return { kind: PT_BSDF_BLACK, reflectance: ZERO, eta: ZERO, k: ZERO, alpha: [0, 0], ior: 1 };
}
