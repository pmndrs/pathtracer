/**
 * Hero-wavelength sampling and the spectrum -> RGB estimator (P5): TS twins of the kernel's `ptSpecLambda*` / `ptSpecToRgb`, plus the dispersion law.
 *
 * Wavelengths: pbrt-v4's `SampleVisibleWavelengths` (Apache-2.0, THIRD_PARTY): the density `0.0039398042 / cosh^2(0.0072 (l - 538))` on [360, 830] nm, a
 * closed-form inverse CDF. A path carries `PT_SPEC_LANES` = 3 wavelengths: the hero `l0 = F(u)` and `F(fract(u + k / 3))` (a stratified rotation of ONE
 * random number, so each lane is marginally distributed by the density; pbrt uses 4 lanes, we use the 3 lanes of the existing vec3 path state: DIVERGES V250).
 *
 * Estimator: `XYZ = (1/3) sum_k L(l_k) cmf(l_k) / p(l_k)`; RGB = `SPEC_XYZ_TO_RGB * XYZ`. After a dispersive event the secondary lanes are terminated:
 * their throughput becomes 0 and the hero's is multiplied by 3, so the average above is the hero's single-wavelength estimate (pbrt's `TerminateSecondary`).
 */
import { cieAt, SPEC_LAMBDA_MAX, SPEC_LAMBDA_MIN, xyzToRgb, SPEC_YD, type Vec3 } from './rgb2spec.ts';

export const PT_SPEC_LANES = 3;

/** pbrt `SampleVisibleWavelengths(u)`: nm. */
export function visibleLambda(u: number): number {
  return 538 - 138.888889 * Math.atanh(0.85691062 - 1.82750197 * u);
}

/** pbrt `VisibleWavelengthsPDF(l)`: 1/nm, zero outside [360, 830]. */
export function visiblePdf(lambda: number): number {
  if (lambda < SPEC_LAMBDA_MIN || lambda > SPEC_LAMBDA_MAX) return 0;
  const c = Math.cosh(0.0072 * (lambda - 538));
  return 0.0039398042 / (c * c);
}

/** The path's wavelengths for one uniform number in [0, 1). */
export function pathLambdas(u: number): Vec3 {
  return [0, 1, 2].map((k) => visibleLambda((u + k / PT_SPEC_LANES) % 1)) as Vec3;
}

/** Per-lane quantities the kernel keeps for a path. */
export interface LaneData {
  lambda: Vec3;
  pdf: Vec3;
  /** D65 / Yd at each lane (the illuminant of an RGB emitter). */
  d65: Vec3;
  /** cmf at each lane, one Vec3 per XYZ component. */
  x: Vec3;
  y: Vec3;
  z: Vec3;
}

export function laneData(u: number): LaneData {
  const lambda = pathLambdas(u);
  const rows = lambda.map((l) => cieAt(l));
  return {
    lambda,
    pdf: lambda.map(visiblePdf) as Vec3,
    d65: rows.map((r) => r[3] / SPEC_YD) as Vec3,
    x: rows.map((r) => r[0]) as Vec3,
    y: rows.map((r) => r[1]) as Vec3,
    z: rows.map((r) => r[2]) as Vec3,
  };
}

/** Radiance of the three lanes -> linear Rec.709 RGB (one sample of the estimator). */
export function laneRadianceToRgb(L: Vec3, d: LaneData): Vec3 {
  const xyz = [0, 0, 0] as [number, number, number];
  for (let k = 0; k < PT_SPEC_LANES; k++) {
    const w = (L[k] as number) / ((d.pdf[k] as number) * PT_SPEC_LANES);
    xyz[0] += w * (d.x[k] as number);
    xyz[1] += w * (d.y[k] as number);
    xyz[2] += w * (d.z[k] as number);
  }
  return xyzToRgb(xyz);
}

// ---- dispersion ----

/** Wavelengths of the Abbe number: Fraunhofer d, F and C lines (nm). */
export const ABBE_LINES = { d: 587.5618, F: 486.1327, C: 656.2725 } as const;

/** KHR_materials_dispersion `dispersion` = 20 / V_d. */
export const dispersionFromAbbe = (abbe: number): number => 20 / abbe;
export const abbeFromDispersion = (dispersion: number): number => 20 / dispersion;
/** The largest dispersion the material record stores (unorm16 of `dispersion / PT_DISPERSION_MAX`; the KHR default range is 0..~2). */
export const PT_DISPERSION_MAX = 8;

/**
 * Cauchy law `n(l) = A + B / l^2` (l in um) through `ior` at the d line with Abbe number `abbe = 20 / dispersion`: `B = (n_d - 1) / V_d / (1/l_F^2 - 1/l_C^2)`.
 * Returns [A, B] with `B` in um^2.
 */
export function cauchyCoefficients(ior: number, dispersion: number): [number, number] {
  if (!(dispersion > 0)) return [ior, 0];
  const um = (nm: number): number => nm * 1e-3;
  const vd = abbeFromDispersion(dispersion);
  const b = (ior - 1) / vd / (1 / um(ABBE_LINES.F) ** 2 - 1 / um(ABBE_LINES.C) ** 2);
  const a = ior - b / um(ABBE_LINES.d) ** 2;
  return [a, b];
}

/** IOR at `lambdaNm` of a dielectric of index `ior` at 587.56 nm and KHR `dispersion` (0 = none). TS twin of `ptSpecIor`. */
export function iorAt(ior: number, dispersion: number, lambdaNm: number): number {
  const [a, b] = cauchyCoefficients(ior, dispersion);
  const um = lambdaNm * 1e-3;
  return a + b / (um * um);
}
