/**
 * Phase functions of a medium beyond one Henyey-Greenstein lobe (P5.v14, docs/internal/CLOUDS-HANDOFF.md item 2): a mixture of two lobes,
 * `(1 - w) * HG(gA) + w * Draine(gB, alpha)` (alpha 0 = a second HG lobe). TS twins of the WGSL in `volume-wgsl.ts`.
 *
 *  - `hg2`: two HG lobes (the clouds runtime's art default: g 0.2 at 60 %, 0.85 at 40 %; Karma's two-lobe volume shader).
 *  - `hgDraine`: HG plus a Draine lobe (Draine 2003, ApJ 598; alpha = 1 is Cornette-Shanks).
 *  - `mie`: the water-droplet fit of Jendersie and d'Eon 2023 ("An Approximate Mie Scattering Function for Fog and Cloud Rendering", SIGGRAPH Talks; supplemental
 *    section 3.1, Eqs. 7-10, large particles, diameter 5 to 50 um), resolved to `hgDraine` by `ptMieFit`. Note the constants 0.428934 and 0.665888 sit INSIDE the
 *    exponent (checked against the supplemental text and Blender Cycles' `phase_mie_fitted_parameters`). No clamp of the forward peak (V298).
 *
 * Convention: `cosTheta = dot(wo, wi)` with wo towards the previous vertex, so forward scattering is cosTheta = -1 (pbrt's, as `hgPhase`). The papers write
 * u = cos(scattering angle) = -cosTheta; the lobe functions below take that `u`.
 */

export type PtPhase =
  | { type: 'hg'; g: number }
  /** `(1 - w2) HG(g1) + w2 HG(g2)`. */
  | { type: 'hg2'; g1: number; g2: number; w2: number }
  /** `(1 - wD) HG(gHG) + wD Draine(gD, alpha)`. */
  | { type: 'hgDraine'; gHG: number; gD: number; alpha: number; wD: number }
  /** Water droplets of mean diameter `diameter` micrometres (5 to 50): `ptMieFit`. */
  | { type: 'mie'; diameter: number };

/** The canonical two-lobe form every phase resolves to: lobe A = HG(gA) with weight 1 - w, lobe B = Draine(gB, alpha) with weight w (alpha 0: HG). */
export interface PtPhaseLobes {
  gA: number;
  gB: number;
  w: number;
  alpha: number;
}

/** Jendersie and d'Eon 2023, supplemental Eqs. 7-10 (large particles, 5 <= d <= 50 um). */
export function ptMieFit(diameter: number): { gHG: number; gD: number; alpha: number; wD: number } {
  const d = diameter;
  if (!(d >= 5 && d <= 50)) throw new RangeError(`mie phase: diameter ${d} um is outside the large-particle fit [5, 50] (the small-particle fits are not ported)`);
  return {
    gHG: Math.exp(-0.0990567 / (d - 1.67154)),
    gD: Math.exp(-2.20679 / (d + 3.91029) - 0.428934),
    alpha: Math.exp(3.62489 - 8.29288 / (d + 5.52825)),
    wD: Math.exp(-0.599085 / (d - 0.641583) - 0.665888),
  };
}

export function ptPhaseLobes(p: PtPhase): PtPhaseLobes {
  switch (p.type) {
    case 'hg':
      return { gA: p.g, gB: 0, w: 0, alpha: 0 };
    case 'hg2':
      return { gA: p.g1, gB: p.g2, w: p.w2, alpha: 0 };
    case 'hgDraine':
      return { gA: p.gHG, gB: p.gD, w: p.wD, alpha: p.alpha };
    case 'mie': {
      const f = ptMieFit(p.diameter);
      return { gA: f.gHG, gB: f.gD, w: f.wD, alpha: f.alpha };
    }
  }
}

export function ptPhaseValidate(l: PtPhaseLobes, what = 'phase'): void {
  if (!(l.gA > -1 && l.gA < 1) || !(l.gB > -1 && l.gB < 1)) throw new RangeError(`${what}: lobe g must be in (-1, 1) (got ${l.gA}, ${l.gB})`);
  if (!(l.w >= 0 && l.w <= 1)) throw new RangeError(`${what}: lobe weight ${l.w} is outside [0, 1]`);
  if (!(l.alpha >= 0 && Number.isFinite(l.alpha))) throw new RangeError(`${what}: Draine alpha ${l.alpha} must be finite and >= 0`);
  // the WGSL Draine sampler is the closed-form inversion only, which divides by g (the Cardano g = 0 case lives in the TS twin)
  if (l.w > 0 && l.alpha >= 1e-3 && Math.abs(l.gB) < 0.01) throw new RangeError(`${what}: a Draine lobe needs |g| >= 0.01 (got ${l.gB}); use alpha 0 for an HG lobe`);
}

const INV_4PI = 1 / (4 * Math.PI);

/** Draine's phase function of u = cos(scattering angle) (forward u = 1), normalised over the sphere; alpha 0 is HG. */
export function draine(u: number, g: number, alpha: number): number {
  const fac = 1 + g * g - 2 * g * u;
  return (INV_4PI * (1 - g * g) * (1 + alpha * u * u)) / ((1 + (alpha * (1 + 2 * g * g)) / 3) * fac * Math.sqrt(Math.max(fac, 1e-300)));
}

/** Mean cosine of a Draine lobe (Draine 2003, Eq. 6): g (1 + alpha (3 + 2 g^2) / 5) / (1 + alpha (1 + 2 g^2) / 3). */
export function draineMeanCosine(g: number, alpha: number): number {
  return (g * (1 + (alpha * (3 + 2 * g * g)) / 5)) / (1 + (alpha * (1 + 2 * g * g)) / 3);
}

/** The mixture's density of the angle between wo and wi (our convention: forward is cosTheta = -1). */
export function phaseEval(l: PtPhaseLobes, cosTheta: number): number {
  const u = -cosTheta;
  return (1 - l.w) * draine(u, l.gA, 0) + (l.w > 0 ? l.w * draine(u, l.gB, l.alpha) : 0);
}

export function phaseMeanCosine(l: PtPhaseLobes): number {
  return (1 - l.w) * l.gA + l.w * draineMeanCosine(l.gB, l.alpha);
}

/** 1 - u of an HG sample (u = cosine with the forward direction), written so it does not cancel near the forward peak (g 0.995: 1 - u is about 1e-5). */
export function hgSampleOneMinusU(g: number, xi: number): number {
  if (Math.abs(g) < 1e-3) return 2 - 2 * xi;
  // t = (1 - g^2) / (1 - g + 2 g xi), u = (1 + g^2 - t^2) / (2 g)  (Witt 1977; the supplemental's Eq. 3), so
  // 1 - u = (t - (1 - g)) (t + (1 - g)) / (2 g) with t - (1 - g) = 2 g (1 - g) (1 - xi) / (1 - g + 2 g xi)
  const den = 1 - g + 2 * g * xi;
  const t = (1 - g * g) / den;
  return ((1 - g) * (1 - xi) * (t + 1 - g)) / den;
}

/** u of a Draine sample: the exact CDF inversion of the supplemental (section 2.2, Eq. 6). Needs alpha > 0 and |g| > 0. */
export function draineSampleU(g: number, alpha: number, xi: number): number {
  const a = alpha;
  const g2 = g * g, g4 = g2 * g2;
  const t0 = a - a * g2;
  const t1 = a * g4 - a;
  const t2 = -3 * (4 * (g4 - g2) + t1 * (1 + g2));
  const t3 = g * (2 * xi - 1);
  const t4 = 3 * g2 * (1 + t3) + a * (2 + g2 * (1 + (1 + 2 * g2) * t3));
  const t5 = t0 * (t1 * t2 + t4 * t4) + t1 * t1 * t1;
  const t6 = t0 * 4 * (g4 - g2);
  const t7 = Math.cbrt(t5 + Math.sqrt(t5 * t5 - t6 * t6 * t6));
  const t8 = (2 * (t1 + t6 / t7 + t7)) / t0;
  const t9 = Math.sqrt(6 * (1 + g2) + t8);
  const r = Math.sqrt(6 * (1 + g2) - t8 + (8 * t4) / (t0 * t9)) - t9;
  return g / 2 + 1 / (2 * g) - (r * r) / (8 * g);
}

/** Sample the mixture: `u` = (lobe pick + rescaled lobe variable, azimuth). Returns 1 - u (u the cosine with the FORWARD direction -wo) and the azimuth. */
export function phaseSample(l: PtPhaseLobes, u: readonly [number, number]): { oneMinusU: number; phi: number } {
  let xi = u[0];
  let g = l.gA, alpha = 0;
  if (xi < l.w) {
    xi = xi / l.w;
    g = l.gB;
    alpha = l.alpha;
  } else {
    xi = l.w > 0 ? (xi - l.w) / (1 - l.w) : xi;
  }
  xi = Math.min(xi, 1 - 2 ** -24);
  let x: number;
  if (alpha < 1e-3) x = hgSampleOneMinusU(g, xi);
  else if (Math.abs(g) < 1e-3) x = 1 - rayleighLikeU(alpha, xi);
  else x = 1 - Math.min(1, Math.max(-1, draineSampleU(g, alpha, xi)));
  return { oneMinusU: Math.min(2, Math.max(0, x)), phi: 2 * Math.PI * u[1] };
}

/** g = 0 Draine (`(1 + alpha u^2)`, normalised): the depressed cubic of its CDF solved by Cardano (Cycles' special case, after Frisvad 2011). */
export function rayleighLikeU(alpha: number, xi: number): number {
  // CDF: (u + alpha u^3 / 3 + 1 + alpha / 3) / (2 (1 + alpha / 3)) = xi  ->  u^3 + (3 / alpha) u + b = 0 with b = (3 + alpha) (1 - 2 xi) / alpha
  const p = 3 / alpha;
  const b = ((3 + alpha) * (1 - 2 * xi)) / alpha;
  const q = b / 2;
  const s = Math.sqrt(q * q + (p * p * p) / 27);
  return Math.cbrt(-q + s) + Math.cbrt(-q - s);
}
