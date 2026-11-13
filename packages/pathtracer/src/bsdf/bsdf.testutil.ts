/** Test and bench helpers for the BSDF twin: seeded RNG, brute-force integration of eval, MC estimates, chi-square. */

import { PT_FLAG_REFLECTION, PT_FLAG_TRANSMISSION, ptBsdfEval, ptBsdfSample, ptFrameFromNormal, ptRefract, type PtBsdfOptions, type PtBsdfParams, type PtVec3 } from './bsdf.ts';

/** mulberry32: small seeded PRNG in [0, 1). */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function dirFromAngles(thetaDeg: number, phiDeg = 0): PtVec3 {
  const t = (thetaDeg * Math.PI) / 180;
  const p = (phiDeg * Math.PI) / 180;
  return [Math.sin(t) * Math.cos(p), Math.sin(t) * Math.sin(p), Math.cos(t)];
}

export interface Albedo {
  /** integral of f |cos wi| over the wo-side hemisphere (reflection) per channel */
  refl: [number, number, number];
  /** same over the opposite hemisphere (transmission) */
  trans: [number, number, number];
  /** integral of the pdf over each hemisphere */
  pdfRefl: number;
  pdfTrans: number;
}

/**
 * Brute-force integral of eval over the sphere (never calls sample). Two passes in polar coordinates around a lobe
 * centre (radius warp psi = pi t^3, dense near the centre so narrow GGX lobes are resolved): pass A around the mirror
 * direction keeps the wo-side hemisphere, pass B around the refracted direction keeps the other one.
 */
export function bruteForceAlbedo(p: PtBsdfParams, wo: PtVec3, options: PtBsdfOptions = {}, nr = 400, nphi = 360): Albedo {
  const out: Albedo = { refl: [0, 0, 0], trans: [0, 0, 0], pdfRefl: 0, pdfTrans: 0 };
  const side = wo[2] >= 0 ? 1 : -1;
  const mirror: PtVec3 = [-wo[0], -wo[1], wo[2]];
  const refr = ptRefract(wo, [0, 0, 1], p.ior);
  const through: PtVec3 = refr ? refr.wt : [-wo[0], -wo[1], -wo[2]];
  for (const pass of [0, 1] as const) {
    const c = pass === 0 ? mirror : through;
    const frame = ptFrameFromNormal(c);
    for (let i = 0; i < nr; i++) {
      const t = (i + 0.5) / nr;
      const psi = Math.PI * t * t * t;
      const dpsi = (3 * Math.PI * t * t) / nr;
      const sp = Math.sin(psi);
      const cp = Math.cos(psi);
      const w = sp * dpsi * ((2 * Math.PI) / nphi);
      for (let j = 0; j < nphi; j++) {
        const chi = ((j + 0.5) / nphi) * 2 * Math.PI;
        const cc = Math.cos(chi) * sp;
        const ss = Math.sin(chi) * sp;
        const wi: PtVec3 = [
          c[0] * cp + frame.t[0] * cc + frame.b[0] * ss,
          c[1] * cp + frame.t[1] * cc + frame.b[1] * ss,
          c[2] * cp + frame.t[2] * cc + frame.b[2] * ss,
        ];
        const sameSide = wi[2] * side > 0;
        if (pass === 0 ? !sameSide : sameSide) continue;
        const e = ptBsdfEval(p, wo, wi, options);
        const target = pass === 0 ? out.refl : out.trans;
        const k = Math.abs(wi[2]) * w;
        target[0] += e.f[0] * k;
        target[1] += e.f[1] * k;
        target[2] += e.f[2] * k;
        if (pass === 0) out.pdfRefl += e.pdf * w;
        else out.pdfTrans += e.pdf * w;
      }
    }
  }
  return out;
}

export interface McEstimate {
  refl: [number, number, number];
  trans: [number, number, number];
  /** standard error per channel of the total (refl + trans) estimate */
  se: [number, number, number];
  /** fraction of samples that succeeded (non-delta lobes: compare with the integral of the pdf) */
  success: number;
  successSe: number;
}

/** Mean of f |cos| / pdf over `n` samples (failures count as 0), split by reflection/transmission flag. */
export function mcAlbedo(p: PtBsdfParams, wo: PtVec3, n: number, seed: number, options: PtBsdfOptions = {}): McEstimate {
  const rand = rng(seed);
  const sumR = [0, 0, 0];
  const sumT = [0, 0, 0];
  const sumSq = [0, 0, 0];
  let ok = 0;
  for (let i = 0; i < n; i++) {
    const s = ptBsdfSample(p, wo, [rand(), rand(), rand()], options);
    if (s.flags === 0) continue;
    ok++;
    const k = Math.abs(s.wi[2]) / s.pdf;
    for (let c = 0; c < 3; c++) {
      const v = (s.f[c] as number) * k;
      if (s.flags & PT_FLAG_REFLECTION) sumR[c] = (sumR[c] as number) + v;
      else if (s.flags & PT_FLAG_TRANSMISSION) sumT[c] = (sumT[c] as number) + v;
      sumSq[c] = (sumSq[c] as number) + v * v;
    }
  }
  const mean = (a: number[]) => a.map((x) => x / n) as [number, number, number];
  const total = [0, 1, 2].map((c) => ((sumR[c] as number) + (sumT[c] as number)) / n);
  const se = [0, 1, 2].map((c) => Math.sqrt(Math.max(0, (sumSq[c] as number) / n - (total[c] as number) ** 2) / n)) as [number, number, number];
  const q = ok / n;
  return { refl: mean(sumR), trans: mean(sumT), se, success: q, successSe: Math.sqrt((q * (1 - q)) / n) };
}

// ---- chi-square ---------------------------------------------------------------------------------------------

function lnGamma(x: number): number {
  const c = [76.1800917294715, -86.5053203294168, 24.0140982408309, -1.231739572450155, 0.001208650973866179, -0.000005395239384953];
  let y = x;
  let tmp = x + 5.5;
  tmp -= (x + 0.5) * Math.log(tmp);
  let ser = 1.000000000190015;
  for (const cj of c) ser += cj / ++y;
  return -tmp + Math.log((2.506628274631 * ser) / x);
}

/** Regularised upper incomplete gamma Q(a, x). */
function gammaQ(a: number, x: number): number {
  if (x <= 0) return 1;
  if (x < a + 1) {
    let sum = 1 / a;
    let del = sum;
    let ap = a;
    for (let n = 0; n < 1000; n++) {
      ap++;
      del *= x / ap;
      sum += del;
      if (Math.abs(del) < Math.abs(sum) * 1e-14) break;
    }
    return 1 - sum * Math.exp(-x + a * Math.log(x) - lnGamma(a));
  }
  let b = x + 1 - a;
  let c = 1 / 1e-300;
  let d = 1 / b;
  let h = d;
  for (let i = 1; i < 1000; i++) {
    const an = -i * (i - a);
    b += 2;
    d = an * d + b;
    if (Math.abs(d) < 1e-300) d = 1e-300;
    c = b + an / c;
    if (Math.abs(c) < 1e-300) c = 1e-300;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < 1e-14) break;
  }
  return Math.exp(-x + a * Math.log(x) - lnGamma(a)) * h;
}

export function chiSquarePValue(stat: number, dof: number): number {
  return gammaQ(dof / 2, stat / 2);
}

export interface ChiSquareResult {
  stat: number;
  dof: number;
  p: number;
  bins: number;
  /** expected mass that the pdf leaves unaccounted (failed samples) */
  failureProb: number;
}

/**
 * pbrt/Mitsuba style goodness of fit of sampled directions against the pdf: bins uniform in (cos theta, phi) over the
 * sphere (so the hemisphere boundary is a bin edge), expected counts from a midpoint sub-grid integral of `ptBsdfEval`'s
 * pdf (of `pdfParams`, default `p`: pass different parameters to prove the test rejects a wrong pdf), one extra bin for failed samples (expected N (1 - sum)). Bins with expected < 5 are pooled.
 */
export function chiSquareTest(p: PtBsdfParams, wo: PtVec3, n: number, seed: number, options: PtBsdfOptions = {}, pdfParams: PtBsdfParams = p, nc = 32, nphi = 32, sub = 8): ChiSquareResult {
  const rand = rng(seed);
  const observed = new Float64Array(nc * nphi);
  let failures = 0;
  for (let i = 0; i < n; i++) {
    const s = ptBsdfSample(p, wo, [rand(), rand(), rand()], options);
    if (s.flags === 0) {
      failures++;
      continue;
    }
    const cb = Math.min(nc - 1, Math.floor(((s.wi[2] + 1) / 2) * nc));
    let phi = Math.atan2(s.wi[1], s.wi[0]);
    if (phi < 0) phi += 2 * Math.PI;
    const pb = Math.min(nphi - 1, Math.floor((phi / (2 * Math.PI)) * nphi));
    observed[cb * nphi + pb] = (observed[cb * nphi + pb] as number) + 1;
  }
  const expected = new Float64Array(nc * nphi);
  let expectedSum = 0;
  for (let cb = 0; cb < nc; cb++) {
    for (let pb = 0; pb < nphi; pb++) {
      let acc = 0;
      for (let a = 0; a < sub; a++) {
        const z = -1 + ((cb + (a + 0.5) / sub) * 2) / nc;
        const sz = Math.sqrt(Math.max(0, 1 - z * z));
        for (let b = 0; b < sub; b++) {
          const phi = ((pb + (b + 0.5) / sub) * 2 * Math.PI) / nphi;
          acc += ptBsdfEval(pdfParams, wo, [sz * Math.cos(phi), sz * Math.sin(phi), z], options).pdf;
        }
      }
      // dω = dz dφ; each sub-cell has area (2/nc/sub) * (2π/nphi/sub)
      const e = acc * ((2 / nc / sub) * ((2 * Math.PI) / nphi / sub)) * n;
      expected[cb * nphi + pb] = e;
      expectedSum += e;
    }
  }
  const failExpected = Math.max(0, n - expectedSum);
  let stat = 0;
  let bins = 0;
  let pooledObs = 0;
  let pooledExp = 0;
  const add = (o: number, e: number) => {
    stat += ((o - e) * (o - e)) / e;
    bins++;
  };
  for (let i = 0; i < expected.length; i++) {
    const e = expected[i] as number;
    if (e < 5) {
      pooledObs += observed[i] as number;
      pooledExp += e;
    } else add(observed[i] as number, e);
  }
  pooledObs += failures;
  pooledExp += failExpected;
  if (pooledExp >= 5) add(pooledObs, pooledExp);
  else if (pooledObs > 0 && pooledExp < 5) {
    // Observations where almost none were expected: fold into the statistic with a floor of 5 (a gross mismatch shows).
    add(pooledObs, Math.max(pooledExp, 5));
  }
  const dof = bins - 1;
  return { stat, dof, p: chiSquarePValue(stat, dof), bins, failureProb: failExpected / n };
}
