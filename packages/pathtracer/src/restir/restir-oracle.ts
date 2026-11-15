/**
 * CPU oracle of the reservoir arithmetic in `restir-wgsl.ts` (initial RIS and `ptRsMerge`), on a discrete sample space so the exact answer is a sum. The WGSL works on
 * (light, u) samples evaluated per pixel; here a sample is an index y, a domain j has a target `phat_j(y)` and every candidate is drawn from the same source pmf `q(y)`.
 * Used by `restir.test.ts` to check that the unbiased combine reproduces `sum_y f(y)` and that the biased one does not when the supports differ.
 */

export interface OracleReservoir {
  y: number; // -1 = empty
  w: number; // contribution weight W
  m: number; // confidence M
}

export interface OracleDomain {
  phat: (y: number) => number;
}

/** Streaming RIS with `count` candidates drawn from the pmf `q` (cumulative table), target `phat`. */
export function oracleInit(count: number, cdf: readonly number[], q: readonly number[], phat: (y: number) => number, rand: () => number): OracleReservoir {
  let wsum = 0;
  let sel = -1;
  let selPh = 0;
  for (let c = 0; c < count; c++) {
    const u = rand();
    let y = cdf.findIndex((v) => u < v);
    if (y < 0) y = cdf.length - 1;
    const ph = phat(y);
    if (!(ph > 0)) continue;
    const w = ph / (q[y] as number);
    wsum += w;
    if (rand() * wsum < w) {
      sel = y;
      selPh = ph;
    }
  }
  return { y: sel, w: sel >= 0 ? wsum / (count * selPh) : 0, m: count };
}

/** `ptRsMerge`: domain 0 is the destination. */
export function oracleMerge(domains: readonly OracleDomain[], rs: readonly OracleReservoir[], unbiased: boolean, rand: () => number, cap = Infinity): OracleReservoir {
  const n = rs.length;
  let wsum = 0;
  let mtot = 0;
  let sel = -1;
  let selPh = 0;
  for (let i = 0; i < n; i++) {
    const r = rs[i] as OracleReservoir;
    mtot += r.m;
    if (!(r.w > 0) || r.y < 0) continue;
    const ph = (domains[0] as OracleDomain).phat(r.y);
    if (!(ph > 0)) continue;
    let w = ph * r.w * r.m;
    if (unbiased) {
      let den = 0;
      let own = 0;
      for (let j = 0; j < n; j++) {
        const rj = rs[j] as OracleReservoir;
        if (!(rj.m > 0)) continue;
        const pj = j === 0 ? ph : (domains[j] as OracleDomain).phat(r.y);
        den += rj.m * pj;
        if (j === i) own = pj;
      }
      w = den > 0 ? ((r.m * own) / den) * ph * r.w : 0;
    }
    wsum += w;
    if (rand() * wsum < w) {
      sel = r.y;
      selPh = ph;
    }
  }
  if (!(selPh > 0 && wsum > 0)) return { y: -1, w: 0, m: Math.min(mtot, cap) };
  return { y: sel, w: unbiased ? wsum / selPh : wsum / (mtot * selPh), m: Math.min(mtot, cap) };
}

/** Deterministic xorshift in [0, 1). */
export function oracleRng(seed: number): () => number {
  let s = seed >>> 0 || 1;
  return () => {
    s ^= s << 13;
    s >>>= 0;
    s ^= s >>> 17;
    s ^= s << 5;
    s >>>= 0;
    return s / 4294967296;
  };
}
