// V4.2: forced in-scatter next-event estimation in homogeneous media (`PtVolumeOptions.forcedNee`, megakernel).
//
// Derivation. Take one free-flight segment [0, L) of a path with throughput beta in a homogeneous medium (extinction sigma_t, scattering sigma_s = sigma_t * albedo, per channel).
// Delta tracking draws a real collision at t with (throughput x density) = beta exp(-sigma_t t) sigma_s per channel (null collisions and the ss / ssMax weights cancel to
// exactly that), and the ordinary kernel then adds beta' Ld(t), Ld = light sample incl. phase, shadow transmittance and MIS weight. Expected light of the first collision in the segment:
//     E = integral_0^L beta exp(-sigma_t t) sigma_s Ld(t) dt.
// The forced estimate draws t from the truncated exponential with the scalar rate mu (the majorant, = sigma_t for grey media):
//     p(t) = mu exp(-mu t) / (1 - exp(-mu L)),
// and adds   beta exp(-sigma_t t) sigma_s Ld(t) / p(t) = beta * sigma_s * exp(-(sigma_t - mu) t) * (1 - exp(-mu L)) / mu * Ld(t).
// For a grey medium that is beta x albedo x (1 - exp(-sigma_t L)) x Ld(t): the chance of a real collision in the segment times the scatter probability. E[forced] = E.
// The collision's own NEE is then SKIPPED (replaced, not added), so the sum stays unbiased; the path's continuation (delta tracking, phase sampling, the MIS on an escape) is unchanged.
// MIS: the weights depend on directions only (V240). For one (t, direction) pair the NEE-side weight ptMis(pl, phasePdf) and the phase-side weight ptMis(phasePdf, pl) sum to 1 whatever
// density the distance came from, so the forced NEE keeps `ptSampleLdMedium` unchanged and the phase-sampled continuation keeps its weight.
// Random numbers: the distance u is the sampler pair (vertex slot PT_SLOT_BSDF_DIRECTION).x of the FORCED vertex index PT_FORCED_VERTEX_BASE + depth, and the light pick / point use the
// same forced vertex's select / lightPoint slots, so none of them repeats a number of the surface or scatter vertex with the same depth (dimension map stays unique; the path stream
// is not touched).

/** 2^24: above any real vertex index (depth caps are far below it; `ptVertexPair` stays in u32 for up to 255 pairs per vertex), so a path deeper than any base cannot reuse a forced vertex's numbers. */
export const PT_FORCED_VERTEX_BASE = 16777216;

/**
 * V7.1 march strata (`PtVolumeOptions.marchSamples`): stratum i of the camera ray's segment uses the forced vertex PT_FORCED_VERTEX_BASE + i * PT_MARCH_STRIDE + depth (stratum 0 is the V4.2 vertex).
 * 2^20 is above any real depth; with ptVertexPair = base + vertex * pairsPerVertex + slot the pair index stays in u32 for up to PT_MARCH_MAX strata (see the unit test).
 */
export const PT_MARCH_STRIDE = 1048576;
export const PT_MARCH_MAX = 64;

/** Resolved stratum count of `marchSamples` (1 = off). */
export function ptMarchCount(v: { marchSamples?: number } | undefined): number {
  const n = v?.marchSamples;
  if (n === undefined || !(n > 1)) return 1;
  if (!Number.isInteger(n) || n > PT_MARCH_MAX) throw new RangeError(`integrator.volumes.marchSamples must be an integer in [1, ${PT_MARCH_MAX}], got ${n}`);
  return n;
}

/** The forced vertex index of stratum i at a path depth. */
export function forcedMarchVertex(i: number, depth: number): number {
  return PT_FORCED_VERTEX_BASE + i * PT_MARCH_STRIDE + depth;
}

/** TS twin of the stratified estimator: mean over strata i of f(u_i), u_i = (i + xi_i) / n. `f` maps a uniform u to the (weighted) single-sample estimate. */
export function marchEstimate(f: (u: number) => number, n: number, xi: (i: number) => number): number {
  let sum = 0;
  for (let i = 0; i < n; i++) sum += f((i + xi(i)) / n);
  return sum / n;
}

/** 1 - exp(-x) without cancellation for small x. */
export function forcedOneMinusExp(x: number): number {
  return x < 1e-3 ? x * (1 - x / 2 + (x * x) / 6) : 1 - Math.exp(-x);
}

/** TS twin: the forced estimate's weight (per channel, multiply by beta and Ld(t)) for a distance t drawn from the truncated exponential of rate mu on [0, L). */
export function forcedNeeWeight(sigmaT: readonly number[], albedo: readonly number[], mu: number, t: number, L: number): number[] {
  const a = forcedOneMinusExp(mu * L);
  return sigmaT.map((s, i) => s * albedo[i]! * Math.exp(-(s - mu) * t) * (a / mu));
}

/** TS twin: inverse CDF of the truncated exponential. */
export function forcedDistance(u: number, mu: number, L: number): number {
  return -Math.log(1 - u * forcedOneMinusExp(mu * L)) / mu;
}

/** TS twin (V4.4): ratio-tracked transmittance over [0, t) of a scalar extinction profile sigma(s) <= mu: the product over the majorant's events of (1 - sigma / mu) (unbiased for exp(-integral sigma)). */
export function ratioTrackTr(sigma: (s: number) => number, mu: number, t: number, rnd: () => number): number {
  let tr = 1;
  let s = 0;
  for (;;) {
    s += -Math.log(1 - rnd()) / mu;
    if (s >= t) return tr;
    tr *= 1 - sigma(s) / mu;
  }
}

/**
 * V4.4 (`volumes.forcedGrid`): sigma_s(t) x T(0, t) at the forced distance t of a GRID segment: the density at t and a ratio-tracked transmittance on [0, t) drawn from its own stream forked off
 * the path's (as a shadow ray's), so it is independent of the light sample's shadow estimate and E[sigma_s T_hat Ld | t] = sigma_s(t) T(0, t) Ld(t): the forced estimate keeps the expectation
 * of the real collision's light sample, which it replaces. Zero density at t: zero (no tracking, no shadow ray).
 */
export const PT_VOL_GRID_SSTR_WGSL = /* wgsl */ `
// V4.4: sigma_s(t) T(0, t) at the forced distance t of a grid segment (density at t, ratio-tracked T on a forked stream; volume/forced-wgsl.ts).
fn ptVolGridSsTr(m: PtVolMedium, id: u32, vo: vec3f, d: vec3f, t: f32) -> vec3f {
    let dens = ptVolDensity(m, vo + d * t);
    if (!(dens > 0.0)) {
        return vec3f(0.0);
    }
    let seed = ptVolFork();
    let keep = ptVolState;
    ptVolState = seed;
    let tr = ptVolTrSegment(id, vo, d, t);
    ptVolState = keep;
    return m.sigmaT * dens * m.albedo * tr;
}
`;

export function ptForcedNeeWgsl(march = 1, grid = false): string {
  if (march > 1) return ptForcedNeeMarchWgsl(march, grid);
  return /* wgsl */ `${grid ? PT_VOL_GRID_SSTR_WGSL : ''}
// V4.2 forced in-scatter NEE for one free-flight segment [0, segMax) in a homogeneous medium (weights: see volume/forced-wgsl.ts). The caller multiplies by beta.
fn ptVolForcedNee(id: u32, vo: vec3f, d: vec3f, segMax: f32, pixelSeed: u32, sampleIndex: u32, depth: u32) -> vec3f {
    let m = ptVolMedium(id);
    if (!(m.mu > 0.0) || !(segMax > 0.0)) {
        return vec3f(0.0);
    }
    let fv = ${PT_FORCED_VERTEX_BASE}u + depth;
    let x = m.mu * segMax;
    let a = select(1.0 - exp(-x), x * (1.0 - 0.5 * x + x * x / 6.0), x < 1.0e-3);
    let u = ptSample2D(pixelSeed, sampleIndex, ptVertexPair(fv, PT_SLOT_BSDF_DIRECTION)).x;
    let t = min(-log(1.0 - u * a) / m.mu, segMax);
${grid ? `    var w = m.sigmaT * m.albedo * exp(-(m.sigmaT - vec3f(m.mu)) * t) * (a / m.mu);
    if (m.info.x != 0.0) {
        w = ptVolGridSsTr(m, id, vo, d, t) * exp(m.mu * t) * (a / m.mu);
        if (!(ptMax3(abs(w)) > 0.0)) {
            return vec3f(0.0);
        }
    }
    return w * ptSampleLdMedium(vo + d * t, -d, id, pixelSeed, sampleIndex, fv);
` : `    let ld = ptSampleLdMedium(vo + d * t, -d, id, pixelSeed, sampleIndex, fv);
    return m.sigmaT * m.albedo * exp(-(m.sigmaT - vec3f(m.mu)) * t) * (a / m.mu) * ld;
`}}
`;
}

/** V7.1: the forced estimate with `march` stratified distances on the camera ray's segment (depth 0); deeper segments take one (stratum 0 = the V4.2 vertex). */
function ptForcedNeeMarchWgsl(march: number, grid: boolean): string {
  return /* wgsl */ `${grid ? PT_VOL_GRID_SSTR_WGSL : ''}
// V7.1 forced in-scatter NEE with ${march} stratified distances on the camera segment (volume/forced-wgsl.ts). The caller multiplies by beta.
fn ptVolForcedNee(id: u32, vo: vec3f, d: vec3f, segMax: f32, pixelSeed: u32, sampleIndex: u32, depth: u32) -> vec3f {
    let m = ptVolMedium(id);
    if (!(m.mu > 0.0) || !(segMax > 0.0)) {
        return vec3f(0.0);
    }
    let n = select(1u, ${march}u, depth == 0u);
    let x = m.mu * segMax;
    let a = select(1.0 - exp(-x), x * (1.0 - 0.5 * x + x * x / 6.0), x < 1.0e-3);
    var acc = vec3f(0.0);
    for (var i = 0u; i < ${march}u; i++) {
        if (i >= n) {
            break;
        }
        let fv = ${PT_FORCED_VERTEX_BASE}u + i * ${PT_MARCH_STRIDE}u + depth;
        let xi = ptSample2D(pixelSeed, sampleIndex, ptVertexPair(fv, PT_SLOT_BSDF_DIRECTION)).x;
        let u = (f32(i) + xi) / f32(n);
        let t = min(-log(1.0 - u * a) / m.mu, segMax);
${grid ? `        var w = m.sigmaT * m.albedo * exp(-(m.sigmaT - vec3f(m.mu)) * t) * (a / m.mu);
        if (m.info.x != 0.0) {
            w = ptVolGridSsTr(m, id, vo, d, t) * exp(m.mu * t) * (a / m.mu);
            if (!(ptMax3(abs(w)) > 0.0)) {
                continue;
            }
        }
        acc += w * ptSampleLdMedium(vo + d * t, -d, id, pixelSeed, sampleIndex, fv);
` : `        let ld = ptSampleLdMedium(vo + d * t, -d, id, pixelSeed, sampleIndex, fv);
        acc += m.sigmaT * m.albedo * exp(-(m.sigmaT - vec3f(m.mu)) * t) * (a / m.mu) * ld;
`}    }
    return acc / f32(n);
}
`;
}
