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

export function ptForcedNeeWgsl(): string {
  return /* wgsl */ `
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
    let ld = ptSampleLdMedium(vo + d * t, -d, id, pixelSeed, sampleIndex, fv);
    return m.sigmaT * m.albedo * exp(-(m.sigmaT - vec3f(m.mu)) * t) * (a / m.mu) * ld;
}
`;
}
