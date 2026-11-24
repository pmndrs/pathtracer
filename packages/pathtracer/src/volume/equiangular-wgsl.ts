// V4.1: equiangular distance sampling (Kulla and Fajardo 2012) for the forced in-scatter NEE (`PtVolumeOptions.equiangular`, megakernel, implies forcedNee).
//
// One free-flight segment [0, L) of a homogeneous medium (see forced-wgsl.ts for E = integral_0^L beta exp(-sigma_t t) sigma_s Ld(t) dt and why replacing the collision's own NEE is unbiased).
// The light is picked FIRST (light select + area point of the forced vertex, exactly the numbers ptSampleLdMedium draws for it), so Ld(t) is the single-scatter integrand toward ONE
// fixed point q on an emissive triangle. Two densities for the distance t on [0, L):
//   (a) truncated exponential of the majorant mu:     p_e(t) = mu exp(-mu t) / (1 - exp(-mu L))                       (what forced NEE draws)
//   (b) equiangular about q (t0 = (q - o).d, h = |q - o - t0 d|, theta_a = atan(-t0 / h), theta_b = atan((L - t0) / h)):
//          t = t0 + h tan(theta_a + u (theta_b - theta_a)),   p_q(t) = h / ((theta_b - theta_a)(h^2 + (t - t0)^2)).
// One-sample MIS with the balance heuristic and choice probabilities 1/2: the strategy is picked from the lower / upper half of the SAME random number u (u < 1/2: exponential with 2u,
// else equiangular with 2u - 1: stratified, no extra dimension), and the estimate is  F = beta sigma_s exp(-sigma_t t) Ld(t) / (0.5 p_e(t) + 0.5 p_q(t)).
// Unbiased: the mixture density is positive on [0, L) wherever the integrand is, and E[F] = integral f(t) (0.5 p_e + 0.5 p_q) / (0.5 p_e + 0.5 p_q) dt = E. The MIS weights of Ld (light vs phase
// sampling, V240) are directions-only and sum to 1 for any (t, wi), so they are untouched. Cost: one light sample and one shadow ray per segment, same as forced NEE (+ atan, tan).
// Light picks that are not a triangle (sun, environment: no point) and a degenerate geometry (ray through the point, h ~ 0) use the exponential distance only: the choice depends on the
// pick and the geometry, never on u, so the per-pick estimator stays unbiased.

import { forcedOneMinusExp, PT_FORCED_VERTEX_BASE, PT_MARCH_STRIDE, PT_VOL_GRID_SSTR_WGSL, ratioTrackTr } from './forced-wgsl.ts';

export interface EquiangularSegment {
  /** Ray parameter of the point on the ray closest to the light point. */
  t0: number;
  /** Distance of the light point to the ray line. */
  h: number;
  thetaA: number;
  thetaB: number;
}

/** TS twin: the equiangular parameters of the segment [0, L) toward the point (relative position `q` = light point - ray origin, unit direction `d`). Null when h is degenerate. */
export function equiangularSegment(q: readonly [number, number, number], d: readonly [number, number, number], L: number): EquiangularSegment | null {
  const t0 = q[0] * d[0] + q[1] * d[1] + q[2] * d[2];
  const px = q[0] - d[0] * t0;
  const py = q[1] - d[1] * t0;
  const pz = q[2] - d[2] * t0;
  const h = Math.hypot(px, py, pz);
  if (!(h > 1e-5 * Math.max(1, L))) return null;
  return { t0, h, thetaA: Math.atan2(-t0, h), thetaB: Math.atan2(L - t0, h) };
}

/** TS twin: inverse CDF of the equiangular density. */
export function equiangularDistance(u: number, s: EquiangularSegment, L: number): number {
  return Math.min(Math.max(s.t0 + s.h * Math.tan(s.thetaA + u * (s.thetaB - s.thetaA)), 0), L);
}

/** TS twin: the equiangular density at t. */
export function equiangularPdf(t: number, s: EquiangularSegment): number {
  return s.h / ((s.thetaB - s.thetaA) * (s.h * s.h + (t - s.t0) * (t - s.t0)));
}

/** TS twin: the truncated exponential's density at t. */
export function expPdf(t: number, mu: number, L: number): number {
  return (mu * Math.exp(-mu * t)) / forcedOneMinusExp(mu * L);
}

/** TS twin: one-sample MIS (balance, 1/2 each). Returns the distance and the mixture density; `seg` null = exponential only (density = p_e). */
export function equiangularMixSample(u: number, mu: number, L: number, seg: EquiangularSegment | null): { t: number; pdf: number } {
  if (!seg) {
    const t = Math.min(-Math.log(1 - u * forcedOneMinusExp(mu * L)) / mu, L);
    return { t, pdf: expPdf(t, mu, L) };
  }
  const useEq = u >= 0.5;
  const u2 = useEq ? 2 * u - 1 : 2 * u;
  const t = useEq ? equiangularDistance(u2, seg, L) : Math.min(-Math.log(1 - u2 * forcedOneMinusExp(mu * L)) / mu, L);
  return { t, pdf: 0.5 * expPdf(t, mu, L) + 0.5 * equiangularPdf(t, seg) };
}

/**
 * TS twin of the V4.4 grid estimate (`volumes.forcedGrid`) for one segment [0, L) of a heterogeneous grey medium with extinction sigmaT(t) <= mu and single-scattering albedo `albedo`:
 * t from the truncated exponential of the MAJORANT mu (seg null) or its 50/50 mixture with equiangular (`equiangularMixSample`: both densities are closed form, so the mixture pdf is exact
 * whatever the density profile), then  F = albedo sigmaT(t) T_hat(0, t) Ld(t) / pdf(t)  with T_hat ratio-tracked against mu. E[F] = integral_0^L T(0, t) sigma_s(t) Ld(t) dt, the expected light
 * sample of the segment's first real collision under delta tracking (which therefore skips its own).
 */
export function gridForcedEstimate(u: number, sigmaT: (t: number) => number, albedo: number, mu: number, L: number, seg: EquiangularSegment | null, ld: (t: number) => number, rnd: () => number): number {
  const m = equiangularMixSample(u, mu, L, seg);
  const st = sigmaT(m.t);
  if (!(st > 0)) return 0;
  return (albedo * st * ratioTrackTr(sigmaT, mu, m.t, rnd) * ld(m.t)) / m.pdf;
}

export function ptEquiangularWgsl(march = 1, grid = false): string {
  if (march > 1) return ptEquiangularMarchWgsl(march, grid);
  return /* wgsl */ `${grid ? PT_VOL_GRID_SSTR_WGSL : ''}
// V4.1 equiangular + exponential one-sample MIS for one free-flight segment [0, segMax) in a homogeneous medium (see volume/equiangular-wgsl.ts). The caller multiplies by beta.
fn ptVolEquiNee(id: u32, vo: vec3f, d: vec3f, segMax: f32, pixelSeed: u32, sampleIndex: u32, depth: u32) -> vec3f {
    let m = ptVolMedium(id);
    if (!(m.mu > 0.0) || !(segMax > 0.0)) {
        return vec3f(0.0);
    }
    let fv = ${PT_FORCED_VERTEX_BASE}u + depth;
    let pick = rtSampleLight(ptSample2D(pixelSeed, sampleIndex, ptVertexPair(fv, PT_SLOT_SELECT_COMPONENT)).x);
    if (pick.index == RT_NONE || !(pick.pmf > 0.0)) {
        return vec3f(0.0);
    }
    let l = rtLight(pick.index);
    let x = m.mu * segMax;
    let a = select(1.0 - exp(-x), x * (1.0 - 0.5 * x + x * x / 6.0), x < 1.0e-3);
    var u = ptSample2D(pixelSeed, sampleIndex, ptVertexPair(fv, PT_SLOT_BSDF_DIRECTION)).x;
    var h = 0.0;
    var t0 = 0.0;
    var thA = 0.0;
    var thB = 0.0;
    if (l.kind == RT_LIGHT_TRIANGLE) {
        let b = rtSampleUniformTriangle(ptSample2D(pixelSeed, sampleIndex, ptVertexPair(fv, PT_SLOT_LIGHT_POINT)));
        let rel = l.position + l.edge1 * b.x + l.edge2 * b.y - vo;
        t0 = dot(rel, d);
        let perp = rel - d * t0;
        h = length(perp);
        if (h > 1.0e-5 * max(1.0, segMax)) {
            thA = atan2(-t0, h);
            thB = atan2(segMax - t0, h);
        } else {
            h = 0.0;
        }
    }
    var t: f32;
    var pdf: f32;
    let pe = m.mu / a;
    if (h > 0.0) {
        let useEq = u >= 0.5;
        u = select(2.0 * u, 2.0 * u - 1.0, useEq);
        t = select(-log(1.0 - u * a) / m.mu, t0 + h * tan(thA + u * (thB - thA)), useEq);
        t = clamp(t, 0.0, segMax);
        pdf = 0.5 * pe * exp(-m.mu * t) + 0.5 * h / ((thB - thA) * (h * h + (t - t0) * (t - t0)));
    } else {
        t = min(-log(1.0 - u * a) / m.mu, segMax);
        pdf = pe * exp(-m.mu * t);
    }
${grid ? `    var w = m.sigmaT * m.albedo * exp(-m.sigmaT * t);
    if (m.info.x != 0.0) {
        w = ptVolGridSsTr(m, id, vo, d, t);
        if (!(ptMax3(abs(w)) > 0.0)) {
            return vec3f(0.0);
        }
    }
    return w * ptSampleLdMedium(vo + d * t, -d, id, pixelSeed, sampleIndex, fv) / pdf;
` : `    let ld = ptSampleLdMedium(vo + d * t, -d, id, pixelSeed, sampleIndex, fv);
    return m.sigmaT * m.albedo * exp(-m.sigmaT * t) * ld / pdf;
`}}
`;
}

/**
 * V7.1: equiangular + exponential MIS with `march` strata on the camera segment. Stratum i draws its distance coordinate u = (i + xi.x) / n and chooses the technique with its own second
 * number xi.y (equiangular when xi.y >= 0.5): both techniques then stratify in their own CDF space, and the one-sample balance MIS stays unbiased. Deeper segments (n = 1) keep the V4.1 split of u.
 */
function ptEquiangularMarchWgsl(march: number, grid: boolean): string {
  return /* wgsl */ `${grid ? PT_VOL_GRID_SSTR_WGSL : ''}
// V7.1 equiangular + exponential one-sample MIS with ${march} stratified distances on the camera segment (volume/equiangular-wgsl.ts). The caller multiplies by beta.
fn ptVolEquiNee(id: u32, vo: vec3f, d: vec3f, segMax: f32, pixelSeed: u32, sampleIndex: u32, depth: u32) -> vec3f {
    let m = ptVolMedium(id);
    if (!(m.mu > 0.0) || !(segMax > 0.0)) {
        return vec3f(0.0);
    }
    let n = select(1u, ${march}u, depth == 0u);
    let x = m.mu * segMax;
    let a = select(1.0 - exp(-x), x * (1.0 - 0.5 * x + x * x / 6.0), x < 1.0e-3);
    let pe = m.mu / a;
    var acc = vec3f(0.0);
    for (var i = 0u; i < ${march}u; i++) {
        if (i >= n) {
            break;
        }
        let fv = ${PT_FORCED_VERTEX_BASE}u + i * ${PT_MARCH_STRIDE}u + depth;
        let pick = rtSampleLight(ptSample2D(pixelSeed, sampleIndex, ptVertexPair(fv, PT_SLOT_SELECT_COMPONENT)).x);
        if (pick.index == RT_NONE || !(pick.pmf > 0.0)) {
            continue;
        }
        let l = rtLight(pick.index);
        let xi = ptSample2D(pixelSeed, sampleIndex, ptVertexPair(fv, PT_SLOT_BSDF_DIRECTION));
        var u = (f32(i) + xi.x) / f32(n);
        var h = 0.0;
        var t0 = 0.0;
        var thA = 0.0;
        var thB = 0.0;
        if (l.kind == RT_LIGHT_TRIANGLE) {
            let b = rtSampleUniformTriangle(ptSample2D(pixelSeed, sampleIndex, ptVertexPair(fv, PT_SLOT_LIGHT_POINT)));
            let rel = l.position + l.edge1 * b.x + l.edge2 * b.y - vo;
            t0 = dot(rel, d);
            let perp = rel - d * t0;
            h = length(perp);
            if (h > 1.0e-5 * max(1.0, segMax)) {
                thA = atan2(-t0, h);
                thB = atan2(segMax - t0, h);
            } else {
                h = 0.0;
            }
        }
        var t: f32;
        var pdf: f32;
        if (h > 0.0) {
            var useEq = xi.y >= 0.5;
            if (n == 1u) {
                useEq = u >= 0.5;
                u = select(2.0 * u, 2.0 * u - 1.0, useEq);
            }
            t = select(-log(1.0 - u * a) / m.mu, t0 + h * tan(thA + u * (thB - thA)), useEq);
            t = clamp(t, 0.0, segMax);
            pdf = 0.5 * pe * exp(-m.mu * t) + 0.5 * h / ((thB - thA) * (h * h + (t - t0) * (t - t0)));
        } else {
            t = min(-log(1.0 - u * a) / m.mu, segMax);
            pdf = pe * exp(-m.mu * t);
        }
${grid ? `        var w = m.sigmaT * m.albedo * exp(-m.sigmaT * t);
        if (m.info.x != 0.0) {
            w = ptVolGridSsTr(m, id, vo, d, t);
            if (!(ptMax3(abs(w)) > 0.0)) {
                continue;
            }
        }
        acc += w * ptSampleLdMedium(vo + d * t, -d, id, pixelSeed, sampleIndex, fv) / pdf;
` : `        let ld = ptSampleLdMedium(vo + d * t, -d, id, pixelSeed, sampleIndex, fv);
        acc += m.sigmaT * m.albedo * exp(-m.sigmaT * t) * ld / pdf;
`}    }
    return acc / f32(n);
}
`;
}
