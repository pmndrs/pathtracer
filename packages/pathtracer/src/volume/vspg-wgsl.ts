// V4.3: volume scattering probability guiding (VSPG, `PtVolumeOptions.guiding: 'vspg'`, megakernel only).
//
// What Cycles does (Apache-2.0, kernel/integrator/shade_volume.h, after Xu et al. 2025): for the PRIMARY ray only, it steps through the whole volume segment collecting candidate scatter
// positions into a reservoir, decides "scatter in this segment" with a guided probability instead of the natural one, and weights the chosen branch by natural / guided. The guided
// probability is the contribution ratio L_scatter / (L_scatter + L_transmit) read from two per-pixel render passes of earlier samples, mixed 75 / 25 with the natural probability
// (defensive resampling), and a majorant scale lifts the guided probability above what the segment's optical depth allows.
//
// What we built (simplest unbiased form for our megakernel). The decision is the EXISTENCE of a majorant collision on the first free-flight segment [0, L) of the primary ray in a
// plain-loop medium (homogeneous or a grid without brick majorants; height and brick media are left alone). Delta tracking draws collisions at rate mu, so
//     natural:  P(no collision) = exp(-mu L) = 1 - a,   P(a collision exists) = a = 1 - exp(-mu L),  first collision ~ truncated exponential on [0, L).
// Guided: draw "exists" with probability P instead; weight a / P when it does (and draw the first collision from the same truncated exponential, so its density cancels), weight
// (1 - a) / (1 - P) when it does not (the path then reaches the segment end with beta unchanged otherwise). Tracking CONTINUES unchanged after the first collision (real scatter, null
// collision, absorption as before). E[weight x f] = P (a / P) E[f | exists] + (1 - P) ((1 - a) / (1 - P)) f0 = a E[f | exists] + (1 - a) f0: independent of P, for any P in (0, 1).
// P = mix(a, Pg, 3/4) so the weights are bounded by 4 (P >= a / 4 and 1 - P >= (1 - a) / 4). Pg = S / (S + T) of the previous samples of the PIXEL (S, T: sums of the luminance of the radiance
// added after the decision, split by the branch taken, already weighted); with no information yet Pg = a (the natural probability). Using earlier samples to choose P stays unbiased:
// each sample's expectation given the past is the pixel value for any past-dependent P (a martingale), the same argument Cycles relies on. Per-pixel state is one vec4 in the accumulator
// tail: x = S, y = T, w = number of guided samples; no readback. Differences from Cycles: no reservoir of candidates and no majorant scale (the existence decision is exact at any P < 1,
// a = 1 included), one extra random number per primary segment, plain-loop media only, and the statistic is the radiance after the decision (forced NEE, added before it, is excluded).
//
// Interaction with forced NEE (V305) / equiangular (V309): combinable. Forced NEE adds a deterministic-in-expectation estimate of the first in-scatter before the decision and the
// collision's own NEE is skipped, so it does not depend on the branch. VSPG then only guides what the forced estimate cannot see: the continuation after a collision (multiple scattering,
// the phase-sampled bounce into emitters) against the pass-through to the surface behind. Without forced NEE VSPG also guides the direct light of the first scatter.

/** TS twin: the effective existence probability from the natural one `a` and the pixel estimate `pg` (Cycles' defensive mix, alpha 3/4). */
export function vspgProbability(a: number, pg: number): number {
  return 0.25 * a + 0.75 * pg;
}

/** TS twin: natural probability of a majorant collision on [0, L). */
export function vspgNatural(mu: number, L: number): number {
  const x = mu * L;
  return x < 1e-3 ? x * (1 - x / 2 + (x * x) / 6) : 1 - Math.exp(-x);
}

/** TS twin of the decision: `u` decides the branch, `u2` draws the first collision. Returns the weight and the first collision distance (null: no collision). */
export function vspgDecide(mu: number, L: number, pg: number, u: number, u2: number): { weight: number; t: number | null } {
  const a = vspgNatural(mu, L);
  const p = vspgProbability(a, pg);
  if (u >= p) return { weight: (1 - a) / (1 - p), t: null };
  return { weight: a / p, t: Math.min(-Math.log(1 - u2 * a) / mu, L) };
}

/** TS twin: the per-pixel guide from the sums (S scatter-branch, T pass-branch contribution); `a` when nothing is known. */
export function vspgGuide(S: number, T: number, a: number): number {
  return S + T > 0 ? S / (S + T) : a;
}

/** WGSL: the private guide state, declared before `ptVolSample`. */
export const PT_VSPG_DECLS_WGSL = `// V4.3 VSPG (volume/vspg-wgsl.ts): the caller sets ptVolGuide (Pg in [0, 1], 2 = unknown) before ptVolSample; the first plain-loop collision decision of that call consumes it (-1) and sets ptVolGuideHit.
var<private> ptVolGuide: f32 = -1.0;
var<private> ptVolGuideHit: bool = false;
`;

/** WGSL: the first step of the tracking loop (replaces `t += ...` at i == 0). Uses `m`, `tmax`, `beta`, `t`. */
export const PT_VSPG_STEP_WGSL = `        if (i == 0u && ptVolGuide >= 0.0) {
            let gx = m.mu * tmax;
            let ga = select(1.0 - exp(-gx), gx * (1.0 - 0.5 * gx + gx * gx / 6.0), gx < 1.0e-3);
            let gp = 0.25 * ga + 0.75 * select(ptVolGuide, ga, ptVolGuide > 1.0);
            ptVolGuide = -1.0;
            if (ptVolRand() >= gp) {
                *beta = *beta * ((1.0 - ga) / (1.0 - gp));
                return PtVolEvent(tmax, 0u);
            }
            ptVolGuideHit = true;
            *beta = *beta * (ga / gp);
            t = min(-log(1.0 - ptVolRand() * ga) / m.mu, tmax);
        } else {
            t += -log(1.0 - ptVolRand()) / m.mu;
        }
`;
