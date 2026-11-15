# ReSTIR DI (P4)

Real-time mode only. Reference mode stays plain NEE + MIS (spec section 6, `research/notes/09`). Code: `packages/pathtracer/src/restir/`
(`restir-wgsl.ts` the four passes, `restir.ts` the host, `restir-oracle.ts` the CPU twin of the reservoir arithmetic), the shade side
in `integrator/integrator-wgsl.ts` (gated: a kernel without `integrator.restir` is byte-identical, `node bench/size-snapshot.ts` + `cmp`).
Checks: `node bench/runner/run.ts --page restir-check --args check=...` (bench/pages/restir-check.ts).

## Pipeline (megakernel; one `encode` pass = one frame = one sample of the progressive mean)

```
ptRestirGbuf      camera ray (same jitter as the shade pass) -> position, view depth, shading normal, material      (2 vec4f / pixel / frame, 2 frames)
ptRestirInit      RIS, M candidates from the light sampler (alias pick + point), target = luminance of f |cos| Le / pdf   -> reservoir A
                  optional: one shadow ray for the chosen sample (initialVisibility), W = 0 if occluded
ptRestirTemporal  reproject the primary hit through the previous camera, same normal / depth tests, merge the previous final reservoir -> B
ptRestirSpatial   merge k neighbours at random offsets inside a disc -> final F[parity]
ptMain (shade)    first vertex: read F, regenerate the light sample, ONE shadow ray, add f |cos| Le / pdf * W; deeper vertices: unchanged NEE + MIS
```

Every pass is one dispatch in the same compute pass as the shade dispatch (one `encode` call = 4 + 1 dispatches per frame; the host switches pipeline and
bind groups between them). The passes are one WGSL module with four entry points (group 0 walk, group 1 scene copy, group 2: params ring, G-buffer, reservoirs),
so they bind 3 + (1 or 2) + 2 storage buffers, inside the 10 of Apple GPUs; the shade kernel needs one more read-only storage binding (`PT_RESTIR_SHADE_BINDING` 11).
Memory at 1280x720: G-buffer 2 x 14.7 MB, reservoirs 4 x 14.7 MB (a vec4u per pixel per region: light index, u as 2 x unorm16, bits(W), M).

## The sample and the target

A reservoir sample is `(light index, u)`, `u` in [0, 1)^2 quantised to 16 bits per axis. The light point / direction is a function of `(light, u)` and of the shading point
only through the solid-angle density that `rtSampleLightLi` returns, so the very same sample can be re-evaluated at any other pixel (an identity shift with Jacobian 1 in
`u`-space). That is what makes reuse correct for triangles, the sun cone and an importance-sampled environment map alike (the map's `u -> direction` warp is a function of `u` only).

- target `F(y) = f(wo, wi) |cos| Le / pdf_omega(wi)` (RGB), `phat = luminance(F)`; unshadowed.
- source density in `u`-space = the light-selection pmf, so a candidate's RIS weight is `phat / pmf`, i.e. the plain NEE estimator's luminance.
- M = 1, no reuse: `W = 1 / pmf` and the estimate `F W` is exactly NEE with no MIS weight (check `equiv`).

## Bias correction (runtime flag, D11: both built)

| `bias` | combine | cost | notes |
|---|---|---|---|
| `biased` | Bitterli et al. Alg. 5: `w_i = phat_dst(y_i) W_i M_i`, `W = wsum / (phat_dst(y) sum M)` | one target evaluation per merged reservoir | biased where a neighbour's sample has no support at the destination (a light behind this pixel's horizon, occluded...), darker there |
| `unbiased` | balance-heuristic ("Talbot") MIS over all contributing domains (Lin et al. 2022, GRIS): `m_i(y) = M_i phat_i(y) / sum_j M_j phat_j(y)`, each `phat_j` on the surface of its own pixel, `W = wsum / phat_dst(y)` | (n^2 target evaluations for n domains, no extra shadow rays: the target is unshadowed) | unbiased given `initialVisibility: false`; with it, the occluded samples' zeroed `W` changes the target (biased, but a lot less noisy in shadow) |

Pairwise MIS (O(n) target evaluations) is not built; with k = 2 neighbours the quadratic cost is 9 evaluations.

## Reuse (R6: decorrelated)

- Temporal: the primary hit position is projected through the previous camera (the locally tracked basis of the last encoded frame: a static scene and a moving camera;
  the real-time mode's motion vectors / previous view-projection replace it through `PtRestirPasses.setPreviousCamera`, DIVERGES V130). Reject on out-of-frame, previous
  G-buffer miss, `dot(n, n_prev) < normalCos` (0.9) or `|depth - depth_prev| > depthRel * depth` (0.1). `temporalPick: 'random'` shifts the projected position by a
  uniform +-0.5 px before the floor (a stochastic bilinear pick: no fixed sub-pixel pattern); `'nearest'` is the baseline. The history confidence is capped: stored `M <= historyCap x candidates` (20).
- Spatial: `k` (default 1, at most 8) neighbours at `neighbours: 'random'` offsets (uniform in a disc of `spatialRadius` = 16 px, a fresh pair per pixel and frame) or `'fixed'` (one golden-angle
  spiral for every pixel and frame: the structured baseline). Same rejection tests. One iteration; its output is next frame's history.

## Not done (open)

- Wavefront scheduler (`PtWavefront.create` throws with `integrator.restir`): the passes sit between the primary hit and the shade of one frame; the wavefront's shade kernels would read the reservoir the same way
  (it is one extra read-only binding at its 8 of 10), but its primary and refill scheduling need the G-buffer pass to be sample-aligned. TASKS P4.2a.
- Textured scenes, OpenPBR mode, alpha MASK (the passes decode the legacy untextured material by index).
- Motion vectors for moving objects, dynamic lights (light indices change), history confidence into the reactive mask (R7).
- The primary hit is traced twice per frame (G-buffer pass and shade); with `primary: 'vbuffer'` support the G-buffer pass would read the V-buffer and the shade would too.
