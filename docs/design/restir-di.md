# ReSTIR DI (P4)

Real-time mode only. Reference mode stays plain NEE + MIS (spec section 6, `research/notes/09`). Code: `packages/pathtracer/src/restir/`
(`restir-wgsl.ts` the four passes, `restir.ts` the host, `restir-oracle.ts` the CPU twin of the reservoir arithmetic), the shade side
in `integrator/integrator-wgsl.ts` (gated: a kernel without `integrator.restir` is byte-identical, `node bench/size-snapshot.ts` + `cmp`).
Checks: `node bench/runner/run.ts --page restir-check --args check=...` (bench/pages/restir-check.ts).

## Pipeline (megakernel; one `encode` pass = one frame = one sample of the progressive mean)

```
ptRestirGbuf      camera ray (same jitter as the shade pass) -> position, view depth, shading normal, material      (20 B / pixel / frame, 2 frames, P4.2c)
ptRestirInit      RIS, M candidates from the light sampler (alias pick + point), target = luminance of f |cos| Le / pdf   -> reservoir A
                  optional: one shadow ray for the chosen sample (initialVisibility), W = 0 if occluded
ptRestirTemporal  reproject the primary hit through the previous camera, same normal / depth tests, merge the previous final reservoir -> B
ptRestirSpatial   merge k neighbours at random offsets inside a disc -> final F[parity]
ptMain (shade)    first vertex: read F, regenerate the light sample, ONE shadow ray, add f |cos| Le / pdf * W; deeper vertices: unchanged NEE + MIS
```

Every pass is one dispatch in the same compute pass as the shade dispatch (one `encode` call = 4 + 1 dispatches per frame; the host switches pipeline and
bind groups between them). The passes are one WGSL module with four entry points (group 0 walk, group 1 scene copy, group 2: params ring, G-buffer, reservoirs),
so they bind 3 + (1 or 2) + 2 storage buffers, inside the 10 of Apple GPUs; the shade kernel needs one more read-only storage binding (`PT_RESTIR_SHADE_BINDING` 11).
Memory at 1280x720: G-buffer 2 x 18.4 MB (was 2 x 29.5), reservoirs 4 x 14.7 MB (a vec4u per pixel per region: light index, u as 2 x unorm16, bits(W), M).

## The sample and the target

A reservoir sample is `(light index, u)`, `u` in [0, 1)^2 quantised to 16 bits per axis. The light point / direction is a function of `(light, u)` and of the shading point
only through the solid-angle density that `rtSampleLightLi` returns, so the very same sample can be re-evaluated at any other pixel (an identity shift with Jacobian 1 in
`u`-space). That is what makes reuse correct for triangles, the sun cone and an importance-sampled environment map alike (the map's `u -> direction` warp is a function of `u` only).

- target `F(y) = f(wo, wi) |cos| Le / pdf_omega(wi)` (RGB), `phat = luminance(F)`; unshadowed.
- source density in `u`-space = the light-selection pmf, so a candidate's RIS weight is `phat / pmf`, i.e. the plain NEE estimator's luminance.
- M = 1, no reuse: `W = 1 / pmf` and the estimate `F W` is exactly NEE with no MIS weight (check `equiv`).

## Bias correction (runtime flag, D11: both built)

**Defaults (D32, provisional):** ReSTIR stays opt-in; when on, `bias: 'unbiased'` (no initial visibility), M 8, temporal, spatial `k` 2 (random offsets); the `realtime` preset adds `sharedHit` on the megakernel. `biased` (with visibility) stays a setting: it darkens 4-17% on the A/B scenes (REALTIME-PIPELINE.md section 4). The table below is the menu, not the default.

| `bias` | combine | cost | notes |
|---|---|---|---|
| `biased` | Bitterli et al. Alg. 5: `w_i = phat_dst(y_i) W_i M_i`, `W = wsum / (phat_dst(y) sum M)` | one target evaluation per merged reservoir | biased where a neighbour's sample has no support at the destination (a light behind this pixel's horizon, occluded...), darker there |
| `unbiased` | balance-heuristic ("Talbot") MIS over all contributing domains (Lin et al. 2022, GRIS): `m_i(y) = M_i phat_i(y) / sum_j M_j phat_j(y)`, each `phat_j` on the surface of its own pixel, `W = wsum / phat_dst(y)` | (n^2 target evaluations for n domains, no extra shadow rays: the target is unshadowed) | unbiased given `initialVisibility: false`; with it, the occluded samples' zeroed `W` changes the target (biased, but a lot less noisy in shadow) |

**Random streams (P4.2b bias hunt).** Every random number a pixel draws in one frame has its own key in the PCG chain (`PT_RESTIR_KEY`: candidates 0..127, sub-pixel shift 128, temporal select 132, spatial offsets 144, spatial select 160). The temporal and spatial merges once
shared a key base, so the spatial merge's "replace the kept sample" decision was correlated with the temporal merge's for the same pixel and frame: the kept sample was no longer chosen with probability proportional to its weight. The error is invisible
while the weights of the domains are alike and a darkening of up to -15% on a face when they differ a lot (a stochastic material pick that changes every frame, `restir-pbr-lamps-mixonly`: -0.23% of the image before, +0.05% +- 0.02% after).

Pairwise MIS (O(n) target evaluations) is not built; with k = 2 neighbours the quadratic cost is 9 evaluations.

## Reuse (R6: decorrelated)

- Temporal: the primary hit position is projected through the previous camera (the megakernel's: `setPrevCamera` / `nextFrame`, else the last encoded frame's; P4.2e, DIVERGES V170). Reject on out-of-frame, previous
  G-buffer miss, `dot(n, n_prev) < normalCos` (0.9) or `|depth - depth_prev| > depthRel * depth` (0.1). `temporalPick: 'random'` shifts the projected position by a
  uniform +-0.5 px before the floor (a stochastic bilinear pick: no fixed sub-pixel pattern); `'nearest'` is the baseline. The history confidence is capped: stored `M <= historyCap x candidates` (20).
- Spatial: `k` (default 2 since D32, was 1; at most 8) neighbours at `neighbours: 'random'` offsets (uniform in a disc of `spatialRadius` = 16 px, a fresh pair per pixel and frame) or `'fixed'` (one golden-angle
  spiral for every pixel and frame: the structured baseline). Same rejection tests. One iteration; its output is next frame's history.

## Unification with the real-time mode (P4.2b, P4.2c, P4.2e)

**One previous camera.** `PtMegakernel` owns it (`setPrevCamera` / `nextFrame`, else the camera of the last encoded frame) and passes it to `PtRestirPasses.encodeFrame`; the frame planes' motion
vectors and the reservoir reprojection are the same camera (V170). `restir-check prevcam`: tracked, `nextFrame` and `setPrevCamera` give bit-identical images, a wrong previous camera does not.

**Blue noise.** The G-buffer pass draws its jitter pair through the same `ptSample2D` as the shade pass, so `sampler: 'bluenoise'` binds the texture to the passes (`PT_BLUENOISE_BINDING` 12 in their
io group); the realtime preset no longer falls back to Owen-Sobol. Candidates / picks / offsets stay PCG (V135, V174). The frame planes need nothing: the reservoir's first-vertex light is part of the
path radiance, which `ptFramePlanes` splits by the primary BSDF kind (`realtime-check restir=unbiased`: diffuse x albedo + specular x factor = radiance).

**G-buffer, packed (V172).** 20 B / pixel / frame: position (f32 x 3), octahedral normal (2 x unorm16), material word; the view depth is derived. Textured scenes add uv + footprint (32 B), a stochastic
material pick adds its random (+4 B). 1080p, two frames: 83.0 MB (was 132.7 MB). Layout: `ptRestirGbufLayout`; one `array<u32>` binding.

**Primary hit.** `primary: 'vbuffer'`: the G-buffer pass resolves the engine's V-buffer (`ptVbufferHit`, binding 9 in the passes' io group), the shade already did: the passes trace no camera ray.
Trace primaries: `restir.sharedHit` makes the G-buffer pass write the hit as a V-buffer texel (own 16 B / pixel buffer, passes binding 3, shade binding 15) and the shade kernel resolve it (`ptVbufferWgsl('restir')`):
one camera-ray trace per frame instead of two, at the price of 16 B / pixel and unorm16 barycentrics (V171). Needs `layout.positions`. Default off until the quiet-pass A/B.

**Materials (V173).** The target uses the very decode of the shade pass: `ptRsBsdf` = `rtApplyMaterialTextures(rtMaterial(m), uv, footprint)` (textured) then `ptBsdfFromMaterial(mat, mixU)` (OpenPBR mode,
`mixU` stored per pixel), evaluated with `ptEval` for layered (albedo) materials, `ptBsdfEval` otherwise; normal-mapped shading normals are stored. Alpha MASK: the G-buffer pass skips cutouts exactly like the shade
(first solid hit), visibility uses `ptVisibleMasked`. `ptRestirUnsupported` says what is refused (stochastic layering, fuzz, thin film, anisotropy, `materialLodDepth` 0, hooks).

**On the wavefront (P4.2a, V212).** `PtWavefront.create(..., { integrator: { restir } })` (plain scheduler) runs the SAME `PtRestirPasses` (module, buffers, previous-camera rule, `restir.stats`, `dropHistory`) inside the sample's compute pass,
before the camera kernel (`encodeFrame(pass, walkGroup, sceneGroup, ...)` with the wavefront's own walk / scene groups, which have the megakernel's layouts). The G-buffer pass traces its own jittered camera ray from the same sampler draw as
the camera kernel (the same arrangement as the megakernel: `sharedHit` is the only join between the two), so path = pixel = G-buffer texel and no alignment work is needed; refill / layers are refused (`create` says why: a slot restarts at its own time,
so its sample would not match the G-buffer's). The shade kernel gets ONE read-only binding (11, the reservoirs; 9 of 10 Apple storage buffers split, 8 packed) and `ptwfRestirLd`: the megakernel's `ptRestirLd` up to the visibility test
(`evalFn`, the reservoir's light through `rtSampleLightLi`), whose shadow ray goes through the shadow queue like a NEE ray, so visibility (and the frame planes' class add) are the shadow kernel's. The first vertex's
BSDF-sampled continuation gets `prevPdf = 0` exactly as in the megakernel, and the final reservoir region travels in the unused `up.w` lane of the params record (`limits.w` is the wavefront's iteration). Unsupported on the wavefront
(`create` throws): `sharedHit` (join the primary hit into the camera kernel: follow-up), alpha MASK (the megakernel tests reservoir visibility with the exact cutout walk whatever the shadow policy; the shadow kernel follows the policy), hooks, refill.
Checked per pixel against the megakernel's frame (same reservoirs expected): `wf-frame-check restir=unbiased` (TASKS P4.2a).

## Not done (open)

- The wavefront's `sharedHit` (one camera-ray trace per frame: the camera kernel would fill its path from the hit buffer) and alpha MASK / hooks with ReSTIR on the wavefront (TASKS P4.2a follow-ups).
- Fuzz, thin film, anisotropy, stochastic layering with ReSTIR (V173); the layered + ReSTIR kernel is over the 50 KB class (shade 50.2 KB plain OpenPBR, 51.8 KB textured + mask).
- Motion vectors for moving objects, dynamic lights (light indices change), history confidence into the reactive mask (R7).
