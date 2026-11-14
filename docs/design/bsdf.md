# BSDFs (P2 slice 1)

The bench material subset in `@pmndrs/pathtracer` (`packages/pathtracer/src/bsdf/`), WGSL prefix `pt`, with a TS twin
(f64) for CPU tests. Ported from pbrt-v4 (Apache-2.0, THIRD_PARTY 1.3) so Tier 0 renders can match the references.

## What exists

| Kind | Model | Notes |
|---|---|---|
| 0 diffuse | Lambert | concentric-disk cosine sampling |
| 1 conductor | GGX (Trowbridge-Reitz), complex Fresnel from RGB eta/k | smooth (delta) when max(alpha) < 1e-3; alpha = roughness², clamped ≥ 1e-4 |
| 2 dielectric | smooth: delta reflect/refract with exact Fresnel; rough: pbrt-v4's generalised half vector, R + T | `ior == 1` is a delta pass-through; no absorption |
| 3 black | nothing | |

API, all in the local shading frame (+Z = normal; `wo.z < 0` = arriving from inside a dielectric):
- `PtBsdfParams { kind, reflectance, eta, k, alpha: vec2, ior }` (plain, unpacked; the P1.4 material table maps to it, P2.2).
- `ptBsdfEval(p, wo, wi) -> {f, pdf}`, `ptBsdfPdf`, `ptBsdfIsDelta`.
- `ptBsdfSample(p, wo, u: vec3) -> {wi, f, pdf, flags, etap}`; flags DELTA=1, REFLECTION=2, TRANSMISSION=4, 0 = failed.
  Throughput is always `f·|cos wi|/pdf`; delta lobes follow pbrt (`f` includes 1/|cos|, `pdf` is the lobe probability).
- Frame helpers `ptFrameFromNormal` (Duff 2017), `ptToLocal`, `ptFromLocal`.
- Module settings (constants in the generated WGSL, so each variant is its own pipeline): `vndf: 'standard' | 'cap'`,
  `transport: 'radiance' | 'importance'`.

## Furnace (single scattering, brute-force integral of eval; sampling agrees to 1e-3)

White conductor (F ≈ 1), albedo (columns are perceptual roughness r, alpha = r²):

| wo | r 0.05 | 0.1 | 0.25 | 0.5 | 1 |
|---|---|---|---|---|---|
| 0° | 1.0000 | 0.9999 | 0.9957 | 0.9158 | 0.3068 |
| 60° | 1.0000 | 0.9997 | 0.9883 | 0.8573 | 0.4507 |
| 80° | 0.9999 | 0.9981 | 0.9354 | 0.8556 | 0.6682 |

Rough dielectric, ior 1.5, importance transport, R + T (perceptual roughness):

| wo | r 0.05 | 0.1 | 0.25 | 0.5 | 1 |
|---|---|---|---|---|---|
| 0° | 1.0000 | 1.0000 | 0.9996 | 0.9920 | 0.8933 |
| 60° | 1.0000 | 0.9999 | 0.9970 | 0.9607 | 0.7416 |
| 80° | 0.9999 | 0.9990 | 0.9648 | 0.9127 | 0.7083 |
| 130° (inside) | 1.0000 | 0.9998 | 0.9888 | 0.8509 | 0.5101 |
| 160° (inside) | 1.0000 | 0.9999 | 0.9955 | 0.9218 | 0.4359 |

The loss at high roughness is single-scatter energy (no multiple-scattering compensation; the references have none).
In radiance transport the dielectric albedo is R + T/eta'² (0.467 at normal incidence from outside, above 1 from
inside); it cancels on exit from a closed object.

Other checks: sample's pdf and f equal eval at the sampled direction (1e-6); reciprocity (1e-9; transmission as
`f(wo,wi)/ior² = f(wi,wo)`); chi-square, 400k samples, 7 parameter sets × 2 VNDFs, all p > 1e-3, with a power test that
rejects slightly wrong parameters at p < 1e-6. GPU (`bench/pages/bsdf-check.ts`, Apple): eval max rel error 2.2e-5,
sample 5.6e-4, over 3 module variants × 3000 cases.

## Open choices (defaults hold until measured)

1. **VNDF: standard or spherical cap.** Default standard (pbrt's exact warp, so renders diff cleanly against pbrt's
   sample sequence). Cap: fewer ops, no sqrt of u, no basis threshold. Same distribution, so noise at equal spp should
   match; the difference is cost per sample. Setting: yes (`vndf`). Settle: P2.3 in the quiet pass (GPU time per sample
   + relMSE at equal spp on a rough conductor and a rough dielectric scene).
2. **Transport mode.** Default radiance (what pbrt and Mitsuba use for camera paths); importance stays for tests and a
   possible light tracer. Zero runtime cost either way (module constant). Settle: the Tier 0 dielectric furnace reads 1
   in radiance mode at high maxDepth once the integrator exists (P2.4).
3. **Parameter packing.** Keep `PtBsdfParams` unpacked (~64 B in registers) and adapt from the packed 96–128 B table
   record (P2.2). Pro: no coupling. Con: unused fields per kind. Settle: register pressure on Metal and DXC once the
   integrator reads the real table.
4. **Anisotropy.** alphaX/alphaY are carried and tested, but there is no tangent frame yet (P2.7, when a Tier 1 scene
   needs it).
5. **Basis threshold in the standard VNDF** (`wh.z < 0.99999`, as pbrt). A branchless Duff basis removes a branch and
   the discontinuity without changing the distribution, but no longer matches pbrt's sequence. Could join the P2.3 A/B
   as a third variant.
