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


## Layered dielectric over diffuse (P2.21, BSDF half)

`packages/pathtracer/src/bsdf/layered.ts` (TS twin), `layered-wgsl.ts`, `layered-lut.ts`; check page `bench/pages/layered-check.ts`.
Not wired into the integrator or the material adapter yet (the OpenPBR material contract is another task); `ptLayeredWgsl(options)`
is appended after `ptBsdfWgsl`. Params: `PtLayeredParams { baseColor, ior, alpha, specularWeight, specularColor, thickness,
internalReflectance }` (`ptLayered(base, ior, roughness, extra)` fills the defaults). Entry points `ptLayeredEval / Pdf / Sample`,
same shapes as the single-lobe ones (WGSL adds a `seed: u32`). `specular_weight * specular_color` scales the interface's Fresnel
reflectance at every angle (OpenPBR; glTF scales F0 only), transmission is `1 - R'`, total internal reflection stays total.

| `layering` | What | Cost (best effort, M5 Pro, 16.8M evals) | WGSL in the 40.4 KB kernel |
|---|---|---|---|
| `albedo` (default) | GGX dielectric reflection lobe on top; Lambert base scaled by `1 - s E(mu_o)` from a 16x16x2 split-sum table (`E = F0 A + B`, sqrt axes in mu and roughness). OpenPBR "layering" by albedo scaling | 0.24 ns/eval | +7.2 KB (3 KB is the table) |
| `albedo` + `albedoBase: 'internal'` | adds the interface's internal reflection: base * keep(mu_o) keep(mu_i) / (eta^2 (1 - rho R_i)); reciprocal; our addition, not OpenPBR | same | same |
| `stochastic` | pbrt-v4 `LayeredBxDF` random walk, `maxDepth` (10), `nSamples` (1), `exitMis: 'pbrt' \| 'none'`; reference mode | 0.52 ns (`none`) / 0.78 ns (`pbrt`), depth 10, 1 walk | +8.3 KB |

Plain rough-dielectric `ptBsdfEval` reads 0.04 ns on the same page (so albedo is ~6x a single lobe, stochastic ~15-20x). Timings are
page-clock, p10 of 15 interleaved repeats, load average 2.3, not a quiet-pass number.

Table accuracy: E vs the exact-Fresnel albedo, max |dE| 0.019 at ior 1.5 (the table's Fresnel shape is the exact one at 1.5, not
Schlick's: Schlick was 0.035 off), 0.030-0.044 over ior 1.33-2.4; below ior ~1.3 it overestimates E at grazing (0.17 at ior 1.1, mu 0.05),
which loses energy, never adds it. A 3D table (ior axis) would fix that for 4x the numbers.

### White furnace (white base, ior 1.5, thickness 0; albedo openpbr / albedo internal / stochastic sampler, depth 200)

| roughness | wo 0 deg | 60 deg | 80 deg |
|---|---|---|---|
| 0.1 | 1.000 / 1.000 / 0.999 | 0.998 / 0.999 / 0.999 | 0.999 / 0.999 / 0.998 |
| 0.3 | 1.000 / 1.013 / 0.936 | 0.999 / 1.011 / 0.935 | 1.000 / 1.010 / 0.903 |
| 0.6 | 1.000 / 1.045 / 0.652 | 1.000 / 1.044 / 0.625 | 0.999 / 1.040 / 0.612 |
| 1 | 1.000 / 1.077 / 0.367 | 1.000 / 1.076 / 0.311 | 1.000 / 1.074 / 0.310 |

openpbr albedo is 1 by construction (to the table error, 0.02). The stochastic loss is the dielectric interface's single-scatter
energy loss compounded over the walk's 2-3 interface crossings (same as pbrt: its interface has no multiple scattering); smooth
interface (roughness 0): eval + delta `F(mu)` = 1 and the sampler reads 1.000. `internal` overshoots up to 8% when rough (its
normalisation assumes the Fresnel `1 - F` for the exit factor).

### Typical plastic: ior 1.5, roughness 0.3, base 0.8, thickness 0: directional albedo vs angle

| wo | spec E (exact) | albedo openpbr | albedo internal | stochastic, sampler | stochastic f, pbrt exit MIS | stochastic f, no exit MIS |
|---|---|---|---|---|---|---|
| 0 | 0.040 | 0.808 | 0.641 | 0.609 | 0.628 | 0.607 |
| 20 | 0.040 | 0.808 | 0.641 | 0.610 | 0.631 | 0.608 |
| 40 | 0.046 | 0.809 | 0.643 | 0.608 | 0.631 | 0.610 |
| 60 | 0.087 | 0.817 | 0.658 | 0.624 | 0.645 | 0.622 |
| 75 | 0.204 | 0.841 | 0.702 | 0.661 | 0.678 | 0.641 |
| 85 | 0.340 | 0.868 | 0.753 | 0.685 | 0.682 | 0.694 |

With pbrt's default thickness 0.01 the stochastic columns drop by another 0.03-0.04 (0.575 at normal incidence). **OpenPBR-style albedo scaling is
0.18-0.20 brighter than the physical layering for this plastic** (a third): it never returns the light the interface reflects back
down, so a bright base keeps all of `rho (1 - E)`; the physical walk keeps `(1 - E) rho (1 - R_i) / (1 - rho R_i)` = 0.585 for a
smooth interface at ior 1.5 (R_i = 0.596). `albedoBase: 'internal'` closes most of it (gap 0.03 at normal incidence, 0.07 at 85).
The two methods agree in the limits that do not depend on the base's bounce count: black base (exactly the existing dielectric
reflection lobe, 1e-12), ior 1 (exactly Lambert), white base at low roughness (both 1); they do NOT agree for a bright base,
whatever the roughness, so "agree at roughness 1" holds only as the base goes to black.

### pbrt quirks found while porting (both kept as pbrt's default, switchable)

- **`f` and `Sample_f` disagree by up to 7%.** The two exit estimators (NEE through the presampled `wis`, and NEE along the
  base's own sample) are weighted with a power heuristic over a density on inside directions and one on outside directions (a
  Jacobian apart), so the weights do not sum to 1: pbrt's own `f` reads +3.5% (roughness 0.3) to +7% (0.5) above its own
  sampler for a white base. `exitMis: 'none'` (weight 1, NEE through `wis` only) makes them agree (0.937 vs 0.936). Default `pbrt` so
  we converge to the reference; an A/B can decide.
- **`PDF` is not a normalised density**: its TRT term has no refraction Jacobian, and the whole is blended 0.1 uniform + 0.9
  estimate. Integrates to 1.5-2.7 over the sphere (only the reflection term is <= 1). Only MIS weights see it (same function on both
  sides, so unbiased), and the sampler's `f |cos| / pdf` weight is exact (`PT_FLAG_PROPORTIONAL`).

### Open choice: which layering is the default for class B

Default `albedo` (deterministic, cheap, fits the megakernel budget) with `stochastic` for reference mode. The numbers above say a
bright dielectric-over-diffuse will not converge to pbrt `coateddiffuse` with `albedo` openpbr (-33% in energy for base 0.8), so for
the gate question the candidates are `stochastic` (matches by construction, up to the quirks above), `albedo + internal` (within 5%
of the walk at base 0.8, 0.04 at 85 deg, but 8% over for white rough bases), and a 3D `E` table. Settle: the P2.21 A/B on one sphere
under a uniform env against pbrt `coateddiffuse` (thickness 0.01 / 0 variants, maxdepth 100) and Mitsuba `roughplastic`
(`nonlinear` true/false), when the material contract and exporters land (`layering`, `albedoBase`, `exitMis` are the three settings).
