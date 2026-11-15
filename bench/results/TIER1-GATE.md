# Tier 1 gate (P2.32, D23): class A scenes, 4096 spp vs pbrt-v4 / Mitsuba 3 (4096 spp)

M5 Pro, 2026-11-15, load average 2-7 (correctness numbers, no timing). Ours: `matrix.ts --renderers pt-megakernel --spp 4096
--snapshots 16,64,256,1024 --extra pt_tint=0,pt_coverage=1` (reports `p232-gate`, `p232-aniso-uv`, `p232-b-stoch`, `p232-b-albint`,
git-ignored). Layering: the adapter default `stochastic` (reference mode).

Rule (tier1.md §6): relMSE(ours, ref) <= 1.5 x floor; |relBias| <= 0.5%; 32 px block-mean rms inside the references' own spread;
0 non-finite pixels; 1/spp slope. The floor of reference X is relMSE(other reference, X) (not symmetric).

**Result: all 7 class A scenes pass.**

| Scene (reference) | relMSE | floor | relBias | block rms ours vs ref / ref vs ref | verdict |
|---|---|---|---|---|---|
| AttenuationTest-notex (Mitsuba; pbrt invalid, V62) | 2.0e-3 | 5.0e-3 | +0.04% | 0.43% / 4.64% | PASS |
| TransmissionThinwallTestGrid-notex (pbrt / Mitsuba) | 7.0e-4 / 7.5e-4 | 8.6e-4 | +0.02% / -0.01% | 0.11% / 0.12% | PASS |
| EmissiveStrengthTest-notex (pbrt / Mitsuba) | 1.5e-4 / 2.3e-4 | 1.8e-4 | +0.03% / +0.01% | 0.04% / 0.05% | PASS |
| IORTestGrid-notex-A (pbrt / Mitsuba) | 2.7e-4 / 2.1e-4 | 3.1e-4 | +0.03% / -0.03% | 0.13% / 0.16% | PASS |
| IORTestGrid-A, textured (pbrt / Mitsuba) | 2.7e-4 / 3.1e-4 | 3.9e-4 / 4.0e-4 | +0.03% / +0.19% | 0.15% / 0.67% | PASS |
| MetalRoughSpheresNoTextures-grey-metal (pbrt / Mitsuba) | 3.9e-4 / 1.8e-4 | 5.3e-4 / 5.5e-4 | +0.02% / +0.07% | 0.15% / 0.55% | PASS |
| AnisotropyStrengthTest-notex, default tangent (pbrt) | 5.7e-4 | 1.1e-3 | +0.02% | 0.13% / 1.49% | PASS |
| AnisotropyStrengthTest-notex, `pt_tangent=uv` (pbrt) | 2.7e-4 | 1.1e-3 | +0.02% | 0.12% / 1.49% | PASS |

All scenes: 0 non-finite, 0 firefly pixels; 1/spp slope k 1.13-1.26 (snapshots share samples with the final image, so slightly
above 1). Anisotropy vs Mitsuba reads relBias +0.62%: Mitsuba's own -0.58% vs pbrt (separable Smith G at large alpha, D15); gated
against pbrt. Attenuation vs pbrt would fail (+1.49%, block rms 4.69%): the known V62 pbrt error on thick saturated absorbers.

## Class B, bracketed (not gated)

d = relMSE; ratio = d(ours, ref) / d(other ref, ref).

| Scene | vs pbrt | ratio | vs Mitsuba | ratio |
|---|---|---|---|---|
| MetalRoughSpheresNoTextures | 7.3e-4, bias -0.08% | 0.49 | 9.6e-4, bias +0.78% | 0.53 |
| MetalRoughSpheres (textured) | 6.2e-4, bias -0.07% | 0.39 | 1.5e-3, bias +0.61% | 0.68 |
| ClearCoatTest-notex | 2.3e-3, bias -0.84% | 0.92 | 8.9e-5, bias +0.12% | 0.03 |
| ClearCoatTest-notex-metal | 5.2e-4, bias +0.37% | 0.04 | 7.4e-3, bias -3.27% (Disney clearcoat stand-in) | 0.88 |
| AttenuationTest (textured) | 4.1e-3, bias +2.03% | n/a (pbrt floor 2.2e-1, V62) | 4.5e-3, bias +3.41% | 0.80 |

Class C (coverage only): SheenTestGrid(-notex) now differ by design (our fuzz is on, the references have no sheen): relMSE 5e-2,
block rms 23-28%. IORTestGrid(-notex) class C spheres unchanged from waves 1-2.

## Finding: the D24 library default overshoots bright rough bases (P2.32b)

With `layering: 'albedo'`, `albedoBase: 'internal'` (report `p232-b-albint`), relBias vs pbrt: AttenuationTest +15.2%,
SheenTestGrid +13.0%, SheenTestGrid-notex +5.9%, MetalRoughSpheres +2.2%, MetalRoughSpheresNoTextures +2.8%. The AttenuationTest
backdrop is a white dielectric-over-diffuse at roughness 0.5-0.8; 32 px blocks read 1.08-1.12x the stochastic walk. The closed form
is approximate for rough interfaces over bright bases; D24's A/B was a single plastic sphere (+0.4-3.7%). OpenPBR albedo is worse
(AttenuationTest mean 0.277 vs internal 0.185, reference 0.159). Not a gate issue (class B is not gated; stochastic is inside the floor).
