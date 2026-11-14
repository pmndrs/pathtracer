# Tier 1 wave 2 (textured, unmodified models): our megakernel vs pbrt-v4 / Mitsuba 3 (P2.24)

M5 Pro, 2026-11-14. Ours: `matrix.ts --renderers pt-megakernel --spp 4096 --extra pt_tint=0,pt_coverage=1` (reports `t2-4096`, `t2-ior-a`); references in `bench/references/<scene>/` at 4096 spp, except Mitsuba
on `MetalRoughSpheres` at 1024 (a 64 spp pass took 31 s, so 4096 would have been ~33 min; pbrt 4096 took 8 min). One native render at a time; load average 14-35 from other jobs
during the renders (these are correctness numbers, not timings). Compare table by `node bench/adapters/openpbr-compare.ts --report t2-4096`; per-region numbers by projected mesh bounds shrunk 15% (nearer wins).
Models are unmodified: textures, labels and backdrops are there, nothing dropped or replaced. None of the four uses `alphaMode`, so MASK is covered by the synthetic `tex-mask` row.

| scene | ours mean RGB | pbrt mean RGB | Mitsuba mean RGB | relBias vs pbrt / Mitsuba | relMSE vs pbrt / Mitsuba | floor (pbrt vs Mitsuba) | block-mean rms vs pbrt / Mitsuba |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tex-mask (320x240, alpha MASK) | 0.7206 0.6317 0.6106 | 0.7204 0.6316 0.6105 | 0.7192 0.6305 0.6100 | +0.02% / +0.16% | 1.9e-5 / 1.3e-4 | 1.3e-4 | 0.02% / 0.32% |
| **IORTestGrid-A** (class C spheres dropped; gate) | 0.1770 0.1811 0.2092 | 0.1769 0.1811 0.2091 | 0.1766 0.1808 0.2087 | +0.03% / +0.19% | 2.7e-4 / 3.1e-4 | 4.0e-4 | 0.15% / 0.69% |
| IORTestGrid (all 20 spheres, coverage) | 0.1713 0.1756 0.2036 | 0.1689 0.1730 0.2001 | 0.1687 0.1727 0.1996 | +1.58% / +1.76% | 5.0e-2 / 5.1e-2 | 6.1e-4 | 22% / 23% |
| SheenTestGrid (coverage: sheen ignored) | 0.1865 0.1946 0.2793 | 0.1872 0.1936 0.2789 | 0.1847 0.1927 0.2741 | -0.11% / +1.37% | 4.5e-4 / 1.1e-3 | 1.3e-3 | 0.62% / 4.5% |
| AttenuationTest | 0.1643 0.1865 0.2570 | 0.1590 0.1864 0.2503 | 0.1579 0.1799 0.2499 | +2.03% / +3.41% | 4.1e-3 / 4.5e-3 | 5.6e-3 | 5.7% / 3.7% |
| MetalRoughSpheres (Mitsuba 1024 spp) | 0.4135 0.4183 0.4678 | 0.4135 0.4186 0.4685 | 0.4109 0.4157 0.4652 | -0.07% / +0.61% | 6.2e-4 / 1.5e-3 | 2.2e-3 | 0.47% / 2.2% |

0 non-finite pixels in all renders.

## Verdicts against tier1.md section 6

- **IORTestGrid-A passes** (relMSE <= 1.5 x the floor: 2.7e-4 and 3.1e-4 against 4.0e-4; |relBias| <= 0.5%; block rms 0.15% / 0.69%). Ten class A spheres: ours/pbrt -0.27..+0.36%, ours/Mitsuba +0.18..+2.16%
  (Mitsuba is 0.2-2.1% under pbrt on the white transmissive ones, its own roughness-1 conductor/Smith G and refraction noise, not texture). The rough-metal checker backdrop (base colour texture on a conductor,
  roughness 1): box between spheres ours/pbrt +0.03%, ours/Mitsuba +0.19%; strip above the spheres +0.02% / +0.53% (Mitsuba is -0.5% under pbrt there: its `k` texture is filtered in k, not in reflectance, V78).
- **IORTestGrid (unmodified) is not gateable**: the ten specularFactor 0.25 spheres are class C (references approximate F0 through the IOR, as in wave 1). They read -11..-18% (black) and +42..+84% (white
  transmissive) against both references; the class A spheres move by up to +3.4% from their reflections. Same finding as `IORTestGrid-notex` in wave 1.
- **MetalRoughSpheres passes** (class A rows by metalness 1; the grid also holds class B dielectric and mixed rows): relMSE 6.2e-4 against pbrt, below the 2.2e-3 floor; relBias -0.07% / +0.61%. 8 x 8 block means of
  the green channel agree with pbrt to the printed 0.01 in every block (bottom rows, rough columns included: 0.17-0.27), Mitsuba: top (metal) rows ours/Mitsuba +0.4%, bottom (dielectric over diffuse) rows +1.4..+1.6% (Mitsuba is 1.3-1.5% under pbrt there: `roughplastic` chain vs `coateddiffuse`, class B).
  Metalness from the B channel, roughness from G (alpha = G^2), clamp wrap, base colour from an sRGB map: all three agree that gold and grey columns land where the texture puts them.
- **SheenTestGrid**: checker backdrop (rough dielectric, base colour texture) and the 16 spheres within the floor of pbrt (0.11%, relMSE 4.5e-4); Mitsuba +1.4% is the layered-model difference (spheres ours/pbrt +4..+7%, ours/Mitsuba
  +3.4..+5.2%, as expected with sheen ignored in all three). Class C feature: coverage only.
- **AttenuationTest**: label sheet and grid backdrop agree; the thin R1 samples and plain backdrop regions are ours/pbrt +0.0..+1.3%, ours/Mitsuba +3.0..+3.8%, with Mitsuba -3.0% under pbrt on exactly those regions
  (`roughplastic` vs `coateddiffuse` under a 20 x 20 grid backdrop; ours follows pbrt's stochastic layering). Thick saturated absorbers: ours/pbrt +5.5..+10.6% on R5 1.0-2.0 (pbrt is the outlier, V62; ours/Mitsuba +2.7..+3.7%).
  `R3_ThicknessTex_Mat`: factor only in all three (V79): ours/pbrt +2.1%, ours/Mitsuba +3.5%. Class B: bracketing, not a gate.

## Texture filtering (V51)

Ours: bilinear at LOD 0 (`textureLod: 0`, the default). Mitsuba `bitmap` bilinear is level 0 too; pbrt `imagemap bilinear` picks a mip level from camera ray differentials scaled by max(1/8, 1/sqrt(spp)). At 4096 spp that
is an eighth of a pixel, so even the densest texture here stays on level 0: IORTestGrid / SheenTestGrid backdrops 256^2 over 6 repeats at ~2.4 texels per pixel, AttenuationTest grid 256^2, MetalRoughSpheres 1024^2 over
768 px with flat texel blocks. No difference in any region can be attributed to filtering: the textured backdrops agree with pbrt to 0.0-0.1% (IOR-A) and the MetalRoughSpheres blocks to the printed precision. The only
filtering-related difference is Mitsuba's baked `k` / alpha textures (V78): within one texel of an edge, <= 0.6% on the IOR backdrop strip.

## Bug this wave found

The first MetalRoughSpheres render of ours had the non-metal rows +10..+40% too bright and flat across roughness: with metalness factor 1 `ptMaterialFeatures` saw only metals, so no layered lobe or stochastic mix was
compiled and dielectric texels fell back to Lambert. The adapter now treats a metallic-roughness texture as `0 < metalness < 1` for the feature scan (`pt-megakernel/index.ts`). Fixed numbers above.

## Exporter exactness by texture kind

| kind | pbrt-v4 | Mitsuba 3 |
| --- | --- | --- |
| base colour (sRGB) | exact: spectrum `imagemap`, bilinear, `encoding sRGB`, wrap repeat/clamp (mirror throws) | exact: `bitmap` bilinear, `raw false`, wrap repeat/clamp |
| base colour on a conductor | exact (`reflectance` texture) | k baked per texel (PFM), filtered in k (V78) |
| roughness (G) | exact: float imagemap per channel, alpha = G x G by a `scale` texture | alpha = (factor G)^2 baked into a float PFM (conductor, dielectric end through <= 8 levels) |
| metalness (B) | exact: float imagemap as the `mix` amount (stochastic pick, exact in expectation) | `blendbsdf` weight bitmap (PFM) |
| emissive (sRGB) | linear-radiance PFM through `AreaLightSource "diffuse" "string filename"`; unit tested, no scene renders it | radiance bitmap (PFM, raw); same |
| alpha MASK | exact hard cutoff (gain-1000 alpha texture) | thresholded 0/255 bitmap on an up to 8x finer grid, `nearest`, `volpath`; contour exact to 1/16 texel at 128 px |
| alpha BLEND | class C: `coverage` renders it opaque in both | same |
| normal map | not exported (throws; `coverage` drops it) | same |
| thickness texture | not read by any renderer (V79) | same |

Reproduce: `node bench/adapters/export-scenes.ts <scene>`; ours `node bench/adapters/matrix.ts --renderers pt-megakernel --scenes IORTestGrid-A,... --spp 4096 --extra pt_tint=0,pt_coverage=1`; references
`node bench/references/render.ts --scenes IORTestGrid,SheenTestGrid --renderers pbrt,mitsuba --spp 4096 --coverage`, `... --scenes AttenuationTest --spp 4096`, `... --scenes MetalRoughSpheres --spp-pbrt 4096 --spp-mitsuba 1024`,
`... --scenes IORTestGrid-A --spp 4096`, `... --scenes tex-mask --spp 4096`.
