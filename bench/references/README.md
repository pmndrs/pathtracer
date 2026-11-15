# Native reference renders (pbrt-v4, Mitsuba 3)

Ground truth for the bench. Scenes come from `bench/scenes` (tier 0 analytic scenes, tier 2 `sponza-white`), are exported by `bench/exporters`, rendered on the
CPU, and cached here as `<scene>/<renderer-id>-<spp>spp.pfm` (git-ignored) plus a committed JSON sidecar
(version, command, spp, seconds, settings, exporter approximations, host). `agreement.md/json` hold the pbrt-vs-Mitsuba
numbers.

## Setup (once; everything lives in git-ignored `.cache/native`)

```
git clone --recursive https://github.com/mmp/pbrt-v4 .cache/native/pbrt-v4
git -C .cache/native/pbrt-v4 apply ../../../bench/references/pbrt-no-rr.patch   # adds "bool rr" and "integer rrdepth" to the path integrator
cmake -S .cache/native/pbrt-v4 -B .cache/native/pbrt-v4/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DPBRT_NO_GPU=ON
ninja -C .cache/native/pbrt-v4/build
python3 -m venv .cache/native/mitsuba-venv && .cache/native/mitsuba-venv/bin/pip install mitsuba numpy
```

## Run

```
pnpm --filter @pathtracer/bench references -- --detach                      # all Tier 0 scenes, pbrt + Mitsuba llvm_ad_rgb, 4096 spp
pnpm --filter @pathtracer/bench assets:fetch                                # once: Sponza (bench/assets/MANIFEST.json, sha256-verified)
pnpm --filter @pathtracer/bench references -- --scenes sponza-white --renderers pbrt,mitsuba --spp-pbrt 1024 --spp-mitsuba 4096 --detach   # Tier 2, ~70 min
pnpm --filter @pathtracer/bench references -- --scenes cornell --renderers pbrt,mitsuba-spectral --spp 512 --force
pnpm --filter @pathtracer/bench references:compare -- --write               # furnace/analytic checks + agreement table
```

Renderer keys: `pbrt` (pbrt-v4-cpu), `mitsuba` (llvm_ad_rgb), `mitsuba-scalar` (scalar_rgb), `mitsuba-spectral` (scalar_spectral).

## Settings that make the renderers agree (verified, see agreement.md)

- **Radiometric scale.** pbrt-v4 is spectral. `rgb L [1 1 1]` becomes an RGBIlluminantSpectrum in the sRGB colour space
  (D65 illuminant) and the light divides its scale by the illuminant's photometric value, so Y = 1. With pbrt's defaults
  (`Film "rgb"`, sensor `cie1931`, `iso 100`, exposure 1, no white balance) the imaging ratio is exactly 1 ("passthrough"),
  and a white L = 1 furnace reads 1.0000 (tested to 1e-4). Do not set `sensor`, `iso`, `whitebalance` or exposure.
- **Handedness.** pbrt is left-handed: the exporter prepends `Scale -1 1 1` to the camera transform; world geometry is
  written unchanged. Verified with the Cornell box (red wall on the left in both renderers).
- **FOV.** pbrt `fov` is the shorter image axis; Mitsuba gets `fov_axis y` explicitly.
- **Depth.** pbrt `maxdepth` = number of scattering events with next-event estimation (1 = direct lighting only);
  Mitsuba `max_depth` counts the camera-ray emitter hit as depth 1, so Mitsuba `max_depth = maxDepth + 1`
  (verified: quad-light direct lighting agrees to 1e-4 only with that offset).
- **Russian roulette.** Stock pbrt-v4 `path` always applies RR for depth > 1 and has no parameter to disable or move it;
  the patch adds `"bool rr" false` and `"integer rrdepth" N` (RR when `depth > N`, stock = 1). Without `rrStartDepth`
  Mitsuba `rr_depth` is 1e6 (never fires). With `rrStartDepth = D` (sponza-white: 4) RR first fires at path vertex D
  (1 = first surface hit): pbrt `rrdepth = D - 1`, Mitsuba `rr_depth = D` (must be > 0). Verified by variance of a
  1-bounce furnace (pbrt rrdepth 0 raises the spread 5x, rrdepth 1 and 2 do not; Mitsuba rr_depth 1 does, 2 does not).
  The survival probabilities differ (pbrt q = 1 - max(beta)); RR is unbiased either way, only noise differs.
- **spp.** pbrt `zsobol` rounds spp up to a power of two, Mitsuba `multijitter` to a square count (32 -> 35 with a
  warning): use 1024 / 4096 (both exact) for references.
- **Meshes.** `file` meshes (Sponza) are loaded by `bench/scenes/gltf-loader.ts`, un-welded with flat geometric normals,
  and written as the same binary PLY to pbrt and Mitsuba; no shading-normal differences are possible.
- **Two-sidedness.** pbrt materials are two-sided, Mitsuba BSDFs are one-sided: Mitsuba diffuse/conductor are wrapped
  in `twosided`. Emitters are one-sided in both unless `twoSided` (Mitsuba: second flipped copy).
- **Pixel filter.** `box` radius 0.5 in both.
- **Roughness.** alpha = roughness²; pbrt `remaproughness false`; Mitsuba `distribution ggx`, `alpha`.
- **Environment.** pbrt-v4 only reads square equal-area octahedral images (and looks up nearest); the exporter resamples
  the equirect PFM into 512x512 octahedral in world coordinates. Mitsuba reads an equirect PFM (bilinear) but looks rows up
  at `v * (H - 1)` instead of `v * H - 0.5` (a vertical shift of `0.5 - v` texels: +0.5 at the zenith, 0 at the equator), so the
  exporter resamples the rows to `v = j / (H - 1)` first (`equirectForMitsuba`, DIVERGES V50). Unfixed it read a sun-lit plane
  +0.39% bright. Checked on a diffuse plane under the env-sphere sky: analytic irradiance (CPU quadrature of the bilinear map) vs
  ours 3e-6, Mitsuba 1e-5, pbrt -0.05% R/-0.06% G (the octahedral map itself is exact to 3e-5; the rest is pbrt's spectral round trip).
  Rotation: Mitsuba's `rotate y` equals our `rotationYDeg` convention to 3e-6 (probed).
- **Textures (P1.4c).** `diffuse.reflectanceTexture` is the only expressible texture (white `reflectance`; PNG written by `exporters/png.ts`).
  pbrt `imagemap` with `filter "bilinear"`, `wrap "repeat"`, `encoding "sRGB"`; Mitsuba `bitmap` with `filter_type bilinear`, `wrap_mode repeat`,
  `raw false`. Uvs go in the PLY: our uv origin is the glTF top-left, pbrt flips t (v written as 1 - v), Mitsuba reads it as is (checked by
  rendering an orange marker in each tile's top-left cell and comparing the three images). Mitsuba never uses mip maps; pbrt `bilinear` picks a mip level
  from the camera ray differentials scaled by max(1/8, 1/sqrt(spp)), so keep texel density low (tex-checker: <= ~8 texels per pixel) when you want pbrt
  on level 0; `tex-checker-hf` deliberately exceeds it. Timings at 4096 spp, 320x240: pbrt 27 s, Mitsuba 15 s.
- **`openpbr` materials and media (P2.20).** `bench/exporters/openpbr.ts` + `pbrtOpenPbr` / `mitsubaOpenPbr` map tier1.md section 2: metal ->
  `conductor` / `roughconductor` (eta 1, k from the reflectance), glass -> `dielectric`, thin-walled -> `thindielectric`, dielectric over diffuse ->
  `coateddiffuse` / `roughplastic` (`nonlinear true`), clearcoat on a metal -> `coatedconductor`, fractional metalness -> `mix` / `blendbsdf`,
  thin-walled `subsurface` and dispersion -> pbrt only. Class C throws; `--coverage` (`ExportOptions.coverage`) writes the closest
  approximation and every deviation is a `coverage:`-prefixed entry of the sidecar's `approximations`. The checks live in `openpbr-check.ts`
  (`node bench/references/openpbr-check.ts [--scenes a,b] [--spp 256] [--coverage] [--furnace]`, 128x128, a few seconds per scene).
  - **Media.** An absorption-only interior (`transmission_color` at `transmission_depth` metres; sigma = -ln(T)/depth) is `MakeNamedMedium
    "homogeneous"` (`sigma_s` 0) + `MediumInterface "med-<material>" ""` on the shape and `Integrator "volpath"` in pbrt, and a homogeneous `interior`
    medium + `volpath` in Mitsuba; scenes without media keep `path`. The mesh must be closed with outward normals (no nesting). Mitsuba's `rgb`
    medium values must be in [0, 1]: sigma_t is `scale` times the normalised rgb. pbrt uplifts RGB `sigma_a` spectrally (V62): grey absorbers are
    exact, saturated ones are off by up to 8% per channel (the closed form is Beer's law through an ior 1.5 slab, (1-R)^2 a / (1-R^2 a^2)).
  - **`volpath` RR.** The no-RR patch covers `volpath` too (`"bool rr" false`, `"integer rrdepth"`, V59); verified by the same one-bounce furnace
    variance (rrdepth 0 gives 3.8x the pixel std, `rr false` / rrdepth 1 / rrdepth 2 equal stock). Mitsuba `volpath` takes the same `max_depth`
    (+1) and `rr_depth` as `path`. After pulling this change rebuild pbrt: `git -C .cache/native/pbrt-v4 apply` is only needed on a fresh clone
    (the patch file is `git diff` of the working tree; `ninja -C .cache/native/pbrt-v4/build`).
  - **`coateddiffuse` / `coatedconductor`.** pbrt's defaults are not energy conserving: `thickness` 0.01 absorbs even with `albedo` 0 (white furnace
    0.9164 converged), and `maxdepth` 10 truncates the walk (0.9277 smooth with thickness 1e-6). The exporter writes `thickness` 1e-6, `maxdepth` 64,
    `nsamples` 1 (`LAYERED_DEFAULTS`; `ExportOptions.layered` overrides). White-base furnace, ior 1.5, 128x128 at 256 spp: smooth coat pbrt
    maxdepth 10 / 16 / 32 / 64 / 128 = 0.9277 / 0.9845 / 0.9994 / 0.9997 / 0.9997, Mitsuba roughplastic 1.0011; rough coat (roughness 0.3)
    0.8842 / 0.9264 / 0.9357 / 0.9358 / 0.9358, Mitsuba 0.9191 (single-scatter GGX loss amplified by the internal bounces; not a maxdepth effect).
  - **Agreement (pbrt vs Mitsuba, centre or whole image, 256 spp):** thin sheet 0.92320/0.92308/0.92297 vs 0.92311 (closed form 0.92308); grey slab
    0.2765 vs 0.2766 (0.2765); emission 2.9996/1.4988/0.7490 vs 3/1.5/0.75 (pbrt's spectral round trip -0.01..-0.14%); grey metal +0.08%; rough
    glass -0.04%; metalness 0.5 mix -0.1%; anisotropic metal +0.05% and relMSE 4.5e-4 (0.20 with alpha_u/alpha_v swapped); coloured metal -0.7% red
    (spectral uplift); clay under a coat red -1.0% (V61).

- **Samplers.** pbrt `zsobol`, Mitsuba `multijitter` (both unbiased; differences are noise, not bias).

## Tier 1 wave 1 (P2.32a; Nov 14 run, 4096 spp both renderers, one native render at a time, load avg 5 before / 20-30 during the all-core renders)

```
node bench/references/render.ts --scenes AttenuationTest-notex,TransmissionThinwallTestGrid-notex,EmissiveStrengthTest-notex,MetalRoughSpheresNoTextures,ClearCoatTest-notex --renderers pbrt,mitsuba --spp 4096
node bench/references/render.ts --scenes IORTestGrid-notex,SheenTestGrid-notex --renderers pbrt,mitsuba --spp 4096 --coverage     # class C: coverage, not references
```

A 64 spp pass of all 14 renders took 60 s in total (2-6 s each), so 4096 spp costs 2.5-9 min per render; no scene needed the 1024 fallback. The
sidecar has no "coverage" flag: coverage renders are recognised by their `coverage:`-prefixed `approximations` (IOR and Sheen only).

| scene (film) | pbrt s / Mitsuba s | mean RGB pbrt | mean RGB Mitsuba | relMSE M vs P (floor) | relBias M/P | 16x16 block-mean max |
|---|---|---|---|---|---|---|
| AttenuationTest-notex (600x720) | 265 / 252 | 0.2639 0.3002 0.3890 | 0.2705 0.3005 0.3960 | 2.43e-2 | +1.45% | 19.7% |
| TransmissionThinwallTestGrid-notex (1000x500) | 478 / 436 | 0.3750 0.3904 0.4639 | 0.3752 0.3906 0.4640 | 8.6e-4 | +0.03% | 0.73% |
| EmissiveStrengthTest-notex (1120x400) | 548 / 285 | 0.2761 0.5311 0.8294 | 0.2763 0.5313 0.8293 | 1.8e-4 | +0.02% | 0.12% |
| MetalRoughSpheresNoTextures (768x768) | 275 / 178 | 0.2910 0.2678 0.2390 | 0.2889 0.2656 0.2365 | 1.5e-3 | -0.85% | 11.8% |
| IORTestGrid-notex (640x768, coverage) | 347 / 373 | 0.2962 0.3069 0.3622 | 0.2965 0.3072 0.3625 | 5.2e-4 | +0.09% | 0.77% |
| IORTestGrid-notex-A (640x768, class A only: S0.25 spheres dropped) | 271 / 238 | 0.3368 0.3487 0.4104 | 0.3370 0.3489 0.4105 | 3.1e-4 | +0.0% | - |
| AnisotropyStrengthTest-notex (768x768, class A: 49 white metals, P2.26; 4096 spp, load avg 14-20) | 259 / 173 | 0.4970 0.5109 0.5941 | 0.4942 0.5079 0.5902 | 1.31e-3 | -0.58% | 11.6% (16x16), 5.8% (32x32) |
| ClearCoatTest-notex (1024x384) | 195 / 141 | 0.2984 0.2465 0.2918 | 0.2885 0.2484 0.2919 | 2.5e-3 | -0.95% | 17.0% |
| SheenTestGrid-notex (768x768, coverage) | 423 / 280 | 0.2422 0.2495 0.3452 | 0.2414 0.2504 0.3421 | 6.0e-4 | -0.35% | 5.2% |

0 non-finite pixels in all 14 images. pbrt writes negative channel values (spectral to sRGB gamut): Attenuation 24178, ClearCoat 3438, MetalRough
1512, Sheen 21; Mitsuba none. Only Thinwall, Emissive and IOR are pure noise floors (relMSE 1.8e-4..8.6e-4, relBias <0.1%); the other four
contain real pbrt-vs-Mitsuba model differences, so their relMSE is not a noise floor and a gate against pbrt there needs the region split below.
Approximated materials (sidecar `approximations`): every dielectric-over-diffuse material is pbrt `coateddiffuse` vs Mitsuba `roughplastic`
(Emissive 5 cubes, MetalRough 84 rows, IOR 8 + 10 class C, ClearCoat 3, Sheen 16); Attenuation's five R1 thin samples are exported untinted
(depth 0 tint); IOR's ten specular 0.25 spheres and Sheen's twelve fuzz spheres are `coverage:`.

Per-region verdicts (masks = projected bounds of each glTF material group, shrunk 15%, nearer groups win; there is no material-id image yet):

- **Attenuation: pbrt is not valid for thick saturated absorbers (V62).** Tint (0.1, 0.5, 0.9) at 1 m: sigma = 2.3 / 0.69 / 0.105 per metre. R5 rows
  1.0 / 1.5 / 2.0 (rows follow Beer's law ordering in both: darker with attenuation 1.0 < 1.5 < 2.0 in the file): pbrt red -0.006 (negative),
  Mitsuba 0.016..0.040; blue pbrt 8-12% lower; green agrees to 0.1-0.5%; luminance M/P +10 / +14 / +17%. Thin rows (0.5, 0.25): +3.8 / +1.4%. The five
  class B R1 thin samples agree to 0.06-0.25%. Use Mitsuba (per-channel, matches the closed form) for volume gates, or fit pbrt `sigma_a` spectra.
- **MetalRoughSpheres:** grey metal row by roughness 0 / .17 / .33 / .5 / .67 / .83 / 1: M/P -0.12 / -0.09 / -0.09 / -0.23 / -0.65 / -1.65 / -4.07% (Smith G,
  see above). Gold (blue base 0.012): -3..-10% luminance, blue -12..-25%, red/green -1..-8%: spectral uplift of a saturated reflectance in pbrt.
  The 84 class B rows (dielectric over diffuse and metalness mixes): M/P -3.0% overall, worst single rows -12..-16%.
- **ClearCoat (Simple row):** Coating (coat-only, class A) 0.02%; Base -2.7% (R -7%, G +2%); Coated -9.1% (R -15%, G +22%): class B layered models differ by colour.
- **IOR:** 20 regions, M/P between -0.20% and +0.58%, relMSE 1.7e-6..1.7e-3 (black bases lowest). Specular 0.25 (class C) agrees as well as specular 1 because both exporters use the same IOR-based F0 scale.
- **IOR-A variant** (`IORTestGrid-notex-A`, the ten S0.25 spheres dropped so no class C reflection reaches the class A spheres; `node bench/references/render.ts --scenes IORTestGrid-notex-A --renderers pbrt,mitsuba --spp 4096`): all ten class A spheres M/P within -0.27..+0.52%; our megakernel agrees to relMSE 2.7e-4 (pbrt) / 2.1e-4 (Mitsuba) against the 3.1e-4 floor, relBias +0.03% / -0.03%. This is the gating IOR reference; the full scene stays coverage for class C (specular 0.25).
- **AnisotropyStrengthTest-notex** (P2.26; `node bench/references/render.ts --scenes AnisotropyStrengthTest-notex --renderers pbrt,mitsuba --spp 4096`; the label plane dropped, no backdrop in the model so the 49 spheres hang in the sky; film 768x768;
  ours: `node bench/adapters/matrix.ts --renderers pt-megakernel --scenes AnisotropyStrengthTest-notex --spp 4096 --extra pt_tint=0 [--extra pt_tangent=uv]`, 20 s). Grid: column x = -3..3 is the glTF roughness 0..1 (step 1/6), row y = 0..6 the anisotropy strength 0..1 (row 0 and the
  roughness 1 column are isotropic); 36 of the 49 materials are anisotropic, all class A (white pure metal). **Floor** (Mitsuba vs pbrt, whole image) relMSE 1.31e-3, so the gate is 1.97e-3 (1.5x); per sphere median floor 4.8e-4, max 1.2e-3. The pbrt-vs-Mitsuba difference is
  the large-alpha end (Smith G, as MetalRoughSpheres): luminance M/P -2.6 .. -2.8% in the roughness 1 column, -1.1 .. -2.0% along strength 1 (alpha_t = 1, alpha_b = 0), 0 .. -0.5% elsewhere, whole image -0.58%. 0 non-finite pixels, no negative pbrt channels.
  **Ours vs pbrt** (tangent = glTF TANGENT, the default / dp/du per triangle, `pt_tangent=uv`): relBias +0.02% / +0.03%, relMSE 5.7e-4 / 2.7e-4 (both under the gate), 16x16 block means max 2.05% / 2.05% (rms 0.21% / 0.18%) against the references' own 11.6% (rms 1.76%); per sphere the
  luminance is within 0.2% everywhere, relMSE median 2.3e-4 / 1.6e-4 (max 4.5e-3 / 7.5e-4). With the vertex TANGENT 12 of the 49 spheres (columns 0-2, rows 2-6, the most anisotropic smooth ones) are above 1.5x their own floor (7e-4 .. 4.5e-3 against 4e-4 .. 9e-4): the
  flat-shaded sphere's triangles have a dp/du a few degrees off the smooth TANGENT (V75), which moves the stripe, not the mean; with per-triangle dp/du none is. Ours vs Mitsuba: relBias +0.63% (= Mitsuba's own -0.58% against pbrt), relMSE 1.5e-3 / 1.2e-3 (floor 1.31e-3). Class A gate against pbrt:
  passes the section 6 rule with `pt_tangent=uv` and the whole-image numbers with the default; the bias against Mitsuba is Mitsuba's G.
- **Sheen:** all 16 spheres M/P -3.6..-4.0% including sheenColor 0: the class B base dielectric, not sheen (neither renderer has it).
- **Thinwall:** 10 regions within +-0.1% (relBias), IOR 1.0 thin sphere invisible (centre pixel -0.1% / -0.2% of the backdrop).
- **Emissive:** G at the five cube centres (strength 1, 2, 4, 8, 16) 0.481 / 1.013 / 2.011 / 4.011 / 8.009 in pbrt, 0.483 / 1.013 / 2.012 / 4.014 / 8.016 in
  Mitsuba: x2 per step both (0.5 x strength + ~0.01 of rough specular sky); R and B scale with 0.1 and 0.9 the same.
- **Thin-wall closed form** (uniform env L=1, 256 spp, thin spheres vs backdrop only): each sphere shows two sheets in series, T2 = T^2 / (1 - (1-T)^2), T = (1-R)/(1+R),
  pixel = T2 Lbg + (1 - T2). Measured/expected: IOR 1.33 -1.8%, 1.5 -2.9%, 1.76 -3.5%, 2.42 -2.7% (pbrt; Mitsuba within 0.1-0.3% of it): the common negative offset is
  the sphere reflecting backdrop light onto the backdrop (not in the expression), same in both.

## Systematic differences to expect

- pbrt-v4 spectral uplift of saturated RGB reflectances (red/green walls, copper) differs slightly from Mitsuba's RGB
  mode; pbrt can output tiny negative values on saturated colours (spectral -> sensor -> sRGB). `scalar_spectral`
  Mitsuba is the cross-check for this.
- Nearest vs bilinear environment lookup (sun hot spot is 5 degrees wide; effect is sub-percent). pbrt's octahedral point
  sampling also turns the sky/ground horizon step (rows 42-49 of env-sphere) into a staircase: per-row differences of +-5-16%
  there that average out (pbrt/ours is -0.03% over the sky above the horizon band).
- env-sphere pbrt vs Mitsuba/ours: clay +1.4%, copper +1.65% brighter in pbrt (spectral uplift of saturated reflectances, as
  Cornell's 0.69%), ground -0.17%, sky -0.03%: the whole-image -0.38% is that, not the map.

## Measured (Apple M-series, 18 threads; see agreement.md for the full tables)

Render times at 4096 spp, pbrt-v4 / Mitsuba llvm_ad_rgb: furnace-diffuse 74 s / 36 s, furnace-dielectric 51 s / 20 s,
furnace-conductor 54 s / 42 s, quad-light 72 s / 25 s, env-sphere 266 s (ran concurrently with other jobs) / 75 s,
cornell (512x512) 1077 s / 533 s. Mitsuba scalar_spectral cross-checks: 1024 spp (512 for cornell).

- Furnaces: diffuse reads 0.50000 (expected 0.5) and dielectric 0.9999 (expected 1) in both renderers; quad-light matches
  the closed-form irradiance image to 1e-4 in both.
- Mitsuba (RGB) vs pbrt relMSE: furnaces 2e-8 .. 3e-5, quad-light 1.8e-6, cornell 4.4e-4, env-sphere 6.5e-4 (6.4e-4 before the Mitsuba envmap row fix).

**sponza-white** (Tier 2, 1280x720, 262,266 triangles, maxDepth 8, RR from vertex 4, sun + uniform sky, image mean 0.246):
pbrt-v4 4096 spp 3431 s (57 min) / Mitsuba llvm_ad_rgb 4096 spp 1456 s (24 min), rendered one after the other. Agreement at
equal spp: relMSE 4.1e-3 (eps 1e-2), SMAPE 0.041, mean bias Mitsuba/pbrt +0.22%. The relMSE is pure noise: Mitsuba 4096 vs
pbrt 1024 / 2048 / 4096 spp gave 1.03e-2 / 6.2e-3 / 4.1e-3, i.e. 8.4/spp (pbrt) + 2.0e-3 (Mitsuba 4096), so the
systematic part is ~0 apart from the small mean bias (the spectral-vs-RGB effect seen on Cornell). Intermediate pbrt
runs took 1284 s (1024 spp) and 1415 s (2048 spp), faster per sample than the 4096 run: wall times on this Mac vary
with whatever else runs (other agents), so quote them as indicative.

Explained systematic differences:

1. **Smith G term.** pbrt-v4 uses height-correlated G = 1/(1+L(wo)+L(wi)); Mitsuba's VNDF weight uses separable
   G1(wi)·G1(wo). For alpha = 0.49 the single-scatter directional albedo (computed numerically) differs by -0.2% at
   30 deg, -0.6% at 45, -1.7% at 60, -3.3% at 70 deg incidence: exactly the radial trend of the rough-conductor furnace
   (-0.003% at the centre to -1.6% at the corners, mean -0.35%). pbrt matches glTF's height-correlated V_SmithGGXCorrelated.
2. **Spectral vs RGB.** pbrt is spectral. Mitsuba scalar_spectral matches pbrt on the saturated diffuse clay sphere
   (0.2775 vs 0.2771) while Mitsuba RGB reads 4% lower in red there, and on Cornell (rel bias +7e-5 spectral, +6.9e-3 RGB).
3. **Coloured conductor eta/k** (copper) cannot be matched in Mitsuba spectral variants (sigmoid uplift of RGB eta/k
   gives a different spectrum: copper reads 0.24/0.12/0.17 instead of 0.34/0.26/0.33). Grey eta/k are exported as a
   constant `spectrum`, which is exact everywhere.
4. pbrt can write tiny negative pixel values on saturated colours (spectral -> sRGB gamut), e.g. -0.0017 on the Cornell red wall.
5. Facet-trapped internal reflections: flat-shaded dielectric sphere needs maxDepth 1000 (0.1% energy lost at 100).

## Tier 1 wave 2 (P2.24: textured, unmodified models; Nov 14, 4096 spp, one native render at a time)

Results and verdicts: `bench/results/TIER1-WAVE2.md`. Scenes `IORTestGrid`, `IORTestGrid-A` (class C spheres dropped, the gate), `SheenTestGrid`, `AttenuationTest`, `MetalRoughSpheres` and the synthetic `tex-mask`.

```
node bench/references/render.ts --scenes IORTestGrid,SheenTestGrid --renderers pbrt,mitsuba --spp 4096 --coverage     # class C spheres / sheen: coverage
node bench/references/render.ts --scenes AttenuationTest,IORTestGrid-A,tex-mask --renderers pbrt,mitsuba --spp 4096
node bench/references/render.ts --scenes MetalRoughSpheres --renderers pbrt,mitsuba --spp-pbrt 4096 --spp-mitsuba 1024   # Mitsuba: 31 s at 64 spp (the 500k-triangle mesh and 8 bitmaps), 4096 would be ~33 min
```

- **Textures in `openpbr` materials** (`exporters/textures.ts`; the exporters wrote only `diffuse.reflectanceTexture` before): base colour, metallic-roughness and emissive slots, base colour alpha MASK. Files are `tex-<texture>[-role].png|pfm`
  next to `scene.pbrt` / `scene.xml`; colour PNGs are written without alpha. A base colour texture needs `baseColor` [1,1,1]; normal maps throw (`coverage` drops them); `mirror` wrap throws (pbrt-v4 has none).
- **pbrt:** float `imagemap` per data channel (channel 0 is what a float texture reads: grey PNG of G and of B, `encoding linear`), roughness alpha = G x G by a `scale` texture (pbrt's own remap is sqrt, so `remaproughness false`), metalness as the `mix` amount
  texture, alpha MASK as a shape `"texture alpha"` = `scale(2, mix(0.5 - K cutoff, imagemap(A, scale K opacity), 0.5))`, K = 1000 (pbrt keeps a hit with probability alpha when 0 < alpha < 1). Emission from a texture: a linear PFM in `AreaLightSource "diffuse" "string filename"`.
- **Mitsuba:** no texture arithmetic, so the baked quantities are PFMs read `raw`: conductor `k` from the base colour, alpha = (factor G)^2, metalness `blendbsdf` weight; a roughness texture over a dielectric becomes a `blendbsdf` chain of `roughplastic`s over at most 8 levels (`roughplastic`
  takes a float alpha). `mask` BSDF with a thresholded opacity bitmap (`nearest`) and `volpath` (shadow rays pass through a `mask` there only). See DIVERGES V77-V79.
- Checks: `bench/exporters/textures.test.ts` (files and numbers per texture kind), `tex-mask` (pbrt vs Mitsuba vs ours: +0.02% / +0.16%, shadows through the holes agree).

