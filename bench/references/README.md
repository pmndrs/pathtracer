# Native reference renders (pbrt-v4, Mitsuba 3)

Ground truth for the bench. Scenes come from `bench/scenes/tier0`, are exported by `bench/exporters`, rendered on the
CPU, and cached here as `<scene>/<renderer-id>-<spp>spp.pfm` (git-ignored) plus a committed JSON sidecar
(version, command, spp, seconds, settings, exporter approximations, host). `agreement.md/json` hold the pbrt-vs-Mitsuba
numbers.

## Setup (once; everything lives in git-ignored `.cache/native`)

```
git clone --recursive https://github.com/mmp/pbrt-v4 .cache/native/pbrt-v4
git -C .cache/native/pbrt-v4 apply ../../../bench/references/pbrt-no-rr.patch   # lets "bool rr false" disable RR
cmake -S .cache/native/pbrt-v4 -B .cache/native/pbrt-v4/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DPBRT_NO_GPU=ON
ninja -C .cache/native/pbrt-v4/build
python3 -m venv .cache/native/mitsuba-venv && .cache/native/mitsuba-venv/bin/pip install mitsuba numpy
```

## Run

```
pnpm --filter @pathtracer/bench references -- --detach                      # all Tier 0 scenes, pbrt + Mitsuba llvm_ad_rgb, 4096 spp
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
- **Russian roulette.** Stock pbrt-v4 `path` always applies RR for depth > 1 and has no parameter to disable it; the
  patch adds `"bool rr" false`. Mitsuba `rr_depth` is set to 1e6 (never fires).
- **Two-sidedness.** pbrt materials are two-sided, Mitsuba BSDFs are one-sided: Mitsuba diffuse/conductor are wrapped
  in `twosided`. Emitters are one-sided in both unless `twoSided` (Mitsuba: second flipped copy).
- **Pixel filter.** `box` radius 0.5 in both.
- **Roughness.** alpha = roughness²; pbrt `remaproughness false`; Mitsuba `distribution ggx`, `alpha`.
- **Environment.** pbrt-v4 only reads square equal-area octahedral images (and looks up nearest); the exporter resamples
  the equirect PFM into 512x512 octahedral in world coordinates. Mitsuba reads the equirect PFM directly (bilinear).
- **Samplers.** pbrt `zsobol`, Mitsuba `multijitter` (both unbiased; differences are noise, not bias).

## Systematic differences to expect

- pbrt-v4 spectral uplift of saturated RGB reflectances (red/green walls, copper) differs slightly from Mitsuba's RGB
  mode; pbrt can output tiny negative values on saturated colours (spectral -> sensor -> sRGB). `scalar_spectral`
  Mitsuba is the cross-check for this.
- Nearest vs bilinear environment lookup (sun hot spot is 5 degrees wide; effect is sub-percent).

## Measured (Apple M-series, 18 threads; see agreement.md for the full tables)

Render times at 4096 spp, pbrt-v4 / Mitsuba llvm_ad_rgb: furnace-diffuse 74 s / 36 s, furnace-dielectric 51 s / 20 s,
furnace-conductor 54 s / 42 s, quad-light 72 s / 25 s, env-sphere 266 s (ran concurrently with other jobs) / 75 s,
cornell (512x512) 1077 s / 533 s. Mitsuba scalar_spectral cross-checks: 1024 spp (512 for cornell).

- Furnaces: diffuse reads 0.50000 (expected 0.5) and dielectric 0.9999 (expected 1) in both renderers; quad-light matches
  the closed-form irradiance image to 1e-4 in both.
- Mitsuba (RGB) vs pbrt relMSE: furnaces 2e-8 .. 3e-5, quad-light 1.8e-6, cornell 4.4e-4, env-sphere 6.4e-4.

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
