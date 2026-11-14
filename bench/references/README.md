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
