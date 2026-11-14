# Megakernel path tracer v1 (P2.4)

`@pmndrs/pathtracer`, `packages/pathtracer/src/integrator/`. Bench adapter `pt-megakernel` (not in the default competitor
set). Tier 0 agreement: `bench/results/reports/pt-tier0/PT-AGREEMENT.md` (viewer:
`?report=/results/reports/pt-tier0/report.json`).

## Shape

- One compute dispatch per sample pass, one thread per pixel, 8×8 workgroups, no sample loop inside a thread (no
  dispatch outlives a watchdog; a pass is the unit of progress).
- Passes of one `encode` share a command encoder and take per-pass uniform records from a 1024-slot ring
  (`hasDynamicOffset`): `queue.writeBuffer` is ordered with `submit`, so a slot is never rewritten before its pass ran.
- Accumulation: f32 `array<vec4f>` (sum, w = sample count). Readback only through `readImage`, outside timed regions.
- Bindings: bvh walk group 0, scene copy group 1, group 2 = `ptParams` (uniform, dynamic offset) + `ptAccum`: 6 storage
  buffers.

```ts
const pt = await PtMegakernel.create(device, caps, { format: sceneFormatOf(trace), layout: copy.layout,
  integrator: { sampler, nee, envSampling, sunVisible, vndf, transport }, minify });
pt.setScene(trace, sceneCopyBuffers); pt.setFilm(w, h); pt.setCamera(cameraDesc);
pt.setLimits({ maxDepth, rrStartDepth }); pt.setSeed(n);
pt.reset(encoder); pt.encode(encoder, passes);
await pt.readImage();
```

## Per path (pbrt-v4 `PathIntegrator`)

Closest hit → emission at the hit (weight 1 for camera and delta paths, else MIS with
`p_light = pmf · dist² / (|cos| · area)`) → black material or `depth ≥ maxDepth` ends → NEE at non-delta vertices →
BSDF sample, `beta *= f·|cos|/pdf`, `etaScale *= etap²` on transmission → Russian roulette
(`q = max(0, 1 − max(beta·etaScale))` when `depth ≥ rrStartDepth`; off when omitted). Mitsuba `max_depth = maxDepth + 1`.

- **NEE:** pick a light (`rtSampleLight`), sample a point/direction (`rtSampleLightLi`), evaluate the BSDF, trace one
  `rtVisible` segment; power heuristic computed as `1/(1 + (other/this)²)` with the ratio clamped to 1e18 (no f32
  overflow). Delta sun unweighted.
- **Offsets:** Wächter-Binder along the geometric normal on the side of the new direction; shadow rays end at an offset
  target point.
- **Radiance and beta** are vec4 (`PtRadiance`, spec §6).
- **Camera:** pinhole per schema (`yFovDeg` vertical, look-at, +Y up, right-handed); one uniform jitter per pixel per
  sample (box 0.5); image row 0 = top.

## Sampler

Owen-scrambled Sobol pairs (`dimensions.ts`; WGSL constants generated from the same table): pair 0 pixel jitter, pair 1
reserved (lens/time), then per vertex `v` pairs `2 + 4v + k`: k=0 light point, k=1 BSDF direction, k=2 (light select,
BSDF component), k=3 (RR). Each pair has its own seed, shuffle (16-bit; the block number joins the seed) and Owen
scramble, so the first 2^m samples of every pair are a (0,m,2)-net. `pcg` is the debug setting.

## Settings

Pipeline constants: `sampler` (owen-sobol | pcg), `nee`, `envSampling` (uniform | importance | none), `envMap` (compile
the environment-map code), `sunVisible`, `vndf`, `transport`. Kernel text (host options, not pipeline constants): `minify`, `prune`, `shortenNames` (all default true; page args `pt_prune=0`, `pt_shorten=0`), `sourceMap` (default false; see Size, reading compiler errors). Per dispatch: size, sample index, seed, `maxDepth`, `rrStartDepth`, camera. Bench page args: `pt_sampler`,
`pt_nee`, `pt_env` (uniform|importance|none; the adapter's default is importance for an `hdri` scene, uniform otherwise), `pt_sun_visible`, `pt_positions`, `pt_vndf`, `pt_transport`, `pt_maxdepth`, `pt_rr`, `pt_seed`,
`pt_passes`, `pt_sun_radius` (radians, 0 = delta sun) (`matrix.ts --extra k=v,...`).

## Agreement (1024 spp, M5; timings best effort)

| Scene | relMSE vs Mitsuba | vs pbrt | pbrt-vs-Mitsuba floor | Notes |
|---|---|---|---|---|
| furnace-diffuse | 1.96e-7 | 1.71e-7 | 2.44e-8 | mean 0.50000 (expected 0.5) |
| furnace-dielectric | 1.93e-7 | 1.26e-7 | 1.53e-7 | mean 0.99990 at maxDepth 1000 |
| furnace-conductor | 2.42e-5 | 7.99e-6 | 3.15e-5 | 0.69285 vs pbrt 0.69286; Mitsuba 0.69040 (its Smith G) |
| quad-light | 4.72e-5 | 4.79e-5 | 1.84e-6 | closed form over the plane to 9e-5 |
| cornell | 7.25e-5 | 4.82e-4 | 4.39e-4 | the 0.69% vs pbrt is pbrt's spectral uplift; Mitsuba RGB agrees with us |
| env-sphere | 6.98e-5 | 6.90e-4 | 6.52e-4 | HDR map, luminance importance sampling + MIS (P2.8, `reports/pt-envmap`); relBias +0.01% / -0.38% (Mitsuba after its envmap row fix V50; it read -0.14% before); the -0.38% is pbrt's spectral uplift of the clay/copper (+1.4% / +1.65% there), not the map (P2.8a item 4); at the pbrt floor; uniform sphere sampling 4.61e-4 (6.6x worse at 1024 spp, 23x at 16 spp), same means |
| sponza-white | 8.11e-3 | 8.28e-3 | 4.09e-3 | at the noise floor (reference noise ~2e-3 + ours; 1/spp slope holds); relBias -0.13% vs Mitsuba; delta sun and positions=ray identical (P2.9, `reports/pt-sponza*`) |

0 non-finite pixels. GPU twins (`pt-check`): sampler bit-identical, adapter 0 mismatches, light sampling max rel 1.3e-5,
end-to-end sun/sky plane within 0.4% of a closed form.

Settings A/B at 1024 spp (relMSE vs Mitsuba): Owen-Sobol beats PCG 2.6–2.8× on Cornell and quad-light; NEE off is 34×
(quad-light) and 126× (Cornell) worse with the same means (MIS unbiased); `envSampling=none` puts furnace-diffuse at the
floor (zero variance); `positions=ray` and `vndf=cap` identical on Tier 0; `rrStartDepth=4` on Cornell: same mean,
3.9 s vs 14.2 s, relMSE 1.13e-4 vs 7.25e-5.

## Environment map (P1.4d, P2.8)

Equirect map in the scene copy `data` buffer (layout and the exact pdf in `packages/raytracer/src/scene/env-map.ts`, DIVERGES V48).
Lookup is bilinear (u wraps, v clamps); `rotationYDeg` turns the map about +Y. Importance density is per pixel `Y(L)·sin θ`:
`u.y` picks a row (marginal CDF, binary search), `u.x` the column (conditional CDF), the leftovers place the point inside the
pixel; solid-angle pdf `Y·sinθ_row·norm/sinθ`, `norm = W·H/(S·2π²)`. Escaped rays take the MIS weight
`ptMis(prevPdf, pmf·pdf_env)`. `envSampling: 'uniform'` with a map is the baseline (uniform sphere sampling, the same lookup).
f32 CDFs: a single alias table over 131k pixels was rejected because the pixel index would use 17 of the 24 bits of `u` and
leave ~7 for the alias coin (distorted probabilities). The light-selection power uses the sin θ-weighted mean radiance.

## Material textures (P1.4c)

Compiled only when the scene has textures (`PtMegakernelOptions.textures = copy.textureLayout`; the untextured kernel text and every image
are unchanged: cornell, env-sphere and sponza-white PFMs are `cmp`-identical at 64 spp). Texture arrays and samplers sit in group 1 after the
two scene buffers (`rtSceneCopyLayoutEntries(0, layout)`; `uploadSceneCopy` creates them; `setScene` checks kernel and scene agree).

Per hit: `rtSurface(.., POSITION | SHADING_NORMAL | UV | TEXFRAME)`, the ray cone's width at the hit, the footprint
`width * uvDensity / max(|n . d|, 1e-3)` (uv units), then `rtApplyMaterialTextures` (base colour, alpha into `opacity`, roughness from G,
metalness from B, emission, all times the factors) and, with a normal texture, `rtApplyNormalMap` around the interpolated shading normal; the
frame, NEE and BSDF sampling use that normal, the geometric normal still offsets rays. Settings: `textureLod: 0` (default, base level, the
reference setting) or `'cone'`; `PtLimits.textureConeScale` (default 1; adapter `pt_texture_cone_scale=auto|N`, `auto` = pbrt-v4's
max(1/8, 1/sqrt(spp))). Ray cone: width 0 and spread = pixel angle x scale at the camera, `width += spread * t` at a hit, a delta bounce keeps the
spread, a non-delta bounce adds `PT_CONE_DIFFUSE_SPREAD` = 0.5 rad. Bench args `pt_texture_lod=0|cone`.

Agreement (1024 spp, M5, `tex-checker`, a sRGB checker with an orientation marker on a quad and a sphere, delta sun + uniform sky, maxDepth 3):
relMSE 4.5e-5 vs Mitsuba 4096 spp (pbrt-vs-Mitsuba floor 4.8e-5), 7.9e-5 vs pbrt, relBias 0.00% / -0.04%, block means 0.06% max;
`pt_texture_lod=cone` with the auto scale 3.7e-5. `tex-checker-hf` (1024 px, up to ~70 texels per pixel): lod 0 1.3e-4 vs Mitsuba (2.8e-4 vs pbrt, whose
bilinear filter picks its own mip level; floor 2.0e-4); LOD A/B in scene-copy.md open choice 9. Timing, best effort (load 3-12, not a result):
0.47 ms/spp for both settings at 320x240.

## OpenPBR materials (P2.21 wiring, P2.22, P2.30)

A scene with an `openpbr` material compiles OpenPBR mode (`PtIntegratorOptions.materials: PtMaterialOptions`; the adapter sets it when any
material of the scene has `type: 'openpbr'`, so every other scene keeps the legacy adapter and the same kernel text: `bench/size-snapshot.ts`
diffed plain, env, pcg, readable and textured kernels `cmp`-identical, and the 64 spp PFMs of cornell, env-sphere, sponza-white and tex-checker are `cmp`-identical). The material
adapter (`ptBsdfFromOpenPbr`, WGSL twin `ptBsdfOpenPbrWgsl`; rules in the header of `bsdf/adapter.ts`) takes a uniform `u` per vertex and picks, metal first:

- **metal**: `u < metalness` (the `mix` feature: pbrt `mix`, the metal end with probability `metalness`, so the expectation is the exact linear blend; `metalness`
  0 and 1 are exact; without the feature the legacy 0.5 threshold). Complex-Fresnel GGX conductor, eta 1, k from `base_weight * base_color` (what pbrt's and
  Mitsuba's `reflectance` do; the OpenPBR F82 tint is not evaluated, DIVERGES V55/V64). `u` is the `.y` of the RR dimension pair of the vertex (RR uses `.x`).
- **thick dielectric**: `transmission_weight > 0.5`, GGX dielectric with `specular_ior` and `specular_roughness`. `transmission_color` at `transmission_depth` 0
  (a non-physical tint) is ignored on thick glass. **Interior medium** (`medium` feature): path state `sigma` (vec3, per metre) = `-ln(transmission_color) /
  transmission_depth` of the material the ray entered; `beta *= exp(-sigma * hit.t)` at every hit (distance to the exit hit); a transmission event sets `sigma` to the material's
  when it enters (`wo.z > 0`, the ray came from outside) and to 0 when it exits. No nesting: a ray inside one medium that enters another replaces it;
  no absorption on NEE shadow rays (they cannot cross a glass boundary anyway; a light inside a medium is unsupported). Beer's law needs closed, consistently oriented meshes (outward normals).
- **thin-walled dielectric** (`geometry_thin_walled`, `thin` feature): `PT_BSDF_THIN` = pbrt's `ThinDielectricBxDF`: R' = R + T^2 R / (1 - R^2), T' = 1 - R', delta lobes,
  transmission goes straight through (`wi = -wo`, etap 1). Transmission is tinted by `transmission_color` (setting `transmissionTint`, default on; pbrt and Mitsuba `thindielectric` cannot tint, so a tinted thin wall differs from the references by construction).
- **dielectric over diffuse** (`layered` feature): `PT_BSDF_LAYERED`, the P2.21 layered BSDF with an interface (`specular_ior`, `specular_roughness`, Fresnel scale
  `specular_color * specular_weight`) over a Lambert base `base_weight * base_color`. A coat (`coat_weight > 0.5`) replaces the interface by (`coat_ior`, `coat_roughness`, scale 1) and drops the base's own specular lobe, which is what the pbrt exporter writes (`coateddiffuse`;
  V67). A black base is still a lobe (IORTestGrid). `PtBsdfParams` carries it without growing: `reflectance` base, `eta` the Fresnel scale, `k = (thickness, internal reflectance, 0)`.
- otherwise Lambert (`base`) or black.

Settings: `layering` `'albedo' | 'stochastic'` (library default `albedo`; **the bench adapter defaults to `stochastic`, the reference mode**, pbrt's `coateddiffuse` walk with the exporters'
`maxdepth` 64, `nsamples` 1 and `thickness` 1e-6; page args `pt_layering`, `pt_layer_depth`, `pt_layer_thickness`, `pt_exit_mis`), `albedoBase` `'openpbr' | 'internal'` (`pt_albedo_base`;
`internal` reads `ptInternalReflectanceFit`, Egan-Hilgeman, within 0.012 of the numeric average), `transmissionTint` (`pt_tint`), `coverage` (`pt_coverage=1`: render a scene whose
materials use a lobe we lack, fuzz / thin film / dispersion / subsurface / anisotropy, with that lobe ignored instead of skipping it). A stochastic layered sample has `PT_FLAG_PROPORTIONAL`: the throughput
uses its `f |cos| / pdf` and the MIS density for the next emission hit is `ptLayeredPdf` (pbrt does the same). The seed of the walk is a hash of (pixel, sample, vertex), set once per vertex in a
private variable (`ptLayerSeed`), so `ptEval/ptSample` keep the single-lobe signatures. Features are derived from the scene's material table (`ptMaterialFeatures`), so a scene without thin walls has no thin code, and so on.

Kernel size, minified with short names (`PT_NO_FEATURES` = OpenPBR mode with no lobe): 40.5 KB (legacy 40.47), `thin` +0.5, `medium` +0.3, `mix` +0.04, `layered` albedo 48.3 (+7.9) / albedo + internal 48.4 / stochastic 49.5 (+9.1);
all four lobes: 49.1 albedo, 50.3 stochastic; with an HDR environment map +2.8 (51.9 / 53.1). The unit-test caps are in `integrator-wgsl.test.ts`. The Tier 1 scenes
use the sky map, so a layered scene is 51-53 KB: over the ~50 KB Windows-DXC class that `CLAUDE.md` names; DXC compile time of the layered kernels is unmeasured (open choice 11).

### Results (1024 spp ours on M5, GPU; timing best effort, load 13-27 from other agents' CPU renders, so none of it is a result)

Closed forms (`bench/scenes/ab/openpbr-check.ts`, 256 spp, field of view 1 degree): thin sheet 0.92313 vs 0.92308; thin furnace 1.00000 exactly; Beer slab (0.59016, 0.11521, 0.00922) vs (0.59021, 0.11521, 0.00922);
layered furnace (white, ior 1.5, roughness 0.1, maxDepth 200) 0.99879 stochastic / 0.99945 albedo, 0 non-finite at 1024 spp. GPU twins (`node bench/runner/run.ts --page openpbr-check`): adapter 85 material x u cases 0 mismatches, thin sample
max rel 5.9e-7, layered sample 9.2e-4 / eval 3.5e-6 (albedo), 0 non-finite in 60 000 stochastic samples. The P2.20 `oc-*` synthetic scenes (ours / pbrt / Mitsuba means): oc-thin 0.9231 / 0.9231 / 0.9231, oc-absorb-grey 0.2764 x 3, oc-absorb (0.4609, 0.2303, 0.7380) /
(0.4898, 0.2114, 0.7220) / (0.4607, 0.2302, 0.7376), oc-emission (3, 1.5, 0.75) x 3, oc-metal 0.5385 / 0.5385 / 0.5381, oc-metal-copper 0.5639 / 0.5598 / 0.5635, oc-glass 0.4070 / 0.4068 / 0.4069, oc-mix 0.4666 / 0.4665 / 0.4671, oc-coated-clay (stochastic)
(0.3415, 0.2061, 0.1830) / (0.3332, 0.2050, 0.1821) / (0.3365, 0.2049, 0.1823); oc-coat-metal ignores the coat (0.5385; pbrt `coatedconductor` 0.4771).

Tier 1 wave 1 (`node bench/adapters/openpbr-compare.ts --report <name>`; relBias = mean ours / mean reference - 1; the 64 spp references are noise-dominated, judge the means and block means; 4096 spp where noted):

| Scene | Lobes | relBias vs pbrt / Mitsuba | block-mean rms (32 px) | relMSE vs pbrt / Mitsuba (floor pbrt-vs-Mitsuba) |
|---|---|---|---|---|
| TransmissionThinwallTestGrid-notex (4096 / 4096 spp) | thin + thick glass | +0.02% / -0.01% | 0.11% / 0.09% | 7.04e-4 / 7.45e-4 (8.59e-4): passes the gate rule |
| AttenuationTest-notex (4096 / 4096), `pt_tint=0` | thin + absorbing blocks | +1.49% / +0.04% | 4.69% / 0.43% | 4.09e-3 / 2.02e-3 (5.00e-3) |
| AttenuationTest-notex, tint on (default) | same | -2.02% / -3.42% | 10.1% / 9.3% | 1.8e-2 / 1.6e-2: the thin R1 row is tinted in ours only (V68) |
| EmissiveStrengthTest-notex (64 spp refs) | emission + layered backdrop/specular | +0.03% / +0.01% | 0.17% / 0.32% | |
| MetalRoughSpheresNoTextures (64) | metal, mix, layered | -0.08% / +0.77% | 0.52% / 2.53% | |
| ClearCoatTest-notex (64) | layered coat | -0.84% / +0.12% | 5.27% / 0.95% | the coated red torus is -10% vs pbrt only in red: pbrt's spectral uplift (below) |
| IORTestGrid-notex (64) | layered, specular weight 0.25 | -0.57% / -0.67% | 8.7% / 8.7% | the S0.25 rows are coverage approximations in both references (class C) |
| SheenTestGrid-notex (64), `pt_coverage=1` | layered, fuzz ignored | +0.11% / +0.46% | 0.63% / 2.84% | |

Layering A/B on the class B scenes (relBias vs pbrt / Mitsuba): ClearCoat stochastic -0.84% / +0.12%, albedo + internal -0.59% / +0.36%, albedo +4.50% / +5.50%; MetalRough stochastic -0.08% / +0.77%, internal +2.78% / +3.65%, albedo +9.90% / +10.83%.
Grey-base smooth `coateddiffuse` (ours vs pbrt 1024 spp, oc-coated-clay geometry) 1.0002: the stochastic port matches pbrt. A saturated base (0.5, 0.02, 0.01) reads 0.937 red / 1.028 green / 1.004 blue against pbrt at any roughness (0, 0.06, 0.3): pbrt is
spectral and the walk multiplies the uplifted spectrum, so the three-channel result differs; ours agrees with Mitsuba (RGB) on that torus (relMSE 4.9e-4). Do not chase it in the layered code.

## Size

**40.4 KB minified** (43.2 KB with an HDR env map; unit-test caps 41.5 / 44.5 KB), under the 50 KB Windows-DXC class. Three steps
(P2.10), all text-only and bit-identical in every image (PFMs `cmp`-equal on Tier 0 and sponza-white at 256 spp):

1. Indentation and spaces around `{}(),;` dropped (`minify`, line breaks kept): 63.3 -> 53.5 KB.
2. `prune` (`rtPruneWgsl`): top-level declarations that `ptMain` and the `@group` bindings cannot reach are dropped: -4.7 KB
   (bvh `bvhClosestPointToTriangle`/`OnEdges`, `rtVisibleSegment`, cone helpers, `rtLightPmfOfHit`, `ptBsdfPdf`, unused consts).
   Needs no scene knowledge; a scene-specialised kernel (drop absent light kinds / lobes) would cut more but recompiles per scene.
3. `shortenNames` (`rtShortenWgsl`): top-level struct/fn/const/alias/var names become `A`..`Zz` by use count (not `override`s,
   bindings, `ptMain`, or a name also used as member/parameter/local): -8.3 KB. Line numbers are unchanged by this step;
   pruning shifts lines after a dropped declaration.

**With textures (P1.4c):** +2.9 KB for the lookups, the glTF material semantics, the normal map and the texture frame in `rtSurface`, +160 B per
texture array: 43.7 KB with one array, 44.8 KB with eight, 46.5 KB with an environment map and one array (unit-test caps 45.5 / 47.5 KB).
The untextured kernel is unchanged (40.4 KB, 40.75 KB with uvs in the layout).

**Reading compiler errors (P2.10b).** The size steps keep a source map, like a TS build: `ptMegakernelBuild(caps, options)` returns
`{ code, map }` (`map.lines`: compiled line -> readable line; `map.names`: short -> original name; `map.sources`: the readable text:
comment-free, nothing pruned, long names, `enable` lines included, i.e. what `prune: false, shortenNames: false, minify: false`
compiles). `PtMegakernel.create` (and `RtBatchTracer.create`) go through `rtCreateComputePipelineAsync`: nothing extra runs on
success, and a compile error is thrown as

```
WGSL (pt megakernel) failed to compile:
error at source line 2045 (compiled line 1843): too few arguments in call to 'ptSample2D', expected 3, got 1
  2044 | fn ptSampleLd(p: vec3f, ...) -> vec3f {
> 2045 |     let selectU = ptSample2D(pixelSeed, ...
```

with the original names and line numbers, the compiled line in brackets and two lines of context. By hand: `rtMapCompilationInfo(await module.getCompilationInfo(), map)`
or `rtCheckShaderModule(module, map)`. Lines and names only: columns do not survive minify (`linePos` is the compiled one). `sourceMap: true`
also passes a Source Map v3 as `GPUShaderModuleDescriptor.sourceMap` (browsers may ignore it; Chrome 157 does). Check it on a real
compiler: `node bench/runner/run.ts --page wgsl-map --args inject=1` (exits non-zero by design: Dawn logs the injected error).

Where the bytes are (pruned, readable names): bvh walk and helpers 14.1 KB (`bvhWalkClosest` 3.3, `bvhWalkAny` 2.9, watertight 1.3,
node/triangle/instance loads 2.5), BSDF 9.8 (`ptBsdfSample` 2.9, `ptBsdfEval` 1.6, `ptBsdfFromMaterial` 0.7), integrator 5.0
(`ptPath` 2.7, `ptSampleLd` 1.2), scene copy 7.7 (`rtSurface` 1.9, `rtMaterialAt` 1.8, `RtMaterial` 0.7), light sampling 3.5,
sampler 1.3, Owen-Sobol 1.1, adapter 0.9. An `hdri` scene adds the map lookup, pdf and CDF sampling (+2.8 KB shortened).
`rayMasks: false`. ~250 ms compile on the Mac; DXC unmeasured (P2.10). Print the table: `node bench/size-breakdown.ts [--env]`.

## Open choices (defaults hold until measured)

1. **Kernel size.** Done: 40.4 KB (43.2 KB with a map) by pruning and short names (see Size). Remaining ideas, not needed for the
   cap: lean material decode (~1.8 KB), light-record decode, bvh closest/any specialisation. Settle: `compile_ms` on Windows
   (prune and no-prune, `pt_prune=0`, are the A/B).
2. **`envSampling` default: uniform + MIS or none (pbrt); with a map: importance (adapter default).** Map: importance beats
   uniform 6.1x relMSE at equal spp (+12% time per sample on env-sphere); CDF vs alias tables is open (P2.8a). Keep uniform until a sun-plus-sky scene shows the cost.
   `none`: zero variance on a uniform sky and cheaper NEE; uniform wastes alias picks next to other lights. Setting
   exists. Settle: P2.12 at equal spp and equal time.
3. **Light selection weights** (P1.4f): **settled, default `pbrt` (D22)**, pbrt-v4's default rule. Each infinite light
   (sun, environment) gets 1/(nInfinite+1), and the area lights share the rest by power, where pbrt uses its light BVH. Mitsuba's
   uniform emitter weights are the same idea. Other settings via `lightSelection` / `pt_light_selection`: `power`, `kind`,
   `cap:0.25` / `cap:0.5`. Noise only (means within 0.03%). relMSE at 1024 spp, 3 seeds (pbrt / power / cap:0.5): Cornell,
   quad-light, env-sphere identical; sponza-white 9.33e-3 / 8.12e-3 / 8.12e-3; `lights-mix` (sun+sky key) 1.67e-5 / 2.10e-5 /
   1.29e-5; `lights-mix-lamps` 1.30e-4 / 8.69e-5 / 8.69e-5. `cap:0.5` measured best overall but is our own heuristic; the
   maintainer chose the common rule. Revisit with a light BVH (spatial importance for the area share) or ReSTIR DI (P4).
4. **Triangle lights: uniform area or spherical triangle.** Keep uniform (matches Mitsuba). Spherical is less noisy close
   to big lights, more ALU. Can be a module constant. Settle: close-up large quad light, relMSE at equal spp.
5. **RR outside reference mode.** Suggest `rrStartDepth` 4 (Cornell 3.6× faster at 1.6× relMSE). Setting exists.
   Settle: time to equal error in the quiet pass (P2.13).
6. **One sample per dispatch vs several per thread vs tiles.** Keep one per dispatch (watchdog-safe). Settle in P3 with
   `pt_passes` and spp-per-thread.
7. **4 dimension pairs per vertex** (RR pair half used). Keep; small cost next to traversal.
8. **Hit position from barycentrics** (P1.8). Keep; identical on Tier 0 and on Sponza (relMSE 8.11e-3 both).
9. **Sponza in the adapter (P2.9):** settled in practice: the glb is loaded in the page (the web exporter path).
10. **Sun: cone (default) or delta like the references.** Keep the cone (physical, free); `pt_sun_radius=0` gives delta. Sponza A/B: mean 0.04% apart, same noise. A large close contact shadow could show more; re-check if a Tier 1 scene has one.
11. **Layered kernel size (P2.21 wiring).** A layered scene with an environment map is 51.9 KB (albedo) / 53.1 KB (stochastic, all lobes): the layered BSDF is +7.9 / +9.1 KB over the Tier 0 kernel, 3 KB of it the
    `albedo` table. Ideas if DXC compile time says so: split the stochastic walk's pdf function out of the albedo variant, a smaller table (8x8), drop `ptLayeredPdf` when `exitMis: 'none'`, a wavefront kernel for
    the layered scenes. Settle: `compile_ms` on Windows for `PT_NO_FEATURES`, albedo and stochastic (`wgsl-map`/`competitor` pages report `compile_ms`).
12. **Thin-walled tint and thick-glass tint at depth 0.** Thin glass applies `transmission_color`; thick glass ignores it at `transmission_depth` 0. OpenPBR applies the tint at each transmission event; the references apply none. A/B: `pt_tint=0` vs 1 on AttenuationTest R1 (tinted thin samples).
