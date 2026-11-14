# Tier 1 plan (glTF `KHR_materials_*`)

Status: proposal, 2026-11-14. Sources: research/INDEX.md section 7. Spec: SPEC-draft section 10 ("Tier 1: glTF-Sample-Assets
`KHR_materials_*`, publishes a feature-coverage matrix"). Gate it feeds: PHASES P2, "converges to pbrt-v4 within the
pbrt-vs-Mitsuba noise floor on Tier 0–1". Open question for the maintainer: section 7.

## 0. Summary

- Tier 1 is a feature-coverage matrix plus a gate on the subset that pbrt-v4 and Mitsuba 3 can both render exactly. Not
  every Tier 1 scene can be a reference scene.
- **Class A** (exact in both references, gate candidates): smooth and rough thick dielectric with an absorption-only
  volume; smooth thin-walled transmission; black-base dielectric with `specularFactor` 1 (pure Fresnel reflection); pure
  metals, including anisotropic ones once the tangent frame exists; emissive strength.
- **Class B** (approximate in both, the references disagree): dielectric-over-diffuse (the glTF default non-metal),
  clearcoat on a non-metal, coloured transmission tint.
- **Class B'** (one reference only): dispersion, diffuse transmission (pbrt only).
- **Class C** (no reference): sheen, iridescence, thin-walled rough transmission, `specularFactor` below 1 or
  `specularColor` ≠ 1.
- The first wave needs no textures: sidecar overrides replace the backdrop and label materials with factor-only ones. That
  decouples Tier 1 from P1.4c.
- A sensitivity check decides whether a scene can gate at all: render with the feature stripped and require a gap of at
  least 10x the floor.
- Competitor support is found by ablation (stripped vs full render), not by reading source.
- **Compare OpenPBR-parameter materials, not glTF's own BRDF.** glTF's `fresnel_mix` uses VdotH Fresnel, OpenPBR uses
  albedo-scaled slabs, and pbrt `coateddiffuse` is stochastic and physical; the three differ by a few percent on a plain
  dielectric. glTF is only the asset and parameter source; the exporters read a neutral `SceneDesc` material, as today.
- The OpenPBR 1.1.2 spec has no glTF mapping section ("glTF" and "KHR" do not occur in it). The mapping in section 2 is
  ours; OpenPBR sections are cited per parameter. The Khronos dispersion README says OpenPBR uses the same 20/Abbe transform.

## 1. Models

Pinned commit `C = 8f492390ddc7a2ae221a1369ef560589cf8c70fb` (still HEAD of glTF-Sample-Assets on 2026-10-23, the same
commit as our Sponza).
- Page: `https://github.com/KhronosGroup/glTF-Sample-Assets/tree/C/Models/<Name>`
- Raw GLB: `https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/C/Models/<Name>/glTF-Binary/<Name>.glb`
- `IridescenceDielectricSpheres` and `IridescenceMetallicSpheres` ship only `glTF/` (`.gltf` + `.bin` + `guides.png`).
- Licence rule, as for Sponza: models are downloaded under `bench/assets/downloads/` (git-ignored); CC-BY-4.0 models go in
  THIRD_PARTY.md with attribution. `metadata.json` and the READMEs are CC-BY-4.0 on every model.

### 1.1 Purpose-built "Test" models (preferred)

| Extension | Model | Licence | Size | Instanced tris | Textures | Loader features needed |
|---|---|---|---|---|---|---|
| core metallic-roughness | `MetalRoughSpheresNoTextures` | CC0-1.0 | 284 KB glb | ~1.04 M (102 meshes sharing ~241 KB of geometry) | none; 98 factor-only materials (metallic x roughness) | NORMAL only. Instancing or baking (~3 M un-welded vertices). Pure-metal rows only for class A |
| core (textured) | `MetalRoughSpheres` | CC-BY-4.0 | 10.96 MB | 501,776 | baseColor, metalRough (8 KB each) | UV0, two textures |
| transmission | `TransmissionThinwallTestGrid` | CC0-1.0 | 1.16 MB | 13,982 | `checker.png` on the backdrop only; materials factor-only, roughness 0 | UV0 for the backdrop; override it in wave 1 |
| transmission (rough, tint) | `TransmissionRoughnessTest` | CC-BY-4.0 | 385 KB | 77,792 | 5 PNGs (labels, roughness grids, backdrop) | UV0, metalRough texture, baseColor tint. Coverage, class B |
| transmission (kitchen sink) | `TransmissionTest` | CC0-1.0 | 1.44 MB | 128,775 | 9 small PNGs | UV0, alpha MASK, transmissionTexture. Coverage only |
| volume | `AttenuationTest` | CC-BY-4.0 | 56 KB | 292 | backdrop, labels, thickness (unused by a path tracer) | none for the volume blocks. Wave 1 drops labels and backdrop texture |
| ior | `IORTestGrid` | CC0-1.0 | 2.57 MB | 30,945 | `checker.png` on the backdrop only | spheres: ior 1–2.42 x specular 0.25/1 x (black, T0) and (white, T1) |
| specular | `SpecularTest` | CC-BY-4.0 | 218 KB | 44,828 | labels, grids | black base, roughness 0, `specularFactor` 0..1, `specularColor`. Class C except factor 1 |
| clearcoat | `ClearCoatTest` | CC-BY-4.0 | 252 KB | 37,116 | 6 images incl. normal maps | NORMAL + TANGENT + UV0. Wave 1 uses the factor-only "Simple" spheres |
| sheen | `SheenTestGrid` | CC0-1.0 | 1.80 MB | 21,646 | `checker.png` backdrop | 16 spheres: sheenColor x sheenRoughness |
| iridescence (dielectric) | `IridescenceDielectricSpheres` | CC0-1.0 | 363 KB `.gltf`+`.bin` | 617,406 | `guides.png` (guides only) | 343 factor-only materials; one BLEND |
| iridescence (metal) | `IridescenceMetallicSpheres` | CC0-1.0 | 337 KB | 617,406 | `guides.png` | as above, metals |
| anisotropy | `AnisotropyStrengthTest` | CC0-1.0 | 92 KB | 97,240 | labels only | NORMAL + TANGENT + UV0; 49 spheres, strength x roughness; white metals (metallic defaults to 1) |
| anisotropy (rotation) | `AnisotropyRotationTest` | CC-BY-4.0 | 813 KB | 7,476 | anisotropy + normal textures | coverage |
| anisotropy (disc) | `AnisotropyDiscTest` | CC0-1.0 | 236 KB | 2,788 | `AnisotropyDiscs.png` | anisotropyRotation, anisotropyTexture |
| emissive_strength | `EmissiveStrengthTest` | CC-BY-4.0 | 10 KB | 90 | backdrop | factor {0.1, 0.5, 0.9} x strength {1..16}, black base. Smallest scene |
| dispersion | `DispersionTest` | CC-BY-4.0 | 2.26 MB | 51,827 | backdrop, labels | transmission + volume + ior + dispersion |
| diffuse_transmission | `DiffuseTransmissionTest` | CC-BY-4.0 | 233 KB | 4,980 | 3 small PNGs | `KHR_lights_punctual` sun, 8 `KHR_materials_unlit` materials, one BLEND |

`KHR_materials_diffuse_transmission` is a Release Candidate; the rest are ratified. `KHR_materials_subsurface` (Initial
Draft) and `ScatteringSkull` are out of scope.

### 1.2 Compare sets

`CompareTransmission`, `CompareVolume`, `CompareIor`, `CompareSpecular`, `CompareClearcoat`, `CompareSheen`,
`CompareIridescence`, `CompareAnisotropy`, `CompareEmissiveStrength`, `CompareDispersion`, `CompareMetallic`,
`CompareRoughness`, `CompareBaseColor`, `CompareNormal` (CC0-1.0 plus the Khronos logo mark; 59 KB–1.5 MB). Each is a
"without / with feature" pair that uses textures, sometimes emissive textures and alpha blend: coverage rows after the
textures wave (P2.24), not exact references.

### 1.3 Other

`DragonAttenuation` / `DragonDispersion` (Stanford licence: download locally, do not redistribute; low priority).
`TransmissionOrderTest`, `NormalTangentTest`, `TextureTransformTest`, `TextureSettingsTest`, `AlphaBlendModeTest`: later
feature waves.

### 1.4 Loader and scene-model features needed

| Feature | Needed by | Today |
|---|---|---|
| UV0, TEXCOORD_1 | textured models | `gltf-loader.ts` reads POSITION + indices only |
| NORMAL (smooth) | normal-map and rounded models | flat geometric normals by design; add `normals: 'geometric' \| 'shading'`, keep the default |
| TANGENT | anisotropy, clearcoat ribs, normal maps | none |
| COLOR_0 | `CompareBaseColor` | none |
| Materials + extension parameters | everything | `MaterialDesc` has four types; needs an OpenPBR-subset type |
| Textures and samplers | wave 2 | slots in the scene copy (P1.4c in progress); exporters write none |
| alphaMode MASK/BLEND | several | scene copy reads alpha (P1.4h GPU check open) |
| `KHR_texture_transform` | two Compare models | none |
| `KHR_lights_punctual`, `KHR_materials_unlit` | `DiffuseTransmissionTest` | none |
| Instancing | `MetalRoughSpheresNoTextures`, `AttenuationTest` | loader bakes every instance world-space. P2.19 note: `MetalRoughSpheresNoTextures` has no shared mesh (102 meshes share *accessors*; 98 spheres of 10,598 triangles = 1,038,604 baked, ~75 MB of positions + normals) and only `AttenuationTest`/`AnisotropyRotationTest` reuse one mesh once, so baking costs nothing; `SceneDesc` has no shared-mesh instances, so an instance path is deferred until a model needs it |
| Camera and lights | all (only `TransmissionTest` has a camera, none has lights) | sidecar per model |

## 2. Mapping to OpenPBR 1.1.2 and reference exactness

Two spec facts used below: lengths default to centimetres (Transmission), while glTF `attenuationDistance` is in scene units
(metres here), so convert. `transmission_depth` 0 means no medium and `transmission_color` is a non-physical tint of the
refraction Fresnel factor, which is exactly glTF's `baseColor` tint on transmission.

| glTF | OpenPBR parameters | pbrt-v4 | Mitsuba 3 | Class |
|---|---|---|---|---|
| core metal (metallic 1) | `base_metalness` 1, `base_color` = F0, `specular_color` 1 (F82-tint reduces to Schlick at defaults), `specular_roughness` | `conductor` (`reflectance`, alpha = r², `remaproughness false`); complex Fresnel from eta = 1, k from reflectance, differs from Schlick at grazing | `roughconductor` | A at normal incidence; grazing difference is a DIVERGES row; A after the P2.17 check |
| core dielectric (metallic 0) | `base_color`, `specular_ior` 1.5, `specular_roughness` | `coateddiffuse` (stochastic; raise `maxdepth`/`nsamples`, furnace-check) | `roughplastic` (`nonlinear true`) or `principled` | B, pbrt closest |
| mixed metalness | `base_metalness` in (0,1) | `mix` (`amount` = metalness, exact linear blend) | `blendbsdf` | A if both ends are exact, else B |
| transmission, thick closed mesh, + ior | `transmission_weight` 1, `specular_ior`, `specular_roughness` | `dielectric` | `dielectric` / `roughdielectric` | A (Tier 0 dielectric furnace already agrees) |
| transmission tint (baseColor ≠ white) | `transmission_color`, `transmission_depth` 0 | no tint | no tint | B (compare white-base only) |
| transmission, thin-walled, roughness 0 | `geometry_thin_walled`, `transmission_weight` 1 | `thindielectric` | `thindielectric` | A (smooth only) |
| transmission, thin-walled, rough | same | none | none | C |
| `KHR_materials_volume` (closed, thickness > 0) | `transmission_color` + `transmission_depth` (Beer's law, `mu_t = -ln(T)/depth`); same pair as glTF `attenuationColor`/`attenuationDistance`; thickness is a rasterizer hint | `dielectric` + homogeneous medium (sigma_s 0) + `volpath` (extend the no-RR patch) | homogeneous medium as `interior` + `volpath` | A (absorption only; measure pbrt's RGB sigma uplift bias) |
| `KHR_materials_ior` | `specular_ior` | `eta` | `int_ior` | A |
| `KHR_materials_specular`, factor 1, colour 1 | defaults | as dielectric | as dielectric | A |
| `specularFactor` < 1 or `specularColor` ≠ 1 | `specular_weight`, `specular_color` (OpenPBR scales F at both ends; glTF scales F0, keeps F90 = 1) | none | `principled` `specular` (F0 only, approx) | C |
| `KHR_materials_clearcoat` | `coat_weight`, `coat_roughness`, `coat_ior` 1.5 | `coatedconductor` (exact for metal base); dielectric base: `coateddiffuse` without the base's own specular lobe | `principled` `clearcoat` (Disney) | metal base A-/B; dielectric base B, pbrt closer |
| `KHR_materials_sheen` | `fuzz_weight` 1, `fuzz_color`, `fuzz_roughness` (Charlie vs OpenPBR fuzz: close, not identical) | none | `principled` `sheen` (Disney lobe) | C; substitute: furnace/energy test and the OpenPBR reference implementation as a CPU oracle |
| `KHR_materials_iridescence` | `thin_film_weight`, `thin_film_ior`, `thin_film_thickness` (µm; glTF is nm) | none | none | C; substitute: analytic Airy-sum CPU oracle on a flat plane under a uniform env |
| `KHR_materials_anisotropy` | `specular_roughness_anisotropy` + mesh tangent. glTF: alpha_t = mix(alpha, 1, s²), alpha_b = alpha; OpenPBR: alpha_t = r²·sqrt(2/(1+(1−a)²)), alpha_b = (1−a)·alpha_t; solve a = 1 − alpha/alpha_t | `conductor` `uroughness`/`vroughness`, tangent = dp/du (verify against TANGENT; check `plymesh` vs `trianglemesh` `S`) | `roughconductor` `alpha_u`/`alpha_v` | A for metal after the tangent check |
| `KHR_materials_emissive_strength` | `emission_luminance` x `emission_color` | `AreaLightSource "diffuse"` L = factor x strength | `area` | A |
| `KHR_materials_dispersion` | `transmission_dispersion_abbe_number` = 20 / dispersion | spectral `eta` from a Cauchy fit of (n_d, Vd) | scalar IOR only | B' (pbrt only); ours needs spectral (D7) or a 3-wavelength trick |
| `KHR_materials_diffuse_transmission` (RC) | thin-walled `subsurface_weight`, `subsurface_color` | `diffusetransmission`, exact only with `specularFactor` 0 | none | B' (pbrt only, as a variant) |

Wave-1 reference finding (P2.32a, `bench/references/README.md`): two class A assignments are weaker than the table says. (1) `KHR_materials_volume` with a
saturated tint at optical depth above about 1 (AttenuationTest R5 1.0-2.0, R2 1.5-2.0): pbrt's RGB `sigma_a` uplift gives negative red and a blue 8-12% low
(V62), so those rows are Mitsuba-only references (class A against Mitsuba, B against pbrt). (2) Metal with a saturated base colour (MetalRoughSpheres gold,
blue 0.012): pbrt and Mitsuba differ by 12-25% in blue, so only grey metal is a class A gate there. The class C rows (specular < 1) agree between the two
references only because both use the same F0-via-IOR approximation.

Which reference is closer: pbrt for every layered case (coat, dielectric-over-diffuse), since its layering is physical and
OpenPBR's albedo-scaling approximates it; Mitsuba's `principled` is Disney-based.

## 3. Feature-coverage matrix

Rows: one per extension (then per model where several make up a row). Columns: ours, three-gpu-pathtracer, mutracer,
pbrt-v4, Mitsuba 3.
- `●` supported: agrees with the reference within the gate threshold (class A), or, without a reference, differs from its own
  feature-stripped render by ≥ 10x the noise floor and passes the oracle/energy check.
- `◐` approximate: renders but outside the threshold, or the exporter approximated (listed).
- `○` missing: full render equals the stripped render, or the renderer errors.
- relMSE vs pbrt / Mitsuba where a reference exists; `ref: pbrt only` or `ref: none` otherwise.

Competitor cells come from ablation runs (stripped vs full), never from reading their source (that is a prior only).

Data: `bench/results/reports/tier1/coverage.json` (`{status, relMSE_pbrt?, relMSE_mitsuba?, floor?, ablationRelMSE?,
notes[], reportRef}` per row and renderer; as built also `noise`, `ratio`, `relBias_*`, `fireflyPixels`, and `reportRef` is the viewer query
`a=<full>&b=<stripped>&r=<reference>`). Page: `bench/results/TIER1-COVERAGE.md` from `bench/adapters/tier1-matrix.ts`.
Viewer: a "Coverage" tab linking each cell to its side-by-side comparison.

## 4. Scene construction

- Scene = model (sha256-pinned) + sidecar `bench/scenes/tier1/<name>.ts` with camera, lighting (`env-sky-512x256.pfm`, as
  env-sphere), film, integrator (`maxDepth` 16 for glass, no RR in reference mode) and overrides.
- Wave 1 overrides (no textures): `dropMaterials` (labels) and `replace` (the backdrop becomes factor-only grey Lambert).
  Overridden scenes are named `<model>-notex`.
- Exactness variants (`specularFactor` 0 for `DiffuseTransmissionTest`, white base for transmission) never replace the
  unmodified scene; both appear in the matrix.
- Geometry policy unchanged: world-space, un-welded, flat geometric normals by default.
- References: pbrt-v4 `volpath` for media (no-RR patch extended and checked by variance, as for `path`); Mitsuba `volpath`;
  spp 1024/4096 per the existing trap.

## 5. Build plan

| ID | Item | Needs | Gate |
|---|---|---|---|
| P2.17 | Tier 1 contract: OpenPBR-subset `MaterialDesc`; the section 2 mapping as one tested function; DIVERGES rows (glTF Fresnel vs OpenPBR, Schlick vs complex Fresnel, length units) | schema only | unit tests for every extension default and the dispersion/anisotropy/thin-film conversions; golden JSON per model |
| P2.18 | Model fetch: MANIFEST (pinned URLs, sha256, licence), THIRD_PARTY attribution | MANIFEST, fetch.ts | `fetch.ts` verifies every hash |
| P2.19 | Loader v2: materials, extensions, UV, TANGENT, COLOR_0, instances, texture transform, punctual sun, unlit-as-emission, sidecar overrides, `normals` option | gltf-loader.ts | synthetic glTFs + each model's table equals the P2.17 golden |
| P2.20 | Exporters: pbrt (`thindielectric`, `mix`, `coated*`, media + `volpath`, `diffusetransmission`, dispersion spectrum), Mitsuba (`blendbsdf`, `roughplastic`, `principled` noted as approximate, media + `volpath`); no-RR patch for `volpath` | pbrt.ts, mitsuba.ts | exporter tests; one-bounce furnace and RR-depth variance check on `volpath` |
| P2.21 | Dielectric-over-diffuse, `specular_weight`/`specular_color`; build both `layering: 'albedo'` (OpenPBR albedo scaling) and `layering: 'stochastic'` (pbrt LayeredBxDF-style, reference mode; Apache-2.0, THIRD_PARTY) | BSDF | A/B vs pbrt `coateddiffuse` and Mitsuba `roughplastic` on one sphere under a uniform env |
| P2.22 | Thin-walled transmission and absorption-only interior medium (inside/outside from hit side; closed meshes, no nesting) | integrator, scene copy | `AttenuationTest-notex`, `TransmissionThinwallTestGrid-notex` |
| P2.23 | Clearcoat lobe | BSDF | `ClearCoatTest` Simple_Coated; metal-base variant vs pbrt `coatedconductor` |
| P2.24 | Textures wave (baseColor, metalRough, emissive, normal, alpha MASK); exporters write `imagemap`/`bitmap` (v flip, sRGB, wrap; pbrt has no mirror wrap, bake it) | P1.4c | unmodified `IORTestGrid`, `SheenTestGrid`, `AttenuationTest` |
| P2.25 | Sheen/fuzz lobe | BSDF | energy/furnace check; coverage only |
| P2.26 | Anisotropy with tangents (supersedes P2.7 for Tier 1) | tangents in scene copy + loader | `AnisotropyStrengthTest` vs both references after the tangent check |
| P2.27 | Iridescence (thin film) | BSDF | Airy-sum CPU oracle; coverage only |
| P2.28 | Dispersion | spectral or 3-wavelength (D7) | deferred; matrix shows pbrt and competitors |
| P2.29 | Diffuse transmission | BSDF | `DiffuseTransmissionTest` (`specularFactor` 0) vs pbrt |
| P2.30 | Emissive strength + metal-rough wave 1 (factor-only) | loader, P2.21 for the dielectric half | class A rows agree with both references |
| P2.31 | Coverage matrix: `tier1-matrix.ts`, `coverage.json`, viewer tab, competitor ablation runs | adapters, viewer | every competitor cell has a number or a reason |
| P2.32 | Tier 1 gate run and report | all above | section 6 |

Order: P2.17–P2.20 in parallel with P2.21 and P1.4c; then P2.22, P2.30, P2.23, P2.24; then P2.26/P2.29; then P2.25/P2.27;
then P2.31 and P2.32. P2.28 waits for spectral.

## 6. Gate (proposal)

Class A scenes gate; class B is reported with a bracketing metric; B' and C are coverage only. Pass rule for a class A
scene, ours at 4096 spp against the 4096 spp references:
- relMSE(ours, pbrt) ≤ 1.5 x relMSE(Mitsuba, pbrt) (the floor);
- |relBias| ≤ 0.5%;
- block means within the references' own spread (as P2.9);
- 0 non-finite pixels.

## 7. Question for the maintainer

**Which Tier 1 scenes count toward "converges to pbrt-v4 within the pbrt-vs-Mitsuba noise floor", and which are coverage
only?**

Suggestion: class A gates (thick dielectric + volume, thin smooth transmission, black-base ior, pure metals, emissive
strength, anisotropic metals after the tangent check). Class B is measured and shown with `d(ours, pbrt) / d(Mitsuba, pbrt)`
but does not gate. B' and C are coverage only.

- Pros: every gate scene has two references that agree to the floor, so a pass means something; no gate result depends on
  an approximation of ours or the exporters; the matrix still shows everything.
- Cons: the default non-metal (dielectric over diffuse), the most common glTF material, would not gate; sheen, iridescence
  and dispersion never gate; the class A list is short.
- Alternative: also gate class B in reference mode with our `layering: 'stochastic'` against pbrt `coateddiffuse`, Mitsuba
  only as the bracketing check.

A/B that settles it:
1. Sensitivity: for each candidate gate scene, relMSE(stripped, full) ≥ 10x the floor, or it cannot gate.
2. Agreement of the references: for each class B candidate, d(pbrt, Mitsuba); above ~5x the class A floor it stays out.
3. Layering: on one dielectric-over-diffuse sphere, `layering: 'albedo'` vs `'stochastic'` against pbrt `coateddiffuse` and
   Mitsuba `roughplastic`. If stochastic lands inside pbrt's floor and albedo does not, class B joins the gate in reference
   mode only.

Until decided, the suggestion holds (class A gates).

## 8. Risks and traps

- OpenPBR has no glTF mapping section; do not cite one. Our mapping is flagged in DIVERGES.
- OpenPBR lengths default to centimetres; the scene is in metres.
- pbrt `path` ignores media: use `volpath` and check its RR patch.
- pbrt `coateddiffuse` truncates at `maxdepth` 10 by default: raise it and run a furnace.
- pbrt and Mitsuba dielectrics cannot tint transmission: compare white-base transmission only.
- glTF thickness is a rasterizer hint; the path tracer needs closed, consistently oriented meshes.
- `MetalRoughSpheresNoTextures` is ~1.04 M instanced triangles (~3 M un-welded vertices): check upload size (P1.4e).
- `KHR_materials_unlit` and BLEND materials need an explicit mapping; never drop them silently.
- Competitor support from source reading is a prior only; trust the ablation.
- Not verified yet: whether pbrt `plymesh` accepts tangents; pbrt's RGB-to-spectrum bias in media; the size of the
  glTF / OpenPBR / `coateddiffuse` difference on a plain dielectric (P2.21 measures it).
