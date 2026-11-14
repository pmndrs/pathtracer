# Scene copy (D4, P1.4 slice 1)

The core's copy of what the GPU needs to shade a hit, beyond bvh's trees. Built on the CPU from the same inputs as
`buildTraceScene` plus a material list, a sun and an environment; a full rebuild per change (incremental updates are
P1.4b). Code: `packages/raytracer/src/scene/`. Bench scenes convert with `bench/scenes/to-scene-copy.ts`.

## Buffers

Two storage buffers, 0 textures:
- `rtSceneTable` (`array<vec4u>`): header (4 vec4u), instance records (8 vec4u: surface record + light-map words),
  mesh records (1 vec4u), materials (8 vec4u = 128 B, default record last), lights (6 vec4u = 96 B).
- `rtSceneData` (`array<u32>`): indices, vertex attributes (f32 bits), light maps for mixed instances.

A kernel with the walk uses 5–6 storage buffers, leaving 2–3 for IO under the 8-buffer fallback (spec §3). The scene
module is 38–39 KB with comments stripped; trace-core's largest kernel would drop from 49,998 to 31,983 B once migrated
(P1.4a).

```ts
const copy = buildSceneCopy({ meshes, instances, materials, sun?, environment?, options? });
const gpu = uploadSceneCopy(device, copy);
rtModuleWgsl(caps, { ...format, scene: { slot, layout: copy.layout, surface?, materials?, lights? } });
```

## Materials (128 B, indexed by bvh's 16-bit id)

| Words | Content |
|---|---|
| 0–2, 3 | baseColor f32×3; ior f32 |
| 4–6, 7 | emissionColor f32×3 (radiance); emissionLuminance f32 |
| 8, 9 | transmissionDepth f32 (0 = no Beer absorption); coatIor f32 |
| 10 | flags: alphaMode (bits 0–1), thinWalled, conductorEtaK, emissionTwoSided, doubleSided |
| 11, 12 | texture slots u16×4: baseColor, roughnessMetal, normal, emission (0xFFFF = none) |
| 13–15 | conductor eta and k, f16×6 |
| 16–28 | 25 unorm16: baseWeight, metalness, specularRoughness, specularWeight, specularColor×3, anisotropy, anisotropyRotation, transmissionWeight, transmissionColor×3, coatWeight, coatRoughness, coatColor×3, sheenWeight, sheenRoughness, sheenColor×3, opacity, alphaCutoff |
| 29–31 | reserved, 0 (subsurface, thin film, fuzz, dispersion) |

`rtMaterial(id)` returns a plain `RtMaterial` with the same field names as the TypeScript twin (a test checks the
lists match), so the P2 BSDF adapter (P2.2) is one line. Roughness is perceptual; GGX alpha = roughness². Ids past the
table, including `RT_NO_MATERIAL`, read the default record. Precision is mixed on purpose (V32).

## Lights (96 B)

Emissive triangles in world space (instance transforms applied; the emitting normal is the authored front side, safe
under mirroring), then one sun, then one environment (constant radiance plus a reserved map slot, P1.4d).

- **Selection:** a Vose alias table over luminance-weighted power, stored in the light records, so `rtSampleLight(u)` is
  one record read plus one pmf read; it returns the leftover `u` rescaled for the next dimension. Power: triangle
  `π·area·Y(L)` (×2 two-sided), sun `Y(E)·π·R²`, uniform env `Y(L)·4π²R²` with `R` the scene bounding-sphere radius,
  scaled by `sunPowerScale` / `envPowerScale` (V31, A/B P1.4f).
- **MIS for a BSDF-sampled hit:** `rtLightPmfOfHit(instance, triangle)` via a per-instance triangle → light map: none,
  dense (index = base + triangle when every triangle emits), or a per-triangle table for mixed instances.
- Other accessors: `rtLight(i)`, `rtSun()`/`rtHasSun()`, `rtEnvironment()`/`rtHasEnvironment()`, `rtLightPmf(i)`,
  `rtLightIndexOfHit`, `rtSceneInfo()` (counts, total power, scene radius).

## Surface

The slice-1 surface layout lives inside the copy; `rtSurface(hit, ray, want)` keeps its signature and flags. The CPU
decode (`readSurfaceFromTable`) and the builder (`buildSurfaceTable`) are shared; the standalone stand-in (upload, bind group, `surface:` module option) was deleted in P1.4a.

## Verification

Pack/unpack twins in TypeScript (80 unit tests: exhaustive f16 round trip, every field, alias statistics, hand-computed
light power for Cornell and quad-light, a scaled/rotated/mirrored synthetic scene, the converter on every Tier 0 scene).
The `scene-copy` page (468,702 checks on the M5) compares every material, light, alias pick and light-of-hit lookup on
the GPU against the twins bit for bit; `rtSurface` interpolation agrees within 2.3e-7 relative and the leftover `u`
within 1.2e-7.

## Open choices (defaults hold until measured)

1. **Light selection weights for sun and environment.** Default pbrt-style power, scales as settings. Risk: a sun or sky
   takes most picks next to small emitters (noisy NEE indoors). Quality only, no speed cost. Measured (P1.4f, `lightSelection`): see docs/design/integrator.md open choice 3; suggestion `cap:0.5`, default `power` until decided.
2. **Material precision** (f16 eta/k, unorm16 weights). Keep: errors far under the reference noise floor. Settle: the
   P2 furnace and copper tests; if bias shows, move eta/k to f32 (record grows to 160 B).
3. **One table + one data buffer vs a buffer per section.** Keep 2 bindings (saves 3). Cost: one 128 MiB binding holds
   all geometry, so big scenes need chunking sooner (P1.4e). Settle: a Bistro-size upload.
4. **`stripComments`** is on by default for every module (was scene-only until the stand-in went in P1.4a). Setting exists.
5. **hdri environments** get black constant radiance and a reserved slot until P1.4d (env-sphere isn't traceable yet).
6. **Default conductor eta/k** is a copper-like placeholder, used only when `conductorEtaK` is set without values.
7. **Mixed-instance light maps** cost one word per triangle; a sorted range list could replace them if a scene has huge
   mixed instances with few lights.
