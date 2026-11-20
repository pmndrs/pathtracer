# Scene copy (D4, P1.4 slice 1)

The core's copy of what the GPU needs to shade a hit, beyond bvh's trees. Built on the CPU from the same inputs as
`buildTraceScene` plus a material list, a sun and an environment; a full rebuild per change (incremental updates are
P1.4b). Code: `packages/raytracer/src/scene/`. Bench scenes convert with `bench/scenes/to-scene-copy.ts`.

## Buffers

Two storage buffers, plus one `texture_2d_array` and one sampler per texture bucket when the scene has textures (see Textures):
- `rtSceneTable` (`array<vec4u>`): header (4 vec4u), instance records (8 vec4u: surface record + light-map words),
  mesh records (1 vec4u), materials (8 vec4u = 128 B, default record last), lights (6 vec4u = 96 B), texture slots
  (1 vec4u each, only with textures; header words 14 and 15 are their base and count, 0 otherwise).
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

## Textures (P1.4c, first slice)

Code: `packages/raytracer/src/scene/texture.ts` (build + CPU twins), `gpu.ts` (`uploadSceneTextures`), `scene-wgsl.ts` (`rtTexturesWgsl`).
Input: `SceneCopyInput.textures` (RGBA8, row 0 = top, `colorSpace: 'srgb' | 'linear'`, `wrap: 'repeat' | 'clamp' | 'mirror'`); a material
names them by index in `baseColorTexture` / `roughnessMetalTexture` / `normalTexture` / `emissionTexture` (the u16 slots). The builder
checks that a slot exists and that colour fields (base colour, emission) point at `srgb` textures and data fields (metallic-roughness,
normal) at `linear` ones. UVs are the mesh's (`SceneCopyMesh.uvs`, glTF TEXCOORD_0: origin top-left, v down; `rtSurface` already
exposed them). Adding textures forces vertex positions into the surface data (the texture frame needs the triangle edges).

**Layout: size buckets of `texture_2d_array`.** A texture is resampled (tent filter) to a power-of-two size (nearest in log space,
at most `maxSize` = 4096), given a full mip chain on the CPU (2x2 box, in linear light for sRGB), and stored as one layer of the
array shared by all textures of the same (width, height, colour space, wrap). One array is one texture binding and one sampler, so a
scene costs `buckets` of the 16 sampled textures and 16 samplers a stage may bind (default cap 8; the cheapest same-space pair is
merged when it needs more, 256 layers per array). Formats: `rgba8unorm-srgb` for colour (the hardware decodes before filtering: filtering in
linear light, as pbrt-v4 and Mitsuba), `rgba8unorm` for data. The slot record is `(bucket, layer, width | height << 16, mip levels)`;
`rtTexSample(slot, uv, lod)` switches on the bucket (`textureSampleLevel`, explicit LOD, so no uniformity rules) and costs about 160 B of
WGSL per bucket plus 2.9 KB once. Mip level is chosen by the integrator (`rtTexLodOf(record, uvFootprint)`).

Shader API (only emitted when the module has `scene.textures`): `rtTexSample`, `rtTexSampleRecord`, `rtTexRecord`, `rtTexLodOf`,
`rtApplyMaterialTextures(m, uv, footprint)` (glTF semantics: base colour rgb x factor, opacity x alpha, roughness x G, metalness x B,
emission x rgb), `rtApplyNormalMap(n, tangent, texel)`, and `rtSurface` with `RT_SURF_TEXFRAME` returns `uvDensity` (sqrt(uv area / world
area), uv units per metre, for the ray-cone LOD) and `tangent` (world dp/du, w = bitangent handedness; glTF green = image up = decreasing v).
A textured emitter is not a light-table entry (NEE needs a constant radiance): BSDF sampling finds it with weight 1 (V52).
Bit-for-bit untextured scenes: no header word, table section, binding or WGSL changes.

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

### Vertex tangents (P2.26)

`SceneCopyMesh.tangents` (`x, y, z, w` per vertex, object space, glTF `TANGENT`: the direction of increasing u, `w` the bitangent sign) is stored when any mesh has
them (`SurfaceTableOptions.tangents`, default derived): the vertex stride grows by 4 floats at the END (`offsets.tangent`, after the position), the mesh record
gets `MESH_FLAG.HAS_TANGENTS`, `SurfaceLayout.tangents` is true. The key is absent otherwise, so scenes without anisotropic materials keep stride, buffers, fingerprints
and kernel text (`bench/size-snapshot.ts`: all 11 existing variants `cmp`-identical; the layout's `tangents` key is added only when true).
`rtSurface(hit, ray, want | RT_SURF_TANGENT)` (opt-in bit 64, not part of ALL) fills `RtSurface.vertexTangent`: the barycentric mix of the three vertex tangents, taken to
world space by the instance's object-to-world 3x3 (a direction, not the normal matrix), normalised, `w` = sign of the mixed `w`; zero xyz when the mesh has none.
It is made orthogonal to the shading normal by the caller (`ptFrameFromTangent`, Gram-Schmidt), because the normal map may change the normal. The decode is a separate
function (`rtSurfaceTangent`), emitted by a JS template condition rather than `#if`: preprocessor-dropped lines leave blank lines, which would change the minified bytes.
`SURFACE_TABLE` records, the scene header and the material record are untouched. The CPU twins are `readSurfaceFromTable` / `referenceSurface` (`tangent`, only when asked).
Per-triangle dp/du of the uvs (the frame `RT_SURF_TEXFRAME` builds for normal maps) is NOT used for anisotropy: the bench makes it a per-vertex tangent instead
(`bench/scenes/tangents.ts` `uvTangents`, setting `pt_tangent=uv`), so there is one decode path (V75).

## Verification

Pack/unpack twins in TypeScript (80 unit tests: exhaustive f16 round trip, every field, alias statistics, hand-computed
light power for Cornell and quad-light, a scaled/rotated/mirrored synthetic scene, the converter on every Tier 0 scene).
The `scene-copy` page (468,702 checks on the M5) compares every material, light, alias pick and light-of-hit lookup on
the GPU against the twins bit for bit; `rtSurface` interpolation agrees within 2.3e-7 relative and the leftover `u`
within 1.2e-7.

Textures: 32 unit tests (`texture.test.ts`: sRGB, bucket sizes, packing and merging, mips, wrap modes, trilinear twin, cone LOD, texture frame, normal map)
plus scene-copy and WGSL tests. The `texture-check` page (3,000 cases) compares `rtTexSample` against the twin over 4 arrays (non-power-of-two
source, clamp / mirror / repeat, uvs from -1.5 to 2.5): max 6.3e-3 absolute (fixed-point filter weights), `rtTexLodOf` 9e-7, the glTF material
semantics 1.4e-2 on emission scaled by 3, `rtApplyNormalMap` 5e-7; end to end under a delta sun every pixel has a closed form (base colour 3.3e-4,
normal maps 1e-7 including a rotated instance, emission 3e-3 on the dim blue channel, metallic-roughness 2e-7). `node bench/runner/run.ts --page texture-check`.

## Open choices (defaults hold until measured)

_D32 (2026-11-18): `DEFAULT_SCENE_PACKING` is `packed` (provisional: per-pixel identical to `split`, one binding freed, timing a wash on Sponza and Cornell +2% megakernel). `split` stays a setting and is the fallback for a scene copy over the one-binding cap._

1. **Light selection weights for sun and environment.** Settled (D22): default `pbrt`, pbrt-v4's rule. Each infinite
   light gets 1/(nInfinite+1), and the area lights share the rest by power. `power`, `kind` and `cap:c` stay as settings.
   Numbers: docs/internal/design/integrator.md open choice 3.
2. **Material precision** (f16 eta/k, unorm16 weights). Keep: errors far under the reference noise floor. Settle: the
   P2 furnace and copper tests; if bias shows, move eta/k to f32 (record grows to 160 B).
3. **One table + one data buffer vs a buffer per section.** Keep 2 bindings (saves 3). Cost: one 128 MiB binding holds
   all geometry, so big scenes need chunking sooner (P1.4e). Settle: a Bistro-size upload.
4. **`stripComments`** is on by default for every module (was scene-only until the stand-in went in P1.4a). Setting exists.
5. **hdri environments** get black constant radiance and a reserved slot until P1.4d (env-sphere isn't traceable yet).
6. **Default conductor eta/k** is a copper-like placeholder, used only when `conductorEtaK` is set without values.
7. **Mixed-instance light maps** cost one word per triangle; a sorted range list could replace them if a scene has huge
   mixed instances with few lights.
8. **Texture storage: size buckets (shipped) vs one fixed-size array vs atlas vs bindless.** Everything below works on every WebGPU device
   except bindless, which is textures-only, Chrome-behind-a-flag and not on macOS (V1).
   - *Size buckets* (now): one array per (size, colour space, wrap). Pro: no wasted texels beyond the pad to a power of two, no bleeding,
     hardware wrap and filtering, mips cut the cache footprint of minified lookups, any texture size mix. Con: up to 8 bindings and a switch
     in the shader (about 160 B of WGSL and a divergent branch when a warp mixes buckets); a scene with many sizes merges buckets and pads.
   - *One fixed-size array* (say 1024x1024, every texture resampled): one binding, no switch. Con: 4x memory for a 512 texture, loses detail
     above the size, and sRGB/data still need two arrays. Cheapest code; worst memory.
   - *Atlas*: one texture, no layers. Con: bleeding across tiles (needs gutters), wrap must be done in the shader (so no hardware repeat),
     mips cross tiles. Not recommended.
   - *Bindless*: no pad, no switch, no merge. Con: not available on macOS; needs `chromium-experimental-sampling-resource-table`.
     **Built (P1.4c2, 2026-11-19)** as `TextureOptions.binding: 'table'`: the size-bucket arrays themselves go into the table, so texels,
     samplers and images equal the `arrays` path whenever `arrays` did not merge (and merging is what the table removes: 20 groups cost
     33.3 MiB as 8 merged arrays, 21.8 MiB in the table). The lookup is one `getResource` line whatever the bucket count. Open before it can
     be a default: Metal fails compute passes with a table (Chrome 160 Canary, V296; its render passes work, and index divergently correctly) and
     SwiftShader breaks divergent indices (V297); the 3060 gives the compute verdict on D3D12 / Vulkan and the timing (W17, `--page bindless-check`).
   Suggestion: keep size buckets; add bindless as Tier 3 behind the same slot API (P1.4c2). Settle with: Sponza-with-textures (P1.4c3) and a
   Bistro-class scene, same camera, spp, seed: ms per spp (quiet pass), texture bytes, and the image difference (must be bit-identical except
   for resampled non-power-of-two sources); buckets vs one fixed array (a setting) vs bindless on the 3060.
9. **Texture LOD for diffuse bounces.** Default `textureLod: 0` (the reference setting, Mitsuba's behaviour). `'cone'` keeps the specular
   chain's cone and widens it by 0.5 rad at a non-delta bounce, with the starting spread scaled by max(1/8, 1/sqrt(spp)) (pbrt-v4's rule).
   Measured on `tex-checker-hf` against Mitsuba 4096 spp (relMSE, lower is better): at 16 spp lod 0 1.08e-2, cone auto 4.89e-3, cone full pixel
   3.85e-2; 64 spp 2.04e-3 vs 1.33e-3; 256 spp 4.74e-4 vs 3.72e-4. A full-pixel cone blurs (a plateau at 3.75e-2 at any spp: the pixel box
   then averages a pre-blurred texture); the pbrt scale removes that. Means agree to 0.02%. Open: the 0.5 rad diffuse spread and the
   curvature term are not tuned. Settle: a GI-heavy textured scene (a corridor of textured walls) at equal spp, lod 0 vs cone, diffuse spread
   0 / 0.25 / 0.5 / 1.
