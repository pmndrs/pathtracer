# MaterialX materials (P5)

Materials authored as MaterialX graphs (`open_pbr_surface`, `standard_surface`, `gltf_pbr` closures plus pattern nodes) rendered by our
path tracer. Everything lives in `packages/pathtracer/src/materialx/` (library) and `bench/scenes/materialx*.ts` (scene glue).
MaterialX itself is Apache-2.0; we read its standard library GLSL / nodegraphs for the node semantics and copy no code (THIRD_PARTY.md).

## 1. Pipeline

```
.mtlx text --parseMaterialX--> MtlxDocument --MtlxLowering--> MtlxIr (DAG: const | op) --lowerSurface--> MtlxSlot per OpenPBR parameter
                                                                                           |
                                         constants folded (foldConstants) -> MtlxMaterial { constants, varying, normal, images }
                                                       |                                      |
                                   (a) bakeMtlxMaterial                        (b) mtlxProgramsWgsl
                         textures in the scene's texture buckets        ptMtlxApply / ptMtlxNormal in the kernel
```

* **No MaterialXJS / WASM.** The reader is 150 lines of TypeScript (`xml.ts`, `document.ts`): the subset of the format we render needs
  elements, attributes and `fileprefix` / `colorspace` inheritance, and a WASM module would not run in a bench page next to the kernel text.
  MaterialXJS and `materialx-fidelity` are the maintainer's references for MaterialX's own renders (section 6).
* **One table of nodes** (`nodes.ts`): for each node its inputs and defaults, a TypeScript implementation and the WGSL expression. The
  TS evaluator (`eval.ts`, the oracle), the baker and the code generator all walk the same lowered DAG, so the twins cannot drift in scoping,
  defaults, output selection or type coercion; only the per-node expression is written twice, side by side.
* **Values are four lanes** (`MtlxValue`, WGSL `vec4f`). A scalar feeding a vector input is splatted, a one-number default fills every lane,
  and type-sensitive nodes (`normalize`, `dotproduct`, `crossproduct`, `magnitude`, `normalmap`) read only the lanes their type has.

## 2. Surfaces to OpenPBR

`open_pbr_surface` maps one to one (names in `surface.ts`). `standard_surface` 1.0.1 and `gltf_pbr` go through the tables in `surface.ts`
(V261, V266); the authored defaults of those shaders that differ from OpenPBR's are applied (`STANDARD_SURFACE_DEFAULTS`: base 0.8, base colour 1,
specular roughness 0.2, coat roughness 0.1, sheen roughness 0.3, thin-film IOR 1.5). Inputs the renderer has no parameter for are listed in
`MtlxSurface.ignored` (transmission scatter, subsurface radius, coat anisotropy, coat darkening, dispersion on standard_surface ...).
Lengths: MaterialX / OpenPBR lengths are centimetres, ours metres: `transmission_depth` is multiplied by `lengthScale` (default 0.01, V262).

## 3. The two approaches

| | (a) bake | (b) codegen |
|---|---|---|
| Kernel | unchanged (material record + texture buckets) | `integrator.materialx`: `ptMtlxApply` after `rtApplyMaterialTextures`, `ptMtlxNormal` for the normal map; text grows by the generated functions |
| What varies | base colour (weight folded in), opacity (alpha), roughness, metalness, emission colour, tangent-space normal; anything else varying is replaced by its mean | every `RtMaterial` field (colour, weights, roughness, IOR, coat, sheen, transmission, opacity, emission); not thin film, diffuse roughness, subsurface |
| Drivers | uv only (position / normal inputs are baked at the origin / default and listed) | uv, position, normal, tangent |
| Fidelity | limited by the texture: 8 bit, `resolution`^2 over the uv square [0, 1]^2 | exact per hit (f32); image nodes are bilinear + mips like any scene texture |
| Cost | none at render time; bake time 1024^2 about 1 s; memory 4 B per texel per baked texture | kernel +1.3 KB (position-driven graph) to +2.5 KB (three images, place2d, normal map) in the OpenPBR layered kernel; wavefront shade 41.6 KB -> 42.9 / 44.2 KB (megakernel 52.7 KB -> 53.9 / 55.2 KB: already over the 50 KB class with textures + layered albedo, integrator.md choice 11) |

The wavefront takes (b) too: `ptShadingModulesWgsl` puts the generated text in the shared parts, so both schedulers see the same
functions (`wf-mtlx-*-shade` in `bench/size-snapshot.ts`).

Scenes: `materialxScenes` (`bench/scenes/ab/materialx-scenes.ts`): `mtlx-const-*` (constants and their native equivalents), `mtlx-uv-bake`
(1024^2), `mtlx-uv-bake-256`, `mtlx-uv-codegen` (the same graph), `mtlx-proc-codegen` (position-driven, no bake possible).
Textures are lazy (`generator: 'lazy'`), so describing a scene parses the document and plans the bake only.

## 4. Supported nodes

Math: `add subtract multiply divide modulo power min max absval floor ceil round fract sign sin cos tan asin acos atan2 sqrt exp ln clamp
invert remap smoothstep mix ifgreater ifgreatereq ifequal normalize magnitude dotproduct crossproduct luminance`. Data: `constant dot convert
combine2/3/4 extract swizzle separate2/3/4` (outputs `outx` and `x` both read). Geometry: `texcoord position normal tangent` (object = model =
world, V260), `transformpoint/vector/normal` between those spaces. Patterns: `image tiledimage place2d rotate2d ramplr ramptb splitlr splittb`
and `normalmap` (tangent or object space). Surfaces: `open_pbr_surface standard_surface gltf_pbr` through `surfacematerial`.

Not supported (the loader throws `MtlxUnsupportedError` naming the node): the noise family (`noise2d/3d fractal3d cellnoise worleynoise
unifiednoise`), `hsvtorgb rgbtohsv colorcorrect contrast saturate`, the matrix and rotate3d nodes, `ramp4 ramp_gradient`, `blackbody`,
`checkerboard` (not in the 1.38 library we read; its 1.39 definition is unverified), logic nodes, `heighttonormal bump`, time and frame.
`node bench/materialx-coverage.ts` counts them over the sibling sample trees: 144 of 218 documents load (81 with spatially varying inputs).

## 5. Conventions (the ones that decide silent errors)

* uv (0, 0) is the **top-left** texel and v grows downwards, as glTF and our scene textures (V263). MaterialX's own `ramptb` blends top to bottom
  with texcoord.y 0 -> 1, which fixes the orientation of its standard library; the GLSL `mx_transform_uv` leaves uv alone. Not compared against
  `materialxview` renders yet (open problem 1).
* `file` inputs: the document's `fileprefix` is prepended; colour images with `colorspace="srgb_texture"` (or `srgb`, `g22_rec709`) are sRGB
  textures (decoded before filtering), everything else is linear data. Address modes `periodic clamp mirror` map to the texture's wrap, `constant` reads as
  clamp, `filtertype` is ignored (always bilinear, mips by ray cone when the kernel has it).
* `tiledimage` follows the 1.38 nodegraph: `((uv * uvtiling - uvoffset) / realworldimagesize) * realworldtilesize`. `place2d` follows
  `NG_place2d_vector2` (SRT and TRS orders, rotation in degrees).
* `normalmap` follows `mx_normalmap.glsl`: zero input decodes to +Z, xy scaled by `scale`, bitangent = cross(N, T); the baked or generated texel is
  re-encoded for `rtApplyNormalMap`, which uses the mesh tangent and its handedness (the glTF green-up convention, `texture-check`).

## 6. Checks

1. Unit tests (`packages/pathtracer/src/materialx/materialx.test.ts`, 21 tests): XML, scoping, interface defaults, multi-output nodes, closed forms of
   `mix`, `clamp`, `remap`, `place2d` (SRT with pivot and rotation), `normalmap`, `tiledimage` texel-exact, sRGB vs linear images, address modes,
   bilinear, the bake against the graph at texel centres, folded base weight, opacity in alpha, baked normal map.
2. `bench/scenes/materialx.test.ts`: constant `open_pbr_surface` and `standard_surface` materials give the same scene-copy records as their native
   `openpbr` descriptions (`toEqual`), bake plan and lazy textures, generated WGSL is balanced, declares no reserved WGSL identifier and no duplicate function.
3. GPU: `node bench/runner/run.ts --page materialx-check` runs the generated functions on random points against the TypeScript twin (section 7).
4. Renders (`matrix.ts --renderers pt-megakernel`): const scenes equal their references per pixel; bake against codegen on the same graph (section 7).
5. Byte identity: `node bench/size-snapshot.ts <dir>` before and after: all 224 existing kernel files are identical; the four `mtlx-*` files are new.

MaterialX's own GLSL renders are not a path tracer (IBL + one directional light), so there is no numeric reference for the whole image. The
maintainer's `materialx-fidelity` renders (materialxview, MaterialXJS, three.js) are the right reference for appearance of the real sample materials;
comparing them needs our renderer under the same IBL and shader ball and is an open follow-up (P5.m7).

## 7. Results (2026-11-18, M5 Pro, Chrome; correctness only: the Mac was shared, load average 30-50, no timing is claimed)

**Check 1, generated WGSL against the TS twin** (`--page materialx-check`, 2000 random points per scene: uv in [0, 1]^2, position in [-1, 1]^3, random
unit normal and tangent; base colour, metalness, roughness, normal texel): 0 mismatches. Position-driven graph (`mtlx-proc-codegen`: sin, remap,
smoothstep, mix): max abs error 7.6e-6. Image graph (`mtlx-uv-codegen`: place2d rotation, sRGB checker, rings, ramp, smoothstep, bump normal map):
max abs error 1.6e-3 (hardware bilinear with 8-bit weights against f64 bilinear; the tolerance is 1.5e-2).

**Check 2, constants** (`mtlx-const-open-pbr` vs `-ref`, `mtlx-const-std` vs `-ref`, 256 spp, `pt_layering=albedo`): the images are bit-identical (mean
difference 0, relMSE 0, 0 pixels differ). `bench/scenes/materialx.test.ts` also has the scene-copy material records `toEqual`.

**Check 3, bake against codegen** (`mtlx-uv-*`, same graph, same sampler seed per pixel, so the difference is the material; A = codegen; `bench/adapters/mtlx-compare.ts`):

| B | mean rel diff (r, g, b) | relMSE(B, A) | rms rel pixel | pixels > 1e-2 | > 5e-2 |
|---|---|---|---|---|---|
| bake 1024^2 | 1.1e-5, 3.2e-4, 1.6e-4 | 2.8e-5 | 8.8e-3 | 2.7% | 0.18% |
| bake 256^2 | -4.6e-4, 1.6e-4, 8.2e-5 | 1.0e-4 | 1.7e-2 | 8.7% | 1.7% |

The bake is unbiased in the mean (differences of 1e-4 to 5e-4) and costs 8-bit quantisation plus bilinear smoothing of the 128^2 source patterns at
the texel scale: 4x more texels cut the error 3.7x in relMSE. The coat, specular colour and similar inputs have no texture slot, so a graph that varies
those is only right in (b) (here nothing does). `mtlx-proc-codegen` has no bake at all (it reads the position).

**Both schedulers** (`pt-wavefront` against `pt-megakernel`, 256 spp, `node bench/adapters/pt-wavefront-agreement.ts --a mtlx-p5 --b mtlx-p5-wf`): the codegen
scenes and the bake agree per pixel, max relative difference 2.6e-6 (uv bake and codegen) and 3.7e-6 (position-driven), relMSE 1.1e-13, no pixel over 1e-3
(the wavefront port's bound is 7e-5): the generated functions are the same code in both shade kernels.

Tracing cost: the matrix rows show `ms_per_spp` 0.95-0.99 for every scene on a loaded machine; nothing here is a timing result. Whether (a) or (b) is
faster per frame at scene scale (texture fetches of baked maps against ALU of the graph plus its image fetches, kernel size and register pressure) is a
quiet-pass A/B (P5.m7).

**Reading the numbers for a default.** Codegen is exact and needs a kernel recompile when a material changes; the bake costs nothing in the kernel and
is bounded by resolution. No default is set (D11): `buildMaterialX({ mode })` takes either; a scene with a position-driven graph needs `codegen`.
