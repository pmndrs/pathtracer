# @pmndrs/raytracer

The trace core: rays in, hits out, on WebGPU, over `@pmndrs/bvh`'s scene buffers and WGSL walk.
WGSL names start with `rt` / `Rt` / `RT_`; the bvh walk underneath keeps its own `bvh*` names.

Status: **P1 slice 1** (L0, L1, `rtSurface`, batch dispatcher). Correct against bvh's CPU oracle on the M5
(`node bench/runner/run.ts --page trace-core`); speed numbers are provisional. Not yet: L2 shading, the
material/light scene copy, L3. Design notes: [docs/internal/design/trace-core.md](trace-core.md).

## API

### Contract (`contract/ray.ts`)

| Name | What |
|---|---|
| `Ray` | `{ origin, tMin, direction (unit), tMax }`, 32 B (`RAY_BYTES`), bvh's `BvhRay` |
| `Hit` | `{ t (< 0 miss), instance, triangle, frontFace, barycentrics (v1, v2), material, normal }`, 40 B (`HIT_BYTES`); compact record 24 B (`HIT_COMPACT_BYTES`) without material and normal |
| `RayCone` | `{ width, spread }`, 8 B. Carried through L1; the megakernel's texture LOD (`textureLod: 'cone'`) consumes it |
| `BVH_LAYOUT_VERSION` | 2. `assertBvhLayout()` runs before any walk is generated |

### Two ways to trace

**Inline**: paste the module into your own compute shader.

```ts
import { rtModuleWgsl, rtWalkEntries, sceneFormatOf, withBindings, rtSceneBindGroupEntries } from '@pmndrs/raytracer';

const format = sceneFormatOf(sceneTrees.traceBuffers); // nodeFormat, triangleLeaves, positionLayout: compiled in
const code = rtModuleWgsl(caps, {
  ...format,
  traceSlot: { group: 0, firstBinding: 0 },          // bvh walk: 3 bindings (4 for indexed leaves)
  scene: { slot: { group: 2, firstBinding: 0 }, layout: copy.layout },    // optional: rtSurface, materials, lights (2 bindings, see Scene copy)
}) + myKernel;
const sceneLayout = device.createBindGroupLayout({ entries: withBindings(rtWalkEntries(format)) });
const sceneGroup = device.createBindGroup({ layout: sceneLayout, entries: rtSceneBindGroupEntries(sceneTrees.traceBuffers) });
```

WGSL it gives you:

```wgsl
struct RtRay  { origin: vec3f, tMin: f32, direction: vec3f, tMax: f32 }
struct RtCone { width: f32, spread: f32 }
struct RtHit  { t: f32, instance: u32, triangle: u32, frontFace: bool, bary: vec2f, material: u32, normal: vec3f, coneWidth: f32 }

fn rtVisible(ray: RtRay, mask: u32) -> bool                       // L0: true = nothing blocks [tMin, tMax]
fn rtVisibleSegment(a: vec3f, b: vec3f, mask: u32, eps: f32) -> bool
fn rtClosest(ray: RtRay, mask: u32) -> RtHit                      // L1: t == RT_MISS_T (< 0) is a miss
fn rtClosestCone(ray: RtRay, mask: u32, cone: RtCone) -> RtHit    // L1 + cone width at the hit
fn rtSurface(hit: RtHit, ray: RtRay, want: u32) -> RtSurface      // lazy decode, RT_SURF_* bits
// helpers: rtRay, rtSegment, rtIsHit, rtConeAt, rtConePropagate; masks: RT_MASK_ALL
```

`rtSurface` returns world position, geometric normal, front face, interpolated shading normal (flipped to the
geometric side) and UV, and the material id (an index into your own table). Only what `want` asks for is loaded.
Its data lives in the scene copy (below); `scene: { ..., materials: false, lights: false }` gives `rtSurface` alone.

**Batch** (queue ABI v0): a GPU ray buffer and a count in, one result per ray out.

```ts
const tracer = await RtBatchTracer.create(device, caps, {
  ...sceneFormatOf(buffers),
  mode: 'closest',          // or 'visible'
  workgroupSize: 64,        // D11 settings: 32 | 64 | 128 | 256
  rayLayout: 'aos',         // or 'soa' ([origin, tMin] and [direction, tMax] arrays)
  hitRecord: 'full',        // or 'compact' (24 B)
  indirect: false,          // true: count read from a GPU buffer
  cones: false,             // closest only: per-ray cones in, cone width out
});
tracer.encode(encoder, buffers, { rays, results, count: n, mask });                  // direct
tracer.encode(encoder, buffers, { rays, results, count: { buffer, word, capacity } }); // indirect
tracer.dispatch(pass, buffers, batch);                                                // into your own pass
```

Results: `closest` writes `Hit` records (`readHits` decodes them), `visible` writes one u32 per ray, 1 = visible.
Everything goes into the caller's encoder; uniforms and bind groups are cached per batch shape, so a steady frame
does no CPU writes and no readback. CPU helpers: `packRays`, `packRaysSoa`, `aosToSoa`, `packCones`, `readHits`,
`readVisibility`, `dispatchGrid`.

### Interval queries (P5.rc1, for gi's Split Radiance Cascades)

`rtModuleWgsl(caps, { intervals: true })` adds `rtClosestInterval(origin, direction, t0, t1, mask) -> RtHit` over the half-open `(t0, t1]` (`rtClosestIntervalLo`: `[t0, t1)`), `rtVisibleInterval*`, `rtCascadeEnd / Start / Index` and
`rtOctEqualAreaDecode`; `RtIntervalTracer` is the batched form (sources `rays` / `probes`, outputs `hit` / `distance` / `visible` / `shaded`, one kernel for many dispatches per frame). Semantics, the boundary
rule, numbers and how gi calls it: [docs/internal/design/split-rc-tracing.md](split-rc-tracing.md). Not boundary exact: the any-hit forms (V290).

### Scene copy (P1.4, D4): materials, lights, surface data

```ts
import { buildSceneCopy, uploadSceneCopy, rtSceneCopyBindGroupEntries, rtSceneCopyLayoutEntries, rtModuleWgsl } from '@pmndrs/raytracer';

const copy = buildSceneCopy({ meshes, instances, materials, sun, environment });   // meshes/instances as for bvh's buildTraceScene
const buffers = uploadSceneCopy(device, copy);                                     // 2 storage buffers: table, data
const code = rtModuleWgsl(caps, { ...format, scene: { slot: { group: 2, firstBinding: 0 }, layout: copy.layout } });
```

Adds `rtMaterial(id) -> RtMaterial` (128 B OpenPBR-subset records, `material.ts`), `rtSampleLight(u)`, `rtLight(i)`, `rtSun()`,
`rtEnvironment()`, `rtLightIndexOfHit(instance, triangle)`, `rtLightPmfOfHit(...)` (light table with an alias table over power,
`lights.ts`) and `rtSurface` over the copy's data.
Bench scenes: `bench/scenes/to-scene-copy.ts`. Layout and WGSL: `scene/constants.ts`, `scene/scene-wgsl.ts`.

#### Material textures (P1.4c)

```ts
const copy = buildSceneCopy({
  meshes /* with uvs: glTF TEXCOORD_0, origin top-left */, instances,
  textures: [{ width: 1024, height: 1024, data /* RGBA8, row 0 = top */, colorSpace: 'srgb', wrap: 'repeat' }, /* ... */],
  materials: [{ baseColorTexture: 0, roughnessMetalTexture: 1, normalTexture: 2, emissionTexture: 3, baseColor: [1, 1, 1] }],
});
const buffers = uploadSceneCopy(device, copy);          // + one texture_2d_array and sampler per size bucket (buffers.textures)
rtSceneCopyLayoutEntries(0, copy.textureLayout);        // bind group layout: table, data, textures, samplers
rtModuleWgsl(caps, { ...format, scene: { slot, layout: copy.layout, textures: copy.textureLayout } });
```

Adds `rtTexSample(slot, uv, lod)`, `rtTexLodOf(record, uvFootprint)`, `rtApplyMaterialTextures(m, uv, footprint)` (glTF semantics),
`rtApplyNormalMap(n, tangent, texel)` and `RT_SURF_TEXFRAME` (`uvDensity`, `tangent`) to `rtSurface`. It works on every WebGPU device (size buckets
of texture arrays, mips, a trilinear sampler per array; at most 8 arrays by default); bindless is a later Tier 3 option behind the same slots.
Textures are sRGB for colours and linear for data (checked against the material field).

**Mips (D34).** `buildSceneCopy` keeps level 0 only (`options.textures.mips: 'gpu'`, the default); `uploadSceneCopy` fills levels 1+ on the GPU with
[`@pmndrs/downsampler`](https://github.com/pmndrs/downsampler) (a dependency of this package: ~630 KB on disk with typings, no required peer dependencies, three.js only for its optional `/three` adapter,
which this package does not import). One `Downsampler` per device is created on first use and cached (`rtMipGenerator(device)`, a WeakMap, so repeated uploads do not recompile;
it keeps no resources of its own beyond the pipelines, so there is nothing to dispose: it goes with the device). Levels 1+ differ from the CPU chain by at most 2 codes (closer to the exact box mean). Settings:

| `textures.mips` | Build | Upload | The CPU twin (`sampleSceneTexture`) |
|---|---|---|---|
| `'gpu'` (default) | level 0 | `rtMipGenerator(device)`, or `uploadSceneCopy({ mipGenerator })` | level 0 only (throws above it) |
| `'cpu'` (or `true`) | every level, ~50 ms per megapixel | the bytes as built | every level; use it for oracles, tests and any consumer that never uploads |
| `false` | level 0, one-level texture | | level 0 |

Scene fingerprints hash level 0 only, so they do not depend on the mode. A custom generator is anything with `encode(encoder, texture)` (`RtMipGenerator`). A textured emitter is not in the light table.
CPU twins and the layout in `scene/texture.ts` (`buildSceneTextures`, `sampleSceneTexture`, `textureLod`, `coneUvFootprint`,
`triangleTextureFrame`, `applyNormalMap`). Design and the open storage choice: [docs/internal/design/scene-copy.md](scene-copy.md).

### L2 shading (P1.3): `rtShade`

```ts
const code = rtModuleWgsl(caps, { ...format, scene: { slot, layout: copy.layout }, shade: { cache: 'external' }, minify: true });
// WGSL: let s = rtShade(hit, ray, RT_SHADE_LAMBERT, vec3f(u0, u1, u2), RT_MASK_ALL);  radiance = rtShadeTotal(s)
```

Emission + one next-event ray + an indirect hook (`rtShadeIndirect`, black by default), in modes `RT_SHADE_FLAT` (no rays), `RT_SHADE_LAMBERT`
and `RT_SHADE_FULL` (builtin diffuse + GGX, or your own `rtShadeBsdf`). TypeScript twin `shade()`. Design, hooks, numbers and open
choices: [docs/internal/design/shade.md](shade.md). The module with shade is 50.2 KB unminified: pass `minify: true`.

### Size transforms and source maps (P2.10, P2.10b)

`rtTransformWgsl(code, { prune, shorten, minify, keep })` -> `{ code, map, dropped }`: `rtPruneWgsl` (call-graph dead declarations),
`rtShortenWgsl` (top-level names to 1-2 letters), `rtMinifyWgsl` (whitespace; line breaks kept). `map` is an `RtWgslSourceMap`
(`lines`: compiled -> source line, `names`: short -> original, `sources`, `enableLines`). Read compiler output with
`rtMapCompilationMessage(msg, map)` / `rtMapCompilationInfo(info, map)` (original line and names, `sourceLine` text),
`rtFormatCompilationMessages`, `rtFormatSourceContext`; `rtCheckShaderModule(module, map, label)` throws the translated text;
`rtCreateComputePipelineAsync(device, { code, map, layout, entryPoint, sourceMap? })` creates the module and pipeline, does nothing extra on
success and throws the translated text on failure. `rtWgslSourceMapV3(map)` makes a Source Map v3 (line level). Composition:
`rtComposeSourceMaps`, `rtComposeLineMaps`. `rtPruneWgsl` now also returns `lineMap`.

### Budgets

Storage buffers per kernel (`rtModuleStorageBuffers`): walk 3 (4 indexed), scene copy 2 (surface, materials and lights together), batch IO 2-6.
Textures add no storage buffers: one sampled texture and one sampler per size bucket (at most 8 by default of the 16 a stage may bind). Generated
kernels stay under 50 KB (largest today ~45 KB, unit-tested for every setting combination).
