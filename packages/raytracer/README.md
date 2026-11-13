# @pmndrs/raytracer

The trace core: rays in, hits out, on WebGPU, over [`@pmndrs/bvh`](../../../bvh)'s scene buffers and WGSL walk.
WGSL names start with `rt` / `Rt` / `RT_`; the bvh walk underneath keeps its own `bvh*` names.

Status: **P1 slice 1** (L0, L1, `rtSurface`, batch dispatcher). Correct against bvh's CPU oracle on the M5
(`node bench/runner/run.ts --page trace-core`); speed numbers are provisional. Not yet: L2 shading, the
material/light scene copy, L3. Design notes: [docs/design/trace-core.md](../../docs/design/trace-core.md).

## API

### Contract (`contract/ray.ts`)

| Name | What |
|---|---|
| `Ray` | `{ origin, tMin, direction (unit), tMax }`, 32 B (`RAY_BYTES`), bvh's `BvhRay` |
| `Hit` | `{ t (< 0 miss), instance, triangle, frontFace, barycentrics (v1, v2), material, normal }`, 40 B (`HIT_BYTES`); compact record 24 B (`HIT_COMPACT_BYTES`) without material and normal |
| `RayCone` | `{ width, spread }`, 8 B. Carried through L1 only; nothing consumes it yet |
| `BVH_LAYOUT_VERSION` | 2. `assertBvhLayout()` runs before any walk is generated |

### Two ways to trace

**Inline**: paste the module into your own compute shader.

```ts
import { rtModuleWgsl, rtWalkEntries, sceneFormatOf, withBindings, rtSceneBindGroupEntries } from '@pmndrs/raytracer';

const format = sceneFormatOf(sceneTrees.traceBuffers); // nodeFormat, triangleLeaves, positionLayout: compiled in
const code = rtModuleWgsl(caps, {
  ...format,
  traceSlot: { group: 0, firstBinding: 0 },          // bvh walk: 3 bindings (4 for indexed leaves)
  surface: { slot: { group: 2, firstBinding: 0 }, layout: table.layout }, // optional rtSurface: 3 bindings
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
Its data comes from `buildSurfaceTable(meshes, instances, options)` + `uploadSurfaceTable(device, table)`, a small
stand-in until the scene copy (D4) lands.

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

### Budgets

Storage buffers per kernel (`rtModuleStorageBuffers`): walk 3 (4 indexed), `rtSurface` 3, batch IO 2-6. Generated
kernels stay under 50 KB (largest today ~45 KB, unit-tested for every setting combination).
