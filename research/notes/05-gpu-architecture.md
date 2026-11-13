# 05 - GPU ray tracing architecture and software traversal performance

Date: 2026-11-13. Area: wavefront vs megakernel, traversal kernels, BVH formats, approximate tracers, GPU-driven
plumbing, WebGPU/Tint pitfalls, and Mrays/s targets. Fits the existing `@pmndrs/bvh` (~/Developer/bvh) layout.

Conventions: **[V]** = number read from a paper/page I opened this session. **[K]** = from my own knowledge, not
re-verified. **[E]** = my estimate or scaling, treat as a hypothesis to benchmark. Anything marked "unverified" must
not go into a spec without a check.

## 1. TL;DR

- **Do not pick one execution model. Ship two kernels on one BVH.** (a) A *megakernel* trace(+minimal shade) for
  utility consumers (GI probes, AO, shadows, reflection rays, visibility). Their shading is trivial, so wavefront
  overhead is pure loss. (b) A *wavefront* path tracer (queues, SoA path state, per-shader-class kernels, indirect
  dispatch) for the full PBR tracer, where materials are expensive and diverse. Laine et al. measured wavefront
  gains of +36% (simple, 9.5K tris), +79% (879K tris), +221% (283K tris, expensive materials) [V]. The gain scales with
  shading cost, not with trace cost.
- **WebGPU wavefront is viable because the per-dispatch cost is small and indirect dispatch exists.** Measured Chrome
  per-dispatch cost with no sync: 33 us (RTX 5090/Linux), 59 us (RTX 2000 laptop/D3D12), 67 us (Intel iGPU/D3D12)
  [V, arXiv 2604.02344 v1, **withdrawn by its author and superseded by arXiv:2608.08730**; the successor repeats the 24-36 us (Vulkan) and 32-71 us (Metal) ranges, Safari/M2 31.7 us and Firefox ~1 ms in the v1 table, but its per-browser table could not be read, so treat the Chrome rows as [V-v1]; see VERIFY.md]. ~40 dispatches per iteration (8 stages x 5 bounces) = 1.3-2.7 ms of CPU/driver floor, which
  is fine for a 10+ ms path-tracing iteration and ruinous for a 0.2 ms GI probe update. This is the quantitative
  reason for the two-kernel split. It also means: record all bounces in one command buffer, never read back.
- **Our traversal is the Aila-Laine BVH2 baseline. The biggest known software win is CWBVH (8-wide, 80 B/node,
  octant-ordered, compressed stack): 1.9-2.1x over Aila-Laine on incoherent rays** (Titan X Pascal/Maxwell, CUDA)
  [V, Ylitie 2017, re-checked against the PDF 2026-11-13: geomean vs Aila 2012 = 2.24x bounce 1, 2.51x bounce 4 on Maxwell (Table 5), 1.33x primary; Pascal (Table 4) 2.36x / 2.70x; abstract 1.9-2.1x holds]. But it is *only* a win when
  memory-bound and incoherent; on primary rays it is 1.3x. Our own quantized-BVH2 variant is 0.93-0.97x of f32 on the
  M5 [V, bvh docs], which matches the theory: compression without width removes bytes, not node fetch latency. **Prototype
  BVH4 (tinybvh `BVH4_GPU`) and CWBVH8 against the current BVH2 as the first traversal experiment.** tinybvh already
  builds both from the binary tree.
- **Target numbers.** Today: 104 Mrays/s closest-hit incoherent on Sponza, RTX 3060 Laptop, Chrome, BVH2 [V, bvh
  README]. Native CUDA CWBVH on a Pascal Titan X (similar raw class) does **658 Mrays/s** on the first diffuse bounce of
  CrySponza [V Ylitie Table 4; the earlier "360" is the *Maxwell* Titan X, Table 5, 361 Mrays/s, and was mislabelled Pascal]. The gap to our 104 is 3.5x against Maxwell and 6.3x against Pascal (baseline Aila-Laine on the same Pascal card does 276); different GPU class and scene, so a hypothesis only; a realistic web target is 1.5-2x of today (150-200 Mrays/s) with wide
  BVH + traversal tuning. Per-class table in section 6.
- **Ray sorting: cheap binning yes, global sorting no.** Meister et al. get 1.5-1.7x *trace-only* speedup on secondary
  rays, but on an RTX 2080 Ti the sort overhead (1.4-3.7 ms on Bistro) ate it: "does not pay off overall" with
  hardware kernels [V]. Their software CUDA kernel did pay off overall (no numbers given) and showed up to +41%
  (secondary) / +112% (shadow) control-flow efficiency [V]. A software tracer is the better case for reordering than
  RTX. Start with free coherence: octant binning (needed anyway for CWBVH child order), tile-coherent generation, and
  workgroup-local bucketing; only then try a coarse (origin cell + direction octant) key with a block-wise sort.
- **SER has no WebGPU equivalent; build the three software analogues:** (1) per-shader-class queues with indirect
  dispatch, (2) workgroup/subgroup-local bucketing by material id, (3) subgroup waterfall loops (`subgroupBroadcastFirst`
  + `subgroupBallot`) for the divergent remainder. NVIDIA SER itself gains 11-24% (Vulkan sample), 20-50% (NVIDIA
  guidance), 24% Indiana Jones, 39% Alan Wake 2, up to 370% Black Myth Wukong [V via search, secondary]. Treat 10-30% as
  the ceiling for a software emulation.
- **Approximate tracers beat triangle BVHs for GI and rough reflections in specific, identifiable cases** (overlapping
  or huge instance counts, far-field, rough/low-frequency consumers). Lumen on PS5 1080p, same content: global-SDF final
  gather tracing 0.94 ms vs HWRT 10.03 ms in an overlapping-mesh scene, but 1.21 vs 1.41 ms in a clean scene [V, Lumen
  SIGGRAPH 2022]. Offer a **voxel/SDF far-field tier** behind the same ray interface; our siblings WebVDB and the DDGI
  repo are natural homes. Triangle BVH stays for mirror/sharp, thin, skinned, exact shadows.
- **WebGPU hazards that will cost real time:** compile time on Windows (DXC/FXC, size-superlinear; see section 5),
  Metal fma contraction breaking the watertight triangle test (already hit and fixed in bvh as `bvhEdgeFunction`, 4,513
  leaking rays per 2M -> 0 [V, bvh HANDOFF]), 16 KiB default workgroup storage, 8 storage buffers per stage, no
  forward-progress guarantee between workgroups, robustness clamps on every storage access.

## 2. Sources

Local paths are under `research/papers/gpu-arch/`. Open-access only.

| Name | Link | Year / venue | Local | Why |
| --- | --- | --- | --- | --- |
| Megakernels Considered Harmful (Laine, Karras, Aila) | https://research.nvidia.com/publication/megakernels-considered-harmful-wavefront-path-tracing-GPUs | 2013 HPG | `laine2013-megakernels-considered-harmful.pdf` | wavefront design, queues, SoA, numbers |
| Understanding the Efficiency of Ray Traversal on GPUs (Aila, Laine) | https://research.nvidia.com/sites/default/files/pubs/2009-08_Understanding-the-Efficiency/aila2009hpg_paper.pdf | 2009 HPG | `aila2009-understanding-efficiency-of-ray-traversal-on-gpus.pdf` | persistent threads, while-while/if-if, speculative traversal, 64 B node layout |
| Efficient Incoherent Ray Traversal on GPUs Through Compressed Wide BVHs (Ylitie, Karras, Laine) | https://research.nvidia.com/sites/default/files/publications/ylitie2017hpg-paper.pdf | 2017 HPG | `ylitie2017-cwbvh-compressed-wide-bvh.pdf` | CWBVH, compressed stack, numbers on Titan X |
| On Ray Reordering Techniques for Faster GPU Ray Tracing (Meister, Bokansky, Guthe, Bittner) | https://arxiv.org/abs/2506.11273 | 2020 I3D, arXiv 2025 | `meister2020-on-ray-reordering-techniques-arxiv2506.11273.pdf` | sort keys (Two Point), overhead analysis |
| Faster Ray Tracing through Hierarchy Cut Code (Xiang et al., not Meister) | https://arxiv.org/abs/2305.16652 | 2023 | `xiang2023-hierarchy-cut-code-arxiv2305.16652.pdf` | cheaper sort key, up to 1.81x secondary |
| H-PLOC (Benthin, Drabinski, Tessari, Meister, Wald) | https://gpuopen.com/download/publications/HPLOC.pdf | 2024 HPG | `benthin2024-hploc.pdf` | 0.8-1 Gtri/s build; quality vs LBVH |
| Fast Parallel Construction of High-Quality BVHs (Karras, Aila) | https://users.aalto.fi/~ailat1/publications/karras2013hpg_paper.pdf | 2013 HPG | `karras2013-fast-parallel-construction-hq-bvh.pdf` | treelet restructuring (quality tier) |
| A Survey on BVH (Meister et al.) | https://meistdan.github.io/publications/bvh_star/paper.pdf | 2021 CGF | `meister2021-survey-bvh.pdf` | builder/traversal survey |
| Robust BVH Ray Traversal (Ize) | https://jcgt.org/published/0002/02/02/paper.pdf | 2013 JCGT | `ize2013-robust-bvh-ray-traversal.pdf` | slab test robustness (1+2 ulp) |
| Watertight Ray/Triangle Intersection (Woop, Benthin, Wald) | https://jcgt.org/published/0002/01/05/paper.pdf | 2013 JCGT | `woop2013-watertight-ray-triangle.pdf` | what bvh's triangle test is |
| Lumen (Wright et al.) | https://advances.realtimerendering.com/s2022/SIGGRAPH2022-Advances-Lumen-Wright%20et%20al.pdf | 2022 SIGGRAPH Advances | `wright2022-lumen-siggraph-advances.pdf` | SWRT vs HWRT ms on PS5 |
| GI with FidelityFX Brixelizer (AMD) | https://gpuopen.com/download/GDC2024_GI_with_AMD_FidelityFX_Brixelizer.pdf | 2024 GDC | `gdc2024-brixelizer-gi.pdf` | brick-atlas SDF + 3-level AABB tree |
| NVIDIA SER whitepaper | https://developer.nvidia.com/sites/default/files/akamai/gameworks/ser-whitepaper.pdf | 2022 | `nvidia-ser-whitepaper.pdf` | semantics of ReorderThread + coherence hints |
| Characterizing WebGPU Dispatch Overhead (arXiv 2604.02344) | https://arxiv.org/abs/2604.02344 | 2026 | link only | per-dispatch us by vendor/browser |
| Dawn/Tint docs (subgroup size control, uniformity) | https://dawn.googlesource.com/dawn/+/HEAD/docs/tint/ | live | link only | subgroup/uniformity behavior |
| tinybvh (Bikker, MIT) | https://github.com/jbikker/tinybvh | live | link only (bvh pkg already vendors 1.9.1) | BVH_GPU / BVH4_GPU / BVH8_CWBVH layouts, H-PLOC |
| mutracer (pissang) | https://github.com/pissang/mutracer | live | not cloned (read-only compare) | uses TinyBVH WASM SBVH + CWBVH; M4 Max demo |
| Wide BVH thesis (Cezner, CTU) + CESCG 2025 | https://cescg.org/wp-content/uploads/2025/04/Cezner-Wide-Bounding-Volume-Hierarchies-for-Ray-Tracing-1.pdf | 2025 | link only | ~20% for 4-ary vs binary in Vulkan compute [secondary, search snippet] |
| Cycles kernel scheduling | https://wiki.blender.org/features/cycles/kernel_scheduling/ | live | link only | Cycles wavefront design |
| pbrt-v4 wavefront chapter | https://pbr-book.org/4ed/Wavefront_Rendering_on_GPUs | 2023 | link only (fetch failed this session; summarized from memory [K]) | queues, SoA |

Not found / not saved: Aila 2012 Kepler tech report, Binder-Keller 2016, Hapala 2011 stackless (URLs 404 or not
open); cited from memory only [K].

## 3. Findings: execution model

### 3.1 Megakernel vs wavefront

What it is. Megakernel: one thread owns a path for its whole life (trace, shade, sample, loop). Wavefront: path state
lives in global SoA arrays; separate kernels per stage (generate, extend/trace, shade per material class, shadow
trace, accumulate); queues route paths to the next stage; each kernel is launched over a compacted queue.

Numbers [V, Laine 2013, GTX 680-class, 1M path pool, Mpaths/s]:

| Scene | tris | Megakernel | Wavefront | Gain |
| --- | --- | --- | --- | --- |
| Carpaint | 9.5K | 42.99 | 58.38 | +36% |
| City | 879K | 5.41 | 9.70 | +79% |
| Conference | 283K | 2.71 | 8.71 | +221% |

Per-iteration time of 1M path segments on City: logic 3.42 ms, new-path 0.86 ms, materials 5.47 ms, ray cast 12.53 ms
[V]. Megakernel thread utilization 23% vs wavefront 53% on Conference [V]. Design details worth copying: SoA layout
alone gave +80% over AoS, with the new-path kernel +790% (writes) [V]; atomic-add coalescing for queue appends gave +40%
total [V]; ray cast kernels sped up 32% from coherence of compacted work alone [V]; pool size ~1M paths saturates
(+5% from larger pools) [V]. Cost: every stage round-trips path state through memory, so memory traffic and per-stage
launch overhead are the price. For simple materials the win is only +36%, and for trivial shading it inverts.

Production: pbrt-v4 GPU is wavefront with work queues and per-material-type queues [K]. Blender Cycles X (3.0) is
wavefront on GPU only, megakernel on CPU, with shader-sorting by local atomics giving 2-3% [V via search of Cycles
source commit, secondary]; Cycles X claims up to ~7x vs old Cycles on NVIDIA OptiX [V via search, secondary; the
gain mixes in hardware RT, so it is not a pure wavefront number].

WebGPU mapping.
- No device-side enqueue; the pattern is: stage kernel writes a counter via `atomicAdd` to a queue buffer; a one-thread
  "args" kernel converts count -> `ceil(count / WG)` into an indirect-args buffer; next stage uses
  `dispatchWorkgroupsIndirect`. Buffer usage `INDIRECT | STORAGE`; the args kernel is a separate dispatch so there is no
  usage conflict. Cap: `maxComputeWorkgroupsPerDimension` 65,535 -> 16.7M threads at WG 256 in 1D; with 1M-path pools
  this is irrelevant [K].
- **The queue must be bounded and over-dispatched safely**: each thread checks `gid < count`, so a stale/over-large
  args value is harmless.
- Per-dispatch floor 33-67 us in Chrome (section 1). Dawn validates/clamps indirect dispatch args; its cost per call
  is not measured here (unverified; add to bench list).
- **One shader class per kernel, not one kernel per material.** Per-material dispatches explode the count. Use
  ~4-6 classes (opaque PBR/layered, transmissive/dielectric, volume, emissive/miss/env, alpha-tested continuation) and
  pick the concrete BSDF inside the kernel from a material table (bindless texture array where available, otherwise
  atlas/array textures). This also helps *compile time*: smaller kernels compile faster and in parallel (section 5).
- Memory: 64-96 B/path in SoA with f16/packed fields => 1M paths = 64-96 MB, under the 128 MiB default
  `maxStorageBufferBindingSize` per buffer if split by field group; the 8-storage-buffers-per-stage default is the tighter
  limit, so bind groups per stage must be designed (bindless removes this).
- Tile/batch the frame: 256K-1M paths per wavefront (Laine's pool), not the whole 1080p frame x bounces, to bound
  memory and keep L2 hot.

Rating: **Adopt now** (wavefront for the full path tracer); **Adopt now** (megakernel `bvhTraceClosest/Any` for
utilities - already what `@pmndrs/bvh` ships).

### 3.2 Persistent threads and work stealing

Aila & Laine 2009 [V] (GTX285, Mrays/s, Conference/Fairy/Sibenik, primary): while-while 88/47.8/81.2, persistent
while-while 135.6/70.5/117.5 (+54%/+47%/+45%), persistent speculative while-while 142.2/74.6/117.5. Diffuse: while-while
44.5 -> 62.4 persistent (+40%). Mechanism: launch exactly enough threads to fill the machine, each fetches ray batches
from a global atomic counter, so a few long rays do not hold whole warps hostage; dynamic fetch when active lanes drop
below a threshold. Simulated ceiling was 85-92% of ideal SIMD efficiency.

Modern relevance [K, not re-verified]: Aila's 2012 follow-up found hardware schedulers (Fermi/Kepler) shrank the gap,
and persistent threads remain useful mainly with long incoherent traversals. Ylitie 2017 still uses persistent
threads with a dynamic-fetch heuristic (fetch when idle lanes cross a threshold) and gets measurable gains from it
(Table 9, not extracted here).

WebGPU: the *fetch-from-atomic-counter* form is safe because it never spin-waits on another workgroup (WebGPU gives
no inter-workgroup forward-progress guarantee, so decoupled-lookback-style waiting is risky; a plain `atomicAdd` work
counter is not). Use workgroup-sized launches = (occupancy-estimate x SM/CU count) -- but WebGPU does not expose SM
count. Practical alternative: launch more workgroups than the machine holds and let each workgroup loop over
`atomicAdd(counter, WG_SIZE)` chunks of the ray queue. Rating: **Prototype** (bench vs plain one-ray-per-thread
indirect dispatch on a 1M-ray incoherent batch; the win could be zero on Apple/modern NVIDIA).

### 3.3 Stream compaction

Needed in two places: compacting alive paths between bounces, and appending paths to per-class queues. Cheapest
correct form in WGSL (needs the `subgroups` feature; Chrome has it, Safari/Firefox not yet per bvh README):
`subgroupBallot` + `countOneBits` for the lane's rank, `subgroupElect` lane does one `atomicAdd(&queue.count, n)`,
`subgroupBroadcastFirst` base, every lane writes at `base + rank`. That is one global atomic per subgroup instead of
per lane (Laine measured +40% total from this idea) [V]. Without subgroups: workgroup-level via shared-memory scan,
one atomic per workgroup (slower by a constant factor, same structure). `subgroupBallot` returns `vec4<u32>`, so it
covers up to 128 lanes [K]; do not hardcode 32. Rating: **Adopt now**.

### 3.4 Ray sorting / binning (coherence)

Meister et al. (I3D 2020; arXiv 2025 copy) [V]: RTX 2080 Ti, 1080p, 8 spp, wavefront, 9 key schemes (Origin, Reis,
Costa, Aila, Aila Compact, Octahedron, Two Point 0.25, Two Point Adaptive, Two Point Real). Best average trace speedup
(DX): 1.63x secondary, ~1.54x shadow; (OptiX) 1.53x secondary, 1.58x shadow (Octahedron). Sponza secondary 1755 ->
3033 Mrays/s; Bistro 655 -> 1329 Mrays/s. Sort throughput ~1.6 GKeys/s at 1M keys (~0.4-0.6 ms per 1M keys); total
reorder overhead on Bistro 1.43-1.59 ms block-wise (only 1.03-1.12x trace gain) vs 3.66 ms device-wise (1.85x, but
overhead larger than the saving). Conclusion: not worth it on RTX at these scene sizes; may pay off on large scenes or
with a slower (software) trace kernel; their earlier CUDA kernel with *indirect sorted indices* (no physical ray move)
paid off overall. **Take-away: sort indices, not ray data; key by (direction octant, coarse origin cell Morton)
or the Two-Point estimate; do it block-wise first.** Xiang et al. 2023 hierarchy-cut-code: up to 1.81x secondary,
claims lower overhead; on open/landscape scenes all reorderings give only 1.1-1.2x [V].

Cheap coherence that needs no global sort (all **Adopt now / Prototype**):
1. Octant binning: 3 bits of direction signs. Required for CWBVH ordered traversal; also free for BVH2 child ordering.
2. Tile-coherent generation: primary rays in 8x8 tiles; for diffuse bounces, share one Sobol/blue-noise dimension
   across a tile's cosine-hemisphere sample so directions cluster (biased only in correlation, not expectation, if
   randomized per tile per frame).
3. Workgroup-local counting sort of 256 rays by (octant, material class, 4-bit origin cell) in shared memory
   (16 KiB default is ample for 256 x 4 B keys + 256 x 4 B indices), then trace in sorted order.
4. Sort-by-material for shading via queues (3.1), not a sort kernel.

### 3.5 SER-style execution reordering in software

NVIDIA SER (`ReorderThread(hit, coherenceHint, nBits)`) reorders threads after trace by hit/material so the shader
runs coherently, with the key = hit-object shader id + hint bits [V, whitepaper]. Not exposed on the web (DXR SM 6.9
standardized it Feb 2026 [V via search]; WebGPU has no ray query/hit-object path, so no analogue is coming soon).
Software analogues, in order of cost:
- queue-per-class with indirect dispatch (3.1);
- workgroup bucketing (3.4 #3);
- subgroup waterfall: `loop { let m = subgroupBroadcastFirst(mat); if mat == m { shade(m); break; } }` bounded by
  the number of distinct materials in the subgroup;
- hint bits: carry "path terminating?" and "needs NEE?" in the queue key (the whitepaper's own recommended hints).
Expected gain ceiling 10-30% for expensive-material scenes [E from V game numbers]. Rating: **Prototype**.

## 4. Findings: traversal kernels and BVH formats

### 4.1 What `@pmndrs/bvh` does now (from README, `docs/concepts/gpu-layout.mdx`, `src/trace/wgsl.ts`)

BVH2, 64 B nodes with both child boxes (Aila-Laine), leaves 1-8 tris (tinybvh/H-PLOC default <=4 in own builder),
48 B watertight triangle records in tree order (or 16 B indexed, 0.75-0.99x speed), optional 32 B quantized nodes
(0.93-0.97x speed, -20% memory), single loop + private `array<u32,64>` stack, near-first for closest, far-first
default for any-hit (1.1-2.8x faster shadows), instance marker on the stack for the 2-level walk, object-space ray not
renormalized so `t` stays in world units, 8-bit instance masks, `bvhTrianglePasses` filter hook, Metal-safe
`bvhEdgeFunction`. Measured: Sponza (5.74M tris) closest-hit 104.3 Mrays/s on RTX 3060 Laptop (Chrome 156/Windows)
= 1.20x three-mesh-bvh; Intel UHD 12.8 Mrays/s; M5 `three/pathtracer` example ~120 Mrays/s (wall, indicative);
H-PLOC build 46.6 ms (123 Mtri/s, vs the paper's 800-1000 Mtri/s native peak [V]). This is a sound, conservative
baseline. The stack lives in private memory (spills to local/scratch on many GPUs) - the largest unexploited lever.

### 4.2 Wide BVHs and CWBVH

- tinybvh layouts [V from README]: `BVH` 32 B nodes; `BVH_GPU` 64 B (Aila-Laine, what we use); `BVH4_GPU` compact
  quantized 4-wide ("may be faster for GPU ray tracing"); `BVH8_CWBVH` 80 B 8-wide (Ylitie); `BVH4_CPU` fastest on
  CPU. Bikker's only headline GPU number: up to 5 Grays/s in Sponza with the basic binary BVH on an RTX 5080 laptop
  [V]; the README does not say ray type (almost certainly coherent primary, so not comparable to our incoherent 104
  Mrays/s). No per-layout table exists in the README; I could not find Bikker's own per-layout timings.
- CWBVH [V, Ylitie 2017]: 8-wide, 10 B/child internal, 1:3.2 compression vs uncompressed BVH2; node = 80 B;
  triangle data 48 B -> or compressed; memory 35-60% of a typical BVH; stack entries compressed to 8 B (up to 12
  entries/thread = 96 B in shared memory, spilling rare); ordered traversal chosen per ray octant with *no per-node
  sort* (child order from octant + occupancy bits); decompress + 8 slab tests per node using popcount and byte-wise
  tricks. Results Titan X Maxwell geomean vs Aila 2012: primary 1.33x, bounce-1 diffuse 2.24x, bounce-4 2.51x; Pascal
  incoherent 2.0-2.7x. Abstract: 1.9-2.1x. Absolute (Maxwell, Mrays/s, CWBVH): CrySponza 616 primary / 361 bounce-1 /
  346 bounce-4; Conference 1416/676/621; Hairball 300/174/67; Sibenik 1059/469/439 [V]. Their comparison also shows
  the 4-wide of Guthe 2014 at 1.1-1.2x and stackless (Binder-Keller 2016) at ~1.2x [V].
- Independent 2025 evidence for *software* wide BVH on Vulkan compute: ~20% for 4-ary over binary [secondary, search
  snippet of CESCG 2025 / CTU thesis, not opened]. This is the realistic expectation for BVH4 in WGSL (not 2x).
- WGSL feasibility: `countOneBits`, `firstLeadingBit`, `extractBits`, `dot4U8Packed`/`pack4xU8`/`unpack4xU8` (language
  feature `packed_4x8_integer_dot_product`, [K]) cover most of the bit tricks; there is no `__byte_perm`/`vsub4`, so child
  box decompression needs ~2x the ALU of CUDA. Shared-memory stack at 12 entries x 8 B = 96 B/thread; at WG 64 it is
  6 KiB, WG 128 = 12 KiB, inside the 16 KiB default `maxComputeWorkgroupStorageSize` [K]; Apple adapters often report
  32 KiB. Occupancy tradeoff is real: workgroup memory competes with registers.
- mutracer already feeds TinyBVH-WASM CWBVH/SBVH to a WebGPU tracer [V, its README]; its only published number is
  an M4 Max, 1080p, 64 spp, 6 bounces, 4.6M tris in 12.1 s (~11 Mpaths/s; ray rate not reported). A like-for-like
  benchmark of its CWBVH path against ours is a cheap way to see whether CWBVH wins on Metal and D3D12 under WGSL.
- Rating: **Prototype** CWBVH8 and BVH4_GPU as additional `nodeFormat` variants next to `'f32' | 'quantized'`, same
  hit contract. Decision gate: >= 1.3x closest-hit on 3060L *and* M5 incoherent rays, else keep BVH2.

### 4.3 Traversal loop structure

- **while-while vs if-if** [V Aila 2009]: on GTX285, if-if had better SIMD efficiency in some cases but while-while
  won by fewer instructions; with persistent threads while-while led (135.6 vs 122.1 persistent packet etc.). Modern
  compilers/hardware blur this; our single-loop is the "if-if/merged" family. Rating: **Reference only**; one
  micro-bench (`while(inner) {...} while(leaf) {...}`) is cheap.
- **Speculative traversal** (postpone leaf, keep traversing while other lanes are in leaf): +4-5% on persistent
  while-while (135.6 -> 142.2) on GTX285 [V]; Aila notes simulated gains need `popc`/ballot compaction. In WGSL, subgroup
  ballot makes it implementable, but expected modern gain is small. Rating: **Later**.
- **Stackless / short-stack / restart trails** [K + V Ylitie table]: Binder-Keller backtracking in constant time ~1.2x
  over Aila baseline for incoherent rays on Maxwell/Pascal [V]; it needs a fixed-topology BVH (8-wide or binary with
  bit-trail) and no early-out ordering tricks. Restart trail (Laine 2010) keeps a bit trail instead of a stack [K].
  Short-stack (e.g. 4-8 entries in registers/shared, restart from root when exhausted) lowers private-memory
  traffic. For us: the stack is a 64-entry `array<u32,64>` in private memory; first experiment = move to
  workgroup memory (Ylitie style, 8-12 entries + overflow to private) and measure; second = bit-trail stackless on
  BVH4. Rating: **Prototype** (shared-memory short stack).
- **Treelet / speculative cache-friendly traversal** (Aila-Karras 2010 treelets, Vaidyanathan stackless multi-BVH)
  [K, links not retrievable this session]: more relevant for ray *queues per treelet* (memory-bound out-of-core), not
  for a 5M-tri scene resident in L2/VRAM. Rating: **Reference only**.
- **Any-hit early out:** terminate on first accepted hit (already done); far-first child order default [V bvh];
  Ize slab test with 1+2 ulp padding for robustness [V paper in folder].

### 4.4 Primitive and box tests

- Ray-box: precomputed `1/d` and `o * (1/d)` fused as `fma(o, invD, -o*invD)` form (`t = fma(b, invD, -oInvD)`)
  saves one op per axis; `min/max` over three axes with NaN-safe forms (`max(a, b)` semantics differ across back ends:
  WGSL `min/max` NaN behavior is implementation-defined [K], so use the Ize padding instead of relying on
  NaN-propagation); `tMax` multiplication by `1 + 2 ulp` (Ize) for robustness. All [K] except Ize [V].
- Ray-triangle: Moller-Trumbore (3 stored vertices -> edges recomputed or precomputed `e1,e2`: 36 B + id),
  Woop watertight (shear to ray space; 48 B records; shared-edge safe) [V paper], Baldwin-Weber (precomputed 48 B
  projection data, fewer flops, not watertight-by-construction) [K]. We use watertight edge functions already; on
  Metal the compiler's fma contraction broke it until `bvhEdgeFunction` [V bvh HANDOFF], proof that precision, not
  ALU count, is the constraint. Keep watertight for correctness-critical paths (shadow/leak) and consider a fast
  precomputed-edge variant for *noisy* consumers (GI probes) behind a flag; measure first. Rating: **Reference only**
  / **Prototype** for the fast-variant.
- AABB in f16 / quantized: our quantized 32 B node is already a "per-mesh grid" variant, outward rounded with 1/1024
  step margin [V bvh docs]. CWBVH uses 8 bit per plane in a local grid per node (15 B grid) [V]. f16 boxes need
  conservative rounding (round min down, max up), and the `shader-f16` feature, which is not on every GPU/adapter.
  Rating: **Later**; do it as part of CWBVH, not standalone.
- Triangle compression: 48 B -> 16 B indexed (ours, 0.83-0.99x), 32 B (Ylitie reports bytes/triangle tables; DGF/AMD
  dense geometry format for HW tier [K]). On GPUs where the BVH working set exceeds L2 (>= 100K tris) bandwidth
  dominates incoherent rays; on small scenes compression loses. Rating: **Prototype** only after CWBVH; keep records.
- SoA vs AoS: node = one 64 B/80 B cache-line-friendly AoS record (what we have); path state = SoA (Laine +80%) [V].

### 4.5 Builds, refit, instancing

- H-PLOC [V Benthin 2024]: 0.8-1.0 billion tris/s, SAH within ~3% of PLOC++; LBVH 1.10-1.38x worse SAH than a PRBVH
  reference; 1M instances TLAS in 2.21 ms vs 1.96 ms for LBVH. Ours: SAH 1.032x tinybvh, closest-hit 0.95x, any-hit
  1.02x; LBVH 0.82x/0.80x [V bvh]. **Keep H-PLOC as default; LBVH only for per-frame rebuilds of dynamic meshes.**
- Refit vs rebuild: refit (0.09 ms for 10k-instance TLAS [V]) preserves topology and degrades with deformation;
  rule of thumb [E]: track refit-time SAH ratio per mesh (cheap, one reduction) and rebuild when >1.25-1.5x or every
  N frames. Treelet restructuring (Karras-Aila 2013 [V paper]) / insertion-based tree rotations (tinybvh `Optimize`:
  ~8% SAH in <30 s offline on Sponza [V]) are the repair path; neither is real-time in WebGPU today.
- TLAS/BLAS in software: one stack, instance marker, ray transformed per instance (12 FMA + pushing a marker).
  Cost grows with *overlap* of instance boxes (Lumen: ~100 meshes overlapping per surface point made HWRT 10x slower
  than merged global SDF [V]). Our `mergeStaticInstances` is the same remedy: 1.85x closest-hit on Sponza surface rays
  [V bvh]. Rating: **Adopt now** (already built); make merge-static the default for static scenes in the path tracer.

## 5. Findings: GPU-driven pipeline and WebGPU specifics

### 5.1 Occupancy, registers, workgroup size

- WebGPU exposes none of the register/occupancy controls. What you control: workgroup size, workgroup memory, loop
  structure, live-state across the trace call, and function inlining by the back end.
- **Live-state across trace dominates register pressure in a megakernel.** The wavefront fix is to store state to SoA
  and reload; NVIDIA's own Indiana Jones SER write-up pairs reordering with "live state reductions" [V title, search].
  In WGSL: keep path state in a `struct` of <= ~24 scalars, f16-pack where `shader-f16` exists.
- Workgroup sizes [K, vendor docs not re-fetched; verify per adapter via `GPUAdapterInfo.subgroupMinSize/MaxSize`,
  which Dawn sources from Vulkan `VkPhysicalDeviceSubgroupSizeControlProperties` [V]]:
  - NVIDIA: warp 32. Use 64-128 for traversal (shared stack) or 32-64 without.
  - AMD RDNA: wave32 default for compute (can be 64); GCN wave64. Use 64 as a safe multiple of both.
  - Apple: fixed subgroup 32 [V, Tint docs: "all the Apple GPUs have a fixed subgroup size of 32"]; 32 KB threadgroup
    memory is common but WebGPU default is 16 KiB unless the adapter reports more [K]. Use 64-128.
  - Intel Xe: subgroup 8/16/32 variable; Dawn exposes a range. Use 64 and never assume a subgroup size; compute rank
    with ballot, not lane math.
  - Qualcomm Adreno: 64 or 128 [K]. Mali Valhall: 16 [K]. Mobile prefers small workgroups (32-64) and avoids large
    workgroup memory. Mobile numbers are unverified.
  - Policy: default WG = 64 for trace/queue kernels, 128 for the CWBVH shared-stack variant, with `override`
    constants so one WGSL source can be retuned per adapter (a single device-class table in the library).
- `subgroup_size_control` exists in Tint (`@subgroup_size`, power-of-two, workgroup size a multiple) [V docs] but is
  not in Chrome stable as far as I verified; do not depend on it.

### 5.2 Subgroup uses (wave-level)

1. queue compaction (3.3); 2. subgroup-uniform BVH descent: `subgroupAll`/`subgroupBroadcastFirst` to detect ray
   groups sharing a node and fetch once (coherent rays, e.g. primary/shadow-to-same-light) [K]; 3. waterfall (3.5);
   4. reductions for refit (already in bvh: refits need `subgroups`) and for SAH-ratio tracking; 5. luminance/
   moment reductions for adaptive sampling. Tint's uniformity analysis is conservative, so barrier-in-loop code can be
   rejected where it is actually uniform; `workgroupUniformLoad` is the sanctioned way to broadcast a loop-control
   value [V docs].

### 5.3 Packing

Octahedral normals (2x u16 / 1 u32), RGB9E5 (1 u32, shared exponent; no native WGSL pack, implement by hand with
`bitcast`/`floor(log2)`; textures can use `rgb9e5ufloat` format [K]), pack hit barycentrics as 2 x u16 unorm, 32-bit
material/instance/triangle ids (our hit record is 24/40 B), f16 for radiance throughput (`shader-f16`, check adapter),
`pack2x16float`/`unpack2x16float` always available [K]. Ray record = 32 B today (`BvhRay`); queue ray = origin (3 f32) +
dir (octahedral u32) + tMax + pixel/path id = 24 B. Rating: **Adopt now** for path state and hits.

### 5.4 Visibility buffer as primary-hit replacement

Rasterize primaries into a visibility buffer (instance id, triangle id, barycentrics; `rg32uint` or 2 x u32 plus
depth), shade with ray-tracing-free material code, and trace secondaries only. Eliminates the primary trace
(typically 15-30% of ray count in a 3-4 bounce path tracer [E]) and gets free AA/depth for denoiser guide buffers and
Hi-Z (our `downsampler`). On the web this requires a rasterization path from the host engine (adapter concern);
for pure compute adopters, a coherent compute primary trace is the fallback (primary rays are 2-5x faster than
incoherent on every published table [V Ylitie]). Rating: **Adopt now** as an adapter option; engine-agnostic core
accepts "hit already known" paths (instance, tri, bary) as the stage-1 output.

### 5.5 WebGPU / Tint / Dawn pitfalls (with provenance)

| Pitfall | Detail | Mitigation |
| --- | --- | --- |
| Compile time on Windows | Tint -> HLSL -> DXC (FXC on old drivers). Fragment-size table: < 20 KB ~0.1 s; 100-200 KB ~1.8 s DXC / ~5.5 s FXC; 1.6 MB never finished > 135 s [V, webgpu-tsl-compile skill, three r186/Chrome 154]. macOS hides it. | Keep every kernel small; wavefront classes compile as separate small modules; avoid string-generated variants multiplying (each distinct WGSL text compiles separately; `override` constants specialize at pipeline creation); `createComputePipelineAsync` everywhere; never first-compile on the load path. |
| Cache reuse is byte-exact | Chrome reuses compiled shaders only for byte-identical WGSL [V skill]. | Deterministic WGSL generation (no ids/capacities/ordering that vary per session); a stable hash of options -> source. |
| Robustness clamps | Spec allows driver robust access or inserting manual clamps on array indexing; on Metal Dawn uses its own bounds handling [V spec text/search]. I found **no published overhead number** for compute BVH walks. | Measure with Dawn's robustness disabled on a dev Chrome flag (toggle name unverified; Chrome `--enable-dawn-features=` accepts toggle names), on Windows and M-series. If >5%, consider fewer dynamic-index accesses (unrolled 4-wide node loads as `vec4` reads instead of scalar array walks). |
| fma contraction / non-IEEE on Metal | Watertight test leaked rays on M5 only: 4,513 per 2M -> 0 after `bvhEdgeFunction` [V bvh HANDOFF]. WGSL has no `precise`. | Keep the edge-function trick; add a "watertightness" test to the GPU gate for every new traversal variant (already `trace-oracle`). |
| Storage buffer limits | defaults: 8 storage buffers/stage, 128 MiB binding, 16 KiB workgroup memory, 4 bind groups [V partial from spec fetch + K]. | Bindless (Chrome flag) first with bind-group fallback; pack path SoA into few buffers; request adapter limits (bvh already does for binding size). |
| Dispatch overhead | 33-67 us/dispatch Chrome (no sync); Firefox ~1 ms [V]. Each `submit` is ~40% of cost on wgpu [V]. | One encoder, many dispatches, one submit per frame; indirect dispatch; avoid CPU readbacks (`mapAsync`) in the render loop. |
| Timestamps | Chrome quantizes `timestamp-query` unless dev features enabled [K]; bvh bench already works around with 20-dispatch fence timing. | Keep that method. |
| No forward-progress guarantee between workgroups | WGSL/WebGPU spec [K] | Use atomic counters for work fetch, never spin-wait on another workgroup (decoupled look-back scans are risky). |
| f16 | `shader-f16` is optional; Apple/NVIDIA/AMD/Intel Chrome mostly expose it [K, unverified per-driver]. | Capability-gated variants with f32 fallback; do not fork the algorithm. |
| 64-bit atomics | Not in core WGSL [K] | 32-bit queue counters; pack (key, index) in two u32 only for sorting via radix (we already have `gpu-radix-sort.ts`). |
| Hardware RT | Not exposed in WebGPU. Native wgpu has experimental ray-query [K, unverified status]; Chrome status unknown. | Software BVH is the design. If a ray-query tier ever appears, it replaces only `bvhTrace*`. |
| Implicit-gradient texture sample in non-uniform flow | forces FXC unroll [V skill] | Use `textureSampleLevel`/`textureLoad`; trace kernels do not need derivatives. |

Published WebGPU-vs-native ray rates: **none found** for like-for-like. Best evidence: our 104 Mrays/s on 3060L WebGPU
vs. native CUDA CWBVH at ~658 Mrays/s (Pascal Titan X, CrySponza bounce 1; 361 on Maxwell) (not same GPU, not same year); the OpenPBR WebGPU paper gives
only path-trace times (100 spp 512x512 1.07M tris: 1058.7 ms RTX 3080, 2319 ms M1 Max, "not optimized") [V]. So a
3080 does ~94 spp/s at 512x512; the M1 Max ~43 spp/s => the M1 Max is ~0.46x a 3080 in the same WebGPU code.

## 6. Findings: approximate tracers (software RT in production)

| System | Mechanism | Numbers | Lesson |
| --- | --- | --- | --- |
| UE5 Lumen SWRT [V slides] | per-mesh SDFs near field, global SDF merged for mid/far; screen traces first; surface cache lighting at hits; screen-space probes + radiance cache | PS5 1080p "Land of Nanite": Lumen total 2.77 ms (High, 1/16 rpp) / 4.63 ms (Epic, 1/4 rpp); GI tracing global SDF 0.94 ms vs HWRT 10.03 ms (heavy overlap); Lyra (clean): 1.21 vs 1.41 ms; Matrix Awakens HWRT+far-field 6.38 ms High, 11.31 ms Epic; Lake House HWRT 2080 Ti 7.25 ms Epic | "Merged" world representation beats per-instance traversal when instances overlap; SDF poor for mirror reflections, fine for GI and rough reflections; lose energy rather than leak on coverage gaps; HWRT needed for mirrors and skinned meshes |
| Godot SDFGI [V docs] | cascaded SDF voxels + probes + cone-like tracing | docs: "one of the most demanding GI" in Godot; no ms published | scales with cascade count and camera motion (cascade scrolling cost) |
| AMD Brixelizer GI [V GDC pdf + GPUOpen text] | cascades of 64^3 voxels, 8^3 R8 brick distance fields in an atlas, 3-level AABB tree per cascade, radiance cache bricks (4^3) with SH propagation | GPUOpen: faster than DXR "for the majority" on RX 7900 XTX, gap narrows in small scenes like Sponza; divergence hurts | A *sparse* grid + short tree gives most of SDF speed; update cost is the weak spot |
| Hybrid SS + WS | trace screen (Hi-Z, `downsampler`) first, fall back to world | Lumen does this [V]; `reflections` repo already planned | keep both in one ray API: `Trace(ray, flags)` tries tier 0..n |

When an approximate tracer wins for *our* consumers [E, reasoned from the table]:
- **DDGI/radiance-cascade/probe GI far field:** rays are low frequency and need only coarse hit position/normal and
  a cached radiance; a brick-atlas SDF or coarse voxel grid (a `WebVDB`-style sparse structure) costs ~constant per ray and
  scales to any triangle count. Use the triangle BVH for the first ~1-2 m (near field contact shadows, thin walls) and
  the SDF for the remainder - Lumen's near/far split [V].
- **Rough reflections (GGX roughness > ~0.5):** same argument; Lumen's reflection tracing cost 0.04-0.46 ms there [V].
- **Scenes with heavy instance overlap or millions of small instances:** a merged volume side-steps the TLAS cost.
- **Triangle BVH stays** for mirrors/glossy < 0.3 roughness, thin/foliage geometry, exact shadow rays, skinned meshes,
  and the reference path tracer. Never replace it in unbiased mode.
- Cost to add: voxelize/SDF-generate per static mesh, update on change (Brixelizer notes update cost dominates;
  Lumen builds mesh SDFs offline), and a radiance cache so that hits need no shading. Rating: **Prototype** as a
  separate tier (`TraceTier.Approx`), only after consumers (ddgi, reflections) ask; not in the first spec cut.

## 7. Recommended architecture with fallback tiers

```
Layer 0  @pmndrs/bvh        build/refit/TLAS, bvhTraceClosest/Any (megakernel inline)   [exists]
Layer 1  trace core         batch/inline trace entry points, hit record, ray queues,   [this repo]
                            args-gen kernel, subgroup/wg compaction, octant binning
Layer 2a utility pipeline   rays -> trace -> minimal shade (albedo/emissive/radiance-cache fetch)
                            as ONE kernel (inline walk). Consumers: DDGI, AO, shadows, SS fallback.
Layer 2b path tracer        wavefront: generate, extend(trace), class kernels, shadow queue, accumulate
Layer 3  approximate tier   (later) brick/SDF far field; hybrid screen-space + world (Hi-Z from downsampler)
```

Tier definitions (selected once at init from adapter features; never branch per ray):

| Tier | Requires | Traversal | Scheduling | Shading | Intended GPUs |
| --- | --- | --- | --- | --- | --- |
| A | `subgroups`, bindless flag, `shader-f16` | CWBVH8 or BVH4 (if the gate in 4.2 passes) with shared-memory short stack; octant-binned queues | wavefront; per-class indirect dispatch; subgroup compaction; block-wise ray bucketing; optional persistent fetch loop; visibility-buffer primaries | per-class kernels, bindless texture table | NVIDIA, AMD, Apple M, Intel Arc in Chrome |
| B | `subgroups` only | BVH2 f32 (today's walk) + shared-memory stack experiment | wavefront with subgroup compaction, bind groups with packed buffers | atlas/array textures | current Chrome on most desktop GPUs; Intel iGPU |
| C | no subgroups (Safari/Firefox today) | BVH2 f32, private stack | megakernel for utilities; wavefront with workgroup-level compaction only if the path tracer is requested; CPU-built trees (tinybvh WASM; bvh already falls back) | classic bind groups, fewer classes | Safari, Firefox, old/mobile GPUs |
| (approx) | any | SDF/voxel far field | n/a | radiance cache | mobile (Adreno/Mali) default for GI |

Design rules (priority order, matching BRIEF):
1. One BVH layout family, selected by `nodeFormat`; traversal functions are generated per format, shared hit contract.
2. Queue ABI fixed early: `Counts` struct (u32 per class + alive), `IndirectArgs` array, 32-byte ray record, SoA path
   state; every stage reads count -> `min(count, capacity)`; overflow counter reported to the CPU lazily (no readback
   in the loop).
3. Bounded recursion: a fixed bounce loop recorded on the CPU as dispatch sequences; early termination from the
   args kernel (count 0 -> next dispatches are no-ops with 0 workgroups, which cost ~the dispatch floor, 33-67 us).
4. Never tune by assumption; every trick gets a GPU-gated page + ratio-vs-baseline row in `bench/results` in the bvh
   style (paired rounds, clock rule).

## 8. Mrays/s targets per GPU class

Metric: **closest-hit, incoherent (random-direction / first diffuse bounce), Sponza-class scene (0.26M-5.7M tris),
large batch >= 1M rays, WebGPU in Chrome, steady state.** Shadow/any-hit rays are typically 1.1-2x faster; primary
rays 2-5x faster [V Ylitie tables (primary/bounce-1 ratio on Pascal CWBVH 1163/658 on CrySponza = 1.8x); V bvh any-hit 1.02x of closest on Sponza so use 1.0-1.5x].

| GPU class | Anchor (verified) | Baseline now | Target Tier B (BVH2 tuned) | Target Tier A (wide + wavefront) | Basis |
| --- | --- | --- | --- | --- | --- |
| Desktop high (RTX 4080/4090/5080, RX 7900) | none measured on our stack | est. 350-600 [E: 3-6x of 3060L by TFLOPS/bandwidth] | 400-700 | 600-1,000 | tinybvh 5 Grays/s (primary, RTX 5080 laptop) bounds the coherent case [V]; scale of CWBVH 2x [V] |
| Desktop mid / laptop dGPU (RTX 3060L, RX 6600, Arc A7xx) | **104 Mrays/s Sponza, 3060L** [V] | 104 | 120-150 | 150-220 | Ylitie Titan X Pascal bounce-1 360 Mrays/s native CWBVH [V] -> web at 40-60% of native (**correction: Pascal CrySponza bounce-1 CWBVH is 658 Mrays/s [V]; 360 is Maxwell; the 40-60%-of-native rule is untested**) |
| Apple M Pro/Max/M5 | M5 pathtracer ~120 Mrays/s wall (indicative) [V]; M1 Max = 0.46x RTX 3080 in a WebGPU PT [V] | 80-120 | 100-150 | 130-220 | no published software-RT Metal Mrays/s found |
| Apple M base / A-series | none | est. 30-60 [E] | 40-70 | 60-100 | scaled from M Pro by GPU core ratio [E] |
| Intel iGPU, Xe-LPG / Iris Xe | UHD: **12.8 Mrays/s** [V] | UHD 12.8; Xe-LPG est. 30-50 [E] | 15 UHD / 40-60 Xe | 20 UHD / 60-90 Xe | UHD is correctness-only per bvh rules |
| Mobile Adreno 7xx / Mali-G7xx | none found | unknown (est. 5-20 [E]) | n/a | 15-40 | no sources; start with SDF/probe tier, trace only <= 0.25 ray/px |

Budget conversion (60 fps, 16.7 ms, all rays in the trace stage only): 100 Mrays/s = 1.67M rays/frame = 0.8
rays/pixel at 1080p; 200 Mrays/s = 1.6 rays/pixel; 1000 Mrays/s = 8 rays/pixel. A DDGI update at 2,048 probes x 128
rays = 0.26M rays = 2.6 ms at 100 Mrays/s. Half-res reflections at 1 ray/px = 0.5M rays = 5 ms at 100 Mrays/s. A
progressive 1080p PT with 4 bounces + NEE (~8 rays/path) needs ~16.6M rays/spp = 166 ms/spp at 100 Mrays/s, i.e.
real-time interactivity needs denoising on 1 spp at reduced resolution or an upscaler (our `upscaler`) - consistent
with the Denoiser/upscaler defaults.

Caveat: every number marked [E] is an extrapolation with a factor-2 error bar. The only hard anchors are 104 Mrays/s
(3060L), 12.8 (UHD), ~120 (M5 PT example), and the native CUDA CWBVH table. Replace [E] with measurements from the
first benchmark pass.

## 9. For the spec

1. Two execution models, one BVH, one hit record: `bvhTraceClosest/Any` inline (utilities) and queue-based
   `TraceBatch` (path tracer). Publish the queue ABI (section 7 rule 2) as part of the layout version.
2. Wavefront path tracer uses 4-6 shader classes, 256K-1M-path batches, SoA path state <= 96 B (f16 where available).
3. Add `nodeFormat: 'cwbvh8' | 'bvh4'` as experiments behind the existing `GPU_LAYOUT_VERSION` mechanism; decision gate:
   >= 1.3x incoherent closest-hit on RTX 3060L *and* M5, else drop. Do not make quantized-BVH2 default (0.93-0.97x).
4. Per-adapter tuning table: workgroup size {64, 128}, stack in private vs workgroup memory, subgroup vs workgroup
   compaction; chosen at init, cached by adapter id, never per ray.
5. Subgroup-based compaction is Tier A/B; Tier C stays correct but slower; CPU-built trees remain the fallback
   (already done in bvh).
6. All WGSL generation deterministic and small; compile via async pipeline creation only; budget: no kernel > 50 KB
   WGSL (below the DXC cliff in the skill's table).
7. Ray coherence: octant binning mandatory; global sort optional, block-wise default, indirect (index) sort, never
   move ray payloads.
8. Visibility-buffer primary path as an adapter capability (engine rasterizes); core accepts pre-resolved primary hits.
9. Approximate tier is a *consumer-driven later milestone* (DDGI far field first); keep `bvhTrace` API tier-agnostic
   (`TraceFlags`: `NEAR_ONLY`, `ALLOW_APPROX`).
10. Watertightness and precision gates stay mandatory on every new traversal/triangle variant, on Metal specifically.

## 10. Bench / comparison ideas

- Incoherent closest-hit Mrays/s, Sponza (5.7M), Bistro, a 262k Sponza, Hairball-like dense scene: BVH2 (current) vs
  BVH4_GPU vs CWBVH8 vs BVH2 + shared-memory short stack; on RTX 3060L, M5, Intel UHD. Paired rounds, clock rule (as in
  `docs/performance/builds-and-tracing.mdx`).
- Workgroup size sweep {32, 64, 128, 256} x stack location, per GPU. Persistent-fetch loop vs plain dispatch.
- Ray order: unsorted vs octant bins vs block-wise Two-Point key vs device-wise radix; end-to-end ms including sort.
  Count active-lane fraction with `subgroupBallot` counters in a debug build.
- Wavefront vs megakernel crossover: sweep shading cost (trivial Lambert -> layered GGX + 8 textures) and bounce
  count; find where wavefront wins on each GPU; per-iteration breakdown like Laine's Table 2.
- Dispatch overhead: stages x bounces vs time at 256K/1M paths; Dawn indirect-validation cost (indirect vs direct
  dispatch same count).
- Robustness toggle on/off (Dawn feature) for the walk.
- Compile time: cold/warm per kernel on Windows with the webgpu-tsl-compile scripts; WGSL bytes per kernel; async
  vs sync creation.
- vs mutracer (WebGPU, CWBVH/SBVH from TinyBVH WASM, M4 Max): same scene and camera, Mpaths/s and ms/spp; vs
  three-gpu-pathtracer (old local copy); vs native: Aila-Laine/CWBVH CUDA (tinybvh GPU samples) and DXR/OptiX on the
  Windows machine.
- SDF/voxel tier vs BVH for DDGI probe rays: ms and error vs reference on Sponza and an overlap-heavy scene.

## 11. Needs Windows / native hardware

- RTX 3060L (and a desktop NVIDIA/AMD card) numbers for any new traversal variant; Intel UHD correctness.
- D3D12 + DXC/FXC compile-time measurements of every kernel; Dawn robustness toggle on D3D12.
- Native comparisons: tinybvh GPU samples (OpenCL/CUDA, CWBVH traversal), OptiX and DXR inline on the same scene and
  rays to quantify the hardware-RT gap and the WebGPU/native software gap (the 3.5x headroom estimate above).
- SER on/off in a DXR build to bound the ceiling of our software analogues (NVIDIA Ada+ or Intel Arc B-series).
- AMD (wave32/64, RDNA3/4) and Qualcomm/Adreno, Mali hardware for subgroup size and workgroup memory behavior;
  none are in the current target list.
- Apple: M-series measurements beyond the M5 (M1-M4 base) for the Tier B/A floor.

## 12. Open questions

- Does CWBVH win under WGSL on Metal and D3D12 (ALU for byte tricks emulation, shared-memory pressure, no
  `__byte_perm`)? The only evidence of it running on WebGPU is mutracer, unmeasured.
- Is the private `array<u32,64>` stack actually spilling on M5/RTX 3060L? A shared-memory/short-stack experiment will
  show it; no published WebGPU data.
- Robustness-clamp overhead for BVH walks: unmeasured anywhere I could find.
- Chrome status of bindless, `subgroup_size_control`, ray query, and `shader-f16` on each OS: needs a feature probe on
  the Windows box (I verified only subgroup size docs in Tint).
- Does indirect-dispatch validation add a hidden pass per dispatch in Chrome on D3D12/Metal (the 3 ms -> 10 us result I
  found was for indirect *draws*)?
- How much of the M5's ~120 Mrays/s is trace vs shade in the `three/pathtracer` example (wall time of one dispatch)?
- Moving the stack off private memory: workgroup-memory pressure on Apple may reduce threadgroup occupancy more than
  it saves.
- Whether sorting by an index key (Meister's CUDA variant) beats physical reordering in WGSL given no `__ldg`-style
  texture-path loads: needs the ray-queue benchmark above.
- Treelet-queue (out-of-core) traversal is only relevant above VRAM; decide the max scene size we promise.
