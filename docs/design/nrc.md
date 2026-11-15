# Neural radiance cache on WebGPU (research track N)

Status: **draft spec + plan, 2026-11-14** (revised the same day around VkNRC, §1b). Nothing here is decided. It turns the "NRC: Prototype at most, needs
coop-matrix" line in [spec §7](../../spec/SPEC-draft.md) into a measurable track. The track has two products:

1. **A small MLP engine in WGSL** (inference + online training) with backends we can A/B: plain f32, plain f16, and
   `subgroup-matrix`. ORT-Web and WebNN are measured as extra arms.
2. **An NRC demo** following [VkNRC](https://github.com/AdamYuan/VkNRC) (a complete Vulkan NRC, §1b): a diffuse path tracer on sponza-white that ends paths in the learned cache, compared against the
   4096 spp references and against plain path tracing at equal time.

The demo is also meant to be evidence: **does `subgroup-matrix` make a learned cache practical on the web, and on which
GPUs?** That answer has to come from paired timings, not from vendor peak numbers.

Companions: [nrc-diverges.md](nrc-diverges.md) (every difference from the paper and VkNRC, with its follow-up) and
[subgroup-matrix-feedback.md](subgroup-matrix-feedback.md) (feature status as of 2026-11-14 and what we would ask the
WebGPU group for).

Markers as in the research notes: **[V]** checked against a source this session, **[M]** from memory, check against the
PDF before quoting, **[U]** unknown, needs a test.

## 1. What NRC is (paper, Müller et al. 2021, arXiv 2106.12372)

Sources we may use: the paper and talks (D10: NVIDIA papers are fine), and **tiny-cuda-nn** (NVlabs, **BSD-3-Clause
[V 2026-11-14, LICENSE.txt]**: may be ported with attribution in THIRD_PARTY.md). Off limits: the NRC SDK and RTXGI code
(NVIDIA RTX SDK licence, D10), including `research/sources/native/rtxgi`. instant-ngp is under a non-commercial NVIDIA
licence: ideas only. VkNRC has no licence: we follow its design and write our own code (§1b, D28).

How it works [M unless marked]:

- A small MLP maps a path vertex to outgoing radiance. Inputs: position, scattered direction, shading normal,
  roughness, diffuse reflectance, specular reflectance (about 14 floats). The paper encodes position with a frequency
  encoding and the rest with one-blob; later NVIDIA work (instant-ngp) and the NRC SDK use a multiresolution hash
  grid. The output is radiance divided by reflectance, multiplied back at the query, so texture detail does not have
  to be learned.
- Network: about 64 neurons wide, about 5 hidden layers, ReLU, no bias; f16 activations, f32 master weights. "Fully
  fused": one kernel runs all layers while activations stay on chip.
- **Render paths** trace 1–3 bounces and stop when the path's footprint spread is large compared with the primary
  footprint (the paper's area-spread heuristic, threshold about 0.01 of the primary). The last vertex queries the
  cache.
- **Training paths**: a sparse set of pixels (a few percent, one random pixel per screen tile) trace longer paths. Each
  vertex becomes a training record whose target is the radiance gathered from there on. The tail of each training path
  is the cache's own prediction ("self-training"), which carries light over many bounces at the cost of a short trace.
- Training runs **every frame, online**: about 4 Adam steps of 16k records (about 65k records per frame). Loss: relative
  L2 (squared error divided by the squared prediction, prediction treated as a constant). Inference uses an
  exponential moving average of the weights.
- Paper cost: about 2–3 ms per 1080p frame for training plus inference on an RTX 3090, with tensor cores.

Read the PDF before quoting any of these numbers (N0.1). The research notes flagged the same gap
([07](../../research/notes/07-denoise-neural.md) "exact NRC cost numbers ... check PDF").

## 1b. Reference implementation: VkNRC (the one we port the design of)

[AdamYuan/VkNRC](https://github.com/AdamYuan/VkNRC), commit `a667859` (2024-09-10), cloned read-only at
`research/sources/native/vknrc` (the native build lives in `.cache/native/vknrc`, §9). A complete NRC in Vulkan: hardware ray queries, a V-buffer for primaries, a
Cook-Torrance path tracer, and a hand-written fully-fused MLP on `VK_NV_cooperative_matrix` (its README says the KHR
extension was "too limited", and that its backward pass beats tiny-cuda-nn's). Its history shows the path we plan to
take: an MLP test harness with a CPU (Eigen) oracle and a "learning an image" test came first, then forward,
backward and Adam, then the integrator.

**Licence: none.** No LICENSE file and no licence text anywhere in the tree [V 2026-11-14]; the README only describes
the project. The maintainer's call (D27, D28), since this track is an experiment to find out whether a web NRC is
possible at all:
- **We follow its design closely**: kernel layout, record formats, pass graph, constants and the order it was built
  and tested in. VkNRC is credited as the design reference.
- **The code is ours.** No VkNRC source text is pasted or translated line by line into this repo. In practice that
  costs little: GLSL with NV cooperative matrices becomes WGSL with subgroup-matrix tiles, no per-element tile access
  and no float atomics, so the kernels are a rewrite either way (table below).
- **We build and run it natively** as the baseline (§9).

What VkNRC does, all [V from its source]. These are its choices, not necessarily the paper's:

| Part | VkNRC |
|---|---|
| Net | 64 inputs → 5 hidden × 64 → 3 outputs, ReLU, no bias, f16 weights and activations; 20,672 weights |
| Input encoding (64 f16) | position normalised to the scene box [−1, 1] (done at load), then a triangle-wave frequency encoding with 12 octaves per axis (36); one-blob with 4 bins (quartic kernel) for the scattered direction (2, spherical), normal (2, spherical) and roughness (1, as `1 − e^−r`) (20); diffuse + specular reflectance (6); two constant ones (2) |
| Output | radiance directly. The resolve is `pixel = bias + throughput × prediction`; it does **not** divide by reflectance as the paper does |
| Kernel shape | 128 records per workgroup, 4 subgroups of 32 (full subgroups required, subgroup size 64 dropped), 16×16×16 f16 tiles; one layer's weights in shared memory at a time (8 KiB); activations 128 × 64 f16 = 16 KiB shared; backward needs 4 × 8 KiB = 32 KiB shared |
| ReLU | applied element by element to the accumulator tiles (the NV extension allows per-element access) |
| ReLU derivative | the forward activations are stored as a mask of 0 or NaN, the backward matmul accumulates onto it, then NaN → 0 |
| Weight gradients | per subgroup dW tiles, summed across the workgroup's subgroups in shared memory, then `atomicAdd` on **f32** into a global gradient buffer |
| Loss | relative L2 on luminance: gradient `2 (p − t) / (lum(p)² + 0.01)`, loss scale 1 |
| Optimiser | Adam, lr 0.002, f32 master weights, gradient divided by the batch count, NaN/inf gradients zeroed; EMA weights optional for inference |
| Training records | 4 batches of 16,384 per frame, trained in sequence; a training path is chosen per **subgroup** (probability 0.03) so control flow stays uniform; each record goes to a random batch |
| Self-training | the inference pass runs before training and writes predictions into the training records' tails as well as into the screen |
| Termination | area-spread heuristic, `c = 0.01`, `a0 = d² / (4π cos)`, up to 8 bounces |
| Dispatch | GPU counts with atomics, a one-thread pass turns them into indirect dispatch arguments |

### What changes on WebGPU

| VkNRC relies on | WebGPU today | Our plan |
|---|---|---|
| Hardware ray queries | none | trace core (`rtClosest`, software BVH) |
| `VK_NV_cooperative_matrix`, 16×16×16 f16 | `chromium-experimental-subgroup-matrix`; tile shape per adapter (Apple 8×8×8) | generate the kernel per `subgroupMatrixConfigs` entry; plain f16 fallback |
| Per-element access to accumulator tiles (ReLU) | not in the proposal: load, store, multiply(-accumulate) and scalar add/sub/mul only | store the tile to workgroup memory, apply ReLU per lane, reload. This is the main extra cost; N3 measures it |
| NaN as a ReLU-derivative mask | WGSL does not guarantee NaN/inf behaviour | an explicit 0/1 mask, multiplied in the same workgroup-memory pass |
| f32 `atomicAdd` | no float atomics | A/B: per-workgroup partial gradients + reduction dispatch, vs fixed-point `atomicAdd` on i32 (the SHaRC trick) |
| Subgroup size fixed at 32, full subgroups | `subgroup-size-control` / `@subgroup_size` (Chrome 151-152) | pin 32 where offered; otherwise derive tiles per subgroup from the adapter's size |
| 32 KiB shared memory for backward | default 16 KiB | request the adapter maximum; if 16 KiB only, 64 records per workgroup |

So the port is mostly mechanical except the element-wise steps. Whether subgroup-matrix wins depends on how much the
workgroup-memory round trips per layer cost compared with the matmuls they surround. That is the number N3 exists
to produce.

## 2. Why this is hard on WebGPU (and the budget it implies)

**Arithmetic.** One forward pass of a 64-wide, 5-hidden-layer net is about 2 · 64 · 64 · 6 ≈ 50k FLOP. One query
per pixel at 1080p is 2.07M queries, about **100 GFLOP per frame for inference alone**. Training adds about
65k records × 3 forward-equivalents (forward, input gradient, weight gradient), about 10 GFLOP. Rough cost at a sustained 25% of peak:

| Device | Peak f16 FMA (approx.) | Inference at 1080p | Inference at 960×540 (upscaler ratio 2) |
|---|---|---|---|
| RTX 3060 Laptop, plain f16 | ~13 TFLOP/s [M] | ~30 ms | ~8 ms |
| RTX 3060 Laptop, tensor cores | ~50+ TFLOP/s dense [M] | ~8 ms | ~2 ms |
| M5 Pro, portable f16 (naive kernel, quiet, 2026-11-15) | 0.44 TFLOP/s achieved | ~195 ms | ~49 ms |
| M5 Pro, `wgsl-sgm` f16, 4 subgroups (quiet, 2026-11-15) | 4.9 TFLOP/s achieved | ~17 ms | ~4.4 ms |

So **without matrix hardware, a full-resolution NRC does not fit a real-time frame; at half resolution it might.**
This is the case for `subgroup-matrix`, and it is also why the demo should run at the upscaler's input resolution.

**Platform constraints** (all affect the kernel design):

| Constraint | Consequence |
|---|---|
| Default `maxComputeWorkgroupStorageSize` is 16 KiB | A 64×64 f16 layer is 8 KiB; a 128-record × 64 f16 activation tile is 16 KiB. Request the adapter maximum (`detectCaps`), keep one layer's weights resident at a time, size the record tile to fit. |
| WGSL has no float atomics | Weight gradients cannot be `atomicAdd`ed in f32 across workgroups (VkNRC does). Either each workgroup writes a partial gradient and a separate reduction dispatch sums them (no "last workgroup" handoff, CLAUDE.md trap), or gradients are scaled to fixed point and `atomicAdd`ed as i32. Both are built and A/B'd. |
| `subgroup-matrix` has no element-wise ops besides scalar add/sub/mul [V proposal] | ReLU and its derivative need a store to workgroup memory, a per-lane pass, and a reload between layers. That is the fully-fused pattern anyway, but it costs bandwidth the scalar path does not pay. |
| `subgroup-matrix` ops need subgroup-uniform control flow and operands; workgroup x size must be a multiple of the max subgroup size [V proposal] | One subgroup owns whole tiles; record counts are padded to the tile size; no early-out per lane. Several subgroups per workgroup failed for the Denoiser (`li / 32` offsets); `@builtin(subgroup_id)` + `subgroup_uniformity` (Chrome 144/145) may fix it: probe in N0.2. |
| Tile shapes are per device (`GPUAdapterInfo.subgroupMatrixConfigs`) [V proposal] | Metal: 8×8×8 (f16, f32). Vulkan on NVIDIA/AMD/Intel: typically 16×16×16 f16 [M]. D3D12 needs SM 6.10 linear algebra, not generally available [V note 06]. The kernel is generated per config. |
| Which backend reaches the matrix units [U] | Metal `simdgroup_matrix` on Apple GPUs before M5 runs on the normal ALUs (the gain is register reuse, not more FLOP/s). M5 adds per-core neural accelerators behind Metal 4 tensor APIs; whether Dawn uses them is unknown. On Windows, Chrome runs Dawn on D3D12, which has no subgroup matrix; tensor cores are reachable only through the Vulkan backend, if Chrome lets us select it. |
| No readback in the frame loop | Query and training-record counts live on the GPU (atomic append, as in the queue ABI); dispatches are indirect. |

**Existing evidence to respect.** The Denoiser's hand-written WGSL beat ORT-Web by about 3× on its U-Net, WebNN
won only on the larger nets (via CoreML), and its subgroup-matrix conv kernel **helped f32 only** on M5 Pro
([01 §3.3](../../research/notes/01-local-ecosystem.md)). So "subgroup-matrix is a big win" is a hypothesis, most
likely to hold on NVIDIA/AMD through Vulkan, and it may not hold on Apple. The plan below is built to find out.

### 2b. What the Denoiser measured on the M5 Pro (github.com/pmndrs/denoiser, `docs/specs/runtimes.md`)

- Peak rates in Chrome: **f16 FMA ~6.6 TFMA/s, subgroup-matrix MMA ~3.8 TMAC/s (f32 8×8×8), f32 FMA ~3.1 TFMA/s.**
  On Apple, plain f16 math outruns the matrix path.
- f16 subgroup matrices exist on Apple only as f16 × f16 → f16 (f16 accumulation), at about the f32 matrix rate.
  VkNRC accumulates in f16 too, so that config is usable for the MLP; it is just not faster there.
- More than one subgroup per workgroup on the matrix path fails WGSL uniformity analysis: matrix load/store offsets
  must be workgroup-uniform and `subgroupBroadcastFirst` does not count. So one subgroup (32 threads) per workgroup,
  and the 128-record, 4-subgroup tile VkNRC uses does not port as is.
- Its subgroup-matrix conv went 57 → 44–50 ms on fp32; for fp16 the portable kernel stayed faster.

**Expectation, before measuring:** on the Mac, `wgsl-f16` is likely to match or beat `wgsl-sgm`; subgroup-matrix should
pay off where it maps to tensor cores (NVIDIA/AMD via Vulkan). The Windows runs decide the demo's headline; the Mac
runs decide the portable default.

## 3. Backends: hand-written WGSL first, ORT and WebNN as A/B arms

The question "ONNX like the Denoiser, or hand-written WGSL?" comes down to one fact: **NRC trains online, every
frame.** Neither ORT-Web nor WebNN trains on the GPU in the browser [U for ORT: its training build was WASM-only].
Both can still run a training step if we export the whole step (forward, backward, Adam update) as an ordinary
inference graph with the weights as inputs and outputs. That keeps them in the A/B, but they pay graph dispatch
overhead and buffer hand-offs every step, on a net of about 20k parameters where overhead dominates.

| Arm | Inference | Training | Runs where | Expectation |
|---|---|---|---|---|
| `wgsl-f32` | fused WGSL, f32 | fused WGSL | everywhere | correctness baseline, slowest |
| `wgsl-f16` | fused WGSL, f16 activations, f32 accumulate where needed | same, f32 master weights | `shader-f16` | the default if matrix ops don't win |
| `wgsl-sgm` | `subgroup_matrix` f16, tile shape from the adapter's config | same | `chromium-experimental-subgroup-matrix` | the demo; win size per vendor unknown |
| `ort-webgpu` | ORT-Web WebGPU EP, I/O binding on GPU buffers | training-step-as-graph | Chrome | comparison only |
| `webnn` | WebNN graph, `MLTensor` export to WebGPU | training-step-as-graph | Chrome behind a flag | comparison only; WebNN is not shipped ([06](../../research/notes/06-web-platform.md)) |

**Suggestion:** hand-write the WGSL (all three precision arms from one generator), and build the ORT and WebNN arms only
for inference plus one training step, as microbench rows. Spec §0 lists "WebNN in the frame loop" as a v1 non-goal;
this track does not change that. It measures WebNN, it does not ship on it.

- Pro: online training needs custom kernels anyway; one generator gives all three WGSL arms; no runtime dependency.
- Con: we own the backward pass and Adam (small: four matmul shapes plus element-wise ops), and a gradient-check oracle.
- Settles it: the N2 microbench table (with the N5 rows). If ORT or WebNN beats `wgsl-f16` on the full train + infer step at equal output,
  revisit.

**Follow the Denoiser's OIDN path.** The Denoiser (`~/Developer/Denoiser`, MIT, ours) went the same way: ORT-Web first,
then a hand-written fused WGSL U-Net that became the default, with WebNN and ORT kept as selectable runtimes behind
one `NetworkRuntime` interface (`wgsl` / `webnn` / `ort`, plus `AutoRuntime`). Copy that shape here: one MLP
interface, the three WGSL arms plus `ort` and `webnn` as runtimes, an auto pick by measured speed. Reuse what it
learned, read on the Mac before N1 (N0.4):
- its parity method (dB against the native fp32 network; fp16 end-to-end measured 53 dB vs fp32) as our tolerance
  model for f16 vs f32 arms;
- its f16 partial sums and single-encoder fusion;
- its subgroup-matrix conv kernel and why it helped only f32 on M5 Pro, before writing `wgsl-sgm`;
- its ORT trap: wrong results on WebGPU Conv with more than 3 input channels (microsoft/onnxruntime#29651). An MLP
  uses MatMul/Gemm rather than Conv, but check our ORT arm against the oracle first.

Since the Denoiser is MIT and ours, its WGSL helpers can be ported directly, with the usual attribution in
THIRD_PARTY.md and the source file named in a comment. If both end up sharing an MLP/conv engine, that is the
"second consumer" in §8.

## 4. Correctness first

- **CPU oracle** (TypeScript, f64): the same net, encodings, loss and Adam. Forward matches within f16 tolerance;
  backward matches the oracle's gradients and central finite differences; one Adam step matches bit for bit in f32
  given the same gradients.
- **Learning test (image fit).** The net learns a fixed HDR image from (u, v) (the classic tiny-cuda-nn demo). Every
  arm trains from the same seed for N steps, then writes the fitted image as PFM. The A/B is fitted image vs target
  (relMSE, FLIP), the loss curve, and ms per step. This is the experiment that produces viewer images before any path
  tracing exists, and it is a good public demo on its own.
- **Determinism.** Fixed seeds and a fixed record order for tests; float atomics never appear (§2), so a run is
  reproducible on one device.

**Match VkNRC first.** Every arm starts with VkNRC's configuration (§1b): net, encoding, loss, Adam settings,
4 × 16,384 records, 0.03 training probability, `c = 0.01`. Then we change one thing at a time, each change recorded in
DIVERGES.md: reflectance factorisation as in the paper, hash-grid encoding, EMA weights, fewer records.

## 5. The NRC demo integrator

**Update 2026-11-15:** main now has the path tracer with cache hooks (`ptOnVertex` / `ptTerminate`, `updateBlock`
training paths, both schedulers). The demo plugs into those instead of carrying its own integrator; the wiring is in
[nrc-plan.md](nrc-plan.md) §5 (TASKS N6a-N6c). The original plan, kept for the pass structure:

- Scene: **sponza-white** (grey diffuse, sun + sky, 262k triangles). Diffuse only means reflectance is a constant and
  the "specular" inputs are zero, so the first demo isolates the cache. References at 4096 spp exist (pbrt + Mitsuba).
- Wavefront stages, all indirect, no readback:
  1. primary + up to 3 bounces with NEE to the sun; stop on the spread heuristic; append a query record
     (position, direction, normal, roughness, reflectance, pixel, throughput) for the last vertex;
  2. training paths on one random pixel per tile (ratio a setting), up to `maxDepth`, tail queried with the EMA net;
     append one training record per vertex with its target radiance;
  3. `K` Adam steps over the training records (batch 16k, indirect count);
  4. inference over the query records; resolve adds `throughput × reflectance × prediction` into the pixel.
- Settings for the A/B: encoding (frequency + one-blob vs hash grid), net width/depth, training ratio, steps per frame,
  termination threshold, MLP backend.
- Uses the cache hooks the spec already reserves (`onVertex`, `terminate`, D3). Where the code lives long-term is
  open (§8).

**Metrics.** At equal frame time: relMSE and FLIP vs Mitsuba (D15), mean bias (NRC is biased; report it), and temporal
stability under a camera path (frame-to-frame difference on a still camera, as in the upscaler's convergence meter).
Rows: plain PT at the same ms; NRC; NRC per backend. Later, a non-neural hash cache (SHaRC-like, from `@pmndrs/gi`) as
the honest competitor: research note 08 rated it "gets 80% of NRC's benefit"; this is where we find out.

## 6. Plan

Rows are in [TASKS.md](../../TASKS.md) section N. Each stage ends with a bench output, not a claim.

| Stage | Output | Gate to continue |
|---|---|---|
| **N0 Groundwork** | NRC paper numbers checked (N0.1); `detectCaps` reads `subgroupMatrixConfigs` and a compile probe for the feature (folds in P0.18); which Dawn backend reaches tensor cores on Windows/NVIDIA and whether Metal reaches M5's neural accelerators (N0.3) | We know which machines can run `wgsl-sgm` at all |
| **N1 MLP engine + oracle** | WGSL generator for forward/backward/Adam (f32, f16), CPU oracle, `mlp-fit` page writing PFMs | Gradient check passes; image fit converges like the oracle |
| **N2 Microbench** | `mlp-timing` page: ms per 1M inferences and per 16k-record training step for every arm, net 64×{2,4,5} hidden, batch 2^14–2^21, ABBA, load-checked | A table of paired ratios on M5 Pro and RTX 3060L |
| **N3 Subgroup-matrix kernels** | `wgsl-sgm` forward, then backward (dW is also a matmul); tile shape from config | Matches `wgsl-f16` output within f16 tolerance; timing row added to N2 |
| **N4 Encodings** | frequency + one-blob and multiresolution hash grid, both on `mlp-fit` | A/B images + ms; both kept as settings |
| **N5 ORT / WebNN arms** | inference + training-step-as-graph rows in the N2 table | Comparison only (§3) |
| **N6 NRC demo** | `nrc` page on sponza-white, all stages of §5, settings exposed | Equal-time relMSE table vs plain PT, bias and stability reported |
| **N7 Hash-cache competitor** | non-neural cache at equal time | Shows whether the MLP earns its cost |
| **N8 Write-up** | `bench/results/NRC.md`: tables, viewer report, "what subgroup-matrix buys per vendor" | Maintainer decides what goes public (D8) |

Order matters: N2 can kill the track cheaply. If `wgsl-sgm` gives nothing on any device we can run, the demo still
works on `wgsl-f16` at the upscaler's resolution, and the write-up says so.

### Status and how to run (2026-11-15)

Built (`packages/mlp`, `@pathtracer/mlp`, private, WGSL prefix `nn`):
- **Portable arms** `wgsl-f32` / `wgsl-f16`: inference, forward/backward, deterministic weight-gradient pass, Adam/EMA,
  GPU-side counts and indirect dispatch (V101). f32 verified against the oracle on SwiftShader.
- **Subgroup-matrix inference** `wgsl-sgm` (N3a, `wgsl-sgm.ts`): tiles from the adapter's config, weights read as a
  transposed column-major view, ReLU/convert round trip through workgroup memory, 1 or N subgroups per workgroup
  (rows split by `subgroup_id`). The same generator emits an **emulated** form in plain WGSL; that form matches the
  oracle on SwiftShader for 8³ and 16³ tiles, 1/2/4 subgroups and ragged counts (inference 3–4e-7 rel). The native
  form compiled first time on the M5 Pro and matches the oracle (below). Training on subgroup matrices is N3b, not started.
- **Pages:** `mlp-check` (GPU vs oracle, `infer=portable|sgm|sgm-emu`), `mlp-fit` (learn an HDR image, PFM snapshots,
  `oracleSteps` replay), `mlp-timing` (GPU ms per inference pass and training step for one arm), `mlp-sgm-probe`
  (configs + which kernel forms compile), `mlp-live` (interactive demo, below).

**First Mac session (2026-11-15, M5 Pro, Chrome 157, Metal).** Every checklist step ran.
- *Probe (N0.2):* configs f32→f32 and f16→f16, 8×8×8; subgroup size fixed at 32. `subgroup-id ×4` compiles with no
  diagnostic override; the `local_invocation_index / 32` control fails and compiles only with
  `diagnostic(off, chromium.subgroup_matrix_uniformity)`.
- *Correctness (N1, N3a):* portable f32 2.5e-7 / f16 1.6e-3 (inference), gradients 3.1e-7 / 1.3e-2; native sgm f32→f32
  3.5e-7, f16→f16 2.8e-3, 1 and 4 subgroups identical. The emulated twin hung the Apple GPU (and the Mac rebooted during
  its 16³ run); rewritten so the 32 lanes share each tile, verified on SwiftShader, not yet re-run on the GPU (N3c).
- *Fit (N1.1):* deterministic, but VkNRC's lr 0.002 gives relMSE 0.84-0.98 after 1024 steps; lr 0.01 gives 0.26. The
  chirp stripes stay unlearned at any lr (encoding follow-up N1.3).
- *Timing (N2, best effort, not a quiet pass):* ms per 1M queries: portable f32 179-183, portable f16 95.4, sgm f32 33.3,
  sgm f16 12.6, sgm f16 with 4 subgroups 8.46. Training step (portable, 16k): f32 8.05, f16 6.78. **On the M5 Pro the
  matrix path beats our portable f16 by 7.5× (11× with 4 subgroups)**, against the expectation in §2b. The caveat is
  that the portable kernel is a naive baseline at 0.45 TFLOP/s (N1.2), so the fair comparison is still open.
- *Live demo (N10):* works in the app's browser pane (portable backends) and in flagged Chrome (all backends).

**Session 2 (2026-11-15).** Main merged (track N IDs renumbered: D27/D28, V101-V104, W11).
- *N3c:* the rewritten emulated form runs on the Apple GPU in 0.5-0.7 s and matches (8³/16³, 1 and 4 subgroups).
- *N2 quiet table* (load 2.5-3.2, no row loaded), ms per 1M queries: portable f32 117-120, portable f16 93.7-94.4,
  sgm f32 32.1-32.3, sgm f16 12.3-12.4, sgm f16 ×4 subgroups 8.4. Training step f32 8.2, f16 6.9. Ratios: matrix path
  11.2× portable f16; 4 subgroups 1.47× one. The loaded f32 number (180) was load, not the kernel.
- *N1.3:* two-phase triangle waves (sin/cos style) and octaves at Nyquist: relMSE 0.26 → 0.16 at lr 0.01, and the
  coarse chirp stripes are learned. Image-task defaults are now 8×2 at lr 0.01; NRC keeps VkNRC's encoding until N6.
- *N1.2:* tiled split-K weight-gradient pass (default): gradient pass 4.4 → 0.20 ms, training step 6.9 → 2.7 ms (f16),
  8.1 → 3.9 ms (f32). Forward-backward is now 92% of a step.
- *N3:* the ReLU round trip through workgroup memory is 20-24% of matrix-path inference with 1 subgroup, 12-13% with 4.
- Denoiser told (pmndrs/denoiser#31) that `subgroup_id` offsets lift its one-subgroup limit.

**Mac session checklist** (each line is one command from the repo root; quiet-machine rules for the timing lines):

```sh
pnpm install && pnpm test                                                   # 22 mlp tests + the rest
node bench/runner/run.ts --page mlp-sgm-probe --json probe.json              # N0.2: configs, which forms compile
node bench/runner/run.ts --page mlp-check --ab precision=f32,f16             # portable arms vs oracle
node bench/runner/run.ts --page mlp-check --args infer=sgm                   # native sgm vs oracle (first compile!)
node bench/runner/run.ts --page mlp-check --args infer=sgm subgroups=4       # only if the probe says subgroup-id x4 compiles
node bench/runner/run.ts --page mlp-fit --args oracleSteps=8 batch=1024 precision=f16
node bench/runner/run.ts --page mlp-fit --ab precision=f32,f16 --repeat 2    # 256², batch 16384, 1024 steps; viewer A/B
node bench/runner/run.ts --page mlp-timing --ab infer=portable,sgm --args precision=f16 --repeat 3
node bench/runner/run.ts --page mlp-timing --ab precision=f32,f16 --repeat 3
pnpm --filter @pathtracer/bench mlp:live                                     # live demo in Chrome (see below)
```

**Live demo** (`bench/pages/mlp-live.html`): the network learns the HDR image while you watch, target left, network
right; backend picker (portable f32/f16, subgroup matrix when Chrome exposes it, emulated), steps per frame, batch,
learning rate, EMA, image/relative-error view, exposure, Reset, and **Benchmark** (pauses, runs a timestamped burst,
reads back once, shows ms per step, ms per million queries and TFLOP/s). Open in a Chrome started with
`--enable-unsafe-webgpu` to get the subgroup-matrix backend. `?offscreen=1` renders to a texture and exposes
`window.__mlpLive.snapshot()` for headless checks; that is how it was verified here, because WebGPU canvas
presentation crashes headless SwiftShader in cloud containers.

Without a GPU (correctness only): `PT_CHROME_FLAGS="--no-sandbox --use-webgpu-adapter=swiftshader
--use-angle=swiftshader --enable-unsafe-swiftshader --enable-unsafe-webgpu"` and `--allow-software`, small sizes.

## 7. Open choices (each one settled by an A/B, not here)

1. Encoding: frequency + one-blob (paper) vs hash grid (later NVIDIA work). Hash grid adds parameters and a
   scattered-gradient write (more partial-sum traffic, §2).
2. Activation storage between layers in the `sgm` path: f16 in workgroup memory vs f32. Bandwidth vs accuracy.
3. Weight gradient reduction: per-workgroup partials + one reduction pass vs a two-level tree. Depends on batch size.
4. Inference at full vs upscaler input resolution (ratio 2). Probably ratio 2 for the demo; measure both.
5. Per-frame training steps `K` and training ratio: the paper's values first, then a sweep.

## 8. Questions for the maintainer

- **Where does the code live?** Suggestion: the MLP engine as a private workspace package `packages/mlp` (WGSL
  prefix `nn*`), the demo integrator as bench pages, until N6. A production cache would follow D3 into `@pmndrs/gi`.
  - Pro: keeps `@pmndrs/raytracer`/`pathtracer` free of neural code; the MLP engine is reusable by the Denoiser.
  - Con: a new prefix and a new package to maintain.
  - Settles it: whether a second consumer (Denoiser, gi) wants the MLP engine after N2.
- **Public evidence for gpuweb.** If N2/N3 show a clear win, the numbers fit gpuweb issue #4195 (subgroup matrix). That is
  outward-facing; nothing gets posted without your go (D8).
- ~~VkNRC licence / native build~~: settled 2026-11-14 (D27, D28): build it natively, follow its design, own code.
- **Windows Vulkan backend.** If tensor cores are only reachable through Dawn's Vulkan backend behind a flag, is a
  flag-only result acceptable for the demo? Suggestion: yes, reported as such.

## 9. Native baseline: VkNRC on the RTX 3060 (N6.1, D27)

What runs where: VkNRC needs `VK_NV_cooperative_matrix`, `VK_EXT_shader_atomic_float` and ray queries, so it only runs
on NVIDIA. The RTX 3060 laptop is the machine (Windows W11). It does **not** run on the Mac.

**Build.** Verified 2026-11-14 on Linux (Ubuntu 24.04, GCC 13.3, CMake 3.28, Ninja, glslc/shaderc 2023.8): a fresh
clone of `a667859` under `.cache/native/vknrc`, `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release`, then
`cmake --build build` builds all 13 shaders and the executable in about 50 s once the packages in the CLAUDE.md trap are
installed. Not run (no GPU in that session). On Windows follow its own CI (`.github/workflows/windows-msvc.yml`):
Vulkan SDK, Visual Studio 2022, x64 Release.

**Run.** `VkNRC <scene.obj>`. It loads one OBJ, normalises it to [-1, 1], has a constant white sky (radiance 10) and no
sun, and offers a split screen (left/right: none, NRC, cache only) with an accumulate toggle. The only number it shows is
FPS.

**Local patches needed for a baseline** (kept in `.cache`, never committed, D27):
1. GPU timestamps around each pass (path tracer, inference, each training step, optimiser), written to CSV, so we
   compare pass by pass rather than FPS.
2. A fixed camera pose and frame count from the command line, and an exit after N frames.
3. A dump of the accumulated image as PFM, so the viewer compares it with our `nrc` page and the references.

**Scene.** Our scenes are not OBJ. Add an OBJ exporter for sponza-white (positions and normals, one grey material), and
note that VkNRC's lighting (white sky, no sun, Cook-Torrance) differs from the sponza-white reference (sun + sky,
Lambert). For a like-for-like comparison our `nrc` page gets a "VkNRC lighting" setting; the references stay ours.

**What we compare.** Same machine, same resolution, same net and batch sizes:
ms per pass (VkNRC native vs our `wgsl-f16` and `wgsl-sgm`), and the images after the same number of frames. The
paired ratio "web / native" per pass is the headline of the track's write-up (N8).
