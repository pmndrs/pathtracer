# Subgroup matrix: where it stands, and what a learned cache needs from it

Checked 2026-11-14 against gpuweb `main` (`proposals/subgroup-matrix.md`, Status: Draft), the open gpuweb issues,
Chrome's "What's New in WebGPU" posts and Chrome 153 in a headless container. Purpose: build research track N
([nrc.md](nrc.md)) for where the feature is heading, and have a concrete, evidence-backed answer ready when the
WebGPU group asks implementers what they need. **Nothing here has been posted.** Posting is outward-facing (D8): the
maintainer decides, and ideally after N2/N3 give us numbers to attach.

## 1. State of play (2026-11-14)

**Shipped in Chrome, no flag (relevant to us):**

| Feature | Chrome | What it gives us |
|---|---|---|
| `subgroups` | 134 | subgroup ops (portable kernels, reductions) |
| `subgroup_id` builtin (`@builtin(subgroup_id)`) | 144 | which subgroup a thread is in, without `local_invocation_index / size` |
| `subgroup_uniformity` WGSL language feature | 145 | uniformity analysis at subgroup scope: control flow on `subgroup_id` is subgroup-uniform. **Verified on Chrome 153** (SwiftShader): branching on `subgroup_id` around `subgroupAdd` compiles; branching on `local_invocation_index / 32` fails |
| `subgroup-size-control` / `@subgroup_size(n)` | 151–152 | pin subgroup size (VkNRC needs 32) |
| `unrestricted_pointer_parameters` | listed in Chrome 153's `wgslLanguageFeatures` | storage pointers as function parameters (we avoided them; no longer needed) |

**Still experimental:** `chromium-experimental-subgroup-matrix` (behind `--enable-unsafe-webgpu`; ~0.04–0.08% of
browsers expose it per web3dsurvey, research/VERIFY.md). WGSL `enable chromium_experimental_subgroup_matrix;`.

**Proposal today (`proposals/subgroup-matrix.md`):** opaque `subgroup_matrix_left/right/result<T, …>`; load/store
(row or column major, stride), multiply, multiply-accumulate, scalar add/sub/mul; configs from
`GPUAdapterInfo.subgroupMatrixConfigs` (component type, result type, M, N, K, min/max subgroup size); component types
f32, f16, i32, u32, i8, u8, plus limited-use `bf16`, `f8<e4m3>`, `f8<e5m2>` (#11093, merged 2026-11-14). Backends:
Vulkan `VK_KHR_cooperative_matrix`; Metal `simdgroup_matrix` (Apple7+, 8×8); D3D12 via SM 6.10 `linalg::Matrix`
(#10964, merged; SM 6.10 is still preview). Uniformity: every call in subgroup-uniform control flow with
subgroup-uniform operands, enforced by a `subgroup_matrix_uniformity` diagnostic that defaults to `error` and is
"triggered based on **workgroup** uniformity violations" — the proposal leaves "should we add subgroup uniformity
analysis?" as a TODO.

**Open investigations (opened Oct–Nov 2026, each with a Metal/D3D12/Vulkan survey and an API sketch):**

| Issue | Topic | Sketch |
|---|---|---|
| [#11077](https://github.com/gpuweb/gpuweb/issues/11077) | Element access | `enable subgroup_matrix_extended;` `subgroupMatrixLength / GetCoordinate / Get / Set` per invocation slot |
| [#11079](https://github.com/gpuweb/gpuweb/issues/11079) | Compose operations | `subgroupMatrixConvert<Destination>(result)`: result → left/result, with type change; row/column reductions; statistic redistribution. No transpose, no → right |
| [#11078](https://github.com/gpuweb/gpuweb/issues/11078) | Workgroup matrices, flexible dimensions | `GPUWorkgroupMatrixConfig` (+ `workgroupInvocations`), dimension ranges |
| [#11096](https://github.com/gpuweb/gpuweb/issues/11096) | Type naming | maybe `group_matrix<Kind, Scope, T, Cols, Rows>` |
| [#11073](https://github.com/gpuweb/gpuweb/issues/11073) | `@unroll` / `@no_unroll` | NVIDIA reports 5×+ from forcing unrolled, constant-indexed fragment loops; today kernels are host-generated |
| [#11051](https://github.com/gpuweb/gpuweb/issues/11051) | FP8 operands | E4M3 operands, f32 accumulate, decode-on-load allowed |
| [#6955](https://github.com/gpuweb/gpuweb/issues/6955) | Assert uniformity | let developers check the analysis result at compile time |
| [#4195](https://github.com/gpuweb/gpuweb/issues/4195) | Tracking | updated 2026-11-09 |

**What the Denoiser measured on M5 Pro (Chrome):** f16 FMA ~6.6 TFMA/s, subgroup-matrix MMA ~3.8 TMAC/s (f32 8×8×8),
f32 FMA ~3.1 TFMA/s; f16 matrices only with f16 accumulation; >1 subgroup per workgroup failed uniformity
(`li / 32` offsets; it predates or did not use `subgroup_id`). The matrix path gave fp32 57 → 44–50 ms; fp16 stayed on
the portable kernel. (pmndrs/denoiser `docs/specs/runtimes.md`.)

## 1b. Windows / NVIDIA (2026-11-16, RTX 3060 Laptop, driver 592.00)

**Chrome on Windows offers no subgroup matrices, with any flag.** Chrome 157 stable and Canary 160 run Dawn on D3D12 (chrome://gpu: D3D12
adapters plus blocklisted D3D11 ones). Dawn's feature doc says "D3D: no support" (HLSL WaveMatrix was withdrawn; the proposal's SM 6.10
`linalg::Matrix` path is not implemented), and `subgroup_id` is "not implemented on D3D". No switch gives Dawn a hardware Vulkan adapter
on Windows (tried `--use-webgpu-adapter=vulkan`, `--force-webgpu-adapter=vulkan`, `--enable-features=Vulkan --use-vulkan=native`,
`--enable-vulkan`; chrome://flags `force-webgpu-adapter` lists only Default and OpenGL ES). web3dsurvey shows the feature at 0% on Windows.

**The hardware and Dawn's Vulkan backend can.** The driver exposes `VK_KHR_cooperative_matrix` (Vulkan 1.4.325). Through dawn.node 0.6.2
(npm `webgpu`, Dawn + Tint as a Node addon) on Vulkan:

| Dawn toggles | `shader-f16` | `subgroupMatrixConfigs` |
|---|---|---|
| none | no | feature absent |
| `allow_unsafe_apis` | no | u8→u32 16×16×32, i8→i32 16×16×32, u8→u32 16×8×32, i8→i32 16×8×32 |
| `allow_unsafe_apis,vulkan_enable_f16_on_nvidia` | yes | f16→f16 and f16→f32 at 16×16×16, 16×8×16, 16×8×8, plus the int8 ones |

Dawn keeps f16 off on NVIDIA Vulkan by default because the driver reports `storageInputOutput16 = false`, and the f16 matrix configs go with
it. No f32-operand config exists (Apple has f32 8×8×8). Our inference kernel compiles with f16→f32 16×16×16 in the one-subgroup and
`subgroup_id ×4` forms; the `local_invocation_index / 32` control fails uniformity unless `diagnostic(off,
chromium.subgroup_matrix_uniformity)` (the unprefixed name is rejected). Tint in that build warns that `subgroupMatrixLoad`/`Store` are
deprecated builtins.

**For the feedback (not posted, N9):** Windows is where most discrete tensor-core GPUs are, and today a WebGPU page there cannot use them
at all. Either Dawn's D3D12 path (SM 6.10) or a Vulkan adapter for Chrome on Windows would close it; f16 on NVIDIA Vulkan additionally
needs the `storageInputOutput16` requirement relaxed (f16 stage IO is not what matrix code uses).

## 2. What an online-trained MLP (NRC) needs, mapped to the open issues

The workload: 64-wide, 5-hidden-layer MLP; inference on ~0.5–2M records per frame and 4 training steps of 16k records
(forward, backward, weight gradient, Adam), every frame. Reference implementation: VkNRC (Vulkan, NV cooperative
matrix). Each "today" cost below is what our WGSL port has to do instead (nrc-diverges.md section B).

| # | Need | Why, in this workload | Today in WGSL | Issue |
|---|---|---|---|---|
| W1 | **Per-element access on result tiles** (Get/Set, or an apply-per-element) | ReLU after every layer; ReLU derivative mask in backward; loss gradient written into the output tile | store tile → workgroup memory → per-lane op → reload, twice per layer (measured, M5 Pro f16 inference: the ReLU pass alone is 20-24% of the time with one subgroup per workgroup, 12-13% with four) | #11077 — **yes, this is our biggest ask** |
| W2 | **Result → left conversion, with type change** (f32/f16 result → f16 left) | layer l's output is layer l+1's input; without it every layer round-trips through workgroup memory | store + reload as left | #11079 — yes |
| W3 | **A transposed operand for backward** (result → right, or a transposing conversion) | backward needs Wᵀ·δ (load column-major works) and dW = δᵀ·A, where A and δ are held as result tiles | store, reload column-major | #11079 explicitly excludes transpose and → right: we'd ask to keep it on the table |
| W4 | **Subgroup-scope uniformity for matrix ops** (offsets from `subgroup_id` accepted). **Update 2026-11-15: Chrome 157 on Metal already accepts them** (compiles, matches the oracle, 1.5× over one subgroup); the ask shrinks to writing that into the proposal | several subgroups per workgroup sharing one staged weight layer (VkNRC: 4 × 32 over 128 records); one subgroup per workgroup re-stages weights 4× as often | unknown: the diagnostic is defined on workgroup uniformity; `subgroup_uniformity` exists since Chrome 145 | proposal TODO "should we add subgroup uniformity analysis?" — **yes, and tie it to `subgroup_uniformity`**. Small, concrete, and unblocks the Denoiser too |
| W5 | **f16 × f16 → f32 on Apple**, and a way to reach M5's neural accelerators (Metal 4 tensors / MPP) | f32 accumulation for training stability; Apple's simdgroup path has no mixed config and runs at about f32 rate | f16 → f16 only, ~3.8 TMAC/s < f16 FMA 6.6 | #11093 adds Metal 4 mappings and #11077 surveys Metal 4 `cooperative_tensor` — ask that this reaches M5's accelerators |
| W6 | **A performance hint per config** (e.g. "hardware accelerated" or a relative-rate class). Note 2026-11-15: for the MLP the Apple matrix path beat our (naive) portable f16 kernel 7.5×; the case for a hint now rests on N1.2's tuned kernel | on Apple the matrix path is slower than plain f16 math; an app cannot know which to pick without timing both | benchmark at startup | new ask (could ride on #11078's config interface) |
| W7 | **Float atomics (at least `atomicAdd` on f32 in storage)** | weight-gradient accumulation across workgroups (VkNRC does exactly this) | separate deterministic reduction pass | not subgroup-matrix; mention as context |
| W8 | **`@unroll`** | tile loops over fragment arrays; we generate unrolled WGSL on the host, like the Denoiser and NVIDIA's GEMM | host generation | #11073 — support |
| W9 | **Workgroup-scope matrices** | 128-record tiles across 4 subgroups with shared weights | many subgroup tiles + manual staging | #11078 — nice to have; W4 matters more |
| W10 | **bf16 operands** | training range without f16 loss scaling | f16 with summed (not averaged) deltas | #11093 merged — good; we'd use bf16 → f32 if offered |
| W11 | **Windows availability** | Chrome on Windows runs Dawn on D3D12, where this needs SM 6.10 (preview); tensor cores sit unused for most Windows users | Vulkan backend only, if selectable | ask about timeline / a Vulkan path on Windows |

Not needed by us (keep out of our ask): FP8 (#11051), reductions/statistic redistribution (#11079 parts), the naming
question (#11096) beyond "scope as a parameter is fine".

## 3. What we build now so the code is ready either way

- **Kernel generator with switchable stages**, so each WGSL gap is one code path we can delete: `relu: 'roundtrip' |
  'element'`, `chain: 'roundtrip' | 'convert'`, `subgroupsPerWorkgroup: 1 | 4`, `accumulate: 'f16' | 'f32'`.
  Today only the round-trip / 1-subgroup paths compile.
- **Measure the round trips** (N3): time of the store/op/reload steps per layer as a share of the layer. That number
  is the evidence for W1/W2.
- **Probe on every device** (N0.2): list `subgroupMatrixConfigs`; compile a one-tile kernel; compile it with offsets
  from `subgroup_id` under `subgroup_uniformity` (W4); try `diagnostic(off, …)` on the uniformity diagnostic with the
  name Chrome actually uses; record results in research note 06.
- **Paired timings** of `wgsl-f16` vs `wgsl-sgm` on M5 Pro and RTX 3060 (N2/N3), and VkNRC native on the 3060 (N6.1)
  as the "what native gets" line.

## 4. Draft comment (not posted; for the maintainer)

Target: #4195 (tracking) or split across #11077 / #11079 / the uniformity TODO. Attach numbers first.

> We're porting an online-trained neural radiance cache (Müller et al. 2021; a 64-wide, 5-hidden-layer MLP trained
> every frame) to WebGPU, following an open Vulkan implementation built on cooperative matrices. Every layer in the
> forward and backward pass needs three things the current proposal lacks, so each costs a workgroup-memory round
> trip today:
> 1. element access on result tiles for activations and their derivatives (#11077);
> 2. result → left conversion with a type change, to feed one layer's output into the next (#11079), and some way to
>    get a transposed operand for the weight gradient (dW = δᵀ·A), which #11079 currently excludes;
> 3. subgroup-scope uniformity for matrix built-ins, so several subgroups in a workgroup can share staged weights
>    using `@builtin(subgroup_id)` offsets now that `subgroup_uniformity` has shipped. The proposal's diagnostic is
>    defined on workgroup uniformity; tying it to `subgroup_uniformity` looks like the smallest change with the
>    biggest effect for us.
>
> Two device-level observations: on Apple M5 Pro in Chrome, the f32 8×8×8 matrix path measures ~3.8 TMAC/s against
> ~6.6 TFMA/s for plain f16 math, and f16 matrices only come with f16 accumulation, so a per-config performance hint
> (or Metal 4 tensor mappings that reach the M5 accelerators) would let apps choose a path without benchmarking it
> themselves. And `@unroll` (#11073) would let us stop generating unrolled kernels on the host.
>
> Measurements: [N2/N3 tables: share of each layer spent in round trips, wgsl-f16 vs wgsl-sgm, M5 Pro and RTX 3060].

## Sources

- gpuweb proposal: https://github.com/gpuweb/gpuweb/blob/main/proposals/subgroup-matrix.md
- Issues: #4195, #6955, #10964, #11051, #11073, #11077, #11078, #11079, #11093, #11096 on github.com/gpuweb/gpuweb
- Chrome: https://developer.chrome.com/blog/new-in-webgpu-144 (subgroup_id),
  https://developer.chrome.com/blog/new-in-webgpu-145 (subgroup_uniformity),
  https://developer.chrome.com/blog/new-in-webgpu-151-152 (subgroup size control);
  blink-dev "Intent to Ship: WebGPU: `subgroup_uniformity` feature"
- Dawn feature doc: https://dawn.googlesource.com/dawn/+/refs/heads/main/docs/dawn/features/subgroup_matrix.md (not
  reachable from cloud sessions)
- Denoiser measurements: github.com/pmndrs/denoiser `docs/specs/runtimes.md`, `packages/denoiser-wgsl/src/mma.ts`
