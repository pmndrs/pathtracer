# Compact NRC inference on WebNN

Use the portable WGSL network on the M5 Pro. Replaying 22,891 real second-hit queries took 0.4 ms median through the existing tiled kernel. The fastest measured WebNN configuration took 44.3 ms for the same queries, including feature packing, cross-API synchronization and returning RGB predictions to WebGPU. The entire raw NRC frame in the control was around 1.3–1.5 ms. The two routes reproduced the same updated network. [Actual renderer replay](../nrc/webnn-replay.ts) · [WebNN interop specification](https://www.w3.org/TR/webnn/)

## Cold build, throughput and parity

The blue-card view was rendered at 320 × 180 on Chrome 157.0.8103.80, Apple M5 Pro. The compact network has four matrices with a 64-wide input, three 32-wide hidden layers and a linear RGB output. The exported fourth output channel is zero. Training and EMA updates stayed in WGSL. Each arm compared four successive live weight snapshots to the native GPU predictions on every valid row and to a double-precision CPU forward pass on a sample of rows. At each successive snapshot the current versus stale CPU error ratio was below 0.023. All four snapshots were within 0.101% relative RMS of the native GPU RGB. The context type is the device **requested** from the browser. Physical execution placement was unverified.

| Rows per slab | WebNN device request | Useful / padded rows on final frame | Graph build | WebNN p50 / p95 | Portable WGSL p50 / p95 |
|---:|---|---:|---:|---:|---:|
| 4,096 | GPU | 22,891 / 1,685 | 52.8 ms | 47.3 / 50.1 ms | 0.4 / 0.6 ms |
| 4,096 | NPU | 22,891 / 1,685 | 54.1 ms | 47.6 / 48.8 ms | 0.4 / 0.6 ms |
| 8,192 | GPU | 22,891 / 1,685 | 84.2 ms | 47.9 / 54.4 ms | 0.4 / 0.7 ms |
| 8,192 | NPU | 22,891 / 1,685 | 94.1 ms | 44.3 / 46.3 ms | 0.4 / 0.5 ms |

Each row timed 20 replays after two warm-ups in the same browser/device session. The packed graph sliced one published weight tensor into the four GEMM operands. At 4,096 rows on the GPU request, publishing four separate layer tensors measured 48.4 ms versus 48.0 ms packed. Publishing the packed weights once per frame measured 47.9 ms versus 47.5 ms when republished for every slab in a separate six-replay comparison. The small differences were within shared-desktop run variation. With these shapes, neither reducing the weight export count nor halving the number of slabs closed the timing gap. At 16,384 rows Chrome lost the external instance reference. At 32,768 rows it rejected the input tensor size.

Both intervals start with raw NRC query records and current GPU-resident weights and end with GPU-resident RGB predictions. WebNN includes WGSL feature encoding into a staging buffer, weight publication, copies to/from exported buffers, synchronization, graph dispatch and unpacking predictions into the consumer's float32 RGB layout. Portable inference consumes the records directly. Both omit the renderer's common final pixel resolve and downstream reconstruction. Readback, CPU checking and graph build are outside the replay timing. A WebNN full-frame integration was not advanced because inference by itself exceeded the complete baseline frame. The separate exact-value synthetic geometry probe remains available. [Portable replay and local probe](../../research/probes/nrc-webnn.html) · [Denoiser interop](../../../Denoiser/docs/specs/runtimes.md)

Mac run times on November 19 UTC were 100940Z (layer exports), 100943Z (publication frequency), 101046Z (4,096 rows), 101228Z (8,192 rows) and 101049Z (larger tensors).

## Windows RTX 3060 (Phase 15, WN8–WN11)

**Decision: keep the portable WGSL network on Windows too.** In the quiet run (timing VALID), WebNN replay of the same records and weights in the same page took 15.1–15.8 ms p50 at 8,192 rows per slab and 28–29 ms at 4,096. WGSL per-record inference took 3.0–3.6 ms: WebNN is 4.4–9.7× slower. WebNN's best p50 is about four times the entire NRC frame (3.0–3.8 ms), so the protocol's integrated-frame step was not triggered. The first set of runs below was taken hot (86–87 °C), with a page in the maintainer's own Chrome holding 80–96% of the GPU. It read 23–64 ms against 3.0–6.0 ms (6.7–14.5×) and is kept as provisional. Under that load WebNN, with its CPU synchronisation every slab, lost more than WGSL did.

Setup: ASUS ROG Zephyrus M16 (GU603HM), Intel i9-11900H (no NPU), NVIDIA GeForce RTX 3060 Laptop GPU (Ampere, 0x2520), driver 592.00 (32.0.15.9200), Windows 11 Home 10.0.26200. Chrome 157.0.8103.79, headed window 1280 × 720, runner defaults (`--enable-unsafe-webgpu --enable-webgpu-developer-features`, adapter pinned by LUID with `--gpu nvidia`) plus `--enable-features=WebMachineLearningNeuralNetwork`. Node 24.13.1. Runs recorded commit `0e4435b`, with the `gemmC` change below in the working tree (committed with this report). The runs were serialised with the other GPU agents on this laptop through the webgpu-ddgi bench lock. The SM clock median under load was 1,567–1,597 MHz, and the power limit applied in at most 1.2% of samples.

### WN8 · Portable baseline (run 20261119T115628Z-nrc-study)

The settings match the protocol at the top of the handoff: compact MLP, `phase10=mlp` (reflected direction, one traced bounce, four position bands, raw RGB, still refinement off), 320 × 180 blue card, ABBA order, 22,686 inference queries per frame.

| WGSL kernel | NRC frame | Inference stage | Training steps | Warm reflection contrast (f16 / f64 ref) | Warm reflection rRMS |
|---|---:|---:|---:|---:|---:|
| tiled (A, A) | 5.8 / 9.4 ms | 3.8 / 4.8 ms | 4.3 / 4.7 ms | 0.958 / 0.960 · 0.965 / 0.964 | 0.223 / 0.223 |
| per-record (B, B) | 6.2 / 5.9 ms | 3.3 / 3.9 ms | 3.4 / 3.8 ms | 0.973 / 0.965 · 0.964 / 0.977 | 0.224 / 0.223 |

Plain path tracing at 1 spp read 0.986 reflection contrast. The two kernels give the same image within noise, and held-out display-only versus with-training-paths images agree to 9e-8 RGB. Per-record was the faster stage in both pairs, which agrees with the earlier D3D12 measurement (17 vs 24 ms per M queries, [NRC.md](NRC.md)). However, 3.3–4.8 ms for 22.7k queries is 150–210 ms per M, far above that quiet figure, so the kernel choice could not be made from this run. WN9 used per-record as its primary control and tiled as a second one.

**Quiet re-time (run 20261119T132754Z-nrc-study, timing VALID).** The foreign page was closed, and the run started at 62 °C with no thermal throttling. The power limit applied in 54% of loaded samples, and the SM clock median under load was 1,620 MHz. The CPU was saturated by the other agents (runner load 16.0, best effort).

| WGSL kernel | NRC frame | Inference stage | Training steps | Plain 1 spp | Warm reflection contrast (f16 / f64 ref) |
|---|---:|---:|---:|---:|---:|
| tiled (A, A) | 5.6 / 3.1 ms | 4.8 / 3.1 ms | 5.0 / 3.4 ms | 4.3 / 3.0 ms | 0.975 / 0.961 · 0.967 / 0.964 |
| per-record (B, B) | 4.9 / 5.1 ms | 3.8 / 4.0 ms | 4.3 / 4.4 ms | 4.2 / 3.9 ms | 0.956 / 0.971 · 0.961 / 0.967 |

**The two kernels tie on this page.** The arm means are within 0.1 ms, while each arm's two runs differ by up to 1.7 ms. The stage times are wall-clock submit-and-wait medians with 0.1 ms resolution, so 3–5 ms for 22.7k queries (130–210 ms per M) is mostly submission latency, not kernel time. The GPU-timestamp kernel bench is the measurement that separates the kernels: per-record was faster on D3D12 (17 vs 24 ms per M, [NRC.md](NRC.md)), and that remains the Windows recommendation. Warm reflection rRMS was 0.223–0.224 and relMSE 4.8e-3 for both.

### WN9 / WN10 · WebNN GPU and NPU requests

**First attempt (runs 20261119T120402Z, 120446Z, 120533Z): every build failed.** The page saw `UnknownError: Failed to execute 'build' on 'MLGraphBuilder': Failed to create session.` Chrome's log has the cause: `[WebNN] ORT status error code: 10 error message: This is an invalid model. In Node, ("fused op (#0 '0_') + (#1 '1_')", DmlFusedGemm, "com.microsoft.dml", -1) ... has input size 2 not in range [min=3, max=3].` Chrome 157's Windows backend is ONNX Runtime with the DirectML execution provider. It fuses `gemm` + `relu` into `DmlFusedGemm`, which requires the bias input C, and our graph has none. The same error appears for the GPU and NPU requests and for the 64-row synthetic probe. `--disable-features=WebNNOnnxRuntime` did not reach a working backend.

**Workaround.** `bench/nrc/webnn-replay.ts` now passes a zero float16 `c` to every `gemm` (`gemmC=zero`). It is the default on Windows, and `gemmC=none` restores the original graph. Adding zero is exact, and the Mac graph is unchanged (default `none` off Windows). Runs with it: 20261119T125543Z (GPU request, per-record control), 20261119T125640Z (GPU request, tiled control), 20261119T125735Z (NPU request, per-record control). Each used ABBA over rows 4,096 / 8,192, 20 timed replays after 2 warm-ups, packed weights published once per frame. All 12 runs passed.

**Parity, all 12 runs × 4 changing snapshots:** relative RMS against the native WGSL predictions on every valid row was 0.072–0.098%. The worst CPU double-precision oracle error on sampled rows was 0.14–0.72% relative. The current-over-stale snapshot error ratio was 0.0032–0.0168 (threshold 0.5). Rows per snapshot were 22,808–22,957; the final frame had 22,871 useful and 1,705 padded rows at both slab sizes.

| Request | WGSL control | Rows / slab | Graph build | WebNN p50 / p95 | WGSL p50 / p95 | p50 ratio |
|---|---|---:|---:|---:|---:|---:|
| GPU | per-record | 4,096 | 616 / 79 ms | 43.7 / 51.0 · 63.9 / 69.7 ms | 3.7 / 4.5 · 4.4 / 6.3 ms | 11.8 · 14.5 |
| GPU | per-record | 8,192 | 95 / 80 ms | 39.9 / 49.8 · 35.6 / 40.8 ms | 6.0 / 7.1 · 4.0 / 5.4 ms | 6.7 · 8.9 |
| GPU | tiled | 4,096 | 316 / 72 ms | 39.5 / 45.5 · 62.4 / 68.0 ms | 3.0 / 4.4 · 4.3 / 5.9 ms | 13.2 · 14.5 |
| GPU | tiled | 8,192 | 75 / 76 ms | 34.6 / 37.3 · 36.0 / 39.3 ms | 4.5 / 6.1 · 3.6 / 5.1 ms | 7.7 · 10.0 |
| NPU | per-record | 4,096 | 191 / 75 ms | 37.1 / 39.8 · 62.9 / 70.2 ms | 3.0 / 4.1 · 4.5 / 5.1 ms | 12.4 · 14.0 |
| NPU | per-record | 8,192 | 80 / 76 ms | 23.4 / 26.1 · 31.9 / 39.8 ms | 3.1 / 4.5 · 4.0 / 6.5 ms | 7.5 · 8.0 |

Each cell lists the two runs of that row in run order (A runs are 1 and 4, B runs 2 and 3). The first graph build in a browser session is slower (190–616 ms), and later builds take 72–95 ms. The last 4,096-row run of each session read 62–64 ms in all three sessions, about 50% above the first, consistent with the GPU heating further during the session. 8,192-row slabs (3 dispatches) beat 4,096 (6 dispatches), as on the Mac.

**Quiet re-time, GPU request (run 20261119T135040Z-nrc-study, timing VALID).** The run started at 62 °C with no thermal throttling. The power limit applied in 75% of loaded samples, and the SM clock median under load was 1,890 MHz. The WGSL control was per-record. The same ABBA protocol was used, and all 4 runs passed.

| Rows / slab | Graph build | WebNN p50 / p95 | WGSL p50 / p95 | p50 ratio | NRC frame |
|---:|---:|---:|---:|---:|---:|
| 4,096 | 213 / 73 ms | 29.0 / 30.0 · 28.1 / 29.7 ms | 3.0 / 4.0 · 3.0 / 4.1 ms | 9.7 · 9.4 | 3.8 · 3.0 ms |
| 8,192 | 77 / 72 ms | 15.1 / 16.8 · 15.8 / 17.8 ms | 3.1 / 4.1 · 3.6 / 4.1 ms | 4.9 · 4.4 | 3.0 · 3.8 ms |

Parity was the same as in the hot runs: 0.074–0.096% relative RMS against native, 0.16–0.47% worst against the CPU oracle, and a current-over-stale ratio of 0.0027–0.0139. Under quiet conditions the 4,096-row cost (6 slabs) is almost exactly twice the 8,192-row cost (3 slabs), so WebNN pays about 5 ms per slab for the export, synchronisation and dispatch round trip. The GEMMs themselves are not the cost. The tiled-control and NPU requests were not re-timed: the WGSL kernels tie on this page, and the NPU request falls back.

**NPU (WN10).** This laptop has no NPU, yet `createContext({ deviceType: 'npu' })` succeeded and the graph ran with the same parity. Chrome falls back silently, so a successful NPU context is not evidence of NPU execution. Physical placement was not verified, as on the Mac.

**Next.** WebNN on this path is not worth revisiting until a slab costs less than the whole WGSL inference stage, which needs WebNN–WebGPU interop without a CPU synchronisation per dispatch. Settling tiled against per-record needs GPU timestamps (the MLP kernel bench), not this page's wall clock.

## Reproduce on Mac or Windows

From the pathtracer root on the Mac:

```sh
PT_CHROME_FLAGS=--enable-features=WebMachineLearningNeuralNetwork,WebNNCoreMLExplicitGPUOrNPU node bench/runner/run.ts --page nrc-study --gpu apple --no-csv --ab device=gpu,npu --args phase10=mlp webnnReplay=1 scene=nrc-materials-card res=320x180 scenario=static frames=16,64 warm=0 flip=0 refSpp=512 upload=0 repeats=20 weights=packed publish=frame rows=4096
```

Windows phase 15 requires the RTX machine. Use this command in PowerShell to run actual renderer records against the WGSL baseline and the Windows WebNN backends. Start with the GPU request if the machine has no NPU. The result JSON reports snapshot parity and GPU-visible end-to-end inference for each supported configuration. Repeat with `infer=portable` to include the per-record backend that performed well on the RTX 3060. Record driver, browser and requested device with the results. The Dawn/Vulkan subgroup-matrix measurements are a separate native-backend comparison. [RTX reference measurements](NRC.md)

```powershell
$env:PT_CHROME_FLAGS = '--enable-features=WebMachineLearningNeuralNetwork'
node bench/runner/run.ts --page nrc-study --gpu nvidia --no-csv --ab device=gpu,npu --args phase10=mlp webnnReplay=1 scene=nrc-materials-card res=320x180 scenario=static frames=16,64 warm=0 flip=0 refSpp=512 upload=0 repeats=20 weights=packed publish=frame rows=4096
```
