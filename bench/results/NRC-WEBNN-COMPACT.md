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
