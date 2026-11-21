# WebNN ↔ WebGPU hand-off: does a better sync close the gap?

Status: **built, not yet measured on a real GPU** (2026-11-21). The arms below need Chrome with WebNN on the Mac (M5 Pro) or the
Windows laptop (RTX 3060); they cannot run in a cloud container (see "Where it runs").

## The question

[NRC-WEBNN-COMPACT.md](NRC-WEBNN-COMPACT.md) measured the compact radiance-cache network through WebNN at 44-48 ms per frame for
~23k queries, against 0.4 ms for our WGSL inference and ~1.3-1.5 ms for the whole cache frame including training, with identical
results. A reviewer asked whether the WebNN/WebGPU interop in the
[MLTensor explainer](https://github.com/webmachinelearning/webnn/blob/main/mltensor-explainer.md) removes the cost of the
hand-off: "why sync and push, it sounds like two steps that are not needed".

The explainer lists "synchronization of work between WebNN and WebGPU without CPU involvement" and "guaranteed zero-copy" as
non-goals, so every crossing stays a sync point through script. Our expectation is a small gain at most. This page tests it
anyway and keeps the numbers, so the answer is a measurement we can share, whichever way it goes.

## What each arm changes

All arms run `bench/nrc/webnn-replay.ts` on the same recorded queries and live weights. The defaults are the November 19
sequence, so every row below pairs one change against it with `--ab` (A B B A order, same browser session per run).

| Arg | A (default) | B | What B tests | Expectation (written before any run) |
|---|---|---|---|---|
| `sync` | `drain`: wait for the WebGPU queue after returning the input tensor, before `dispatch` | `none`: rely on `destroy()` alone | Whether the extra wait costs anything, and whether it is still needed | A few ms at most. Chrome 157 / Mac returned the previous frame's answers without it ([13-radiance-cache.md](../../research/notes/13-radiance-cache.md)): B may report `parity_failures` > 0, which means "faster but wrong" |
| `tensors` | `shared`: one input/output pair, 6 chunks one after another (3 waits per chunk, ~20 per frame) | `chunk`: a pair per chunk; all exports, copies, returns and dispatches batched (~5 waits per frame) | Whether the number of round trips matters | Small: halving the chunks (8,192 rows) did not change the time on Nov 19 |
| `ctx` | `type`: `createContext({ deviceType })` | `device`: `createContext(gpuDevice)` | The explainer's best case for zero-copy export | Unknown. The Denoiser saw a CPU fallback, 7-8x slower, on the Mac ([07-denoise-neural.md](../../research/notes/07-denoise-neural.md)); may differ on Windows or newer Chrome |
| `route` | `export`: `exportToGPU` + GPU copies | `cpu`: `mapAsync` read-back, `writeTensor` / `readTensor`, `queue.writeBuffer` | Whether the interop path beats the naive one at all | `export` faster, but not by the 100x the gap needs |

Always add `breakdown=1`: it reports `webnn_phase_<name>_p50_ms` for `encode`, `weights`, `in`, `wait`, `dispatch`, `out` and `finish`
(timestamps around the existing awaits, no extra waits). WebNN's own compute time lands in `out` on the `export` route, because
`exportToGPU` waits for the graph to finish.

`parity=report` records stale or wrong answers in `parity_failures` instead of stopping the run. Use it for `sync=none`. Every other
arm should read 0.

## Commands

Mac (Chrome with WebNN + Core ML device selection). Run each line once; `--repeat 2` gives two ABBA rounds (4 runs). Without
`--no-csv` the rows land in `bench/results/nrc-study.csv`, with the arm in the config hash.

```sh
FLAGS=--enable-features=WebMachineLearningNeuralNetwork,WebNNCoreMLExplicitGPUOrNPU
BASE="phase10=mlp webnnReplay=1 scene=nrc-materials-card res=320x180 scenario=static frames=16,64 warm=0 flip=0 refSpp=512 upload=0 repeats=20 weights=packed publish=frame rows=4096 device=gpu breakdown=1"
bash -c "PT_CHROME_FLAGS=$FLAGS node bench/runner/run.ts --page nrc-study --gpu apple --repeat 2 --ab sync=drain,none --args $BASE parity=report"
bash -c "PT_CHROME_FLAGS=$FLAGS node bench/runner/run.ts --page nrc-study --gpu apple --repeat 2 --ab tensors=shared,chunk --args $BASE"
bash -c "PT_CHROME_FLAGS=$FLAGS node bench/runner/run.ts --page nrc-study --gpu apple --repeat 2 --ab ctx=type,device --args $BASE"
bash -c "PT_CHROME_FLAGS=$FLAGS node bench/runner/run.ts --page nrc-study --gpu apple --repeat 2 --ab route=export,cpu --args $BASE"
```

(`bash -c` because zsh does not split `$BASE` into separate `k=v` arguments; see CLAUDE.md Traps. Check the echoed args on each PASS line.)

Windows (PowerShell, RTX machine, Chrome stable): the same four lines with `$env:PT_CHROME_FLAGS = '--enable-features=WebMachineLearningNeuralNetwork'`
and `--gpu nvidia`. Record the driver and the start temperature; the 3060 heats up quickly under back-to-back runs.

Then, if any B arm wins, run the winners together against the default (for example `--ab tensors=shared,chunk --args $BASE ctx=device`).

## Reading a result

- Quote `webnn_p50_ms` A vs B from the same run pair, and `wgsl_p50_ms` next to it as the target. Rows the runner marks `loaded`
  are not results.
- `parity_failures` > 0 means the arm returned stale or wrong predictions. Report its time, but it is not a usable option.
- The bar for WebNN to matter: below ~1.4 ms for this replay (the whole WGSL cache frame), and below 0.4 ms to beat WGSL inference.
  WebNN still cannot train, so weights cross every frame in every arm.

## Where it runs

- **Mac / Windows Chrome with the WebNN flag:** all arms.
- **Cloud container (Linux, SwiftShader):** none. The cache network is f16 and SwiftShader has no `shader-f16`, and exportable
  tensors lose the WebNN context there (`createExportableTensor`: "Context is lost", Chrome 157). Plain WebNN works there: the
  WebNN calls of the `cpu` route (`writeTensor` from a sliced buffer, shared weights across chunk dispatches, batched
  `readTensor`) were checked against a CPU result to f16 rounding (8e-4) in a standalone probe, shared and per-chunk alike.

## Results

None yet. Add one row per `--ab` run pair; keep failed or `loaded` runs out of the table but note them below it.

| Date | Machine | Chrome | WebNN device | Arm (A vs B) | A p50 ms | B p50 ms | B / A | WGSL p50 ms | B parity failures | Load / temp | Run ids |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|

## Conclusion

Open until the table has Mac and Windows rows.
