# WebNN ↔ WebGPU hand-off: does a better sync close the gap?

Status: **measured on the Mac (M5 Pro, Chrome 158) on 2026-11-24; Windows (RTX 3060) still open.** The short answer is in
[Conclusion](#conclusion): the `exportToGPU` hand-off itself was the cost. Going through JavaScript instead takes about 3 ms, not 47 ms.
The arms need Chrome with WebNN on a real GPU; they cannot run in a cloud container (see "Where it runs").

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

One row per `--ab` run pair (two rounds, A B B A). The p50 columns give the two runs of each arm; B / A is the ratio of their means.
Every figure is the wall-clock time of one replay of 22,679 queries (6 chunks of 4,096 rows unless stated), measured in the page with a
0.1 ms clock. The WGSL column is our own inference timed the same way in the same run.

All rows: 2026-11-24, MacBook Pro M5 Pro (18 cores), headless Chrome 158.0.8127.88, Metal. The Mac was in normal use, not a quiet machine:
load average 2.9-4.2, one browser GPU process, nine desktop apps open. The runner marked no run `loaded` and reported no thermal limit.
Treat the absolute numbers as best effort; the ratios are paired within one session and the two rounds agree to a few percent.

Rows 1-4 are the planned arms, each against the November 19 default. Rows 5-10 are follow-ups on the winner of row 4. Rows 11-15
(run later the same day at load 4.4-5.6) ask the review question "why not one dispatch per frame?" with the batched tensors.

| # | WebNN device | Fixed args | Arm (A vs B) | A p50 ms | B p50 ms | B / A | WGSL p50 ms | B parity failures | Load | Run id |
|---:|---|---|---|---:|---:|---:|---:|---:|---|---|
| 1 | gpu | defaults | `sync` drain vs none | 49.9, 46.3 | 37.0, 40.3 | 0.80 | 0.4-0.5 | **7, 7 (wrong)** | 2.9-3.0 | 20261124T003344Z |
| 2 | gpu | defaults | `tensors` shared vs chunk | 47.3, 44.4 | 40.2, 38.8 | 0.86 | 0.4 | 0 | 4.1 | 20261124T003420Z |
| 3 | gpu | defaults | `ctx` type vs device | 48.2, 45.7 | 47.5, 46.6 | 1.00 | 0.5-0.6 | 0 | 3.8-4.0 | 20261124T003430Z |
| 4 | gpu | defaults | `route` export vs cpu | 48.2, 46.7 | **3.0, 3.0** | **0.063** | 0.5 | 0 | 3.9-4.2 | 20261124T003442Z |
| 5 | gpu | `route=cpu` | `tensors` shared vs chunk | 3.0, 3.0 | **2.8, 2.7** | 0.92 | 0.5 | 0 | 3.4-3.5 | 20261124T003517Z |
| 6 | gpu | `route=cpu` | `rows` 4096 vs 32768 (6 chunks vs 1) | 3.0, 3.0 | 3.0, 3.0 | 1.00 | 0.5 | 0 | 3.1-3.4 | 20261124T003523Z |
| 7 | gpu | `route=cpu tensors=chunk` | `rows` 4096 vs 32768 | 2.7, 2.8 | 3.2, 3.0 | 1.13 | 0.4-0.5 | 0 | 3.3 | 20261124T003530Z |
| 8 | gpu vs npu | `route=cpu` | `device` gpu vs npu | 2.9, 3.1 | 2.9, 3.0 | 0.98 | 0.5 | 0 | 3.3-3.4 | 20261124T003537Z |
| 9 | gpu | `route=cpu` | `ctx` type vs device | 3.0, 3.3 | 3.0, 3.2 | 0.98 | 0.5 | 0 | 3.3-3.4 | 20261124T003543Z |
| 10 | npu | defaults | `route` export vs cpu | 50.8, 46.2 | 3.0, 3.0 | 0.062 | 0.5 | 0 | 3.4-3.8 | 20261124T003550Z |
| 11 | gpu | `tensors=chunk` (export) | `rows` 4096 vs 32768 (6 dispatches vs 1) | 37.6, 37.3 | **failed, failed** | — | 0.4 | — | 4.8 | 20261124T090941Z |
| 12 | gpu | `route=cpu tensors=chunk` | `rows` 4096 vs 32768 (6 dispatches vs 1) | 2.7, 3.0 | 3.0, 2.8 | 1.02 | 0.3-0.4 | 0 | 4.4 | 20261124T090952Z |
| 13 | gpu | `tensors=chunk` (export) | `rows` 4096 vs 16384 (6 dispatches vs 2) | 36.4, 33.5 | **failed, failed** | — | 0.4 | — | 5.6 | 20261124T090959Z |
| 14 | gpu | `route=cpu tensors=chunk` | `rows` 4096 vs 16384 (6 dispatches vs 2) | 2.7, 2.7 | 2.8, 2.9 | 1.06 | 0.4 | 0 | 5.5 | 20261124T091007Z |
| 15 | gpu | `tensors=chunk` (export) | `rows` 4096 vs 8192 (6 dispatches vs 3) | 35.3, 39.0 | 38.8, 37.3 | 1.02 | 0.4 | 0 | 5.6 | 20261124T091040Z |

Other numbers from the same runs:

- p95 (ms): default export route 47.3-52.9. Script route 3.0-3.5 in the calm runs, and 4.8-6.3 in about half of them (one or two slow replays out
  of 20), so its tail is less steady than its median.
- WGSL inference p95: 0.5-0.8 ms. The whole WGSL cache frame including training (`ms_cache_frame`): 1.0-1.2 ms in these runs.
- WebNN graph build: 46-54 ms in 35 of 38 runs, 63-94 ms in the other three (a one-off cost per graph).
- Parity in every arm except `sync=none`: relative RMS against the WGSL predictions 0.0008-0.0012 (f16 rounding), and each of the three
  consecutive weight snapshots was told apart from the previous one (error against current weights 0.5-2.3% of the error against stale weights).

Failed or excluded runs in rows 1-10: none. No run crashed, no run was marked `loaded`, and WebNN and `exportToGPU` were available in every arm.
Rows 11 and 13 (B arm, both rounds) failed, and the failure is the finding: an exportable tensor of shape [rows, 64] is backed by an IOSurface
whose height is the row count, and Chrome 158 on macOS caps that at 8,192. At 32,768 rows `createExportableTensor` throws
`TypeError: Tensor size is too large`. At 16,384 rows WebGPU reports `IOSurface height (16384) exceeds maxTextureDimension2D (8192)` and the
device is lost (`A valid external Instance reference no longer exists`). So with this tensor shape the export route cannot take a
frame of 22,679 queries in fewer than three dispatches. Plain tensors (the script route) have no such limit.
`sync=none` (row 1) ran but is not usable: all four snapshots in both runs returned answers unrelated to the correct ones (relative RMS about 1.0,
neither the current nor the previous frame's predictions), so the wait is still required in Chrome 158.

Not established: whether `device=npu` or `ctx=device` changed what Core ML ran on. Both arms gave the same times and the same results as the
default, and the API does not report the processor it picked.

### Where the time goes

Per-phase medians (ms) of one replay. `in` = getting the 3 MB of encoded queries into WebNN, `wait` = the WebGPU queue wait before `dispatch`,
`out` = running the graph and getting 196 KB of results back, `finish` = the final unpack pass and queue wait.

| Variant | weights | in | wait | dispatch | out | finish | total p50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Default: `exportToGPU`, shared tensors (rows 1-4, arm A) | 0.1 | 13.1-13.4 | 26.3-29.1 | 0.0-0.1 | 1.2-1.7 | 3.3-3.8 | 44-50 |
| `sync=none` (wrong answers) | 0.1 | 13.0-13.4 | — | 0.1 | 20.0-23.8 | 3.5-3.7 | 37-40 |
| `exportToGPU`, a tensor pair per chunk | 0.1 | 0.5 | 20.8-22.5 | 0.1 | 1.4 | 15.7-15.8 | 39-40 |
| Through JavaScript, shared tensors | 0.2 | 0.9-1.0 | — | 0.0 | 1.2-1.4 | 0.5 | 3.0 |
| Through JavaScript, a tensor pair per chunk | 0.2 | 1.0 | — | 0.0 | 0.9-1.0 | 0.5 | 2.7-2.8 |

What this says:

- **WebNN's own compute is not the problem.** On the script route, running all six chunks and reading the results back takes 0.9-1.4 ms
  in total. On the export route, the six output exports (which wait for the graph) take 1.2-1.7 ms. The network itself costs about a millisecond.
- **The export route spends about 45 of its 47 ms handing buffers back and forth.** Each time a WebGPU buffer exported from a tensor is returned
  to WebNN, roughly 4.5 ms passes before the queue is idle again (28 ms of `wait` over six chunks), and exporting the input tensor costs
  another 2.2 ms per chunk (13 ms). Exporting the output is cheap (about 0.2 ms per chunk, compute included).
- **The cost moves instead of going away.** Dropping the wait pushes the time into the next export and gives wrong answers. Batching all six
  chunks removes the per-chunk input cost (13 ms to 0.5 ms), but the single remaining wait takes 21-22 ms and the final one 16 ms.
- **Fewer dispatches do not help either route.** In the batched export arm, three dispatches of 8,192 rows take the same time as six of 4,096
  (row 15: `wait` 20.6-20.7 ms and `finish` 15.2-16.4 ms against 17.7-21.2 and 15.5-16.0). So the two stalls are a roughly fixed cost
  per hand-off round on this machine. They do not scale with the number of buffers exported, which corrects the per-buffer reading an
  earlier version of this note took from the unbatched arm (28 ms of `wait` over six chunks). On the script route one dispatch for the
  whole frame takes 2.8-3.0 ms against 2.7-3.0 ms for six (rows 6, 7, 12, 14).
- **The script route has a floor of its own.** Reading the query features back from the GPU and writing them to WebNN is 1.0 ms, the weight
  read-back 0.2 ms, the results 0.9-1.4 ms, the final upload and unpack 0.5 ms. About 1.7 ms of the 2.7 ms would remain even if the network
  took no time at all. One chunk of 32,768 rows instead of six of 4,096 does not help (rows 6-7).

## Conclusion

Mac only; the Windows rows are still open, and the numbers come from a Mac in normal use (see the note above the table).

**Does a better hand-off between WebNN and WebGPU close the gap? Partly, and not in the way anyone expected.**

- The 44-48 ms measured on November 19 was almost entirely the cost of Chrome's `exportToGPU` interop on macOS, not of WebNN running the
  network. That path was chosen because it is the one designed for sharing data with WebGPU. On this machine it is the slow one.
- The plain route, copying the inputs and results through JavaScript, runs the same replay in **3.0 ms**, and **2.7-2.8 ms** with one tensor
  pair per chunk. That is 16-17 times faster than the export route, with identical results. It closes about 95% of the gap.
- It is still short of both bars. The whole of our WGSL cache frame, training included, takes about 1.1-1.4 ms; WebNN inference alone
  takes about twice that. Against WGSL inference alone (0.4-0.5 ms) it is about 6 times slower. And WebNN cannot train, so the WGSL training
  pass stays and the weights still cross every frame.
- One dispatch per frame is not the missing piece. Through JavaScript, one dispatch of the whole frame takes the same 2.8-3.0 ms as six.
  Through `exportToGPU` it cannot be run with this tensor shape (8,192-row cap, see the failed rows), and going from six dispatches to
  three changed nothing. Not tested: folding the frame into a wider tensor (for example [3072, 512] reshaped in the graph) to get one
  exported dispatch, and expressing training as WebNN graph operations. Training stayed in WGSL in every arm.
- Of the three ideas taken from the MLTensor explainer, none made the export route competitive: dropping the extra wait is 20% faster but
  returns wrong answers; a tensor pair per chunk is 14% faster (39-40 ms); creating the context from the `GPUDevice` changes nothing.
  The best export-route arm is still about 28 times over the 1.4 ms bar.

Where the written expectations were wrong:

| Arm | Expected | Measured |
|---|---|---|
| `sync=none` | A few ms at most; may be wrong | 8-10 ms faster, and wrong in every snapshot. The wait is still needed in Chrome 158. Right in kind, low on size |
| `tensors=chunk` | Small | 14% on the export route, 8% on the script route. Right |
| `ctx=device` | Unknown, possibly a CPU fallback 7-8 times slower | No change either way. No fallback seen, no gain |
| `route=cpu` | `export` faster, though not by enough | **Wrong, by a wide margin.** The script route is 16 times faster than `export` |

What this does and does not change for us: the earlier statement "WebNN is about 100 times slower than our WGSL inference" should now read
"about 6 times slower when the data is copied through JavaScript; about 100 times slower through `exportToGPU` on macOS". The decision to
keep the network in WGSL stands on the measured numbers: WebNN inference alone costs twice our whole cache frame. Two limits on the script
route's number: it was measured at 320x180 (3 MB of query features read back per frame, and that copy grows with the query count), and it
puts a GPU read-back in every frame, which our frame loop otherwise avoids.

Still open: the same arms on Windows (RTX 3060, Chrome stable), where the WebNN backend and the export path are different code, and a rerun
on a quiet Mac if the absolute numbers are to be quoted as final.
