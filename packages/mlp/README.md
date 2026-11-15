# @pathtracer/mlp (private, research track N)

A small fully-fused MLP with online training in WGSL, plus an f64 CPU oracle. Built to answer whether a neural
radiance cache (NRC) is practical on WebGPU. Spec, plan and status: [docs/design/nrc.md](../../docs/design/nrc.md).
Design follows VkNRC (D23); the code is ours.

- `config.ts`: network shape (default 64 → 5 × 64 → 3, ReLU, no bias) and Adam/EMA/loss settings.
- `encoding.ts`: triangle-wave frequency and quartic one-blob encodings, CPU and WGSL; NRC and "learn an image" layouts.
- `oracle.ts`: forward, backward, relative-L2 luminance loss, Adam, EMA in f64. The reference for every GPU arm.
- `wgsl.ts`: portable kernels (`wgsl-f32`, `wgsl-f16`): prepare, infer, forward/backward, weight gradient, Adam, sync.
  Caller hooks in bind group 1 supply records (`nnTrainLoad`, `nnInferLoad`, `nnInferStore`). WGSL prefix `nn`.
- `wgsl-sgm.ts`: subgroup-matrix inference (`wgsl-sgm`, N3a): tiles from the adapter config, transposed weight view,
  workgroup-memory round trips where WGSL lacks element access / conversion, 1 or N subgroups; plus an emulated
  plain-WGSL form for correctness on any device. `pickSubgroupMatrixConfig` reads `adapter.info`.
- `engine.ts`: `MlpEngine` owns buffers and pipelines and encodes training steps and inference; counts are read
  from a GPU buffer (no readback).

Bench pages: `mlp-check` (GPU vs oracle), `mlp-fit` (learn an HDR image, PFM snapshots, `oracleSteps=N` replay),
`mlp-timing` (GPU ms per arm), `mlp-sgm-probe` (what compiles), `mlp-live` (interactive demo:
`pnpm --filter @pathtracer/bench mlp:live`). Mac checklist: docs/design/nrc.md, "Status and how to run".
