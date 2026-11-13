# Third-party code and assets

Everything in this repo that was ported, adapted or downloaded from somewhere else, and everything we only run.
Rules are in [CLAUDE.md](CLAUDE.md) (Licences and clean room). Keep this file current when you port something; record
the technical departure in [DIVERGES.md](DIVERGES.md) as well.

Last checked: 2026-11-13, by reading file headers (`grep -rniE "ported|adapted|licen|BSD|MIT|Apache"`).

## 1. Ported or adapted code

### 1.1 From our own sibling repos (MIT, same author, Dennis Smolek)

No licence obligation beyond keeping this record; listed so the lineage is not lost. Sibling commits are the HEAD at
the time of this check, not necessarily the commit each file was ported from.

| Our file | Source | Licence | Source commit (at check) | Notes |
|---|---|---|---|---|
| `bench/runner/run.ts` | `~/Developer/bvh/tests/gpu/run.ts` | MIT | bvh `498a6c1` | One result per run instead of pass/fail pages |
| `bench/runner/chrome.ts` | `~/Developer/bvh/tests/gpu/chrome.ts` | MIT | bvh `498a6c1` | macOS first, Apple/AMD choices |
| `bench/runner/cdp.ts` | `~/Developer/bvh/tests/gpu/cdp.ts` | MIT | bvh `498a6c1` | |
| `bench/runner/cleanup.ts` | `~/Developer/bvh/tests/gpu/cleanup.ts` | MIT | bvh `498a6c1` | |
| `bench/runner/evaluate.ts` | `~/Developer/bvh/tests/gpu/evaluate.ts` | MIT | bvh `498a6c1` | |
| `bench/runner/session.ts` | `~/Developer/bvh/tests/gpu/session.ts` | MIT | bvh `498a6c1` | page URL + args, `BenchReport` |
| `bench/runner/server.ts` | `~/Developer/bvh/tests/gpu/server.ts` | MIT | bvh `498a6c1` | adds the PFM image upload endpoint |
| `bench/runner/measured-commit.ts` | `~/Developer/bvh/tests/gpu/measured-commit.ts` | MIT | bvh `498a6c1` | |
| `bench/runner/clock-note.ts` (+ test) | `~/Developer/webgpu-ddgi/tools/gpu/clock-note.ts`, via `~/Developer/bvh/tests/gpu/clock-note.ts` | MIT | ddgi `9dddefe`, bvh `498a6c1` | `PT_` env names, live sampler, our CSV row builder |
| `bench/runner/contract.ts` | shape follows `~/Developer/bvh/tests/gpu/contract.ts` | MIT | bvh `498a6c1` | reduced to what the bench needs |
| `bench/runner/page-client.ts` | shape follows `~/Developer/bvh/tests/gpu/client.ts` | MIT | bvh `498a6c1` | |
| `bench/runner/gpu-lock.ts` (+ test) | same protocol as `~/Developer/Denoiser/tools/gpu-lock.sh` (a shell script, reimplemented in TypeScript) | MIT | Denoiser `7014384` | takes over locks of dead owners (V5) |
| `bench/metrics/flip.ts` | `~/Developer/Denoiser/examples/realtime-denoiser/src/flip.ts` (our own implementation), written after NVIDIA's FLIP reference | MIT (ours), reference BSD-3, see 1.2 | Denoiser `7014384` | LDR-FLIP only; reference is `github.com/NVlabs/flip` |

Ideas, not code (no licence issue, kept for lineage): `bench/runner/csv.ts` columns follow `~/Developer/webgpu-ddgi/bench/results`;
`bench/runner/stats.ts` (ABBA, median of blocks) and `bench/pages/timer-calibration.ts` follow
`~/Developer/upscaler` (`scripts/measure-timer-overhead.mjs`, `bench/timer-overhead.html`); `packages/raytracer/src/contract/ray.ts`
mirrors the `@pmndrs/bvh` GPU layout v2 (`~/Developer/bvh/src/trace/layout.ts`). All of those siblings are MIT, same author.

### 1.2 FLIP (NVIDIA), BSD-3-Clause

- Algorithm and constants: Andersson, Nilsson, Akenine-Moller, Oskarsson, Astrom, Fairchild, "FLIP: A Difference Evaluator
  for Alternating Images", HPG 2020.
- Reference implementation: <https://github.com/NVlabs/flip>, BSD-3-Clause, Copyright (c) 2020-2024 NVIDIA CORPORATION &
  AFFILIATES (the upstream LICENSE at HEAD says 2020-2025). Pinned: commit `d3ab02a4425eb167d8a4021b382a2bafc6e6c838`
  (flip 1.7); validated against its images and its pip package `flip-evaluator` 1.7 (TASKS P0.14, DIVERGES V6).
- What we have: `bench/metrics/flip.ts`, an LDR-FLIP port through our Denoiser implementation (see 1.1), plus HDR-FLIP
  (P0.14: exposure range, tonemap coefficients and per-pixel max over exposures follow `FLIP.h` of the pinned commit;
  `flip-parallel.ts`/`flip-worker.ts` are ours). No NVIDIA source file is copied verbatim; the pipeline and the
  constants follow the reference, so we carry the BSD-3 notice.
- Licence text: [licenses/NVlabs-FLIP-BSD-3-Clause.txt](licenses/NVlabs-FLIP-BSD-3-Clause.txt), verbatim from the pinned commit.
- Not used: the reference's PyTorch/C++/CUDA code. The clone and the pip venv live in `.cache/flip-validation/` (git-ignored) and are only used to produce validation outputs.

### 1.3 Clean-room status

- NVIDIA RTX SDK repos (RTXPT, RTXDI, RTXGI, SHaRC, NRD): nothing ported. NRD code never opened.
- mutracer: nothing ported (see section 2).
- Third-party clones under `research/sources/` are read-only and git-ignored; no code from them is in the tree.

## 2. Run, not copied (competitor and dependency packages)

Installed from npm as `bench/` devDependencies and called through their public APIs by the adapters. None of their code is
in our packages. Versions, hashes and adapter ids are kept in [bench/adapters/COMPETITORS.md](bench/adapters/COMPETITORS.md).

| Package | Version | Licence | Used by |
|---|---|---|---|
| `three-gpu-pathtracer` (Garrett Johnson) | 0.0.28 (gitHead `9f5b910d`) | MIT | `bench/adapters/tgpt-*` (`tgpt-wavefront`, `tgpt-megakernel`) |
| `three-mesh-bvh` | 0.9.18 | MIT | dependency of the above |
| `three` / `@types/three` | 0.187.0 | MIT | dependency of the above |
| `@meimu/mutracer` (github.com/pissang/mutracer, Yi Shen) | 0.3.1 | MIT | `bench/adapters/mutracer` (we never copy its code, CLAUDE.md) |
| `oidn-webgpu` | pulled by mutracer | Apache-2.0 | not loaded: we never attach a denoiser |

Tooling (dev only, not shipped): vite, vitest, typescript, eslint, `@webgpu/types` (all MIT/Apache-2.0/BSD under their own
packages; see `pnpm-lock.yaml` for versions). Native reference renderers pbrt-v4 (BSD-2-Clause) and Mitsuba 3 (BSD-3-Clause)
are built under `.cache/native/` (git-ignored) and only their rendered images are used; no code or scene data is copied.

## 3. Third-party assets

| Asset | Used by | Licence | Source | Committed? |
|---|---|---|---|---|
| Sponza (glTF + `.bin`; geometry only, every material replaced by a grey Lambertian) | scene `sponza-white` (`bench/scenes/tier2/sponza-white.ts`) | Cryengine Limited License Agreement (Crytek, 2016) for the model, CC-BY-4.0 for the glTF metadata | Khronos glTF-Sample-Assets, commit `8f492390ddc7a2ae221a1369ef560589cf8c70fb`, `Models/Sponza` <https://github.com/KhronosGroup/glTF-Sample-Assets/tree/8f492390ddc7a2ae221a1369ef560589cf8c70fb/Models/Sponza> | **No.** Downloaded by `bench/assets/fetch.ts`, git-ignored under `bench/assets/downloads/`, sha256 pinned in [bench/assets/MANIFEST.json](bench/assets/MANIFEST.json). Do not commit or redistribute the model files. |
| Cornell box dimensions and colours | scene `cornell` (`bench/scenes/tier0/cornell.ts`) | measurement data, no licence file; rebuilt as our own geometry code | Goral et al., as distributed by the Cornell PCG lab | Numbers only, in our own scene code |

Attribution line for Sponza (required by the manifest): "Sponza by Crytek (2016), via Khronos glTF-Sample-Assets".
The Cryengine licence text is fetched with the asset (`downloads/sponza/LicenseRef-CRYENGINE-Agreement.txt`); read it before
any redistribution of rendered images that show the model (reference renders are git-ignored today).

Generated assets (`bench/assets/generated/`, env maps and noise made by our own scripts) are ours.

## 4. Not yet in the tree, with a licence plan

- Blue noise landed (TASKS P0.17) as **our own code**, written from the papers: Ulichney 1993 (void-and-cluster),
  Georgiev & Fajardo 2016 (vec2 swap energy), Wolfe et al. 2022 / Donnelly et al. 2024 (separate spatial + temporal
  filters), Burley 2020 (hash-based Owen scrambling, `src/sampling/owen-sobol.ts`; only the published construction
  and its four hash constants). The textures in `packages/raytracer/assets/bluenoise/` are generated by
  `packages/raytracer/tools/bluenoise/` and are ours. No EA FAST (BSD-3) code was read or ported and no NVIDIA STBN
  code or texture is used (D16), so no notice is owed. If FAST code is ever ported, add its BSD-3 text under
  `licenses/` and a section 1 entry.
