# Research brief (shared by every research agent)

Date of this brief: 2026-11-13. Search for recent work (2024–2026) explicitly; do not stop at what you already know.

## What we are building

A **web-based ray and path tracing library** (WebGPU, WGSL, engine-agnostic core with adapters) plus a suite of
shared resources. It must serve two audiences:

1. **Utility consumers** — sibling libraries that need rays: reflections, GI (DDGI / radiance cascades / probes),
   AO, shadows, baking, visibility queries. They need a fast, shared "trace + minimal shade" core rather than each
   building its own tracer on top of the BVH.
2. **A full PBR path tracer** — progressive, physically based, eventually spectral, competitive in quality with
   pbrt-v4 / Mitsuba on reference scenes and faster than any other web path tracer.

### Core rules (in priority order)

1. **Speed above all.** Be as fast as a software (non-hardware-RT) GPU ray tracer can be. Every trick from the last
   40 years of RT research is in scope: noise/sample patterns, weird math, traversal tricks, memory layout,
   scheduling, reuse (ReSTIR, caching), reconstruction.
2. **Quality.** Unbiased/consistent modes for reference, plus fast biased real-time modes.
3. **Good DX.** Clean APIs, engine-agnostic, debuggable.

### Platform stance

- **As GPU-driven as possible** (indirect dispatch, GPU-side queues, minimal CPU round trips).
- **Push what the web can do.** Chrome has a feature flag for **bindless** resources — we build for bindless first,
  with the core *falling back* to classic bind groups.
- WebNN exists and gives access to NPU/GPU ML; our denoiser will likely be neural (OIDN 2/3 networks).
- Native comparisons and some native tooling will run later on a **Windows machine** (DX12/Vulkan, CUDA/OptiX,
  OIDN native). Flag anything that needs Windows/NVIDIA/AMD hardware.

## Our sibling packages (all in ~/Developer, mostly WebGPU, many headed to pmndrs)

- `bvh` (@pmndrs/bvh) — tinybvh in WASM, GPU LBVH + H-PLOC builders, GPU refit, WGSL ray walk + CPU port, one shared
  GPU layout. **Scope ends at the hit**; shading/sampling/lights/accumulation belong to this path tracer repo.
- `downsampler` — the "SPD" (single-pass downsampler) library: mip chains, reduction pyramids, Hi-Z.
- `Denoiser` — WebGPU denoising (OIDN networks, FidelityFX real-time denoisers, three.js nodes). Ours ran ~30% faster
  than mutracer's OIDN.
- `upscaler` (@pmndrs/upscaler) — FSR1 + FSR2/3-style temporal upscaling as raw WGSL compute.
- `reflections` (@pmndrs/reflections) — probes + SSR + ray-traced fallbacks, hybrid.
- `webgpu-ddgi` (@pmndrs/gi) — WebGPU GI platform, DDGI first; `ddgi-research`, `gi-school`, `gitestbench`.
- `homefig/gputex` — vendored gpuTex: compressed texture arrays, near-streaming textures by writing into them at runtime.
- Also present: `Capsaicin` (AMD GI-1.0 framework, native), `rt-denoiser-sources`, `three-gpu-pathtracer` (old local
  copy), `WebVDB` (NanoVDB on web), `Baker`, `MaterialX`/`MaterialXJS`, `fsr3`, `ssgiDev`.

Defaults: we will use our own BVH and downsampler as the core, and our denoiser + upscaler by default.

## Known external references

- **three-gpu-pathtracer** (gkjohnson, close friend) — three.js only, legacy baggage. Study, compare, don't inherit.
- **mutracer** (https://github.com/pissang/mutracer) — new, WebGPU, very interesting, near-competitor. Code is open:
  read, cite, compare and benchmark against it. **Do not copy code from it and do not use it as our starting point.**
  Notable: during motion it shows a blurry representation instead of classic grain (likely temporal
  accumulation/reprojection behavior) — worth understanding.
- **pbrt-v4** — the usual quality reference; not speed-focused.

## Rules for gathering source

- Clone into `research/sources/web/<name>` or `research/sources/native/<name>` with
  `git clone --depth 1 --single-branch` (no `--recursive`). For huge monorepos (Blender, Unreal, Chromium, Dawn)
  do **not** clone whole: use a sparse checkout of the relevant subdirectory, or record links only.
- **Never run, build or install anything from a cloned repo** (no npm install, no scripts, no cmake). Read only.
- Before cloning, check the directory doesn't already exist (other agents work in parallel). If it exists, reuse it.
- Append one line per clone to `research/sources/MANIFEST.md`:
  `| name | url | commit (git rev-parse --short HEAD) | license | why |` (create the table header if missing;
  use `>>` appends so parallel agents don't clobber each other).
- Open-access papers (arXiv, author/project pages, publisher open access) may be saved as PDFs under
  `research/papers/<your-area>/` with a descriptive filename. No paywalled or pirate sources. Links are always fine.
- Record **licenses**. GPL/LGPL/non-commercial code: read for ideas only, never port; flag it.
- After cloning and before reading, run `node research/tools/provenance-scan.ts <clone> --git` and classify the hits
  (docs/internal/CLEAN-ROOM.md §3). Never open a flagged file to check it; report the hit to the lead.
- **Clean room: research agents never clone, open or summarize NVIDIA NRD code or any third-party NRD port** (e.g.
  Rayzee's), nor `~/Developer/nrd-quarantine`. Rules: `~/Developer/Denoiser/docs/research/rt-denoiser/RULES.md`.
  NRD *is* needed for native runs and comparisons: it lives only with the Windows "harness builder" role
  (`Denoiser/docs/research/rt-denoiser/NATIVE-HARNESS.md`), and our bench consumes its outputs (EXRs, timings) as a
  black box. Other NVIDIA RTX
  SDK repos (RTXPT, RTXDI, RTXGI, SHaRC): ideas only, no code transcription, until their license is reviewed.

## Output format

Write your findings to the notes file named in your task (`research/notes/NN-topic.md`). Structure:

1. **TL;DR** — 5–10 bullets: what matters most for us.
2. **Sources** — table: name, link, year, license (code) / venue (paper), local path if cloned/downloaded, one-line why.
3. **Techniques / findings** — for each: what it is, why it is fast (or what it costs), how it maps to WebGPU/WGSL
   today (and with bindless / subgroups / f16), and a rating:
   `Adopt now` / `Prototype` / `Later` / `Reference only`.
4. **For the spec** — concrete recommendations and decisions we should make.
5. **Bench / comparison ideas** — what to measure and against what.
6. **Needs Windows / native hardware** — list.
7. **Open questions.**

Be concrete: cite file paths and line ranges in cloned code, equations, numbers from papers (ms, Mrays/s, GPU used).
Prefer depth over breadth on the items that matter most to "speed above all". Mark anything you could not verify.
