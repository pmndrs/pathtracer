# VERIFY: P0.8 literature verification pass

Date: 2026-11-13. Literature only: no code run, no clones, no NRD code opened. Tags: **[V]** read at the primary source this
session, **[V-v1]** verified in a source that has since been withdrawn, **[S]** search snippet only, **NV** could not verify.
Not legal advice: the licence summary below is a reading of the texts for engineering decisions.

## 1. NVIDIA licences

Texts read in full: `NVIDIA-RTX/RTXPT/LICENSE.txt` (v. 2023-01-23, 25 KB) and `NVIDIA-RTX/RTXDI/LICENSE.txt`. First part of
`NVIDIA-RTX/RTXGI/License.md` and `NVIDIA-RTX/SHARC/License.md` also read (clauses 1, 2, 3, 4a on). All four are titled "NVIDIA RTX
SDKs LICENSE"; the RTXGI/SHARC copies add "RTX Video SDK" to the product list. The NRD SDK is named in the same text, but NRD's own
repo was not opened (clean room). Sources: https://github.com/NVIDIA-RTX/RTXPT/blob/main/LICENSE.txt,
https://github.com/NVIDIA-RTX/RTXDI/blob/main/LICENSE.txt, https://github.com/NVIDIA-RTX/RTXGI/blob/main/License.md,
https://github.com/NVIDIA-RTX/SHARC/blob/main/License.md.

**Operative terms (cl. numbers from the text):**
- 1 (grant): non-exclusive, non-transferable, no sublicense. (a) install and use the SDK; (b) *modify and create derivative works of
  **sample source code** delivered in the SDK*; (c) distribute SDK software/material, other than internal developer tools, **"as
  incorporated in object code format into a software application"**.
- 2 (distribution): (a) the application must have "material additional functionality" beyond the SDK portions; (b) modified or derivative
  *source* that is distributed must carry "This software contains source code provided by NVIDIA Corporation."; (c) distribute the SDK
  under terms "at least as protective" as this licence (grant, restrictions, IP protection); (d) report known non-compliant
  distribution and enforce the terms downstream.
- 4 (limits): (a) no reverse engineering, no removal of notices; (b) except as granted, no copying, selling, sublicensing,
  distributing, modifying or deriving from any part of the SDK, and no distribution as a stand-alone product; (e) **may not use the SDK in
  a manner that makes it subject to an open-source licence, with examples: source disclosure, licensed for derivative works, or
  redistributable at no charge**; (c) no implied NVIDIA endorsement; (f) no life-critical use.
- 8.1 all rights not expressly granted are reserved (this includes patents: no patent grant appears). 8.2 you own your derivative works
  of the *sample* source. 12: terminates on non-compliance or if you start a legal proceeding against NVIDIA over the SDK.
- Supplement: apps must interoperate with NVIDIA GPUs; the DLSS/NGX-specific terms do not apply to RTXDI/RTXGI/SHARC; marketing clause
  6 covers DLSS/NGX.

**(a) Porting code to WGSL in an MIT library: not allowed.** A WGSL translation of RTXDI/RTXGI/SHaRC/RTXPT source is a derivative work. The
licence only lets you make derivatives of *sample* code (not the SDK core), only distribute SDK material in *object-code* form inside an
application with material additional functionality, requires downstream terms at least as protective as the licence (2c), and
forbids any use that puts the SDK under an open-source licence (4e). Publishing MIT-licensed WGSL source derived from it breaks 1(c),
2(c) and 4(b)/(e) at once. **The "no NVIDIA SDK code" rule is correct and also covers transliteration.**

**(b) Reading and learning from samples: the text does not forbid it.** Cl. 1(a) permits use, and nothing restricts studying ideas. Two
cautions: the licence binds "by using the SDK" (cloning and building counts), and ideas are separate from copyright only if the
re-implementation is written from the paper or from your own notes, not by line-by-line paraphrase. The licence is silent on
patents (8.1 reserves all rights); NVIDIA is known to patent ReSTIR-family methods **[S, NV: not checked in this pass]**, so "papers are
fair game" is a copyright statement, not a patent one. Flag for the maintainer.

**(c) Redistribution:** object code only, inside an application that adds material functionality, under protective terms; never
stand-alone; source forms only for modified *sample* code with the NVIDIA notice. Incompatible with an MIT npm package of WGSL source.

**Does anything contradict or narrow the maintainer's stance** ("no NVIDIA SDK code; examples, books, presentations and papers are fair
game")? The RTX SDKs licence does not narrow it: it is looser than the stance for apps, stricter than MIT for a library, and
says nothing about papers or talks. But "NVIDIA examples" is not uniformly fair game; found while checking:
1. **NVIDIA-RTX/STBN (spatiotemporal blue noise): "NVIDIA Source Code License ... Non-Commercial Use"** (`License.txt`, repo-wide,
   including `Assets/STBN.zip` and the generator). Not MIT-compatible, not usable commercially. Generate our own STBN from the paper
   (Wolfe et al. 2022; arXiv 2112.09629 is saved). Note 04 line 28 now says so. https://github.com/NVIDIA-RTX/STBN/blob/main/License.txt
2. **Ray Tracing Gems chapters are CC BY-NC-ND 4.0** (Springer chapter page for ch. 6). The book is open access, but copying the
   listings verbatim into an MIT library is not covered (NC, ND). Re-derive from the description. Code listings may carry a separate
   licence **[NV: the Apress code repository licence was not checked]**.
3. NVIDIA Research papers carry per-paper licences (ReSTIR PT Enhanced is CC BY 4.0 [V]); the ideas are usable, text and figures need attribution.
4. `NVIDIA-RTX/SHARC`, `RTXGI`, `RTXDI`, `NRC`, `OMM`, `Streamline` are all GitHub-"Other" (the proprietary text). By contrast
   NVRHI, Donut, Donut-Samples, RTXMU, MathLib, NRI, ShaderMake, RTXDI-Assets are MIT (GitHub API). Donut/NVRHI are MIT so
   *portable*, but they are framework glue, not algorithms. Verify per file before porting.

**Falcor** (`NVIDIAGameWorks/Falcor/LICENSE.md`, copyright 2020): verbatim **BSD-3-Clause** text. It then adds that Falcor "links to"
DLSS, RTXGI, RTXDI and NRD, *which are not under this licence*. So Falcor's own files are portable with the notice (INDEX §5's
"BSD-3-like" is confirmed), but any file that wraps RTXGI/RTXDI/NRD or carries a different header is under the RTX SDKs licence:
check the header of each file before porting. https://github.com/NVIDIAGameWorks/Falcor/blob/master/LICENSE.md

**NVlabs/flip** (`LICENSE`): **BSD-3-Clause**, "Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES" with SPDX header. Fine to run,
port and vendor with attribution (THIRD_PARTY.md). https://github.com/NVlabs/flip/blob/main/LICENSE
Note 10 line 30 changed from "(unverified)" to [V].

Other licences checked via the GitHub licence API / LICENSE text on 2026-11-13: pbrt-v4 Apache-2.0 [V]; OpenPGL Apache-2.0 [V];
OIDN Apache-2.0 [V]; MaterialX, OpenPBR Apache-2.0 [V]; Cycles (standalone) Apache-2.0 [V]; tinybvh MIT [V]; ltc-sheen Apache-2.0 [V];
Mitsuba 3 LICENSE is a BSD-3-style text (GitHub reports NOASSERTION; read before porting); erichlof/THREE.js-PathTracing-Renderer is
**CC0-1.0, not MIT** (note 10 line 178 fixed); glTF-Sample-Assets and glTF-Render-Fidelity: no repo-level licence detected, per-asset
licences (CC BY 4.0 for the Wayfair example), **NV**.

## 2. Chrome and WebGPU platform

| Item | Result | Source |
|---|---|---|
| Timestamp quantization | 100 us quantization is **automatically disabled when "WebGPU Developer Features" is enabled**: `chrome://flags/#enable-webgpu-developer-features` (restart). The flag alone does not expose the feature: also need `chrome://flags/#enable-unsafe-webgpu` while `timestamp-query` was experimental, and request `timestamp-query` on the device. Dawn toggle name is `timestamp_quantization`, **enabled by default**; native code disables it via `disabledToggles`. [V] | https://developer.chrome.com/docs/web-platform/webgpu/developer-features ; https://developer.chrome.com/blog/new-in-webgpu-120 |
| Command-line spelling | `--enable-webgpu-developer-features` is a real switch: Chromium's Blink web_tests config uses `--enable-webgpu-developer-features --enable-unsafe-webgpu --enable-dawn-features=allow_unsafe_apis` (older snapshot). Not in public docs. `--disable-dawn-features=timestamp_quantization` follows the Dawn `--enable-dawn-features=` / `--disable-dawn-features=` pattern but I found no document using it for this toggle: unverified. [V / NV] | https://chromium.googlesource.com/codesearch/chromium/src/+/901b47f22c2070757a567c167eefc60dc0f6e149/third_party/blink/web_tests/FlagSpecificConfig |
| Works headless? | **NV.** No document says. The switch is used by Chromium's headless-capable test harness, and our sibling harnesses already launch `--headless=new --enable-unsafe-webgpu` (bvh `tests/gpu/chrome.ts:74`; upscaler CLAUDE.md). Plan: add the switch to the runner and keep the calibration page (empty pass vs known long pass, detect the quantum, log it in run metadata). Do not claim ns timing until the page confirms it. | CLAUDE.md trap stays valid |
| `chromium-experimental-sampling-resource-table` | **Still prefixed and experimental** (Oct 2026): `chromium-experimental-sampling-resource-table` behind `chrome://flags/#enable-unsafe-webgpu`; Windows, Linux, Android; **macOS "coming soon"**; sampled textures + samplers only; heterogeneous tables not implemented. Khronos SIGGRAPH Jul 2026 deck lists bindless as early-to-mid design. No origin trial or ship date found. Matches CLAUDE.md trap and note 06. [V blog; S for deck] | https://toji.dev/2026/10/06/webgpu-bindless.html ; https://github.com/gpuweb/gpuweb/blob/main/proposals/bindless.md |
| `subgroup-matrix` | **NV (no change found).** Survey lists `chromium-experimental-subgroup-matrix` at ~0.04-0.08% of browsers. Chrome 151-152 notes (2026-08-12) cover only `subgroup-size-control` / `@subgroup_size`, `setImmediates` error type, Dawn snorm10-10-10-2. No shipping or origin-trial entry found. Keep as optional Prototype. | https://developer.chrome.com/blog/new-in-webgpu-151-152 ; https://web3dsurvey.com/webgpu/features/chromium-experimental-subgroup-matrix |
| WebNN | **Not shipped.** Chrome origin trial planned M146 (beta Feb 2026, Android excluded), re-planned M147-149, then disabled again for release-blocking issues; no later start found. Edge trial page: closed, expiry 2026-08-11. W3C: CR Snapshot 2026-01-22, CRD 2026-03-27. [V via blink-dev/W3C, S for the last status] | https://groups.google.com/a/chromium.org/g/blink-dev/c/5CWKSChYo98 ; https://www.w3.org/TR/webnn |
| Safari 26 | WebGPU on by default on macOS/iOS/iPadOS/visionOS 26 [V]. **NV: no source lists optional features (subgroups, shader-f16, timestamp-query).** WebKit's WWDC25 post, the gpuweb implementation-status page and MDN give only "26". Detect at runtime; record `adapter.features` in `caps.html`. | https://webkit.org/blog/16993/news-from-wwdc25-web-technology-coming-this-fall-in-safari-26-beta/ |
| 64-bit atomics in WGSL | **Correction to notes 06/08: a proposal exists and is merged.** `atomic_vec2u_min_max`: `atomic<vec2u>` as a u64 surrogate with `atomicStoreMin` / `atomicStoreMax` only, no return value, storage buffers only, compute/fragment. Needs Vulkan int64 atomics, Metal Apple 9 family (not M1), D3D12 SM 6.6; gpuweb spec PR #5610, tracking #5071; added to the spec repo (merge noted in the 2026 W3C WG activity log, gpuweb#11080 for the proposals index). Not found shipped in a browser. No add/CAS/exchange 64-bit atomics are proposed. [V proposal text] | https://github.com/gpuweb/gpuweb/blob/main/proposals/atomic-64-min-max.md |

Chrome per-dispatch cost (arXiv 2604.02344): see §4.

## 3. OIDN 3

Not released as of 2026-11-13: latest GitHub release is v2.5.1 (the listing shows "18 Aug"; 2.5.0 "2 Jun"; 2.4.1 "13 Jan" with CG Channel
confirming 2.4.1 as latest in Jan 2026). **The maintainer's 2026-11-30 is the "3.0 Alpha" milestone due date, not the release.**
Confirmed from the official GitHub milestones (https://github.com/RenderKit/oidn/milestones):
- **3.0 Alpha, due 2026-11-30**: #294 viewport spatial filter, #295 lightmap spatial filter, #296 final-frame spatial filter, #297
  composable AOV support (CPUs x86/ARM64), #298 viewport temporal + upscaler filter, **Intel GPUs only**.
- **3.0 Release, due 2027-03-31**: #299 viewport temporal + upscaler filter, on x86 CPUs with AMX-FP16 and Intel, NVIDIA, AMD and Apple GPUs.
- **3.1 Release, due 2027-06-30**: #300 final-frame temporal filter.
Issue bodies were not readable beyond hardware lines: no input list, model format or weight sizes confirmed on GitHub.
Licence Apache-2.0 for the repo [V, openimagedenoise.org and GitHub]; the 3.0 weights licence is **NV** (current weights ship under
the repo licence).
Press only (CG Channel, 2026-01-30, quoting Intel; https://www.cgchannel.com/2026/01/open-image-denoise-3-will-support-temporal-denoising/):
real-time variant takes previous denoised frame, backward motion vectors and depth; final-quality variant adds future frames and
forward motion vectors; U-Net moves to kernel prediction; supersampling (Blender Conference 2026 talk abstract: "up to 3x"); the
article said "second half of 2026", already missed. Third-party video recap [S]: alpha Oct-Nov, beta Dec, final Q1 2027: consistent with the milestones.
Effect on us: the first alpha will not give a temporal filter outside Intel GPUs; a temporal OIDN 3 for our purposes lands with 3.0
Release (2027-03-31 target). Note 07 line 8 updated; model format (`.tza`?) and aux inputs remain **NV**.

## 4. Tagged numbers

| Number | Where | Result | Source |
|---|---|---|---|
| CWBVH speedup 1.9-2.1x incoherent; geomean 2.24x bounce 1, 2.51x bounce 4 (Maxwell), 1.33x primary | 05 L23-25, 4.2 | **[V]** Table 5 geomean row: 1.33 / 2.24 / 2.51. Pascal Table 4: 2.36 / 2.70 (bounce 1 / 4). Abstract 1.9-2.1x | `papers/gpu-arch/ylitie2017-...pdf` |
| "Pascal Titan X 360 Mrays/s on CrySponza bounce 1" and the "3.5x gap" | 05 L30-32, L370, §8 table | **WRONG label.** 361 is the **Maxwell** Titan X (Table 5). Pascal CWBVH on CrySponza is **1163 primary / 658 bounce 1 / 557 bounce 4** (baseline 918/276/187). Gap vs our 104 Mrays/s is 3.5x against Maxwell, 6.3x against Pascal. Maxwell absolute rows in 4.2 (616/361/346 etc.) are correct. Changes the headroom story, not the tiers. | same |
| Laine 2013: +36% / +79% / +221% | 05 L16, §3 | **[V]** Table 1: Carpaint 42.99 -> 58.38 (36%), City 5.41 -> 9.70 (79%), Conference 2.71 -> 8.71 (221%) Mpaths/s | `laine2013-...pdf` |
| 104 Mrays/s (3060L), 12.8 (UHD), 1.20x / 1.97x vs three-mesh-bvh | 05 §8 | **[V local]** `~/Developer/bvh/README.md:109`, `docs/internals/design/trace.md:190` (median of 3) | local |
| Per-GPU Mrays/s for 4080/4090, Apple base, Xe-LPG, mobile | 05 §8 | still **[E]**; no external source exists; leave as hypotheses (factor 2) | none |
| Chrome per-dispatch 33 / 59 / 67 us | 05 L19, L361; 06 L35 | arXiv 2604.02344 is **withdrawn and superseded by 2608.08730** (v3 2026-10-26). v1 table [V-v1]: Chrome RTX 5090 Linux 32.8 us, RTX 2000 D3D12 58.7, Intel iGPU D3D12 66.5, Safari M2 31.7, Firefox ~1040 (rate limiting), single-op sync numbers ~20x inflated. v3 text repeats 24-36 us Vulkan, 32-71 us Metal; its per-browser table was not readable. The 06 note's "24-58 us" is a paraphrase of this. Use as a range; measure ourselves. Also: submit is ~40% of CPU cost, one platform for the fusion study. | https://arxiv.org/abs/2604.02344 , https://arxiv.org/abs/2608.08730 |
| Lumen High 1/16 rpp 2.77 ms, Epic 1/4 rpp 4.63 ms (PS5, 1080p) | 08 L10, L163 | **[V]** exact in the saved course notes. Correction: these are **software** Global-SDF traces, not hardware RT (note called them "hardware RT"). Final Gather 2.07 / 3.46 ms | `lumen-siggraph2022-wright.pdf` |
| GI-1.0 1.9-3.4 ms total, 0.13-0.68 ms DXR | 08 L163 | **[V]** Table 1 (RX 6900 XT / RTX 3080) | `gi-1.0-gpuopen-2022.pdf` |
| DDGI 6 ms full GI (2080 Ti 1080p); ray counts | 08 L59, L163 | **[V]** 6 ms is Fig.1 total; indirect passes 2.6 ms (Table 2); 32x8x32 probes x 64 rays used for largest scenes = 524k rays/frame (0.25 ray/pixel at 1080p); throughput 1.5-1.65 GRays/s on 32-256 rays/probe. "RTXGI default 256, up to 512" stays **NV** (needs SDK docs). | `majercik-ddgi-jcgt-2019.pdf` |
| ReSTIR GI "~2-4 ms at 1080p on 3090, 1 spp" | 08 L112 | **WRONG (too low).** [V] RTX 3080, 1920x1050: initial sampling 3.2 ms (1 bounce) / 4.2 ms (2 bounce) + reuse 4.6 ms, total frame 14 ms. Fig.1 (3090, 1080p): ReSTIR GI 8.9 ms vs plain 1 spp PT 8.0 ms. Reservoir (sample) 48 B **[V]**. "1-3 extra visibility rays" remains **NV**. | `restir-gi-ouyang-2021.pdf` (the file in the repo was an HTML page; replaced by the real PDF) |
| ReSTIR PT Enhanced 37.1 -> 12.6 ms (Fig.1), 2-3x, 88 -> 64 B reservoir, 3.7-6.2 ms sub-passes; baseline 40-80 ms on 3090 | 04 L11, L101 | **[V]**. GPU is an **RTX 5880 Ada at 1080p** (note had none). Other scenes 30.1 -> 14.5, 47.1 -> 21.4, 50.0 -> 16.4 ms. Storage is 2 x (88+16) B -> 2 x 64 B | `lin2026-restir-pt-enhanced.pdf` |
| Spectral overhead "10-30% (estimate)" | 09 L199 | **Replace:** Peters measured 2-36% (RTX 5070 Ti, 4 wavelengths, 1080p, 1 spp, locked clocks): Cornell +19% / +36%, Bistro +7% / +2%, <= 0.3 ms absolute | https://momentsingraphics.de/SpectralRendering2Rendering.html |
| STBN 128x128x64 shipped sizes, "4 MB" | 04 L75 | still **[M]** (could not read the repo zip); **licence is non-commercial** (§1) | https://github.com/NVIDIA-RTX/STBN |
| SHaRC "MIT-style" licence | 04 L38 | **WRONG:** NVIDIA RTX SDKs licence (§1). "2^22 entries default" and "4% of pixels" stay [S]/NV | github |
| FLIP BSD-3 | 10 L30 | **[V]** | §1 |
| OpenPBR 1.1 "~2025" | 09 L74 | **WRONG:** 1.1 = 2024-06-28; 1.1.1 = 2026-04-17; 1.1.2 = 2026-11-05 | CHANGELOG.md |

Still unverified after this pass: ReSTIR PG and BDPT claims [S], Area ReSTIR, NASG/neural guiding [S]; SHaRC internals; Mallett-Yuksel
numbers; Hairball/PipersAlley outliers in Ylitie (not needed); Bikker's 5 Grays/s (README, ray type unknown); all per-scene
triangle counts and scene licences in note 10 (not in scope).

## 5. Papers saved (all first pages checked with `pdftotext -f 1 -l 1`)

`research/papers/` is git-ignored; files are local only.

| Paper | File | Source |
|---|---|---|
| EARS, Rath et al. 2022 (not on arXiv; 2205.04435 was wrong) | `sampling/rath2022-ears-efficiency-aware-russian-roulette-splitting.pdf` (32 MB) | https://graphics.cg.uni-saarland.de/papers/rath-2022-ears.pdf |
| Wolfe et al., "Scalar Spatiotemporal Blue Noise Masks" (the arXiv preprint of the STBN work; EGSR 2022 version has a different title) | `sampling/wolfe2021-scalar-spatiotemporal-blue-noise-masks-arxiv2112.09629.pdf` | https://arxiv.org/abs/2112.09629 |
| Heitz et al. 2019, low-discrepancy sampler / blue noise in screen space | `sampling/heitz2019-low-discrepancy-sampler-blue-noise-screen-space.pdf` | https://perso.liris.cnrs.fr/david.coeurjolly/publication/heitz-19/ |
| ReSTIR GI, Ouyang et al. 2021 (replaces an HTML page mis-saved as PDF) | `gi/restir-gi-ouyang-2021.pdf` (165 MB) | NVIDIA research CDN |
| ReSTIR PT Enhanced, Lin et al. 2026 (was link-only) | `sampling/lin2026-restir-pt-enhanced.pdf` (126 MB) | https://research.nvidia.com/labs/rtr/publication/lin2026restirptenhanced/ |
| Wilkie et al. 2014, Hero Wavelength Spectral Sampling | `transport/wilkie2014-hero-wavelength-spectral-sampling.pdf` | https://cgg.mff.cuni.cz/~wilkie/Website/EGSR_14.html |
| Wilkie et al. 2021, fitted radiance and attenuation model (Prague sky) | `transport/wilkie2021-fitted-radiance-attenuation-model-atmospheres.pdf` | https://cgg.mff.cuni.cz/publications/skymodel-2021/ |
| Zeltner, Burley, Chiang 2022, LTC sheen | `transport/zeltner2022-practical-multiple-scattering-sheen-ltc.pdf` | https://tizianzeltner.com/projects/Zeltner2022Practical/ |
| Belcour 2018, layered materials, atomic decomposition | `transport/belcour2018-layered-materials-atomic-decomposition-statistical-operators.pdf` | HAL hal-01785457 (HAL's bot check blocked direct download; fetched the Internet Archive copy of the same HAL file) |
| Belcour & Barla 2017, varying iridescence / thin film | `transport/belcour-barla2017-microfacet-varying-iridescence-thin-film.pdf` | HAL hal-01518344, same route |
| Weier & Belcour 2020, layered materials with anisotropic interfaces (extra, JCGT) | `transport/weier-belcour2020-layered-materials-anisotropic-interfaces-jcgt.pdf` | https://jcgt.org/published/0009/02/03/paper.pdf |
| Kaplanyan & Dachsbacher 2013, path space regularization | `transport/kaplanyan2013-path-space-regularization.pdf` | https://cg.ivd.kit.edu/publications/p2013/PSR_Kaplanyan_2013/PSR_Kaplanyan_2013.pdf |
| Wächter & Binder 2019, RTG ch. 6, self-intersection (pp. 113-121 cut from the open-access book PDF; Springer's own download sits behind a bot check) | `gpu-arch/wachter-binder2019-rtg-ch6-avoiding-self-intersection.pdf` | book: http://raytracinggems.com/unofficial_RayTracingGems_v1.9.pdf (the editors' site), chapter licence CC BY-NC-ND 4.0 per https://doi.org/10.1007/978-1-4842-4427-2_6 |
| EON, Portsmouth/Kutz/Hill 2025 (extra) | `transport/portsmouth2025-eon-energy-preserving-rough-diffuse-arxiv2410.18026.pdf` | https://arxiv.org/abs/2410.18026 |
| OpenPBR companion paper 2026 (extra) | `transport/portsmouth2026-openpbr-novel-features-implementation-arxiv2512.23696.pdf` | https://arxiv.org/abs/2512.23696 |

HAL and Springer both serve a bot challenge to scripted downloads; I did not try to bypass it. The two Belcour PDFs came from the
Internet Archive snapshot of the HAL document URL. The Wächter-Binder pages come from a copy of the open-access book hosted by its editors.

## 6. Notes changed

| File | Lines (after edit) | Change |
|---|---|---|
| `notes/04-sampling-reuse.md` | 28, 32, 38, 75-97 area, 101, 272 | STBN licence and arXiv file; EARS source and file; SHaRC licence; ReSTIR GI cost and reservoir [V]; ReSTIR PT Enhanced GPU/res [V]; open-item 6 |
| `notes/05-gpu-architecture.md` | 19-25, 30-32, 370, 431, 436 | withdrawn dispatch paper; Ylitie Pascal vs Maxwell correction, 658 Mrays/s; geomean table cites |
| `notes/06-web-platform.md` | 12, 15, 35, 69 | timestamp switch spelling; WebNN status [V]; dispatch paper withdrawn; 64-bit atomics proposal exists |
| `notes/07-denoise-neural.md` | 8 | OIDN 3 milestones: 11-30 is the alpha |
| `notes/08-realtime-gi.md` | 59, 112, 163, 261-262 | DDGI rays [V]; ReSTIR GI numbers corrected; Lumen is software RT; 64-bit atomics; subgroup status |
| `notes/09-materials-transport.md` | 74, 199, 283, 289-290, new section 8 | OpenPBR versions; spectral overhead [V]; failed downloads resolved; "2024-26 update" appended |
| `notes/10-benchmarks.md` | 11, 30, 151, 178 | flag spellings, FLIP licence, erichlof licence |

Not touched (as instructed): TASKS.md, STATUS.md, CLAUDE.md, DECISIONS.md, DIVERGES.md, research/INDEX.md, spec/.
For the maintainer to carry over: INDEX §4 backlog items now resolved are the licence question, OIDN 3 date, Chrome flags,
64-bit atomics, WebNN status and the missing papers; CLAUDE.md trap on timestamps stays right but should add the exact switch once
the calibration page confirms it headless.
