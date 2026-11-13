# 10 - Benchmarks, reference scenes, metrics, comparison methodology

Date: 2026-11-13. Scope: how we test and compare a WebGPU path tracer against web competitors (mutracer, three-gpu-pathtracer, others) and native references (pbrt-v4, Mitsuba 3, Cycles, Falcor/RTXPT).
Status of facts: sibling-repo findings were read directly from disk (paths cited). External facts (scene licenses, tool behaviour, Chrome flags) are from my background knowledge and were NOT re-verified online in this pass; each is marked (unverified) where it matters. Verify before relying on a license.

## 1. TL;DR

- **Compare in linear HDR, never in tonemapped 8-bit.** Our own `ddgi-research/comparisons/linear-format.md` + `scripts/compare-linear.py` already do native-vs-web comparison this way; reuse the idea. Every renderer dumps an EXR/PFM of radiance (pre-tonemap, pre-exposure, fixed sensor/camera model); the comparer applies one shared tonemap only for pictures.
- **The primary quality contract is "converged equals pbrt-v4/Mitsuba", measured as relMSE/FLIP against a cached high-spp reference** on a small set of analytic and Bitterli scenes. Furnace + Cornell + glTF material-extension scenes give correctness; Sponza/Bistro/Emerald give speed under load; Moana is a scale/stress sidecar only.
- **Primary speed metric is "time to equal quality" (error vs wall-clock, equal-error crossing), not Mrays/s.** Mrays/s is reported only as a per-kernel diagnostic, with ray type, coherence and counting rules stated. Efficiency = 1/(variance x time).
- **GPU timing on web is the hard part.** Chrome quantizes `timestamp-query` to ~100 us by default; precise timestamps need developer features: `chrome://flags/#enable-webgpu-developer-features` [V Chrome docs, "quantization is automatically disabled when the flag is enabled"], command line `--enable-webgpu-developer-features` [V in Chromium Blink web_tests FlagSpecificConfig, alongside `--enable-unsafe-webgpu` and `--enable-dawn-features=allow_unsafe_apis`; not in public docs]; the Dawn toggle is `timestamp_quantization` (default enabled) so `--disable-dawn-features=timestamp_quantization` is the plausible equivalent [toggle name V in Chrome 120 blog; CLI form unverified]. Always run the calibration page. Siblings already built timer-overhead and "validity" gating (thermal/clock/idle). Copy it.
- **Siblings give us ~70% of the harness.** `webgpu-ddgi/tools/gpu/chrome.ts` (Chrome launch, LUID pinning), `bench/results/*.csv` schema (date, commit, device, browser, scene, metric, value; append-only; 3 runs + median; INVALID flags), `Denoiser/tools/scorecard` (temporal metrics, FLIP, gpu-lock), `Denoiser/tools/eval` (native OIDN vs web), `bvh/compare` (adapter interface, agreement checking, interleaved timing), `materialx-fidelity` (renderer-package pattern + viewer website).
- **Windows is required for**: NVIDIA/AMD timing on DX12/Vulkan, native pbrt-v4 (GPU/OptiX build), Mitsuba 3 CUDA/OptiX variant, Falcor/RTXPT, native OIDN GPU, Cycles OptiX/HIP. Mac (M-series) covers pbrt-v4 CPU, Mitsuba `llvm_ad`/scalar CPU variants, Cycles CPU/Metal, and all web runs on Metal.
- **A web path tracer that wants a fair mutracer/three-gpu-pathtracer comparison must fix the camera, resolution, bounce count, light setup, sample count/time budget and tonemap, and disable the denoiser on both sides** for the raw-sample comparison, then enable each side's denoiser for the second table.

## 2. Sources

| Name | Link | Year | License | Local | Why |
|---|---|---|---|---|---|
| Benedikt Bitterli rendering resources | https://benedikt-bitterli.me/resources/ | 2016 | per-scene CC (mostly CC BY 3.0 / CC0; check each page) (unverified per scene) | not cloned (large downloads) | Canonical cross-renderer scenes: pbrt-v3/v4, Mitsuba, Tungsten formats with reference images |
| McGuire Computer Graphics Archive | https://casual-effects.com/data/ | 2017 | per-scene; mixed CC BY, some non-commercial (unverified) | links only | Sponza, Bistro (older), Dragon, Buddha, San Miguel, Rungholt, Conference |
| Intel Sponza (new, "Atrium") | https://www.intel.com/content/www/us/en/developer/topic-technology/graphics-research/samples.html | 2022 | CC BY 4.0 (base) (unverified) | not cloned | Modern PBR textured Sponza; sibling `bvh/webgpu-ddgi` already use Sponza |
| Amazon Lumberyard Bistro | https://developer.nvidia.com/orca/amazon-lumberyard-bistro | 2017 | CC BY 4.0 (unverified) | links only | Many lights/emissives, foliage alpha, interior+exterior; webgpu-ddgi already has "Intel Bistro" |
| NVIDIA ORCA | https://developer.nvidia.com/orca | 2017+ | per-scene (Emerald Square, Zero Day: CC BY-NC variants for some, check) (unverified) | links only | Large stress scenes (Emerald Square, Zero Day = animated); license flag |
| Disney Moana Island | https://www.disneyanimation.com/resources/moana-island-scene/ | 2018 | CC BY-SA 3.0 (data) (unverified) | not cloned (~50 GB pbrt) | Scale test: instancing, ptex, memory. Not realistic for WebGPU; use for "does it load / what is the cap" sidecar only |
| Khronos glTF-Sample-Assets | https://github.com/KhronosGroup/glTF-Sample-Assets | 2023- (repo; replaced glTF-Sample-Models) | per-asset CC/Apache; repo Apache-2.0-ish (unverified) | not cloned | Material-extension tests: `KHR_materials_*` (transmission, volume, clearcoat, sheen, iridescence, anisotropy, ior, specular, emissive_strength, dispersion, diffuse_transmission), alpha modes, texture transform, many-lights |
| glTF-Render-Fidelity | https://github.com/KhronosGroup/glTF-Render-Fidelity | 2023- | Apache-2.0 (unverified) | not cloned | Khronos harness comparing glTF viewers/renderers side by side on the sample assets; pattern for our web-vs-web table; includes path-traced references (Blender Cycles / Dassault Enterprise PBR) (details unverified) |
| pbrt-v4-scenes | https://github.com/mmp/pbrt-v4-scenes | 2021+ | per-scene (mixed CC; check) | not cloned | Ready pbrt-v4 reference scenes (Barcelona pavilion, Sanmiguel, Zero Day, Villa, Kitchen, Cornell, Sportscar, Hair...), incl. `pbrt-v4` files we can feed pbrt directly |
| Mitsuba 3 test scenes | https://github.com/mitsuba-renderer/mitsuba3 (`resources/data/scenes`, `src/*/tests`) and mitsuba-renderer/mitsuba-data | 2022+ | BSD-3-Clause (code), per-scene data (unverified) | not cloned | Small unit-style scenes with known expected behaviour; spectral/polarized tests |
| NVIDIA FLIP | https://github.com/NVlabs/flip | 2020; HDR-FLIP 2021 | BSD-3-Clause [V LICENSE text 2020-2025] | not cloned | Perceptual error metric; python + C++ + CUDA |
| pbrt-v4 | https://github.com/mmp/pbrt-v4 | 2023 | Apache-2.0 | not cloned (other agents) | Primary quality reference |
| Mitsuba 3 | https://github.com/mitsuba-renderer/mitsuba3 | 2022 | BSD-3-Clause | not cloned | Second reference, scriptable (Python), spectral variants |
| mutracer | https://github.com/pissang/mutracer | 2026 | see `sources/MANIFEST.md` | check `research/sources/web/` | Web competitor; read-only |
| three-gpu-pathtracer | local `~/Developer/three-gpu-pathtracer`; upstream gkjohnson | | MIT | local | Web competitor; has `example/` scenes and a visual regression set (unverified) |
| Sibling harnesses | see section 7 | 2026 | ours | local | Reuse |

Per the brief, no scene repos were cloned: scene data is large, and licenses must be checked before vendoring. Manifest should store URL + SHA-256 + license, not the bytes (section 8).

## 3. Scene library: what stresses what

Suggested tiers. "Fmt" = what exists natively; our canonical input is glTF 2.0 (+extensions), with converters to pbrt-v4 and Mitsuba (section 6).

### Tier 0 - analytic/correctness (tiny, run in CI)

| Scene | Tests | Expected answer |
|---|---|---|
| White furnace: sphere (and a rough conductor/dielectric/sheen/clearcoat variants), uniform env L=1, albedo 1 (or 0.5 enclosed emissive box, as `gitestbench` does: wall radiance 1, albedo 0.5 -> converge to 2) | Energy conservation of every BSDF lobe, MIS weights, Russian roulette unbiasedness, env sampling | Pixel = 1 (or analytic) everywhere; deviation = bug. Pass bar e.g. max relative error <= 0.5% at N spp. Existing precedent: `gitestbench/TESTING.md` test 1; `ddgi-research` analytic E=pi*L tests with 1% tolerance |
| Furnace with white-but-rough dielectric, multiple-scattering on/off | Whether we implement multiscatter energy compensation; expected: single-scatter loses energy at high roughness, compensated version ~ 1 | Document which is expected per material model |
| Cornell box (Bitterli `cornell-box`, also the classic with spheres) | Color bleeding, area light sampling, GI | Compare to pbrt/Mitsuba converged |
| Quad light + diffuse plane | Next event estimation correctness (analytic form-factor) | Analytic irradiance |
| Environment-lit sphere on plane (HDRI) | Env importance sampling, MIS | Converged compare |
| Texture/UV checker, alpha test, normal map plane | Texture sampling parity, alpha modes, tangent frame | Pixel-exact vs reference at 1 spp primary hit |
| Null-bounce/volume cube (homogeneous medium, absorbing + scattering) | Volume transmittance, analytic Beer-Lambert for absorbing-only | Analytic transmittance |

### Tier 1 - material feature scenes

- **glTF-Sample-Assets** (names as I recall; verify against the repo): `MetalRoughSpheres`, `MetalRoughSpheresTextured`, `NormalTangentTest`, `TextureSettingsTest`, `AlphaBlendModeTest`, `TextureTransformTest`, `TransmissionTest`, `TransmissionRoughnessTest`, `ClearCoatTest`, `ClearCoatCarPaint`, `SheenChair`, `SheenCloth`, `IridescenceLamp`, `IridescenceAbalone`, `IridescentDishWithOlives`, `AnisotropyBarnLamp`, `AnisotropyDiscTest`, `AnisotropyRotationTest`, `StainedGlassLamp`, `GlamVelvetSofa`, `DiffuseTransmissionTeacup`, `CommercialRefrigerator`, `DragonAttenuation`, `AttenuationTest`, `IORTestGrid`, `SpecularTest`, `EmissiveStrengthTest`, `DispersionTest`, `LightsPunctualLamp`, `MaterialsVariantsShoe`, `MultiUVTest`, `TextureLinearInterpolationTest`, `UnlitTest`, `CompareAmbientOcclusion`, `CompareBaseColor` ... (the `Compare*` and `*Test` assets are designed for fidelity checking).
- Gives: per-extension pass/fail, a "feature coverage matrix" (renderer x extension x ok/partial/absent) as deliverable, which doubles as a marketing table vs. three-gpu-pathtracer and mutracer.
- **Khronos glTF-Render-Fidelity**: use its list of models and the style of its results page (side-by-side image grid + difference); our output becomes another "renderer" in that grid. Worth checking whether it accepts external renderers, and which references (Cycles? Enterprise PBR?) it ships; unverified.
- Also relevant: our own `materialx-fidelity` monorepo (MaterialX samples through renderers, viewer website). Same architecture; MaterialX standard-surface vs glTF PBR mapping is a separate fidelity axis.

### Tier 2 - realistic, medium

| Scene | Fmt | Stresses |
|---|---|---|
| Bitterli Veach MIS (`veach-mis`), Veach Bidir (`veach-bidir`) | pbrt/Mitsuba/Tungsten | Light sampling vs BSDF sampling, glossy highlight lobes, NEE (variance at small lights); bidir = caustics-from-small-light, needs path-space care |
| Bitterli Staircase, Bathroom, Kitchen (`kitchen`), Living Room (`living-room`, `living-room-2/3`), Bedroom, Dining Room, Classroom, Salle de bain | pbrt/Mitsuba/Tungsten | Indoor GI, small window light, many materials, textures; "hard" light transport (indirect-dominated) -> neural/ReSTIR/radiance caching benchmarks |
| Bitterli Glass of Water (`glass-of-water`), Water Caustic (`water-caustic`), Ring (`ring`) | | Caustics, dielectric nesting, specular chains; expected hard for unidirectional PT. Good "known-hard" negative set |
| Bitterli Dragon/Hair/Curly Hair (`curly-hair`), Teapot Full, Coffee, Spaceship, Car/Car2, Bunny, Material Testball, Hairball | | Geometry variety, hair (curve primitives: not in glTF), large flat BVH |
| Bitterli Volumetric Caustic, Fireplace/Hair | | Volumes, participating media (needs medium support) |
| Furnished room (already used in `bvh/compare`) and Two-Room (`ddgi-research`, `gitestbench`) | ours | Doorway light leak, sealed rooms; GI correctness tests; direct reuse |

### Tier 3 - large / speed under load

| Scene | Triangles (approx; unverified) | Stresses |
|---|---|---|
| Intel Sponza + curtains, ivy | ~1.0M (base) / 5.7M merged with curtains (bvh/compare measured 5,744,384) | Texture memory, alpha-tested foliage (ivy), instancing; indoor; the DDGI/bvh siblings already load it |
| Amazon Lumberyard Bistro (interior + exterior) | ~2.8M | Many emissive lights (hundreds), alpha-tested foliage, glass; light sampling strategy test |
| NVIDIA Emerald Square | ~10M | BVH memory, tree foliage, WebGPU buffer limits (maxStorageBufferBindingSize), streaming |
| Zero Day (ORCA, animated) | ~10M per frame | BLAS refit/rebuild per frame (refit is core to our `bvh` package) |
| San Miguel, Rungholt, Conference (McGuire) | 5-10M | Classical BVH quality comparisons (SAH cost, traversal steps) |
| Powerplant, Dragon-Buddha (Stanford) | up to 12M | Single dense meshes; BVH build time; coherence |
| Disney Moana Island | ~15B instanced triangles; ~29 GB-ish (unverified) | Cannot be loaded into a browser. Use a subset (e.g. one instance set, "beach" camera) only to track the practical scene-size ceiling; treat as a "can't, here is why" data point |

### What stresses what (matrix)

| Stress | Scenes |
|---|---|
| BVH size/quality | Emerald, San Miguel, Zero Day, Powerplant, Moana subset |
| BVH build/refit speed | Zero Day (refit), any dynamic scene, Sponza rebuild |
| Many lights / light sampling | Bistro (interior), Zero Day, `living-room`, synthetic grid of 1k-100k emissive triangles (we must author this: no standard one) |
| Caustics / specular paths | `glass-of-water`, `water-caustic`, `ring`, glTF `TransmissionTest`/`DragonAttenuation` |
| Volumes | Bitterli volumetric, glTF `KHR_materials_volume` assets, homemade homogeneous cube; NanoVDB (sibling `WebVDB`) for heterogeneous |
| Textures / memory | Sponza (4k PBR), Bistro, Emerald, Moana; test bindless vs bind groups; compressed textures via `homefig/gputex` |
| Alpha-tested geometry | Sponza ivy, Bistro foliage, Emerald trees: any-hit cost |
| Sheen/clearcoat/iridescence/aniso | Specific glTF extension assets |
| Hair/curves | Bitterli hair (not glTF) - Reference only unless we add curve primitives |
| Difficult GI | Staircase, bathroom, kitchen, Two-Room, dark room with doorway |

## 4. Metrics

### 4.1 Image quality vs a reference (all on linear HDR)

| Metric | Formula / notes | When to use | Caveats |
|---|---|---|---|
| MSE / RMSE | mean((x-r)^2) over pixel channels | Basic | Dominated by fireflies/bright regions; useless on HDR without clamping |
| **relMSE** | mean((x-r)^2 / (r^2 + eps)), eps ~ 1e-2 | Rendering-literature standard (Rousselle/Rathinam; used by the Mitsuba and denoiser papers); good for convergence curves | Sensitive to eps; fix and record it. Compute on per-pixel mean over frame buffers, clamp or use trimmed mean to avoid a single firefly |
| **SMAPE** | mean(abs(x-r) / (abs(x)+abs(r)+eps)) | Robust to outliers; used in OIDN/denoiser training/evals | Not variance-weighted; eps; report with relMSE |
| **FLIP / HDR-FLIP** | NVIDIA ꟻLIP; LDR-FLIP on tonemapped; HDR-FLIP computes over exposure range (start/stop exposure, `--tone-mapper`) | Perceptual "does it look equal"; what artists see | Needs pixels-per-degree assumption (default 0.7 deg/px-ish, 67 PPD on 4K at 0.7 m; set explicitly and record); python (`pip install flip-evaluator`) is easy, cross-platform |
| SSIM / MS-SSIM | structural similarity | Secondary, for denoiser/upscaler outputs | Inflated on noisy-but-smooth; tonemapped LDR only |
| PSNR | | Secondary only | Same as MSE |
| Mean bias (signed) | mean(x - r) per channel and in regions | **Unbiased/consistent mode check.** Distinguishes "noisy" from "wrong". Do this with high-spp web output vs reference | Requires converged web output (>= 10k spp or enough per-pixel averaging) |
| Energy/furnace residual | max abs(pixel - analytic) | Tier 0 | Needs analytic scenes |

Practice: report relMSE + FLIP-LDR (+ HDR-FLIP for HDR-relevant scenes) and a bias map. Always show the difference/FLIP heatmap images, not only the numbers (siblings: `ddgi-research` writes JSON + difference heatmaps; `webgpu-ddgi` keeps CSV).

Existing code: `Denoiser/examples/realtime-denoiser/src/flip.ts` (a TypeScript FLIP port, used by the scorecard with `image spans 46.4 degrees`); `Denoiser/tools/eval/native.py` and `oidn-native-compare/compare.py` (PSNR + PNGs). We should extract a shared metrics module (Node + Python) rather than reimplement.

### 4.2 Convergence and efficiency

- **Convergence curves**: error (relMSE/FLIP) vs spp AND vs wall time, log-log, on a fixed set of checkpoints (1,2,4,...,4096 spp; or time-sampled 0.1 s ... 60 s). For an unbiased estimator with no denoiser, relMSE ~ 1/N slope -1 on log-log; deviations show correlation (QMC beats it: slope between -1 and -2, e.g. Owen-scrambled Sobol / blue-noise dithering gets there only for low-dimension integrands) or bias floors (plateau => bias, e.g. clamping, light leaks, wrong shading normals).
- **Time-to-equal-quality (TTEQ)**: choose a target (e.g. relMSE = 1e-2 or FLIP mean = 0.05 against the reference), report the wall-clock seconds each renderer needs. Where a renderer never reaches it (bias floor), report "never, floor = X". This is *the* comparison figure for "faster than any other web path tracer".
- **Efficiency**: eps = 1 / (variance x time), same as Veach / pbrt book Ch. 2; variance from independent runs (>= 16) at a fixed spp or time, per-pixel across runs, then the mean over pixels (relative variance preferred). Report speedup = eps_ours / eps_other. Use *same scene, same bounces, same light/BSDF set*.
- Variance measurement protocol: N independent runs with different seeds (>= 16 for stable estimates; 64 for publication-grade), compute per-pixel sample variance of the mean. Avoid estimating variance from a single image by comparing against the reference; that confuses bias and variance (use MSE = variance + bias^2 decomposition, report both).
- Fireflies: report the 99.9th percentile pixel value / count of pixels > k x reference, since mean metrics hide them.

### 4.3 Real-time/biased modes (denoised, reprojected, ReSTIR)

- **Static-camera convergence** with the denoiser enabled: error vs frame count.
- **Motion**: replay a camera/object path, compare each frame to a per-frame high-spp reference (frames rendered offline with the same animation). Metrics: per-frame FLIP, relMSE, then mean/p95 over the sequence.
- **Temporal stability / flicker / "boil"**: existing in `Denoiser/tools/scorecard/lib/metrics.mjs`: boil = RMS of the second temporal difference `x_t - 2 x_(t-1) + x_(t-2)` along reprojected history / sqrt(6), measured on tonemapped values, per region (moving / disoccluded / static), relative to the clean reference's own boil; plus "still sigma" = per-pixel temporal sigma of luma while static. Also `lag` for time to respond to light changes. Reuse these exact definitions, so denoiser and path tracer reports are comparable. `webgpu-ddgi/bench/results/flicker.csv` has a simpler flicker measurement.
- **Temporal FLIP**: FLIP on each frame plus frame-to-frame FLIP; the NVIDIA FLIP repo also has video-oriented usage; unverified. Frame-difference variance is an easy proxy.
- **Ghosting/lag**: sudden light on/off or object-move event, measure frames until error < threshold.
- mutracer-style "blurry representation during motion": measure explicitly: same motion path, report FLIP vs per-frame reference and the sharpness (gradient energy) of the output; this is a *design* difference (temporal accumulation visible blur vs noise), so the table must have "noise" and "blur" columns, not one number.

### 4.4 Speed

| Metric | Definition | Pitfalls |
|---|---|---|
| Frame time at fixed spp/res | GPU ms (timestamps) and CPU submit ms, p50/p95/p99, min 200 frames after 60 warm-up | Include first-frame compile cost separately (pipeline creation; `webgpu-tsl-compile` skill shows Windows D3D12/DXC compile can take seconds) |
| Samples/s (spp x pixels / s) | Our and competitors' *comparable* metric: path samples with identical max-bounce/RR/NEE | Not comparable if their path is shorter (RR, bounce cap) |
| **Mrays/s** | rays (counted by the shader with an atomic or counter, or computed as pixels x spp x average path length x (1 + shadow ray)) / GPU seconds | Pitfalls: (1) primary rays are coherent and ~5-10x faster than incoherent secondary, so a Mrays/s number is only meaningful when stating the ray type mix; (2) shadow/any-hit rays are cheaper than closest-hit: count separately; (3) counting rays with atomics perturbs the timing: use a counted run and a timed run separately; (4) high Mrays/s with a bad sampler is useless; (5) different scenes/BVHs not comparable; (6) wavefront vs megakernel differences (rays/s of the traversal kernel in isolation vs end-to-end including shading/sort); (7) bounce count, RR, alpha-test any-hit loops distort counts |
| Traversal kernel Mrays/s | `bvh` package already has `bench/trace.csv` and `compare/gpu`: closest-hit on fixed seeded ray sets (`outside` etc.) | Keep it the diagnostic for the BVH, not the headline |
| Memory | peak GPU buffer bytes by category: BVH nodes, triangles, materials, textures, accumulation/G-buffers; `device.limits` hit; `bvh/compare/node/memory.ts` pattern | WebGPU does not expose VRAM use: sum our own allocations; Chrome task manager / `chrome://gpu` / `performance.measureUserAgentSpecificMemory` for sanity |
| Load/build time | glTF parse, BVH build, texture upload, first pixel, "time to first converged frame" | Include pipeline compilation, cold vs warm cache (reload) |
| Power/energy | Windows: HWiNFO/NVML board power; Mac: `powermetrics`; report J per converged frame | Optional |

### 4.5 Timestamp and timing methodology on WebGPU

- `timestamp-query` feature: request at `requestDevice`; use `GPUQuerySet` + `timestampWrites` on compute/render passes (per-pass), not between passes.
- **Quantization**: Chrome rounds timestamps (~100 microseconds, to mitigate timing side channels) unless developer features/flag enabled. Options (spellings checked 2026-11-13, see research/VERIFY.md §2: `--enable-webgpu-developer-features` is confirmed in Chromium's test config, the Dawn-toggle CLI form is not): launch with `--enable-webgpu-developer-features` and/or `--disable-dawn-features=timestamp_quantization`, plus `--enable-unsafe-webgpu`. In the Dawn toggle list the toggle is `timestamp_quantization`; verify with a calibration page that submits an empty pass and a known long pass and checks the delta granularity. Sibling `upscaler/bench/timer-overhead.html` already measures timer overhead: adopt it as the calibration step at the start of every bench session, record the detected quantum in the results CSV metadata.
- Without precise timestamps: time N back-to-back dispatches and divide (amortize the quantum), then `device.queue.onSubmittedWorkDone()` wall-clock; the quantization error is then <= quantum/N. For a 5 ms frame and 100 us quantum run >= 50 frames per sample.
- On Apple (Metal) timestamps have per-pass granularity only (no mid-pass), and the `timestamp-query` encoder-level support differs; report that tile-based GPUs measure per-pass, include the whole-frame sum.
- Use only GPU timestamps for GPU work (not `performance.now` around `submit`), and report CPU submission time separately (`webgpu-ddgi` does: CPU submission vs GPU passes).
- Avoid `mapAsync` readback in the timed region (zero mandatory readback; `ddgi-research` TESTING says bake/update readback must be zero and export readback measured outside).
- Headless: Chrome `--headless=new` works with WebGPU (what `webgpu-ddgi/tools/gpu/chrome.ts` and `bvh/tests/gpu/chrome.ts` do). Flags used there: `--headless=new --enable-unsafe-webgpu --remote-debugging-port=0 --user-data-dir=<profile> --disable-background-timer-throttling --disable-renderer-backgrounding --disable-backgrounding-occluded-windows --window-size=1280,720 --force-device-scale-factor=1 [--use-adapter-luid=high,low] [--disable-gpu]`. On macOS headless `--use-angle=metal` (upscaler docs); on Windows pin the discrete GPU via LUID on dual-GPU laptops (their `parseGpuLuidLines`).
- Playwright: `chromium.launch({ channel: 'chrome', args: [...] })`, `headless: false` is more representative (headless uses a different compositor path); keep one headed mode run to confirm presentation cost. Use `page.evaluate` for results, and expose a `window.__bench` API returning JSON (siblings use a bench page + Vite server on a dedicated port, `5199`).
- **Run discipline** (copied from siblings): (a) bench lock (FIFO GPU lock: `Denoiser/tools/gpu-lock.sh`, `/tmp/rt-gpu.lock`); (b) preflight: three consecutive samples with GPU <= 5% and CPU <= 20% (`ddgi-research/TESTING.md`); (c) record clocks, temperature, thermal flag during the run: `webgpu-ddgi/PROGRESS.md` flags a run VALID only if started at ~46 C, 0% thermal throttle in all samples, clock 780 MHz stable; INVALID verdicts exist for "card throttled to 900 MHz"; (d) 3 alternating baseline/candidate (ABBA) pairs; (e) median of 3 runs; (f) first-start (cold) kept separate.
- **Laptop power/thermal**: plug in AC; set power plan to max performance; fixed fan profile if possible; cap to a stable clock (Windows: `nvidia-smi -lgc <min,max>`, requires admin; Mac: cannot cap, so monitor with `powermetrics`/`pmset -g thermlog`, take only data in cool-start window, and interleave competitors rather than running them sequentially so drift hits all equally). Report device name, driver, Chrome version, OS, power state in each row (siblings' CSV schema has date, commit, device, browser).

### 4.6 Fairness rules for web competitors

1. Same geometry: feed identical triangle soup/glTF; same camera (position, FOV, sensor); same resolution (e.g. 1280x720 and 1920x1080, devicePixelRatio 1).
2. Same lighting: convert env map/area lights identically; disable features the other lacks (or report as "feature not present").
3. Same path settings: max bounces (e.g. 8), RR on/off, NEE on, no clamping or the same clamp value; report each renderer's own sampler (PCG, Sobol, blue noise).
4. Same output transform: take linear radiance buffer (readback of accumulation texture, rgba32f) -> EXR/PFM. If the competitor can't export raw, hook its accumulation target via a small patch/adapter in our harness (not in their repo), or screenshot with a neutral view transform + inverse sRGB (flag as lossy, 8-bit).
5. Same time budget: wall-clock measured with identical `requestAnimationFrame` cadence, vsync off if possible (headless or `--disable-frame-rate-limit --disable-gpu-vsync`; verify).
6. Denoiser off for the raw comparison; denoiser on (each side's default) for a second comparison, ours using the Denoiser package. Ours ran ~30% faster than mutracer's OIDN (brief); measure it in the same harness, same network (large/base/small), same resolution.
7. Publish adapter code and results so it is reproducible; record versions/commits.

## 5. Comparison targets and adapters

| Target | How to run | Output | Notes |
|---|---|---|---|
| Ours (web) | Playwright + bench page | linear EXR/PFM per checkpoint + timing JSON | |
| mutracer | clone (read-only) + serve statically via its own dev server in a separate process (do not install into cloned dir: per brief the repo itself must not be built there; copy the *built release* if available, or build under `bench/competitors/` outside `research/sources`, record). | page-driven; needs a render-to-buffer adapter or screenshot | cannot copy code; adapter only drives UI/API |
| three-gpu-pathtracer | local `~/Developer/three-gpu-pathtracer` (or npm tag); three.js WebGL2 | `pathTracer.target.texture` readback float (it renders to float RGBA); has `example/` scenes | WebGL; uses a different BVH (three-mesh-bvh); texture array/limits differ |
| Other web PTs (unverified list: Babylon.js/Blender-Cycles-in-web? `Needle`, `Threepipe` path tracer plugin, `Raytracing.js`/`webgl-path-tracing` (erichlof/THREE.js-PathTracing-Renderer), `lightmapper`) | per case | | erichlof's renderer is an important feature-comparison/perf-legacy reference (WebGL2); license **CC0-1.0 [V, GitHub licence file; not MIT]** |
| pbrt-v4 | `pbrt scene.pbrt --spp N --outfile out.exr` (CPU), `--gpu` (OptiX, Windows/Linux NVIDIA, wavefront GPU path) | EXR | Mac: CPU only (build from source; arm64 builds fine, unverified); GPU needs CUDA/OptiX |
| Mitsuba 3 | Python `mitsuba` wheels; variants `scalar_rgb`, `llvm_ad_rgb` (CPU LLVM), `cuda_ad_rgb` (OptiX) and `*_spectral`; `mi.render(scene, spp=N)` -> `mi.Bitmap` -> EXR | EXR | Mac works for LLVM; CUDA/OptiX requires Windows/Linux NVIDIA |
| Cycles | `blender -b scene.blend -o //out -F OPEN_EXR -f 1` with `--cycles-device OPTIX/HIP/METAL/CPU`; or standalone Cycles; Cycles uses Principled BSDF, not identical to glTF PBR | EXR | GPL/Apache? Blender is GPL, Cycles standalone Apache-2.0 (unverified); run as black box = fine; do not port code. Material mapping is lossy; use for "common artist reference", not for quantitative gating |
| Falcor / RTXPT | Windows only; Falcor Mogwai with path tracer pass; RTXPT is NVIDIA sample with DXR + NRD/ReSTIR | EXR via capture | Use for real-time speed targets (ms at 1080p) rather than quality parity |
| Capsaicin (AMD GI-1.0) | local `~/Developer/Capsaicin`; sibling already has `scripts/native/run-capsaicin.ps1` | | Real-time GI baseline |
| RTXGI-DDGI | sibling `ddgi-research/native/references/rtxgi-ddgi`, `webgpu-ddgi/native/rtxgi-ddgi`; PowerShell runner scripts in `ddgi-research/scripts/native/` | | Reuse for consumer comparisons (DDGI uses us for rays) |

Fair-comparison caveat: unbiased reference renderers should use the same *exact* BSDF models. pbrt-v4 `DiffuseMaterial`, `CoatedDiffuse`, `Conductor`, `Dielectric` and Mitsuba `principled`/`roughconductor`/`roughdielectric`; glTF metallic-roughness maps to pbrt's `CoatedDiffuseMaterial`-ish or Mitsuba `principled` only approximately (see 6). Prefer authoring the *reference scenes using only a common subset* (diffuse, rough conductor GGX, rough dielectric GGX, plus point/area/env lights) for the quantitative gate. The extension-rich glTF material tests get *visual/FLIP-bounded* comparison, not a tight numeric gate.

## 6. Reference pipeline

```
glTF (canonical) --+--> our renderer (web)
                   +--> converter --> pbrt-v4 .pbrt  --> pbrt (CPU/GPU) --> ref.exr
                   +--> converter --> Mitsuba XML/py --> mitsuba (llvm/cuda) --> ref.exr
                   +--> Blender import (io_scene_gltf2) --> Cycles --> ref.exr   (qualitative)
```

Conversion paths (status unverified; check each before committing):
- **glTF -> pbrt-v4**: existing converters: `assimp` can export `.pbrt`-like? No. pbrt-v4 ships `pbrt --toply`/`imgtool` and an assimp-based importer for geometry only (`pbrt-v4/src/pbrt/..` "assimp" used in `scenes`? unverified). Realistic path: our own small script (Node or Python with `pygltflib`/`trimesh`) emitting `.pbrt` with `Shape "plymesh"` or inline `trianglemesh`, materials mapped per table below, `Camera "perspective"` with explicit `fov`, `LookAt`, `Film "rgb"` (`"float iso"`/`exposure` fixed, `sensor "cie1931"`, `maxcomponentvalue` inf to avoid clamping), `Sampler "zsobol"` or `"independent"` for convergence tests, `Integrator "volpath"` (handles media) / `"path"` with `maxdepth` equal ours, `regularize false`.
- **glTF -> Mitsuba 3**: Python script building a scene dict: `mi.load_dict({...})`, `bsdf: principled` (maps well to glTF metallic/roughness/clearcoat/sheen/transmission) or `twosided`, `perspective` sensor with `fov_axis`, `hdrfilm` with `pixel_format=rgb`, `component_format=float32`, `rfilter: box` (set explicitly: box vs gaussian changes error), `integrator: path` with `max_depth`.
- **Blender**: `io_scene_gltf2` import for Cycles; Principled BSDF is the nearest; set view transform = Standard, film filter 1.5 px blackman-harris -> choose `box` 1.0 for parity (Cycles filter width setting).
- **Camera consistency**: use vertical FOV, aspect, resolution, pixel center convention (pbrt: y down; Mitsuba film origin top-left; glTF y-up, -z forward). Common bugs: FOV axis, sensor fit, handedness flips for pbrt (`Scale -1 1 1`), texture v-flip, vertex normal vs shading normal, per-face vs smooth normals, tangent sign, doubleSided, and area-light "two-sided" flags.
- **Lights**: map KHR_lights_punctual (candela/lux units!) to pbrt `point`/`spot` `"scale"`/`"power"` with a documented unit conversion (candela->W/sr; pbrt `I`; and `illuminance`). Emissive meshes: `AreaLightSource "diffuse"` with `"L"` = `emissiveFactor x emissive_strength`; same in Mitsuba `area` emitter. Env: pbrt `infinite` with `"filename"` + `"illuminance"` or `scale`; Mitsuba `envmap` with `to_world` rotation. Verify with the furnace: render uniform env on a sphere in all three and confirm pixel = analytic.
- **Material mapping table** (to be filled and tested): base color (sRGB -> linear), metallic/roughness (glTF roughness is perceptual: alpha = roughness^2, GGX; pbrt `roughness` with `remaproughness true` has similar semantics, verify), `KHR_materials_ior`, `KHR_materials_specular`, transmission (+volume: attenuationDistance/Color -> sigma_a), clearcoat -> pbrt `CoatedDiffuse`/Mitsuba `principled clearcoat`, sheen, anisotropy (Mitsuba principled `anisotropic`; pbrt: `ConductorMaterial` with `uroughness/vroughness`), iridescence (Mitsuba: not in principled; may need thin-film `RoughConductor` with `thin-film` only in some builds, unverified; pbrt-v4: no iridescence) -> expect partial support; document each as "no reference exists" instead of silently dropping.
- **Spectral**: our renderer is RGB now, spectral later. pbrt-v4 and Mitsuba spectral variants render in spectral; RGB-vs-spectral differences (esp. saturated colors, metals) cause a small persistent bias vs reference. Use `scalar_rgb`/`llvm_ad_rgb` Mitsuba variants for the RGB-vs-RGB comparison and spectral for later; pbrt-v4 is always spectral (RGB albedo uplifted via RGBAlbedoSpectrum), so expect it to deviate from RGB Mitsuba by small amounts; record the delta of pbrt vs Mitsuba-rgb on each scene as the *floor* below which our error comparison is meaningless.
- **Tonemap parity**: all renderers write linear scene-referred RGB (Rec.709/sRGB primaries, linear). Compare in linear. Compute metrics on linear for relMSE/SMAPE; for FLIP-LDR and display-oriented metrics, apply one shared tonemap (e.g. exposure-fixed Khronos PBR Neutral or ACES via a single Python/TS function; same code for all) then sRGB OETF; HDR-FLIP handles tonemapping internally (`--tone-mapper aces`, etc.).
- **Cache high-spp references once**: reference = pbrt-v4 (and Mitsuba as a second) at e.g. 65,536 spp on Windows (GPU), produce EXR + variance estimate (render two independent halves, report their difference as the reference noise floor). Store: `reference/<scene>/<renderer>-<spp>.exr`, `meta.json` (renderer version/commit, variant, command line, seed, resolution, wall-clock time, machine), `sha256`. Large EXRs go in an external store (Git LFS, or R2/S3 bucket keyed by hash, referenced in manifest). Never recompute unless the scene hash or converter version changes (key = hash(scene + converter version + renderer version + settings)).
- Reference noise floor must be well below target error: aim for reference relMSE vs a hypothetical infinite-spp reference <= 1/20 of the smallest error we claim.

## 7. Siblings: what exists and what to reuse

| Sibling | Path | What it gives us |
|---|---|---|
| bvh bench | `~/Developer/bvh/bench/{closest-point,query-parity,three-raycast}.ts`, `results/{trace,compare}.csv` | Raycast micro-benchmarks, CSV format |
| bvh compare | `~/Developer/bvh/compare/{adapters,core,gpu,node}` + `RESULTS.md` | **Adapter-interface pattern** (`adapters/types.ts`, features table), agreement checker (`core/agree.ts`: same / tie / disagree), seeded ray sets (`core/rays.ts`), interleaved timing (5 timed runs, warm-up 250 ms), memory accounting, GPU trace page (`compare/gpu`). Rather than reinventing: our PT compare reuses `scene-suite`, `rays`, `report` |
| bvh tests/gpu/chrome.ts | `~/Developer/bvh/tests/gpu/chrome.ts` (same file in webgpu-ddgi/tools/gpu) | Headless Chrome launcher incl. LUID pinning |
| webgpu-ddgi bench | `~/Developer/webgpu-ddgi/bench/results/*.csv`, `README.md` | **Results schema** (date, commit, device, browser, scene, metric, value), append-only, run1..3 + median, `note/gpu_busy`, `note/cpu_core_busy` flags; metric categories: timing, trace, memory, flicker, response, saturation, shading, native, baselines |
| webgpu-ddgi native | `~/Developer/webgpu-ddgi/native/rtxgi-ddgi`; `native.csv` | Native harness feeding the same CSV with `browser=native` |
| ddgi-research | `~/Developer/ddgi-research/{TESTING.md,comparisons,scripts,native}` | Tiered tests (fast/baseline/gpu/acceptance), per-step deadlines and exit codes, frozen acceptance thresholds, raw artifact + hash integrity, linear-format capture and `compare-linear.py`, PowerShell runners for Windows native (`run-capsaicin.ps1`, `run-f7.ps1`, `play-two-room.ps1`), parity attribution methodology ("do not fit intensity to native output") |
| gitestbench | `~/Developer/gitestbench/{TESTING.md,HANDOFF.md,src}` | Eyeball test scenes: furnace, Cornell kitchen, two-room leak; exact procedures; failure-reading tables. Port furnace + cornell to our Tier 0 |
| Denoiser scorecard | `~/Developer/Denoiser/tools/scorecard/{run.mjs,lib/metrics.mjs,scenes.json}` | Boil/still-sigma/lag, FLIP, per-region temporal metrics, crop strips, `index.html` report, GPU lock; **the temporal-stability definition** to adopt |
| Denoiser eval/oidn-compare | `~/Developer/Denoiser/tools/{eval,oidn-native-compare}`, `~/Developer/Denoiser-oidn-compare` | Native OIDN (CPU ground truth + Metal ceiling) vs web runtimes on same fp16-rounded inputs; dump from PT via `__dumpForOIDN()` -> PFM; `compare.py` |
| Denoiser rt-capture | `~/Developer/Denoiser/tools/rt-capture` | Headless capture of RT frames, `gen-synthetic.mjs` synthetic data |
| upscaler bench | `~/Developer/upscaler/bench/{src,results,timer-overhead.html,exposure-ceiling.html}` + `bench/results/` (E00: ABBA ordering, blinded A/B `review.html`, manifest hash, working-tree digest) | Timer-overhead calibration; ABBA protocol; blinded human review page with ROI outlines; `raw/E00/<timestamp>/run.json` provenance (manifest SHA-256, baseline git SHA, digest of tracked+untracked files) |
| materialx-fidelity | `~/Developer/materialx-fidelity` | Monorepo: `core` renderer interface + orchestration, `cli`, `viewer` website, `renderer-*` packages; model for "renderer adapters + gallery site" |
| three-gpu-pathtracer | `~/Developer/three-gpu-pathtracer/example` | Competitor scenes/UI; use as a runnable target |

Recommendation: do not copy their code into the new repo piecemeal. Put the shared parts (Chrome launcher, GPU lock, CSV schema writer, metrics module, FLIP, EXR IO, timer calibration) into one internal `bench-kit` package that this repo and the siblings can import (or vendor with a manifest), since five siblings have already re-implemented pieces (`chrome.ts` is literally duplicated in `bvh` and `webgpu-ddgi`).

## 8. Proposed bench harness for this repo

### 8.1 Layout

```
bench/
  README.md
  scenes/
    manifest.json            # id, tier, source URL, license, SHA-256, format(s), tri count, texture MB, camera set, light notes, converters + versions, expected-feature tags
    cameras/<scene>.json     # N named cameras (fov, pos, target, up, res), shared by all renderers
    local/                   # tiny scenes committed (furnace, cornell, two-room, material grid)
    fetch.mjs                # downloads big scenes by manifest (hash-checked) into $PT_SCENES_DIR (not in git)
  converters/
    gltf-to-pbrt.mjs
    gltf-to-mitsuba.py
    gltf-to-blender.py
    TESTS.md                 # furnace/camera/light unit conversion tests
  reference/                 # EXR cache (git-lfs or external); meta.json + sha256 each
  adapters/                  # renderer adapters, one interface (like bvh/compare/adapters/types.ts)
    ours.ts  mutracer.ts  three-gpu-pathtracer.ts  pbrt4.ts  mitsuba3.ts  cycles.ts  falcor.ts
  harness/
    chrome.ts                # from webgpu-ddgi/tools/gpu/chrome.ts (Playwright or CDP)
    calibrate-timers.html    # timestamp quantum + overhead (from upscaler/bench/timer-overhead.html)
    lock.sh  preflight.mjs   # GPU lock + idle/thermal/clock sampling
    run.mjs                  # CLI: --suite quick|ci|full --scenes ... --renderers ... --device ...
    page/                    # bench page: window.__bench.{load(scene), render(spp|ms), readbackEXR(), timings()}
  metrics/                   # Node + Python: relmse, smape, flip (ts port + flip-evaluator), ssim, bias, boil/still-sigma/lag (from Denoiser scorecard), convergence fit, efficiency
  results/
    raw/<run-id>/            # run.json (provenance: manifest hash, commit, dirty-tree digest, flags, GPU info), images, per-sample json
    *.csv                    # append-only, siblings' schema: date,commit,device,browser,scene,metric,value
  report/
    build.mjs                # static site: tables, convergence plots, side-by-side + FLIP heatmaps, feature-coverage matrix; review.html blinded A/B (from upscaler)
```

### 8.2 Suites

| Suite | Content | Runs | Where |
|---|---|---|---|
| `unit` (CI, no GPU) | metric math, converter tests, EXR roundtrip, CPU port of sampler/BSDF vs analytic | seconds | any |
| `furnace` (GPU, quick) | Tier 0 analytic scenes, assertions with tolerances, exit code != 0 on failure (ddgi-research rule: skipped required check = fail) | < 2 min | Mac + Windows |
| `quality` | Tier 1+2 to N spp, relMSE/FLIP/bias vs cached refs, convergence curves | 10-30 min | Mac + Windows |
| `speed` | Tier 3 + Tier 2: frame ms, samples/s, Mrays/s (counted), build/load/compile times, memory | 30-60 min with ABBA x3 | Windows (primary), Mac |
| `realtime` | Camera paths, denoiser on, boil/lag/FLIP per frame | 15 min | Mac + Windows |
| `compare-web` | ours vs mutracer vs three-gpu-pathtracer, TTEQ + efficiency | 1 h | Mac + Windows |
| `compare-native` | ours vs pbrt/Mitsuba/Cycles rendering time and quality, relative to cached refs | refs built once on Windows | Windows (+Mac CPU) |

### 8.3 Result storage and dashboards

- Append-only CSV per family as siblings do (keeps diff-able history in git; "numbers live here, not in report files"), plus `raw/<run-id>/` for images/JSON (git-ignored or LFS), each run with provenance (manifest hash, commit + dirty-tree digest, flags, adapter info from `adapter.info`/Dawn, Chrome version, OS, power state, preflight samples, VALID/INVALID verdict).
- Dashboard: static HTML report generated from CSV + raw (like `scorecard/index.html`, `materialx-fidelity` viewer, or an Artifact page), with: TTEQ table, convergence plots (log-log), speed bars with p50/p95, side-by-side image grid with FLIP maps (glTF-Render-Fidelity style), feature-coverage matrix, history chart per metric across commits (regression gate: fail CI if median slower > X% or error up > Y% on a given device).
- CI: GitHub Actions has no GPU; run `unit` only. The GPU suites run on our two machines, triggered manually / nightly, results committed or uploaded.

### 8.4 Reuse list

Chrome launcher (`webgpu-ddgi/tools/gpu/chrome.ts`), GPU lock (`Denoiser/tools/gpu-lock.sh`), CSV conventions (`webgpu-ddgi/bench/results/README.md`), timer calibration (`upscaler/bench/timer-overhead.html`), ABBA/provenance/blinded review (`upscaler/bench/results`), metrics (`Denoiser/tools/scorecard/lib/metrics.mjs`, `examples/realtime-denoiser/src/flip.ts`), adapters/agreement (`bvh/compare`), tiered gate runner and artifact hashing (`ddgi-research/scripts/test-suite.mjs`, `test-g*.mjs`), linear comparison (`ddgi-research/scripts/compare-linear.py`), furnace/cornell/two-room scenes (`gitestbench`), PowerShell native runners (`ddgi-research/scripts/native/*.ps1`), renderer-adapter site (`materialx-fidelity`).

## 9. Needs Windows / native hardware

- All NVIDIA/AMD GPU timing on DX12/Vulkan via Chrome/Dawn (the sibling runs are RTX 3060 Laptop; note laptop throttling, LUID pin, 780 MHz vs 1200 MHz clock states observed in `webgpu-ddgi/PROGRESS.md`).
- Windows-specific web issues: first-frame pipeline compile through DXC/FXC can take seconds (`webgpu-tsl-compile` skill); measure cold and warm.
- pbrt-v4 GPU (`--gpu`, OptiX + CUDA) reference at high spp, and large scenes (Moana) on pbrt CPU with lots of RAM.
- Mitsuba 3 `cuda_ad_rgb`/`cuda_ad_spectral` (OptiX) references.
- Cycles with OptiX/HIP/Metal-RT (Mac Metal possible) for artist reference.
- Falcor / RTXPT (DXR 1.1, NRD, SER, ReSTIR PT) for real-time speed ceiling; hardware RT numbers are a *ceiling/reference only* (our rule: software RT).
- Native OIDN GPU (CUDA/SYCL/HIP) vs our denoiser; sibling `oidn-native-compare` has the Mac recipe.
- Capsaicin / RTXGI-DDGI native runs (sibling PowerShell scripts).
- Power measurement (NVML/HWiNFO), clock locking (`nvidia-smi -lgc`, admin), multiple vendors (Intel Arc/AMD) for portability of subgroup sizes, f16, bindless flag behavior.
- Chrome bindless feature flag testing across platforms (D3D12 vs Metal vs Vulkan): needs each OS; confirm flag name per Chrome version (unverified).

## 10. Recommendations for the spec

1. Define the **bench contract** early: one `Renderer` adapter interface with `load(scene, camera)`, `renderTo(spp|ms) -> linear RGBA f32 buffer + timings`, `features`, `info`. Every competitor and native reference sits behind it; results table is auto-generated.
2. Make the path tracer expose **debug/bench hooks** natively: `readbackAccumulation()` (linear f32), per-pass GPU timestamps (named passes), ray counters in a debug build, fixed seed + deterministic mode (no time-dependent noise), bounce/RR/clamp parameters in one options struct, "no denoise / no tonemap / no jitter-from-time" switches.
3. Require **deterministic** replays for camera paths and fixed-seed comparisons (needed for the boil metric and for reference-image regression tests).
4. Treat **Tier 0 furnace tests as a merge gate** for every BSDF/sampler change (cheap, analytic, no reference needed).
5. Choose and document **reference noise-floor and eps** for relMSE once (suggest eps = 1e-2 on linear radiance, mean over 3 channels; 16 independent runs for variance).
6. Publish a **feature coverage matrix** vs the glTF extensions as a first-class output.
7. Pin the benchmark machine states (Mac M-series, Windows RTX 3060 Laptop as already used by siblings) and keep results per device; never compare numbers across machines.

## 11. Open questions

- Chrome flag spelling for precise timestamps in current stable (and whether Playwright's bundled Chromium honours it); verify with calibration page. Also whether `timestamp-query` inside passes is quantized on Metal.
- Does glTF-Render-Fidelity accept additional renderers, and which reference renderer(s) does it ship? (It may already contain Chrome model-viewer, three.js, Babylon, Filament, and path-traced Blender/Dassault references.)
- Exact license terms (CC BY vs CC BY-NC) for each ORCA/McGuire/Bitterli scene we intend to redistribute: store links + hashes only until verified.
- Mitsuba/pbrt iridescence/sheen/anisotropy parity: references may not exist for some KHR extensions; decide whether to render references in Blender (Principled has sheen, clearcoat, anisotropic, thin-film in 4.2+? unverified) or accept "visual only".
- How to count rays fairly across wavefront and megakernel designs, and across mutracer (unknown path structure): propose counting only at the traversal kernel from a counter, and primarily use spp/s + TTEQ.
- Bindless vs bind-group A/B in the same harness: flag availability in headless Chrome.
- Where to host cached reference EXRs (LFS vs bucket) and who has the Windows time for 65k-spp pbrt GPU runs of ~15 scenes.
- Whether to unify the siblings' duplicated bench pieces into a shared `bench-kit` package now or later.
