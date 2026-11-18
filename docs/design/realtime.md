# Real-time mode foundation (P4.1)

What the library gives a real-time pipeline (spec section 7: demod -> SVGF-class temporal + spatial filter -> upscaler) and
what it leaves to the engine. Checked by `node bench/runner/run.ts --page realtime-check`. Sections 1-4 are the contract (P4.1);
section 5 is the chain that consumes it: the filter (P4.5), the upscaler (P4.6) and the preview pipeline (P4.7), measured in
[bench/results/REALTIME-PIPELINE.md](../../bench/results/REALTIME-PIPELINE.md).

## 1. Mode presets (`integrator/modes.ts`)

`mode: 'reference' | 'preview' | 'realtime'` on `PtMegakernelOptions` / `PtWavefrontOptions`. A key the caller sets wins; a key
left `undefined` takes the preset's value. `ptModePreset(mode)` returns the preset, `PT_MODE_NOTES` carries one status line per
entry (a measured number and its TASKS / DIVERGES row, or "pending A/B").

| entry | reference | preview | realtime | status |
|---|---|---|---|---|
| `alphaPolicy` + `alphaShadow` | exact | exact + transmit 0.45 | exact + transmit 0.45 | D31 (refines D27, R2b): shadows at about the solid walk's cost within 0.2% of exact on foliage; `alphaPolicy: 'opaqueShadow'` is the max-speed opt-in (-5..-17% on cards, 0.76 / 0.89 of exact time on foliage-mask) |
| `materialLod` | full | secondary (depth >= 1) | secondary | R1, furnace within ~1%; timing in QUIET-PASS; pending A/B (R9) |
| `textureLod` | 0 | cone, scale 1 | cone, scale 1 | pending A/B |
| layered materials | stochastic walk (adapter) | albedo + internal base + `texture` correction | same | D24 + P2.32b, measured |
| `maxDepth`, RR | the scene's | 4, off | 2, off | pending A/B (R3) |
| `scenePacking` | split | packed | packed | P3.11, per-pixel identical; timing in QUIET-PASS. NOT applied by `create` (see below) |
| `aovs` | off | on (OIDN, P4.7) | off (frame planes instead) | |
| `sampler` | owen-sobol | owen-sobol | bluenoise when `blueNoise` is passed | this task, V120 |
| `frame` | off | off | on | this task, V122 |

`scenePacking` has to be the same for `uploadSceneCopy({ packing })` and the kernel, so `create` never changes it: use
`ptModePreset(mode).scenePacking` for both. The limits (`PtLimits`) start from the mode's depth (`setLimits` overrides).
The wavefront takes the mode too, but drops what only the megakernel has (`frame`, `bluenoise`).

## 2. Blue-noise sampler (`sampler: 'bluenoise'`, V120, V121)

* Texture: our own jointly optimised `gf-vec2-64` / `-128` (D16, never STBN), passed as `PtMegakernelOptions.blueNoise`
  (`fetchBlueNoise` / `decodeBlueNoisePng`; 16-bit PNG, never through a canvas). Uploaded as `rg32float` (group 2 binding 12,
  `textureLoad`, no filtering). Power-of-two sides.
* Pair `k < blueNoisePairs` (default 6 = jitter, lens, and ALL of vertex 0: light point, BSDF direction, selection + component,
  RR) reads texel `(pixel + hash(k)) mod size`: each pair is blue on its own and pairs read different texels (no correlation
  between them). Pairs `>= blueNoisePairs` are the Owen-Sobol pair of the pixel seed (the frame index is the sample index).
* Per-frame animation, the same for every pixel (so the spatial structure survives): `value = fract(texel + shift(k, frame))`.
  `shift: 'owen'` (default, section 4) = the Owen-Sobol pair of the frame index with a seed shared by all pixels (a (0,m,2)-net per pair over 2^m frames);
  `shift: 'kronecker'` = frame times an irrational step (R2 for pair 0, `fract(sqrt(prime))` for the others: a shared step would
  move every pair along a diagonal and the time average would converge to the wrong 4D integral).
* Megakernel only. `ptBnPixel` is a `var<private>` set at the start of `ptPath`, so `ptSample2D`'s signature is unchanged.
* TS twin: `ptSampleBlueNoise`, `ptBlueNoiseShift`; `realtime-check` uses it for the exact jitter.

## 3. Per-frame outputs (`integrator.frame`, V122)

One sample per pixel per dispatch (`accumulate` defaults to false with it: the accumulator is overwritten), one `encode(encoder, 1)`
per frame. Per frame the host calls `setCamera` (first) or `nextFrame(camera)` (the old camera becomes the previous one; or
`setPrevCamera`). The camera is the UNJITTERED pinhole of the integrator; with the traced primary the sampler's pair 0 is the
only jitter owner (spec: one jitter owner per pipeline); with `primary: 'vbuffer'` the engine owns it and motion is still
jitter-free because it is the projection of a world point through two unjittered cameras.

Buffers (`kernel.frameBuffers`, or `readFrame()` for a CPU copy): `accum` = beauty (vec4f per pixel: total radiance of the
sample, a = 1), `planes` = 7 planes (`PT_FRAME_PLANES`) of `width * height` vec4f, plane-major (plane p at byte `p * w * h * 16`), row 0 = top.
One buffer must fit `maxStorageBufferBindingSize` (half-res 1080p = 960 x 540 x 112 B = 58 MB).

| plane | rgb | a |
|---|---|---|
| beauty (`accum`) | total radiance | 1 |
| 0 diffuse | diffuse-class radiance / albedo factor | hit distance of the first bounce ray (0 = miss, or a specular-class path) |
| 1 specular | specular-class radiance / specular factor | the same distance (0 = miss, or a diffuse-class path) |
| 2 normal, viewZ | world-space unit shading normal of the primary hit (0 on a miss) | viewZ, negative in front (0 on a miss) |
| 3 motion | x, y = NDC delta current - previous (+y up, jitter-free); z = perceptual roughness | motionZ = viewZ(previous) - viewZ(current) |
| 4 albedo factor | the diffuse demodulation factor (albedo of the primary hit, floored at `PT_DEMOD_EPS` = 1e-3; 1 on a miss) | 1 when the primary lobe is specular-class |
| 5 specular factor | the specular demodulation factor (approximate pre-integrated specular term: F0 = mix(0.04, base, metalness), 1 for glass; floored; 1 on a miss) | 1 when the primary ray hit a surface |
| 6 emission (P4.5a, V200) | radiance that is NOT demodulated: emission of an emitter hit by the primary ray, the environment / sun behind a primary miss and (`frameEmission: 'mirror'`) the emission / environment reached through a chain of perfect-mirror bounces, throughput included. Noise-free at its source. Added back AFTER the filter | 0 (reserved) |

Class split: the primary hit's BSDF kind decides. A conductor or dielectric lobe is specular-class, Lambert and layered
(coated) are diffuse-class. Everything after the primary vertex (all its bounces, NEE) goes to that class, EXCEPT the emission plane
(below). So `diffuse * albedoFactor + specular * specularFactor + emission = radiance` per pixel (checked to 1.1e-7 relative);
the stored factor is the floored one, so the identity is exact whatever the albedo.

**The emission plane (P4.5a, V200-V203).** Before it, emission seen at the primary hit and the sky behind a primary miss were diffuse-class: demodulated by the
albedo factor, which for an emitter is its (black) base colour floored at 1e-3, so a lamp of radiance 480 became 4.8e5 in the diffuse plane, hit the f16 clamp of the pack
(65000) and was spread by the filter (sponza-lamps with the lamps in view: denoised relMSE 228, Cornell with its light in view: 1e4). The light the primary BSDF does not
modulate is now accumulated in `fEmit` and written to plane 6 instead (`frameEmission`, an integrator option, default `'mirror'` with `frame`):

| `frameEmission` | plane 6 holds | the other planes |
|---|---|---|
| `'none'` | 0 (the P4.1 behaviour, the A/B baseline) | emission demodulated by the albedo factor; a specular-class primary puts its own emission in the diffuse plane |
| `'primary'` | emitter at depth 0 (`beta` = 1: its radiance), environment / sun on a depth-0 miss | diffuse = (radiance - emission) / albedo factor for a diffuse-class primary; specular = (radiance - emission) / specular factor for a specular-class one; the other is 0 |
| `'mirror'` (default) | the above, plus every emission / environment hit while ALL bounces so far were perfect-mirror (smooth CONDUCTOR, `PT_FLAG_DELTA`) samples: `ptFramePureNext(pure, specularBounce, bp.kind)` | as `'primary'`; the mirror's own reflection of a lit surface stays in the specular plane |

Glass is deliberately NOT a pure chain: a smooth dielectric picks reflect or refract at random per sample, so what is seen through it is noisy and must reach the filter
(its specular factor is 1, so nothing is divided). A rough conductor's emission goes to the specular plane too (it is MIS-sampled and noisy). The beauty (`accum`) is the
total (`radiance + emission`). Per-pixel plane write, the same in every scheduler: `ptFramePlanes(pixel, n, radiance, direct, emit, ...)` in `outputs-wgsl.ts` (`radiance` = everything
but `emit`; the legacy `none` passes `direct` = primary emission and `emit` = 0). **A scheduler that writes the planes (the wavefront's shade kernel) needs: one extra vec3
per path (`emit`, summed where the path adds emission / escape radiance while the path is pure) and, for `'mirror'`, one `pure` bit (set at the camera kernel, cleared
by `ptFramePureNext` after each BSDF sample; the wavefront's depth word keeps depth in bits 0-28 with specular / `aovDone` / live in 31 / 30 / 29, so the bit must come from the top depth bit (a 2^28 path limit is no limit) or from a spare lane of the state vec4s); pure sites add to `emit` instead of `radiance`
and the final write is `ptFramePlanes(..., radiance, vec3f(0.0), emit, ...)` with the beauty `radiance + emit`.** The accumulating `ptAccum` of a frame kernel is `radiance + emit`.

What the Denoiser's `denoiser-realtime` wants (`DiffuseSpecularInputs`, `FrameGuides`, `FrameCamera`; read-only, its CAPTURE-FORMAT):
the caller demodulates; diffuse input = radiance / albedo (`guides.albedo` = plane 4 rgb), specular input = radiance /
pre-integrated specular term (`inputs.specular.demodulation` = plane 5 rgb), `hitDistance` = the `a` of planes 0 / 1, `guides.normal` =
plane 2 rgb (world space), `guides.viewZ` = plane 2 a (negative in front, the Denoiser's three.js view space), `guides.motion` =
plane 3 rg (NDC delta current - previous), `guides.motionZ` = plane 3 a, `guides.roughness` = plane 3 b, `FrameCamera.view` / `projection` (UNJITTERED) /
`prevView` / `prevProjection` from the host's cameras, `jitter` = the sampler's pair 0 (or the engine's). The Denoiser's
`depth` guide is required: derive device depth from viewZ with the projection (or pass the engine's raster depth).
Textures: the planes are buffers; a `copyBufferToTexture` (rgba32float rows are 256-aligned for `width >= 16`) or a small compute pass
makes the `rg16float` / `rgba16float` / `r32float` textures it wants (P4.5).

The edge case the plane does NOT solve: a sub-pixel emitter (Sponza's lamps are 1.5 px across at 960 x 540) is a binary 1-spp coverage sample at its silhouette (the per-pixel jitter makes a
pixel on the lamp's edge 480 or 0 from frame to frame). The emission plane is not filtered, so that aliasing reaches the upscaler as it is (raw-sample level on the lamp's pixels, no worse); the
upscaler's temporal accumulation is its only filter. Temporal accumulation of the emission plane in the chain (reprojected, neighbourhood-clamped) is the fix if it matters (TASKS P4.5c).

Not in the frame outputs, by design: instance motion (static scenes first; the scene copy keeps only the current instance
transforms, so a moving object needs last frame's `objectToWorld` or a per-instance delta: P4.5), motion of a mirror's reflection
(the virtual-motion variants of the Denoiser's `recurrent` filter need the reflected hit; the first bounce's `hitDistance` is there),
and `primary: 'vbuffer'` with frame outputs is untested (the frame planes read the same `surf` and `hit`, so it should work).

## 4. Open choices (build both, then decide)

* Blue-noise shift: `owen` (default) vs `kronecker`. A/B on Cornell 96 x 96, maxDepth 2, 128 one-spp frames per sampler against a 2048 spp
  reference (`realtime-check check=bluenoise`; relMSE; blur = Gaussian sigma on test and reference):

  | sampler | frame relMSE raw | after blur sigma 2 | after blur sigma 4 | 128-frame mean relMSE | mean bias |
  |---|---|---|---|---|---|
  | owen-sobol | 0.162 | 3.38e-3 | 9.9e-4 | 3.1e-4 | +0.04% |
  | pcg | 0.161 | 2.94e-3 | 7.5e-4 | 12.6e-4 | +0.21% |
  | bluenoise, kronecker shift | 0.163 | 2.30e-3 (1.47x better than owen-sobol) | 6.8e-4 (1.45x) | 5.4e-4 (1.75x WORSE) | -0.01% |
  | bluenoise, owen shift | 0.156 | 2.29e-3 (1.47x) | 7.5e-4 (1.32x) | 3.3e-4 (same) | +0.02% |

  The raw per-frame error is the same (blue noise moves it to high frequencies, it does not remove it); after a low-pass, which is
  what a spatial filter does, it is 1.3-1.5x lower than Owen-Sobol. Both are unbiased
  (|mean bias| <= 0.2% over 128 frames). The kronecker shift converges 1.75x worse under accumulation (a per-pair irrational step
  is not a good 12-dimensional low-discrepancy sequence), the owen shift matches Owen-Sobol, so `owen` is the default. One scene, one depth:
  suggestion, not a D-row; the A/B that would settle it for good is P4.5's after the temporal filter (steadier in motion, R9).
* Blue noise for deeper pairs: 6 pairs = vertex 0 only. More pairs (vertex 1) need a third texture read per pair; A/B `blueNoisePairs` 6 vs 10.
* Frame planes as a storage buffer (here) vs `rgba16float` storage textures (half the bandwidth, no copy for the filter): needs
  `texture-formats-tier1` / `bgra8unorm-storage` style caps on some devices; P4.5 decides when it knows what the filter reads.
* Megakernel only. The wavefront shade kernel is at 8 of the 10 Apple storage buffers; 1 more (frame planes) fits only with
  `scenePacking: 'packed'`. A wavefront frame mode is P4.1b.

## 5. The chain: kernel -> filter -> upscaler (P4.5, P4.6, P4.7)

The library stays dependency-free: the Denoiser and the upscaler are siblings, so the chain lives in the bench (`bench/realtime/chain.ts`, the code track N's `bench/nrc/pipeline.ts`
had; `nrc-pipeline` and `realtime-pipeline` import the same file) and this section is how an engine wires it. `bench/realtime/session.ts` (`RealtimePipelineSession`) is the whole
thing for a plain kernel with no cache; page `realtime-pipeline` checks it.

```
PtMegakernel { mode: 'realtime', maxDepth 2, [restir] }   960 x 540       one encode = one frame = one sample per pixel
   frameBuffers { accum, planes }  (storage buffers, plane-major)
        |  PipelineArm.accumulate + pack (compute, ~1.4 ms at 960 x 540):  planes -> rgba16float / r32float textures through ONE buffer and
        |  copyBufferToTexture (rows 256-aligned), device depth from viewZ (one render pass, frag_depth, near 0.05 / far 200)
        v
   Denoiser DiffuseSpecularDenoiser.dispatchBoth (SVGF, anti-lag auto, remodulate: true)    diffuse = plane 0 / albedo factor, specular = plane 1 / specular factor
        |  composite (diffuse + specular; the raw sky where every sample missed the scene)         + reactive mask pass (optional)
        v
   pmndrs/upscaler temporal (jitter: false), render -> display (2x)  [color, depth, velocity = the motion plane, reactive]
        v
   tonemap (exposure, ACES fit, sRGB) -> rgba8unorm
```

An engine's calls per frame, in order: `kernel.nextFrame(camera)` (first frame and after a cut: `setCamera`, `setPrevCamera(camera)`, `restir.setPreviousCamera(null)`, `arm.resetHistory()`);
`arm.beginFrame(e); kernel.encode(e, 1); arm.accumulate(e, kernel.frameBuffers); arm.pack(e, 1)`; submit; `await arm.post(camera, { tonemap: true })` (the Denoiser and the upscaler submit
their own command buffers; nothing is read back). The camera is the UNJITTERED pinhole of the kernel; the host's cameras give `FrameCamera.view / projection / prevView / prevProjection`
(`chain-math.ts`: the same conventions as the kernel's `ptCameraBasis`, three's WebGPU projection for the upscaler). History resets on a cut; instance motion is not in the planes (static scenes).

**Jitter owner (P4.6 decision): the kernel's sampler, as in track N.** Pair 0 jitters every pixel's primary ray inside the pixel (blue noise without ReSTIR, Owen-Sobol with it); the upscaler
runs `jitter: false`, the denoiser gets `jitter [0, 0]`, motion is jitter-free by construction. The alternatives, both built in the library but not wired here: the upscaler as owner (one Halton offset per frame
through a uniform camera shift in the kernel: N11.2, not built) and `primary: 'vbuffer'` + `setVbufferJitter` (needs a raster pass and `encode(1)`; the planes are untested with it). Reasons to keep the sampler:
no kernel change, no second owner by construction, and per-pixel blue-noise jitter already gives the upscaler distinct sub-pixel samples; what it cannot do is reconstruct above the render
resolution from known offsets (the 2x reconstruction floor of the control row, relMSE 0.015 on sponza-white, is the price; A/B in N11.2).

**Reactive mask (R7, `settings.reactive`, off by default).** `r = strength * clamp(1 - (n - 1) / cap, 0, 1)` per render pixel from the SVGF history length n (alpha of its accumulation texture), written as rgba8unorm and
passed as the upscaler's `reactive` input: fresh and disoccluded pixels favour the current frame. On the orbit it lowers the upscaler's darkening (-4.8% against -7.0%) and the error at the last pose slightly (FLIP 0.107 against 0.110) but
costs 2.3x the boil (0.0082 against 0.0036): off by default, a quality-vs-steadiness setting. It reads the Denoiser's internal `accum` debug texture.

**P4.7, the preview pipeline,** is the same chain with `denoise: 'oidn'` (`moving: 'fast'` every frame, `stillSamples: Infinity`, `hold: false`) at a lower render size and maxDepth 4 (`RealtimePipelineSession { preview: true }`, page args `res=640x360 maxDepth=4 variants=svgf,oidn-fast`).
It runs the real-time frame kernel (OIDN's albedo and normal guides come from the frame planes), not `mode: 'preview'` (whose AOV path serves `pt_denoise=oidn`, the converged-frame variant). Numbers: REALTIME-PIPELINE.md. OIDN small costs 4.6 ms at 640 x 360 and 11.8 ms at 960 x 540 on the M5 Pro (wall clock, awaited, provisional) against 1.2 ms for SVGF at 960 x 540; the preview frame (640 x 360 -> 1280 x 720, maxDepth 4) is 20.1 ms (13.3 trace + 0.6 pack + 5.0 OIDN + 0.4 upscale), SVGF in the same configuration 14.9 ms.

**What the chain does not do yet.** Instance motion (moving objects: the scene copy has only the current transforms), motion of reflections (the virtual-motion variants), wavefront frame mode (P4.1b), the wavefront with ReSTIR.
Emitters seen directly by the primary ray are handled (P4.5a): plane 6 is summed over the frame's samples by `pipeAccumulate`, packed as an `rgba16float` texture (clamped at 65000: a visible sun disc, 7e5, saturates) and added by the composite AFTER the
filter (`diffuse + specular + emission` where the primary ray hit, the raw beauty, which holds the sky, where it missed); the upscaler sees the full image. Only the sub-pixel emitter's silhouette aliasing remains (above).

### Results (M5 Pro, provisional timing; own depth-2 references at 1920 x 1080)

Detail and every table: [REALTIME-PIPELINE.md](../../bench/results/REALTIME-PIPELINE.md). sponza-white, 960 x 540 -> 1920 x 1080, maxDepth 2:

| stage | ms (load 3-10; batched wall clock, PROVISIONAL) |
|---|---|
| path trace (NEE) | 15.2 |
| path trace + ReSTIR DI (biased + visibility / unbiased) | 20.6 / 17.5 |
| accumulate + pack (frame planes -> textures, depth) | 1.35 |
| denoise + composite (SVGF) | 1.2 |
| upscale | 0.75 |
| tonemap | 0.04 |
| **whole frame, NEE / ReSTIR biased / ReSTIR unbiased** | **19.6 (51 fps) / 31.0 (32 fps) / 21.0 (48 fps)** |

Quality (static, frame 64, against our own 1024 spp depth-2 reference): raw 4.5, denoised 0.065, final 0.056 relMSE (resolution floor 0.015), final FLIP 0.103; moving camera: boil 0.0036 (k = 1)
against 0.0090 for OIDN small, the preview pipeline's denoiser. P4 exit gate (about 30 fps on a 3060L-class GPU, half-res -> 1080p, 2 bounces + ReSTIR DI; steadier in motion than preview mode):
the second half holds (2.5x / 1.9x lower boil, 20% lower FLIP than OIDN small at the same size); the first half is an ESTIMATE: this Mac's trace is 15.2 ms, the only 3060L-over-M5 ratio we have is 74.1 ms (3060L, 8 vertices with RR from 4, throttled, N11) against
55.5 ms (M5 Pro, 8 vertices, no RR, load 14), i.e. 1.3x (more once RR is accounted for: about 2.1x), so the 3060L trace is about 20-32 ms; with the track N chain's 11 ms of pack, denoise and upscale on that GPU the frame is 31-43 ms without ReSTIR (23-32 fps) and
34-54 ms with it (19-29 fps): borderline without ReSTIR, short with it on a full Sponza. The M5 Pro is comfortably inside the gate (32-51 fps). Settle on the 3060 (`realtime-pipeline` with `--gpu nvidia`; TASKS P4.5b) before calling the gate.
