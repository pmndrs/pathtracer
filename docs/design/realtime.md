# Real-time mode foundation (P4.1)

What the library gives a real-time pipeline (spec section 7: demod -> SVGF-class temporal + spatial filter -> upscaler) and
what it leaves to the engine. Checked by `node bench/runner/run.ts --page realtime-check`. The filter (P4.5) and the upscaler
(P4.6) are separate tasks; this note is the contract they read.

## 1. Mode presets (`integrator/modes.ts`)

`mode: 'reference' | 'preview' | 'realtime'` on `PtMegakernelOptions` / `PtWavefrontOptions`. A key the caller sets wins; a key
left `undefined` takes the preset's value. `ptModePreset(mode)` returns the preset, `PT_MODE_NOTES` carries one status line per
entry (a measured number and its TASKS / DIVERGES row, or "pending A/B").

| entry | reference | preview | realtime | status |
|---|---|---|---|---|
| `alphaPolicy` | exact | opaqueSecondary | opaqueSecondary | D27, measured (R2): ~20-23% of trace time on a foliage scene, biased -3..-17% on cards |
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
sample, a = 1), `planes` = 6 planes of `width * height` vec4f, plane-major (plane p at byte `p * w * h * 16`), row 0 = top.
One buffer must fit `maxStorageBufferBindingSize` (half-res 1080p = 960 x 540 x 96 B = 50 MB).

| plane | rgb | a |
|---|---|---|
| beauty (`accum`) | total radiance | 1 |
| 0 diffuse | diffuse-class radiance / albedo factor | hit distance of the first bounce ray (0 = miss, or a specular-class path) |
| 1 specular | specular-class radiance / specular factor | the same distance (0 = miss, or a diffuse-class path) |
| 2 normal, viewZ | world-space unit shading normal of the primary hit (0 on a miss) | viewZ, negative in front (0 on a miss) |
| 3 motion | x, y = NDC delta current - previous (+y up, jitter-free); z = perceptual roughness | motionZ = viewZ(previous) - viewZ(current) |
| 4 albedo factor | the diffuse demodulation factor (albedo of the primary hit, floored at `PT_DEMOD_EPS` = 1e-3; 1 on a miss) | 1 when the primary lobe is specular-class |
| 5 specular factor | the specular demodulation factor (approximate pre-integrated specular term: F0 = mix(0.04, base, metalness), 1 for glass; floored; 1 on a miss) | 1 when the primary ray hit a surface |

Class split: the primary hit's BSDF kind decides. A conductor or dielectric lobe is specular-class, Lambert and layered
(coated) are diffuse-class. Everything after the primary vertex (all its bounces, NEE) goes to that class. Emission seen at the
primary hit and the sky behind a primary miss are diffuse-class (they are demodulated by the albedo factor, which recomposes
them exactly). So `diffuse * albedoFactor + specular * specularFactor = radiance` per pixel (checked to 1.1e-7 relative);
the stored factor is the floored one, so the identity is exact whatever the albedo.

What the Denoiser's `denoiser-realtime` wants (`DiffuseSpecularInputs`, `FrameGuides`, `FrameCamera`; read-only, its CAPTURE-FORMAT):
the caller demodulates; diffuse input = radiance / albedo (`guides.albedo` = plane 4 rgb), specular input = radiance /
pre-integrated specular term (`inputs.specular.demodulation` = plane 5 rgb), `hitDistance` = the `a` of planes 0 / 1, `guides.normal` =
plane 2 rgb (world space), `guides.viewZ` = plane 2 a (negative in front, the Denoiser's three.js view space), `guides.motion` =
plane 3 rg (NDC delta current - previous), `guides.motionZ` = plane 3 a, `guides.roughness` = plane 3 b, `FrameCamera.view` / `projection` (UNJITTERED) /
`prevView` / `prevProjection` from the host's cameras, `jitter` = the sampler's pair 0 (or the engine's). The Denoiser's
`depth` guide is required: derive device depth from viewZ with the projection (or pass the engine's raster depth).
Textures: the planes are buffers; a `copyBufferToTexture` (rgba32float rows are 256-aligned for `width >= 16`) or a small compute pass
makes the `rg16float` / `rgba16float` / `r32float` textures it wants (P4.5).

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
