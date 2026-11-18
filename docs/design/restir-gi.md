# ReSTIR GI (P5)

Real-time mode only, on the megakernel and (P5 wavefront GI, below) the plain wavefront, on top of ReSTIR DI (`docs/design/restir-di.md`): `integrator.restir.gi`, default off. Reference mode and every kernel without `gi`
are byte-identical (`node bench/size-snapshot.ts`, `diff -r` before / after: no file of the old set changed). Code: `packages/pathtracer/src/restir/restir-gi-wgsl.ts` (options, the three
passes, the shade-side read), the host in `restir.ts` (`PtRestirPasses`: GI buffer, pipelines, steps), the shade hook in `integrator/integrator-wgsl.ts`, the CPU oracle in
`restir-gi-oracle.ts`. Checks: `restir-check check=gi` and `realtime-pipeline restir=gi,giu` (both below).
Written from Ouyang et al. 2021 (ReSTIR GI), Lin et al. 2022 (GRIS) and the Wyman course notes; no RTXGI code.

## What a sample is

```
x0 (eye) -> x1 (the primary hit, G-buffer) -> x2 (one BSDF-sampled ray from x1) -> light      sample y = (x2, n2, Lo, W)
```

`Lo` is the direct light x2 reflects toward x1, computed exactly as the path tracer's depth-2 vertex does (one light sample MIS-weighted against the BSDF, plus one BSDF sample
whose emission / sky is MIS-weighted against the light sampler). So `F W` of a fresh candidate is the plain path-tracing estimate `beta Lo`, and **GI + DI is a depth-2 path tracer**:
direct light at x1 is DI's (every light, no MIS, as before), everything after x1 is the GI sample. The shade pass, at the first non-delta vertex, adds the final GI reservoir's
contribution and **ends the path** (no BSDF continuation, no ray at all for the GI term). Paths whose first vertex is a delta lobe (mirror, smooth glass) are not touched: they
continue as plain path tracing, and their later vertices have no GI. Emission seen directly from x1 is DI's (the BSDF-sampled continuation already gets no MIS weight at the first vertex).

Deeper bounces than 2 are not in the sample (open problem 1): the image is the depth-2 estimate, as the real-time chain's `maxDepth: 2` already was. `nee_deep` in the checks says how
much that is (Cornell +11.8%, Sponza-white +39%, env-sphere +1.1%).

## Measure and the reconnection Jacobian

The sample is a point on a surface, so it lives in the AREA measure at x2. The target of pixel `d` is `F_d(y) = f_d(wo_d, wi_d) |cos1_d| Lo G_d V_d(y)` with `G_d = |cos phi2_d| / |x2 - x1_d|^2`,
`phat = luminance(F)`, and the contribution weight W is per unit area (a fresh candidate: `W = r^2 / (pdf_omega |cos phi2|) = 1 / (pdf_omega G)`). The ratio `G_d / G_s` between two pixels is
exactly the reconnection Jacobian of Ouyang et al. (`|cos phi2'| |x2 - x1|^2 / (|cos phi2| |x2 - x1'|^2)`): it is not a separate term, it falls out of evaluating `phat` of the same y at another
domain. The CPU oracle (`restir-gi-oracle.ts`, `restir-gi.test.ts`) checks it on a plane with a closed-form answer: a fresh candidate averages to the quadrature, the Talbot combine of three pixels with
different heights, visibility and a non-shiftable region is unbiased to 1%, the `1/sum M` combine on the same world is not.

## Shift validity (what makes the unbiased mode honest)

`Lo` is computed once, toward the original x1. It is exact at another pixel only if x2 is Lambert (a glossy x2 would have to re-evaluate its BSDF toward the new x1). The sample stores `diffuse`
(`bp2.kind == PT_BSDF_DIFFUSE`), and a sample is re-evaluated at another domain only if: x2 is Lambert, x2 is seen from its front side by both pixels, both distances are at least
`minDistance` (0.05 m: G = cos / r^2 explodes), and the shadow ray between that pixel and x2 is clear (the target is **shadowed**: unlike DI's unshadowed target, a reused sample must be visible
from its new pixel, which is why there is no shadow ray at shade time). Anything else has `phat = 0` at every domain but its own. The rule is symmetric in the two pixels, so the balance-heuristic
weights of the unbiased mode are still a partition of unity. x2 on a non-Lambert surface (plastic, metal, glass; OpenPBR layered) and x2 on a delta lobe: the glossy ones live and die at
their own pixel (no reuse, noise of a plain path tracer there); a delta x2 is dropped (open problem 2).

## Passes

```
ptRestirGbuf (DI)      the shared G-buffer (+ one word: the geometric normal, only with gi: the secondary ray's origin offset)
ptRestirGiInit         x1 BSDF sample (one PCG pair), ray to x2, Lo (NEE + BSDF sample at x2), reservoir A: M = 1, W = r^2 / (pdf cos)
ptRestirGiTemporal     reproject x1 through the previous camera (the DI pass's rejection tests, 0.9 / 0.1), merge the previous final GI reservoir
ptRestirGiSpatial      merge k neighbours (random offsets in a disc, <= 4)
ptMain (shade)         first non-delta vertex: DI term as before, + f cos Lo W G (final GI reservoir), then the path ends
```

The GI passes are their own module (`ptRestirBuild(.., 'gi')`: the DI text with the DI entry points left out, pruned): DI module 41.0 KB, GI module 47.7 KB (50.4 with an environment map; 56.1 KB with
OpenPBR materials, where the DI passes are already 47.7 KB). They share the DI params record (a GI step writes its own settings into the same 208 bytes: `cfg.w` carries `minDistance`), the
G-buffer and the camera rules; the passes bind one more storage buffer (binding 4, the GI reservoirs), the shade kernel one more read-only buffer (binding 19; it was 16 until the P5 merge, which gave volumes 16-17 and spectral 18). Reservoirs: 2 vec4u per pixel per
region, 4 regions: x2 (3 x f32), n2 (octahedral), Lo as f16 / s with s = max(Lo) (f32), W (f32), M (8 bit), flag. 1280 x 720: 118 MB (limit of one storage binding: 128 MiB; the passes bind it whole).
The PCG keys of every random draw are new (`PT_RESTIR_GI_KEY`, 176..215, disjoint from DI's 0..168: P4.2b's lesson).

## Bias modes (D11: both built)

| `gi.bias` | combine | cost | notes |
|---|---|---|---|
| `unbiased` | balance-heuristic MIS over the domains with shadowed targets (`W = wsum / phat`), `phat_j(y_i)` for every ordered pair | one shadow ray per ordered pair of domains (k = 1 spatial + temporal: 2 + 2) | unbiased up to the shift-validity rule; converges to the plain depth-2 estimator (below) |
| `biased` (default) | `W = wsum / (phat sum M)`, only the destination's `phat` | one shadow ray per reused sample | darkens where neighbours disagree on visibility: Cornell -0.45% (temporal), -0.95% (+ spatial), Sponza-white -5.9%, env-sphere -1.6% |

`gi.spatial` is at most 4 (the unbiased merge is quadratic). `temporal` and `spatial` can be flipped at run time through `PtRestirPasses.giOverrides` (the check pages do).

## Checks (all provisional on a loaded Mac: relBias and relMSE are not timings)

Machine: M5 Pro, Chrome 157, **shared and often loaded (load average 3 to 90 during these runs): every number here is provisional**; relBias / relMSE do not depend on the load, the ms do. Raw rows: `bench/results/restir-check.csv`, `realtime-pipeline.csv`.

**(a) + (b) Reuse off equals the plain depth-2 estimator; unbiased converges** (`restir-check check=gi`: static camera, 128 accumulated frames, 6 seeds, k = 1, one 2048-frame reference of OUR plain path tracer at `maxDepth` 2;
relBias mean +- standard error over the seeds. The reference is Owen-Sobol, so the plain estimators themselves sit a few 1e-4 from it: that band, not the standard error, is the resolution):

| scene | plain NEE (Owen / PCG) | gi_off (no reuse) | gi_ts_unb | gi_ts_bia | plain depth (`nee_deep`) |
|---|---|---|---|---|---|
| cornell | +3.6e-5 / +1.5e-5 | -5.4e-5 +- 0.9e-5 | -1.1e-4 +- 0.7e-4 | **-0.95%** | +11.8% |
| sponza-white | +4.3e-4 / -6.5e-4 | -3.7e-4 +- 0.3e-4 | -3.9e-4 +- 2.6e-4 | **-5.9%** | +39% |
| env-sphere | +2e-6 / +2.6e-4 | -5.8e-5 +- 9e-5 | -1.9e-4 +- 1.5e-4 | **-1.6%** | +1.1% |
| quad-light, lights-mix-lamps | flat plane: no second bounce, GI is a no-op (all modes bit-identical) | | | | |

Temporal only (Cornell): unbiased -1.3e-4 +- 0.6e-4, biased -0.45%. Non-finite pixels: 0 in every run. The first Sponza run (before the secondary vertex got its own BSDF sample + MIS, i.e. `Lo` = light sampling only) read -0.16% at gi_off: the sky of Sponza
is better sampled by the BSDF, and the missing MIS was both the noise (2.4x PCG NEE -> 1.9x) and most of that bias (the unbiased arithmetic was never in question: the CPU oracle). A ray-origin offset by the geometric instead of the shading normal changed nothing there (bit-identical), the
G-buffer word for it is kept because a grazing secondary ray from a smooth-normal surface needs it in principle.

**(c) Per-frame noise, 1 spp per frame, 960x540-class chain input** (`restir-check check=giframes`, 3 seeds, 256-frame reference per pose, orbit 1 degree per frame; frame t alone = t x mean_t - (t-1) x mean_(t-1); all kernels `maxDepth` 2):

| relMSE of one frame | cornell f1 / f4 / f8 | moving (block p99 bias) | sponza-white f1 / f4 / f8 | moving (block p99) |
|---|---|---|---|---|
| plain NEE (1 frame) | 0.140 | 0.137 (0.33) | 4.64 | 2.66 (1.31) |
| DI alone, unbiased (`di_unb`) | 0.128 / 0.125 / 0.127 | 0.129 (0.33) | 8.74 / 8.73 / 8.87 | 3.44 (1.38) |
| DI alone, biased + visibility | 0.127 / 0.123 / 0.125 | 0.127 (0.38), bias -1.3% | 6.84 / 4.19 / 4.08 | 1.87 (1.03), bias -2.0% |
| DI + GI, both unbiased | 0.079 / 0.021 / 0.017 | **0.020 (0.15), bias +0.2%** | 7.41 / 5.47 / 5.34 | 2.26 (1.33), bias -0.4% |
| DI + GI, both biased | 0.080 / 0.018 / 0.013 | 0.016 (0.35), bias -2.6% | 5.41 / 0.84 / 0.45 | 0.64 (0.82), bias -3.1% (static camera -9.4%) |
| DI biased + GI unbiased | 0.078 / 0.019 / 0.015 | 0.018 (0.32), bias -1.2% | 5.51 / 0.93 / 0.54 | 0.68 (0.83), bias -2.0% |

Reading: on Cornell GI cuts the per-frame error 7x against both plain NEE and DI alone (DI alone does nothing for the indirect light), ghosting-free when unbiased (moving block bias +0.2%, block p99 half of plain NEE's). On Sponza-white the unbiased DI (no visibility, the sun) is itself the noisy part
(8.7 > plain 4.6): pair the unbiased GI with DI's biased + visibility mode and GI is 8.5x better than plain NEE per frame at f8 (0.54 vs 4.6) with no bias of its own (the remaining -3% is DI's biased darkening, P4.2g). The fully biased GI is the lowest noise (0.45) and darkens the static camera by 9%.

**The real-time chain** (`realtime-pipeline sponza-white`, 960x540 -> SVGF -> 2x upscale, 16 frames, references 512 spp of our plain depth-2; relMSE of the denoised stage at frame 1 / 4 / 16 (relBias), raw stage in the ratio column):

| arm | f1 | f4 | f16 | raw f16 | path, SVGF final pose 12 / 23 (relBias) | path, no denoiser |
|---|---|---|---|---|---|---|
| plain NEE | 0.120 | 0.075 | 0.070 (-1.3%) | 4.48 | 0.120 (-5.6%) / 0.040 (-6.0%) | 0.328 / 0.272 |
| DI (biased) | 0.137 | 0.091 | 0.056 (-8.6%) | 3.87 | 0.089 (-12.8%) / 0.041 (-14.7%) | 0.245 / 0.209 |
| DI + GI, both biased | 0.142 | 0.088 | 0.046 (-13.0%) | 0.44 | 0.083 (-16.6%) / 0.044 (-22.8%) | 0.134 / 0.094 |
| DI biased + GI unbiased | 0.143 | 0.090 | **0.043 (-8.0%)** | 0.54 | 0.086 (-12.1%) / 0.040 (-14.2%) | 0.127 / 0.084 |
| DI + GI, both unbiased | 0.152 | 0.125 | 0.093 (-3.4%) | 5.29 | 0.125 (-6.7%) / 0.046 (-9.5%) | 0.192 / 0.112 |

(The bias column of every row includes the chain's own (SVGF, upscaler: P4.6a): -1.3% for plain NEE static, -6% on the path.) Boil (the Denoiser's calm scorecard, final stage k = 1): 0.0054 plain, 0.0055 DI, 0.0053 GI, 0.0057 GI unbiased: no flicker cost.
After the denoiser the picture is closer than the raw one: GI buys 35-40% of the final error at frame 16 and nothing at frame 1-4 on Sponza (the denoiser already removes the noise GI removes); the raw (no denoiser) path is 2.4-3.3x better than plain NEE.

**Time** (`realtime-pipeline scenarios=timing`, trace stage of one frame, sponza-white 960x540, M5 Pro, load 3.7, PROVISIONAL, 250 ms batches x 5, median; no GPU timestamps): plain NEE 15.0 ms, DI 21.0 ms, DI + GI biased 28.6 ms, DI + GI unbiased 30.6 ms (whole chain frame 20.0 / 29.5 / 37.3 / 38.4 ms).
GI costs 7.6 ms (biased) / 9.6 ms (unbiased) over DI at this size: three more passes plus the shadow rays of every merged sample (the shade kernel gets cheaper: its path ends at the first vertex). A QUIET-PASS row exists (P5.5).

## On the wavefront (P5 wavefront GI, V280)

`PtWavefront.create(..., { integrator: { restir: { gi: {} } } })` (plain scheduler, as DI: refill / layers throw) runs the SAME `PtRestirPasses` (DI + GI passes modules, buffers, previous-camera rule, `giOverrides`) inside the sample's compute pass, before the camera
kernel; nothing in the passes changed. Only the shade side is new, and it is the megakernel's text: `ptRestirGiShadeWgsl(evalFn, base)` (the final GI region index is a parameter: `ptParams.limits.w` in the megakernel, `bitcast<u32>(ptParams.up.w)` in the wavefront, where
`limits.w` is the iteration; DI's final region rides in the same lane and is the same region number), bound read-only at group 2 binding 19 in the shade kernel only (DI's binding 11 + this one: 10 of 10 Apple storage buffers per stage with a split scene copy, 9 packed; `integrator.counters` or AOV buffers on top would not fit a split copy). At vertex 0 of a non-delta lobe the shade kernel queues DI's shadow ray (visibility through the shadow
kernel, as DI), adds `beta * ptRestirGiLi(...)`, writes the GI distance into the frame plane's alpha (`ptFrameBounce(path, n, ptGiDist)`, the megakernel's `fBounce = ptGiDist`) and ends the path: no extension ray, so GI + DI is exactly the megakernel's depth-2 estimate (the plain wavefront's iteration loop then has only the shadow queue's work left after the first shade). Frame planes need nothing else (the GI term lands in `radiance`, which the class add splits by the primary BSDF kind). Not supported on the wavefront (as DI): `sharedHit`, alpha MASK, hooks, refill.
The kernel text of every build without `gi` is byte-identical (`node bench/size-snapshot.ts`, `diff -r` ignoring `Only in`: no file changed).

Sizes (minified, `node bench/size-snapshot.ts`; wavefront shade kernel / megakernel shade kernel): plain GI 32.4 / 43.1 KB, + env map 35.1 / 45.9, real-time preset (frame planes, blue noise) 38.7 / 50.1 (the real-time megakernel is 50.2 KB in the bench adapter: over the DXC class), real-time + OpenPBR albedo
48.3 / 59.5 (the megakernel with OpenPBR + GI is 52.4 KB even without the frame planes). The wavefront's other four kernels are 1.1 - 12.6 KB. The GI PASSES module is the same on both schedulers (47.6 KB, 50.3 with an env map, 56.9 KB with OpenPBR): the wavefront split does not help there (open problem 5).

Checks: `wf-frame-check restir=gi|giu` (cornell, sponza-white, lights-mix-lamps, pbr-full, many-lights, lamps; six frames, orbit / pan): every plane within 1.4e-7 relative of the megakernel's, i.e. the same reservoirs and the same pixels (the check turns `sharedHit` off on both: the megakernel's real-time preset adds it and its unorm16 barycentrics move viewZ by 7e-5);
split and packed scene copies, atomic and subgroup appends, `emit` none / primary / mirror all the same. `restir-check check=gi scheduler=wavefront` reproduces the megakernel's table to the printed digits (static camera, 128 frames, 6 seeds, k = 1, 2048-frame depth-2 reference): Cornell gi_off -5.4e-5, gi_ts_unb -1.1e-4 +- 0.7e-4,
gi_ts_bia -0.95%; Sponza-white gi_off -3.7e-4, gi_ts_unb -3.9e-4, gi_ts_bia -5.9%; env-sphere -5.8e-5, -1.9e-4, -1.6% (the same numbers as above: same seeds, same per-pixel images). The chain (`realtime-pipeline scheduler=wavefront restir=gi,giu`, sponza-white, 16 frames, SVGF and none, static and the moving path, references
512 spp): every relMSE (raw / denoised / final, frame 1 / 4 / 16, two path poses) is within 0.2% of the megakernel arm's (ratio 0.9996 .. 1.0021).

Time, wavefront vs megakernel (`realtime-pipeline scenarios=timing timingMin=500`, trace stage, sponza-white 960x540, M5 Pro, three paired rounds megakernel then wavefront, **load 17-35 from other jobs: PROVISIONAL**, no GPU timestamps; wf / mk per round):
plain NEE 0.84 / 0.84 / 0.96, DI 0.98 / 0.93 / 1.09, DI + GI biased 1.05 / 1.00 / 1.14 (26.9 vs 25.7 ms in round 1), DI + GI unbiased 1.06 / 1.01 / 1.16. The wavefront wins only without ReSTIR; with GI it is level to 14% slower (the megakernel's real-time preset also has `sharedHit`, which the wavefront
does not). The reason to run GI there is size (the shade kernel is 38.7 KB against the megakernel's 50.1 KB, 48.3 against 59.5 with OpenPBR), not speed; `ptwfAutoPolicy` is unchanged (V281), its `kernelBytes` rule already routes the over-50 KB megakernels. A QUIET-PASS row exists (P5.6).

## Open problems

1. **One bounce only.** `Lo` is direct light at x2. A full ReSTIR GI sample continues the path from x2 (`Lo` from a longer path, the path length as part of the sample). That needs the megakernel's vertex loop inside the init pass (textures, cutouts, hooks) and re-evaluates the BSDF at x2 toward the new x1 only for Lambert x2 anyway.
2. **Specular chains.** A diffuse x1 -> mirror / glass x2 -> light path is dropped (it is a depth-2 contribution of the plain path tracer; a 1-sample GI cannot reconnect through a delta lobe). Needs a "delta-chain" extension of the sample (Ouyang et al.'s rough-surface reconnection condition, or a second path sample for paths through specular vertices).
3. **Glossy x2 has no reuse** (see shift validity). A footprint rule (reconnect only above a roughness, evaluate the BSDF at the new wo with the stored incoming direction) would give them reuse at a small bias.
4. **Textures and alpha MASK are refused** (`ptRestirUnsupported` says so): x2 needs the texture footprint of the primary cone and the cutout loop. Wavefront: supported (no textures / MASK there either). `materialLod` with OpenPBR materials: refused.
5. **Size.** The GI module is 47.7 KB (50.4 with an environment map, 56.1 with OpenPBR), the shade kernel 43 KB (50.2 KB in the real-time preset with `sharedHit`, blue noise and frame planes): at the DXC class, over it with OpenPBR.
6. History correlation: a static camera accumulates correlated frames; the useful comparison is the per-frame image (below), not the accumulated mean.
