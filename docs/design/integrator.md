# Megakernel path tracer v1 (P2.4)

`@pmndrs/pathtracer`, `packages/pathtracer/src/integrator/`. Bench adapter `pt-megakernel` (not in the default competitor
set). Tier 0 agreement: `bench/results/reports/pt-tier0/PT-AGREEMENT.md` (viewer:
`?report=/results/reports/pt-tier0/report.json`).

## Shape

- One compute dispatch per sample pass, one thread per pixel, 8×8 workgroups, no sample loop inside a thread (no
  dispatch outlives a watchdog; a pass is the unit of progress).
- Passes of one `encode` share a command encoder and take per-pass uniform records from a 1024-slot ring
  (`hasDynamicOffset`): `queue.writeBuffer` is ordered with `submit`, so a slot is never rewritten before its pass ran.
- Accumulation: f32 `array<vec4f>` (sum, w = sample count). Readback only through `readImage`, outside timed regions.
- Bindings: bvh walk group 0, scene copy group 1, group 2 = `ptParams` (uniform, dynamic offset) + `ptAccum`: 6 storage
  buffers.

```ts
const pt = await PtMegakernel.create(device, caps, { format: sceneFormatOf(trace), layout: copy.layout,
  integrator: { sampler, nee, envSampling, sunVisible, vndf, transport }, minify });
pt.setScene(trace, sceneCopyBuffers); pt.setFilm(w, h); pt.setCamera(cameraDesc);
pt.setLimits({ maxDepth, rrStartDepth }); pt.setSeed(n);
pt.reset(encoder); pt.encode(encoder, passes);
await pt.readImage();
```

## Per path (pbrt-v4 `PathIntegrator`)

Closest hit → emission at the hit (weight 1 for camera and delta paths, else MIS with
`p_light = pmf · dist² / (|cos| · area)`) → black material or `depth ≥ maxDepth` ends → NEE at non-delta vertices →
BSDF sample, `beta *= f·|cos|/pdf`, `etaScale *= etap²` on transmission → Russian roulette
(`q = max(0, 1 − max(beta·etaScale))` when `depth ≥ rrStartDepth`; off when omitted). Mitsuba `max_depth = maxDepth + 1`.

- **NEE:** pick a light (`rtSampleLight`), sample a point/direction (`rtSampleLightLi`), evaluate the BSDF, trace one
  `rtVisible` segment; power heuristic computed as `1/(1 + (other/this)²)` with the ratio clamped to 1e18 (no f32
  overflow). Delta sun unweighted.
- **Offsets:** Wächter-Binder along the geometric normal on the side of the new direction; shadow rays end at an offset
  target point.
- **Radiance and beta** are vec4 (`PtRadiance`, spec §6).
- **Camera:** pinhole per schema (`yFovDeg` vertical, look-at, +Y up, right-handed); one uniform jitter per pixel per
  sample (box 0.5); image row 0 = top.

## Sampler

Owen-scrambled Sobol pairs (`dimensions.ts`; WGSL constants generated from the same table): pair 0 pixel jitter, pair 1
reserved (lens/time), then per vertex `v` pairs `2 + 4v + k`: k=0 light point, k=1 BSDF direction, k=2 (light select,
BSDF component), k=3 (RR). Each pair has its own seed, shuffle (16-bit; the block number joins the seed) and Owen
scramble, so the first 2^m samples of every pair are a (0,m,2)-net. `pcg` is the debug setting.

## Settings

Pipeline constants: `sampler` (owen-sobol | pcg), `nee`, `envSampling` (uniform | none), `sunVisible`, `vndf`,
`transport`. Per dispatch: size, sample index, seed, `maxDepth`, `rrStartDepth`, camera. Bench page args: `pt_sampler`,
`pt_nee`, `pt_env`, `pt_sun_visible`, `pt_positions`, `pt_vndf`, `pt_transport`, `pt_maxdepth`, `pt_rr`, `pt_seed`,
`pt_passes` (`matrix.ts --extra k=v,...`).

## Agreement (1024 spp, M5; timings best effort)

| Scene | relMSE vs Mitsuba | vs pbrt | pbrt-vs-Mitsuba floor | Notes |
|---|---|---|---|---|
| furnace-diffuse | 1.96e-7 | 1.71e-7 | 2.44e-8 | mean 0.50000 (expected 0.5) |
| furnace-dielectric | 1.93e-7 | 1.26e-7 | 1.53e-7 | mean 0.99990 at maxDepth 1000 |
| furnace-conductor | 2.42e-5 | 7.99e-6 | 3.15e-5 | 0.69285 vs pbrt 0.69286; Mitsuba 0.69040 (its Smith G) |
| quad-light | 4.72e-5 | 4.79e-5 | 1.84e-6 | closed form over the plane to 9e-5 |
| cornell | 7.25e-5 | 4.82e-4 | 4.39e-4 | the 0.69% vs pbrt is pbrt's spectral uplift; Mitsuba RGB agrees with us |

0 non-finite pixels. GPU twins (`pt-check`): sampler bit-identical, adapter 0 mismatches, light sampling max rel 1.3e-5,
end-to-end sun/sky plane within 0.4% of a closed form.

Settings A/B at 1024 spp (relMSE vs Mitsuba): Owen-Sobol beats PCG 2.6–2.8× on Cornell and quad-light; NEE off is 34×
(quad-light) and 126× (Cornell) worse with the same means (MIS unbiased); `envSampling=none` puts furnace-diffuse at the
floor (zero variance); `positions=ray` and `vndf=cap` identical on Tier 0; `rrStartDepth=4` on Cornell: same mean,
3.9 s vs 14.2 s, relMSE 1.13e-4 vs 7.25e-5.

## Size

52.7 KB minified (55.0 KB indentation-stripped), over the ~50 KB target: bvh walk + `rtVisible`/`rtClosest` 21.1 KB,
scene copy 10.5 KB (surface 3.5, materials + lights 7.0), light sampling 3.4, BSDF 10.8, integrator 5.8, sampler 1.3,
Owen-Sobol 1.1, adapter 0.9. `rayMasks: false`. ~250 ms compile on the Mac; DXC unmeasured (P2.10). A unit test caps it
at 54 KB.

## Open choices (defaults hold until measured)

1. **Kernel size over 50 KB.** Keep; measure DXC compile on the 3060 first (W10). Cuts touch other owners' code (bvh
   walk, scene decode); possible trims: lean material decode (~1.8 KB), light-record decode, closest/any specialisation.
   No speed/quality effect from text cuts. Settle: `compile_ms` on Windows.
2. **`envSampling` default: uniform + MIS or none (pbrt).** Keep uniform until a sun-plus-sky scene shows the cost.
   `none`: zero variance on a uniform sky and cheaper NEE; uniform wastes alias picks next to other lights. Setting
   exists. Settle: P2.12 at equal spp and equal time.
3. **Light selection weights** (P1.4f): run now; quality only.
4. **Triangle lights: uniform area or spherical triangle.** Keep uniform (matches Mitsuba). Spherical is less noisy close
   to big lights, more ALU. Can be a module constant. Settle: close-up large quad light, relMSE at equal spp.
5. **RR outside reference mode.** Suggest `rrStartDepth` 4 (Cornell 3.6× faster at 1.6× relMSE). Setting exists.
   Settle: time to equal error in the quiet pass (P2.13).
6. **One sample per dispatch vs several per thread vs tiles.** Keep one per dispatch (watchdog-safe). Settle in P3 with
   `pt_passes` and spp-per-thread.
7. **4 dimension pairs per vertex** (RR pair half used). Keep; small cost next to traversal.
8. **Hit position from barycentrics** (P1.8). Keep; identical on Tier 0, re-test on Sponza scale.
9. **Sponza in the adapter (P2.9):** load the glb in the page (matches the web exporter path).
