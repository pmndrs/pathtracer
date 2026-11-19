# NRC reflection consumer on the Mac

The Reflections raytracer provider can supply a mirror hit to a diffuse hash. The consumer evaluates the hit's full direct lighting, adds strictly indirect cached outgoing radiance, and applies the originating mirror's Fresnel factor once. On the offscreen card, the cache reduced reflection-crop relMSE from 0.0558 with full direct lighting alone to 0.00641. It used 3.00 rays per pixel across the mirror-only view, compared with 4.49 for the regular path tracer. The existing environment hit shader used 2.00 rays per pixel and missed the fixture's area lighting and interreflection. Raw path tracing at the same sample count had lower error than the cache in this view.

## Transport gate

The existing L2 hit shader takes a full next-event sample. The megakernel instead weights next-event estimation and BSDF-sampled direct light with MIS. Adding a megakernel-style tail to the full L2 estimate double counts the part of direct light that the BSDF would sample. For a Lambert hit, the bench estimates strictly indirect light by continuing cosine-sampled paths to later surfaces and taking full next-event samples there. The immediate sky or emitter contribution belongs to the receiver's full next-event estimate. Emission and sky reached after a delta step remain in the tail because a delta step cannot sample them by next-event estimation. [Transport shader](../nrc/reflection-transport.ts) · [Transport gate](../pages/nrc-reflection-transport.ts)

The Lambert receiver and independent megakernel both used a 48 × 48 film and 4,096 samples per pixel. Relative bias uses all RGB channels in linear HDR. The column with the MIS complement adds only the measured first-edge BSDF-sampled direct term to the otherwise correct estimate.

| Lighting | Full direct alone bias | Full direct + strict indirect bias | With MIS complement added | Strict relMSE |
|---|---:|---:|---:|---:|
| Constant sky | +0.001% | +0.001% | +82.29% | 4.04e−9 |
| Area emitter | +0.004% | +0.004% | +2.99% | 0.0000430 |
| Area emitter and red bounce wall | −7.17% | +0.17% | +3.43% | 0.00191 |

The sky produced the closed-form receiver RGB (0.20, 0.15, 0.12). Strict indirect was zero under both the sky and the isolated area emitter. With the bounce wall, the strict indirect mean RGB was (0.03592, 0.00562, 0.00175). The bench trains its own hash on this strict suffix. The existing NRC training target includes MIS-weighted transport and requires separate accounting before a consumer can add L2 direct lighting to it. [Path integrator](../../packages/pathtracer/src/integrator/integrator-wgsl.ts) · [L2 hit shading](../../packages/raytracer/src/shade/shade-wgsl.ts)

## Provider comparison

The bench calls Reflections' existing `traceFirstHit` provider for the camera and specular segments and the checkout's `environmentHitShading` for the environment arm. It uses the real geometry, material, and light tables on the provider. Each trained record stores outgoing indirect RGB generated after the receiver. One 32,768-cell world-space hash (1,310,804 bytes) accepts those records from visible diffuse surfaces and mirror-hit diffuse surfaces. The hit shader reads its resolved snapshot. Cell size targets four pixels with two coarse levels and a 16-sample acceptance threshold. A miss traces the complete strict suffix. A second mirror also continues explicitly. [Consumer page](../pages/nrc-reflection-consumer.ts) · [Provider loader](../nrc/reflection-provider.ts) · [Hash grid](../nrc/hash-grid.ts)

Chrome 157 headless on Apple M5 Pro, 128 × 72, 128 training frames, 256-spp probes, and independently seeded 4,096-spp path-traced references. Reflection crops exclude visible diffuse primary surfaces. Errors are linear RGB relMSE with denominator reference RGB squared plus 0.01.

| Fixture | Environment hit shader | Full direct only | Full direct and traced indirect | Full direct and hash | 256-spp PT |
|---|---:|---:|---:|---:|---:|
| Offscreen blue card in mirror | 0.52226 | 0.05579 | 0.01099 | 0.00641 | 0.00223 |
| 8 mm partition, red face | 0.46485 | 0.09279 | 0.00914 | 0.00428 | 0.00187 |
| 8 mm partition, blue face | 0.46486 | 0.09245 | 0.01072 | 0.00559 | 0.00220 |
| Card before move | 0.74074 | 0.09800 | 0.01897 | 0.00452 | 0.00554 |
| Card moved, stale cache | 0.71988 | 0.09299 | 0.01854 | 0.02994 | 0.00529 |
| Card moved, 64 updates | 0.71988 | 0.09299 | 0.01854 | 0.00590 | 0.00529 |
| Red-to-blue cut, first view on blue | 0.46486 | 0.09245 | 0.01072 | 0.01072 | 0.00220 |
| Red-to-blue cut, 64 blue-side updates | 0.46486 | 0.09245 | 0.01072 | 0.00638 | 0.00220 |

The environment arm is Reflections' no-probe shader. The fixtures use area emitters and a dim constant sky. Reflections' scene probes and screen-space integrations were outside this experiment. Full direct alone was dark by 30.6% in the offscreen crop, by 41.6–42.2% at the partition, and by 33.1% after the card moved. The warmed hash biases on those crops were −0.82%, +1.27% and −0.56%, and −0.11%, respectively.

The moving fixture calls the provider's `setTransforms`, which updates traversal and shading, then renders a new same-camera path-traced reference. The stale cache accepted 85.17% of diffuse queries, including 13.05% at coarse cells, and was 2.97% dark. After 64 updates, it accepted all queries and the coarse share fell below 0.1%. The cut warms only the red side before switching cameras. The blue side initially had zero usable cache cells and matched the strict trace fallback. Blue-side coverage rose to 99.995% after 64 updates. Counts and geometric confidence were checked independently from cache acceptance. [Consumer shader](../nrc/reflection-consumer-wgsl.ts)

The offscreen view showed the floor on all camera rays. The reflection segment hit the card on 72.4% of mirror samples. Its center-ray hit distance agreed with the analytical plane intersection to 5.8e−7 m. Across all shading arms the fixture checks identical hit distances and geometric confidence values. The sky-only mirror gave confidence 0, a miss distance of −1, and reflected sky within 3.2e−7 absolute RGB of the Fresnel calculation. The one-sided emitter behind one mirror agreed within 6.4e−5 absolute RGB. A separate two-mirror path to an emitter agreed with the product of both Fresnel factors within 1.2e−4 absolute RGB. Those checks count terminal emission even when no diffuse cache query is involved.

| Offscreen card | Rays per pixel | Wall time p50 / p95 |
|---|---:|---:|
| Environment hit shader | 2.00 | 0.30 / 0.60 ms |
| Full direct only | 3.00 | 0.30 / 0.60 ms |
| Full direct and traced indirect | 4.79 | 0.40 / 0.80 ms |
| Full direct and frozen hash | 3.00 | 0.30 / 0.60 ms |
| Plain PT | 4.49 | 0.40 / 0.70 ms |
| Strict-suffix training and hash update | — | 0.40 / 0.70 ms |

Wall times measure command encoding, submission, and queue completion, with diagnostic counters enabled in the render probes. The offscreen view consists entirely of the mirror and provides a comparable ray count. The other views shade only the reflection plane in the consumer while their PT arm renders the whole image. The small, headless runs have approximately 0.1 ms timing granularity and do not establish a production speed improvement. The benchmark is limited to delta conductor origins and Lambert receivers. It leaves rough GGX origins, probe-backed shading, and denoised or upscaled output for subsequent comparisons.

## Reproduction

Run from the pathtracer root with the historical Reflections provider checkout at revision `36e206a`. The browser runner serves that checkout and keeps its usual shared GPU lock. Both pages can upload linear PFM crops and references.

```sh
refl_checkout="$PWD/../reflections/.claude/worktrees/p1.5-ab-baseline"
REFLECTIONS_REPO="$refl_checkout" node bench/runner/run.ts --page nrc-reflection-transport --gpu apple --no-csv --args reflections="$refl_checkout" size=48 spp=4096 refSpp=4096 upload=1
for fixture in mirror emitter thin moving cut delta miss; do
  REFLECTIONS_REPO="$refl_checkout" node bench/runner/run.ts --page nrc-reflection-consumer --gpu apple --no-csv --args reflections="$refl_checkout" fixture="$fixture" res=128x72 spp=256 refSpp=4096 warm=128 upload=1
done
```

Saved runs under `bench/results/incoming` are `20261119T101803Z-nrc-reflection-transport` and `20261119T101913Z`, `101916Z`, `101918Z`, `101921Z`, `101925Z`, `101928Z`, and `101931Z` on November 19 UTC with the `-nrc-reflection-consumer` suffix, in the fixture order above.
