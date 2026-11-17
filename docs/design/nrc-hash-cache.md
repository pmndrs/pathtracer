# Hash-grid cache: the non-neural competitor (track N, M7 / N7)

Question Q4 of [nrc-plan.md](nrc-plan.md): does the neural cache beat a plain hash-grid radiance cache (SHaRC-like) in the
same time? This note says what we built and how the A/B is set up. Results go to
[NRC-STUDY.md](../../bench/results/NRC-STUDY.md) ("Hash grid vs network").

## 1. One cache class, two stores

The hash grid lives inside `NrcCache` (`bench/nrc/cache.ts`) as `model: 'hash'` (`bench/nrc/hash-grid.ts`), not as a
separate renderer. Everything except the store is the same code:

| Piece | Network (`model: 'mlp'`) | Hash grid (`model: 'hash'`) |
|---|---|---|
| Render paths | terminate at vertex `queryDepth` (1), append a query record | same |
| Training paths | inline: every 16th 8 × 8 tile, vertices 0..M−1 (M = 4) recorded, gathered radiance summed per vertex | same |
| Targets | target_k = (gathered after k + β_M · cache(x_M)) / β_k, cache as its own tail | same (the tail is a grid lookup) |
| "Inference" | MLP forward over the query records | grid lookup over the query records |
| "Training" | `stepsPerFrame` Adam steps on the frame's records | one update (add every record into its cell) + one resolve |
| Into the image | resolve: `β · max(prediction, 0)`; fix-up for training pixels | same |

So both arms see the same paths, the same records and the same termination, and the A/B isolates the store. That is
deliberately narrower than "SHaRC vs NRC": SHaRC's own path logic differs (§3), and those differences are listed as
divergences to try next, not mixed into the first answer.

## 2. The grid

Sources: Binder et al., "Massively parallel path space filtering" (arXiv 1902.05942); SHaRC's integration notes as
summarised in [research/notes/08-realtime-gi.md](../../research/notes/08-realtime-gi.md) §3.3. Ideas only, no SDK code
(CLAUDE.md licence rules).

- **Cell** = (level, integer cell of size `minCell · 2^level`, normal bin). The level grows with distance to the camera
  so a cell covers about `cellPixels` pixels wherever it is: `size = max(d · pixelAngle · cellPixels, minCell)`,
  `level = floor(log2(size / minCell))` (SHaRC's LOD rule). Normal bin: dominant axis and sign (6 bins), or one bin.
- **Table**: 2^20 slots (default; 40 MiB), buckets of 8. The slot word is a 32-bit checksum of the cell (0 = empty), a
  second hash chain independent of the bucket index; claimed with `atomicCompareExchangeWeak`. WGSL has no 64-bit
  atomics, so SHaRC's 64-bit key becomes bucket index + 32-bit checksum (a false match needs both to collide).
  Lookups scan the whole bucket, so evicting a cell never hides another (no tombstones needed).
- **Update**: each training record adds its target as u32 fixed point (×256, stochastic rounding, clamped at 1000 per
  sample) plus a count. WGSL has no float atomics; this is SHaRC's trick too.
- **Resolve** (every cell, every frame): with n new samples of sum S, `w = w_old · (1 − 1/historyFrames) + n`,
  `L = (L_old · w_old · (1 − 1/historyFrames) + S) / w`. An exponentially weighted sample mean: noise-optimal within its
  window, time constant `historyFrames` frames. The weight ages only in frames with new samples. A cell with no samples
  for `staleFrames` frames is evicted.
- **Pyramid** (`pyramid`, default on): each record also adds into its `fallbackLevels` coarser cells, so the grid is
  multi-resolution and a fallback always finds data. Without it the update writes one level, and a coarser cell exists
  only where records from farther-away vertices happened to use it.
- **Lookup**: the finest of the query's cell and its `fallbackLevels` coarser cells that holds at least `minSamples`
  sample weight; else 0. Optional tangent-plane jitter (Binder; it hurt, §6).
- **Counters** (debug readback, never in a frame loop): misses, coarse hits, dropped records (full bucket), cells in
  use.

## 3. Where this differs from SHaRC (to try next)

Recorded in [nrc-diverges.md](nrc-diverges.md) §D (N-H1..N-H6).

- **Miss → 0, not "continue the path".** SHaRC terminates only where the cell has enough samples and traces on
  otherwise. Ours terminates at `queryDepth` like the network arm. Continue-on-miss needs the lookup inside the render
  hook (the grid bound in the megakernel's hook group: one more storage buffer there).
- **Update from the network's training records**, not from SHaRC's own update paths (every vertex of ~4% of paths,
  radiance propagated back over a few bounces). Ours: every 16th tile (6.25% of pixels), 4 vertices, cache as tail. Close
  in spirit; the ratio and depth are the network arm's.
- **No termination heuristic** (SHaRC: segment longer than the voxel, cone spread for glossy lobes). Fixed depth, like
  the network arm.
- **View-independent radiance per cell**, no material demodulation; the network conditions on direction and albedo. On
  sponza-white (one white diffuse material) this costs nothing; on a glossy or textured scene it would.

## 4. Correctness

`node bench/runner/run.ts --page nrc-check --args model=hash`: the closed emissive room, where every vertex's indirect
radiance is ρE/(1−ρ). Tuned defaults (pyramid, 64 samples): mean 1.99998 vs 2 (1.2e-5) at `queryDepth` 1 and 0, worst pixel
1.7e-5. The first defaults (one level, 1 sample): 0.03% mean, 3.1% worst pixel (2026-11-18, RTX 3060).

## 5. Settings and page args

`hashGridArgs` (page args → `HashGridSettings`): `hgLog2` (20), `hgMinCell` (0.02 m), `hgCellPx` (16), `hgNormals` (1),
`hgHistory` (32 frames), `hgStale` (64), `hgFallback` (3), `hgPyramid` (1), `hgMinSamples` (64), `hgJitter` (0).
Pages: `nrc-check`, `nrc-study` (`model=hash`), `nrc-live` (Cache menu: "Hash grid (no network)", so
`bench/live-bench.ts --backends hash` times it).

## 6. Tuning (sponza-white 640×360, RTX 3060, 2026-11-18)

`nrc-study scenario=adapt res=640x360 warm=150 ref=ours refSpp=1024`, single frames (relMSE against our own 1024 spp
image; one run each, deterministic for a seed). "Worst transient" = the largest single-frame error over cold start
(t 1-64), the cut and the fly-through. Network for comparison: steady 2.51, bias +2.0%, worst transient 2.47.

| Variant | Steady | Bias | Worst transient | Misses after a cut (frame 1) |
|---|---|---|---|---|
| first defaults (8 px, one level, `minSamples` 1) | 2.88 | +0.07% | 10.6 | 34% |
| `minSamples` 16 | 2.67 | −4.2% | 2.67 | 41% |
| 16 px cells | 2.64 | +0.7% | 11.1 | 33% |
| 4 px cells | 4.38 | −1.1% | 7.7 | 36% |
| history 8 / 128 frames | 3.35 / 2.83 | | ~10.5 | |
| one normal bin | 2.79 | | 9.7 | |
| jitter 1 cell | 2.90 | | 46.5 | |
| pyramid, `minSamples` 16, fallback 3 | 2.74 | +0.5% | 2.82 | 3.4% |
| pyramid, 16 px, `minSamples` 16, fallback 3 | 2.62 | +0.8% | 2.83 | 2.0% |
| pyramid, 16 px, `minSamples` 64, fallback 2 | 2.53 | −0.4% | 2.53 | 40% |

At 1280×720 (the study) `minSamples` 64 with 3 fallback levels beat 16 in every row (steady 2.48 vs 2.53, fresh cache frame 4 2.45
vs 2.77, misses after a cut 6.4% vs 1.8%), so 64 is the default. Results against the network: NRC-STUDY.md "Hash grid vs network".

What mattered: blotches of single noisy samples in young cells (after a cut, a cold start, or while the camera moves
and the LOD level of a surface changes) caused every spike; a sample threshold removes them, and the pyramid keeps the
threshold from turning into dark misses. Bigger cells (16 px) help steady state. Jitter makes it much worse (it smears
young cells across neighbours' lookups).

## 7. Through the M9 chain (N7c)

`nrc-pipeline-study model=hash` and the `nrc-pipeline` demo's "Hash grid" cache entry (auto warm-up gate: 10 frames). After SVGF and OIDN
the grid beats the network at every checkpoint, keeps a 1.5× lead over plain under OIDN on an accumulated still view (the network's falls to
1×), and is ahead of plain on every measure from 10 training frames. Tables: [NRC-PIPELINE.md](../../bench/results/NRC-PIPELINE.md) "Hash grid
in the same chain".
