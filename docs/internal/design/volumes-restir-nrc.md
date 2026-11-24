# Volumetric ReSTIR and NRC at medium vertices (V5.5, research design note)

Status: research, 2026-11-24. Nothing built. Tags: [R] read in the cited source, [E] our estimate (untested).
Sources: Lin, Wyman, Yuksel 2021, "Fast Volume Rendering with Spatiotemporal Reservoir Resampling" (TOG 40(6), 278; `graphics.cs.utah.edu/research/projects/volumetric-restir/volumetric_restir.pdf`,
main text only, the supplement was not read; cited as VR §n). GRIS: Lin et al. 2022 (`research/papers/sampling/gris-lin2022.pdf`). NRC: Müller et al. 2021
(`research/papers/denoise/nrc-mueller2021-real-time-neural-radiance-caching.pdf`). Repo: `restir-di.md`, `restir-gi.md`, `volumes.md`, `volumes-plan.md` (V4, V5, V7), `nrc.md`,
`nrc-pipeline.md`, `research/notes/14-volumes-survey.md`, `15-volumes-vs-ours.md`. No NVIDIA SDK code was read.

## 0. Where the noise is (decides everything below)

- At 16 spp the orders 0-1 plane carries 0% of the error on `vol-vdb-wdas-cloud` and 4% on `vol-cornell-fog`: multiple scattering is the noise (`volumes-plan.md`, V7 "Where the noise is").
- V7.1 (forced NEE, equiangular, `marchSamples`) already attacks order 1. V7.3 (split denoising of orders 2+) is in progress; a blur stand-in took Cornell fog from 0.556 to 0.027 at 4 spp.
- So the useful reuse targets are order-2+ paths, not first-scatter light samples. Both ideas below must be judged against V7.3 plus a denoiser, not against raw 1 spp.

## 1. What volumetric ReSTIR does

### 1.1 The sample is a path, not a light sample

- A random walk from the camera: free-flight distance z_i, then NEE at each vertex (a "scattering path" ending at a light), plus an "emission path" per vertex when the medium emits
  (VR §3.2, Fig. 3). K vertices give up to 2K candidates.
- One candidate is picked per walk by RIS, treating the n_j paths as stratified samples with p̄ = p / n (VR §3.3, eq. 19-21). M = 4 walks per pixel (VR §5.3).
- Target p̂ = F̃: F with a cheap transmittance T̃ on the last (NEE) segment only (VR eq. 22-23). It cancels every other T and σ_t in w = p̂ / p, so w = W_k Π σ_s / σ_t (VR eq. 24-25).
- Only the chosen path pays for the exact T in final shading (VR eq. 26).
- The target is a scalar: chroma is sampled identically, so colour noise remains (VR §7, a listed limitation).

### 1.2 Why delta tracking is replaced

- Reuse must re-evaluate p(λ) at another pixel. Delta tracking's pdf T σ_t (VR eq. 18) has no cheap closed form (VR §4.1).
- The paper uses regular tracking through a piecewise-constant volume σ* (VR eq. 27-28), for candidate generation and p only. σ* must be non-zero wherever the true density is, or the
  estimate is biased (VR §4.1). It may be coarser than the render grid; for p̂ in reuse the paper ray-marches mip 1, BC4-compressed, one voxel diagonal per step (VR §5.1).
- Transmittance at three levels (VR Fig. 5, §4.3): T* (piecewise constant) at generation, T̃ (trilinear ray march) as p̂ in reuse, exact T once in final shading. Because p̂_q ≠ p̂_q'
  even at q = q', the reuse weight carries the ratio of different transmittances (VR §4.3, eq. 29).

### 1.3 Reuse

- Path mapping (VR §4.2, Fig. 6): same distance z_1 along the new pixel's primary ray, then "vertex reuse" (x_i = x_i') or "direction reuse" (ω_i, z_i copied). Vertex reuse has an
  unbounded G(x1, x2) = 1 / |x2 - x1|² and gives fireflies; the paper uses direction reuse everywhere. Direction reuse re-marches T* on every later segment (cost grows with path length).
- Spatial reuse (VR §4.3): 3 neighbours within 10 px, low-discrepancy offsets, Talbot (balance-heuristic) MIS. Stochastic MIS was "excessively noisy in volumes" (VR Fig. 7). Normal / depth
  rejection does not work because those features are stochastic in volumes.
- Temporal reuse (VR §4.4, §5.2): motion of x_1 as a particle in the volume's velocity field; "velocity resampling" replaces an x_1 that landed on the background; history cap Q = 4 (10 with surfaces).
- Results: 1 spp per frame vs a decomposition-tracking / residual-ratio baseline at 4-14 spp at equal time (VR §6, Fig. 17-19).

### 1.4 What GRIS says about media

GRIS ("ReSTIR PT" section): "Volumetric media requires a volumetric shift map; Lin et al. [2021] implicitly defines one possibility and Gruson et al. [2018] propose another, though finding
fast volumetric shifts for resampling remains interesting future work." So the 2021 mapping is the only practical shift, and it is not GRIS-grade (no generalized MIS proof, no Jacobian
discussion). No ReSTIR-PT-with-media follow-up was found; consistent with `research/notes/14-volumes-survey.md`.

## 2. What it would reuse and cost here

### 2.1 Reusable from the ReSTIR DI / GI work

| piece | where | status for media |
|---|---|---|
| pass layout Gbuf / Init / Temporal / Spatial in one pruned module (`ptRestirBuild(.., part)`) | `restir-di.md`, `restir-gi.md` | reuse as a third part |
| Talbot MIS combine, per-domain p̂ | `restir-di.md` (bias correction) | reuse; the paper agrees (VR §4.3) |
| previous camera ownership, `PT_RESTIR_KEY` streams (a new key per draw) | `restir-di.md`, CLAUDE.md Traps (P4.2b) | reuse |
| CPU oracle + bias scan (`restir-check biasscan seeds=24+`) | `restir-gi.md`, CLAUDE.md Traps | reuse as the first check |
| frame / motion semantics at a medium primary vertex | V5.1c | reuse for the camera-only motion of x_1 |
| normal / depth rejection | `restir-di.md` | not reusable (stochastic in media) |
| (light index, u as 2 x unorm16) sample | `restir-di.md` | not reusable: a volumetric sample is a walk |

### 2.2 What a sample would store

- A, explicit: z_1, then (ω_i, z_i) per vertex, plus light index and u: ~40 B at K = 3. [E]
- B, replay: store the walk's PCG seed and k (`ptVolSeed`) and re-run the walk at the new pixel (a random-replay shift in GRIS terms). Reservoir stays one vec4u (seed, z_1, W, M | k | flags),
  DI's size; the cost is re-marching T* on every re-evaluation. [E]
- Suggestion: B first. It keeps memory at DI's 14.7 MB per region at 1280x720, inside the 128 MiB binding cap.

### 2.3 What our code already gives

- The brick majorant texture (binding 17, max density per 8^3 brick) is a valid conservative σ*: ≥ density everywhere, never zero where density is non-zero (the VR §4.1 condition).
- Regular tracking through bricks reuses the DDA (`ptDdaInit` / `ptDdaNext`, `volume-wgsl.ts`); a new sampler and its pdf are needed (not delta tracking, not `ptVolSampleBrick`).
- Homogeneous media: σ* = σ_t exactly, the pdf is the closed-form exponential (the forced-NEE pdf in `forced-wgsl.ts`).
- T̃ can be ratio tracking with a fixed step count and a shared random order (Cycles / Misso route, `research/notes/15`), since T̃ is only a target and may be biased.

### 2.4 Bindings and bytes [E]

- G-buffer: DI's 20 B per pixel; for a medium pixel "position" is x_1 and the normal word becomes a flag. Reservoirs: 1 vec4u per pixel per region (option B).
- The shade kernel's final region needs a new group-2 binding (20+; 16-19 are taken). Apple's 10 storage buffers: passes module = walk 3 + scene 1-2 + params + G-buffer + reservoirs;
  the grid and majorants are textures. Fits with a packed scene copy; count bindings in a unit test before any GPU run.
- Kernel bytes: DI passes 41.0 KB, GI 47.7 KB; regular tracking + DDA + T̃ adds ~4-5 KB. The volume shade kernels are already 43.7-50.5 KB: a new gated variant, defaults byte-identical.
- Volumes refuse `restir` today except homogeneous ReSTIR DI (V5.4); `restir.volumetric` would be its own lifted refusal.

## 3. Interaction with our estimator stack

### 3.1 V240 (delta tracking + direction-only MIS)

- V240 is unbiased because MIS weights depend on directions only. Volumetric ReSTIR needs closed-form distance pdfs, so its path samples use regular tracking; the two estimators coexist,
  the non-ReSTIR path is unchanged.
- Chromatic media: our per-channel weights (ss / ssMax) have no counterpart in a scalar p̂. Use the luminance of F̃ for p̂ (as DI / GI do) and keep the per-channel W; greys stay exact,
  saturated σ gets colour noise (VR §7).
- Exact T in final shading can be our stochastic ratio tracking: unbiased and independent of the selection.

### 3.2 Forced NEE, equiangular, V7.1 marching

- They estimate the same order-1 quantity. In ReSTIR terms each is a candidate generator with a closed-form pdf; they fit into one RIS, balance-MIS on the source pdf. Their random slots
  (`PT_FORCED_VERTEX_BASE + i * PT_MARCH_STRIDE + depth`) must not be shared with ReSTIR draws (`PT_RESTIR_KEY`).
- V7 two-part estimator: orders 0-1 stay outside ReSTIR; ReSTIR (K ≥ 2) targets orders 2+ only, without double-counting order 1. Needs V7.3's outputs to accept forced / equiangular first.

### 3.3 Where bias creeps in

1. p̂ must be non-zero wherever F is: a compressed / coarsened T̃ can zero a thin filament (darkening). Brick majorants as σ* have no such zero; T̃ needs a floor.
2. Neighbour domains without support: fixed only by Talbot MIS (n² p̂ evaluations, each re-marching T̃); stochastic MIS is noisy, 1/ΣM reads dark.
3. History cap and stale reservoirs (VR Fig. 14); our DI biased mode read -12% at frame 64 on sponza-lamps.
4. Random streams: every draw its own key (DI's shared key gave -15%, P4.2b). Walk replay (option B) makes this stricter.
5. Direction reuse through surfaces: implicit Jacobian (VR §4.2), not blessed by GRIS; the 24-seed bias scan is the only judge.
6. Temporal reprojection mismatch: halos at silhouettes (VR §5.2), a quality issue when MIS is exact.

## 4. NRC at medium vertices

### 4.1 State of the art

- NRC caches surface radiance only (inputs: position, direction, normal, roughness, diffuse + specular reflectance; §3.6); the paper does not mention media [R].
- Neural cloud work uses other inputs (Deep Scattering: hierarchical density features; MRPNN 2023) and is not online-trained. NRC with media: not found (notes 14 and 15).

### 4.2 Inputs for a medium query (our 64-wide layout, `nrc.md` §1b)

| NRC slot | surface | medium vertex (proposal) |
|---|---|---|
| position (frequency encoding) | x | unchanged |
| direction (one-blob 2 x 4) | ω_o | unchanged; at g ≥ 0.85 the 4-bin blob cannot resolve the forward peak |
| normal (one-blob 2 x 4) | shading normal | zeroed + a flag |
| roughness (one-blob 1 - e^-r) | roughness | phase anisotropy g, mapped like roughness |
| diffuse reflectance (3) | albedo | single-scatter albedo σ_s / σ_t (NRC's factorisation multiplies the output by it, §5) |
| specular reflectance (3) | F0 | zero |
| constant ones (2) | | one becomes σ_t (scaled), so the cache can tell optical thickness (a layout change: refit) |

Density features would be a further input for dynamic volumes, not a first step.

### 4.3 Paths

- Termination: NRC stops a path when its spread exceeds c a_0, c = 0.01 (§3.4, eq. 3), using |cos| and the BSDF pdf. In media: the phase pdf, cos = 1, plus a distance term. New, tune by A/B.
- Training: online, 4 Adam steps per frame, EMA weights (§3.3); one suffix in 16 unbiased by roulette (§3.2). Hooks `ptOnVertex` / `ptTerminate` (`nrc.md` §5); the cache returns the
  vertex's whole outgoing radiance including NEE (`nrc-pipeline.md` §2). Volumes refuse hooks today: lifting that (megakernel) is the first step.
- Delta-tracking throughput weights stay outside the cache (applied before the query).

### 4.4 Does it help?

- Dense clouds: yes in principle: walks of tens to hundreds of collisions are truncated, and the field inside a dense cloud is smooth in position. Weak spot: directionally sharp silver lining
  and forward peaks (g 0.85-0.99, droplet phases).
- Thin fog: no (paths end after 1-2 collisions; order 1 is handled by V4.1, V4.2, V7.1).
- Dynamic media (V6.6 runtime params): the cache re-learns (NRC: ~8 frames), but behind SVGF / OIDN a young cache is the dominant low-frequency error (N11.6: ~150 frames for OIDN).
- Biased by design: real-time only; the reference stays the unbiased path tracer.

## 5. Comparison

| | volumetric ReSTIR | NRC at medium vertices |
|---|---|---|
| removes noise from | all orders it reuses | orders beyond the cache vertex |
| new infrastructure | regular-tracking sampler, T̃ march, a third ReSTIR part | the NRC stack on media, a new training-record term |
| bias | controlled by MIS, tested by bias scan | structural (net approximation) |
| failure | stale / incoherent reuse, chroma noise | sharp directional lobes, young cache |
| thin fog | single scatter only (V7.1 has it) | no |
| dense cloud | via multi-order walks, unproven | yes in principle |
| size | L | L |
| precedent | 1 spp beats 4-13 spp baselines (VR §6) | none found |

## 6. Recommendation (maintainer's decision)

Neither is worth building as a full feature now: order 2+ noise is the problem (§0) and V7.3 + a denoiser is the competitor to beat. Do the cheap steps that can fail first.

Volumetric ReSTIR, each step S:
1. R1 CPU oracle (vitest): homogeneous slab, K = 1, M walks, regular tracking, RIS without reuse; equals `slabSingleScatterPhase` within SE; M = 1 equals plain NEE + distance sampling.
2. R2 reuse in the oracle: 2-3 pixel domains, direction reuse, Talbot vs 1/ΣM (the `restir-gi-oracle.ts` pattern): Talbot unbiased within 1%, 1/ΣM visibly dark.
3. R3 GPU, homogeneous, K = 1, spatial only, walk replay: `restir-check` bias scan on `vol-cornell-fog` (24 seeds). Gate: unbiased, and per-frame noise below V7.1 forced + equiangular at equal cost.
4. R4 temporal reuse (x_1 motion from V5.1c), history cap Q.
5. R5 grid media, σ* = brick majorant, T̃ = fixed-step ratio march; `vol-cl-cloud256` at 256 spp.
6. R6 K ≥ 2 and the order split with V7.3: the real payoff and the biggest risk; not before R3 works.

NRC at media (research):
1. N1 lift the hook refusal for volumes on the megakernel; a hook returning 0 equals the plain image bit for bit.
2. N2 offline fit, no GPU: export vertex records (position, ω, albedo, g, σ_t, radiance) on `vol-cl-sph-g85-a99-t200` and `vol-cl-cloud256`, fit the MLP with the CPU replay
   (`mlp-fit oracleSteps=`), held-out relMSE at g 0 vs g 0.85. Answers "can the net represent it" in minutes.
3. N3 only if N2 reaches relMSE ≤ 0.05 at g 0.85: termination rule and query on the megakernel, `nrc-pipeline-study`-style checks. Otherwise record the negative and stop.

Stays research: NRC in media beyond N3, GRIS-grade volume shifts, per-channel p̂, density-feature inputs, ReSTIR GI with media.

The A/B that decides which gets more work: equal-time per-frame relMSE (not accumulated) on `vol-cornell-fog` and `vol-vdb-wdas-cloud` at 1 and 4 spp against (a) V7.1 forced + equiangular,
(b) V7.3 split denoise; relBias next to any biased mode; timings provisional (QUIET-PASS).

## 7. Open questions

- Store the seed or the vertices (§2.2).
- Reuse only above some optical depth (a thin ray in a thin medium has nothing to reuse).
- Speed vs bias of T̃ (step size, mip level; VR §4.3 tunes the step to the resampling budget).
- Read the VR supplement (storage layout, derivation of eq. 20) before R1.
