# Track N handoff: cloud session → local Mac agent (2026-11-15)

Branch `claude/eager-brown-8uj9yy`. Research track N (a neural radiance cache on WebGPU) was started in a GPU-less
cloud session. Everything from here needs a real GPU, the `~/Developer/bvh` sibling or the reference renders, so the
local agent on the M5 Pro owns the track from now on. Read this file, then [nrc.md](nrc.md) (spec, plan, Mac
checklist), [nrc-diverges.md](nrc-diverges.md) (every difference from the paper and VkNRC, with follow-ups) and
TASKS.md section N.

**Update 2026-11-15:** the "First session on the Mac" list below has been run. Results are in nrc.md (Status, "First
Mac session") and TASKS N0.2/N1.1/N2/N3/N10; the "Verified vs not" table below is the cloud session's view.

## What exists

| Piece | Where | State |
|---|---|---|
| Spec + plan | `docs/design/nrc.md` | Paper numbers marked [M] are from memory: check them against `research/papers/gi/nrc-*.pdf` (N0.1) |
| Divergence register | `docs/design/nrc-diverges.md` | 30 rows, each with a follow-up; DIVERGES V139–V142 point here |
| Subgroup-matrix status + wishlist | `docs/design/subgroup-matrix-feedback.md` | **Draft. Never post anything to gpuweb or anywhere else without the maintainer (N9, D8).** |
| MLP engine | `packages/mlp` (`@pathtracer/mlp`, private, WGSL prefix `nn`) | see below |
| Bench pages | `bench/pages/mlp-{check,fit,timing,sgm-probe,live}.ts`, shared `mlp-common.ts`, `mlp-image.ts` | all run on SwiftShader; none has run on a real GPU |
| VkNRC reference | read-only clone `research/sources/native/vknrc` (re-clone from MANIFEST); native build only under `.cache/native/` | no licence: follow its design, never paste its code (D29); native build allowed (D28) |

Engine (`packages/mlp/src`): `config.ts` (VkNRC-shaped defaults: 64 → 5×64 → 3, ReLU, no bias, Adam 0.002), `encoding.ts`
(triangle-wave frequency + quartic one-blob, CPU and WGSL), `oracle.ts` (f64 forward/backward/Adam/EMA), `wgsl.ts`
(portable f32/f16 kernels), `wgsl-sgm.ts` (subgroup-matrix inference + emulated twin), `engine.ts` (`MlpEngine`, GPU-side
counts, indirect dispatch, no readback). 22 unit tests (`pnpm test`).

## Verified vs not

| Claim | Evidence | Where it could still break |
|---|---|---|
| f32 portable kernels are correct | SwiftShader vs oracle: inference 2.3e-7, gradient 3.9e-7, Adam p99 6.9e-5 (rel); 12-step training replay p99 1.1e-5 | nothing known |
| sgm kernel logic (tiling, transposed weight view, output padding, N subgroups, ragged counts) | emulated twin vs oracle on SwiftShader: 8³ and 16³, 1/2/4 subgroups, 77/200 records, ≤ 4e-7 | the **native** form has never compiled |
| f16 portable kernels | none (SwiftShader has no `shader-f16`) | f16 delta underflow (N-D10), accuracy tolerance in `mlp-check` |
| Live page | rendered offscreen + training loop in headless SwiftShader | canvas presentation (crashes SwiftShader), UI on a real screen |
| Image fit quality | 64², batch 1024, 512 steps: relMSE 25 → 1.5, PSNR 13.6, stripes and checker blurred | just too few steps? Or the HDR relative-L2 loss / lr. Check at 256², batch 16384, 1024 steps |
| Any timing | none | everything |

## First session on the Mac (in this order)

Quiet-machine rules apply to every timing line (CLAUDE.md: load average, foreign GPU processes, min..max).

1. `git fetch origin claude/eager-brown-8uj9yy && git checkout claude/eager-brown-8uj9yy && pnpm install && pnpm test`
2. `node bench/runner/run.ts --page timer-calibration` (once per session).
3. **Probe (N0.2):** `node bench/runner/run.ts --page mlp-sgm-probe --json probe.json`. Record the configs, subgroup
   sizes and which forms compiled in research note 06 §3.5 and TASKS N0.2. The key question: does `subgroup-id x4`
   compile? (It would also lift the Denoiser's one-subgroup limit; tell the maintainer, do not edit the Denoiser.)
4. **Correctness:** `mlp-check --ab precision=f32,f16`, then `mlp-check --args infer=sgm`, then (if step 3 said so)
   `mlp-check --args infer=sgm subgroups=4`. If the native sgm kernel fails to compile, likely suspects, in order:
   load/store argument rules (stride minimums, `MinStorageSize` of the weight buffer), the matrix type parameter order
   (`subgroup_matrix_left<T, K, M>` etc.; only visible on non-square configs), f16 enables, workgroup-memory size.
   Fix in `wgsl-sgm.ts`, keep the emulated twin in step (`infer=sgm-emu` must still pass), and note the cause in
   CLAUDE.md Traps.
5. **Fit:** `mlp-fit --args oracleSteps=8 batch=1024 precision=f16`, then `mlp-fit --ab precision=f32,f16 --repeat 2`
   and look at the PFMs in the viewer. If the fit still blurs the stripes at 1024 steps, sweep lr (N-D6) before
   anything else.
6. **Timing (N2):** `mlp-timing --ab infer=portable,sgm --args precision=f16 --repeat 3` and
   `mlp-timing --ab precision=f32,f16 --repeat 3`. Expectation from the Denoiser's measurements: on Apple, plain f16
   (~6.6 TFMA/s) may match or beat the matrix path (~3.8). Either answer is a result.
7. **Live demo:** `pnpm --filter @pathtracer/bench mlp:live`, in a Chrome started with `--enable-unsafe-webgpu`; check
   every backend in the picker, the error view and Benchmark.

Then update TASKS (N0.2, N1.1, N2, N3, N10), nrc.md "Status", nrc-diverges.md statuses, and STATUS.md if the picture
changed.

## After that (suggested order)

1. **N1.2** faster portable kernels: several threads per record (the 64-float private arrays likely spill) and a
   tiled weight-gradient pass. The current kernels are a correctness baseline.
2. **N3** finish: measure the round-trip share per layer (store → ReLU → reload) — that number is the evidence for
   the gpuweb draft's W1/W2 asks. Then **N3b** (subgroup-matrix training), emulated twin first, as for N3a.
3. **N6** the actual demo: diffuse wavefront path tracer on sponza-white over the trace core (`rtClosest`,
   `rtVisible`), spread termination, training paths per subgroup with the self-training tail, inference before
   training each frame, equal-time comparison with plain path tracing against the Mitsuba reference. Settings and
   follow-ups for it are rows N-D4/D5/D8/D11 and N-I1–I8 in nrc-diverges.md. N6.2 adds the OBJ export and a
   "VkNRC lighting" setting.
4. **N0.1** paper check whenever convenient (closes every [M]).
5. Windows (maintainer's laptop): N0.3, N6.1 / W11 (VkNRC native on the RTX 3060; Linux build recipe in CLAUDE.md
   Traps), N2 on NVIDIA. The 3060 is the likeliest place subgroup matrices show a large win.

## Rules that are easy to miss

- D29: VkNRC has no licence. Read it, follow its design, never paste or line-translate its code into the repo.
- D28: building VkNRC natively is allowed under `.cache/native/` only; its local patches are never committed.
- N9 / D8: the subgroup-matrix feedback stays a draft until the maintainer says otherwise.
- `nn*` is the WGSL prefix for `packages/mlp`; keep generated kernels under ~50 KB (Windows DXC).
- Every new departure from the paper, VkNRC or the Denoiser gets a row in nrc-diverges.md with a follow-up.

## Open questions for the maintainer (unchanged)

- Where the MLP engine lives long-term (private `packages/mlp` now; nrc.md §8).
- Whether a flag-only Windows result (Vulkan backend for tensor cores) is acceptable for the demo.
- When and where to post the gpuweb feedback (N9).
