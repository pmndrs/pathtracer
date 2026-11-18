# Track N back on the Mac: handoff from the RTX 3060 agent (2026-11-18)

_Written on Windows at the end of M9 and M8. The reverse of [nrc-windows-handoff.md](nrc-windows-handoff.md). Read
[CLAUDE.md](../../CLAUDE.md) (rules, licences, Traps; the "Windows:" traps do not apply on the Mac) and
[nrc-plan.md](nrc-plan.md) first. The one-page summary of the whole track is [bench/results/NRC.md](../../bench/results/NRC.md)._

## Where things stand

Every milestone of [nrc-plan.md](nrc-plan.md) is done (M0-M9). The maintainer accepted the write-up (M8) on 2026-11-18.
What is left is Mac-side reruns, the quiet pass and optional follow-ups.

Branch `windows-nrc` was fast-forwarded into `main` on 2026-11-18 on the maintainer's go (typecheck clean, 1152 unit
tests pass on Windows). No file in the index has CRLF line endings.

## Built on Windows, never run on Apple

| Piece | Where | Check on the Mac |
|---|---|---|
| Hash-grid cache (M7, SHaRC-like, no network) | `bench/nrc/hash-grid.ts`, `NrcSettings.model: 'hash'`, `nrc-study model=hash`, demo "Hash grid" entries | `node bench/runner/run.ts --page nrc-check --args model=hash` (closed room), then N7b below |
| Real per-vertex roughness in the training/query records (N-I10) | `nrcRoughness` in `bench/nrc/cache.ts` | Changes the network's input on every scene with a metal: the Mac's sponza-white numbers are unaffected (no metals), glossy ones are new |
| `sponza-glossy` (metal floor 0.15, metal columns 0.3) | `bench/scenes/tier2/sponza-glossy.ts` | `node bench/adapters/export-scenes.ts sponza-glossy` before any page uses it |
| M9 chain: SVGF and OIDN arms, warm-up gate (16 steps per gated frame, D30) | `bench/nrc/pipeline*.ts`, `nrc-pipeline` demo, `nrc-pipeline-study` twin | needs `pnpm install` for the `@pmndrs/upscaler` link (sibling checkout) |
| `Caps.subgroupMatrixConfigs` (N0.2) | `packages/raytracer/src/caps/detect.ts` | `pnpm test`; the Mac should list f32→f32 and f16→f16 8×8×8 |
| Viewer report of the A/B images | `bench/nrc/nrc-report.ts` | built from the Windows runs' git-ignored PFMs: on the Mac, point `RUNS` at the Mac's own runs |

## Mac tasks, in order

1. **Pull** `main` (already merged), `pnpm install`, `pnpm typecheck`, `pnpm test` on the Mac.
2. **N11.4**: the M9 twin against the Mitsuba reference (the PFM is on the Mac):
   `node bench/runner/run.ts --page nrc-pipeline-study --timeout 3600000`. Confirms or corrects the 3060's numbers, which
   used our own 4096 spp render as the reference.
3. **N7b, Mac half**: hash grid vs network with subgroup-matrix inference (the Mac's cache frame is 0.61 of a plain
   sample, against 1.04 on the 3060, so Q4's per-time answer may flip there): `nrc-study model=hash` and the network run,
   on `sponza-white` and `scene=sponza-glossy`. Add a Mac column to NRC.md Q4.
4. **N2r-e**: the blocked subgroup-matrix kernels on Apple (8×8 f16→f16) against `row`, then decide the defaults
   (DECISIONS.md); constant-trip lane loops for the portable kernels.
5. **Quiet pass** (maintainer, [QUIET-PASS.md](../QUIET-PASS.md)): N11.1 (the M9 chain timing on both GPUs), N2r-e
   timings, N7d (hash resolve over active cells only).

## Open but not tied to a machine

N7a (hash: continue the path on a miss), N11.2 (upscaler-owned jitter), N11.5 (upscaler darkens 2.5-4%), N11.8 (ramp
instead of the warm-up switch), N11.9 (saved network per scene), N11.10 (cut into an unseen region under OIDN).
N9 (posting the subgroup-matrix feedback to gpuweb) is the maintainer's call. Details in TASKS.md section N.
