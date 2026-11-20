# Quiet pass 2026-11-14 15:43, trace step only (M5 Pro, Chrome 157, commit 7056f80 + P1.6a batch timing)

Rerun of P1.6 with trace-timing's new batch mode (P1.6a): back-to-back passes of one dispatch in one command buffer,
≥0.5 s per sample, 6 samples per run, 5 ABBA rounds per A/B; sponza-white, 1 M rays. Other apps closed except the Claude
app; load (1 min) 1.8–2.2 during the runs (9.1 at the start from the build agents finishing, settled before the first
timed run: the step waited on a stale lock until 15:54). Every run valid, results identical to the single-pass snapshot.

Paired B/A batch throughput per round, min..max, and the worst within-run spread (max/min of 6 samples):

| A/B | primary | bounce | shadow (L0) | AO (L0) | worst spread |
|---|---|---|---|---|---|
| workgroup 128 / 64 | 0.996..1.003 | 0.996..1.002 | 1.000..1.004 | 1.000..1.004 | 1.003 |
| SoA / AoS rays | 0.998..1.004 | 0.997..1.002 | 0.991..0.999 | 0.993..1.000 | 1.002 |
| compact / full hit record | **0.956..0.963** | **0.922..0.928** | 0.996..1.007 | 0.993..1.010 | 1.002 |

Absolute (median Mrays/s, A): primary 284, bounce 136, shadow 163, AO 235.

**Reading:**
- Workgroup size: no difference (±0.4%).
- Ray layout: no difference on closest-hit rays; SoA is up to 1% slower on the any-hit sets.
- Hit record: the compact record is 4% slower on primary and 7.5% slower on bounce rays (the sets that write it); the
  control sets that don't write it don't move. Packing costs more than the bytes it saves on this GPU.
- **Suggestion for the maintainer:** keep the current defaults (workgroup 64, AoS, full record); they are the best or tied
  on every set. The 3060 may differ (NVIDIA was bandwidth-bound in bvh's record tests): rerun there before closing
  open choices 1–3 in docs/internal/design/trace-core.md.
