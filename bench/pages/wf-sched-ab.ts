/**
 * Scheduling A/B in ONE page (P3.3 / P3.4 / P3.10 / R1b): several of our renderer ids on one scene, interleaved ABBA inside the page, so a run takes the shared GPU
 * lock once instead of once per (renderer, round) like `matrix.ts --interleave`. The first id is the baseline of the paired ratios.
 *
 *   node bench/runner/run.ts --page wf-sched-ab --suite wf-sched-ab --timeout 1800000 --args scene=cornell renderers=pt-megakernel,pt-wavefront,pt-wf-sg spp=256 rounds=5 [pt_layers=2 pt_maxdepth=4]
 *
 * Metrics: `<id>_ms_per_spp_min / _med / _max`, `<id>_ratio_min / _max` (paired against the first id, round by round), `<id>_max_rel_pixel` (the final image of round 0 against the
 * baseline's, per pixel: the correctness check of the very same runs; a variant that is not mathematically the same image shows up here) and `<id>_non_finite`.
 * The wall time is the adapter's (`render()`: submit + `onSubmittedWorkDone` + `finish()` drain, no readback). One warm-up render per id before the rounds.
 */
import { getArgs, numberArg, reportError, reportResult } from '../runner/page-client.ts';
import { createAdapter } from '../adapters/registry.ts';
import { getScene } from '../scenes/index.ts';

try {
  const args = getArgs();
  const ids = (args['renderers'] ?? '').split(',').filter(Boolean);
  const sceneName = args['scene'];
  if (ids.length < 2 || !sceneName) throw new Error('args scene=<name> and renderers=<id,id,...> (at least two) are required');
  const spp = numberArg(args, 'spp', 64);
  const rounds = numberArg(args, 'rounds', 5);
  const pt = {
    ...(args['pt_maxdepth'] ? { maxDepth: numberArg(args, 'pt_maxdepth', 1) } : {}),
    ...(args['pt_layers'] ? { layers: numberArg(args, 'pt_layers', 1) } : {}),
    ...(args['pt_refill_threshold'] ? { refillThreshold: numberArg(args, 'pt_refill_threshold', 0.25) } : {}),
  };
  const scene = getScene(sceneName);
  const adapters = [];
  for (const id of ids) {
    const adapter = await createAdapter(id, { pt });
    await adapter.load(scene, { baseUrl: '/bench/assets' });
    adapters.push(adapter);
  }
  const first: (Float32Array | null)[] = ids.map(() => null);
  const channels = [4];
  for (const [i, adapter] of adapters.entries()) {
    const r = await adapter.render({ spp, snapshotsAtSpp: [] }); // warm-up (pipelines, caches); its image is the correctness sample
    first[i] = r.image.data;
    channels[0] = r.image.channels;
  }
  const ms: number[][] = ids.map(() => []);
  for (let round = 0; round < rounds; round++) {
    const order = ids.map((_, i) => i);
    if (round % 2 === 1) order.reverse(); // ABBA
    for (const i of order) {
      const r = await adapters[i]!.render({ spp, snapshotsAtSpp: [] });
      ms[i]!.push(r.timings.wallMs / spp);
    }
  }
  const metrics: Record<string, number> = { spp, rounds };
  const stat = (v: number[]) => ({ min: Math.min(...v), max: Math.max(...v), med: [...v].sort((a, b) => a - b)[Math.floor(v.length / 2)]! });
  const base = first[0]!;
  const ch = channels[0]!;
  for (const [i, id] of ids.entries()) {
    const s = stat(ms[i]!);
    metrics[`${id}_ms_per_spp_min`] = s.min;
    metrics[`${id}_ms_per_spp_med`] = s.med;
    metrics[`${id}_ms_per_spp_max`] = s.max;
    const ratios = ms[i]!.map((v, r) => v / ms[0]![r]!);
    metrics[`${id}_ratio_min`] = Math.min(...ratios);
    metrics[`${id}_ratio_max`] = Math.max(...ratios);
    const img = first[i]!;
    let nonFinite = 0;
    let maxRel = 0;
    let mean = 0;
    for (let p = 0; p < img.length; p += ch) {
      for (let c = 0; c < 3; c++) {
        const a = base[p + c]!;
        const b = img[p + c]!;
        if (!Number.isFinite(b)) { nonFinite++; continue; }
        mean += Math.abs(a) / (img.length / ch);
        if (Number.isFinite(a)) maxRel = Math.max(maxRel, Math.abs(b - a) / Math.max(Math.abs(a), 1e-3));
      }
    }
    metrics[`${id}_max_rel_pixel`] = i === 0 ? 0 : maxRel;
    metrics[`${id}_non_finite`] = nonFinite;
  }
  reportResult({ scene: sceneName, renderer: 'wf-sched-ab', config: { renderers: ids, spp, rounds, ...pt }, valid: true, metrics, notes: { perRound: Object.fromEntries(ids.map((id, i) => [id, ms[i]!.map((v) => Number(v.toFixed(4))).join(' ')])) } });
  for (const a of adapters) a.dispose();
} catch (error) {
  reportError(error);
}
