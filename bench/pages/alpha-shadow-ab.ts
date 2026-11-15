/**
 * R2b page: A/B of cutout shadow modes in ONE page (one GPU lock hold, every variant loaded, rounds interleaved with the order reversed every other round).
 *
 *   node bench/runner/run.ts --page alpha-shadow-ab --no-csv --args scene=foliage-mask renderers=pt-megakernel,pt-mk-shadow,pt-mk-roulette,... spp=1024 repeat=5
 *
 * Why not `matrix.ts --interleave`: it takes the shared GPU lock once per run, and on this shared Mac other agents' long pages made 200 runs take hours.
 * Reports `notes.rounds[id]` (wall ms per round), `notes.means[id]`, and uploads `<id>__<scene>` (the image of the last round; the sampler is deterministic, so
 * every round renders the same image). Read it with `bench/adapters/alpha-shadow-ab.ts --runs <dir>,...`. The timing is the adapter's own wall time for `spp`
 * samples (submit -> finish -> readback), the same number `competitor` reports.
 */
import { drainDevice, getArgs, numberArg, reportError, reportResult, uploadImage, watchDevice } from '../runner/page-client.ts';
import type { AdapterSummary } from '../runner/contract.ts';
import { createAdapter } from '../adapters/registry.ts';
import { getScene } from '../scenes/index.ts';
import type { RendererAdapter } from '../src/adapter.ts';

let adapterSummary: AdapterSummary | null = null;
let deviceErrors: string[] = [];

try {
  const args = getArgs();
  const sceneName = args['scene'];
  const ids = (args['renderers'] ?? '').split(',').filter(Boolean);
  if (!sceneName || ids.length === 0) throw new Error('args scene=<name> and renderers=<id,id,...> are required');
  const spp = numberArg(args, 'spp', 1024);
  const repeat = numberArg(args, 'repeat', 5);
  const scene = getScene(sceneName);
  const adapters = new Map<string, RendererAdapter>();
  for (const id of ids) {
    // `<id>@<beta>` sets `alphaTransmit` (the calibrated `transmit` rows); the label keeps the suffix
    const [base, beta] = id.split('@') as [string, string | undefined];
    const adapter = await createAdapter(base, { emitters: 'rect', mutracer: {}, pt: beta !== undefined ? { alphaTransmit: Number(beta) } : {} });
    await adapter.load(scene, { baseUrl: '/bench/assets' });
    const gpu = (adapter as unknown as { gpuContext?: { summary: AdapterSummary; device: GPUDevice } | null }).gpuContext;
    if (gpu) {
      adapterSummary ??= gpu.summary;
      deviceErrors.push(...watchDevice(gpu.device));
    }
    adapters.set(id, adapter);
  }
  const rounds: Record<string, number[]> = Object.fromEntries(ids.map((id) => [id, [] as number[]]));
  const means: Record<string, number[]> = {};
  const images = new Map<string, Awaited<ReturnType<RendererAdapter['render']>>['image']>();
  // one untimed warm-up round (pipeline / clock ramp), then `repeat` timed rounds; the order is reversed every other round (ABBA)
  for (let round = -1; round < repeat; round++) {
    const order = round % 2 === 0 ? ids : [...ids].reverse();
    for (const id of order) {
      const adapter = adapters.get(id) as RendererAdapter;
      const result = await adapter.render({ spp });
      if (round >= 0) (rounds[id] as number[]).push(result.timings.wallMs / result.spp);
      images.set(id, result.image);
      const sum = [0, 0, 0];
      const d = result.image.data;
      for (let i = 0; i < d.length; i += result.image.channels) for (let c = 0; c < 3; c++) sum[c]! += d[i + c]!;
      means[id] = sum.map((s) => s / (d.length / result.image.channels));
    }
  }
  for (const id of ids) await uploadImage(`${id.replace("@", "_b")}__${sceneName}`, images.get(id)!);
  for (const adapter of adapters.values()) {
    const gpu = (adapter as unknown as { gpuContext?: { device: GPUDevice } | null }).gpuContext;
    if (gpu) await drainDevice(gpu.device);
  }
  reportResult({ scene: sceneName, renderer: 'alpha-shadow-ab', config: { spp, repeat, renderers: ids, scene: sceneName }, metrics: { spp, repeat }, notes: { rounds, means } }, { adapter: adapterSummary, deviceErrors });
  for (const adapter of adapters.values()) adapter.dispose();
} catch (error) {
  reportError(error, { adapter: adapterSummary, deviceErrors });
}
