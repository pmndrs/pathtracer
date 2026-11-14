/**
 * Competitor page: runs a web renderer from bench/adapters on a Tier 0 scene and uploads the linear HDR result.
 *
 *   node bench/runner/run.ts --page competitor --args renderer=mutracer scene=cornell spp=256 [snapshots=16,64]
 *
 * Args: renderer=<tgpt-wavefront|tgpt-megakernel|mutracer>  scene=<any bench scene name>  spp=N  snapshots=a,b,c
 *       budget=N (tgpt frame budget, path steps per call)  bounces=N (maxBounces override, for calibration)  emitters=rect|mesh (tgpt: RectAreaLight, default, or emissive mesh)  mt_candidates/mt_sampler/mt_bevel/mt_dispersion/mt_offset (mutracer experiments, P0.26)  force=1 (render even if the scene is not exact for this renderer)
 *
 *       pt-megakernel (ours): pt_sampler=owen-sobol|pcg  pt_nee=0|1  pt_env=uniform|importance|none  pt_sun_visible=0|1  pt_positions=barycentric|ray  pt_vndf=standard|cap
 *       pt_transport=radiance|importance  pt_sun_radius=R (radians, 0 = delta sun)  pt_light_selection=power|kind|cap:0.25  pt_maxdepth=N  pt_rr=N (0 = off)  pt_seed=N  pt_passes=N (sample passes per submit)
 *
 * Uploads `<renderer>__<scene>` (final image) and `<renderer>__<scene>__spp<N>` per snapshot. Unsupported scenes report
 * `skipped=1` and upload nothing.
 */
import { drainDevice, getArgs, numberArg, reportError, reportResult, uploadImage, watchDevice } from '../runner/page-client.ts';
import type { AdapterSummary } from '../runner/contract.ts';
import { createAdapter, unsupportedReason } from '../adapters/registry.ts';
import { parseLightSelection } from '@pmndrs/raytracer';
import { getScene } from '../scenes/index.ts';

let adapterSummary: AdapterSummary | null = null;
let deviceErrors: string[] = [];

function channelMeans(data: Float32Array, channels: number): { mean: number[]; nonFinite: number } {
  const sum = [0, 0, 0];
  let nonFinite = 0;
  let count = 0;
  for (let i = 0; i < data.length; i += channels) {
    let ok = true;
    for (let c = 0; c < 3; c++) if (!Number.isFinite(data[i + c]!)) ok = false;
    if (!ok) {
      nonFinite++;
      continue;
    }
    for (let c = 0; c < 3; c++) sum[c]! += data[i + c]!;
    count++;
  }
  return { mean: sum.map((s) => s / Math.max(1, count)), nonFinite };
}

try {
  const args = getArgs();
  const rendererId = args['renderer'];
  const sceneName = args['scene'];
  if (!rendererId || !sceneName) throw new Error('args renderer=<id> and scene=<name> are required');
  const spp = numberArg(args, 'spp', 64);
  const snapshots = (args['snapshots'] ?? '').split(',').filter(Boolean).map(Number);
  if (snapshots.some((s) => !Number.isInteger(s) || s <= 0)) throw new Error(`snapshots="${args['snapshots']}" must be positive integers`);
  snapshots.sort((a, b) => a - b);
  const frameBudget = args['budget'] ? numberArg(args, 'budget', 250000) : undefined;

  const scene = getScene(sceneName);
  const emitters = args['emitters'] === 'mesh' ? 'mesh' : 'rect';
  const bounces = args['bounces'] ? numberArg(args, 'bounces', 0) : undefined;
  const mutracer = {
    ...(args['mt_candidates'] ? { candidates: numberArg(args, 'mt_candidates', 4) } : {}),
    ...(args['mt_sampler'] ? { sampler: args['mt_sampler'] as 'sobol' | 'zsobol' | 'random' } : {}),
    ...(args['mt_bevel'] ? { bevel: args['mt_bevel'] === '1' } : {}),
    ...(args['mt_dispersion'] ? { dispersion: (numberArg(args, 'mt_dispersion', 3) === 1 ? 1 : 3) as 1 | 3 } : {}),
    ...(args['mt_premul'] ? { premultiply: args['mt_premul'] === '1' } : {}),
    ...(args['mt_envwidth'] ? { envWidth: numberArg(args, 'mt_envwidth', 4) } : {}),
    ...(args['mt_offset'] ? { offset: numberArg(args, 'mt_offset', 0) } : {}),
  };
  const pt = {
    ...(args['pt_sampler'] ? { sampler: args['pt_sampler'] as 'owen-sobol' | 'pcg' } : {}),
    ...(args['pt_nee'] ? { nee: args['pt_nee'] !== '0' } : {}),
    ...(args['pt_env'] ? { envSampling: args['pt_env'] as 'uniform' | 'importance' | 'none' } : {}),
    ...(args['pt_sun_visible'] ? { sunVisible: args['pt_sun_visible'] === '1' } : {}),
    ...(args['pt_sun_radius'] !== undefined ? { sunAngularRadius: numberArg(args, 'pt_sun_radius', 0) } : {}),
    ...(args['pt_positions'] ? { positions: args['pt_positions'] as 'barycentric' | 'ray' } : {}),
    ...(args['pt_vndf'] ? { vndf: args['pt_vndf'] as 'standard' | 'cap' } : {}),
    ...(args['pt_transport'] ? { transport: args['pt_transport'] as 'radiance' | 'importance' } : {}),
    ...(args['pt_maxdepth'] ? { maxDepth: numberArg(args, 'pt_maxdepth', 1) } : {}),
    ...(args['pt_rr'] ? { rrStartDepth: numberArg(args, 'pt_rr', 0) } : {}),
    ...(args['pt_light_selection'] ? { lightSelection: parseLightSelection(args['pt_light_selection']) } : {}),
    ...(args['pt_seed'] ? { seed: numberArg(args, 'pt_seed', 0) } : {}),
    ...(args['pt_prune'] ? { prune: args['pt_prune'] !== '0' } : {}),
    ...(args['pt_shorten'] ? { shortenNames: args['pt_shorten'] !== '0' } : {}),
    ...(args['pt_passes'] ? { passesPerSubmit: numberArg(args, 'pt_passes', 8) } : {}),
  };
  const adapter = await createAdapter(rendererId, { frameBudget, bounces, emitters, mutracer, pt });
  const reason = unsupportedReason(adapter.features, scene);
  const config = { ...adapter.info.config, spp, snapshots, scene: sceneName };

  if (reason && args['force'] !== '1') {
    reportResult({ scene: sceneName, renderer: adapter.info.id, config, metrics: { skipped: 1 }, notes: { skipped: reason } });
  } else {
    const loaded = await adapter.load(scene, { baseUrl: '/bench/assets' });
    const gpu = (adapter as unknown as { gpuContext?: { summary: AdapterSummary; device: GPUDevice } | null }).gpuContext;
    if (gpu) {
      adapterSummary = gpu.summary;
      deviceErrors = watchDevice(gpu.device);
    }
    const result = await adapter.render({ spp, snapshotsAtSpp: snapshots });
    if (gpu) await drainDevice(gpu.device);

    const name = `${adapter.info.id}__${sceneName}`;
    await uploadImage(name, result.image);
    for (const snap of result.snapshots ?? []) await uploadImage(`${name}__spp${snap.spp}`, snap.image);

    const { mean, nonFinite } = channelMeans(result.image.data, result.image.channels);
    const approximations = (adapter as unknown as { approximations?: string[] }).approximations ?? [];
    reportResult(
      {
        scene: sceneName,
        renderer: adapter.info.id,
        config,
        valid: nonFinite === 0 && deviceErrors.length === 0,
        metrics: {
          spp: result.spp,
          wall_ms: result.timings.wallMs,
          ms_per_spp: result.timings.wallMs / result.spp,
          load_ms: loaded.loadMs,
          compile_ms: loaded.compileMs ?? 0,
          mean_r: mean[0]!,
          mean_g: mean[1]!,
          mean_b: mean[2]!,
          non_finite_pixels: nonFinite,
          ...Object.fromEntries((result.snapshots ?? []).map((s) => [`wall_ms_at_${s.spp}`, s.wallMs])),
        },
        notes: {
          image: name,
          version: adapter.info.version,
          commit: adapter.info.commit,
          forced: reason ? reason : undefined,
          stats: result.stats,
          exporterApproximations: approximations.length,
        },
      },
      { adapter: adapterSummary, deviceErrors },
    );
    adapter.dispose();
  }
} catch (error) {
  reportError(error, { adapter: adapterSummary, deviceErrors });
}
