/**
 * Competitor page: runs a web renderer from bench/adapters on a Tier 0 scene and uploads the linear HDR result.
 *
 *   node bench/runner/run.ts --page competitor --args renderer=mutracer scene=cornell spp=256 [snapshots=16,64]
 *
 * Args: renderer=<tgpt-wavefront|tgpt-megakernel|mutracer>  scene=<any bench scene name>  spp=N  snapshots=a,b,c
 *       budget=N (tgpt frame budget, path steps per call)  bounces=N (maxBounces override, for calibration)  emitters=rect|mesh (tgpt: RectAreaLight, default, or emissive mesh)  mt_candidates/mt_sampler/mt_bevel/mt_dispersion/mt_offset (mutracer experiments, P0.26)  force=1 (render even if the scene is not exact for this renderer)
 *
 *       pt-megakernel (ours): pt_sampler=owen-sobol|pcg|bluenoise  pt_nee=0|1  pt_env=uniform|importance|none  pt_sun_visible=0|1  pt_positions=barycentric|ray  pt_vndf=standard|cap
 *       pt_transport=radiance|importance  pt_sun_radius=R (radians, 0 = delta sun)  pt_light_selection=pbrt|power|kind|cap:0.25  pt_texture_lod=0|cone  pt_texture_cone_scale=auto|N  pt_layering=albedo|stochastic  pt_albedo_base=openpbr|internal  pt_albedo_correction=none|lut|texture  pt_exit_mis=pbrt|none  pt_layer_depth=N  pt_layer_thickness=T  pt_tint=0|1  pt_coverage=0|1  pt_maxdepth=N  pt_rr=N (0 = off)  pt_seed=N  pt_passes=N (sample passes per submit)  pt_refill=0|1|inline|batch|ordered pt_refill_threshold=F pt_layers=N (wavefront only: refill, slots per pixel; P3.2, ordered = P3.10)  pt_appends=atomic|subgroup|auto (P3.3)  pt_sort=none|material|octant|both (P3.4)  pt_alpha_policy=exact|opaqueShadow|opaqueIndirect|opaqueSecondary (alpha MASK scenes; BIASED unless exact, R2)  pt_packing=split|packed (scene copy as two storage buffers or one; P3.11)  pt_primary=trace|vbuffer (primary hits: traced camera rays, or a bench-rasterised visibility buffer; P3.6)  pt_aovs=0|1 (albedo/normal/depth extra images)  pt_counters=0|1 (counter heatmaps, row INVALID for timing)  pt_denoise=oidn (extra `denoised` image; raw stays the result)  pt_lod=full|secondary (R1, biased) pt_lod_depth=N  pt_restir=1 (ReSTIR DI at the first vertex, P4, megakernel; one sample pass = one frame: static scene, accumulated mean) pt_restir_m=N pt_restir_temporal=0|1 pt_restir_spatial=K pt_restir_radius=PX pt_restir_bias=biased|unbiased pt_restir_vis=0|1 pt_restir_cap=N pt_restir_nb=random|fixed pt_restir_tpick=random|nearest
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
    ...(args['pt_sampler'] ? { sampler: args['pt_sampler'] as 'owen-sobol' | 'pcg' | 'bluenoise' } : {}),
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
    ...(args['pt_texture_lod'] ? { textureLod: args['pt_texture_lod'] === 'cone' ? ('cone' as const) : (0 as const) } : {}),
    ...(args['pt_texture_cone_scale'] ? { textureConeScale: args['pt_texture_cone_scale'] === 'auto' ? ('auto' as const) : numberArg(args, 'pt_texture_cone_scale', 1) } : {}),
    ...(args['pt_layering'] ? { layering: args['pt_layering'] as 'albedo' | 'stochastic' } : {}),
    ...(args['pt_albedo_base'] ? { albedoBase: args['pt_albedo_base'] as 'openpbr' | 'internal' } : {}),
    ...(args['pt_albedo_correction'] ? { albedoCorrection: args['pt_albedo_correction'] as 'none' | 'lut' | 'texture' } : {}),
    ...(args['pt_exit_mis'] ? { exitMis: args['pt_exit_mis'] as 'pbrt' | 'none' } : {}),
    ...(args['pt_layer_depth'] ? { layerDepth: numberArg(args, 'pt_layer_depth', 64) } : {}),
    ...(args['pt_layer_thickness'] ? { layerThickness: numberArg(args, 'pt_layer_thickness', 1e-6) } : {}),
    ...(args['pt_tint'] ? { transmissionTint: args['pt_tint'] !== '0' } : {}),
    ...(args['pt_lod'] ? { materialLod: args['pt_lod'] === 'secondary' ? ('secondary' as const) : ('full' as const) } : {}),
    ...(args['pt_lod_depth'] ? { materialLodDepth: numberArg(args, 'pt_lod_depth', 1) } : {}),
    ...(args['pt_restir'] === '1'
      ? {
          restir: {
            ...(args['pt_restir_m'] ? { candidates: numberArg(args, 'pt_restir_m', 8) } : {}),
            ...(args['pt_restir_temporal'] ? { temporal: args['pt_restir_temporal'] !== '0' } : {}),
            ...(args['pt_restir_spatial'] ? { spatial: numberArg(args, 'pt_restir_spatial', 1) } : {}),
            ...(args['pt_restir_radius'] ? { spatialRadius: numberArg(args, 'pt_restir_radius', 16) } : {}),
            ...(args['pt_restir_bias'] ? { bias: args['pt_restir_bias'] as 'biased' | 'unbiased' } : {}),
            ...(args['pt_restir_vis'] ? { initialVisibility: args['pt_restir_vis'] !== '0' } : {}),
            ...(args['pt_restir_cap'] ? { historyCap: numberArg(args, 'pt_restir_cap', 20) } : {}),
            ...(args['pt_restir_nb'] ? { neighbours: args['pt_restir_nb'] as 'random' | 'fixed' } : {}),
            ...(args['pt_restir_tpick'] ? { temporalPick: args['pt_restir_tpick'] as 'random' | 'nearest' } : {}),
          },
        }
      : {}),
    ...(args['pt_tangent'] ? { tangent: args['pt_tangent'] as 'attribute' | 'uv' } : {}),
    ...(args['pt_coverage'] ? { coverage: args['pt_coverage'] === '1' } : {}),
    ...(args['pt_seed'] ? { seed: numberArg(args, 'pt_seed', 0) } : {}),
    ...(args['pt_prune'] ? { prune: args['pt_prune'] !== '0' } : {}),
    ...(args['pt_shorten'] ? { shortenNames: args['pt_shorten'] !== '0' } : {}),
    ...(args['pt_alpha_shadow'] ? { alphaShadow: args['pt_alpha_shadow'] as 'policy' | 'roulette' | 'coverage' | 'stochastic' | 'firstK' | 'transmit' | 'classified' } : {}),
    ...(args['pt_alpha_transmit'] ? { alphaTransmit: numberArg(args, 'pt_alpha_transmit', 0.5) } : {}),
    ...(args['pt_alpha_roulette'] ? { alphaRoulette: numberArg(args, 'pt_alpha_roulette', 0.25) } : {}),
    ...(args['pt_alpha_k'] ? { alphaFirstK: numberArg(args, 'pt_alpha_k', 1) } : {}),
    ...(args['pt_alpha_policy'] ? { alphaPolicy: args['pt_alpha_policy'] as 'exact' | 'opaqueShadow' | 'opaqueIndirect' | 'opaqueSecondary' } : {}),
    ...(args['pt_packing'] ? { scenePacking: args['pt_packing'] as 'split' | 'packed' } : {}),
    ...(args['pt_primary'] ? { primary: args['pt_primary'] as 'trace' | 'vbuffer' } : {}),
    ...(args['pt_counters'] ? { counters: args['pt_counters'] === '1' } : {}),
    ...(args['pt_aovs'] ? { aovs: args['pt_aovs'] === '1' } : {}),
    ...(args['pt_denoise'] ? { denoise: args['pt_denoise'] as 'oidn' } : {}),
    ...(args['pt_passes'] ? { passesPerSubmit: numberArg(args, 'pt_passes', 8) } : {}),
    ...(args['pt_refill'] ? { refill: args['pt_refill'] === '0' ? (false as const) : args['pt_refill'] === 'batch' ? ('batch' as const) : args['pt_refill'] === 'ordered' ? ('ordered' as const) : (true as const) } : {}),
    ...(args['pt_auto'] ? { auto: args['pt_auto'] === '1' } : {}),
    ...(args['pt_appends'] ? { appends: args['pt_appends'] as 'atomic' | 'subgroup' | 'auto' } : {}),
    ...(args['pt_sort'] ? { sort: args['pt_sort'] as 'none' | 'material' | 'octant' | 'both' } : {}),
    ...(args['pt_refill_threshold'] ? { refillThreshold: numberArg(args, 'pt_refill_threshold', 0.25) } : {}),
    ...(args['pt_layers'] ? { layers: numberArg(args, 'pt_layers', 1) } : {}),
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
    // Extra images of our own renderer (AOVs, counter heatmaps, the OIDN-denoised frame): next to the raw image, which stays the one the metrics read.
    for (const extra of (adapter as unknown as { extraImages?: { name: string; image: Parameters<typeof uploadImage>[1] }[] }).extraImages ?? []) await uploadImage(`${name}__${extra.name}`, extra.image);

    const { mean, nonFinite } = channelMeans(result.image.data, result.image.channels);
    const approximations = (adapter as unknown as { approximations?: string[] }).approximations ?? [];
    reportResult(
      {
        scene: sceneName,
        renderer: adapter.info.id,
        config,
        // Counters make a render instrumented: its time is not a result (R4), so the row is invalid for timing (notes.stats.instrumented = 1).
        valid: nonFinite === 0 && deviceErrors.length === 0 && !result.stats?.instrumented,
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
