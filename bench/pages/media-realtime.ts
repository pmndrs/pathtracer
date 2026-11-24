/**
 * V7.4: SVGF on media, whole vs split (docs/internal/design/volumes-plan.md V7.4; results in bench/results/SPLIT-DENOISE.md). Quality only: no timings.
 *
 *   node bench/runner/run.ts --page media-realtime --timeout 1200000 --load-wait 0 --args scenes=fog,cloud scenarios=identity,static,moving
 *
 * Arms (`bench/realtime/media-rig.ts`): raw (1 spp frame), raw-acc (mean of the raw frames, static only), svgf-all (chain as is), svgf-split (camera-ray single scattering = emission plane
 * with a temporal history and no spatial filter; SVGF on the rest). References: OUR plain tracer, same estimator and path length, 256x256 (`refSpp`).
 *
 * Args: scenes=fog,cloud (fog = vol-cornell-fog, forced + equiangular; fogmarch = + march 4; cloud = vol-vdb-wdas-cloud, default estimator)  scenarios=identity,ref,static,moving
 *       res=256 maxDepth=12 seeds=1,2  frames=64 checkpoints=1,2,4,8,16,32,64 stableFrom=32  moveDeg=4 moveFrames=32 moveCheckpoints=1,2,4,8,16,32  refSpp=2048 moveRefSpp=1024
 *       refFile=<scene>@<url>,...  moveRefFiles=<scene>@<pose>@<url>,...  alpha=0.1 clip=0  flip=1 upload=1 uploadFrames=4,16
 */
import type { PtLookAtCamera, PtMegakernel } from '@pmndrs/pathtracer';
import { PtMegakernelAdapter } from '../adapters/pt-megakernel/index.ts';
import { compareImages } from '../metrics/metrics.ts';
import { flipLdr } from '../metrics/flip.ts';
import { readPFM } from '../metrics/pfm.ts';
import type { Image } from '../src/adapter.ts';
import type { AdapterSummary } from '../runner/contract.ts';
import { getArgs, numberArg, reportError, reportResult, uploadImage, watchDevice } from '../runner/page-client.ts';
import { getScene } from '../scenes/index.ts';
import type { SceneDesc } from '../scenes/schema.ts';
import { MediaRig, type MediaArm, type MediaImg } from '../realtime/media-rig.ts';
import { edgeMask, maskedRelMse, temporalExcess, type MImg } from '../realtime/media-metrics.ts';

let adapterSummary: AdapterSummary | null = null;
let deviceErrors: string[] = [];
const metrics: Record<string, number> = {};
const notes: Record<string, unknown> = {};

/** Scene keys of this page: scene name + the estimator settings of the adapter. */
export const MEDIA_SCENES: Record<string, { scene: string; media: Record<string, unknown> }> = {
  fog: { scene: 'vol-cornell-fog', media: { volumeForcedNee: true, volumeEquiangular: true } },
  fogmarch: { scene: 'vol-cornell-fog', media: { volumeForcedNee: true, volumeEquiangular: true, volumeMarch: 4 } },
  cloud: { scene: 'vol-vdb-wdas-cloud', media: {} },
};

type Img = MediaImg | { width: number; height: number; channels: number; data: Float32Array };

async function fetchPfm(url: string): Promise<Img> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  const buf = new Uint8Array(await r.arrayBuffer());
  if (buf[0] !== 0x50) throw new Error(`${url}: not a PFM`);
  return readPFM(buf) as unknown as Img;
}

/** Our own reference: the accumulating plain kernel at `spp`, size w x h, with the page's estimator. */
async function ownReference(scene: SceneDesc, media: Record<string, unknown>, w: number, h: number, spp: number, camera: PtLookAtCamera, maxDepth: number, seed: number): Promise<Img> {
  const adapter = new PtMegakernelAdapter({ nee: true, ...media });
  await adapter.load(scene);
  const kernel = adapter.renderer as PtMegakernel;
  const device = adapter.gpuContext!.device;
  kernel.setFilm(w, h);
  kernel.setCamera(camera);
  kernel.setLimits({ maxDepth, rrStartDepth: 0 });
  kernel.setSeed(seed);
  {
    const e = device.createCommandEncoder();
    kernel.reset(e);
    device.queue.submit([e.finish()]);
  }
  for (let n = 0; n < spp; n += 8) {
    const e = device.createCommandEncoder();
    kernel.encode(e, Math.min(8, spp - n));
    device.queue.submit([e.finish()]);
    await device.queue.onSubmittedWorkDone();
  }
  const img = await kernel.readImage();
  adapter.dispose();
  return { width: img.width, height: img.height, channels: 4, data: img.data };
}

/** A slow pan: yaw about the camera's own position (the view turns, the parallax is zero), `deg` degrees over `frames` frames, linear. */
function panCamera(base: PtLookAtCamera, f: number, frames: number, deg: number): PtLookAtCamera {
  const yaw = ((deg * Math.PI) / 180) * ((f - 1) / Math.max(1, frames - 1));
  const d = [base.target[0] - base.position[0], base.target[1] - base.position[1], base.target[2] - base.position[2]] as const;
  const c = Math.cos(yaw);
  const s = Math.sin(yaw);
  return { ...base, target: [base.position[0] + d[0] * c + d[2] * s, base.position[1] + d[1], base.position[2] - d[0] * s + d[2] * c] };
}

const lum = (d: Float32Array, i: number, c: number) => 0.2126 * (d[i * c] as number) + 0.7152 * (d[i * c + 1] as number) + 0.0722 * (d[i * c + 2] as number);
const asImage = (i: Img) => i as unknown as Image;

/** Metrics of one arm image against a reference (edge mask precomputed). */
function score(prefix: string, img: Img, ref: Img, mask: Uint8Array, doFlip: boolean): void {
  const m = compareImages(asImage(img), asImage(ref));
  metrics[`${prefix}_relmse`] = m.relMse;
  metrics[`${prefix}_relbias`] = m.relBias;
  metrics[`${prefix}_edge`] = maskedRelMse(img as MImg, ref as MImg, mask, true);
  metrics[`${prefix}_flat`] = maskedRelMse(img as MImg, ref as MImg, mask, false);
  if (doFlip) metrics[`${prefix}_flip`] = flipLdr(asImage(img), asImage(ref)).mean;
}

try {
  const args = getArgs();
  const keys = (args.scenes ?? 'fog').split(',');
  const scenarios = (args.scenarios ?? 'identity,static').split(',');
  const res = numberArg(args, 'res', 256);
  const maxDepth = numberArg(args, 'maxDepth', 12);
  const seeds = (args.seeds ?? '1,2').split(',').map(Number);
  const frames = numberArg(args, 'frames', 64);
  const checkpoints = (args.checkpoints ?? '1,2,4,8,16,32,64').split(',').map(Number).filter((f) => f <= frames);
  const stableFrom = numberArg(args, 'stableFrom', Math.min(32, frames - 1));
  const moveDeg = numberArg(args, 'moveDeg', 4);
  const moveFrames = numberArg(args, 'moveFrames', 32);
  const moveCheckpoints = (args.moveCheckpoints ?? '1,2,4,8,16,32').split(',').map(Number).filter((f) => f <= moveFrames);
  const refSpp = numberArg(args, 'refSpp', 2048);
  const moveRefSpp = numberArg(args, 'moveRefSpp', 1024);
  const alpha = numberArg(args, 'alpha', 0.1);
  const clip = numberArg(args, 'clip', 0);
  const dn = args.dn ? Object.fromEntries(args.dn.split(';').map((kv) => { const [k, v] = kv.split(':') as [string, string]; return [k, Number.isNaN(Number(v)) ? v : Number(v)]; })) : undefined;
  const tag = args.tag ?? '';
  const doFlip = numberArg(args, 'flip', 1) === 1;
  const doUpload = numberArg(args, 'upload', 1) === 1;
  const uploadFrames = (args.uploadFrames ?? '4,16').split(',').map(Number);
  const refFiles = new Map((args.refFile ?? '').split(',').filter(Boolean).map((e) => [e.split('@')[0] as string, e.split('@')[1] as string]));
  const moveRefFiles = new Map((args.moveRefFiles ?? '').split(',').filter(Boolean).map((e) => { const [k, p, u] = e.split('@'); return [`${k}@${p}`, u as string] as const; }));
  const allDevices: GPUDevice[] = [];
  const track = (rig: MediaRig) => (rig.a ? [rig.a.device] : []).concat(rig.b ? [rig.b.device] : []).forEach((d) => { allDevices.push(d); deviceErrors.push(...watchDevice(d)); });

  for (const key of keys) {
    const def = MEDIA_SCENES[key];
    if (!def) throw new Error(`unknown scene key ${key} (${Object.keys(MEDIA_SCENES).join(', ')})`);
    const scene = getScene(def.scene);
    const baseCam = scene.camera as PtLookAtCamera;
    const W = res;
    const H = res;
    notes[`${key}_config`] = `${def.scene} ${W}x${H}, maxDepth ${maxDepth}, no RR, estimator ${JSON.stringify(def.media)}, emission history alpha ${alpha} clip ${clip}`;

    // ---- identity: plane identity and the guides of a medium first vertex
    if (scenarios.includes('identity')) {
      const rig = await MediaRig.create({ scene, width: W, height: H, maxDepth, media: def.media, seed: 1, alpha, clip });
      track(rig);
      adapterSummary = rig.a!.adapter.gpuContext!.summary;
      const frame2: { viewZ: Float32Array; radiance: Float32Array }[] = [];
      const frame2s: { viewZ: Float32Array; radiance: Float32Array }[] = [];
      let aRad: Float32Array | null = null;
      for (let f = 1; f <= 2; f++) {
        await rig.frame(baseCam);
        for (const [name, s] of [['all', rig.a!], ['split', rig.b!]] as const) {
          const fr = await s.kernel.readFrame();
          let worst = 0;
          let sumRad = 0;
          let sumEmi = 0;
          let fogPixels = 0;
          let viewZMiss = 0;
          const n = fr.width * fr.height;
          for (let i = 0; i < n; i++) {
            let err = 0;
            for (let c = 0; c < 3; c++) {
              const rad = fr.radiance[i * 4 + c] as number;
              const sum = (fr.diffuse[i * 4 + c] as number) * (fr.albedo[i * 4 + c] as number) + (fr.specular[i * 4 + c] as number) * (fr.specularFactor[i * 4 + c] as number) + (fr.emission[i * 4 + c] as number);
              err = Math.max(err, Math.abs(sum - rad) / (Math.abs(rad) + 1e-3));
            }
            worst = Math.max(worst, err);
            sumRad += lum(fr.radiance, i, 4);
            sumEmi += lum(fr.emission, i, 4);
            // a pixel whose demodulation factor is the medium's (albedo 0.9 / 0.99 plus its floor) and whose specular flag is off is a first-vertex medium scatter
            if ((fr.normalViewZ[i * 4 + 3] as number) === 0) viewZMiss++;
            else if ((fr.albedo[i * 4 + 3] as number) === 0 && Math.abs((fr.albedo[i * 4] as number) - 0.9) < 1e-3) fogPixels++;
          }
          metrics[`${key}_identity_${name}_f${f}_max_rel_err`] = worst;
          metrics[`${key}_identity_${name}_f${f}_emission_share`] = sumEmi / sumRad;
          metrics[`${key}_identity_${name}_f${f}_medium_first_vertex_frac`] = fogPixels / n;
          metrics[`${key}_identity_${name}_f${f}_miss_frac`] = viewZMiss / n;
          if (name === 'all') {
            frame2.push({ viewZ: fr.normalViewZ, radiance: fr.radiance });
            if (f === 1) aRad = fr.radiance;
          } else {
            frame2s.push({ viewZ: fr.normalViewZ, radiance: fr.radiance });
          }
          if (name === 'split' && f === 1 && aRad) {
            let worstAB = 0;
            let sumAB = 0;
            for (let i = 0; i < n * 4; i++) {
              if (i % 4 === 3) continue;
              const d = Math.abs((fr.radiance[i] as number) - (aRad[i] as number)) / (Math.abs(aRad[i] as number) + 1e-3);
              worstAB = Math.max(worstAB, d);
              sumAB += d;
            }
            metrics[`${key}_identity_split_vs_all_f1_max_rel_diff`] = worstAB;
            metrics[`${key}_identity_split_vs_all_f1_mean_rel_diff`] = sumAB / (n * 3);
          }
        }
      }
      // guides of a medium first vertex: how much does viewZ of the same pixel move between two frames of a still camera? (all = scatter-point guides, split = guide ray, V7.5)
      for (const [nm, fs] of [['', frame2], ['split_', frame2s]] as const) {
        const [g1, g2] = fs as [typeof frame2[0], typeof frame2[0]];
        if (!g1 || !g2) continue;
        let moved = 0;
        let seen = 0;
        for (let i = 0; i < W * H; i++) {
          const z1 = g1.viewZ[i * 4 + 3] as number;
          const z2 = g2.viewZ[i * 4 + 3] as number;
          if (z1 === 0 && z2 === 0) continue;
          seen++;
          if (Math.abs(z1 - z2) > 0.02 * Math.max(Math.abs(z1), Math.abs(z2))) moved++;
        }
        metrics[`${key}_identity_${nm}guide_viewz_unstable_frac`] = moved / Math.max(1, seen);
      }
      rig.destroy();
    }

    // ---- reference of the static camera
    let staticRef: Img | null = null;
    if (scenarios.some((s) => ['static', 'ref'].includes(s))) {
      if (refFiles.has(key)) {
        staticRef = await fetchPfm(refFiles.get(key) as string);
        notes[`${key}_reference`] = `refFile ${refFiles.get(key)}`;
      } else {
        const t0 = performance.now();
        staticRef = await ownReference(scene, def.media, W, H, refSpp, baseCam, maxDepth, 4242);
        notes[`${key}_reference`] = `ours: plain megakernel, NEE, maxDepth ${maxDepth}, no RR, ${refSpp} spp, ${W}x${H}, estimator ${JSON.stringify(def.media)}`;
        notes[`${key}_reference_render_s`] = (performance.now() - t0) / 1000;
        if (doUpload) await uploadImage(`reference-${key}-${refSpp}spp`, staticRef as never);
      }
      let mean = 0;
      for (let i = 0; i < W * H; i++) mean += lum(staticRef.data, i, staticRef.channels);
      metrics[`${key}_reference_mean_luma`] = mean / (W * H);
    }

    // ---- static camera
    if (scenarios.includes('static')) {
      const ref = staticRef as Img;
      const { mask, fraction } = edgeMask(ref as MImg);
      metrics[`${key}_edge_fraction`] = fraction;
      const arms: MediaArm[] = ['raw', 'raw-acc', 'svgf-all', 'svgf-split'];
      for (const seed of seeds) {
        const rig = await MediaRig.create({ scene, width: W, height: H, maxDepth, media: def.media, seed, alpha, clip, trackRaw: true, ...(dn ? { denoiser: dn } : {}) });
        track(rig);
        adapterSummary = rig.a!.adapter.gpuContext!.summary;
        { const o = rig.a!.arm.denoiser.options.diffuse as unknown as Record<string, number>; metrics.dn_atrous = o.atrousIterations as number; metrics.dn_keys = Object.keys(dn ?? {}).length; metrics.dn_s = JSON.stringify((rig.a as any).options.denoiser ?? null).length; metrics.dn_alpha = o.temporalAlpha as number; metrics.dn_sigmaL = o.sigmaLuminance as number; }
        rig.cut();
        for (let f = 1; f <= frames; f++) {
          await rig.frame(baseCam);
          if (f === stableFrom) { rig.a!.arm.startFlicker(); rig.b!.arm.startFlicker(); }
          if (f >= stableFrom) { rig.a!.arm.stepFlicker(); rig.b!.arm.stepFlicker(); }
          if (checkpoints.includes(f)) {
            for (const arm of arms) {
              const img = await rig.image(arm);
              score(`${key}_static_s${seed}_f${f}_${arm}`, img, ref, mask, doFlip);
              if (doUpload && seed === seeds[0] && uploadFrames.includes(f)) await uploadImage(`${key}-static-s${seed}-f${f}-${arm}${tag}`, img as never);
            }
          } else {
            for (const d of await rig.device()) await d.queue.onSubmittedWorkDone();
          }
        }
        if (frames > stableFrom) {
          for (const [arm, s, stage] of [['raw', rig.a!, 'raw'], ['svgf-all', rig.a!, 'denoised'], ['svgf-split', rig.b!, 'denoised']] as const) {
            const fl = await s.arm.readFlicker();
            const { map } = fl[stage];
            let sum = 0;
            for (let i = 0; i < map.length; i++) sum += (map[i] as number) / (lum(ref.data, i, ref.channels) + 0.01);
            metrics[`${key}_static_s${seed}_${arm}_flicker`] = sum / map.length;
          }
        }
        rig.destroy();
      }
    }

    // ---- slow pan
    if (scenarios.includes('moving')) {
      const poses = [...new Set([...moveCheckpoints, ...moveCheckpoints.filter((c) => c > 1 && c >= moveFrames / 4).map((c) => c - 1)])].sort((a, b) => a - b);
      const refs = new Map<number, Img>();
      for (const p of poses) {
        const url = moveRefFiles.get(`${key}@${p}`);
        if (url) refs.set(p, await fetchPfm(url));
        else {
          const t0 = performance.now();
          const r = await ownReference(scene, def.media, W, H, moveRefSpp, panCamera(baseCam, p, moveFrames, moveDeg), maxDepth, 777 + p);
          refs.set(p, r);
          notes[`${key}_moveref_${p}_s`] = (performance.now() - t0) / 1000;
          if (doUpload) await uploadImage(`moveref-${key}-p${p}-${moveRefSpp}spp`, r as never);
        }
      }
      const arms: MediaArm[] = ['raw', 'svgf-all', 'svgf-split'];
      for (const seed of seeds) {
        const rig = await MediaRig.create({ scene, width: W, height: H, maxDepth, media: def.media, seed, alpha, clip });
        track(rig);
        rig.cut();
        const prev = new Map<MediaArm, Img>();
        for (let f = 1; f <= moveFrames; f++) {
          await rig.frame(panCamera(baseCam, f, moveFrames, moveDeg));
          const wantScore = moveCheckpoints.includes(f);
          for (const arm of arms) {
            if (!poses.includes(f)) continue;
            const img = await rig.image(arm);
            const ref = refs.get(f);
            if (ref && wantScore) score(`${key}_moving_s${seed}_f${f}_${arm}`, img, ref, edgeMask(ref as MImg).mask, doFlip);
            const p0 = prev.get(arm);
            const r0 = refs.get(f - 1);
            if (ref && p0 && r0 && f > 1) metrics[`${key}_moving_s${seed}_f${f}_${arm}_excess`] = temporalExcess(img as MImg, p0 as MImg, ref as MImg, r0 as MImg);
            prev.set(arm, img);
            if (doUpload && seed === seeds[0] && uploadFrames.includes(f)) await uploadImage(`${key}-moving-s${seed}-f${f}-${arm}`, img as never);
          }
          for (const d of await rig.device()) await d.queue.onSubmittedWorkDone();
        }
        rig.destroy();
      }
    }
  }
  reportResult(
    { scene: keys.join('+'), renderer: 'media-realtime', config: { scenes: keys, scenarios, res, maxDepth, seeds, frames, checkpoints, stableFrom, moveDeg, moveFrames, moveCheckpoints, refSpp, moveRefSpp, alpha, clip }, valid: deviceErrors.length === 0, metrics, notes },
    { adapter: adapterSummary, deviceErrors },
  );
} catch (error) {
  reportError(error, { adapter: adapterSummary, deviceErrors });
}
