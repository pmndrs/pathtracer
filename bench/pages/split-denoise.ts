/**
 * V7.3 split denoising, GPU half: OIDN (the Denoiser sibling, colour only, HDR) on a media render and on its multiple-scattering plane.
 *
 *   node bench/runner/run.ts --page split-denoise --args files=<key>@<prefix>,... [guides=0|1]
 *
 * `<prefix>` is a repo-relative path stem of a `pt-megakernel` run with `pt_volume_outputs=1`: `<prefix>.pfm` is the beauty, `<prefix>__vol_order01.pfm` the orders 0-1 plane.
 * Uploads, per key: `<key>__oidn_all` (OIDN on the beauty) and `<key>__split_oidn` (OIDN on max(beauty - order01, 0), then + order01), and the same two with the orders 0-1 plane
 * as a pseudo-albedo guide (`__oidn_all_g`, `__split_oidn_g`; guide = o / (1 + o) clamped to [0, 1]: the clean part of the image gives OIDN the edges that noisy colour does not).
 * The metrics (relMSE, relBias, edge mask, Gaussian baselines) are computed by `bench/split-denoise.ts` in Node from these images. Quality only.
 */
import { detectCaps, requestDeviceForCaps } from '@pmndrs/raytracer';
import { denoiseColorWithOidn } from '../adapters/shared/oidn.ts';
import { readPFM } from '../metrics/pfm.ts';
import { getArgs, reportError, reportResult, summarizeAdapter, uploadImage, watchDevice } from '../runner/page-client.ts';

try {
  const args = getArgs();
  const files = (args.files ?? '').split(',').filter(Boolean).map((e) => {
    const [key, prefix] = e.split('@');
    if (!key || !prefix) throw new Error(`bad files entry "${e}" (want key@prefix)`);
    return { key, prefix };
  });
  if (files.length === 0) throw new Error('files= is empty');
  if (!navigator.gpu) throw new Error('navigator.gpu is undefined: no WebGPU in this browser.');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('requestAdapter() returned null: no usable GPU.');
  const { device } = await requestDeviceForCaps(adapter, detectCaps(adapter), { label: 'split-denoise' });
  const deviceErrors = watchDevice(device);
  const load = async (path: string) => readPFM(new Uint8Array(await (await fetch(`/${path}`)).arrayBuffer()));
  const metrics: Record<string, number> = {};
  const finite = (x: number): number => (Number.isFinite(x) ? x : 0);
  for (const { key, prefix } of files) {
    const beauty = await load(`${prefix}.pfm`);
    const o01 = await load(`${prefix}__vol_order01.pfm`);
    const n = beauty.width * beauty.height;
    // clean the inputs: non-finite pixels become 0 (counted), the difference plane is clamped at 0 (counted)
    let bad = 0;
    let negative = 0;
    const beautyClean = { ...beauty, channels: 3 as const, data: new Float32Array(n * 3) };
    const ms = { ...beauty, channels: 3 as const, data: new Float32Array(n * 3) };
    for (let i = 0; i < n * 3; i++) {
      const b = beauty.data[i] as number;
      const o = o01.data[i] as number;
      if (!Number.isFinite(b) || !Number.isFinite(o)) bad++;
      beautyClean.data[i] = finite(b);
      const d = finite(b) - finite(o);
      if (d < 0) negative++;
      ms.data[i] = Math.max(d, 0);
    }
    const guide = new Float32Array(n * 4);
    for (let p = 0; p < n; p++) {
      for (let c = 0; c < 3; c++) {
        const o = finite(o01.data[p * 3 + c] as number);
        guide[p * 4 + c] = Math.min(1, Math.max(0, o / (1 + o)));
      }
      guide[p * 4 + 3] = 1;
    }
    const all = await denoiseColorWithOidn(device, beautyClean, null);
    const msDen = await denoiseColorWithOidn(device, ms, null);
    const allG = await denoiseColorWithOidn(device, beautyClean, { albedo: guide });
    const msDenG = await denoiseColorWithOidn(device, ms, { albedo: guide });
    const outAll = { width: beauty.width, height: beauty.height, channels: 3 as const, data: new Float32Array(n * 3) };
    const outSplit = { width: beauty.width, height: beauty.height, channels: 3 as const, data: new Float32Array(n * 3) };
    for (let p = 0; p < n; p++) {
      for (let c = 0; c < 3; c++) {
        outAll.data[p * 3 + c] = all.data[p * 4 + c] as number;
        outSplit.data[p * 3 + c] = Math.max(msDen.data[p * 4 + c] as number, 0) + finite(o01.data[p * 3 + c] as number);
      }
    }
    const outAllG = { width: beauty.width, height: beauty.height, channels: 3 as const, data: new Float32Array(n * 3) };
    const outSplitG = { width: beauty.width, height: beauty.height, channels: 3 as const, data: new Float32Array(n * 3) };
    for (let p = 0; p < n; p++) {
      for (let c = 0; c < 3; c++) {
        outAllG.data[p * 3 + c] = allG.data[p * 4 + c] as number;
        outSplitG.data[p * 3 + c] = Math.max(msDenG.data[p * 4 + c] as number, 0) + finite(o01.data[p * 3 + c] as number);
      }
    }
    await uploadImage(`${key}__oidn_all`, outAll);
    await uploadImage(`${key}__split_oidn`, outSplit);
    await uploadImage(`${key}__oidn_all_g`, outAllG);
    await uploadImage(`${key}__split_oidn_g`, outSplitG);
    metrics[`${key}_nonfinite_in`] = bad;
    metrics[`${key}_ms_negative_values`] = negative;
  }
  reportResult({ scene: 'split-denoise', renderer: 'oidn', config: { files: files.length }, metrics, notes: {} }, { adapter: summarizeAdapter(adapter), deviceErrors });
} catch (error) {
  reportError(error);
}
