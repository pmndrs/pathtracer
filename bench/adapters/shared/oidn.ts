/**
 * OIDN on a converged frame (P2 exit item: `pt_denoise=oidn`). The library only produces the AOVs (`integrator.aovs`); this bench helper hands the frame to the
 * Denoiser package (`~/Developer/Denoiser`, `packages/denoiser`, linked as a bench dev dependency: OIDN networks as WGSL on the same `GPUDevice`) with color + albedo + normal.
 *
 *  - color: our linear HDR image (rgba32float), `hdr: true` (OIDN's PU transfer and autoexposure);
 *  - albedo: the first non-specular vertex's albedo, averaged over the samples (the AOV, [0, 1]);
 *  - normal: OIDN's aux wants VIEW-space normals in [-1, 1] (Denoiser guide: `normalView`; the pathtracer's AOV is world space, the spec says adapters convert), so
 *    n_view = (n . right, n . up, n . -forward) from the camera basis; pixels without a hit stay (0, 0, 0);
 *  - rows: both top-down (row 0 = top), so no flips;
 *  - the net is `rt_hdr_calb_cnrm` (clean aux, chosen automatically when albedo and normal are both given): our aux are noise-free to the extent the averaged AOVs are.
 *
 * Quality only. Run outside any timed region (the denoiser, the texture uploads and the readback all cost).
 */
import { ptCameraBasis, type PtAovs, type PtLookAtCamera } from '@pmndrs/pathtracer';
import type { Image } from '../../src/adapter.ts';

declare const __BENCH_DENOISER_TZAS__: string | null | undefined;

const TEXTURE_BINDING = 0x4;
const COPY_DST = 0x2;
const COPY_SRC = 0x1;
const STORAGE_BINDING = 0x8;

/** Half float bits to a number. */
export function halfToFloat(h: number): number {
  const s = (h & 0x8000) >> 15;
  const e = (h & 0x7c00) >> 10;
  const f = h & 0x03ff;
  if (e === 0) return (s ? -1 : 1) * 2 ** -14 * (f / 1024);
  if (e === 0x1f) return f ? NaN : (s ? -1 : 1) * Infinity;
  return (s ? -1 : 1) * 2 ** (e - 15) * (1 + f / 1024);
}

/** The aux planes OIDN takes, as rgba32float arrays: albedo rgb, normal in view space. */
export function oidnAux(aovs: PtAovs, camera: PtLookAtCamera): { albedo: Float32Array; normal: Float32Array } {
  const basis = ptCameraBasis(camera, aovs.width, aovs.height);
  const unit = (v: readonly number[]): number[] => {
    const l = Math.hypot(v[0]!, v[1]!, v[2]!);
    return [v[0]! / l, v[1]! / l, v[2]! / l];
  };
  const right = unit(basis.right);
  const up = unit(basis.up);
  const back = unit(basis.forward).map((v) => -v);
  const n = aovs.width * aovs.height;
  const albedo = new Float32Array(n * 4);
  const normal = new Float32Array(n * 4);
  for (let p = 0; p < n; p++) {
    for (let c = 0; c < 3; c++) albedo[p * 4 + c] = Math.min(1, Math.max(0, aovs.albedo[p * 3 + c] as number));
    albedo[p * 4 + 3] = 1;
    const x = aovs.normal[p * 3] as number;
    const y = aovs.normal[p * 3 + 1] as number;
    const z = aovs.normal[p * 3 + 2] as number;
    normal[p * 4] = x * right[0]! + y * right[1]! + z * right[2]!;
    normal[p * 4 + 1] = x * up[0]! + y * up[1]! + z * up[2]!;
    normal[p * 4 + 2] = x * back[0]! + y * back[1]! + z * back[2]!;
    normal[p * 4 + 3] = 1;
  }
  return { albedo, normal };
}

export interface OidnOptions {
  /** Base URL of the OIDN `.tza` weights. Default: the linked Denoiser checkout's `tzas`, served by the bench server (`/@fs/...`). */
  tzaUrl?: string;
  /** `fp16` needs `shader-f16` on the device. Default `fp32` (the reference quality). */
  precision?: 'fp32' | 'fp16';
}

let cached: { device: GPUDevice; denoiser: { denoiseTextures(o: unknown): Promise<GPUTexture | undefined> }; key: string } | null = null;

/** Denoises `image` (linear HDR, row 0 = top) with OIDN `rt_hdr_calb_cnrm` on `device`. Returns linear RGBA float32. */
export async function denoiseWithOidn(device: GPUDevice, image: Image, aovs: PtAovs, camera: PtLookAtCamera, options: OidnOptions = {}): Promise<Image> {
  const { width, height } = image;
  if (aovs.width !== width || aovs.height !== height) throw new Error('denoiseWithOidn: AOV and image sizes differ');
  const tzaUrl = options.tzaUrl ?? (typeof __BENCH_DENOISER_TZAS__ === 'string' ? __BENCH_DENOISER_TZAS__ : undefined);
  if (!tzaUrl) throw new Error('denoiseWithOidn: no OIDN weights: link the Denoiser checkout (bench/package.json `denoiser`) or pass tzaUrl');
  const precision = options.precision ?? 'fp32';
  const key = `${tzaUrl}|${precision}`;
  if (!cached || cached.device !== device || cached.key !== key) {
    const { Denoiser } = (await import('denoiser')) as unknown as { Denoiser: { create(o: unknown): Promise<{ denoiseTextures(o: unknown): Promise<GPUTexture | undefined> }> } };
    cached = { device, key, denoiser: await Denoiser.create({ device, precision, tzaUrl }) };
  }
  const aux = oidnAux(aovs, camera);
  const upload = (data: Float32Array): GPUTexture => {
    const texture = device.createTexture({ size: { width, height }, format: 'rgba32float', usage: TEXTURE_BINDING | COPY_DST });
    device.queue.writeTexture({ texture }, data as unknown as GPUAllowSharedBufferSource, { bytesPerRow: width * 16, rowsPerImage: height }, { width, height });
    return texture;
  };
  const color = new Float32Array(width * height * 4);
  for (let p = 0; p < width * height; p++) for (let c = 0; c < 4; c++) color[p * 4 + c] = c === 3 ? 1 : (image.data[p * image.channels + c] as number);
  const colorTexture = upload(color);
  const albedoTexture = upload(aux.albedo);
  const normalTexture = upload(aux.normal);
  const output = device.createTexture({ size: { width, height }, format: 'rgba16float', usage: STORAGE_BINDING | COPY_SRC | TEXTURE_BINDING });
  const result = await cached.denoiser.denoiseTextures({ color: colorTexture, albedo: albedoTexture, normal: normalTexture, hdr: true, output, transfer: 'linear' });
  if (!result) throw new Error('denoiseWithOidn: the denoiser run was aborted');
  // rgba16float readback: rows padded to 256 bytes
  const bytesPerRow = Math.ceil((width * 8) / 256) * 256;
  const staging = device.createBuffer({ size: bytesPerRow * height, usage: 0x1 /* GPUBufferUsage.MAP_READ */ | 0x8 /* GPUBufferUsage.COPY_DST */ });
  const encoder = device.createCommandEncoder({ label: 'oidn readback' });
  encoder.copyTextureToBuffer({ texture: result }, { buffer: staging, bytesPerRow, rowsPerImage: height }, { width, height });
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(1 /* GPUMapMode.READ */);
  const halves = new Uint16Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  const out = new Float32Array(width * height * 4);
  const stride = bytesPerRow / 2;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      for (let c = 0; c < 4; c++) out[(y * width + x) * 4 + c] = c === 3 ? 1 : halfToFloat(halves[y * stride + x * 4 + c] as number);
    }
  }
  for (const t of [colorTexture, albedoTexture, normalTexture, output]) t.destroy();
  return { width, height, channels: 4, data: out };
}
