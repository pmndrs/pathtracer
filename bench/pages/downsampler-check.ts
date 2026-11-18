/**
 * downsampler-check (X5): `@pmndrs/downsampler` against our own code.
 *
 *  A. Texture mips. `mips: 'cpu'` builds every mip level of the texture arrays on the CPU (`buildSceneTextures`, 2x2 box in linear
 *     light for colour, on the raw bytes for data). The same textures built with the default `mips: 'gpu'` (level 0 only, D34) and uploaded
 *     (default `rtMipGenerator(device)`, or `mipGenerator: new Downsampler(device)`) get levels 1+ from the library. Every level of every bucket is read
 *     back and compared byte for byte with the CPU chain: levels, bytes that differ, the largest difference in codes (a rounding step is
 *     the documented difference; anything above 2 fails), per colour space. Both `averageAccess: 'sample'` (hardware bilinear, default)
 *     and `'load'` (exact weights).
 *  B. Cost (provisional wall time, one machine): CPU build + upload against the level-0 build + GPU generate + upload, 24 textures
 *     of 1024x1024 and 24 of 2048x2048 (a third linear), min of `repeats` runs.
 *  C. Scalar reduction. The auto-exposure of the real-time chain (mean log luma through a `load` / `finish` hook pair) and a plain
 *     per-channel mean, on an rgba32float image of an odd size, against f64 CPU twins (`autoExposureCpu`).
 *
 * Args: repeats=3  size=1920x1081 (part C)  big=1 (part B at 2048x2048 too; default 1)
 */
import { Downsampler, ScalarReduction, requiredFeatures, requiredLimits } from '@pmndrs/downsampler/webgpu';
import { buildSceneTextures, rtMipGenerator, uploadSceneTextures, type SceneTextures, type TextureInput } from '@pmndrs/raytracer';
import type { AdapterSummary } from '../runner/contract.ts';
import { drainDevice, getArgs, numberArg, reportError, reportResult, summarizeAdapter, watchDevice } from '../runner/page-client.ts';
import { autoExposureCpu } from '../realtime/chain-math.ts';

let adapterSummary: AdapterSummary | null = null;
let deviceErrors: string[] = [];

/** A texture with smooth gradients, hard edges and noise, saturated in places (so rounding and clamping both show). */
function pattern(width: number, height: number, srgb: boolean, seed: number): TextureInput {
  const data = new Uint8Array(width * height * 4);
  let state = (seed * 2654435761) >>> 0;
  const rand = () => ((state = (Math.imul(state, 1664525) + 1013904223) >>> 0) / 4294967296);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;
      const edge = ((x >> 4) + (y >> 4)) & 1 ? 1 : 0;
      for (let c = 0; c < 3; c++) {
        const smooth = 0.5 + 0.5 * Math.sin((x / width) * (3 + c + seed) * Math.PI + (y / height) * (2 + seed));
        const v = 0.6 * smooth + 0.3 * edge + 0.25 * (rand() - 0.5) + (c === seed % 3 ? 0.2 : 0);
        data[i + c] = Math.round(255 * Math.min(1, Math.max(0, v)));
      }
      data[i + 3] = (x + y) % 7 === 0 ? 255 : Math.round(255 * rand());
    }
  }
  return { width, height, colorSpace: srgb ? 'srgb' : 'linear', data, label: `pattern ${seed}` };
}

const align256 = (n: number) => Math.ceil(n / 256) * 256;

async function readLevel(device: GPUDevice, texture: GPUTexture, level: number, layers: number): Promise<Uint8Array> {
  const w = Math.max(1, texture.width >> level);
  const h = Math.max(1, texture.height >> level);
  const stride = align256(w * 4);
  const staging = device.createBuffer({ size: stride * h * layers, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer({ texture, mipLevel: level }, { buffer: staging, bytesPerRow: stride, rowsPerImage: h }, [w, h, layers]);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const raw = new Uint8Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  const out = new Uint8Array(w * h * layers * 4);
  for (let layer = 0; layer < layers; layer++) for (let y = 0; y < h; y++) out.set(raw.subarray((layer * h + y) * stride, (layer * h + y) * stride + w * 4), ((layer * h + y) * w) * 4);
  return out;
}

async function readBuffer(device: GPUDevice, buffer: GPUBuffer, size: number): Promise<ArrayBuffer> {
  const staging = device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, staging, 0, size);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const out = staging.getMappedRange().slice(0);
  staging.unmap();
  staging.destroy();
  return out;
}

try {
  const args = getArgs();
  const repeats = numberArg(args, 'repeats', 3);
  const [sizeW, sizeH] = (args.size ?? '1920x1081').split('x').map(Number) as [number, number];
  const big = (args.big ?? '1') !== '0';
  if (!navigator.gpu) throw new Error('navigator.gpu is undefined: no WebGPU in this browser.');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('requestAdapter() returned null.');
  adapterSummary = summarizeAdapter(adapter);
  const device = await adapter.requestDevice({ label: 'downsampler-check', requiredFeatures: requiredFeatures(adapter) as GPUFeatureName[], requiredLimits: requiredLimits(adapter) as Record<string, number> });
  deviceErrors = watchDevice(device);

  const metrics: Record<string, number> = {};
  const notes: Record<string, unknown> = {};
  const failures: string[] = [];

  // ------------------------------------------------------------------------------------------------ A. mips
  {
    const inputs: TextureInput[] = [
      ...[1, 2, 3, 4, 5, 6].map((seed) => pattern(512, 512, true, seed)),
      ...[7, 8, 9].map((seed) => pattern(512, 512, false, seed)),
      ...[10, 11].map((seed) => pattern(256, 128, true, seed)),
      pattern(1024, 256, false, 12),
      pattern(640, 480, true, 13), // not a power of two: stored 512x512 or 1024x512 after the resample (the same for both builds)
    ];
    const cpu = buildSceneTextures(inputs, { mips: 'cpu' });
    // 'sample' / 'load' take the buffer path for the sRGB buckets (all below the 1920x1080 `renderPathMinTexels`); 'render' lowers that
    // threshold so every sRGB chain goes through render passes instead (linear chains stay on storage). The two must give the same bits.
    const variants: Array<[string, ConstructorParameters<typeof Downsampler>[1]]> = [['sample', {}], ['load', { averageAccess: 'load' }], ['render', { renderPathMinTexels: 1 }], ['render-load', { renderPathMinTexels: 1, averageAccess: 'load' }]];
    const srgbLevels: Record<string, Uint8Array[]> = {};
    for (const [name, options] of variants) {
      const srgbOut: Uint8Array[] = (srgbLevels[name] = []);
      const downsampler = new Downsampler(device, options);
      const gpuBuilt = buildSceneTextures(inputs); // the default is 'gpu' (D34)
      const uploaded = uploadSceneTextures(device, gpuBuilt, `mips ${name}`, downsampler, true);
      const uploadedFor = (_bucket: number) => ({ textures: uploaded.textures.slice(_bucket, _bucket + 1) });
      await device.queue.onSubmittedWorkDone();
      notes[`mipsPaths_${name}`] = uploaded.textures.map((t, b) => `${cpu.buckets[b]!.colorSpace}: ${downsampler.outputPath(t)}`);
      const stats: Record<string, { bytes: number; differing: number; maxDiff: number; over1: number }> = {};
      let worst = 0;
      for (let b = 0; b < cpu.buckets.length; b++) {
        const bucket = cpu.buckets[b]!;
        const texture = uploaded.textures[b]!;
        for (let level = 0; level < bucket.mipLevels; level++) {
          const gpuBytes = await readLevel(device, texture, level, bucket.layers);
          const cpuBytes = bucket.mips[level]!;
          if (bucket.colorSpace === 'srgb') srgbOut.push(gpuBytes);
          const key = `${bucket.colorSpace}_L${level === 0 ? '0' : level <= 2 ? '1-2' : level <= 5 ? '3-5' : '6+'}`;
          const s = (stats[key] ??= { bytes: 0, differing: 0, maxDiff: 0, over1: 0 });
          for (let i = 0; i < cpuBytes.length; i++) {
            const d = Math.abs((gpuBytes[i] as number) - (cpuBytes[i] as number));
            s.bytes++;
            if (d) s.differing++;
            if (d > 1) s.over1++;
            if (d > s.maxDiff) s.maxDiff = d;
            if (d > worst) worst = d;
          }
        }
      }
      // Which chain is closer to the exact box mean of level 0? (linear data textures, power-of-two buckets: mean over the 2^L x 2^L block, no intermediate rounding.)
      const linear = cpu.buckets.findIndex((b) => b.colorSpace === 'linear' && b.width === b.height);
      {
        const bucket = cpu.buckets[linear]!;
        let cpuErr = 0;
        let gpuErr = 0;
        let n = 0;
        for (let level = 1; level < bucket.mipLevels; level++) {
          const w = Math.max(1, bucket.width >> level);
          const h = Math.max(1, bucket.height >> level);
          const gpuBytes = await readLevel(device, uploadedFor(linear).textures[0]!, level, bucket.layers);
          const base = bucket.mips[0]!;
          const step = bucket.width / w;
          for (let layer = 0; layer < bucket.layers; layer++)
            for (let y = 0; y < h; y += Math.max(1, h >> 4))
              for (let x = 0; x < w; x += Math.max(1, w >> 4))
                for (let c = 0; c < 3; c++) {
                  let sum = 0;
                  for (let by = 0; by < step; by++) for (let bx = 0; bx < step; bx++) sum += base[((layer * bucket.height + y * step + by) * bucket.width + x * step + bx) * 4 + c] as number;
                  const exact = sum / (step * step);
                  const at = ((layer * h + y) * w + x) * 4 + c;
                  cpuErr += Math.abs((bucket.mips[level]![at] as number) - exact);
                  gpuErr += Math.abs((gpuBytes[at] as number) - exact);
                  n++;
                }
        }
        metrics[`mips_${name}_linear_cpu_chain_mean_abs_err`] = cpuErr / n;
        metrics[`mips_${name}_linear_gpu_chain_mean_abs_err`] = gpuErr / n;
      }
      for (const [key, s] of Object.entries(stats)) {
        metrics[`mips_${name}_${key}_differing_frac`] = s.differing / s.bytes;
        metrics[`mips_${name}_${key}_max_diff`] = s.maxDiff;
        metrics[`mips_${name}_${key}_over1`] = s.over1;
      }
      metrics[`mips_${name}_worst_diff`] = worst;
      if (worst > 2) failures.push(`mips ${name}: a code differs by ${worst} from the CPU chain (documented: a rounding step)`);
      if (stats['srgb_L0']!.differing || stats['linear_L0']!.differing) failures.push(`mips ${name}: level 0 changed`);
      for (const t of uploaded.textures) t.destroy();
    }
    // sRGB render path vs buffer path, per source access: the library documents the same bits (srgbRounding 'exact', the default).
    for (const [access, buffer, render] of [['sample', 'sample', 'render'], ['load', 'load', 'render-load']] as const) {
      let differing = 0;
      let maxDiff = 0;
      let bytes = 0;
      const a = srgbLevels[buffer]!;
      const b = srgbLevels[render]!;
      for (let l = 0; l < a.length; l++)
        for (let i = 0; i < a[l]!.length; i++) {
          const d = Math.abs((a[l]![i] as number) - (b[l]![i] as number));
          bytes++;
          if (d) differing++;
          if (d > maxDiff) maxDiff = d;
        }
      metrics[`mips_srgb_${access}_render_vs_buffer_differing_bytes`] = differing;
      metrics[`mips_srgb_${access}_render_vs_buffer_differing_frac`] = differing / bytes;
      metrics[`mips_srgb_${access}_render_vs_buffer_max_diff`] = maxDiff;
    }
    // The library default (D34): no generator passed, `rtMipGenerator(device)` fills the levels; same bytes as an explicit default Downsampler.
    {
      const explicit = uploadSceneTextures(device, buildSceneTextures(inputs), 'mips explicit', new Downsampler(device), true);
      const byDefault = uploadSceneTextures(device, buildSceneTextures(inputs), 'mips default', undefined, true);
      const again = uploadSceneTextures(device, buildSceneTextures(inputs), 'mips default again', undefined, true); // reuses the cached Downsampler
      let differing = 0;
      for (let b = 0; b < cpu.buckets.length; b++)
        for (let level = 0; level < cpu.buckets[b]!.mipLevels; level++) {
          const a = await readLevel(device, explicit.textures[b]!, level, cpu.buckets[b]!.layers);
          for (const other of [byDefault, again]) {
            const o = await readLevel(device, other.textures[b]!, level, cpu.buckets[b]!.layers);
            for (let i = 0; i < a.length; i++) if (a[i] !== o[i]) differing++;
          }
        }
      metrics.mips_default_vs_explicit_differing_bytes = differing;
      if (differing) failures.push(`mips: the default generator differs from an explicit Downsampler in ${differing} bytes`);
      if (rtMipGenerator(device) !== rtMipGenerator(device)) failures.push('mips: rtMipGenerator is not cached per device');
      for (const u of [explicit, byDefault, again]) for (const t of u.textures) t.destroy();
    }
    metrics.mips_buckets = cpu.buckets.length;
    notes.mipsBuckets = cpu.buckets.map((b) => `${b.width}x${b.height}x${b.layers} ${b.colorSpace} ${b.mipLevels} levels`);
  }

  // ------------------------------------------------------------------------------------------------ B. cost
  {
    const downsampler = new Downsampler(device);
    const sizes = big ? [1024, 2048] : [1024];
    for (const size of sizes) {
      const inputs: TextureInput[] = [...Array.from({ length: 16 }, (_, i) => pattern(size, size, true, i + 1)), ...Array.from({ length: 8 }, (_, i) => pattern(size, size, false, i + 20))];
      const run = async (mode: 'cpu' | 'gpu'): Promise<{ build: number; upload: number }> => {
        const t0 = performance.now();
        const built: SceneTextures = buildSceneTextures(inputs, { mips: mode });
        const t1 = performance.now();
        const uploaded = uploadSceneTextures(device, built, `cost ${mode}`, mode === 'gpu' ? downsampler : undefined);
        await device.queue.onSubmittedWorkDone();
        const t2 = performance.now();
        for (const t of uploaded.textures) t.destroy();
        return { build: t1 - t0, upload: t2 - t1 };
      };
      await run('gpu'); // compile the pipelines once, outside the numbers
      for (const mode of ['cpu', 'gpu'] as const) {
        const rows: Array<{ build: number; upload: number }> = [];
        for (let r = 0; r < repeats; r++) rows.push(await run(mode));
        const best = rows.reduce((a, b) => (a.build + a.upload <= b.build + b.upload ? a : b));
        metrics[`cost_${size}_${mode}_build_ms`] = best.build;
        metrics[`cost_${size}_${mode}_upload_ms`] = best.upload;
        metrics[`cost_${size}_${mode}_total_ms`] = best.build + best.upload;
      }
      metrics[`cost_${size}_cpu_over_gpu`] = (metrics[`cost_${size}_cpu_total_ms`] as number) / (metrics[`cost_${size}_gpu_total_ms`] as number);
    }
  }

  // ------------------------------------------------------------------------------------------------ C. scalar reductions
  {
    const w = sizeW;
    const h = sizeH;
    const image = new Float32Array(w * h * 4);
    let state = 12345;
    const rand = () => ((state = (Math.imul(state, 1664525) + 1013904223) >>> 0) / 4294967296);
    for (let i = 0; i < w * h; i++) {
      const s = Math.exp(3 * (rand() - 0.5) * 4 - 1);
      image[i * 4] = s * (0.5 + rand());
      image[i * 4 + 1] = s * (0.5 + rand());
      image[i * 4 + 2] = s * (0.5 + rand());
      image[i * 4 + 3] = i % 11 === 0 ? 0 : 1; // a few black texels (the luma floor), the rest lit
      if (i % 11 === 0) image.fill(0, i * 4, i * 4 + 3);
    }
    const texture = device.createTexture({ size: [w, h], format: 'rgba32float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    device.queue.writeTexture({ texture }, image, { bytesPerRow: w * 16 }, [w, h]);

    // C1: the chain's auto-exposure (hooks), result into a storage buffer.
    const out = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const exposure = new ScalarReduction(device, {
      bindings: {
        wgsl: '@group(1) @binding(0) var colour: texture_2d<f32>;\n@group(1) @binding(1) var<storage, read_write> exposureOut: array<f32>;',
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        ],
      },
      load: `fn load(texel: vec2i, layer: u32) -> vec4f {
  let c = max(textureLoad(colour, texel, 0).rgb, vec3f(0.0));
  return vec4f(log(max(dot(c, vec3f(0.2126, 0.7152, 0.0722)), 6.1e-5)), 0.0, 0.0, 0.0);
}`,
      finish: `fn finish(value: vec4f, weight: f32) {
  exposureOut[0] = clamp(0.18 / exp(value.x), 0.015625, 64.0);
  exposureOut[1] = value.x;
  exposureOut[2] = weight;
}`,
    });
    const group = device.createBindGroup({ layout: exposure.bindGroupLayouts[1] as GPUBindGroupLayout, entries: [{ binding: 0, resource: texture.createView() }, { binding: 1, resource: { buffer: out } }] });
    const encoder = device.createCommandEncoder();
    exposure.encode(encoder, { width: w, height: h, bindGroup: group });
    device.queue.submit([encoder.finish()]);
    const got = new Float32Array(await readBuffer(device, out, 16));
    const cpu = autoExposureCpu(image);
    metrics.exposure_gpu = got[0] as number;
    metrics.exposure_cpu = cpu.exposure;
    metrics.exposure_rel_err = Math.abs((got[0] as number) - cpu.exposure) / cpu.exposure;
    metrics.mean_log_luma_abs_err = Math.abs((got[1] as number) - cpu.meanLogLuma);
    metrics.weight_gpu = got[2] as number;
    if (got[2] !== w * h) failures.push(`scalar weight ${got[2]} is not the texel count ${w * h} (an odd size must count every texel once)`);
    if (metrics.exposure_rel_err > 1e-5) failures.push(`auto exposure off by ${metrics.exposure_rel_err}`);

    // C2: plain per-channel mean of the source (no hooks), result to a buffer.
    const mean = new ScalarReduction(device, {});
    const meanOut = device.createBuffer({ size: mean.resultBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const encoder2 = device.createCommandEncoder();
    mean.encode(encoder2, { source: texture, output: { buffer: meanOut } });
    device.queue.submit([encoder2.finish()]);
    const m = new Float32Array(await readBuffer(device, meanOut, mean.resultBytes));
    const twin = [0, 0, 0, 0];
    for (let i = 0; i < w * h; i++) for (let c = 0; c < 4; c++) twin[c]! += image[i * 4 + c]!;
    let worst = 0;
    for (let c = 0; c < 4; c++) worst = Math.max(worst, Math.abs(m[c]! - twin[c]! / (w * h)) / (twin[c]! / (w * h)));
    metrics.mean_rel_err = worst;
    if (worst > 1e-5) failures.push(`plain mean off by ${worst}`);
    texture.destroy();
    out.destroy();
    meanOut.destroy();
  }

  await drainDevice(device);
  if (deviceErrors.length) failures.push(`device errors: ${deviceErrors.join('; ')}`);
  if (failures.length) throw new Error(`downsampler-check failed: ${failures.join('; ')}`);
  reportResult({ scene: 'synthetic', renderer: 'downsampler-check', config: { repeats, size: `${sizeW}x${sizeH}`, big }, metrics, notes }, { adapter: adapterSummary, deviceErrors });
} catch (error) {
  reportError(error, { adapter: adapterSummary, deviceErrors });
}
