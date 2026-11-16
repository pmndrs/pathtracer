/** Shared helpers for the research-track-N pages (mlp-check, mlp-fit, mlp-timing, mlp-live, mlp-sgm-probe). */

import { pickSubgroupMatrixConfig, type ForwardKind, type GradientKind, type MlpEngineOptions, type MlpShape, type Precision, type SgmScalar } from '@pathtracer/mlp';

/** A device with every adapter feature and the adapter's max limits (the MLP wants 16-32 KiB of workgroup memory). */
export async function requestMlpDevice(): Promise<{ adapter: GPUAdapter; device: GPUDevice }> {
  if (!navigator.gpu) throw new Error('navigator.gpu is undefined: no WebGPU in this browser.');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('requestAdapter() returned null: no usable GPU.');
  const requiredLimits: Record<string, number> = {};
  for (const name of ['maxComputeWorkgroupStorageSize', 'maxStorageBufferBindingSize', 'maxBufferSize', 'maxStorageBuffersPerShaderStage', 'maxComputeInvocationsPerWorkgroup']) {
    const v = (adapter.limits as unknown as Record<string, number>)[name];
    if (typeof v === 'number') requiredLimits[name] = v;
  }
  const device = await adapter.requestDevice({ requiredFeatures: [...adapter.features] as GPUFeatureName[], requiredLimits, label: 'mlp' });
  return { adapter, device };
}

/** Copies a buffer out and maps it. Only after the measured work, never inside a loop. */
export async function readBuffer(device: GPUDevice, src: GPUBuffer, bytes = src.size): Promise<ArrayBuffer> {
  const staging = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(src, 0, staging, 0, bytes);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const out = staging.getMappedRange().slice(0);
  staging.unmap();
  staging.destroy();
  return out;
}

export function storageBuffer(device: GPUDevice, data: ArrayBufferView | number, label: string): GPUBuffer {
  const size = Math.max(16, Math.ceil((typeof data === 'number' ? data : data.byteLength) / 16) * 16);
  const buffer = device.createBuffer({ label, size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  if (typeof data !== 'number') device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);
  return buffer;
}

/** f16 bits → number. */
export function halfToFloat(h: number): number {
  const s = h & 0x8000 ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const f = h & 0x3ff;
  if (e === 0) return s * f * 2 ** -24;
  if (e === 31) return f ? NaN : s * Infinity;
  return s * (1 + f / 1024) * 2 ** (e - 15);
}


export type InferArm = 'portable' | 'sgm' | 'sgm-emu';

export function hasSubgroupMatrix(device: GPUDevice): boolean {
  return device.features.has('chromium-experimental-subgroup-matrix' as GPUFeatureName) || device.features.has('subgroup-matrix' as GPUFeatureName);
}

/**
 * Engine precision + sgm options for an inference arm. `sgm` picks a config from the adapter (operand type decides
 * the precision); `sgm-emu` is the plain-WGSL stand-in (f32, any device, correctness only).
 */
export function inferArmOptions(
  arm: string,
  adapter: GPUAdapter,
  device: GPUDevice,
  shape: MlpShape,
  precision: Precision,
  opts: { records?: number; subgroups?: number; tile?: number; roundTrip?: 'relu' | 'skip' | 'double'; accumulate?: SgmScalar; layout?: 'row' | 'blocked'; pad?: number; inputProbe?: 'skip' | 'none'; reluRun?: number } = {},
): { precision: Precision; sgm?: MlpEngineOptions['sgm']; infer?: 'perRecord' | 'tiled'; description: string } {
  const records = opts.records ?? 64;
  const subgroups = opts.subgroups ?? 1;
  // 'portable' stays the per-record kernel (older result rows mean that); the engine's default portable kernel is 'tiled'.
  if (arm === 'portable') return { precision, infer: 'perRecord', description: `portable ${precision}` };
  if (arm === 'tiled') return { precision, infer: 'tiled', description: `portable tiled ${precision}` };
  if (arm === 'sgm-emu') {
    const t = opts.tile ?? 8;
    return {
      precision: 'f32',
      sgm: { config: { componentType: 'f32', resultComponentType: 'f32', M: t, N: t, K: t, minSubgroupSize: 32, maxSubgroupSize: 32 }, records, subgroups, emulate: true },
      description: `sgm emulated ${t}³ f32, ${records} rec × ${subgroups} sg`,
    };
  }
  if (arm === 'sgm') {
    if (!hasSubgroupMatrix(device)) throw new Error('infer=sgm needs chromium-experimental-subgroup-matrix (Chrome with --enable-unsafe-webgpu)');
    const config = pickSubgroupMatrixConfig(adapter.info, shape, precision === 'f16' && device.features.has('shader-f16'), opts.accumulate);
    if (!config) throw new Error('no usable subgroupMatrixConfigs entry on this adapter');
    if (opts.accumulate && config.resultComponentType !== opts.accumulate) throw new Error(`acc=${opts.accumulate}: no ${config.componentType}->${opts.accumulate} config on this adapter`);
    const enableName = device.features.has('subgroup-matrix' as GPUFeatureName) ? 'subgroup_matrix' : 'chromium_experimental_subgroup_matrix';
    return {
      precision: config.componentType,
      sgm: { config, records, subgroups, enableName, ...(opts.roundTrip ? { roundTrip: opts.roundTrip } : {}), ...(opts.layout ? { layout: opts.layout } : {}), ...(opts.pad !== undefined ? { pad: opts.pad } : {}), ...(opts.inputProbe ? { inputProbe: opts.inputProbe } : {}), ...(opts.reluRun ? { reluRun: opts.reluRun } : {}) },
      description: `sgm ${config.M}×${config.N}×${config.K} ${config.componentType}→${config.resultComponentType}, ${records} rec × ${subgroups} sg${opts.layout === 'blocked' ? `, blocked pad ${opts.pad ?? 8}` : ''}${opts.roundTrip && opts.roundTrip !== 'relu' ? `, roundTrip=${opts.roundTrip} (TIMING ONLY)` : ''}${opts.inputProbe ? `, inputProbe=${opts.inputProbe} (TIMING ONLY)` : ''}`,
    };
  }
  throw new Error(`infer must be portable, tiled, sgm or sgm-emu, got ${arm}`);
}

/**
 * Training forward-backward arm from the page arg `forward=perRecord|tiled|sgm|sgm-emu` (N3b). `sgm` / `sgm-emu` use the
 * subgroup-matrix kernel (config from the adapter, or the plain-WGSL stand-in at tile=8|16, f32). `fwdRecords` defaults
 * to 64 records per workgroup (f16) or 32 (f32 / emulated: 64 × 64 × 8 B would be the whole 32 KiB), rounded up to a
 * multiple of M × subgroups.
 */
export function forwardArmOptions(
  arg: string,
  adapter: GPUAdapter,
  device: GPUDevice,
  shape: MlpShape,
  precision: Precision,
  opts: { records?: number; subgroups?: number; tile?: number; accumulate?: SgmScalar; layout?: 'row' | 'blocked'; pad?: number; reluRun?: number; trainProbe?: 'noMaskRead' | 'noStores' | 'forwardOnly'; trainMask?: 'global' | 'bits' } = {},
): { forward: ForwardKind; precision: Precision; trainSgm?: MlpEngineOptions['trainSgm']; description: string } {
  if (arg === 'perRecord' || arg === 'tiled') return { forward: arg, precision, description: `${arg} ${precision}` };
  if (arg !== 'sgm' && arg !== 'sgm-emu') throw new Error(`forward must be perRecord, tiled, sgm or sgm-emu, got ${arg}`);
  const subgroups = opts.subgroups ?? 1;
  const base = inferArmOptions(arg, adapter, device, shape, precision, { records: 64, subgroups, tile: opts.tile ?? 8, ...(opts.accumulate ? { accumulate: opts.accumulate } : {}), ...(opts.layout ? { layout: opts.layout } : {}), ...(opts.pad !== undefined ? { pad: opts.pad } : {}), ...(opts.reluRun ? { reluRun: opts.reluRun } : {}) });
  const sgm = base.sgm as NonNullable<MlpEngineOptions['trainSgm']>;
  const unit = sgm.config.M * subgroups;
  const wanted = opts.records ?? (base.precision === 'f16' ? 64 : 32);
  const records = Math.ceil(wanted / unit) * unit;
  return { forward: 'sgm', precision: base.precision, trainSgm: { ...sgm, records, ...(opts.trainProbe ? { trainProbe: opts.trainProbe } : {}), ...(opts.trainMask ? { trainMask: opts.trainMask } : {}) }, description: base.description.replace(/\d+ rec/, `${records} rec`) + (opts.trainProbe ? `, trainProbe=${opts.trainProbe} (TIMING ONLY)` : '') };
}

/**
 * `forwardArmOptions` extras from page args: `fwdRecords`, `fwdSubgroups` (else `subgroups`, else `subgroups0`), `tile`, `acc`,
 * `fwdLayout=row|blocked`, `fwdPad` (else `pad`), `reluRun` (N2r-d: the blocked layout for the sgm forward-backward).
 */
export function forwardArgs(args: Record<string, string>, subgroups0 = 1): Parameters<typeof forwardArmOptions>[5] {
  const num = (k: string) => {
    const v = Number(args[k]);
    if (!Number.isFinite(v)) throw new Error(`Argument ${k}="${args[k]}" is not a number.`);
    return v;
  };
  const pad = args.fwdPad ?? args.pad;
  return {
    ...(args.fwdRecords ? { records: num('fwdRecords') } : {}),
    subgroups: args.fwdSubgroups ? num('fwdSubgroups') : args.subgroups ? num('subgroups') : subgroups0,
    tile: args.tile ? num('tile') : 8,
    ...(args.acc ? { accumulate: args.acc as SgmScalar } : {}),
    ...(args.fwdLayout ? { layout: args.fwdLayout as 'row' | 'blocked' } : {}),
    ...(pad !== undefined ? { pad: Number(pad) } : {}),
    ...(args.reluRun ? { reluRun: num('reluRun') } : {}),
    ...(args.fwdMask ? { trainMask: args.fwdMask as 'global' | 'bits' } : {}),
    ...(args.fwdProbe && args.fwdProbe !== 'none' ? { trainProbe: args.fwdProbe as 'noMaskRead' | 'noStores' | 'forwardOnly' } : {}),
  };
}

/** Weight-gradient arm from the page arg `gradient=perWeight|tiled|sgm` (sgm: f32-result config with the MLP's operand type, N2r-d). */
export function gradientArmOptions(
  arg: string,
  adapter: GPUAdapter,
  device: GPUDevice,
  shape: MlpShape,
  precision: Precision,
): { gradient: GradientKind; gradientSgm?: MlpEngineOptions['gradientSgm']; description: string } {
  if (arg === 'perWeight' || arg === 'tiled') return { gradient: arg, description: arg };
  if (arg !== 'sgm') throw new Error(`gradient must be perWeight, tiled or sgm, got ${arg}`);
  if (!hasSubgroupMatrix(device)) throw new Error('gradient=sgm needs chromium-experimental-subgroup-matrix');
  const config = pickSubgroupMatrixConfig(adapter.info, shape, precision === 'f16' && device.features.has('shader-f16'), 'f32');
  if (!config || config.componentType !== precision || config.resultComponentType !== 'f32') throw new Error(`gradient=sgm: no ${precision}->f32 subgroup-matrix config on this adapter`);
  const enableName = device.features.has('subgroup-matrix' as GPUFeatureName) ? 'subgroup_matrix' : 'chromium_experimental_subgroup_matrix';
  return { gradient: 'sgm', gradientSgm: { config, enableName }, description: `sgm ${config.M}×${config.N}×${config.K} ${config.componentType}→f32` };
}
