/**
 * Research track N, N0.2: what this browser + device offer for subgroup matrices, and which kernel forms compile.
 * Reports: adapter features, WGSL language features, subgroup sizes, every `subgroupMatrixConfigs` entry, and for each
 * usable config the compile + pipeline result of the generated inference kernel (nnInferSgm) in these forms:
 *   one-subgroup          1 subgroup per workgroup (the Denoiser's working form)
 *   subgroup-id ×4        4 subgroups, row blocks split by @builtin(subgroup_id) (needs subgroup_uniformity)
 *   local-index ×4        4 subgroups, split by local_invocation_index / size (expected to FAIL: the control case)
 *   local-index ×4 + diagnostic(off, <name>) for each candidate name of the matrix uniformity diagnostic
 * Never fails because a form does not compile: that is the information. Fails only if the page itself breaks.
 * Run with Chrome's --enable-unsafe-webgpu so the experimental feature is exposed.
 */

import { BUFFER_HOOKS_INFER, DEFAULT_SHAPE, inferSgmWgsl, pickSubgroupMatrixConfig, sgmWorkgroupBytes, type SgmConfig, type SgmOptions } from '@pathtracer/mlp';
import { drainDevice, reportError, reportResult, summarizeAdapter, watchDevice } from '../runner/page-client.ts';
import type { AdapterSummary } from '../runner/contract.ts';
import { hasSubgroupMatrix, requestMlpDevice } from './mlp-common.ts';

const DIAGNOSTIC_NAMES = ['subgroup_matrix_uniformity', 'chromium.subgroup_matrix_uniformity'];

let adapterSummary: AdapterSummary | null = null;
let deviceErrors: string[] = [];

function plainConfigs(info: GPUAdapterInfo): Array<Record<string, unknown>> {
  const raw = (info as unknown as { subgroupMatrixConfigs?: ArrayLike<object> }).subgroupMatrixConfigs;
  if (!raw) return [];
  return Array.from(raw, (c) => {
    const out: Record<string, unknown> = {};
    for (const key of ['componentType', 'resultComponentType', 'M', 'N', 'K', 'minSubgroupSize', 'maxSubgroupSize']) out[key] = (c as Record<string, unknown>)[key];
    return out;
  });
}

async function tryCompile(device: GPUDevice, code: string): Promise<{ ok: boolean; stage: string; message: string }> {
  device.pushErrorScope('validation');
  const module = device.createShaderModule({ code });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((m) => m.type === 'error');
  if (errors.length) {
    await device.popErrorScope();
    return { ok: false, stage: 'shader', message: errors.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join(' | ').slice(0, 600) };
  }
  try {
    await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'nnInferSgm' } });
  } catch (e) {
    await device.popErrorScope();
    return { ok: false, stage: 'pipeline', message: String((e as Error).message ?? e).slice(0, 600) };
  }
  const err = await device.popErrorScope();
  return err ? { ok: false, stage: 'validation', message: err.message.slice(0, 600) } : { ok: true, stage: 'pipeline', message: '' };
}

try {
  const { adapter, device } = await requestMlpDevice();
  adapterSummary = summarizeAdapter(adapter);
  deviceErrors = watchDevice(device);
  const info = adapter.info as GPUAdapterInfo & { subgroupMinSize?: number; subgroupMaxSize?: number };
  const languageFeatures = [...navigator.gpu.wgslLanguageFeatures].sort();
  const configs = plainConfigs(info);
  const notes: Record<string, unknown> = {
    userAgent: navigator.userAgent,
    subgroupMatrixFeature: hasSubgroupMatrix(device),
    subgroupSizes: [info.subgroupMinSize, info.subgroupMaxSize],
    shaderF16: device.features.has('shader-f16'),
    maxWorkgroupStorage: device.limits.maxComputeWorkgroupStorageSize,
    languageFeatures,
    configs,
  };
  const metrics: Record<string, number> = { configs: configs.length, forms_ok: 0, forms_failed: 0 };

  if (hasSubgroupMatrix(device)) {
    const enableName = device.features.has('subgroup-matrix' as GPUFeatureName) ? 'subgroup_matrix' : 'chromium_experimental_subgroup_matrix';
    const results: Array<Record<string, unknown>> = [];
    const usable = new Map<string, SgmConfig>();
    for (const preferF16 of [true, false]) {
      const c = pickSubgroupMatrixConfig(info, DEFAULT_SHAPE, preferF16 && device.features.has('shader-f16'));
      if (c) usable.set(`${c.componentType}->${c.resultComponentType} ${c.M}x${c.N}x${c.K}`, c);
    }
    for (const [label, config] of usable) {
      const forms: Array<[string, Partial<SgmOptions>]> = [
        ['one-subgroup', { subgroups: 1 }],
        ['subgroup-id x4', { subgroups: 4 }],
        ['local-index x4 (control)', { subgroups: 4, offsetsFromLocalIndex: true }],
        ...DIAGNOSTIC_NAMES.map((name): [string, Partial<SgmOptions>] => [`local-index x4 + diagnostic(off, ${name})`, { subgroups: 4, offsetsFromLocalIndex: true, uniformityDiagnosticOff: name }]),
        ...DIAGNOSTIC_NAMES.map((name): [string, Partial<SgmOptions>] => [`subgroup-id x4 + diagnostic(off, ${name})`, { subgroups: 4, uniformityDiagnosticOff: name }]),
      ];
      for (const [form, partial] of forms) {
        // Records: 64 per workgroup, or as few as the tile allows when workgroup memory is short.
        let records = 64;
        const opts = (r: number): SgmOptions => ({ shape: DEFAULT_SHAPE, config, records: r, subgroups: 1, enableName, ...partial });
        while (records > config.M * (partial.subgroups ?? 1) && sgmWorkgroupBytes(opts(records)) > device.limits.maxComputeWorkgroupStorageSize) records /= 2;
        let code = '';
        let r: { ok: boolean; stage: string; message: string };
        try {
          code = inferSgmWgsl(opts(records), BUFFER_HOOKS_INFER);
          r = await tryCompile(device, code);
        } catch (e) {
          r = { ok: false, stage: 'generator', message: String((e as Error).message ?? e) };
        }
        metrics[r.ok ? 'forms_ok' : 'forms_failed'] = (metrics[r.ok ? 'forms_ok' : 'forms_failed'] as number) + 1;
        results.push({ config: label, form, records, ...r, wgslBytes: code.length });
      }
    }
    notes.results = results;
  } else {
    notes.results = 'subgroup matrix feature not exposed: start Chrome with --enable-unsafe-webgpu (and a recent Chrome)';
  }
  await drainDevice(device);
  // Compile failures are findings, so they do not fail the run; but device errors from a failed pipeline would, so
  // they were captured in error scopes above.
  reportResult({ scene: 'sgm-probe', renderer: 'probe', metrics, notes }, { adapter: adapterSummary, deviceErrors });
} catch (error) {
  reportError(error, { adapter: adapterSummary, deviceErrors });
}
