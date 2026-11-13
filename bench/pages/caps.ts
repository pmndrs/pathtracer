/**
 * Capability diagnostics page. Prints everything detectCaps sees, the derived tier, and whether a trivial
 * pipeline using each feature actually compiles on this browser + GPU. JSON is on `window.__benchResult`.
 */
import {
  buildCaps, capsHash, capsDefines, detectCaps, requestDeviceForCaps, wgslEnableLines,
  type Caps,
} from '../../packages/raytracer/src/caps/index.ts';

interface Probe { name: string; status: 'ok' | 'fail' | 'skip'; detail: string; ms: number }
declare global { interface Window { __benchResult?: unknown } }

const esc = (s: unknown) => String(s).replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' })[c]!);
const yn = (b: boolean) => `<span class="${b ? 'ok' : 'bad'}">${b ? 'yes' : 'no'}</span>`;
const row = (k: string, v: string) => `<tr><th>${esc(k)}</th><td>${v}</td></tr>`;
const table = (rows: string[]) => `<table>${rows.join('')}</table>`;
const chips = (xs: string[]) => (xs.length ? xs.map((x) => `<span class="chip">${esc(x)}</span>`).join('') : '<span class="mute">none</span>');
const mb = (n: number) => (n >= 1 << 30 ? `${(n / 2 ** 30).toFixed(2)} GiB` : `${(n / 2 ** 20).toFixed(0)} MiB`);

async function guarded(device: GPUDevice, fn: () => Promise<void> | void): Promise<string | null> {
  device.pushErrorScope('validation');
  let thrown: string | null = null;
  try { await fn(); } catch (e) { thrown = e instanceof Error ? e.message : String(e); }
  const err = await device.popErrorScope();
  return thrown ?? err?.message ?? null;
}

async function compileProbe(device: GPUDevice, code: string, immediateSize = 0): Promise<string | null> {
  return guarded(device, async () => {
    const module = device.createShaderModule({ code });
    const info = await module.getCompilationInfo();
    const errs = info.messages.filter((m) => m.type === 'error');
    if (errs.length) throw new Error(errs.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join('; '));
    const layout = device.createPipelineLayout({
      bindGroupLayouts: [device.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }] })],
      ...(immediateSize ? { immediateSize } : {}),
    } as GPUPipelineLayoutDescriptor);
    await device.createComputePipelineAsync({ layout, compute: { module, entryPoint: 'main' } });
  });
}

function probes(caps: Caps, device: GPUDevice): Array<[string, boolean, () => Promise<string | null>]> {
  const enables = wgslEnableLines(caps);
  const wrap = (enable: string, body: string) => `${enable}\n@group(0) @binding(0) var<storage, read_write> o: array<f32>;\n@compute @workgroup_size(32) fn main(@builtin(local_invocation_index) i: u32) { ${body} }`;
  const texFmt = (format: GPUTextureFormat) => async () =>
    guarded(device, () => { device.createTexture({ size: [4, 4], format, usage: GPUTextureUsage.TEXTURE_BINDING }).destroy(); });
  return [
    ['subgroups', caps.subgroups, () => compileProbe(device, wrap('enable subgroups;', 'o[i] = f32(subgroupAdd(i)) + f32(subgroupBallot(true).x);'))],
    ['subgroupSizeControl', caps.subgroupSizeControl, async () => null],
    ['shaderF16', caps.shaderF16, () => compileProbe(device, wrap('enable f16;', 'let h: f16 = f16(i); o[i] = f32(h * 2.0h);'))],
    ['timestampQuery', caps.timestampQuery, async () => guarded(device, () => { device.createQuerySet({ type: 'timestamp', count: 2 }).destroy(); })],
    ['immediates', caps.immediates, () => compileProbe(device, `var<immediate> k: vec4u;\n@group(0) @binding(0) var<storage, read_write> o: array<f32>;\n@compute @workgroup_size(32) fn main(@builtin(local_invocation_index) i: u32) { o[i] = f32(k.x + i); }`, 16)],
    ['subgroupMatrix', caps.subgroupMatrix, () => compileProbe(device, wrap(enables.split('\n').find((l) => l.includes('subgroup_matrix')) ?? '', 'var m: subgroup_matrix_result<f32, 8, 8>; o[i] = 0.0;'))],
    ['float32Filterable', caps.float32Filterable, async () => guarded(device, () => {
      const t = device.createTexture({ size: [4, 4], format: 'r32float', usage: GPUTextureUsage.TEXTURE_BINDING });
      const l = device.createBindGroupLayout({ entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } }] });
      device.createBindGroup({ layout: l, entries: [{ binding: 0, resource: t.createView() }, { binding: 1, resource: device.createSampler({ magFilter: 'linear' }) }] });
      t.destroy();
    })],
    ['compressed.bc', caps.compressed.bc, texFmt('bc7-rgba-unorm')],
    ['compressed.bcSliced3d', caps.compressed.bcSliced3d, async () => guarded(device, () => { device.createTexture({ size: [4, 4, 4], dimension: '3d', format: 'bc7-rgba-unorm', usage: GPUTextureUsage.TEXTURE_BINDING }).destroy(); })],
    ['compressed.astc', caps.compressed.astc, texFmt('astc-4x4-unorm')],
    ['compressed.etc2', caps.compressed.etc2, texFmt('etc2-rgb8unorm')],
    ['textureFormatsTier1', caps.textureFormatsTier1, async () => guarded(device, () => { device.createTexture({ size: [4, 4], format: 'r8snorm', usage: GPUTextureUsage.RENDER_ATTACHMENT }).destroy(); })],
  ];
}

async function main() {
  const out = document.getElementById('out')!;
  const result: Record<string, unknown> = { schema: 'caps/v1', userAgent: navigator.userAgent };
  window.__benchResult = result;
  if (!navigator.gpu) {
    out.innerHTML = '<span class="bad">navigator.gpu is undefined: no WebGPU in this browser/context (needs https or localhost).</span>';
    result.error = 'no-webgpu';
    return;
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) {
    out.innerHTML = '<span class="bad">requestAdapter() returned null.</span>';
    result.error = 'no-adapter';
    return;
  }
  const adapterCaps = detectCaps(adapter);
  const { device, caps } = await requestDeviceForCaps(adapter, adapterCaps, { label: 'caps-page' });
  const deviceErrors: string[] = [];
  device.addEventListener('uncapturederror', (e) => deviceErrors.push((e as GPUUncapturedErrorEvent).error.message));

  const probeResults: Probe[] = [];
  for (const [name, present, run] of probes(caps, device)) {
    const t0 = performance.now();
    if (!present) { probeResults.push({ name, status: 'skip', detail: 'feature absent', ms: 0 }); continue; }
    const err = await run();
    probeResults.push({ name, status: err ? 'fail' : 'ok', detail: err ?? (name === 'subgroupSizeControl' ? 'API-level feature, no WGSL to compile' : ''), ms: +(performance.now() - t0).toFixed(1) });
  }

  const allLimits: Record<string, number> = {};
  for (const k in adapter.limits) allLimits[k] = (adapter.limits as unknown as Record<string, number>)[k]!;
  const deviceLimits: Record<string, number> = {};
  for (const k in device.limits) deviceLimits[k] = (device.limits as unknown as Record<string, number>)[k]!;
  const hash = capsHash(caps);
  const adapterOnlyHash = capsHash(buildCaps({ features: adapter.features as ReadonlySet<string>, limits: adapter.limits as unknown as Record<string, unknown> }, new Set(caps.wgslLanguageFeatures), adapter.info as unknown as Record<string, unknown>));

  Object.assign(result, {
    tier: caps.tier, capsHash: hash, adapterCapsHash: adapterOnlyHash, caps, adapterCaps,
    defines: capsDefines(caps), enableLines: wgslEnableLines(caps),
    adapterFeatures: [...adapter.features].sort(), deviceFeatures: [...device.features].sort(),
    adapterLimits: allLimits, deviceLimits, wgslLanguageFeatures: caps.wgslLanguageFeatures,
    probes: probeResults, deviceErrors,
    isFallbackAdapter: (adapter.info as { isFallbackAdapter?: boolean }).isFallbackAdapter ?? null,
  });

  const flag = (label: string, b: boolean) => row(label, yn(b));
  const probeCls = { ok: 'ok', fail: 'bad', skip: 'mute' } as const;
  out.className = '';
  out.innerHTML = [
    `<div class="tier">Tier ${caps.tier} <span class="mute" style="font-size:14px">label only; baseline ${caps.meetsBaseline ? 'met' : '<span class="bad">NOT met</span>'}; caps hash ${hash}</span></div>`,
    '<h2>Adapter</h2>',
    table([
      row('vendor', esc(caps.adapter.vendor)), row('architecture', esc(caps.adapter.architecture)),
      row('device', esc(caps.adapter.device)), row('description', esc(caps.adapter.description)),
      row('subgroup size min / max', `${caps.subgroupMinSize} / ${caps.subgroupMaxSize}`),
      row('user agent', esc(navigator.userAgent)),
    ]),
    '<h2>Detected flags (as granted to the device)</h2>',
    table([
      flag('subgroups', caps.subgroups), flag('subgroupSizeControl', caps.subgroupSizeControl),
      flag('shaderF16', caps.shaderF16), flag('timestampQuery', caps.timestampQuery),
      flag('immediates (WGSL immediate_address_space)', caps.immediates),
      row('resourceTable', caps.resourceTable === 'none' ? '<span class="bad">none</span>' : `<span class="ok">${caps.resourceTable}</span>`),
      flag('subgroupMatrix', caps.subgroupMatrix), flag('float32Filterable', caps.float32Filterable),
      flag('textureFormatsTier1', caps.textureFormatsTier1), flag('accelerationStructure', caps.accelerationStructure),
      row('compressed', Object.entries(caps.compressed).map(([k, v]) => `<span class="chip ${v ? 'ok' : 'mute'}">${k}</span>`).join('')),
    ]),
    '<h2>Limits (device / adapter max)</h2>',
    table(Object.entries(caps.limits).map(([k, v]) => row(k, `${v.toLocaleString()} <span class="mute">${k.includes('Size') ? mb(v) : ''} / adapter ${adapterCaps.limits[k as keyof typeof caps.limits].toLocaleString()}</span>`))),
    '<h2>Compile probes</h2>',
    table(probeResults.map((p) => row(p.name, `<span class="${probeCls[p.status]}">${p.status}</span> ${esc(p.detail)} <span class="mute">${p.ms ? p.ms + ' ms' : ''}</span>`))),
    '<h2>WGSL helpers for this device</h2>',
    table([row('enable lines', `<pre>${esc(wgslEnableLines(caps) || '(none)')}</pre>`), row('defines', chips(Object.entries(capsDefines(caps)).map(([k, v]) => `${k}=${v}`)))]),
    '<h2>WGSL language features</h2>', chips(caps.wgslLanguageFeatures),
    `<h2>Raw adapter features (${adapter.features.size})</h2>`, chips([...adapter.features].sort()),
    `<h2>Raw device features (${device.features.size})</h2>`, chips([...device.features].sort()),
    '<h2>All adapter limits</h2>', table(Object.entries(allLimits).map(([k, v]) => row(k, v.toLocaleString()))),
    deviceErrors.length ? `<h2>Device errors</h2><pre class="bad">${esc(deviceErrors.join('\n'))}</pre>` : '',
    '<h2>JSON</h2><pre id="json"></pre>',
  ].join('');
  document.getElementById('json')!.textContent = JSON.stringify(result, null, 2);
  result.done = true;
}

main().catch((e) => {
  const msg = e instanceof Error ? e.stack ?? e.message : String(e);
  document.getElementById('out')!.innerHTML = `<pre class="bad">${esc(msg)}</pre>`;
  (window.__benchResult as Record<string, unknown> | undefined) ??= {};
  (window.__benchResult as Record<string, unknown>).error = msg;
});
