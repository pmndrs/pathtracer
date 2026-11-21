/**
 * GPU report: what this browser and GPU offer the libraries, as `detectCaps(adapter)` sees it, and what each optional feature is used for.
 * Paste the report into a bug report. `requestDeviceForCaps(adapter, caps)` requests every optional feature found here and the limits
 * the kernels can use; the tier (0-3) is a label for the feature set, not a speed rating.
 */
import { capsHash, type Caps } from '@pmndrs/raytracer';
import { EXAMPLES } from '../../catalog.ts';
import { defineExample } from '../../shared/example.ts';

const FLAGS: [keyof Caps, string, string][] = [
  ['subgroups', 'Subgroups', 'Wavefront queue appends (`appends: \'subgroup\'`), the auto policy, the MLP kernels'],
  ['shaderF16', 'shader-f16', 'f16 storage and maths in the research MLP; not needed by the path tracer'],
  ['timestampQuery', 'Timestamp queries', 'GPU timings in the bench; the examples time on the wall clock'],
  ['float32Filterable', 'float32-filterable', 'Filtered reads of float textures'],
  ['textureFormatsTier1', 'texture-formats-tier1', 'More storage texture formats'],
  ['subgroupSizeControl', 'Subgroup size control', 'Fixed subgroup sizes for kernels that need them'],
  ['subgroupMatrix', 'Subgroup matrices (experimental)', 'Tensor-core style matrix maths in the research MLP'],
  ['immediates', 'Immediates (WGSL)', 'Small per-dispatch constants without a uniform buffer'],
];
const LIMITS = ['maxStorageBufferBindingSize', 'maxBufferSize', 'maxStorageBuffersPerShaderStage', 'maxComputeWorkgroupStorageSize', 'maxComputeInvocationsPerWorkgroup', 'maxComputeWorkgroupsPerDimension', 'maxSampledTexturesPerShaderStage', 'maxBindGroups', 'maxTextureDimension2D', 'maxTextureArrayLayers'];

function el<K extends keyof HTMLElementTagNameMap>(tag: K, text?: string, className?: string): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (text !== undefined) e.textContent = text;
  if (className) e.className = className;
  return e;
}

function table(rows: (string | Node)[][], head?: string[]): HTMLTableElement {
  const t = el('table');
  if (head) {
    const tr = el('tr');
    for (const h of head) tr.append(el('th', h));
    t.append(tr);
  }
  for (const r of rows) {
    const tr = el('tr');
    for (const c of r) {
      const td = el('td');
      td.append(c);
      tr.append(td);
    }
    t.append(tr);
  }
  return t;
}

const yes = (on: boolean) => el('span', on ? 'yes' : 'no', on ? 'yes' : 'no');
const bytes = (n: number) => (n >= 1 << 20 ? `${(n / (1 << 20)).toFixed(0)} MiB` : n >= 1024 ? `${(n / 1024).toFixed(0)} KiB` : String(n));

defineExample(async (ctx) => {
  const { caps, adapter, device, pane, stats } = ctx;
  const style = el('style');
  style.textContent = `
    .report { position: fixed; inset: 64px 16px 16px; overflow: auto; max-width: 860px; margin: 0 auto; padding: 20px 24px; background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); }
    .report h2 { margin: 20px 0 8px; font-size: 15px; } .report h2:first-child { margin-top: 0; }
    .report table { border-collapse: collapse; width: 100%; font-size: 13px; }
    .report td, .report th { text-align: left; padding: 4px 8px; border-bottom: 1px solid var(--border); vertical-align: top; }
    .report th { color: var(--muted); font-weight: 500; }
    .report .yes { color: var(--accent); } .report .no { color: var(--muted); }
    .report code, .report .mono { font-family: var(--mono); font-size: 12px; }
    .report .tier { font-size: 28px; font-weight: 700; color: var(--accent); margin-right: 12px; }
  `;
  document.head.append(style);
  const root = el('div', undefined, 'report');
  document.body.append(root);

  const a = caps.adapter;
  root.append(el('h2', 'Adapter'));
  const tier = el('div');
  tier.append(el('span', `Tier ${caps.tier}`, 'tier'), el('span', caps.meetsBaseline ? 'Meets the baseline (128 MiB storage bindings, 16 KiB workgroup memory, 8 storage buffers)' : 'Below the baseline: some kernels will not fit'));
  root.append(tier, table([
    ['Vendor', a.vendor || '(hidden)'],
    ['Architecture', a.architecture || '(hidden)'],
    ['Device', a.device || '(hidden by the browser)'],
    ['Description', a.description || '(hidden without developer features)'],
    ['Fallback (software) adapter', yes(!!(adapter as GPUAdapter & { isFallbackAdapter?: boolean }).isFallbackAdapter || /swiftshader/i.test(`${a.description} ${a.architecture}`))],
    ['Subgroup sizes', caps.subgroupMaxSize ? `${caps.subgroupMinSize} – ${caps.subgroupMaxSize}` : '(not reported)'],
    ['Caps hash', el('span', capsHash(caps), 'mono')],
  ]));

  root.append(el('h2', 'Optional features'));
  root.append(table(FLAGS.map(([key, name, use]) => [name, yes(!!caps[key]), use]), ['Feature', 'Here', 'Used for']));

  root.append(el('h2', 'Limits of this device'));
  root.append(table(LIMITS.map((k) => [el('span', k, 'mono'), bytes(Number((device.limits as unknown as Record<string, number>)[k] ?? 0)), bytes(Number((adapter.limits as unknown as Record<string, number>)[k] ?? 0))]), ['Limit', 'Device', 'Adapter max']));

  root.append(el('h2', 'Examples'));
  const has = new Set(caps.features);
  root.append(table(EXAMPLES.map((e) => {
    const missing = (e.requires ?? []).filter((f) => !has.has(f));
    return [e.title, missing.length ? `needs ${missing.join(', ')}` : e.heavy ? 'runs (heavy)' : 'runs'];
  }), ['Example', 'Here']));

  root.append(el('h2', 'Raw'));
  const raw = el('pre', JSON.stringify({ ...caps, deviceLimits: Object.fromEntries(LIMITS.map((k) => [k, (device.limits as unknown as Record<string, number>)[k]])) }, null, 2), 'mono');
  raw.style.whiteSpace = 'pre-wrap';
  root.append(raw);

  const actions = { copy: () => void navigator.clipboard?.writeText(raw.textContent ?? '') };
  pane.addButton({ title: 'Copy the report' }).on('click', actions.copy);
  stats.set('tier', String(caps.tier));
  (globalThis as { __caps?: Caps }).__caps = caps;
});
