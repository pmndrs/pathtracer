/**
 * kernel-compile (D26 / W10): how long the driver's shader compiler takes for each megakernel variant. No dispatches, no readback.
 *
 * For every variant of `bench/kernel-variants.ts` (built with the REAL device's caps) and every repeat:
 *  - cold: `createShaderModule` + `await createComputePipelineAsync` on text with a unique first-line comment (nonce), so Chrome/Dawn's
 *    shader cache cannot match (its key is a hash of the source text);
 *  - warm: the identical text again straight after. warm << cold proves the cache works and that cold is a real compile.
 * Compiles run one at a time. `layout: 'auto'`, entry point `ptMain`.
 *
 * The nonce must change the GENERATED code, not only the WGSL text: a comment dies in Tint, so the HLSL/DXIL (D3D12) or MSL (Metal) is
 * identical and a backend/driver cache (system-wide on Metal, NVIDIA's DXIL cache) turns repeat 2+ into a lookup. Measured on this Mac
 * with a comment nonce: repeat 1 = 130-475 ms, repeats 2-3 = 9-12 ms, i.e. a cache hit. So the default `nonce=code` rewrites the live
 * literal `vec4f(l.xyz, 1.0)` in `ptMain` to `1 + n * 2^-23` (a different f32 each compile, never folded away); `nonce=comment` is the old
 * comment-only mode, kept to show the difference.
 *
 * `set=wavefront` (V1.8) compiles the wavefront kernels instead (`wavefrontKernelVariants`, `wf-<variant>-<kernel>`, one entry point each, found in the
 * text). They have no `ptMain` accumulate line, so their default nonce is `literal`: every standalone `1.0` becomes a unique f32 near 1 (the kernels are
 * compiled, never dispatched, so the value does not matter; `1.0` appears in live code of every kernel). Check `cold_ms_per_repeat` for a cache hit.
 *
 * Args: `set=megakernel|wavefront`, `variants=a,b` (exact names) or `prefix=vol-,plain` (name prefixes), `repeat=3`, `warm=1` (0 skips the warm
 * compile), `nonce=code|literal|comment` (default code for the megakernel, literal for the wavefront), `packing=split|packed`.
 * Provisional unless the machine was quiet (the runner records the load check).
 */
import { detectCaps, requestDeviceForCaps } from '@pmndrs/raytracer';
import type { AdapterSummary } from '../runner/contract.ts';
import { drainDevice, getArgs, numberArg, reportError, reportResult, summarizeAdapter, watchDevice } from '../runner/page-client.ts';
import { kernelVariants, wavefrontKernelVariants } from '../kernel-variants.ts';

const ENTRY_POINT = 'ptMain';
let adapterSummary: AdapterSummary | null = null;
let deviceErrors: string[] = [];

const stats = (xs: number[]) => {
  const s = [...xs].sort((a, b) => a - b);
  const mid = s.length >> 1;
  return { min: s[0] ?? NaN, median: s.length % 2 ? (s[mid] as number) : ((s[mid - 1] as number) + (s[mid] as number)) / 2, max: s[s.length - 1] ?? NaN };
};
const round = (x: number) => Math.round(x * 100) / 100;

try {
  if (!navigator.gpu) throw new Error('navigator.gpu is undefined: no WebGPU in this browser.');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('requestAdapter() returned null: no usable GPU.');
  adapterSummary = summarizeAdapter(adapter);
  const { device, caps } = await requestDeviceForCaps(adapter, detectCaps(adapter), { label: 'kernel-compile' });
  deviceErrors = watchDevice(device);

  const args = getArgs();
  const repeat = Math.max(1, numberArg(args, 'repeat', 3));
  const warmOn = numberArg(args, 'warm', 1) !== 0;
  const filter = args.variants ? args.variants.split(',').map((s) => s.trim()).filter(Boolean) : null;
  const prefixes = args.prefix ? args.prefix.split(',').map((s) => s.trim()).filter(Boolean) : null;
  if (args.packing && args.packing !== 'split' && args.packing !== 'packed') throw new Error(`packing must be split or packed, got '${args.packing}'`);
  const set = args.set ?? 'megakernel';
  if (set !== 'megakernel' && set !== 'wavefront') throw new Error(`set=${set}: use megakernel or wavefront`);
  const all = (set === 'wavefront' ? wavefrontKernelVariants : kernelVariants)(caps, args.packing as 'split' | 'packed' | undefined);
  const names = Object.keys(all).filter((n) => (!filter || filter.includes(n)) && (!prefixes || prefixes.some((p) => n.startsWith(p))));
  if (names.length === 0) throw new Error(`No variant matches; known: ${Object.keys(all).join(', ')}`);
  if (filter) for (const f of filter) if (!(f in all)) throw new Error(`Unknown variant "${f}"; known: ${Object.keys(all).join(', ')}`);

  const nonceMode = args.nonce ?? (set === 'wavefront' ? 'literal' : 'code');
  if (nonceMode !== 'code' && nonceMode !== 'literal' && nonceMode !== 'comment') throw new Error(`nonce=${nonceMode}: use code, literal or comment`);
  if (set === 'wavefront' && nonceMode === 'code') throw new Error('nonce=code needs the megakernel (ptMain); use literal for set=wavefront');
  const base = Date.now() % 8_000_000;
  let counter = 0;
  const ONE = /(?<![\w.])1\.0(?![\w.])/;
  const LIVE = /(ptAccum\[index\]\s*=\s*ptAccum\[index\]\s*\+\s*vec4f\(l\.xyz,\s*)1\.0\)/;
  // Kernels with no standalone 1.0 (the integer-only args kernel) fall back to the comment nonce; they are listed in
  // notes.commentNonce, and their cold numbers may be cache answers.
  const commentNonce = new Set<string>();
  const withNonce = (text: string, name: string): string => {
    const n = base + counter++;
    const comment = `// kernel-compile nonce ${n}\n`;
    if (nonceMode === 'comment') return comment + text;
    if (nonceMode === 'literal') {
      if (!ONE.test(text)) {
        commentNonce.add(name);
        return comment + text;
      }
      return comment + text.replace(new RegExp(ONE.source, 'g'), (1 + n / 2 ** 23).toPrecision(9));
    }
    if (!LIVE.test(text)) throw new Error('kernel-compile: the ptMain accumulate line changed; update LIVE in kernel-compile.ts');
    return comment + text.replace(LIVE, `$1${(1 + n / 2 ** 23).toPrecision(9)})`);
  };
  const entryOf = (text: string): string => {
    if (set === 'megakernel') return ENTRY_POINT;
    const m = /@compute[^\n]*\n?\s*fn\s+(\w+)/.exec(text);
    if (!m) throw new Error('kernel-compile: no @compute entry point in a wavefront kernel');
    return m[1] as string;
  };
  const compile = async (text: string): Promise<number> => {
    const t0 = performance.now();
    const module = device.createShaderModule({ label: 'kernel-compile', code: text });
    await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: entryOf(text) } });
    return performance.now() - t0;
  };

  const texts: Record<string, string> = {};
  const failed: Record<string, string> = {};
  for (const n of names) {
    try {
      texts[n] = (all[n] as () => string)();
    } catch (e) {
      failed[n] = String(e).slice(0, 200);
    }
  }
  const cold: Record<string, number[]> = {};
  const warm: Record<string, number[]> = {};
  // Repeat-outer so slow drift (clock ramp, other apps) spreads over variants instead of landing on one.
  for (let r = 0; r < repeat; r++) {
    for (const n of Object.keys(texts)) {
      if (failed[n]) continue;
      const text = withNonce(texts[n] as string, n);
      try {
        (cold[n] ??= []).push(await compile(text));
        if (warmOn) (warm[n] ??= []).push(await compile(text));
      } catch (e) {
        failed[n] = String(e).slice(0, 300);
      }
    }
  }
  await drainDevice(device);

  const metrics: Record<string, number> = {};
  const table: Record<string, unknown>[] = [];
  for (const n of Object.keys(texts)) {
    const bytes = new TextEncoder().encode(texts[n] as string).length;
    metrics[`${n}_wgsl_bytes`] = bytes;
    const c = cold[n];
    if (!c || failed[n]) continue;
    const cs = stats(c);
    metrics[`${n}_cold_ms_min`] = round(cs.min);
    metrics[`${n}_cold_ms_median`] = round(cs.median);
    metrics[`${n}_cold_ms_max`] = round(cs.max);
    const w = warm[n];
    const ws = w ? stats(w) : null;
    if (ws) metrics[`${n}_warm_ms_median`] = round(ws.median);
    table.push({ variant: n, bytes, nonce: commentNonce.has(n) ? 'comment' : nonceMode, cold_min: round(cs.min), cold_median: round(cs.median), cold_max: round(cs.max), warm_median: ws ? round(ws.median) : null, cold_over_warm: ws ? round(cs.median / ws.median) : null, cold_ms_per_repeat: c.map(round), warm_ms_per_repeat: w?.map(round) ?? null });
  }
  const capsNote = { tier: caps.tier, subgroups: caps.subgroups, shaderF16: caps.shaderF16, timestampQuery: caps.timestampQuery, subgroupSizes: [caps.subgroupMinSize, caps.subgroupMaxSize], features: caps.features };
  reportResult(
    {
      scene: 'kernel-compile',
      renderer: 'ours',
      config: { set, repeat, warm: warmOn, nonce: nonceMode, variants: names },
      valid: Object.keys(failed).length === 0 && table.length > 0,
      metrics,
      notes: { entryPoint: set === 'megakernel' ? ENTRY_POINT : 'per kernel (@compute fn)', layout: 'auto', caps: capsNote, userAgent: navigator.userAgent, adapter: adapterSummary, failed, table, nonce: nonceMode, commentNonce: [...commentNonce], cacheDefeat: nonceMode === 'literal' ? 'literal nonce: every standalone 1.0 replaced by a unique f32 per compile (plus a comment)' : 'code nonce: a unique f32 literal in ptMain per compile (plus a comment); warm = identical text compiled again immediately' },
    },
    { adapter: adapterSummary, deviceErrors },
  );
} catch (error) {
  reportError(error, { adapter: adapterSummary, deviceErrors });
}
