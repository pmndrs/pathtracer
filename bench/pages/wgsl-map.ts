/**
 * wgsl-map (P2.10b): the megakernel's WGSL source map on a real compiler. Correctness only, no timing.
 *  1. `PtMegakernel.create` compiles the pruned, shortened, minified kernel (positive control).
 *  2. A type error is injected into the compiled (minified) text at the first line of `ptSampleLd`'s body, calling a shortened
 *     function (`ptSample2D`) with one argument (the compiler names it in the message). `rtCreateComputePipelineAsync` must throw a message with the ORIGINAL function name and the ORIGINAL line number,
 *     and print the original source line. The same is read through `rtCheckShaderModule` and `rtMapCompilationInfo`.
 *
 * Step 2 and 3 only run with `inject=1`: Dawn logs every compile error to the browser console and the runner counts that as a failure,
 * so an `inject=1` run exits non-zero BY DESIGN; read `valid`, `metrics.problems` and `notes.thrown` in its run.json. Without it
 * (default) the page is a clean pass: the control kernel and the `sourceMap: true` kernel compile.
 *  4. Also checks `sourceMap: true` (Source Map v3 in the descriptor) does not break compilation.
 */
import { detectCaps, requestDeviceForCaps, rtCheckShaderModule, rtCreateComputePipelineAsync, rtMapCompilationInfo } from '@pmndrs/raytracer';
import { PtMegakernel, ptMegakernelBuild } from '@pmndrs/pathtracer';
import type { AdapterSummary } from '../runner/contract.ts';
import { drainDevice, getArgs, numberArg, reportError, reportResult, summarizeAdapter, watchDevice } from '../runner/page-client.ts';

let adapterSummary: AdapterSummary | null = null;
let deviceErrors: string[] = [];
const notes: Record<string, unknown> = {};
const problems: string[] = [];
const expectThat = (ok: boolean, what: string) => {
  if (!ok) problems.push(what);
};

try {
  if (!navigator.gpu) throw new Error('navigator.gpu is undefined: no WebGPU in this browser.');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('requestAdapter() returned null: no usable GPU.');
  adapterSummary = summarizeAdapter(adapter);
  const { device, caps } = await requestDeviceForCaps(adapter, detectCaps(adapter), { label: 'wgsl-map' });
  deviceErrors = watchDevice(device);
  const options = { format: {}, layout: { normals: true, uvs: false, positions: true }, integrator: {} };

  const inject = numberArg(getArgs(), 'inject', 0) !== 0;
  // 1. The real kernel compiles; and with a Source Map v3 attached.
  await PtMegakernel.create(device, caps, { ...options, label: 'wgsl-map control' });
  await PtMegakernel.create(device, caps, { ...options, sourceMap: true, label: 'wgsl-map sourceMap' });

  const { code, map } = ptMegakernelBuild(caps, options);
  const sources = map.sources as string[];
  const out = code.split('\n');
  if (inject) {
  // 2. Inject an error into the compiled text.
  const fnLine = sources.findIndex((l) => l.startsWith('fn ptSampleLd'));
  expectThat(fnLine >= 0, 'fn ptSampleLd not in the source');
  const bodyLine = fnLine + 1; // first line of the body, 0-based source line
  const at = out.findIndex((_, i) => map.lines[i] === bodyLine);
  expectThat(at >= 0, 'body line was pruned');
  const short = Object.entries(map.names).find(([, original]) => original === 'ptSample2D')?.[0];
  expectThat(!!short, 'ptSample2D was not shortened');
  const broken = [...out.slice(0, at), `let ptInjected=${short}(1u);${out[at]}`, ...out.slice(at + 1)].join('\n');
  notes.compiledLine = at + 1;
  notes.sourceLine = bodyLine + 1;
  notes.shortName = short;

  // The compile errors below are expected (the two broken modules report 2 uncaptured errors). They are checked, then removed from
  // the device-error list, which fails a run. (Error scopes hang under the headless runner, so none is used.)
  const errorsBefore = deviceErrors.length;
  let thrown = '';
  try {
    await rtCreateComputePipelineAsync(device, { label: 'wgsl-map injected', layout: 'auto', code: broken, map, entryPoint: 'ptMain' });
  } catch (error) {
    thrown = String((error as Error).message);
  }
  notes.thrown = thrown;
  expectThat(thrown.includes('failed to compile'), 'no translated error thrown');
  expectThat(thrown.includes('ptSample2D'), 'original function name missing from the message');
  expectThat(thrown.includes(`at source line ${bodyLine + 1}`), `source line ${bodyLine + 1} missing from the message`);
  expectThat(thrown.includes(`compiled line ${at + 1}`), 'compiled line missing');
  expectThat(thrown.includes(sources[bodyLine] as string), 'original source line text missing');
  expectThat(short !== undefined && !new RegExp(`(?<![\\w.])${short}(?!\\w)`).test(thrown), 'the short name leaked into the message');

  // 3. The same through getCompilationInfo directly.
  const module = device.createShaderModule({ label: 'wgsl-map info', code: broken });
  const mapped = rtMapCompilationInfo(await module.getCompilationInfo(), map);
  notes.mapped = mapped.map((m) => ({ type: m.type, line: m.lineNum, compiled: m.outputLineNum, message: m.message }));
  expectThat(mapped.some((m) => m.type === 'error' && m.lineNum === bodyLine + 1), 'mapped info has no error on the source line');
  let checked = '';
  try {
    await rtCheckShaderModule(module, map, 'wgsl-map info');
  } catch (error) {
    checked = String((error as Error).message);
  }
  expectThat(mapped.some((m) => m.type === 'error' && m.message.includes('ptSample2D')), `the message text does not name ptSample2D: ${mapped.map((m) => m.message).join(' | ')}`);
  expectThat(checked.includes('ptSample2D'), 'rtCheckShaderModule message lacks the original name');

  await drainDevice(device);
  const expected = deviceErrors.splice(errorsBefore);
  expectThat(expected.length === 2 && expected.every((e) => e.includes('too few arguments')), `expected 2 compiler errors from the device, got ${expected.length}`);
  notes.expectedDeviceErrors = expected.length;
  }
  reportResult(
    { scene: 'wgsl-map', renderer: 'ours', config: {}, valid: problems.length === 0, metrics: { problems: problems.length, short_names: Object.keys(map.names).length, output_lines: out.length, source_lines: sources.length }, notes: { problems, ...notes } },
    { adapter: adapterSummary, deviceErrors },
  );
  if (problems.length > 0) throw new Error(`wgsl-map: ${problems.join('; ')}`);
} catch (error) {
  reportError(error, { adapter: adapterSummary, deviceErrors });
}
