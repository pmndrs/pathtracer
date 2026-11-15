/**
 * Runs one bench page in dawn.node (the npm `webgpu` package: Dawn + Tint as a Node addon) instead of Chrome.
 *
 * Why (track N, M6): official Chrome on Windows runs Dawn on D3D12 only, and Dawn's D3D12 backend has no subgroup
 * matrices ("D3D: no support", Dawn docs/dawn/features/subgroup_matrix.md). Dawn's Vulkan backend has them on NVIDIA
 * (VK_KHR_cooperative_matrix), but Chrome on Windows never offers a Vulkan hardware adapter. dawn.node can pick the
 * backend, so the same WGSL goes through the same Dawn and Tint on Vulkan. This is an out-of-browser arm: it is not
 * what a Windows Chrome user gets (nrc-diverges.md).
 *
 * Spawned by DawnSession (dawn-session.ts), one process per page run, so every run starts from a fresh device and a
 * fresh module graph (pages run their work at import time). Protocol on stdout, one JSON object per line, prefixed:
 *   @@dawn-host {"kind":"preflight","adapter":{...}}       (with --preflight: then exits)
 *   @@dawn-host {"kind":"message","level":"error","text":"..."}
 *   @@dawn-host {"kind":"report","report":{...}}
 *
 * The page sees: navigator.gpu = dawn.node's GPU, `location.search` = the query the runner built, a `window` that
 * collects the report, a do-nothing `document`, and `fetch` that resolves relative URLs against the runner's server
 * (image uploads still land in bench/results/incoming/<run>/).
 *
 * Usage (normally by DawnSession): node bench/runner/dawn-host.ts --backend vulkan --toggles allow_unsafe_apis
 *   --origin http://127.0.0.1:5291 (--preflight | --page mlp-check --query 'token=..&infer=sgm')
 * dawn.node: DAWN_NODE=<path to webgpu/index.js>, else <repo>/.cache/dawn-node/node_modules/webgpu/index.js.
 */
import { existsSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { pathToFileURL } from 'node:url';

export const DAWN_HOST_PREFIX = '@@dawn-host ';

export function defaultDawnNodePath(repoRoot: string, env: NodeJS.ProcessEnv = process.env): string {
  return env.DAWN_NODE ? resolve(env.DAWN_NODE) : join(repoRoot, '.cache', 'dawn-node', 'node_modules', 'webgpu', 'index.js');
}

/** dawn.node `create()` flags for a backend and a toggle list. */
export function dawnCreateFlags(backend: string, toggles: readonly string[]): string[] {
  return [`backend=${backend}`, ...(toggles.length ? [`enable-dawn-features=${toggles.join(',')}`] : [])];
}

/** One record per line. Dawn's native code prints compilation logs to stdout without a final newline, so start a fresh line. */
function emit(object: unknown): void {
  process.stdout.write(`\n${DAWN_HOST_PREFIX}${JSON.stringify(object)}\n`);
}

/** A do-nothing DOM element: pages may build a summary view; nobody sees it here. */
function fakeElement(): Record<string, unknown> {
  const element: Record<string, unknown> = {
    style: {},
    textContent: '',
    innerHTML: '',
    append: () => {},
    appendChild: () => {},
    addEventListener: () => {},
    setAttribute: () => {},
    getContext: () => null,
  };
  return element;
}

async function main(): Promise<void> {
  const { values } = parseArgs({
    options: {
      backend: { type: 'string' },
      toggles: { type: 'string' },
      origin: { type: 'string' },
      page: { type: 'string' },
      query: { type: 'string' },
      preflight: { type: 'boolean' },
    },
    strict: true,
  });
  const repoRoot = resolve(import.meta.dirname, '..', '..');
  const dawnPath = defaultDawnNodePath(repoRoot);
  if (!existsSync(dawnPath)) {
    throw new Error(`dawn.node not found at ${dawnPath}. Install it: mkdir .cache/dawn-node, npm install webgpu there (or set DAWN_NODE).`);
  }
  const toggles = (values.toggles ?? '').split(',').filter((t) => t !== '');
  const dawn = (await import(pathToFileURL(dawnPath).href)) as { create: (flags: string[]) => GPU; globals: Record<string, unknown> };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create(dawnCreateFlags(values.backend ?? 'vulkan', toggles));

  // Console output becomes runner messages (the runner judges errors the same way as Chrome's).
  for (const level of ['log', 'info', 'warn', 'error', 'debug'] as const) {
    console[level] = (...parts: unknown[]) => {
      const text = parts.map((p) => (p instanceof Error ? (p.stack ?? p.message) : typeof p === 'string' ? p : JSON.stringify(p))).join(' ');
      emit({ kind: 'message', level: level === 'warn' ? 'warning' : level, text });
    };
  }

  if (values.preflight) {
    const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) {
      emit({ kind: 'preflight', problem: 'no-adapter' });
    } else {
      const { vendor, architecture, device, description } = adapter.info;
      emit({ kind: 'preflight', adapter: { vendor, architecture, device, description, isFallbackAdapter: false, features: [...adapter.features].sort() } });
    }
    process.exit(0);
  }

  const origin = values.origin ?? 'http://127.0.0.1';
  const g = globalThis as unknown as Record<string, unknown>;
  const listeners: Record<string, Array<(event: unknown) => void>> = {};
  let reported = false;
  const windowObject = new Proxy(g, {
    set(target, key, value) {
      target[key as string] = value;
      if (key === '__benchReport' && !reported) {
        reported = true;
        emit({ kind: 'report', report: value });
        // Let the page's last awaits settle (its drain window), then end: dawn.node keeps the loop alive otherwise.
        setTimeout(() => process.exit(0), 200);
      }
      return true;
    },
  });
  g.window = windowObject;
  g.self = windowObject;
  g.addEventListener = (type: string, fn: (event: unknown) => void) => ((listeners[type] ??= []).push(fn));
  g.location = new URL(`${origin}/bench/pages/${values.page}.html?${values.query ?? ''}`);
  Object.defineProperty(globalThis, 'navigator', { value: { gpu, userAgent: `dawn.node (${values.backend})`, hardwareConcurrency: 1 }, configurable: true });
  g.document = { title: values.page, body: fakeElement(), createElement: fakeElement, getElementById: () => fakeElement(), querySelector: () => fakeElement() };
  g.requestAnimationFrame = (fn: (t: number) => void) => setTimeout(() => fn(performance.now()), 16);
  const realFetch = globalThis.fetch;
  g.fetch = (input: string | URL | Request, init?: RequestInit) =>
    realFetch(typeof input === 'string' && input.startsWith('/') ? `${origin}${input}` : input, init);

  process.on('unhandledRejection', (reason) => {
    for (const fn of listeners.unhandledrejection ?? []) fn({ reason });
    emit({ kind: 'message', level: 'error', text: `Unhandled rejection: ${reason instanceof Error ? (reason.stack ?? reason.message) : String(reason)}` });
  });
  process.on('uncaughtException', (error) => {
    for (const fn of listeners.error ?? []) fn({ message: error.message, error });
    emit({ kind: 'message', level: 'error', text: `Uncaught error: ${error.stack ?? error.message}` });
  });

  // dawn.node does not hold the event loop open while GPU promises (mapAsync, onSubmittedWorkDone, compilation) are
  // pending: without this the process ends quietly mid-page with exit code 0. The report handler exits.
  setInterval(() => {}, 1000);
  await import(pathToFileURL(join(repoRoot, 'bench', 'pages', `${values.page}.ts`)).href);
}

// `import.meta.main` needs Node 24; this works on 22 too.
if (process.argv[1] && resolve(process.argv[1]) === resolve(import.meta.filename)) {
  main().catch((error: unknown) => {
    emit({ kind: 'message', level: 'error', text: `dawn-host: ${error instanceof Error ? (error.stack ?? error.message) : String(error)}` });
    process.exit(1);
  });
}
