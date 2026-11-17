/**
 * Presses the nrc-live demo's Benchmark button for each resolution and network backend in a WINDOWED Chrome (the
 * maintainer's setting for interactive timing), driven over DevTools, and records the per-stage GPU frame times plus the
 * time the page spends compiling the cache kernels on first load (WN5).
 *
 *   node bench/live-bench.ts [--gpu nvidia] [--res 960x540,1280x720] [--backends tiled-f16,portable-f16] [--steps 4]
 *                            [--repeat 2] [--cool 60] [--json out.json]
 *
 * Each (resolution, backend) cell rebuilds the cache, runs 60 frames, then Benchmark (23 frames, median of the last 20, one
 * readback at the end, as the button does) after the card has cooled below --cool °C with the demo's loop paused; `--repeat` runs the whole grid again, in reverse order every other time.
 * Wall-clock per submit, best effort: quote with the start temperature and the load the script prints.
 */
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { CdpClient } from './runner/cdp.ts';
import { discoverGpuLuid, findBrowser, removeProfile, startChrome } from './runner/chrome.ts';
import type { GpuChoice } from './runner/contract.ts';
import { startBenchServer } from './runner/server.ts';
import { execFileSync } from 'node:child_process';

const { values } = parseArgs({
  options: {
    gpu: { type: 'string', default: 'nvidia' },
    res: { type: 'string', default: '960x540,1280x720' },
    backends: { type: 'string', default: 'tiled-f16,portable-f16' },
    steps: { type: 'string', default: '4' },
    repeat: { type: 'string', default: '1' },
    scene: { type: 'string', default: 'sponza-white' },
    cool: { type: 'string', default: '60' },
    json: { type: 'string' },
  },
  strict: true,
});
const resolutions = values.res!.split(',');
const backends = values.backends!.split(',');
const repeat = Number(values.repeat);
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

function gpuState(): string {
  try {
    return execFileSync('nvidia-smi', ['--query-gpu=temperature.gpu,clocks.sm,power.draw', '--format=csv,noheader'], { encoding: 'utf8' }).trim();
  } catch {
    return 'n/a';
  }
}

/** Waits until the card is below --cool degrees C (nvidia-smi), at most 10 min. */
async function coolDown(): Promise<void> {
  const limit = Number(values.cool);
  const t0 = Date.now();
  for (;;) {
    const t = Number(gpuState().split(',')[0]);
    if (!Number.isFinite(t) || t < limit || Date.now() - t0 > 600_000) return;
    await sleep(3000);
  }
}

const server = await startBenchServer(`live-bench-${new Date().toISOString().replace(/[-:]/g, '').replace(/\..*/, 'Z')}`);
server.setSlot('nrc-live');
const luid = process.platform === 'win32' ? await discoverGpuLuid(values.gpu as GpuChoice) : null;
const profileDir = mkdtempSync(join(tmpdir(), 'pt-live-bench-'));
const chrome = startChrome(findBrowser(null), { profileDir, luid, withoutGpu: false, headed: true });
let cdp: CdpClient | null = null;
const rows: Record<string, unknown>[] = [];
try {
  await chrome.ready();
  const targets = (await (await fetch(`${chrome.devtoolsOrigin}/json/list`)).json()) as Array<{ type: string; webSocketDebuggerUrl: string }>;
  cdp = await CdpClient.connect(targets.find((t) => t.type === 'page')!.webSocketDebuggerUrl);
  await cdp.send('Page.enable');
  await cdp.send('Runtime.enable');
  const page = cdp;
  const status = () => page.evaluate<string>(`document.getElementById('status')?.textContent ?? 'loading'`);
  /** Waits until the page is idle (status empty) and returns how long it said "Compiling the cache kernels". */
  const settle = async (): Promise<{ compileMs: number; totalMs: number }> => {
    const t0 = Date.now();
    let compileStart = 0;
    let compileMs = 0;
    for (;;) {
      const s = await status().catch(() => 'loading');
      if (s.startsWith('Compiling') || s.startsWith('Rebuilding')) compileStart ||= Date.now();
      else if (compileStart && !compileMs) compileMs = Date.now() - compileStart;
      const ready = await page.evaluate<boolean>(`!!window.__nrcLive?.samples`).catch(() => false);
      // Idle = no build message (the status then shows the fps line or nothing); give a just-dispatched change 300 ms to start.
      if (ready && !/^(Loading|Compiling|Rebuilding)/.test(s) && Date.now() - t0 > 300) return { compileMs, totalMs: Date.now() - t0 };
      if (/error/i.test(s)) throw new Error(`page: ${s}`);
      if (Date.now() - t0 > 300_000) throw new Error(`page did not settle: "${s}"`);
      await sleep(50);
    }
  };
  const t0 = Date.now();
  await page.send('Page.navigate', { url: `${server.origin}/bench/pages/nrc-live.html?scene=${values.scene}` });
  const first = await settle();
  console.log(`first load ${(Date.now() - t0) / 1000}s (cache kernels compiling ${first.compileMs} ms)`);
  rows.push({ kind: 'first-load', res: '960x540', backend: await page.evaluate<string>(`document.getElementById('backend').value`), compileMs: first.compileMs, totalMs: first.totalMs });
  const offered = await page.evaluate<string[]>(`[...document.getElementById('backend').options].map((o) => o.value)`);
  console.log(`backends offered: ${offered.join(', ')}`);
  for (let r = 0; r < repeat; r++) {
    const cells = resolutions.flatMap((res) => backends.map((backend) => ({ res, backend })));
    if (r % 2) cells.reverse();
    for (const { res, backend } of cells) {
      if (!offered.includes(backend)) {
        console.log(`skip ${backend}: not offered`);
        continue;
      }
      await page.evaluate(`(() => {
        const set = (id, v) => { const el = document.getElementById(id); if (el.value !== v) { el.value = v; el.dispatchEvent(new Event('change')); return true; } return false; };
        document.getElementById('steps').value = '${values.steps}';
        document.getElementById('steps').dispatchEvent(new Event('change'));
        document.getElementById('view').value = 'split';
        const full = set('res', '${res}');
        if (!full) set('backend', '${backend}');
        return full;
      })()`);
      let built = await settle();
      // A resolution change rebuilds everything with the remembered backend; set the backend after it.
      const now = await page.evaluate<string>(`document.getElementById('backend').value`);
      if (now !== backend) {
        await page.evaluate(`(() => { const el = document.getElementById('backend'); el.value = '${backend}'; el.dispatchEvent(new Event('change')); })()`);
        built = await settle();
      }
      // The demo's own frame loop would heat the card between cells: pause it, cool down, then warm up and time.
      await page.evaluate(`window.__nrcLive.pause(true)`);
      await coolDown();
      await page.evaluate(`window.__nrcLive.step(60)`);
      const before = gpuState();
      await page.evaluate(`window.__nrcLive.benchmark()`);
      const bench = await page.evaluate<Record<string, number | string>>(`window.__nrcBench`);
      const row = { kind: 'benchmark', round: r, res, backend, steps: Number(values.steps), rebuildMs: built.compileMs, gpuBefore: before, ...bench };
      rows.push(row);
      const f = (v: unknown) => (typeof v === 'number' ? v.toFixed(2) : String(v));
      console.log(
        `${res} ${backend.padEnd(13)} paths ${f(bench.trainerMs)}  train ${f(bench.trainMs)}  render ${f(bench.renderMs)}  infer ${f(bench.inferMs)}  = cache ${f(bench.nrcMs)} ms | plain/spp ${f(bench.perSpp)} ms | equal-time spp ${bench.equal}  [${before}]`,
      );
    }
  }
} finally {
  cdp?.close();
  await chrome.stop();
  await removeProfile(profileDir);
  await server.close();
}
if (values.json) writeFileSync(values.json, JSON.stringify({ when: new Date().toISOString(), gpu: values.gpu, rows }, null, 1));
