#!/usr/bin/env node
/**
 * Opens every example of examples/catalog.ts in headless Chromium and checks it: no error overlay, no console error, the image (or frame)
 * reaches its sample count with no NaN / Inf pixel, and the canvas is not one flat colour. With `--thumbs` it also writes each example's
 * gallery card image (`examples/<path>/thumb.webp`, 640 x 360, from the page's own canvas output; needs python3 with Pillow).
 *
 *   node scripts/examples-smoke.mjs                       # starts the examples dev server, SwiftShader (no GPU needed)
 *   node scripts/examples-smoke.mjs --only hello,ao       # some examples
 *   node scripts/examples-smoke.mjs --thumbs --gpu        # card images on the real GPU (a Mac or a desktop with a display)
 *   node scripts/examples-smoke.mjs --url http://localhost:5310/   # an already running server
 *
 * Pages open on their DEFAULT scene (a check on the Cornell box alone missed a bug that only textured models hit), so most stream a Khronos model.
 * Options: --url, --only, --skip, --thumbs, --gpu (no SwiftShader flags), --no-network (skip examples that stream models), --timeout <s>
 * per example (default 240), --port (default 5320). Playwright: `playwright` from node_modules, else PLAYWRIGHT_MODULE (a path to the package).
 * Checks pass on SwiftShader; it is slow (minutes for a path tracer at a few samples) and never a timing.
 */
import { spawn, spawnSync } from 'node:child_process';
import { createRequire } from 'node:module';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { EXAMPLES } from '../examples/catalog.ts';

const root = fileURLToPath(new URL('..', import.meta.url));
const args = process.argv.slice(2);
const flag = (name) => args.includes(name);
const value = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 && args[i + 1] ? args[i + 1] : fallback;
};
const list = (name) => (value(name, '') || '').split(',').filter(Boolean);
const only = list('--only');
const skip = list('--skip');
const thumbs = flag('--thumbs');
const timeoutMs = Number(value('--timeout', '240')) * 1000;

async function playwright() {
  try {
    return await import('playwright');
  } catch {
    const path = process.env.PLAYWRIGHT_MODULE ?? '/opt/node-tools/node_modules/playwright';
    return createRequire(import.meta.url)(path);
  }
}

/** Starts the examples' Vite dev server and resolves with its base URL. */
function startServer(port) {
  return new Promise((resolve, reject) => {
    // Its own process group: pnpm wraps vite, and killing only the wrapper leaves the server running.
    const child = spawn('pnpm', ['--filter', '@pathtracer/examples', 'exec', 'vite', '--port', String(port), '--strictPort'], { cwd: root, stdio: ['ignore', 'pipe', 'pipe'], detached: true });
    const stop = () => { try { process.kill(-child.pid, 'SIGTERM'); } catch { /* already gone */ } };
    let log = '';
    const timer = setTimeout(() => reject(new Error(`vite did not start:\n${log}`)), 60000);
    const onData = (d) => {
      log += d;
      // Vite colours its banner on some terminals (CI): match the text without the escape codes.
      if (/Local:/.test(log.replace(/\x1b\[[0-9;]*m/g, ''))) {
        clearTimeout(timer);
        resolve({ url: `http://localhost:${port}/`, stop });
      }
    };
    child.stdout.on('data', onData);
    child.stderr.on('data', onData);
    child.on('exit', (code) => reject(new Error(`vite exited (${code}):\n${log}`)));
  });
}

/** In the page: the state of the example (error overlay, progress, samples, image statistics). */
async function probe(page) {
  return page.evaluate(() => {
    const error = document.querySelector('.ex-overlay.error:not([hidden])');
    if (error) return { error: error.textContent?.slice(0, 1500) ?? 'error' };
    const progress = document.querySelector('.ex-overlay.progress:not([hidden])');
    const g = globalThis;
    const samples = g.__progressive?.samples ?? g.__realtime?.frames ?? (g.__caps ? 1e9 : 0);
    return { progress: progress ? progress.textContent : null, samples };
  });
}

/** In the page: mean and non-finite count of the example's own linear image (path tracer, accumulation or real-time view). */
async function readLinear(page) {
  return page.evaluate(async () => {
    const g = globalThis;
    let data = null;
    if (g.__progressive) data = (await g.__progressive.readImage()).data;
    else if (g.__realtime) data = await g.__realtime.read();
    if (!data) return null;
    let sum = 0, bad = 0, n = 0;
    for (let i = 0; i < data.length; i += 4) {
      const v = data[i] + data[i + 1] + data[i + 2];
      if (Number.isFinite(v)) { sum += v / 3; n++; } else bad++;
    }
    return { mean: n ? sum / n : 0, nonFinite: bad, pixels: data.length / 4 };
  });
}

/** In the page: the RGBA8 bytes the example last drew (the offscreen target of `?offscreen=1`), as base64, plus its statistics. */
async function capture(page) {
  return page.evaluate(async () => {
    const t = globalThis.__exampleTarget;
    if (!t) return null;
    const { device, texture } = t;
    const w = texture.width, h = texture.height, bpr = Math.ceil((w * 4) / 256) * 256;
    const buffer = device.createBuffer({ size: bpr * h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow: bpr }, [w, h]);
    device.queue.submit([encoder.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);
    const src = new Uint8Array(buffer.getMappedRange());
    const out = new Uint8Array(w * h * 4);
    for (let y = 0; y < h; y++) out.set(src.subarray(y * bpr, y * bpr + w * 4), y * w * 4);
    buffer.unmap();
    buffer.destroy();
    const bgra = texture.format.startsWith('bgra');
    let sum = 0, sq = 0;
    for (let i = 0; i < w * h; i++) {
      const l = (out[i * 4] + out[i * 4 + 1] + out[i * 4 + 2]) / 3;
      sum += l;
      sq += l * l;
    }
    const mean = sum / (w * h);
    let bin = '';
    for (let i = 0; i < out.length; i += 0x8000) bin += String.fromCharCode(...out.subarray(i, i + 0x8000));
    return { width: w, height: h, bgra, mean, std: Math.sqrt(Math.max(0, sq / (w * h) - mean * mean)), base64: btoa(bin) };
  });
}

function writeThumb(entry, shot) {
  const dir = mkdtempSync(join(tmpdir(), 'thumb-'));
  try {
    const raw = join(dir, 'frame.rgba');
    writeFileSync(raw, Buffer.from(shot.base64, 'base64'));
    const out = join(root, 'examples', entry.path, 'thumb.webp');
    const py = `
import sys
from PIL import Image
w, h, bgra, src, dst = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3] == '1', sys.argv[4], sys.argv[5]
img = Image.frombytes('RGBA', (w, h), open(src, 'rb').read())
if bgra:
    b, g, r, a = img.split(); img = Image.merge('RGBA', (r, g, b, a))
img = img.convert('RGB')
tw = min(w, h * 16 // 9); th = tw * 9 // 16
left, top = (w - tw) // 2, (h - th) // 2
img = img.crop((left, top, left + tw, top + th)).resize((640, 360), Image.LANCZOS)
img.save(dst, 'WEBP', quality=82, method=6)
`;
    const r = spawnSync('python3', ['-I', '-c', py, String(shot.width), String(shot.height), shot.bgra ? '1' : '0', raw, out], { encoding: 'utf8' });
    if (r.status !== 0) throw new Error(`thumbnail failed: ${r.stderr}`);
    return out;
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

const entries = EXAMPLES.filter((e) => !e.external && (!only.length || only.includes(e.slug)) && !skip.includes(e.slug) && !(flag('--no-network') && e.network && !e.check?.query));
const server = value('--url', '') ? { url: value('--url', ''), stop: () => {} } : await startServer(Number(value('--port', '5320')));
const { chromium } = await playwright();
const gpuArgs = flag('--gpu') ? ['--enable-unsafe-webgpu', ...(process.env.CHROME_PATH ? ['--headless=new', '--use-angle=metal'] : [])] : ['--no-sandbox', '--enable-unsafe-webgpu', '--use-webgpu-adapter=swiftshader', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'];
const browser = await chromium.launch({ headless: true, args: gpuArgs, ...(process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : {}) });
const results = [];
try {
  for (const entry of entries) {
    const check = entry.check ?? {};
    const query = (thumbs ? (check.thumbQuery ?? check.query) : check.query) ?? '';
    const target = thumbs ? (check.thumbSamples ?? Math.max(64, check.samples ?? 0)) : (check.samples ?? 4);
    const url = `${server.url}${entry.path}?offscreen=1${query ? `&${query.replace(/^[?&]/, '')}` : ''}`;
    const page = await browser.newPage({ viewport: thumbs ? { width: 960, height: 540 } : { width: 480, height: 270 } });
    const consoleErrors = [];
    page.on('console', (m) => m.type() === 'error' && consoleErrors.push(m.text().slice(0, 400)));
    page.on('pageerror', (e) => consoleErrors.push(`page error: ${e.message}`));
    const started = Date.now();
    const result = { slug: entry.slug, ok: false, notes: [] };
    try {
      // A page waits for its module graph (the dev server compiles on first request) and the GPU process may still be busy with the last page.
      await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 180000 });
      let state = null;
      while (Date.now() - started < timeoutMs) {
        state = await probe(page);
        if (state.error || (state.samples >= target && !state.progress)) break;
        await page.waitForTimeout(500);
      }
      if (state?.error) throw new Error(state.error);
      if (!(state && state.samples >= target)) throw new Error(`timed out at ${state?.samples ?? 0} of ${target} samples (${state?.progress ?? 'no progress message'})`);
      const linear = await readLinear(page);
      if (linear) {
        result.notes.push(`mean ${linear.mean.toFixed(4)}`);
        if (linear.nonFinite) throw new Error(`${linear.nonFinite} non-finite pixels`);
      }
      if (check.image !== false) {
        const shot = await capture(page);
        if (!shot) throw new Error('no offscreen target (does the page draw through ctx.target()?)');
        result.notes.push(`canvas mean ${shot.mean.toFixed(1)}, std ${shot.std.toFixed(1)}`);
        if (shot.std < 1) throw new Error(`the canvas is one flat colour (mean ${shot.mean.toFixed(1)})`);
        if (thumbs) result.notes.push(`wrote ${writeThumb(entry, shot).replace(root, '')}`);
      }
      if (consoleErrors.length) throw new Error(`console errors:\n${consoleErrors.join('\n')}`);
      result.ok = true;
    } catch (error) {
      result.error = error instanceof Error ? error.message : String(error);
    } finally {
      result.seconds = Math.round((Date.now() - started) / 1000);
      await page.close().catch(() => {});
    }
    results.push(result);
    console.log(`${result.ok ? 'PASS' : 'FAIL'} ${entry.slug.padEnd(18)} ${String(result.seconds).padStart(4)} s  ${result.notes.join('; ')}${result.error ? `\n     ${result.error.replace(/\n/g, '\n     ')}` : ''}`);
  }
} finally {
  await browser.close();
  server.stop();
}
const failed = results.filter((r) => !r.ok);
console.log(`\n${results.length - failed.length} of ${results.length} examples passed${failed.length ? `; failed: ${failed.map((r) => r.slug).join(', ')}` : ''}`);
process.exit(failed.length ? 1 : 0);
