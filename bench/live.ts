/**
 * Serves one interactive bench page with the runner's Vite server (linked bvh / Denoiser allowed, the PFM upload
 * endpoint on, so "Save" buttons write to bench/results/incoming/live-<time>/) and opens it in the default browser.
 *
 *   node bench/live.ts nrc-live          (pnpm --filter @pathtracer/bench nrc:live)
 *   node bench/live.ts mlp-live --no-open
 *
 * For the subgroup-matrix backends open the URL in a Chrome started with --enable-unsafe-webgpu.
 */
import { execFile } from 'node:child_process';
import { startBenchServer } from './runner/server.ts';

const page = process.argv[2];
if (!page || page.startsWith('-')) {
  console.error('Usage: node bench/live.ts <page> [--no-open]   (a page in bench/pages/, e.g. nrc-live)');
  process.exit(2);
}
const runId = `live-${new Date().toISOString().replace(/[-:]/g, '').replace(/\..*/, 'Z')}`;
const server = await startBenchServer(runId);
server.setSlot(page);
const url = `${server.origin}/bench/pages/${page}.html`;
console.log(`${page}: ${url}`);
console.log(`Saved images go to bench/results/incoming/${runId}/${page}/. Ctrl+C to stop.`);
if (!process.argv.includes('--no-open')) {
  const opener = process.platform === 'darwin' ? 'open' : process.platform === 'win32' ? 'explorer' : 'xdg-open';
  execFile(opener, [url], () => {});
}
process.on('SIGINT', () => void server.close().then(() => process.exit(0)));
