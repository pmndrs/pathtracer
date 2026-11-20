import { copyFile, mkdir, readFile, rename } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'vite';

const root = fileURLToPath(new URL('../../', import.meta.url));
const outDir = join(root, 'dist/pages');
const base = (process.env.PAGES_BASE ?? '/pathtracer').replace(/\/$/, '') + '/';

await build({
  root: join(root, 'bench/pages'),
  configFile: false,
  publicDir: false,
  base,
  define: {
    __BENCH_SITE_BASE__: JSON.stringify(base),
    __BENCH_DENOISER_TZAS__: JSON.stringify(base + 'weights'),
  },
  build: {
    outDir, emptyOutDir: true, target: 'esnext',
    rollupOptions: { input: join(root, 'bench/pages/nrc-pipeline.html') },
  },
});
await rename(join(outDir, 'nrc-pipeline.html'), join(outDir, 'index.html'));

async function copyAssets(source, target, names) {
  const destination = join(outDir, target);
  await mkdir(destination, { recursive: true });
  for (const name of names) await copyFile(join(source, name), join(destination, name));
}

const scenes = 'bench/assets/generated/web';
const scene = 'hero-classroom';
const meta = JSON.parse(await readFile(join(root, scenes, scene + '.json'), 'utf8'));
await copyAssets(join(root, scenes), scenes, [
  scene + '.scene.json', scene + '.json', scene + '.glb', ...meta.textures.map(texture => texture.file),
]);
const noise = 'packages/raytracer/assets/bluenoise';
await copyAssets(join(root, noise), noise, ['manifest.json', 'gf-vec2-64.png']);

const weights = resolve(root, 'bench/node_modules/denoiser/tzas');
await copyAssets(weights, 'weights', ['LICENSE.txt', 'rt_ldr.tza', 'rt_ldr_small.tza', 'rt_hdr_alb.tza', 'rt_hdr_alb_small.tza',
  'rt_hdr_calb_cnrm.tza', 'rt_hdr_calb_cnrm_small.tza']);
