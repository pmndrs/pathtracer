/**
 * Dev server for the A/B viewer:  pnpm exec vite bench/viewer   (or: pnpm --filter @pathtracer/bench viewer)
 * Serves bench/results at /results/, bench/references at /references/ and .cache/viewer-samples at /samples/, and a
 * file index at /__files (JSON) that the start screen lists. Static builds (vite build) just lack the index.
 */
import { createReadStream, existsSync, readdirSync, statSync } from 'node:fs';
import { extname, join, normalize, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig, type Plugin } from 'vite';

const here = fileURLToPath(new URL('.', import.meta.url));
const bench = normalize(join(here, '..'));
const repo = normalize(join(bench, '..'));
const MOUNTS: Record<string, string> = {
  '/results/': join(bench, 'results'),
  '/references/': join(bench, 'references'),
  '/samples/': join(repo, '.cache', 'viewer-samples'),
};
const TYPES: Record<string, string> = { '.json': 'application/json', '.png': 'image/png', '.pfm': 'application/octet-stream' };

function walk(dir: string, urlBase: string, out: Array<{ path: string; size: number; mtime: number }>, depth = 0) {
  if (!existsSync(dir) || depth > 6) return;
  for (const name of readdirSync(dir)) {
    const full = join(dir, name);
    const st = statSync(full);
    if (st.isDirectory()) walk(full, `${urlBase}${name}/`, out, depth + 1);
    else if (/\.pfm$|report\.json$/.test(name)) out.push({ path: `${urlBase}${name}`, size: st.size, mtime: st.mtimeMs });
  }
}

function benchFiles(): Plugin {
  return {
    name: 'bench-files',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const url = decodeURIComponent((req.url ?? '').split('?')[0]!);
        if (url === '/__files') {
          const out: Array<{ path: string; size: number; mtime: number }> = [];
          for (const [prefix, dir] of Object.entries(MOUNTS)) walk(dir, prefix, out);
          res.setHeader('content-type', 'application/json');
          res.end(JSON.stringify(out));
          return;
        }
        for (const [prefix, dir] of Object.entries(MOUNTS)) {
          if (!url.startsWith(prefix)) continue;
          const file = normalize(join(dir, url.slice(prefix.length)));
          if (!file.startsWith(dir + sep) || !existsSync(file) || !statSync(file).isFile()) break;
          res.setHeader('content-type', TYPES[extname(file)] ?? 'application/octet-stream');
          res.setHeader('cache-control', 'no-cache');
          createReadStream(file).pipe(res);
          return;
        }
        next();
      });
    },
  };
}

export default defineConfig({
  root: here,
  plugins: [benchFiles()],
  server: { fs: { allow: [bench] } },
  build: { target: 'es2022', outDir: join(here, 'dist'), emptyOutDir: true },
  worker: { format: 'es' },
});
