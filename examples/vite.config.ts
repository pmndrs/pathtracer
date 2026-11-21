import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig, type Plugin } from 'vite';
import { EXAMPLES } from './catalog.ts';

// The examples gallery. Serves this folder; @pmndrs/raytracer and @pmndrs/pathtracer come from the workspace as TypeScript
// source, so library edits reload here. The site build (scripts/build-site.mjs) sets EXAMPLES_BASE to the deployed path
// (`/pathtracer/examples/` on GitHub Pages); local dev and builds use '/'.
const root = fileURLToPath(new URL('.', import.meta.url));
const local = EXAMPLES.filter((e) => !e.external);

/**
 * Every example page gets the frame (nav pill with the pmndrs logo, breadcrumb, Docs / Source / Fullscreen) and its catalog slug, injected
 * here so an example cannot forget them and stays a plain page. The gallery itself does not.
 */
function exampleFrame(): Plugin {
  return {
    name: 'example-frame',
    transformIndexHtml: {
      order: 'pre',
      handler(_html, ctx) {
        const path = ctx.path.replace(/^\//, '').replace(/index\.html$/, '');
        const entry = local.find((e) => e.path === path);
        if (!entry) return;
        return [
          { tag: 'meta', attrs: { name: 'example', content: entry.slug }, injectTo: 'head' },
          { tag: 'link', attrs: { rel: 'icon', type: 'image/svg+xml', href: '/shared/pmndrs-logo.svg' }, injectTo: 'head' },
          { tag: 'script', attrs: { type: 'module', src: '/shared/nav.ts' }, injectTo: 'body' },
        ];
      },
    },
  };
}

export default defineConfig({
  root,
  base: process.env.EXAMPLES_BASE ?? '/',
  plugins: [exampleFrame()],
  // bvh's build workers: excluded from pre-bundling or Vite rewrites their URL (CLAUDE.md, Traps).
  optimizeDeps: { exclude: ['@pmndrs/bvh'] },
  cacheDir: resolve(root, '../node_modules/.vite-examples'),
  server: { port: 5310, fs: { allow: [resolve(root, '..')] } },
  build: {
    target: 'esnext',
    outDir: process.env.EXAMPLES_OUT ?? resolve(root, 'dist'),
    emptyOutDir: true,
    // The libraries are one shared chunk (about 0.9 MB, mostly WGSL text and bvh's builder).
    chunkSizeWarningLimit: 1200,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('/packages/raytracer/') || id.includes('/packages/pathtracer/') || id.includes('@pmndrs/')) return 'pmndrs';
          if (id.includes('tweakpane')) return 'tweakpane';
          return undefined;
        },
      },
      input: {
        index: resolve(root, 'index.html'),
        nrcGuide: resolve(root, 'learn/nrc/index.html'),
        ...Object.fromEntries(local.map((e) => [e.slug, resolve(root, e.path, 'index.html')])),
      },
    },
  },
});
