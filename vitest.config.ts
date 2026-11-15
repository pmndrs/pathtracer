import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    include: ['packages/*/src/**/*.test.ts', 'packages/*/tools/**/*.test.ts', 'bench/**/*.test.ts'],
    exclude: ['**/node_modules/**', 'research/**', '.cache/**'],
    // The Mac is shared: CPU-heavy oracle tests (BSDF furnaces, the layered walk, the Sponza loader) take 5-7 s alone and time out
    // at the 5 s default under other agents' load (load 8-18). 30 s keeps a real hang visible.
    testTimeout: 30_000,
  },
});
