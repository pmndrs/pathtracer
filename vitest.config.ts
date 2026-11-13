import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    include: ['packages/*/src/**/*.test.ts', 'packages/*/tools/**/*.test.ts', 'bench/**/*.test.ts'],
    exclude: ['**/node_modules/**', 'research/**', '.cache/**'],
  },
});
