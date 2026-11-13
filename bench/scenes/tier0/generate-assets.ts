/** Writes the procedural assets the Tier 0 scenes reference. Run: `node bench/scenes/tier0/generate-assets.ts`. */
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { mkdirSync, writeFileSync } from 'node:fs';
import { writePFM } from '../../metrics/pfm.ts';
import { ENV_FILE, ENV_HEIGHT, ENV_WIDTH, generateSkyImage } from './env-sky.ts';

export const ASSETS_DIR = join(dirname(fileURLToPath(import.meta.url)), '..', '..', 'assets');

export function generateAssets(): string[] {
  const path = join(ASSETS_DIR, ENV_FILE);
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, writePFM({ width: ENV_WIDTH, height: ENV_HEIGHT, channels: 3, data: generateSkyImage() }));
  return [path];
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  for (const p of generateAssets()) console.log('wrote', p);
}
