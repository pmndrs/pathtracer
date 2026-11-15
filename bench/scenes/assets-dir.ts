import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

/** bench/assets: downloads (git-ignored, see MANIFEST.json) and generated assets. */
export const ASSETS_DIR = resolve(dirname(fileURLToPath(import.meta.url)), '..', 'assets');
