import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { ASSETS_DIR } from './assets-dir.ts';
import { allScenes, getScene } from './index.ts';
import { interiorNames, interiorPath, isInterior, missingInterior, parseInterior } from './interiors.ts';

export function localSceneNames(): string[] {
  return [...allScenes.map(s => s.name), ...interiorNames.filter(id => existsSync(join(ASSETS_DIR, interiorPath(id))))];
}

export function getLocalScene(name: string) {
  if (!isInterior(name)) return getScene(name);
  const file = join(ASSETS_DIR, interiorPath(name));
  if (!existsSync(file)) throw new Error(missingInterior(name));
  return parseInterior(name, readFileSync(file, 'utf8'));
}
