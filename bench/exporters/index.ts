import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import type { ExportResult } from './types.ts';

export { toPbrt } from './pbrt.ts';
export { toMitsuba } from './mitsuba.ts';
export { toGltf } from './gltf.ts';
export type { ExportOptions, ExportResult, ExportedFile, Exporter } from './types.ts';

/** Write an export to `dir` (created); returns the absolute scene-file path. */
export function writeExport(result: ExportResult, dir: string): string {
  for (const f of result.files) {
    const p = join(dir, f.path);
    mkdirSync(dirname(p), { recursive: true });
    writeFileSync(p, f.data);
  }
  return join(dir, result.sceneFile);
}
