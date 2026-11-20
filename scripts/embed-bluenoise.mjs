// Writes packages/raytracer/src/noise/builtin/<name>.ts from the committed blue-noise PNGs (run from the repo root after
// regenerating them: node scripts/embed-bluenoise.mjs). bluenoise-builtin.test.ts fails if the two drift apart.
import { readFileSync, writeFileSync } from 'node:fs';
const dir = 'packages/raytracer/assets/bluenoise';
const manifest = JSON.parse(readFileSync(`${dir}/manifest.json`, 'utf8'));
for (const name of ['gf-vec2-64', 'gf-vec2-128']) {
  const meta = manifest.textures.find((t) => t.name === name);
  const b64 = readFileSync(`${dir}/${meta.png}`).toString('base64');
  const id = name.replace(/-/g, '_');
  writeFileSync(`packages/raytracer/src/noise/builtin/${name}.ts`,
`// Generated from assets/bluenoise/${meta.png} (our own texture, D16). Do not edit: node scripts/embed-bluenoise.mjs rewrites it; ../bluenoise-builtin.test.ts checks it against the asset.
import type { BlueNoiseMeta } from '../bluenoise.ts';

export const meta: BlueNoiseMeta = ${JSON.stringify(meta)};
/** The 16-bit PNG, base64. */
export const png: string = '${b64}';
`);
  console.log(name, b64.length);
}
