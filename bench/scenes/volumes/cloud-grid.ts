/**
 * The density file of `vol-cl-cloud256` (P5.v13): `cloudDensity(256)` (the procedural blobs of `vol-grid-cloud` at 256^3, a 2 voxel zero border) remapped by
 * `clamp((v - 0.25) / 0.5, 0, 1)` so the cloud has an edge and most of the box is empty (as the clouds page's bakes are), as raw little-endian float32, x fastest. Node only; the output is git-ignored (64 MB) and deterministic.
 *
 *   node bench/scenes/volumes/cloud-grid.ts      -> bench/assets/generated/volumes/cloud256.{f32,json}
 */
import { createHash } from 'node:crypto';
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { CLOUD256 } from './clouds.ts';
import { cloudDensity } from './index.ts';

const ASSETS = join(dirname(fileURLToPath(import.meta.url)), '..', '..', 'assets');

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  const t0 = performance.now();
  const d = cloudDensity(CLOUD256.n).map((v) => Math.min(1, Math.max(0, (v - 0.25) / 0.5)));
  const out = join(ASSETS, CLOUD256.file);
  mkdirSync(dirname(out), { recursive: true });
  const bytes = Buffer.from(d.buffer, d.byteOffset, d.byteLength);
  writeFileSync(out, bytes);
  let max = 0, occupied = 0, sum = 0;
  for (const v of d) { max = Math.max(max, v); if (v > 0) occupied++; sum += v; }
  const meta = { file: CLOUD256.file, resolution: [CLOUD256.n, CLOUD256.n, CLOUD256.n], max, occupancy: occupied / d.length, mean: sum / d.length, sha256: createHash('sha256').update(bytes).digest('hex'), ms: Math.round(performance.now() - t0) };
  writeFileSync(out.replace(/\.f32$/, '.json'), `${JSON.stringify(meta, null, 2)}\n`);
  console.log(JSON.stringify(meta));
}
