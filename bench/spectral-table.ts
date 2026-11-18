// Fits the RGB -> spectrum table of the spectral mode (P5) and writes `bench/assets/generated/rgb2spec-<res>.f32` (git-ignored), so a page loads it instead of
// fitting it (res 64: ~4 s).   node bench/spectral-table.ts [res=64]
import { mkdirSync, writeFileSync } from 'node:fs';
import { buildRgb2SpecTable } from '@pmndrs/pathtracer';

const res = Number(process.argv[2] ?? 64);
const t = performance.now();
const table = buildRgb2SpecTable(res);
mkdirSync(new URL('./assets/generated/', import.meta.url), { recursive: true });
const file = new URL(`./assets/generated/rgb2spec-${res}.f32`, import.meta.url);
writeFileSync(file, new Uint8Array(table.data.buffer));
console.log(`rgb2spec-${res}.f32: ${table.data.byteLength} bytes in ${Math.round(performance.now() - t)} ms`);
