// Writes the shipped kernel text of the Tier 0 / env / textured variants to a directory (byte-identity check between two commits).
//   node bench/size-snapshot.ts <outDir>
import { writeFileSync, mkdirSync } from 'node:fs';
import { buildCaps } from '@pmndrs/raytracer';
import { kernelVariants, wavefrontKernelVariants } from './kernel-variants.ts';

const out = process.argv[2] ?? '.';
mkdirSync(out, { recursive: true });
const caps = buildCaps({ features: new Set(['subgroups', 'shader-f16', 'timestamp-query']), limits: {} }, new Set());
const variants = { ...kernelVariants(caps), ...wavefrontKernelVariants(caps) };
for (const [name, build] of Object.entries(variants)) {
  try { const code = build(); writeFileSync(`${out}/${name}.wgsl`, code); console.log(name, code.length); } catch (e) { console.log(name, 'skipped', String(e).slice(0, 80)); }
}
