// Writes the shipped kernel text of the Tier 0 / env / textured variants to a directory (byte-identity check between two commits).
//   node bench/size-snapshot.ts <outDir> [split|packed]   (scene copy packing, P3.11; default: the library default. `split` text must stay byte-identical across commits)
import { writeFileSync, mkdirSync } from 'node:fs';
import { buildCaps } from '@pmndrs/raytracer';
import { kernelVariants, materialxKernelVariants, restirKernelVariants, wavefrontKernelVariants } from './kernel-variants.ts';

const out = process.argv[2] ?? '.';
mkdirSync(out, { recursive: true });
const caps = buildCaps({ features: new Set(['subgroups', 'shader-f16', 'timestamp-query']), limits: {} }, new Set());
const packing = process.argv[3] === 'packed' || process.argv[3] === 'split' ? process.argv[3] : undefined;
const variants = { ...kernelVariants(caps, packing), ...wavefrontKernelVariants(caps, packing), ...restirKernelVariants(caps, packing), ...materialxKernelVariants(caps, packing) };
for (const [name, build] of Object.entries(variants)) {
  try { const code = build(); writeFileSync(`${out}/${name}.wgsl`, code); console.log(name, code.length); } catch (e) { console.log(name, 'skipped', String(e).slice(0, 80)); }
}
