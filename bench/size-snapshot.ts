// Writes the shipped kernel text of the Tier 0 / env / textured variants to a directory (byte-identity check between two commits).
//   node bench/size-snapshot.ts <outDir>
import { writeFileSync, mkdirSync } from 'node:fs';
import { buildCaps } from '@pmndrs/raytracer';
import { ptMegakernelWgsl } from '@pmndrs/pathtracer';

const out = process.argv[2] ?? '.';
mkdirSync(out, { recursive: true });
const caps = buildCaps({ features: new Set(['subgroups', 'shader-f16', 'timestamp-query']), limits: {} }, new Set());
const layout = { normals: true, uvs: false, positions: true };
const texLayout = { normals: true, uvs: true, positions: true };
const bucket = (srgb: boolean, wrap: 'repeat' | 'clamp' | 'mirror' = 'repeat') => ({ width: 256, height: 256, colorSpace: srgb ? ('srgb' as const) : ('linear' as const), wrap, layers: 2, mipLevels: 9 });
const textures = { buckets: [bucket(false), bucket(true), bucket(true, 'clamp')] };
const variants: Record<string, () => string> = {
  plain: () => ptMegakernelWgsl(caps, { format: {}, layout }),
  env: () => ptMegakernelWgsl(caps, { format: {}, layout, integrator: { envMap: true, envSampling: 'importance' } }),
  pcg: () => ptMegakernelWgsl(caps, { format: {}, layout, integrator: { sampler: 'pcg', nee: false } }),
  readable: () => ptMegakernelWgsl(caps, { format: {}, layout, minify: false, shortenNames: false }),
};
// OpenPBR-mode variants (layered lobe, P2.21; coated metal, P2.23): the byte-identity check also covers scenes with layered materials.
const feat = { mix: true, thin: true, layered: true, medium: true };
const mat = (layered: object, extra: object = {}) => ({ integrator: { materials: { features: { ...feat, ...extra }, layered } } });
for (const [name, layered] of Object.entries({ albedo: { layering: 'albedo' }, albedoInternal: { layering: 'albedo', albedoBase: 'internal' }, stochastic: { layering: 'stochastic', maxDepth: 64 } })) {
  variants[`pbr-${name}`] = () => ptMegakernelWgsl(caps, { format: {}, layout, ...mat(layered) } as never);
  variants[`pbr-${name}-coatedMetal`] = () => ptMegakernelWgsl(caps, { format: {}, layout, ...mat(layered, { coatedMetal: true }) } as never);
}
variants.tex = () => ptMegakernelWgsl(caps, { format: {}, layout: texLayout, textures });
for (const [name, build] of Object.entries(variants)) {
  try { const code = build(); writeFileSync(`${out}/${name}.wgsl`, code); console.log(name, code.length); } catch (e) { console.log(name, 'skipped', String(e).slice(0, 80)); }
}
