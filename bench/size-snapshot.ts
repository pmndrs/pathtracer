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
// Anisotropic metals (P2.26): the layout carries tangents and the material table has an anisotropic metal.
variants['pbr-aniso'] = () => ptMegakernelWgsl(caps, { format: {}, layout: { ...layout, tangents: true }, ...mat({ layering: 'albedo' }, { aniso: true }) } as never);
// Fuzz (P2.25) and thin film (P2.27): only compiled for scenes that use them (the variants above stay byte-identical).
variants['pbr-albedo-fuzz'] = () => ptMegakernelWgsl(caps, { format: {}, layout, ...mat({ layering: 'albedo' }, { fuzz: true }) } as never);
variants['pbr-stochastic-fuzz'] = () => ptMegakernelWgsl(caps, { format: {}, layout, ...mat({ layering: 'stochastic', maxDepth: 64 }, { fuzz: true }) } as never);
variants['pbr-fuzz-only'] = () => ptMegakernelWgsl(caps, { format: {}, layout, integrator: { materials: { features: { mix: false, thin: false, layered: false, medium: false, fuzz: true }, layered: { layering: 'albedo' } } } } as never);
variants['pbr-albedo-thinFilm'] = () => ptMegakernelWgsl(caps, { format: {}, layout, ...mat({ layering: 'albedo' }, { thinFilm: true }) } as never);
variants['pbr-albedo-fuzz-thinFilm'] = () => ptMegakernelWgsl(caps, { format: {}, layout, ...mat({ layering: 'albedo' }, { fuzz: true, thinFilm: true }) } as never);
variants.tex = () => ptMegakernelWgsl(caps, { format: {}, layout: texLayout, textures });
// Alpha MASK (P1.4c5): the textured kernel + the cutout pass-through and the masked shadow-ray loop (only compiled for scenes with a MASK material).
variants['tex-mask'] = () => ptMegakernelWgsl(caps, { format: {}, layout: texLayout, textures, integrator: { alphaMask: true } });
for (const [name, build] of Object.entries(variants)) {
  try { const code = build(); writeFileSync(`${out}/${name}.wgsl`, code); console.log(name, code.length); } catch (e) { console.log(name, 'skipped', String(e).slice(0, 80)); }
}
