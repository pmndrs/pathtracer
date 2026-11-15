// R1: kernel bytes (WGSL, minified, short names) with and without `integrator.materialLod: 'secondary'`.
//   node bench/lod-sizes.ts
// Megakernel: the reduced branch is added next to the full lobes. Wavefront: `shade` (depth < N) is the default shade kernel, `shadeSecondary` (depth >= N) has no
// layered walk / fuzz / thin film / aniso / coated-metal code. The D26 budget is 50 KB per kernel.
import { buildCaps } from '@pmndrs/raytracer';
import { ptMegakernelWgsl, ptWavefrontBuild, ptWavefrontKernelBytes } from '@pmndrs/pathtracer';

const caps = buildCaps({ features: new Set(['subgroups', 'shader-f16', 'timestamp-query']), limits: {} }, new Set());
const layout = { normals: true, uvs: false, positions: true, tangents: true };
const texLayout = { normals: true, uvs: true, positions: true, tangents: true };
const bucket = (srgb: boolean) => ({ width: 256, height: 256, colorSpace: srgb ? ('srgb' as const) : ('linear' as const), wrap: 'repeat' as const, layers: 2, mipLevels: 9 });
const textures = { buckets: [bucket(false), bucket(true)] };
const feat = { mix: true, thin: true, layered: true, medium: true };
const all = { ...feat, coatedMetal: true, aniso: true, fuzz: true, thinFilm: true };

interface Variant {
  name: string;
  options: Record<string, unknown>;
}
const variants: Variant[] = [
  { name: 'layered albedo', options: { layout, integrator: { materials: { features: feat, layered: { layering: 'albedo' } } } } },
  { name: 'layered stochastic', options: { layout, integrator: { materials: { features: feat, layered: { layering: 'stochastic', maxDepth: 64 } } } } },
  { name: 'albedo + coatedMetal + aniso + fuzz', options: { layout, integrator: { materials: { features: { ...feat, coatedMetal: true, aniso: true, fuzz: true }, layered: { layering: 'albedo' } } } } },
  { name: 'albedo + all (thin film too)', options: { layout, integrator: { materials: { features: all, layered: { layering: 'albedo' } } } } },
  { name: 'albedo + all + env map', options: { layout, integrator: { envMap: true, envSampling: 'importance', materials: { features: all, layered: { layering: 'albedo' } } } } },
  { name: 'albedo + all + env + textures (cone LOD)', options: { layout: texLayout, textures, integrator: { envMap: true, envSampling: 'importance', textureLod: 'cone', materials: { features: all, layered: { layering: 'albedo' } } } } },
  { name: 'stochastic + coatedMetal + aniso + fuzz? (fuzz over walk)', options: { layout, integrator: { materials: { features: { ...feat, coatedMetal: true, aniso: true, fuzz: true }, layered: { layering: 'stochastic', maxDepth: 64 } } } } },
];

const kb = (n: number): string => `${(n / 1000).toFixed(1)}`;
console.log('| variant | mega full | mega lod | wf shade (full) | wf shade secondary | secondary / shade |');
console.log('|---|---|---|---|---|---|');
for (const v of variants) {
  const mk = (lod: boolean) => ptMegakernelWgsl(caps, { format: {}, ...v.options, integrator: { ...(v.options.integrator as object), ...(lod ? { materialLod: 'secondary' } : {}) } } as never).length;
  const wf = ptWavefrontBuild(caps, { format: {}, ...v.options, integrator: { ...(v.options.integrator as object), materialLod: 'secondary' } } as never);
  const bytes = ptWavefrontKernelBytes(wf);
  console.log(`| ${v.name} | ${kb(mk(false))} | ${kb(mk(true))} | ${kb(bytes.shade)} | ${kb(bytes.shadeSecondary as number)} | ${((bytes.shadeSecondary as number) / bytes.shade).toFixed(2)} |`);
}
