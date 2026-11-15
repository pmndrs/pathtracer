// The megakernel variants (Tier 0 / env / OpenPBR / textured) shared by `size-snapshot.ts` (Node, fixed caps) and the `kernel-compile` page
// (browser, the device's caps). Browser-safe: no node:fs.
import type { Caps } from '@pmndrs/raytracer';
import { ptMegakernelWgsl, ptWavefrontBuild } from '@pmndrs/pathtracer';

export type KernelVariants = Record<string, () => string>;

/** Name -> builder of the shipped WGSL text for `caps`. Order is the print order. */
export function kernelVariants(caps: Caps): KernelVariants {
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
  // Optional integrator outputs (R10 hooks, R4 counters, AOVs): only compiled when asked; the variants above stay byte-identical.
  const hookDemo = { wgsl: 'fn ptOnVertex(v: PtPathVertex) -> u32 { return select(PT_HOOK_CONTINUE, PT_HOOK_TERMINATE, v.depth >= 3u); }\nfn ptTerminate(v: PtPathVertex) -> vec3f { return vec3f(0.5); }' };
  for (const [name, integrator] of Object.entries(outputVariantOptions(hookDemo))) variants[`out-${name}`] = () => ptMegakernelWgsl(caps, { format: {}, layout, integrator } as never);
  variants['out-pbr-albedo-all'] = () => ptMegakernelWgsl(caps, { format: {}, layout, integrator: { materials: { features: feat, layered: { layering: 'albedo' } }, hooks: hookDemo, counters: true, aovs: true } } as never);
  variants.tex = () => ptMegakernelWgsl(caps, { format: {}, layout: texLayout, textures });
  // Alpha MASK (P1.4c5): the textured kernel + the cutout pass-through and the masked shadow-ray loop (only compiled for scenes with a MASK material).
  variants['tex-mask'] = () => ptMegakernelWgsl(caps, { format: {}, layout: texLayout, textures, integrator: { alphaMask: true } });
  return variants;
}

/** Integrator option sets for the optional outputs (plain Tier 0 kernel): hooks, counters, AOVs, all three. */
function outputVariantOptions(hookDemo: { wgsl: string }): Record<string, object> {
  return { hooks: { hooks: hookDemo }, counters: { counters: true }, aovs: { aovs: true }, all: { hooks: hookDemo, counters: true, aovs: true } };
}

/**
 * The wavefront kernels (P3.1) of a few variants, as `wf-<variant>-<kernel>` -> text, for the byte-identity check (`size-snapshot.ts`). Kept apart from
 * `kernelVariants`: the kernel-compile page compiles each entry of that one as a self-contained module, and a wavefront kernel is one of five.
 */
export function wavefrontKernelVariants(caps: Caps): KernelVariants {
  const layout = { normals: true, uvs: false, positions: true };
  const texLayout = { normals: true, uvs: true, positions: true };
  const bucket = (srgb: boolean, wrap: 'repeat' | 'clamp' | 'mirror' = 'repeat') => ({ width: 256, height: 256, colorSpace: srgb ? ('srgb' as const) : ('linear' as const), wrap, layers: 2, mipLevels: 9 });
  const textures = { buckets: [bucket(false), bucket(true), bucket(true, 'clamp')] };
  const feat = { mix: true, thin: true, layered: true, medium: true };
  const base: Record<string, object> = {
    plain: { format: {}, layout },
    env: { format: {}, layout, integrator: { envMap: true, envSampling: 'importance' } },
    pcg: { format: {}, layout, integrator: { sampler: 'pcg', nee: false } },
    tex: { format: {}, layout: texLayout, textures },
    'pbr-albedo': { format: {}, layout, integrator: { materials: { features: feat, layered: { layering: 'albedo' } } } },
    'pbr-stochastic-fuzz': { format: {}, layout, integrator: { materials: { features: { ...feat, fuzz: true }, layered: { layering: 'stochastic', maxDepth: 64 } } } },
  };
  const hookDemo = { wgsl: 'fn ptOnVertex(v: PtPathVertex) -> u32 { return select(PT_HOOK_CONTINUE, PT_HOOK_TERMINATE, v.depth >= 3u); }\nfn ptTerminate(v: PtPathVertex) -> vec3f { return vec3f(0.5); }' };
  for (const [name, integrator] of Object.entries(outputVariantOptions(hookDemo))) base[`out-${name}`] = { format: {}, layout, integrator };
  base['out-pbr-albedo-all'] = { format: {}, layout, integrator: { materials: { features: feat, layered: { layering: 'albedo' } }, hooks: hookDemo, counters: true, aovs: true } };
  const variants: Record<string, () => string> = {};
  for (const [name, options] of Object.entries(base)) {
    for (const kernel of ['camera', 'intersect', 'shade', 'shadow', 'args'] as const) {
      variants[`wf-${name}-${kernel}`] = () => ptWavefrontBuild(caps, options as never)[kernel].code;
    }
  }
  return variants;
}
