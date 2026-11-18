// The megakernel variants (Tier 0 / env / OpenPBR / textured) shared by `size-snapshot.ts` (Node, fixed caps) and the `kernel-compile` page
// (browser, the device's caps). Browser-safe: no node:fs.
import type { Caps, SceneCopyPacking } from '@pmndrs/raytracer';
import { loadMtlxMaterial, mtlxProgramsWgsl, ptMegakernelWgsl, ptRestirBuild, ptWavefrontBuild } from '@pmndrs/pathtracer';
import { DOC_PROC, DOC_UV } from './scenes/ab/materialx-scenes.ts';

/** R2b cutout shadow modes (`integrator.alphaShadow`), shared by the megakernel and wavefront variant lists. */
const ALPHA_SHADOW_VARIANTS = {
  roulette: { alphaShadow: 'roulette' },
  coverage: { alphaShadow: 'coverage', alphaCoverage: [1, 0.5] },
  stochastic: { alphaShadow: 'stochastic', alphaCoverage: [1, 0.5] },
  firstK: { alphaShadow: 'firstK', alphaFirstK: 1 },
  transmit: { alphaShadow: 'transmit' },
  classified: { alphaShadow: 'classified' },
} as const;

export type KernelVariants = Record<string, () => string>;

/** Name -> builder of the shipped WGSL text for `caps`. Order is the print order. */
export function kernelVariants(caps: Caps, scenePacking?: SceneCopyPacking): KernelVariants {
  const mk = (c: Caps, o: Parameters<typeof ptMegakernelWgsl>[1]): string => ptMegakernelWgsl(c, scenePacking ? { ...o, scenePacking } : o);
  const layout = { normals: true, uvs: false, positions: true };
  const texLayout = { normals: true, uvs: true, positions: true };
  const bucket = (srgb: boolean, wrap: 'repeat' | 'clamp' | 'mirror' = 'repeat') => ({ width: 256, height: 256, colorSpace: srgb ? ('srgb' as const) : ('linear' as const), wrap, layers: 2, mipLevels: 9 });
  const textures = { buckets: [bucket(false), bucket(true), bucket(true, 'clamp')] };
  const variants: Record<string, () => string> = {
    plain: () => mk(caps, { format: {}, layout }),
    env: () => mk(caps, { format: {}, layout, integrator: { envMap: true, envSampling: 'importance' } }),
    pcg: () => mk(caps, { format: {}, layout, integrator: { sampler: 'pcg', nee: false } }),
    readable: () => mk(caps, { format: {}, layout, minify: false, shortenNames: false }),
  };
  // OpenPBR-mode variants (layered lobe, P2.21; coated metal, P2.23): the byte-identity check also covers scenes with layered materials.
  const feat = { mix: true, thin: true, layered: true, medium: true };
  const mat = (layered: object, extra: object = {}) => ({ integrator: { materials: { features: { ...feat, ...extra }, layered } } });
  for (const [name, layered] of Object.entries({ albedo: { layering: 'albedo' }, albedoInternal: { layering: 'albedo', albedoBase: 'internal' }, stochastic: { layering: 'stochastic', maxDepth: 64 } })) {
    variants[`pbr-${name}`] = () => mk(caps, { format: {}, layout, ...mat(layered) } as never);
    variants[`pbr-${name}-coatedMetal`] = () => mk(caps, { format: {}, layout, ...mat(layered, { coatedMetal: true }) } as never);
  }
  // P2.32b: the fitted internal-base correction (lut: const array, texture: group 2 binding 10); the variants above stay byte-identical.
  for (const albedoCorrection of ['lut', 'texture'] as const) variants[`pbr-albedoInternal-corr-${albedoCorrection}`] = () => mk(caps, { format: {}, layout, ...mat({ layering: 'albedo', albedoBase: 'internal', albedoCorrection }) } as never);
  // Anisotropic metals (P2.26): the layout carries tangents and the material table has an anisotropic metal.
  variants['pbr-aniso'] = () => mk(caps, { format: {}, layout: { ...layout, tangents: true }, ...mat({ layering: 'albedo' }, { aniso: true }) } as never);
  // Fuzz (P2.25) and thin film (P2.27): only compiled for scenes that use them (the variants above stay byte-identical).
  variants['pbr-albedo-fuzz'] = () => mk(caps, { format: {}, layout, ...mat({ layering: 'albedo' }, { fuzz: true }) } as never);
  variants['pbr-stochastic-fuzz'] = () => mk(caps, { format: {}, layout, ...mat({ layering: 'stochastic', maxDepth: 64 }, { fuzz: true }) } as never);
  variants['pbr-fuzz-only'] = () => mk(caps, { format: {}, layout, integrator: { materials: { features: { mix: false, thin: false, layered: false, medium: false, fuzz: true }, layered: { layering: 'albedo' } } } } as never);
  variants['pbr-albedo-thinFilm'] = () => mk(caps, { format: {}, layout, ...mat({ layering: 'albedo' }, { thinFilm: true }) } as never);
  variants['pbr-albedo-fuzz-thinFilm'] = () => mk(caps, { format: {}, layout, ...mat({ layering: 'albedo' }, { fuzz: true, thinFilm: true }) } as never);
  // Participating media (P5): only compiled when `integrator.volumes` is set (the variants above stay byte-identical). One homogeneous medium bounded by an interface material, and the same plus a grid medium.
  const volHomog = { media: [{ sigmaT: [1, 1, 1], albedo: [0.9, 0.9, 0.9], g: 0.3 }], materials: [{ interior: 0, exterior: 0, boundary: false }, { interior: 1, exterior: 0, boundary: true }] };
  const volGrid = { media: [{ sigmaT: [1, 1, 1], albedo: [0.9, 0.9, 0.9], g: 0.3, grid: { resolution: [8, 8, 8], worldToIndex: [8, 0, 0, 3.5, 0, 8, 0, 3.5, 0, 0, 8, 3.5], densityMax: 1 } }], materials: volHomog.materials };
  variants['vol-homogeneous'] = () => mk(caps, { format: {}, layout, integrator: { volumes: volHomog } } as never);
  variants['vol-grid'] = () => mk(caps, { format: {}, layout, integrator: { volumes: volGrid } } as never);
  variants['vol-grid-brick'] = () => mk(caps, { format: {}, layout, integrator: { volumes: { ...volGrid, majorant: 'brick' } } } as never);
  // Optional integrator outputs (R10 hooks, R4 counters, AOVs): only compiled when asked; the variants above stay byte-identical.
  const hookDemo = { wgsl: 'fn ptOnVertex(v: PtPathVertex) -> u32 { return select(PT_HOOK_CONTINUE, PT_HOOK_TERMINATE, v.depth >= 3u); }\nfn ptTerminate(v: PtPathVertex) -> vec3f { return vec3f(0.5); }' };
  for (const [name, integrator] of Object.entries(outputVariantOptions(hookDemo))) variants[`out-${name}`] = () => mk(caps, { format: {}, layout, integrator } as never);
  variants['out-pbr-albedo-all'] = () => mk(caps, { format: {}, layout, integrator: { materials: { features: feat, layered: { layering: 'albedo' } }, hooks: hookDemo, counters: true, aovs: true } } as never);
  variants.tex = () => mk(caps, { format: {}, layout: texLayout, textures });
  // Alpha MASK (P1.4c5): the textured kernel + the cutout pass-through and the masked shadow-ray loop (only compiled for scenes with a MASK material).
  variants['tex-mask'] = () => mk(caps, { format: {}, layout: texLayout, textures, integrator: { alphaMask: true } });
  // Alpha policy (R2): the exact kernel above is the default; these drop the cutout test for shadow rays and / or indirect extension rays.
  for (const alphaPolicy of ['opaqueShadow', 'opaqueIndirect', 'opaqueSecondary'] as const) variants[`tex-mask-${alphaPolicy}`] = () => mk(caps, { format: {}, layout: texLayout, textures, integrator: { alphaMask: true, alphaPolicy } });
  // Cutout shadow modes (R2b): only the shadow visibility differs from `tex-mask`.
  for (const [name, integrator] of Object.entries(ALPHA_SHADOW_VARIANTS)) variants[`tex-mask-${name}`] = () => mk(caps, { format: {}, layout: texLayout, textures, integrator: { alphaMask: true, ...integrator } });
  // V-buffer primaries (P3.6): the first hit comes from a texture; the traced variants above are unchanged.
  variants.vbuffer = () => mk(caps, { format: {}, layout, integrator: { primary: 'vbuffer' } });
  variants['tex-mask-vbuffer'] = () => mk(caps, { format: {}, layout: texLayout, textures, integrator: { alphaMask: true, primary: 'vbuffer' } });
  // Real-time mode (P4.1, megakernel only): per-frame outputs, the blue-noise sampler, both, and the whole realtime preset on a textured layered scene.
  variants['rt-frame'] = () => mk(caps, { format: {}, layout, integrator: { frame: true } } as never);
  variants['rt-bluenoise'] = () => mk(caps, { format: {}, layout, integrator: { sampler: 'bluenoise' } } as never);
  variants['rt-bluenoise-kronecker'] = () => mk(caps, { format: {}, layout, integrator: { sampler: 'bluenoise', blueNoiseShift: 'kronecker' } } as never);
  // P4.5a: the emission plane's three settings (`rt-frame` is the default 'mirror'; 'none' is the pre-P4.5a text apart from the extra plane write).
  variants['rt-frame-emission-primary'] = () => mk(caps, { format: {}, layout, integrator: { frame: true, frameEmission: 'primary' } } as never);
  variants['rt-frame-emission-none'] = () => mk(caps, { format: {}, layout, integrator: { frame: true, frameEmission: 'none' } } as never);
  variants['rt-frame-bluenoise'] = () => mk(caps, { format: {}, layout, integrator: { frame: true, sampler: 'bluenoise' } } as never);
  variants['rt-pbr-albedo-frame-bluenoise'] = () => mk(caps, { format: {}, layout, integrator: { frame: true, sampler: 'bluenoise', materials: { features: feat, layered: { layering: 'albedo', albedoBase: 'internal', albedoCorrection: 'texture' } } } } as never);
  // Spectral transport (P5, megakernel only): only compiled with `integrator.spectral`; every variant above stays byte-identical.
  variants['spectral'] = () => mk(caps, { format: {}, layout, integrator: { spectral: true } } as never);
  variants['spectral-nodisp'] = () => mk(caps, { format: {}, layout, integrator: { spectral: { dispersion: false } } } as never);
  variants['spectral-env'] = () => mk(caps, { format: {}, layout, integrator: { spectral: true, envMap: true, envSampling: 'importance' } } as never);
  variants['spectral-pbr-albedo'] = () => mk(caps, { format: {}, layout, integrator: { spectral: true, ...mat({ layering: 'albedo' }).integrator } } as never);
  variants['spectral-pbr-stochastic'] = () => mk(caps, { format: {}, layout, integrator: { spectral: true, ...mat({ layering: 'stochastic', maxDepth: 64 }).integrator } } as never);
  variants['spectral-tex'] = () => mk(caps, { format: {}, layout: texLayout, textures, integrator: { spectral: true } } as never);
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
export function wavefrontKernelVariants(caps: Caps, scenePacking?: SceneCopyPacking): KernelVariants {
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
    // Alpha MASK (P3.7, R2): cutout pass-through in intersect, masked shadow loop; the policy variants drop one or both.
    'tex-mask': { format: {}, layout: texLayout, textures, integrator: { alphaMask: true } },
    'tex-mask-opaqueShadow': { format: {}, layout: texLayout, textures, integrator: { alphaMask: true, alphaPolicy: 'opaqueShadow' } },
    'tex-mask-opaqueIndirect': { format: {}, layout: texLayout, textures, integrator: { alphaMask: true, alphaPolicy: 'opaqueIndirect' } },
    'tex-mask-opaqueSecondary': { format: {}, layout: texLayout, textures, integrator: { alphaMask: true, alphaPolicy: 'opaqueSecondary' } },
    ...Object.fromEntries(Object.entries(ALPHA_SHADOW_VARIANTS).map(([name, integrator]) => [`tex-mask-${name}`, { format: {}, layout: texLayout, textures, integrator: { alphaMask: true, ...integrator } }])),
    // V-buffer primaries (P3.6): the camera kernel reads the jitter from the params, the intersect kernel resolves depth-0 paths from the V-buffer.
    vbuffer: { format: {}, layout, integrator: { primary: 'vbuffer' } },
    'tex-mask-vbuffer': { format: {}, layout: texLayout, textures, integrator: { alphaMask: true, primary: 'vbuffer' } },
    'pbr-albedo': { format: {}, layout, integrator: { materials: { features: feat, layered: { layering: 'albedo' } } } },
    'pbr-stochastic-fuzz': { format: {}, layout, integrator: { materials: { features: { ...feat, fuzz: true }, layered: { layering: 'stochastic', maxDepth: 64 } } } },
    // P2.32b: the fitted internal-base correction in the shade kernel.
    'pbr-albedoInternal-corr-lut': { format: {}, layout, integrator: { materials: { features: feat, layered: { layering: 'albedo', albedoBase: 'internal', albedoCorrection: 'lut' } } } },
    'pbr-albedoInternal-corr-texture': { format: {}, layout, integrator: { materials: { features: feat, layered: { layering: 'albedo', albedoBase: 'internal', albedoCorrection: 'texture' } } } },
    // P4.1b / P4.2a: the real-time options on the plain scheduler (frame planes in the accumulator tail, blue noise, ReSTIR DI at the first vertex).
    // P4.1c: R1 (`materialLod: 'secondary'`) x cone texture LOD x alpha MASK: the intersect kernel's cutout test takes the widened footprint of depth >= N (`PtwfConfig.lodTexDepth`); the other kernels are the plain `tex-mask` text.
    'tex-mask-cone-lod': { format: {}, layout: texLayout, textures, integrator: { alphaMask: true, textureLod: 'cone', materialLod: 'secondary' } },
    'rt-frame': { format: {}, layout, integrator: { frame: true } },
    'rt-frame-bluenoise': { format: {}, layout, integrator: { frame: true, sampler: 'bluenoise' } },
    'rt-restir': { format: {}, layout, integrator: { restir: { bias: 'unbiased' } } },
    'rt-realtime-restir': { format: {}, layout, integrator: { frame: true, sampler: 'bluenoise', restir: { bias: 'unbiased' } } },
    'rt-realtime-tex-restir': { format: {}, layout: texLayout, textures, integrator: { frame: true, sampler: 'bluenoise', restir: { bias: 'unbiased' } } },
    'rt-realtime-pbr-albedo': { format: {}, layout, integrator: { frame: true, sampler: 'bluenoise', materials: { features: feat, layered: { layering: 'albedo' } } } },
  };
  const hookDemo = { wgsl: 'fn ptOnVertex(v: PtPathVertex) -> u32 { return select(PT_HOOK_CONTINUE, PT_HOOK_TERMINATE, v.depth >= 3u); }\nfn ptTerminate(v: PtPathVertex) -> vec3f { return vec3f(0.5); }' };
  for (const [name, integrator] of Object.entries(outputVariantOptions(hookDemo))) base[`out-${name}`] = { format: {}, layout, integrator };
  base['out-pbr-albedo-all'] = { format: {}, layout, integrator: { materials: { features: feat, layered: { layering: 'albedo' } }, hooks: hookDemo, counters: true, aovs: true } };
  const variants: Record<string, () => string> = {};
  for (const [name, options] of Object.entries(base)) {
    for (const kernel of ['camera', 'intersect', 'shade', 'shadow', 'args'] as const) {
      variants[`wf-${name}-${kernel}`] = () => ptWavefrontBuild(caps, (scenePacking ? { ...options, scenePacking } : options) as never)[kernel].code;
    }
  }
  return variants;
}

/**
 * ReSTIR DI (P4): the shade kernel (megakernel with `integrator.restir`) and the module of the four reservoir passes, for the size cap (50 KB) and the byte-identity check
 * (`size-snapshot.ts`). Apart from `kernelVariants` because the kernel-compile page compiles every entry there as a stand-alone `ptMain` module.
 */
export function restirKernelVariants(caps: Caps, scenePacking?: SceneCopyPacking): KernelVariants {
  const layout = { normals: true, uvs: false, positions: true };
  const base = scenePacking ? { format: {}, layout, scenePacking } : { format: {}, layout };
  const variants: KernelVariants = {
    'restir-shade': () => ptMegakernelWgsl(caps, { ...base, integrator: { restir: {} } }),
    'restir-passes': () => ptRestirBuild(caps, { ...base, integrator: { restir: {} } }).code,
    'restir-passes-env': () => ptRestirBuild(caps, { ...base, integrator: { restir: {}, envMap: true, envSampling: 'importance' } }).code,
  };
  // P4.2: the blue-noise sampler, V-buffer primaries, the shared primary hit, textures (+ alpha MASK), OpenPBR mode (layered albedo + mix + thin): shade kernel and passes module of each.
  const bucket = (srgb: boolean) => ({ width: 256, height: 256, colorSpace: srgb ? ('srgb' as const) : ('linear' as const), wrap: 'repeat' as const, layers: 2, mipLevels: 9 });
  const tex = { ...base, layout: { normals: true, uvs: true, positions: true }, textures: { buckets: [bucket(false), bucket(true)] } };
  const pbr = { features: { mix: true, thin: true, layered: true, medium: false }, layered: { layering: 'albedo' as const, albedoBase: 'internal' as const } };
  const sets: Record<string, Record<string, unknown>> = {
    bluenoise: { ...base, integrator: { restir: {}, sampler: 'bluenoise' } },
    vbuffer: { ...base, integrator: { restir: {}, primary: 'vbuffer' } },
    shared: { ...base, integrator: { restir: { sharedHit: true } } },
    tex: { ...tex, integrator: { restir: {} } },
    'tex-mask': { ...tex, integrator: { restir: {}, alphaMask: true } },
    'tex-mask-shared': { ...tex, integrator: { restir: { sharedHit: true }, alphaMask: true } },
    pbr: { ...base, integrator: { restir: {}, materials: pbr } },
    'tex-pbr-mask': { ...tex, integrator: { restir: {}, alphaMask: true, materials: pbr } },
    realtime: { ...tex, integrator: { restir: { sharedHit: true }, sampler: 'bluenoise', frame: true, alphaMask: true, materials: pbr } },
    // P5: ReSTIR GI on top of DI (no textures / alpha MASK: refused).
    gi: { ...base, integrator: { restir: { gi: {} } } },
    'gi-unbiased': { ...base, integrator: { restir: { bias: 'unbiased', initialVisibility: false, gi: { bias: 'unbiased' } } } },
    'gi-env': { ...base, integrator: { restir: { gi: {} }, envMap: true, envSampling: 'importance' } },
    'gi-pbr': { ...base, integrator: { restir: { gi: {} }, materials: pbr } },
    'gi-realtime': { ...base, integrator: { restir: { sharedHit: true, gi: {} }, sampler: 'bluenoise', frame: true } },
  };
  for (const [name, o] of Object.entries(sets)) {
    variants[`restir-shade-${name}`] = () => ptMegakernelWgsl(caps, o as never);
    variants[`restir-passes-${name}`] = () => ptRestirBuild(caps, o as never).code;
    if (name.startsWith('gi')) variants[`restir-gipasses-${name}`] = () => ptRestirBuild(caps, o as never, 'gi').code;
  }
  return variants;
}

/**
 * MaterialX codegen (P5, `integrator.materialx`): the megakernel and the wavefront shade kernel of the two codegen scenes' graphs (`mtlx-proc`: position-driven, no image; `mtlx-uv`: three
 * images and a normal map), OpenPBR mode with the layered albedo lobes. A kernel without `materialx` is byte-identical to before, so these are new files only.
 */
export function materialxKernelVariants(caps: Caps, scenePacking?: SceneCopyPacking): KernelVariants {
  const layout = { normals: true, uvs: true, positions: true };
  const bucket = (srgb: boolean) => ({ width: 128, height: 128, colorSpace: srgb ? ('srgb' as const) : ('linear' as const), wrap: 'repeat' as const, layers: 1, mipLevels: 8 });
  const textures = { buckets: [bucket(false), bucket(true)] };
  const base = scenePacking ? { format: {}, layout, textures, scenePacking } : { format: {}, layout, textures };
  const pbr = { features: { mix: true, thin: true, layered: true, medium: false }, layered: { layering: 'albedo' as const, albedoBase: 'internal' as const } };
  const program = (doc: string) => mtlxProgramsWgsl([{ material: 1, mtl: loadMtlxMaterial(doc) }], { textureSlot: (r) => (r.file.includes('checker') ? 1 : 0) });
  const variants: KernelVariants = {};
  for (const [name, doc] of Object.entries({ 'mtlx-proc': DOC_PROC, 'mtlx-uv': DOC_UV })) {
    const p = program(doc);
    const o = { ...base, integrator: { materials: pbr, materialx: { wgsl: p.wgsl, normal: p.normal } } };
    variants[name] = () => ptMegakernelWgsl(caps, o as never);
    variants[`wf-${name}-shade`] = () => ptWavefrontBuild(caps, o as never).shade.code;
  }
  return variants;
}
