/**
 * The megakernel path tracer's WGSL (prefix `pt`), one thread per pixel, one sample per dispatch. The integrator is pbrt-v4's
 * `PathIntegrator` (Apache-2.0, THIRD_PARTY 1.3) in structure: emission at the hit, depth check, next-event estimation with
 * the power heuristic against BSDF sampling, BSDF sampling, Russian roulette; Mitsuba's `max_depth` is ours + 1.
 * See docs/internal/design/integrator.md for the semantics line by line and DIVERGES.md for where we leave pbrt.
 *
 * Bindings: bvh walk at group 0, scene copy at group 1 (two buffers), group 2 = `ptParams` (uniform, dynamic offset) and
 * `ptAccum` (read-write `array<vec4f>`, one f32 sum per pixel). Radiance is `PtRadiance` = vec4f (rgb + a reserved lane,
 * spec section 6: spectral-ready); the accumulator adds `(L.rgb, 1)`, so `accum.w` is the sample count.
 *
 * Specialised per pipeline (module constants): sampler, NEE on/off, environment sampling, sun visibility, VNDF and transport
 * of the BSDF. Per dispatch (uniform): image size, sample index, seed, maxDepth, rrStartDepth, camera.
 */
import type { PtMaterialOptions } from '../bsdf/adapter.ts';
import type { PtBsdfOptions } from '../bsdf/bsdf.ts';
import type { PtBlueNoiseOptions, PtBlueNoiseShift, PtSamplerKind } from './dimensions.ts';
import { ptHookVertexExpr, ptHookWantsRadiance, type PtHooksOptions } from './outputs-wgsl.ts';
import { ptVolPhaseSampleWgsl, ptVolumeWgsl } from '../volume/volume-wgsl.ts';
import type { PtVolumeOptions } from '../volume/volume.ts';
import { ptSpectralWgsl, resolveSpectralOptions, type PtSpectralOptions, type ResolvedSpectralOptions } from '../spectral/spectral-wgsl.ts';
import { PT_RESTIR_HIT_BINDING, PT_RESTIR_OCT_WGSL, PT_RESTIR_SHADE_BINDING, PT_RESTIR_UNPACK_WGSL, type PtRestirOptions } from '../restir/restir-wgsl.ts';
import { PT_RESTIR_GI_SHADE_BINDING, ptRestirGiShadeWgsl } from '../restir/restir-gi-wgsl.ts';

export interface PtIntegratorOptions extends PtBsdfOptions {
  /** `'owen-sobol'` (reference), `'pcg'` (debug) or `'bluenoise'` (P4.1, real-time frames: the `blueNoise` option, default the package's `gf-vec2-64` via `loadBlueNoise`; both schedulers, the wavefront without refill / layers; `dimensions.ts`). */
  sampler?: PtSamplerKind;
  /** `sampler: 'bluenoise'`: how many leading pairs read the texture (default 6, at most 15). */
  blueNoisePairs?: number;
  /** `sampler: 'bluenoise'`: the per-frame shift (`'owen'` default, `'kronecker'`; P4.1 A/B, `docs/internal/design/realtime.md`). */
  blueNoiseShift?: PtBlueNoiseShift;
  /**
   * Per-frame guide outputs for a temporal denoiser / upscaler (P4.1, real-time mode; both schedulers, the wavefront without refill / layers, where the planes live in the accumulator's tail; docs/internal/design/realtime.md): the kernel writes ONE sample per pixel per
   * dispatch (no accumulation: `accumulate` defaults to false with it) plus 7 planes in `ptFrame`: diffuse and specular radiance demodulated, world normal + viewZ, motion
   * + roughness, the demodulation factors, the emission that bypasses them (`frameEmission`). Default false: no frame code at all (the kernel text is unchanged).
   */
  frame?: boolean;
  /**
   * With `frame` (P4.5a, docs/internal/design/realtime.md section 3): which light bypasses the demodulated planes into plane 6 (`emission`, added back AFTER the filter): radiance the primary
   * BSDF does not modulate, noise-free at the primary hit, which dividing by the albedo factor (floored at 1e-3 for an emitter) turns into a huge value the filter would spread.
   *  - `'none'`: off, plane 6 is zero and emission is demodulated like everything else (the P4.1 behaviour; the A/B baseline).
   *  - `'primary'`: emission of an emitter hit by the primary ray, and the environment / sun seen by a primary ray that missed.
   *  - `'mirror'` (default): also what a chain of perfect-mirror bounces (smooth conductors) from the primary hit reaches (an emitter or the sky seen in a mirror, throughput included).
   */
  frameEmission?: 'none' | 'primary' | 'mirror';
  /** `false`: the accumulator is overwritten with this dispatch's sample instead of summed (a frame, not a progressive pass). Default true, or false with `frame`. */
  accumulate?: boolean;
  /** Next-event estimation + MIS. Default true. False = BSDF sampling only (debug; delta suns contribute nothing). */
  nee?: boolean;
  /**
   * `'uniform'` (sphere; NEE with MIS), `'importance'` (an environment map sampled by its luminance; needs `envMap`, otherwise the
   * same as uniform) or `'none'` (pbrt's `allowIncompletePDF` for a uniform sky: BSDF sampling only for it). Default 'uniform'.
   */
  envSampling?: 'uniform' | 'importance' | 'none';
  /** The scene has an equirect environment map: compile its lookup, pdf and sampling (kernel size +). Default false (constant radiance only). */
  envMap?: boolean;
  /** A sun with an angular radius is visible to camera and specular paths (default false: the references' sun is a delta light and invisible). */
  sunVisible?: boolean;
  /**
   * The scene has material textures (P1.4c; set by `PtMegakernel.create` from `textures`): the kernel fetches them in the material
   * decode and tracks a ray cone. Default false: no texture code at all (the kernel text is unchanged).
   */
  textures?: boolean;
  /**
   * Level of detail of texture lookups: `0` (default) always the base level, the reference setting (pbrt `bilinear` and Mitsuba `bilinear`
   * do not use mips); `'cone'` ray cones (Akenine-Moller et al. 2021): the width starts at the pixel angle, grows with distance, a
   * specular bounce keeps its spread, a non-delta bounce adds `PT_CONE_DIFFUSE_SPREAD` radians. The starting spread is the pixel angle times
   * `PtLimits.textureConeScale` (default 1; pbrt-v4 uses max(1/8, 1/sqrt(spp)): with many samples per pixel the pixel box already
   * averages the texture, so a full-pixel filter on top blurs it twice).
   */
  textureLod?: 'cone' | 0;
  /**
   * The scene has `alphaMode` MASK materials (P1.4c5; set by the adapter from the scene copy; needs `textures`): a hit whose base colour alpha
   * times `opacity` is below `alphaCutoff` is passed through without costing a bounce (a hard cutoff, as pbrt's `alpha` texture and Mitsuba's `mask`
   * BSDF), and shadow rays pass through such hits too (`ptVisibleMasked`). Default false: no mask code at all (the kernel text is unchanged).
   */
  alphaMask?: boolean;
  /**
   * Which rays honour the cutout of an alpha MASK hit (R2; only with `alphaMask`). `'exact'` (default): all of them, the reference behaviour.
   * The other three are BIASED speed settings, never for reference renders: a masked-out texel is treated as solid for the named ray class,
   * so those rays stop at the leaf card instead of passing through its holes.
   *  - `'opaqueShadow'`: shadow rays (next-event visibility) are plain any-hit walks (`rtVisible`, no closest-hit loop, no texture fetch). Foliage
   *    shadows become the shadows of the whole card (darker, too much shadow); camera and bounce rays are exact.
   *  - `'opaqueIndirect'`: extension rays after the first scattering vertex (depth >= 1: bounce, mirror and glass chains) stop at a cutout card.
   *    Camera rays (depth 0) and shadow rays stay exact (holes are still seen, shadows still pass through them).
   *  - `'opaqueSecondary'`: both.
   * Without `alphaMask` the setting has no effect and the kernel text is unchanged; `'exact'` keeps every kernel byte-identical.
   */
  alphaPolicy?: PtAlphaPolicy;
  /**
   * Shadow-ray handling of alpha MASK cutouts (R2b; only with `alphaMask`). `'policy'` (default): as `alphaPolicy` says (exact closest-hit loop, or solid any-hit),
   * the kernel text is unchanged. The others replace the shadow side of the policy (the indirect side still follows `alphaPolicy`) and are about recovering
   * the light the solid walk loses at (almost) its cost; see DIVERGES V144-V148 and TASKS R2b for the A/B.
   *  - `'roulette'`: UNBIASED. The solid any-hit walk runs first; when it says blocked, with probability `alphaRoulette` (q) the exact closest-hit loop runs and a
   *    visible result counts 1/q. Solid blockers are a superset of the exact ones, so the estimator is exact in the mean; the cost is noise.
   *  - `'coverage'`: BIASED, one closest-hit walk, no texture fetch, no loop: the first hit's material transmits `1 - alphaCoverage[material]` (a solid material
   *    has coverage 1). A ray through several cards sees only the first one's transmittance (too bright behind a second card).
   *  - `'stochastic'`: closest-hit loop without texture fetches: at every hit the ray is blocked with probability `alphaCoverage[material]`, else it steps over the card
   *    (stochastic opacity with the mean coverage in place of the texel). Unbiased for holes independent between cards.
   *  - `'firstK'`: BIASED dark: the first `alphaFirstK` cutout hits are tested exactly (closest hit + texture), the rest of the ray is a solid any-hit walk.
   *  - `'transmit'`: BIASED by a constant, at about the solid walk's cost: the adapter puts every cutout card in an instance with mask `alphaMixedMask`. A shadow ray is an
   *    any-hit walk over everything else (blocked: 0), then one over the cards only: blocked by cards alone counts `alphaTransmit` (beta, default 0.5 = 1 - the mean coverage)
   *    instead of 0. Right on average when beta is the share of card-blocked rays that the exact test would pass (calibrate it: the mean image is linear in beta),
   *    blind to which texel was hit, so the hole pattern of a card's shadow becomes a uniform partial shadow.
   *  - `'classified'`: EXACT (to the classification's conservatism): the scene was split by the adapter into `alphaMixedMask` instances (cards whose triangles
   *    straddle the cutoff) and everything else (solid, the fully opaque triangles of cards included). The shadow ray is a solid any-hit over the second set,
   *    then the exact loop over the first only.
   */
  alphaShadow?: PtAlphaShadow;
  /** `alphaShadow: 'roulette'`: probability q that a solid-blocked shadow ray is re-tested exactly (default 0.25). */
  alphaRoulette?: number;
  /** `alphaShadow: 'firstK'`: number of cutout hits tested exactly (default 1). */
  alphaFirstK?: number;
  /** `alphaShadow: 'coverage' | 'stochastic'`: per-material fraction of the card that is solid (1 for a non-cutout material), indexed by material id (bench: `alphaCoverageOf`). */
  alphaCoverage?: readonly number[];
  /** `alphaShadow: 'transmit'`: visibility of a shadow ray blocked only by cutout cards (default 0.5). */
  alphaTransmit?: number;
  /** `alphaShadow: 'classified'`: the bvh instance mask bit of the mixed cards (default 0x04); all other instances are solid. */
  alphaMixedMask?: number;
  /**
   * OpenPBR mode (P2.21, P2.22, P2.30): the scene has `openpbr` materials. Selects the OpenPBR material adapter (`ptBsdfFromOpenPbr`) and compiles
   * only the lobes `features` names: stochastic metal/dielectric mix, thin-walled dielectric, layered dielectric-over-diffuse
   * (`layered.layering` albedo | stochastic) and the absorbing interior medium (path state: Beer's law over the distance to the exit
   * hit; inside/outside from the hit side, no nesting). Default null: the legacy adapter and the unchanged kernel text.
   */
  materials?: PtMaterialOptions | null;
  /**
   * L3 path hooks (R10, spec section 8): user WGSL `ptOnVertex(v: PtPathVertex) -> u32` and `ptTerminate(v: PtPathVertex) -> vec3f`, called by both
   * schedulers at every scattering vertex after emission and next-event estimation (`outputs-wgsl.ts`; `docs/internal/design/integrator.md`). A hook that returns
   * `PT_HOOK_TERMINATE` ends the path and adds `throughput * ptTerminate(v)`. Default null: no hook code at all (the kernel text is unchanged).
   */
  hooks?: PtHooksOptions | null;
  /** Per-pixel debug counters (R4): depth, rays, termination reason, in one storage buffer (`readCounters`). Never in a timed run. Default false: the kernel text is unchanged. */
  counters?: boolean;
  /** First non-specular albedo + normal and the primary viewZ, accumulated like radiance in the tail of the accumulation buffer (`readAovs`). Default false: the kernel text is unchanged. */
  aovs?: boolean;
  /**
   * Where the primary hit comes from (P3.6). `'trace'` (default): a camera ray, traced; no V-buffer code at all (the kernel text is unchanged). `'vbuffer'`: the engine
   * rasterised a visibility buffer (`vbuffer.ts`: instance + triangle + material + barycentrics per pixel) and the kernel resolves the primary hit from it; the engine owns
   * the sub-pixel jitter (`setVbufferJitter`), the camera-ray sampler pair is not consumed, bounces trace as usual. Needs `layout.positions`; the wavefront scheduler needs
   * `refill: false`.
   */
  primary?: PtPrimary;
  /**
   * R1, ray-class material LOD (a real-time setting, BIASED: never for reference renders). `'full'` (default): every vertex evaluates the full
   * material and the kernel text is unchanged. `'secondary'`: vertices at depth >= `materialLodDepth` (the hits of indirect rays) evaluate a reduced
   * material (`ptBsdfReduced`, adapter.ts): the layered lobe / coat becomes a stochastic GGX-or-Lambert choice with the same directional albedo, a coated
   * metal a plain conductor with the stack's reflectance, anisotropy its mean alpha, and fuzz and thin film are skipped; texture footprints
   * are widened by `PT_LOD_TEX_BIAS` when `textureLod` is `'cone'`. Needs OpenPBR mode (`materials`) to change a BSDF (without it only the texture bias applies).
   * The megakernel branches at run time (all lobes stay compiled for depth < N); the wavefront compiles a second, smaller shade kernel for depth >= N.
   */
  materialLod?: 'full' | 'secondary';
  /** First scattering depth that uses the reduced material (`materialLod: 'secondary'`). Default 1 (the hits of indirect rays); 0 reduces every vertex (debug / furnace checks). */
  materialLodDepth?: number;
  /**
   * ReSTIR DI at the first scattering vertex (P4, real-time mode; both schedulers, the wavefront without refill / layers; restir/restir-wgsl.ts). Setting it compiles the shade side here (read the final
   * reservoir, regenerate the light sample, one shadow ray; no MIS weight on emission after that vertex) and makes `PtMegakernel` run the G-buffer / candidate /
   * temporal / spatial passes before every sample dispatch. Default null: no ReSTIR code at all (the kernel text is unchanged). Needs `nee`, no hooks; textures and OpenPBR materials work, except anisotropy, fuzz, thin film and
   * stochastic layering (`ptRestirUnsupported` says which).
   */
  restir?: PtRestirOptions | null;
  /**
   * Participating media (P5, both schedulers; docs/internal/design/volumes.md): homogeneous and dense-grid media bounded by closed meshes, delta tracking for free flight,
   * Henyey-Greenstein scattering with next-event estimation and MIS, transmittance on shadow rays, medium-boundary ("interface") surfaces that rays pass through.
   * Default null: no volume code at all (the kernel text is unchanged). Cannot be combined with `restir`, `frame`, `primary: 'vbuffer'`, `alphaMask`, hooks or the
   * `materials.features.medium` absorber (`materials` may be set for other lobes).
   */
  volumes?: PtVolumeOptions | null;
  /**
   * A floor (metres) for the self-intersection offset of every secondary ray origin, added along the side normal before the ulp offset (Waechter and Binder).
   * The ulp offset alone assumes the hit point is accurate to a few ulps of ITSELF; a point near the origin on a large triangle (a 2 km ground under the camera)
   * carries the error of its far vertices instead (~2e-7 x their magnitude), so bounce and shadow rays re-hit the surface from behind (P5.v17: the ground
   * under an opaque box received 100x the sky light pbrt does). Set it to about 2^-19 x the scene's largest coordinate. Default 0: off, the kernel text is unchanged.
   */
  originOffset?: number;
  /**
   * Spectral transport (P5, `docs/internal/design/spectral.md`): paths carry three wavelengths (hero sampling) in the lanes of the vec3 throughput; every RGB reflectance,
   * conductor eta / k, transmission / coat / sheen colour is uplifted to its sigmoid-polynomial spectrum (Jakob and Hanika 2019, as pbrt-v4), emission and the
   * environment / sun to the same times D65, and the sample is converted to linear Rec.709 by the CIE 1931 matching functions. Opt-in (default off: the kernel text is
   * unchanged). Needs the table texture (`PtMegakernelOptions.spectralTable`, group 2 binding `PT_SPEC_BINDING`); both schedulers; no hooks / frame / AOVs / ReSTIR.
   * `true` = `{ res: 64, dispersion: true }`; with dispersion a dielectric with `RtMaterial.dispersion > 0` follows a Cauchy law and keeps only the hero wavelength.
   */
  spectral?: boolean | PtSpectralOptions;
  /**
   * P5, MaterialX approach (b): per-material WGSL generated from MaterialX graphs (`materialx/wgsl.ts`, `mtlxProgramsWgsl`). `wgsl` is the generated text (`ptMtlxApply`, and
   * `ptMtlxNormal` when `normal`); the kernel calls `ptMtlxApply` after the texture lookups of the hit and `ptMtlxNormal` for the shading normal. Needs `textures` (uv,
   * position, tangent, texture lookups). Default null: no MaterialX code, the kernel text is unchanged. Alpha cut-outs and shadow rays do not see a generated opacity.
   */
  materialx?: { wgsl: string; normal?: boolean } | null;
  /** @internal Set by the wavefront builder on its secondary shade kernel: every vertex it shades is reduced (no run-time branch). */
  lodKernel?: 'secondary';
}

export type PtPrimary = 'trace' | 'vbuffer';

/** R1: factor on the texture footprint (`fp`) of a reduced vertex with `textureLod: 'cone'`: 2 = one mip level coarser. */
export const PT_LOD_TEX_BIAS = '2.0';

/**
 * R1: the lines `ptPath` (megakernel) and `ptwfShadePath` (wavefront) share to reduce a vertex. `expr` is the run-time condition (`true` in the wavefront's
 * secondary kernel). Every field is '' when `o.lod` is 'off', so the default kernel text does not change.
 */
export function ptLodSnippets(o: ResolvedIntegratorOptions, indent: string): { decl: string; texBias: string; bsdf: (mixU: string, shadingNormal: string, hasMats: boolean) => string | null; and: string; fuzzW: string } {
  if (o.lod === 'off') return { decl: '', texBias: '', bsdf: () => null, and: '', fuzzW: 'mat.sheenWeight' };
  const always = o.lod === 'always';
  const mats = !!o.materials;
  return {
    decl: `${indent}let lod = ${always ? 'true' : 'depth >= PT_LOD_DEPTH'};\n`,
    texBias: o.textures ? ' * select(1.0, PT_LOD_TEX_BIAS, lod)' : '',
    bsdf: (mixU, shadingNormal, hasMats) => {
      if (!(mats && hasMats)) return null;
      const reduced = `ptBsdfReduced(mat, ${mixU}, abs(dot(${shadingNormal}, -ray.direction)))`;
      return always ? `${indent}let bp = ${reduced};\n` : `${indent}var bp: PtBsdfParams;\n${indent}if (lod) {\n${indent}    bp = ${reduced};\n${indent}} else {\n${indent}    bp = ptBsdfFromMaterial(mat, ${mixU});\n${indent}}\n`;
    },
    and: ' && !lod',
    fuzzW: 'select(mat.sheenWeight, 0.0, lod)',
  };
}

export type PtAlphaPolicy = 'exact' | 'opaqueShadow' | 'opaqueIndirect' | 'opaqueSecondary';
export type PtAlphaShadow = 'policy' | 'roulette' | 'coverage' | 'stochastic' | 'firstK' | 'transmit' | 'classified';
/** Resolved shadow mode of alpha cutouts: `exact` (closest-hit loop), `solid` (any-hit walk), or one of the R2b modes. */
export type PtAlphaShadowMode = 'exact' | 'solid' | Exclude<PtAlphaShadow, 'policy'>;

export interface ResolvedIntegratorOptions {
  sampler: PtSamplerKind;
  blueNoise: PtBlueNoiseOptions;
  frame: boolean;
  /** `frame`'s plane 6: `'none'` without `frame` (see `PtIntegratorOptions.frameEmission`). */
  frameEmission: 'none' | 'primary' | 'mirror';
  accumulate: boolean;
  nee: boolean;
  envSampling: 'uniform' | 'importance' | 'none';
  envMap: boolean;
  sunVisible: boolean;
  textures: boolean;
  textureLod: 'cone' | 0;
  alphaMask: boolean;
  /** Always 'exact' without `alphaMask`. */
  alphaPolicy: PtAlphaPolicy;
  /** Shadow handling of cutouts (R2b). Always 'exact' without `alphaMask`. */
  alphaShadow: PtAlphaShadowMode;
  alphaRoulette: number;
  alphaFirstK: number;
  alphaCoverage: readonly number[];
  alphaMixedMask: number;
  alphaTransmit: number;
  materials: PtMaterialOptions | null;
  hooks: PtHooksOptions | null;
  counters: boolean;
  aovs: boolean;
  primary: PtPrimary;
  /** R1: `'off'` (full material everywhere: no code changes), `'branch'` (megakernel and wavefront primary kernel: `depth >= lodDepth` at run time), `'always'` (wavefront secondary shade kernel). */
  lod: 'off' | 'branch' | 'always';
  lodDepth: number;
  /** ReSTIR DI at the first vertex (`PtIntegratorOptions.restir`). */
  restir: boolean;
  /** ReSTIR with `sharedHit` and trace-mode primaries: the shade kernel resolves its primary hit from the V-buffer texel the G-buffer pass wrote (P4.2c). */
  restirShared: boolean;
  /** P5 participating media (`PtIntegratorOptions.volumes`). */
  volumes: PtVolumeOptions | null;
  /** `PtIntegratorOptions.originOffset` (0 = off). */
  originOffset: number;
  /** Spectral transport (P5), or null. */
  spectral: ResolvedSpectralOptions | null;
  /** ReSTIR GI on top of DI (`PtRestirOptions.gi`, P5): the shade ends the path at the first non-delta vertex and adds the final GI reservoir's contribution. */
  restirGi: boolean;
  /** MaterialX codegen (`PtIntegratorOptions.materialx`) is compiled in. */
  mtlx: boolean;
  /** ... and some material has a generated normal map. */
  mtlxNormal: boolean;
}

export function resolveIntegratorOptions(o: PtIntegratorOptions = {}): ResolvedIntegratorOptions {
  const alphaMask = (o.alphaMask ?? false) && (o.textures ?? false);
  const alphaPolicy: PtAlphaPolicy = alphaMask ? o.alphaPolicy ?? 'exact' : 'exact';
  const override = o.alphaShadow ?? 'policy';
  const alphaShadow: PtAlphaShadowMode = !alphaMask ? 'exact' : override !== 'policy' ? override : alphaPolicy === 'exact' || alphaPolicy === 'opaqueIndirect' ? 'exact' : 'solid';
  const alphaRoulette = o.alphaRoulette ?? 0.25;
  if (alphaShadow === 'roulette' && !(alphaRoulette > 0 && alphaRoulette <= 1)) throw new RangeError(`integrator.alphaRoulette must be in (0, 1], got ${alphaRoulette}`);
  const alphaFirstK = Math.floor(o.alphaFirstK ?? 1);
  if (alphaShadow === 'firstK' && !(alphaFirstK >= 1 && alphaFirstK <= 32)) throw new RangeError(`integrator.alphaFirstK must be in 1..32, got ${alphaFirstK}`);
  const alphaCoverage = o.alphaCoverage ?? [];
  if ((alphaShadow === 'coverage' || alphaShadow === 'stochastic') && alphaCoverage.length === 0) throw new RangeError(`integrator.alphaShadow '${alphaShadow}' needs alphaCoverage (one fraction per material)`);
  if (o.volumes) {
    const clash = o.restir ? 'restir' : o.frame ? 'frame' : (o.primary ?? 'trace') === 'vbuffer' ? "primary 'vbuffer'" : alphaMask ? 'alphaMask' : o.hooks ? 'hooks' : o.materials?.features.medium ? 'the Beer-law absorber (materials.features.medium)' : o.counters ? 'counters' : o.aovs ? 'aovs' : '';
    if (clash) throw new Error(`integrator.volumes cannot be combined with ${clash} yet`);
    if (o.volumes.outputs && o.accumulate === false) throw new Error('integrator.volumes.outputs needs accumulate (the outputs are sums over samples)');
  }
  const alphaMixedMask = o.alphaMixedMask ?? 0x04;
  const alphaTransmit = o.alphaTransmit ?? 0.5;
  if (alphaShadow === 'transmit' && !(alphaTransmit >= 0 && alphaTransmit <= 1)) throw new RangeError(`integrator.alphaTransmit must be in [0, 1], got ${alphaTransmit}`);
  const spectral = resolveSpectralOptions(o.spectral);
  if (spectral && (o.hooks || o.frame || o.aovs || o.restir)) throw new Error('integrator.spectral does not combine with hooks, frame, aovs or restir (their RGB planes / callbacks are not wavelength aware)');
  if (spectral && o.volumes) throw new Error('integrator.spectral does not combine with volumes yet (media sigma_a / sigma_s are RGB and the delta-tracking lanes are not wavelength aware)');
  return { sampler: o.sampler ?? 'owen-sobol', blueNoise: { ...(o.blueNoisePairs !== undefined ? { pairs: o.blueNoisePairs } : {}), ...(o.blueNoiseShift ? { shift: o.blueNoiseShift } : {}) }, frame: o.frame ?? false, frameEmission: o.frame ? (o.frameEmission ?? 'mirror') : 'none', accumulate: o.accumulate ?? !(o.frame ?? false), nee: o.nee ?? true, envSampling: o.envSampling ?? 'uniform', envMap: o.envMap ?? false, sunVisible: o.sunVisible ?? false, textures: o.textures ?? false, textureLod: o.textureLod ?? 0, alphaMask, alphaPolicy, alphaShadow, alphaRoulette, alphaFirstK, alphaCoverage, alphaMixedMask, alphaTransmit, materials: o.materials ?? null, hooks: o.hooks ?? null, counters: o.counters ?? false, aovs: o.aovs ?? false, primary: o.primary ?? 'trace', lod: (o.materialLod ?? 'full') === 'secondary' ? (o.lodKernel === 'secondary' ? 'always' : 'branch') : 'off', lodDepth: Math.max(0, Math.floor(o.materialLodDepth ?? 1)), restir: !!o.restir, restirShared: !!(o.restir && o.restir.sharedHit && (o.primary ?? 'trace') === 'trace'), restirGi: !!(o.restir && o.restir.gi), volumes: o.volumes ?? null, originOffset: ptOriginOffset(o.originOffset), spectral, mtlx: !!o.materialx && (o.textures ?? false), mtlxNormal: !!o.materialx?.normal && (o.textures ?? false) };
}

/** Shadow rays of this kernel honour cutouts (`ptVisibleMasked`); false: plain any-hit `rtVisible`. Only meaningful with `alphaMask`. */
export const ptMaskShadow = (o: ResolvedIntegratorOptions): boolean => o.alphaMask && o.alphaShadow === 'exact';
/** Shadow rays use the R2b weighted visibility `ptShadowWeight` (a weight in [0, 1/q], not a bool). */
export const ptShadowWeighted = (o: ResolvedIntegratorOptions): boolean => o.alphaMask && o.alphaShadow !== 'exact' && o.alphaShadow !== 'solid';
/** The shadow kernel / path needs the scene copy (material and texture fetches) for its alpha handling. */
export const ptShadowNeedsScene = (o: ResolvedIntegratorOptions): boolean => ptMaskShadow(o) || (ptShadowWeighted(o) && o.alphaShadow !== 'coverage' && o.alphaShadow !== 'stochastic' && o.alphaShadow !== 'transmit');
/** Extension rays after the first scattering vertex honour cutouts. Camera rays always do. */
export const ptMaskIndirect = (o: ResolvedIntegratorOptions): boolean => o.alphaMask && (o.alphaPolicy === 'exact' || o.alphaPolicy === 'opaqueShadow');

/** Bytes of one `ptParams` record (5 vec4 + 1 vec4u pair). */
export const PT_PARAMS_BYTES = 96;
export const PT_WORKGROUP = [8, 8] as const;

export const PT_PARAMS_WGSL = /* wgsl */ `
struct PtParams {
    dims: vec4u,
    limits: vec4u,
    origin: vec4f,
    right: vec4f,
    up: vec4f,
    forward: vec4f,
}
`;

/**
 * The pieces of the integrator the wavefront scheduler (`../wavefront/`) shares with the megakernel, so both compile the same text:
 * constants, ray-origin offset, MIS weight and the escaped-ray (environment / sun) term. Moving them here left the megakernel's text byte-identical.
 */
function ptOriginOffset(v: number | undefined): number {
  if (v === undefined || v === 0) return 0;
  if (!(v > 0 && Number.isFinite(v))) throw new RangeError(`integrator.originOffset must be a finite number >= 0 (metres), got ${v}`);
  return Math.fround(v);
}

export function ptCommonHeadWgsl(o: ResolvedIntegratorOptions): string {
  const tex = o.textures;
  const mask = o.alphaMask;
  return /* wgsl */ `const PT_FAR: f32 = 1.0e30;
const PT_NEE: bool = ${o.nee};
const PT_ENV_SAMPLING: bool = ${o.envSampling !== 'none'};
const PT_SUN_VISIBLE: bool = ${o.sunVisible};
const PT_ORIGIN: f32 = 0.03125;
const PT_FLOAT_SCALE: f32 = 0.0000152587890625;
const PT_INT_SCALE: f32 = 256.0;
${mask ? `const PT_MASK_SKIPS: u32 = 64u;
` : ''}${tex ? `const PT_TEX_CONE: bool = ${o.textureLod === 'cone'};
const PT_CONE_DIFFUSE_SPREAD: f32 = 0.5;
` : ''}${o.lod !== 'off' ? `const PT_LOD_DEPTH: u32 = ${o.lodDepth}u;
${tex ? `const PT_LOD_TEX_BIAS: f32 = ${PT_LOD_TEX_BIAS};
` : ''}` : ''}
${o.originOffset > 0 ? `const PT_ORIGIN_FLOOR: f32 = ${o.originOffset};

fn ptOffsetOrigin(p0: vec3f, n: vec3f) -> vec3f {
    let p = p0 + PT_ORIGIN_FLOOR * n;` : `fn ptOffsetOrigin(p: vec3f, n: vec3f) -> vec3f {`}
    let ofs = vec3i(PT_INT_SCALE * n);
    let pi = bitcast<vec3f>(bitcast<vec3i>(p) + select(ofs, -ofs, p < vec3f(0.0)));
    return select(pi, p + PT_FLOAT_SCALE * n, abs(p) < vec3f(PT_ORIGIN));
}

fn ptSideNormal(n: vec3f, w: vec3f) -> vec3f {
    return select(-n, n, dot(n, w) > 0.0);
}`;
}

export const PT_MIS_WGSL = /* wgsl */ `fn ptMis(pThis: f32, pOther: f32) -> f32 {
    if (!(pThis > 0.0)) {
        return 0.0;
    }
    let r = min(pOther / pThis, 1.0e18);
    return 1.0 / (1.0 + r * r);
}

fn ptMax3(v: vec3f) -> f32 {
    return max(v.x, max(v.y, v.z));
}`;

export const PT_ESCAPED_WGSL = /* wgsl */ `fn ptEscaped(dir: vec3f, direct: bool, prevPdf: f32) -> vec3f {
    var le = vec3f(0.0);
    if (rtHasEnvironment()) {
        let e = rtEnvironment();
        var w = 1.0;
        if (!direct && PT_NEE && PT_ENV_SAMPLING) {
            w = ptMis(prevPdf, e.pmf * rtEnvironmentPdf(e, dir));
        }
        le += rtEnvironmentLe(e, dir) * w;
    }
    if (rtHasSun()) {
        let s = rtSun();
        if (rtSunContains(s, dir) && (PT_SUN_VISIBLE || !direct)) {
            var w = 1.0;
            if (!direct && PT_NEE) {
                w = ptMis(prevPdf, s.pmf / s.solidAngle);
            }
            le += rtSunRadiance(s) * w;
        }
    }
    return le;
}`;

/** Shadow-ray visibility through cutouts (alpha MASK): closest-hit loop that steps over masked-out texels. Shared by the megakernel and the wavefront shadow kernel. */
export const PT_VISIBLE_MASKED_WGSL = /* wgsl */ `fn ptMaskedOut(hit: RtHit, ray: RtRay) -> bool {
    let m = rtMaterial(hit.material);
    if (m.alphaMode != RT_ALPHA_MASK) {
        return false;
    }
    let s = rtSurface(hit, ray, RT_SURF_UV);
    let a = rtApplyMaterialTextures(m, s.uv, 0.0);
    return a.opacity < a.alphaCutoff;
}

fn ptVisibleMasked(ray0: RtRay, mask: u32) -> bool {
    var ray = ray0;
    for (var i = 0u; i < PT_MASK_SKIPS; i++) {
        let hit = rtClosest(ray, mask);
        if (hit.t < 0.0) {
            return true;
        }
        if (!ptMaskedOut(hit, ray)) {
            return false;
        }
        let p = ptOffsetOrigin(ray.origin + ray.direction * hit.t, ptSideNormal(hit.normal, ray.direction));
        ray = RtRay(p, 0.0, ray.direction, ray.tMax - length(p - ray.origin));
        if (ray.tMax <= 0.0) {
            return true;
        }
    }
    return false;
}

`;

const f32Lit = (v: number): string => (Number.isInteger(v) ? `${v}.0` : String(Math.fround(v)));
const hex32 = (v: number): string => `0x${(v >>> 0).toString(16)}u`;

/**
 * R2b: weighted shadow-ray visibility for cutouts, `ptShadowWeight(ray, h) -> f32` (0 blocked, 1 visible, 1/q for a roulette survivor; the caller multiplies the
 * light contribution by it). `h` seeds the per-ray random number (`ptAlphaRand`). Shared by the megakernel's NEE and the wavefront shadow kernel; '' unless
 * `ptShadowWeighted(o)`, so every other kernel is unchanged. Needs `PT_VISIBLE_MASKED_WGSL` for the modes that test the texture (`ptShadowNeedsScene`).
 */
export function ptShadowWeightWgsl(o: ResolvedIntegratorOptions): string {
  if (!ptShadowWeighted(o)) return '';
  const rand = `// per-ray uniform number in [0, 1) (pcg3d, Jarzynski and Olano 2020)
fn ptAlphaRand(a: u32, b: u32, c: u32) -> f32 {
    var v = vec3u(a, b, c) * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v ^= v >> vec3u(16u);
    v.x += v.y * v.z;
    return f32(v.x >> 8u) * 5.9604644775390625e-8;
}

`;
  const advance = `        let p = ptOffsetOrigin(ray.origin + ray.direction * hit.t, ptSideNormal(hit.normal, ray.direction));
        ray = RtRay(p, 0.0, ray.direction, ray.tMax - length(p - ray.origin));
        if (ray.tMax <= 0.0) {
            return 1.0;
        }
`;
  const n = o.alphaCoverage.length;
  const coverageTable = (): string => `var<private> PT_ALPHA_COVERAGE: array<f32, ${n}> = array<f32, ${n}>(${o.alphaCoverage.map((c) => f32Lit(Math.min(1, Math.max(0, c)))).join(', ')});
fn ptAlphaCoverage(material: u32) -> f32 {
    return select(1.0, PT_ALPHA_COVERAGE[min(material, ${n - 1}u)], material < ${n}u);
}

`;
  switch (o.alphaShadow) {
    case 'roulette':
      return `${rand}const PT_ALPHA_Q: f32 = ${f32Lit(o.alphaRoulette)};
fn ptShadowWeight(ray: RtRay, h: vec3u) -> f32 {
    if (rtVisible(ray, RT_MASK_ALL)) {
        return 1.0;
    }
    if (ptAlphaRand(h.x, h.y, h.z) >= PT_ALPHA_Q) {
        return 0.0;
    }
    return select(0.0, 1.0 / PT_ALPHA_Q, ptVisibleMasked(ray, RT_MASK_ALL));
}

`;
    case 'coverage':
      return `${coverageTable()}fn ptShadowWeight(ray: RtRay, h: vec3u) -> f32 {
    let hit = rtClosest(ray, RT_MASK_ALL);
    if (hit.t < 0.0) {
        return 1.0;
    }
    return 1.0 - ptAlphaCoverage(hit.material);
}

`;
    case 'stochastic':
      return `${rand}${coverageTable()}fn ptShadowWeight(ray0: RtRay, h: vec3u) -> f32 {
    var ray = ray0;
    for (var i = 0u; i < PT_MASK_SKIPS; i++) {
        let hit = rtClosest(ray, RT_MASK_ALL);
        if (hit.t < 0.0) {
            return 1.0;
        }
        let c = ptAlphaCoverage(hit.material);
        if (c >= 1.0 || ptAlphaRand(h.x, h.y, h.z + i) < c) {
            return 0.0;
        }
${advance}    }
    return 0.0;
}

`;
    case 'firstK':
      return `const PT_ALPHA_K: u32 = ${o.alphaFirstK}u;
fn ptShadowWeight(ray0: RtRay, h: vec3u) -> f32 {
    var ray = ray0;
    for (var i = 0u; i < PT_ALPHA_K; i++) {
        let hit = rtClosest(ray, RT_MASK_ALL);
        if (hit.t < 0.0) {
            return 1.0;
        }
        if (!ptMaskedOut(hit, ray)) {
            return 0.0;
        }
${advance}    }
    return select(0.0, 1.0, rtVisible(ray, RT_MASK_ALL));
}

`;
    case 'transmit':
      return `const PT_ALPHA_CARDS: u32 = ${hex32(o.alphaMixedMask)};
const PT_ALPHA_BETA: f32 = ${f32Lit(o.alphaTransmit)};
fn ptShadowWeight(ray: RtRay, h: vec3u) -> f32 {
    if (!rtVisible(ray, RT_MASK_ALL & ~PT_ALPHA_CARDS)) {
        return 0.0;
    }
    return select(1.0, PT_ALPHA_BETA, !rtVisible(ray, PT_ALPHA_CARDS));
}

`;
    case 'classified':
      return `const PT_ALPHA_MIXED: u32 = ${hex32(o.alphaMixedMask)};
fn ptShadowWeight(ray: RtRay, h: vec3u) -> f32 {
    if (!rtVisible(ray, RT_MASK_ALL & ~PT_ALPHA_MIXED)) {
        return 0.0;
    }
    return select(0.0, 1.0, ptVisibleMasked(ray, PT_ALPHA_MIXED));
}

`;
    default:
      return '';
  }
}

/** The integrator module text, to append after the raytracer module, the light sampling, the Owen-Sobol functions, the BSDF, its adapter and the sampler. */
export function ptIntegratorWgsl(options: PtIntegratorOptions = {}): string {
  const o = resolveIntegratorOptions(options);
  const spec = o.spectral;
  const tex = o.textures;
  const mask = o.alphaMask;
  const maskShadow = ptMaskShadow(o);
  const maskIndirect = ptMaskIndirect(o);
  const shadowWeighted = ptShadowWeighted(o);
  const shadowWeightedScene = shadowWeighted && ptShadowNeedsScene(o);
  const shadingNormal = tex ? 'shadingN' : 'surf.shadingNormal';
  const mats = o.materials;
  const layered = !!mats?.features.layered;
  const stochastic = layered && (mats?.layered?.layering ?? 'albedo') === 'stochastic';
  const medium = !!mats?.features.medium;
  const vol = !!o.volumes;
  const volOut = vol && !!o.volumes!.outputs;
  const aniso = !!mats?.features.aniso;
  const fuzz = !!mats?.features.fuzz;
  const film = !!mats?.features.thinFilm;
  const evalFn = fuzz ? 'ptFuzzedEval' : layered ? 'ptEval' : 'ptBsdfEval';
  const sampleFn = fuzz ? 'ptFuzzedSample' : layered ? 'ptSample' : 'ptBsdfSample';
  const lodS = ptLodSnippets(o, '        ');
  const mixU = mats?.features.mix || (o.lod !== 'off' && mats) ? 'ptSample2D(pixelSeed, sampleIndex, ptVertexPair(depth, PT_SLOT_RR)).y' : '0.0';
  const hooks = o.hooks;
  // The optional radiance callback (`ptOnRadiance`): call sites only when the hook text defines it, so every other kernel keeps its exact text.
  const onRadiance = ptHookWantsRadiance(o.hooks);
  const addRadiance = (value: string, index: string, indent: string): string =>
    onRadiance ? `${indent}{\n${indent}    let ptAdded = ${value};\n${indent}    radiance += ptAdded;\n${indent}    ptOnRadiance(pxIndex, sampleIndex, ${index}, ptHookFlags, ptAdded);\n${indent}}\n` : `${indent}radiance += ${value};\n`;
  const counters = o.counters;
  const aovs = o.aovs;
  const restir = o.restir;
  const gi = o.restirGi;
  const frame = o.frame;
  // P4.5a: light the primary BSDF does not modulate (an emitter or the sky seen directly, or through perfect mirrors) goes to `fEmit` instead of `radiance`; the beauty is their sum.
  const emit = o.frameEmission !== 'none';
  const emitMirror = o.frameEmission === 'mirror';
  const pureCond = emitMirror ? 'fPure' : 'depth == 0u';
  const addPure = (value: string, index: string, indent: string): string =>
    !emit
      ? addRadiance(value, index, indent)
      : onRadiance
        ? `${indent}{\n${indent}    let ptAdded = ${value};\n${indent}    if (${pureCond}) {\n${indent}        fEmit += ptAdded;\n${indent}    } else {\n${indent}        radiance += ptAdded;\n${indent}    }\n${indent}    ptOnRadiance(pxIndex, sampleIndex, ${index}, ptHookFlags${o.hooks?.radianceEmissionFlag ? ` | select(0u, PT_HOOK_RADIANCE_EMISSION, ${pureCond})` : ''}, ptAdded);\n${indent}}\n`
        : `${indent}if (${pureCond}) {\n${indent}    fEmit += ${value};\n${indent}} else {\n${indent}    radiance += ${value};\n${indent}}\n`;
  const bn = o.sampler === 'bluenoise';
  const vb = o.primary === 'vbuffer';
  // The hit comes from a V-buffer texel: the engine's texture (`vb`) or the one the ReSTIR G-buffer pass wrote (`restir.sharedHit`).
  const vbHit = vb || o.restirShared;
  const black = fuzz ? '(bp.kind == PT_BSDF_BLACK && !ptFuzzActive(1.0))' : 'bp.kind == PT_BSDF_BLACK';
  const rrMedium = `                if (rrStart > 0u && depth >= rrStart) {
                    let rrBeta = beta * etaScale;
                    let m = ptMax3(rrBeta);
                    if (m < 1.0) {
                        let q = max(0.0, 1.0 - m);
                        if (ptSample2D(pixelSeed, sampleIndex, ptVertexPair(vertex, PT_SLOT_RR)).x < q) {
                            break;
                        }
                        beta = beta / (1.0 - q);
                    }
                }
`;
  // P5: the free-flight segment of the ray in the current medium (before the hit it ends at is looked at), and the medium-boundary handling at that hit.
  const volSegment = vol
    ? `        if (curMed != 0u) {
            let segMax = select(PT_FAR, hit.t, hit.t >= 0.0);
            let ev = ptVolSample(curMed, ray.origin, ray.direction, segMax, &beta);
            if (ev.kind == 2u) {
                break;
            }
            if (ev.kind == 1u) {
                if (depth >= maxDepth) {
                    break;
                }
                let vertex = depth;
${volOut ? `                ptVoCapture(depth, radiance);
` : ''}                depth += 1u;
                let pm = ray.origin + ray.direction * ev.t;
${volOut ? `                if (vertex == 0u && ptVoHas == 0.0) {
                    ptVoDist = length(pm - origin);
                    ptVoHas = 1.0;
                }
` : ''}                let woM = -ray.direction;
                if (PT_NEE) {
                    radiance += beta * ptSampleLdMedium(pm, woM, curMed, pixelSeed, sampleIndex, vertex);
                }
${ptVolPhaseSampleWgsl(o.volumes!, '                ', 'curMed', 'woM', 'ptSample2D(pixelSeed, sampleIndex, ptVertexPair(vertex, PT_SLOT_BSDF_DIRECTION))', 'wiM', 'prevPdf')}                specularBounce = false;
                prevP = pm;
                ray = RtRay(pm, 0.0, wiM, PT_FAR);
${rrMedium}                continue;
            }
        }
`
    : '';
  const volSurface = vol
    ? `        let mv = ptVolMat(hit.material);
        let medIn = select(curMed, mv.x, mv.z != 0u);
        let medOut = select(curMed, mv.y, mv.z != 0u);
        if (mv.w != 0u) {
            ifaceSkips += 1u;
            if (ifaceSkips > 256u) {
                break;
            }
            curMed = select(medOut, medIn, dot(ray.direction, hit.normal) < 0.0);
            ray = RtRay(ptOffsetOrigin(ray.origin + ray.direction * hit.t, ptSideNormal(hit.normal, ray.direction)), 0.0, ray.direction, PT_FAR);
            continue;
        }
`
    : '';
  const term = (reason: number): string => (counters ? `            termReason = ${reason}u;\n` : '');
  return /* wgsl */ `
alias PtRadiance = vec4f;
${PT_PARAMS_WGSL}
@group(2) @binding(0) var<uniform> ptParams: PtParams;
@group(2) @binding(1) var<storage, read_write> ptAccum: array<vec4f>;
${restir ? `@group(2) @binding(${PT_RESTIR_SHADE_BINDING}) var<storage, read> ptRestirRes: array<vec4u>;
` : ''}${gi ? `@group(2) @binding(${PT_RESTIR_GI_SHADE_BINDING}) var<storage, read> ptRestirGiRes: array<vec4u>;
` : ''}${o.restirShared ? `@group(2) @binding(${PT_RESTIR_HIT_BINDING}) var<storage, read> ptRestirHit: array<vec4u>;
` : ''}
${spec ? `${ptSpectralWgsl(spec)}\n` : ''}${ptCommonHeadWgsl(o)}

${maskShadow || shadowWeightedScene || (o.restir && mask) ? PT_VISIBLE_MASKED_WGSL : ''}${shadowWeighted ? ptShadowWeightWgsl(o) : ''}${PT_MIS_WGSL}${vol ? ptVolumeWgsl(o.volumes!) : ''}
${restir ? `
${PT_RESTIR_UNPACK_WGSL}

fn ptRestirLd(p: vec3f, ng: vec3f, frame: PtFrame, wo: vec3f, bp: PtBsdfParams, pxIndex: u32) -> vec3f {
    let r = ptRestirRes[ptParams.limits.w + pxIndex];
    let rw = bitcast<f32>(r.z);
    if (r.x == RT_NONE || !(rw > 0.0)) {
        return vec3f(0.0);
    }
    let li = rtSampleLightLi(r.x, p, ptRsUnpackU(r.y));
    if (!li.valid || !(li.pdf > 0.0)) {
        return vec3f(0.0);
    }
    let wiLocal = ptToLocal(frame, li.wi);
    let f = ${evalFn}(bp, wo, wiLocal).f * abs(wiLocal.z);
    if (!(ptMax3(f) > 0.0)) {
        return vec3f(0.0);
    }
    let origin = ptOffsetOrigin(p, ptSideNormal(ng, li.wi));
    var ray: RtRay;
    if (li.distance < 1.0e30) {
        let end = ptOffsetOrigin(li.point, ptSideNormal(li.normal, -li.wi));
        ray = rtSegment(origin, end, 0.0, 0.0);
    } else {
        ray = RtRay(origin, 0.0, li.wi, PT_FAR);
    }
${counters ? `    ptCountRay(pxIndex, 0u, 1u);
` : ''}    if (!${mask ? 'ptVisibleMasked' : 'rtVisible'}(ray, RT_MASK_ALL)) {
        return vec3f(0.0);
    }
    return li.radiance * f * (rw / li.pdf);
}
` : ''}${gi ? `
${PT_RESTIR_OCT_WGSL}
${ptRestirGiShadeWgsl(evalFn)}` : ''}
fn ptSampleLd(p: vec3f, ng: vec3f, frame: PtFrame, wo: vec3f, bp: PtBsdfParams, pixelSeed: u32, sampleIndex: u32, vertex: u32${counters || restir ? ', pxIndex: u32' : ''}${vol ? ', medIn: u32, medOut: u32' : ''}) -> vec3f {
${restir ? `    if (vertex == 0u) {
        return ptRestirLd(p, ng, frame, wo, bp, pxIndex);
    }
` : ''}    let selectU = ptSample2D(pixelSeed, sampleIndex, ptVertexPair(vertex, PT_SLOT_SELECT_COMPONENT)).x;
    let pointU = ptSample2D(pixelSeed, sampleIndex, ptVertexPair(vertex, PT_SLOT_LIGHT_POINT));
    let pick = rtSampleLight(selectU);
    if (pick.index == RT_NONE || !(pick.pmf > 0.0)) {
        return vec3f(0.0);
    }
    if (!PT_ENV_SAMPLING && pick.index == rtSceneTable[2].z) {
        return vec3f(0.0);
    }
    let li = rtSampleLightLi(pick.index, p, pointU);
    if (!li.valid || !(li.pdf > 0.0)) {
        return vec3f(0.0);
    }
    let wiLocal = ptToLocal(frame, li.wi);
    let e = ${evalFn}(bp, wo, wiLocal);
    let f = e.f * abs(wiLocal.z);
    if (!(ptMax3(f) > 0.0)) {
        return vec3f(0.0);
    }
    let origin = ptOffsetOrigin(p, ptSideNormal(ng, li.wi));
    var ray: RtRay;
    if (li.distance < 1.0e30) {
        let end = ptOffsetOrigin(li.point, ptSideNormal(li.normal, -li.wi));
        ray = rtSegment(origin, end, 0.0, 0.0);
    } else {
        ray = RtRay(origin, 0.0, li.wi, PT_FAR);
    }
${counters ? `    ptCountRay(pxIndex, 0u, 1u);
` : ''}${vol ? `    let tr = ptVolTr(ray, select(medOut, medIn, dot(li.wi, ng) < 0.0));
    if (!(ptMax3(tr) > 0.0)) {
        return vec3f(0.0);
    }
` : shadowWeighted ? `    let vis = ptShadowWeight(ray, vec3u(pixelSeed, sampleIndex, vertex));
    if (!(vis > 0.0)) {
        return vec3f(0.0);
    }
` : `    if (!${maskShadow ? 'ptVisibleMasked' : 'rtVisible'}(ray, RT_MASK_ALL)) {
        return vec3f(0.0);
    }
`}    let pl = pick.pmf * li.pdf;
    var w = 1.0;
    if (!li.delta) {
        w = ptMis(pl, e.pdf);
    }
    return ${vol ? `${spec ? 'ptSpecIllum(li.radiance)' : 'li.radiance'} * f * tr * (w / pl)` : `${spec ? 'ptSpecIllum(li.radiance)' : 'li.radiance'} * f * (w ${shadowWeighted ? '* vis ' : ''}/ pl)`};
}

${spec ? PT_ESCAPED_WGSL.replace('le += rtEnvironmentLe(e, dir) * w;', 'le += ptSpecIllum(rtEnvironmentLe(e, dir)) * w;').replace('le += rtSunRadiance(s) * w;', 'le += ptSpecIllum(rtSunRadiance(s)) * w;') : PT_ESCAPED_WGSL}

fn ptPath(px: vec2u, sampleIndex: u32) -> PtRadiance {
${bn ? `    ptBnPixel = px;
` : ''}    let dims = ptParams.dims;
    let maxDepth = ptParams.limits.x;
    let rrStart = ptParams.limits.y;
    let pixelSeed = ptPixelSeed(px.x, px.y, dims.w);
${spec ? `    ptSpecBegin(ptSample2D(pixelSeed, sampleIndex, 1u).x);
` : ''}    let jitter = ${vb ? 'vec2f(ptParams.origin.w, ptParams.right.w)' : 'ptSample2D(pixelSeed, sampleIndex, PT_PAIR_JITTER)'};
    let ndc = vec2f(((f32(px.x) + jitter.x) / f32(dims.x)) * 2.0 - 1.0, 1.0 - ((f32(px.y) + jitter.y) / f32(dims.y)) * 2.0);
    let origin = ptParams.origin.xyz;
    var ray = RtRay(origin, 0.0, normalize(ptParams.forward.xyz + ptParams.right.xyz * ndc.x + ptParams.up.xyz * ndc.y), PT_FAR);

${counters || aovs || hooks || restir || frame ? `    let pxIndex = px.y * dims.x + px.x;
` : ''}${counters ? `    var termReason = 0u;
` : ''}${aovs ? `    var aovDone = false;
` : ''}${frame ? `${emit ? `    var fEmit = vec3f(0.0);
${emitMirror ? `    var fPure = true;
` : ''}` : `    var fDirect = vec3f(0.0);
`}    var fBounce = 0.0;
    var fSeen = false;
    var fIsSpec = false;
    var fNormal = vec3f(0.0);
    var fViewZ = 0.0;
    var fMotion = vec3f(0.0);
    var fRough = 0.0;
    var fAlbedo = vec3f(1.0);
    var fSpec = vec3f(1.0);
` : ''}    var radiance = vec3f(0.0);
${onRadiance ? `    let ptHookFlags = ptHookUpdateFlag(px.x, px.y, sampleIndex);
` : ''}    var beta = vec3f(1.0);
    var specularBounce = false;
    var depth = 0u;
    var etaScale = 1.0;
${mask ? `    var skips = 0u;
` : ''}    var prevP = origin;
    var prevPdf = 0.0;
${medium ? `    var sigma = vec3f(0.0);
` : ''}${vol ? `    var curMed = ${o.volumes!.cameraMedium ?? 0}u;
    var ifaceSkips = 0u;
    ptVolSeed(pixelSeed, sampleIndex);
${volOut ? `    ptVoBegin(ray, curMed, pixelSeed, sampleIndex);
` : ''}` : ''}${tex ? `    var cone = RtCone(0.0, bitcast<f32>(ptParams.limits.z) * 2.0 * length(ptParams.up.xyz) / (length(ptParams.forward.xyz) * f32(dims.y)));
` : ''}
${vbHit ? `    var primary = true;
` : ''}    loop {
${vbHit ? `        var hit: RtHit;
        if (primary) {
            primary = false;
            hit = ptVbufferHit(px, ray);${o.restirShared ? `
            if (hit.t >= 0.0) {
                // The G-buffer pass traced this hit along ITS jittered ray: continue along the ray through the hit point (the two draw the same sampler pair, but nothing may depend on it).
                let hp = rtSurface(hit, ray, RT_SURF_POSITION).position;
                ray = RtRay(origin, 0.0, normalize(hp - origin), PT_FAR);
                hit.t = length(hp - origin);
                hit.frontFace = dot(hit.normal, ray.direction) < 0.0;
            }` : ''}
        } else {
            hit = rtClosest(ray, RT_MASK_ALL);
${counters ? `            ptCountRay(pxIndex, 1u, 0u);
` : ''}        }
` : `        let hit = rtClosest(ray, RT_MASK_ALL);
${counters ? `        ptCountRay(pxIndex, 1u, 0u);
` : ''}`}${volSegment}        if (hit.t < 0.0) {
${addPure('beta * ptEscaped(ray.direction, depth == 0u || specularBounce, prevPdf)', 'depth', '            ')}            break;
        }
${frame ? `        if (depth == 1u && fBounce == 0.0) {
            fBounce = hit.t;
        }
` : ''}${volSurface}${lodS.decl}${tex ? `        let surf = rtSurface(hit, ray, RT_SURF_POSITION | RT_SURF_SHADING_NORMAL | RT_SURF_UV | RT_SURF_TEXFRAME${aniso ? ' | RT_SURF_TANGENT' : ''});
        let coneW = rtConeAt(cone, hit.t);
        let fp = select(0.0, coneW * surf.uvDensity / max(abs(dot(hit.normal, ray.direction)), 0.001)${lodS.texBias}, PT_TEX_CONE);
        let mat = ${spec ? 'ptSpecMaterial(' : ''}${o.mtlx ? 'ptMtlxApply(' : ''}rtApplyMaterialTextures(rtMaterial(hit.material), surf.uv, fp)${o.mtlx ? ', hit.material, surf.uv, surf.position, surf.shadingNormal, surf.tangent.xyz, fp)' : ''}${spec ? ', hit.material)' : ''};
        var shadingN = surf.shadingNormal;
        if (mat.normalTexture != RT_TEXTURE_NONE) {
            let nr = rtTexRecord(mat.normalTexture);
            shadingN = rtApplyNormalMap(surf.shadingNormal, surf.tangent, rtTexSampleRecord(nr, surf.uv, rtTexLodOf(nr, fp)).xyz);
        }${o.mtlxNormal ? `
        let mxN = ptMtlxNormal(hit.material, surf.uv, surf.position, surf.shadingNormal, surf.tangent.xyz, fp);
        if (mxN.w > 0.0) {
            shadingN = rtApplyNormalMap(surf.shadingNormal, surf.tangent, mxN.xyz);
        }` : ''}` : `        let surf = rtSurface(hit, ray, RT_SURF_POSITION | RT_SURF_SHADING_NORMAL${aniso ? ' | RT_SURF_TANGENT' : ''});
        let mat = ${spec ? 'ptSpecMaterial(rtMaterial(hit.material), hit.material)' : 'rtMaterial(hit.material)'};`}
${medium ? `        beta = beta * exp(-sigma * hit.t);
` : ''}${mask ? `        if (${maskIndirect ? '' : 'depth == 0u && '}mat.alphaMode == RT_ALPHA_MASK && mat.opacity < mat.alphaCutoff) {
            skips += 1u;
            if (skips > PT_MASK_SKIPS) {
${counters ? `                termReason = 6u;
` : ''}                break;
            }
            cone = RtCone(coneW, cone.spread);
            ray = RtRay(ptOffsetOrigin(surf.position, ptSideNormal(hit.normal, ray.direction)), 0.0, ray.direction, PT_FAR);
            continue;
        }
` : ''}
${spec?.dispersion ? `        if (ptSpecDisp) {
            ptSpecDisp = false;
            if (!ptSpecHero) {
                ptSpecHero = true;
                beta = beta * vec3f(3.0, 0.0, 0.0);
            }
        }
` : ''}${hooks ? `        var hookEmitted = vec3f(0.0);
` : ''}        if (mat.emissionLuminance > 0.0 && (hit.frontFace || mat.emissionTwoSided)) {
            var w = 1.0;
            if (!(depth == 0u || specularBounce) && PT_NEE) {
                let lightIndex = rtLightIndexOfHit(hit.instance, hit.triangle);
                if (lightIndex != RT_NONE) {
                    let light = rtLight(lightIndex);
                    let dist = length(surf.position - prevP);
                    let cosLight = abs(dot(hit.normal, ray.direction));
                    w = ptMis(prevPdf, light.pmf * rtTriangleSolidAnglePdf(light.area, dist, cosLight));
                }
            }
${addPure('beta * mat.emissionColor * w', 'depth', '            ')}${frame && !emit ? `            if (depth == 0u) {
                fDirect = beta * mat.emissionColor * w;
            }
` : ''}${hooks ? `            hookEmitted = beta * mat.emissionColor * w;
` : ''}        }

${lodS.bsdf(mixU, shadingNormal, true) ?? `        let bp = ptBsdfFromMaterial(mat${mats ? `, ${mixU}` : ''});
`}${film ? `        ptFilm = select(vec3f(0.0), ptMaterialFilm(hit.material), mat.coatWeight <= 0.5 && mat.transmissionWeight <= 0.5${lodS.and});
` : ''}${fuzz ? `        ptFuzzW = ${lodS.fuzzW};
        ptFuzzC = vec4f(mat.sheenColor, mat.sheenRoughness);
` : ''}${aovs ? `        if (depth == 0u) {
            ptAovPrimary(pxIndex, dims.x * dims.y, -dot(surf.position - origin, normalize(ptParams.forward.xyz)));
        }
        if (!aovDone && !ptBsdfIsDelta(bp)) {
            aovDone = true;
            ptAovGuide(pxIndex, dims.x * dims.y, ptAovAlbedo(mat), ${shadingNormal});
        }
` : ''}${frame ? `        if (depth == 0u) {
            fSeen = true;
            fNormal = ${shadingNormal};
            fViewZ = -dot(surf.position - origin, normalize(ptParams.forward.xyz));
            fMotion = ptFrameMotion(surf.position);
            fRough = mat.specularRoughness;
            fAlbedo = ptAovAlbedo(mat);
            fSpec = ptAovSpecular(mat);
            fIsSpec = bp.kind == PT_BSDF_CONDUCTOR || bp.kind == PT_BSDF_DIELECTRIC;
        }
` : ''}        if (${black} || depth >= maxDepth) {
${counters ? `            termReason = select(2u, 1u, ${black});
` : ''}            break;
        }
        let vertex = depth;
${volOut ? `        ptVoCapture(depth, radiance);
` : ''}        depth += 1u;
${layered ? `        ptLayerSeed = pixelSeed ^ (sampleIndex * 2654435761u) ^ (vertex * 2246822519u);
` : ''}
${aniso ? `        var frame = ptFrameFromNormal(${shadingNormal});
        if (bp.kind == PT_BSDF_CONDUCTOR && mat.specularAnisotropy > 0.0${lodS.and}) {
            frame = ptFrameFromTangent(${shadingNormal}, surf.vertexTangent, mat.anisotropyRotation * 6.283185307179586);
        }
` : `        let frame = ptFrameFromNormal(${shadingNormal});
`}        let wo = ptToLocal(frame, -ray.direction);
        if (PT_NEE && ${fuzz ? '(!ptBsdfIsDelta(bp) || ptFuzzActive(wo.z))' : '!ptBsdfIsDelta(bp)'}) {
${addRadiance(`beta * ptSampleLd(surf.position, hit.normal, frame, wo, bp, pixelSeed, sampleIndex, vertex${counters || restir ? ', pxIndex' : ''}${vol ? ', medIn, medOut' : ''})`, 'vertex', '            ')}        }

${gi ? `        if (vertex == 0u && !ptBsdfIsDelta(bp)) {
            // ReSTIR GI: the final reservoir stands for everything after this vertex (the 2-bounce estimate); the DI term above is the direct light.
            radiance += beta * ptRestirGiLi(surf.position, frame, wo, bp, pxIndex);
${frame ? `            fBounce = ptGiDist;
` : ''}${term(7)}            break;
        }

` : ''}${hooks ? `        let hv = ${ptHookVertexExpr({ x: 'px.x', y: 'px.y', pixel: 'pxIndex', sampleIndex: 'sampleIndex' }, hooks)};
        if (ptOnVertex(hv) == PT_HOOK_TERMINATE) {
${addRadiance('beta * ptTerminate(hv)', 'vertex + 1u', '            ')}${term(4)}            break;
        }

` : ''}        let dirU = ptSample2D(pixelSeed, sampleIndex, ptVertexPair(vertex, PT_SLOT_BSDF_DIRECTION));
        let compU = ptSample2D(pixelSeed, sampleIndex, ptVertexPair(vertex, PT_SLOT_SELECT_COMPONENT)).y;
        let s = ${sampleFn}(bp, wo, vec3f(compU, dirU.x, dirU.y));
        if (s.flags == 0u || !(s.pdf > 0.0)) {
${term(5)}            break;
        }
        beta = beta * s.f * (abs(s.wi.z) / s.pdf);
        specularBounce = (s.flags & PT_FLAG_DELTA) != 0u;
${emitMirror ? `        fPure = ptFramePureNext(fPure, specularBounce, bp.kind);
` : ''}${tex ? `        cone = RtCone(coneW, cone.spread + select(PT_CONE_DIFFUSE_SPREAD, 0.0, specularBounce));
` : ''}        if ((s.flags & PT_FLAG_TRANSMISSION) != 0u) {
            etaScale = etaScale * s.etap * s.etap;
${medium ? `            sigma = select(vec3f(0.0), ptMediumSigma(mat), wo.z > 0.0);
` : ''}        }
        prevPdf = ${restir ? 'select(s.pdf, 0.0, vertex == 0u)' : 's.pdf'};${stochastic ? `
        if ((s.flags & PT_FLAG_PROPORTIONAL) != 0u) {
            prevPdf = ptLayeredPdf(ptLayeredOf(bp), wo, s.wi, ptLayerSeed);
        }` : ''}
        prevP = surf.position;
        let wiWorld = ptFromLocal(frame, s.wi);
        ray = RtRay(ptOffsetOrigin(surf.position, ptSideNormal(hit.normal, wiWorld)), 0.0, wiWorld, PT_FAR);
${vol ? `        curMed = select(medOut, medIn, dot(wiWorld, hit.normal) < 0.0);
` : ''}
        if (rrStart > 0u && depth >= rrStart) {
            let rrBeta = beta * etaScale;
            let m = ptMax3(rrBeta);
            if (m < 1.0) {
                let q = max(0.0, 1.0 - m);
                if (ptSample2D(pixelSeed, sampleIndex, ptVertexPair(vertex, PT_SLOT_RR)).x < q) {
${counters ? `                    termReason = 3u;
` : ''}                    break;
                }
                beta = beta / (1.0 - q);
            }
        }
    }
${counters ? `    ptCountEnd(pxIndex, depth, termReason);
` : ''}${frame ? `    ptFramePlanes(pxIndex, dims.x * dims.y, radiance, ${emit ? 'vec3f(0.0), fEmit' : 'fDirect, vec3f(0.0)'}, fBounce, fSeen, fIsSpec, fNormal, fViewZ, fMotion, fRough, fAlbedo, fSpec);
` : ''}${volOut ? `    ptVoEnd(radiance);
` : ''}    return PtRadiance(${spec ? 'ptSpecToRgb(radiance)' : emit ? 'radiance + fEmit' : 'radiance'}, 0.0);
}

@compute @workgroup_size(${PT_WORKGROUP[0]}, ${PT_WORKGROUP[1]})
fn ptMain(@builtin(global_invocation_id) id: vec3u) {
    let dims = ptParams.dims;
    if (id.x >= dims.x || id.y >= dims.y) {
        return;
    }
    let l = ptPath(id.xy, dims.z);
    let index = id.y * dims.x + id.x;
    ptAccum[index] = ${o.accumulate ? 'ptAccum[index] + ' : ''}vec4f(l.xyz, 1.0);${volOut ? `
    ptVoAccumulate(index, dims.x * dims.y, l.xyz);` : ''}
}
`;
}
