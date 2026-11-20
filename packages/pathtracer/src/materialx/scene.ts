/**
 * A MaterialX document as scene-copy data (experimental): `buildMaterialX` turns a `.mtlx` text into a scene-copy material and the textures
 * it needs, in one of three modes, and `mtlxKernelOptions` gives the kernel options the codegen mode needs.
 *
 *  - `constants`: every input is constant (folded); the material is exactly the equivalent OpenPBR record. No kernel option.
 *  - `bake`: inputs that vary over uv are baked into textures over the uv square [0, 1]^2 (`bakeMtlxMaterial`); no kernel option.
 *  - `codegen`: the record holds the parameters at uv (0.5, 0.5) and the kernel evaluates the graph at every hit (generated WGSL,
 *    `integrator.materialx`); image nodes read scene textures made here. Position- or normal-driven graphs need this mode.
 *
 *     const built = buildMaterialX(text, { mode: 'codegen', images: mtlxSyntheticImage });
 *     const use = mtlxAddToScene(input, built);               // appends the material and its textures to a SceneCopyInput
 *     const scene = await buildGpuScene(device, input);
 *     const pt = await PtMegakernel.create(device, caps, ptSceneOptions(scene, mtlxKernelOptions([use])));
 */
import { openPbrToRtMaterial, type OpenPbrParams, type RtMaterialInput, type TextureInput } from '@pmndrs/raytracer';
import { bakeMtlxMaterial, type BakedTexture } from './bake.ts';
import { imageSampler, type MtlxImage } from './image.ts';
import { loadMtlxMaterial, mtlxConstantParams, type MtlxMaterial } from './material.ts';
import type { MtlxImageRequest } from './nodes.ts';
import { mtlxFallbackParams, mtlxProgramsWgsl } from './wgsl.ts';

export type MtlxBuildMode = 'constants' | 'bake' | 'codegen';
/** Image files of the document -> RGBA8 texels (row 0 = v 0). `undefined`: not available (the image node's default applies). */
export type MtlxImageLoader = (file: string) => MtlxImage | undefined;

export interface MtlxBuildOptions {
  /** Default: `constants` when nothing varies over the surface, else `codegen`. */
  mode?: MtlxBuildMode;
  /** Bake resolution (a square). Default 256. */
  resolution?: number;
  /** Image loader for `image` / `tiledimage` nodes. Default: none (every image node falls back to its default). */
  images?: MtlxImageLoader;
  /** `surfacematerial` or shader node name. Default: the first material of the document. */
  material?: string;
  /** Metres per MaterialX length unit. Default 0.01 (OpenPBR's centimetre). */
  lengthScale?: number;
}

export interface MtlxSceneMaterial {
  mode: MtlxBuildMode;
  /** The parsed and lowered material. */
  mtl: MtlxMaterial;
  /** The scene-copy record. Its texture fields index `textures` (local: `mtlxAddToScene` offsets them). */
  material: RtMaterialInput;
  textures: TextureInput[];
  /** Codegen: the images the generated code reads, and which of `textures` each one is. */
  images: { request: MtlxImageRequest; texture: number }[];
  /** Approximations and limits, one line each. */
  notes: string[];
}

/** One built material placed in a scene: its scene-copy material index and where its textures start in the scene's texture list. */
export interface MtlxSceneUse {
  material: number;
  textureOffset: number;
  built: MtlxSceneMaterial;
}

export const mtlxImageKey = (r: MtlxImageRequest): string => `${r.file}|${r.srgb ? 's' : 'l'}|${r.uaddress}|${r.vaddress}`;

const wrapOf = (address: string): TextureInput['wrap'] => (address === 'clamp' ? 'clamp' : address === 'mirror' ? 'mirror' : 'repeat');

/** Parses `source` and builds the material in the chosen mode. Baking and image decoding happen here, on the CPU. */
export function buildMaterialX(source: string, options: MtlxBuildOptions = {}): MtlxSceneMaterial {
  const loader: MtlxImageLoader = options.images ?? (() => undefined);
  const mtl = loadMtlxMaterial(source, { ...(options.material ? { material: options.material } : {}), ...(options.lengthScale !== undefined ? { lengthScale: options.lengthScale } : {}) });
  const varies = Object.keys(mtl.varying).length > 0 || !!mtl.normal;
  const mode: MtlxBuildMode = options.mode ?? (varies ? 'codegen' : 'constants');
  const notes: string[] = [];
  const textures: TextureInput[] = [];
  const images: MtlxSceneMaterial['images'] = [];
  const keys = new Map<string, number>();
  const addTexture = (key: string, t: TextureInput): void => {
    keys.set(key, textures.length);
    textures.push(t);
  };
  let params: OpenPbrParams;

  if (mode === 'constants') {
    if (varies) throw new Error(`MaterialX: mode 'constants' but ${Object.keys(mtl.varying).join(', ') || 'the normal'} varies; use 'bake' or 'codegen'`);
    params = { ...(mtlxConstantParams(mtl) as OpenPbrParams) };
  } else if (mode === 'bake') {
    const baked = bakeMtlxMaterial(mtl, { resolution: options.resolution ?? 256, sample: imageSampler(loader) });
    params = { ...(baked.params as OpenPbrParams) };
    notes.push(...baked.approximations);
    const slot = (field: 'baseColorTexture' | 'metallicRoughnessTexture' | 'emissiveTexture' | 'normalTexture', t: BakedTexture | undefined): void => {
      if (!t) return;
      addTexture(field, { width: t.width, height: t.height, data: t.data, colorSpace: t.colorSpace, label: `materialx ${field}` });
      params[field] = field;
    };
    slot('baseColorTexture', baked.baseColorTexture);
    slot('metallicRoughnessTexture', baked.metallicRoughnessTexture);
    slot('emissiveTexture', baked.emissiveTexture);
    slot('normalTexture', baked.normalTexture);
    if (baked.alphaMode) params.alphaMode = baked.alphaMode;
  } else {
    params = { ...(mtlxFallbackParams(mtl) as OpenPbrParams) };
    if (mtl.varying.emissionColor || mtl.varying.emissionLuminance) {
      // The program sets the emission at the hit; the record must not make an area light of it.
      params.emissionLuminance = 0;
      notes.push('emission varies: the record emits nothing, the generated function emits (no next-event sampling of this surface)');
    }
    if (mtl.varying.geometryOpacity) notes.push('geometry_opacity varies: shading sees it, alpha cutouts and shadow rays do not (codegen)');
    for (const request of mtl.images) {
      const img = loader(request.file);
      if (!img) {
        notes.push(`image '${request.file}' is not available: the node's default applies`);
        continue;
      }
      if (request.uaddress !== request.vaddress) notes.push(`image '${request.file}': u and v address modes differ (${request.uaddress}, ${request.vaddress}); the texture uses the u mode`);
      images.push({ request, texture: textures.length });
      addTexture(mtlxImageKey(request), { width: img.width, height: img.height, data: img.data, colorSpace: request.srgb ? 'srgb' : 'linear', wrap: wrapOf(request.uaddress), label: request.file });
    }
    // The kernel decodes uv, position and tangent only in a textured scene: a graph that reads no image still gets a texture.
    if (textures.length === 0) addTexture('placeholder', { width: 1, height: 1, data: new Uint8Array([255, 255, 255, 255]), colorSpace: 'linear', label: 'materialx placeholder' });
  }
  const material = openPbrToRtMaterial(params, (key) => {
    const i = keys.get(key);
    if (i === undefined) throw new RangeError(`MaterialX: texture '${key}' was not built`);
    return i;
  });
  return { mode, mtl, material, textures, images, notes };
}

/** Appends a built material and its textures to a scene-copy input (`materials`, `textures`), offsetting its texture indices. */
export function mtlxAddToScene(input: { materials: RtMaterialInput[]; textures?: readonly TextureInput[] }, built: MtlxSceneMaterial): MtlxSceneUse {
  const textureOffset = input.textures?.length ?? 0;
  input.textures = [...(input.textures ?? []), ...built.textures];
  const shift = (i: number | undefined) => (i === undefined ? undefined : i + textureOffset);
  const m: RtMaterialInput = { ...built.material };
  for (const field of ['baseColorTexture', 'roughnessMetalTexture', 'normalTexture', 'emissionTexture'] as const) {
    const v = shift(m[field]);
    if (v !== undefined) m[field] = v;
  }
  input.materials.push(m);
  return { material: input.materials.length - 1, textureOffset, built };
}

/** Records a program varies a lobe weight in: the kernel must be compiled for that lobe, whatever the record's fallback value says. */
export function mtlxFeatureProbes(mtl: MtlxMaterial): RtMaterialInput[] {
  const probes: RtMaterialInput[] = [];
  const v = mtl.varying;
  if (v.coatWeight || v.coatRoughness || v.coatColor) probes.push({ coatWeight: 0.5, coatRoughness: 0.1 });
  if (v.transmissionWeight) probes.push({ transmissionWeight: 0.5 });
  if (v.fuzzWeight) probes.push({ sheenWeight: 0.5 });
  if (v.baseMetalness) probes.push({ metalness: 0.5 });
  if (v.specularRoughnessAnisotropy) probes.push({ specularAnisotropy: 0.5 });
  return probes;
}

/**
 * The kernel options of the codegen materials among `uses` (spread into `ptSceneOptions` overrides): `integrator.materialx` with the generated
 * WGSL, and `materialProbes` so the kernel compiles the lobes the programs can switch on. Empty when no material uses codegen.
 */
export function mtlxKernelOptions(uses: readonly MtlxSceneUse[]): { integrator?: { materialx: { wgsl: string; normal: boolean } }; materialProbes?: RtMaterialInput[]; notes: string[] } {
  const codegen = uses.filter((u) => u.built.mode === 'codegen');
  const notes = uses.flatMap((u) => u.built.notes);
  if (codegen.length === 0) return { notes };
  const slots = new Map<string, number>();
  for (const u of codegen) for (const im of u.built.images) slots.set(mtlxImageKey(im.request), im.texture + u.textureOffset);
  const programs = mtlxProgramsWgsl(codegen.map((u) => ({ material: u.material, mtl: u.built.mtl })), { textureSlot: (r) => slots.get(mtlxImageKey(r)) ?? -1 });
  return { integrator: { materialx: { wgsl: programs.wgsl, normal: programs.normal } }, materialProbes: codegen.flatMap((u) => mtlxFeatureProbes(u.built.mtl)), notes: [...notes, ...programs.notes] };
}
