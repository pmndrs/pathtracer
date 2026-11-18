/**
 * MaterialX materials as bench scene materials (P5). A MaterialX document becomes an `openpbr` `MaterialDesc` plus the textures it needs, by one of
 * three modes (docs/design/materialx.md):
 *
 *  - `constants`: every input is constant (folded); the material is exactly the equivalent `openpbr` description.
 *  - `bake` (approach a): inputs that vary over uv are baked to textures in the scene's texture buckets (`bakeMtlxMaterial`); no kernel change.
 *  - `codegen` (approach b): the record holds the parameters at uv (0.5, 0.5), the material carries `materialx: <key>`, and the adapter passes the
 *    generated WGSL (`mtlxProgramsWgsl`) to the kernel (`integrator.materialx`); image nodes read scene textures made here.
 *
 * Texels are lazy (`generator: 'lazy'`): describing a scene parses the document only; baking and image decoding happen when a renderer builds the
 * scene copy. Everything here is page-safe (no `node:` imports); Node-only image loaders are passed in by the caller.
 */
import { bakeMtlxMaterial, imageSampler, loadMtlxMaterial, mtlxConstantParams, mtlxFallbackParams, mtlxProgramsWgsl, type MtlxImage, type MtlxImageRequest, type MtlxMaterial, type MtlxPrograms } from '@pmndrs/pathtracer';
import type { RtMaterialInput } from '@pmndrs/raytracer';
import { syntheticImage } from './materialx-images.ts';
import type { MaterialDesc, OpenPbrParams, SceneDesc, TextureDesc } from './schema.ts';
import { registerLazyTexture } from './textures.ts';

export type MaterialXMode = 'constants' | 'bake' | 'codegen';
export type MtlxImageLoader = (file: string) => MtlxImage | undefined;

export interface MaterialXSource {
  /** Registry key (also the prefix of the texture names). Unique per material. */
  key: string;
  /** The `.mtlx` text. */
  source: string;
  mode: MaterialXMode;
  /** Bake resolution (a square; default 256). */
  resolution?: number;
  /** Image files -> texels (default: the synthetic `gen:` images). */
  images?: MtlxImageLoader;
  /** `surfacematerial` / shader name; default the first material. */
  material?: string;
  /** Metres per MaterialX length unit (default 0.01). */
  lengthScale?: number;
}

export interface MtlxEntry {
  key: string;
  mode: MaterialXMode;
  mtl: MtlxMaterial;
  /** Codegen: the images the programs read and the scene texture each one is. */
  images: { request: MtlxImageRequest; texture: string }[];
  /** Approximations and limits, one line each. */
  notes: string[];
}

const entries = new Map<string, MtlxEntry>();
export const getMtlxEntry = (key: string): MtlxEntry => {
  const e = entries.get(key);
  if (!e) throw new Error(`MaterialX material '${key}' is not registered (import the module that defines the scene)`);
  return e;
};

const imageKey = (r: MtlxImageRequest): string => `${r.file}|${r.srgb ? 's' : 'l'}|${r.uaddress}|${r.vaddress}`;

/** Parses, plans and registers; returns the material description and its textures (lazy). */
export function buildMaterialX(src: MaterialXSource): { material: Extract<MaterialDesc, { type: 'openpbr' }>; textures: Record<string, TextureDesc>; entry: MtlxEntry } {
  const loader = src.images ?? syntheticImage;
  const mtl = loadMtlxMaterial(src.source, { ...(src.material ? { material: src.material } : {}), ...(src.lengthScale !== undefined ? { lengthScale: src.lengthScale } : {}) });
  const notes: string[] = [];
  const textures: Record<string, TextureDesc> = {};
  const entry: MtlxEntry = { key: src.key, mode: src.mode, mtl, images: [], notes };
  const out: Record<string, unknown> = {};
  let params: Record<string, unknown>;

  if (src.mode === 'constants') {
    if (Object.keys(mtl.varying).length > 0 || mtl.normal) throw new Error(`MaterialX '${src.key}': mode 'constants' but ${Object.keys(mtl.varying).join(', ') || 'the normal'} varies; use 'bake' or 'codegen'`);
    params = { ...mtlxConstantParams(mtl) };
  } else if (src.mode === 'bake') {
    const resolution = src.resolution ?? 256;
    const sample = imageSampler(loader);
    const plan = bakeMtlxMaterial(mtl, { resolution, texels: false, sample });
    params = { ...plan.params };
    notes.push(...plan.approximations);
    const full = (() => {
      let cached: ReturnType<typeof bakeMtlxMaterial> | undefined;
      return () => (cached ??= bakeMtlxMaterial(mtl, { resolution, sample }));
    })();
    const lazy = (slot: 'baseColorTexture' | 'metallicRoughnessTexture' | 'emissiveTexture' | 'normalTexture', field: string, name: string): void => {
      const t = plan[slot];
      if (!t) return;
      const key = `mtlx:${src.key}:${name}`;
      registerLazyTexture(key, () => {
        const b = full()[slot]!;
        return { width: b.width, height: b.height, data: b.data };
      });
      textures[`${src.key}.${name}`] = { generator: 'lazy', key, width: t.width, height: t.height, colorSpace: t.colorSpace };
      out[field] = `${src.key}.${name}`;
    };
    lazy('baseColorTexture', 'baseColorTexture', 'base');
    lazy('metallicRoughnessTexture', 'metallicRoughnessTexture', 'mr');
    lazy('emissiveTexture', 'emissiveTexture', 'emission');
    lazy('normalTexture', 'normalTexture', 'normal');
    if (plan.alphaMode) out.alphaMode = plan.alphaMode;
  } else {
    params = { ...mtlxFallbackParams(mtl) };
    if (mtl.varying.emissionColor || mtl.varying.emissionLuminance) {
      params.emissionLuminance = 0; // the program sets the emission at the hit; the record must not make an area light of it (a textured emitter is not in the light table, V52)
      notes.push('emission varies: the record emits nothing, the generated function emits (no next-event sampling of this surface)');
    }
    if (mtl.varying.geometryOpacity) notes.push('geometry_opacity varies: shading sees it, alpha cut-outs and shadow rays do not (codegen)');
    // Images the programs read: one scene texture per request.
    mtl.images.forEach((request, i) => {
      const name = `${src.key}.img${i}`;
      const img = loader(request.file);
      if (!img) {
        notes.push(`image '${request.file}' is not available: the node's default applies`);
        return;
      }
      const key = `mtlx:${src.key}:img${i}:${imageKey(request)}`;
      registerLazyTexture(key, () => ({ width: img.width, height: img.height, data: img.data }));
      textures[name] = { generator: 'lazy', key, width: img.width, height: img.height, colorSpace: request.srgb ? 'srgb' : 'linear', wrap: request.uaddress === 'clamp' ? 'clamp' : request.uaddress === 'mirror' ? 'mirror' : 'repeat' };
      entry.images.push({ request, texture: name });
      if (request.uaddress !== request.vaddress) notes.push(`image '${request.file}': u and v address modes differ (${request.uaddress}, ${request.vaddress}); the texture uses the u mode`);
    });
    // The kernel needs a texture layout for uv / position / tangent even when the graph reads no image.
    if (Object.keys(textures).length === 0) textures[`${src.key}.placeholder`] = { generator: 'solid', color: [255, 255, 255, 255], width: 1, height: 1, colorSpace: 'linear' };
    out.materialx = src.key;
  }
  const desc = { type: 'openpbr' as const, ...(params as OpenPbrParams), ...(out as OpenPbrParams) };
  entries.set(src.key, entry);
  return { material: desc, textures, entry };
}

/** Feature probes for `ptMaterialFeatures`: a program that varies a lobe weight needs the kernel compiled for that lobe, whatever the record's fallback value is. */
function featureProbes(mtl: MtlxMaterial): RtMaterialInput[] {
  const probes: RtMaterialInput[] = [];
  const v = mtl.varying;
  if (v.coatWeight || v.coatRoughness || v.coatColor) probes.push({ coatWeight: 0.5, coatRoughness: 0.1 });
  if (v.transmissionWeight) probes.push({ transmissionWeight: 0.5 });
  if (v.fuzzWeight) probes.push({ sheenWeight: 0.5 });
  if (v.baseMetalness) probes.push({ metalness: 0.5 });
  if (v.specularRoughnessAnisotropy) probes.push({ specularAnisotropy: 0.5 });
  return probes;
}

export interface MaterialXKernelInput {
  /** Pass as `integrator.materialx`. */
  programs: MtlxPrograms;
  /** Extra material rows for `ptMaterialFeatures`. */
  featureProbes: RtMaterialInput[];
}

/**
 * The generated programs of a scene: for each scene-copy material (by `materialNames`, `<bench material>` or `<bench material>+emission(...)`) whose bench
 * material has `materialx`, the entry's program. Null when the scene has none.
 */
export function collectMaterialXPrograms(scene: SceneDesc, materialNames: readonly string[]): MaterialXKernelInput | null {
  const programs: { material: number; mtl: MtlxMaterial }[] = [];
  const probes: RtMaterialInput[] = [];
  const textureNames = Object.keys(scene.textures ?? {});
  const slots = new Map<string, number>();
  const notes: string[] = [];
  materialNames.forEach((name, index) => {
    const base = scene.materials[name.split('+')[0]!];
    if (!base || base.type !== 'openpbr' || !base.materialx) return;
    const entry = getMtlxEntry(base.materialx);
    programs.push({ material: index, mtl: entry.mtl });
    probes.push(...featureProbes(entry.mtl));
    notes.push(...entry.notes.map((n) => `${base.materialx}: ${n}`));
    for (const im of entry.images) {
      const slot = textureNames.indexOf(im.texture);
      if (slot < 0) throw new Error(`scene '${scene.name}': texture '${im.texture}' of MaterialX '${base.materialx}' is not in the scene`);
      slots.set(imageKey(im.request), slot);
    }
  });
  if (programs.length === 0) return null;
  const generated = mtlxProgramsWgsl(programs, { textureSlot: (r) => slots.get(imageKey(r)) ?? -1 });
  return { programs: { ...generated, notes: [...notes, ...generated.notes] }, featureProbes: probes };
}
