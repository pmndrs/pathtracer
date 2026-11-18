/**
 * A MaterialX material read for rendering (P5): parse, lower the shader's inputs, fold the constant ones, keep the ones that depend on
 * the shading point (uv, position, normal) as expressions. `bake.ts` turns the varying ones into textures, `wgsl.ts` into a WGSL function.
 */
import { parseMaterialX, type MtlxDocument } from './document.ts';
import { collectCategories, collectImages, constantContext, dependsOnGeometry, evalIr, foldConstants, MtlxLowering, type LowerOptions, type MtlxIr } from './eval.ts';
import { findShaderNode, GLTF_PBR_DEFAULTS, lowerSurface, STANDARD_SURFACE_DEFAULTS, type MtlxSlot, type MtlxSurface, type OpenPbrKey } from './surface.ts';
import type { MtlxImageRequest, MtlxTsContext } from './nodes.ts';

export type MtlxParamValue = number | boolean | [number, number, number];
/** Resolved OpenPBR parameters (the bench's `OpenPbrParams` names; absent = the OpenPBR default). */
export type MtlxParams = Partial<Record<OpenPbrKey, MtlxParamValue>>;

export interface MtlxLoadOptions extends LowerOptions {
  /** `surfacematerial` (or shader node) name; default the first material of the document. */
  material?: string;
  /**
   * Metres per MaterialX length unit, applied to `transmission_depth` (OpenPBR's default unit is the centimetre, ours the metre).
   * Default 0.01 (V262).
   */
  lengthScale?: number;
}

export interface MtlxMaterial {
  name: string;
  /** `open_pbr_surface` or `standard_surface`. */
  category: string;
  doc: MtlxDocument;
  lowering: MtlxLowering;
  surface: MtlxSurface;
  lengthScale: number;
  /** Slots whose value does not depend on the shading point, folded (an expression that is a `const`). */
  constants: Partial<Record<OpenPbrKey, MtlxSlot>>;
  /** Slots that depend on the shading point, with their constant subexpressions folded. */
  varying: Partial<Record<OpenPbrKey, MtlxSlot>>;
  /** The tangent-space / object-space normal expression, folded, if the surface has one. */
  normal?: { ir: MtlxIr; type: string };
  /** Images the varying slots (and the normal) read. */
  images: MtlxImageRequest[];
  /** Node categories used by the varying part, sorted: the support report. */
  categories: string[];
}

const COLOR_KEYS: ReadonlySet<OpenPbrKey> = new Set(['baseColor', 'specularColor', 'transmissionColor', 'subsurfaceColor', 'coatColor', 'fuzzColor', 'emissionColor']);
export const isColorKey = (k: OpenPbrKey): boolean => COLOR_KEYS.has(k);

/** Reads `source`, lowers the chosen material's shader. Throws `MtlxUnsupportedError` for a node or shader that is not supported. */
export function loadMtlxMaterial(source: string, options: MtlxLoadOptions = {}): MtlxMaterial {
  const doc = parseMaterialX(source);
  const shader = findShaderNode(doc, options.material);
  const lowering = new MtlxLowering(doc, options);
  const surface = lowerSurface(lowering, shader);
  const ctx = constantContext();
  const constants: MtlxMaterial['constants'] = {};
  const varying: MtlxMaterial['varying'] = {};
  const dep = new Map<MtlxIr, boolean>();
  for (const slot of Object.values(surface.slots)) {
    const folded = foldConstants(slot.ir, ctx, new Map(), dep);
    (folded.kind === 'const' ? constants : varying)[slot.key] = { ...slot, ir: folded };
  }
  const images: MtlxImageRequest[] = [];
  const cats = new Set<string>();
  for (const slot of Object.values(varying)) {
    for (const r of collectImages(slot.ir)) if (!images.some((o) => o.file === r.file && o.srgb === r.srgb && o.uaddress === r.uaddress && o.vaddress === r.vaddress)) images.push(r);
    for (const c of collectCategories(slot.ir)) cats.add(c);
  }
  const material: MtlxMaterial = { name: shader.name, category: shader.category, doc, lowering, surface, lengthScale: options.lengthScale ?? 0.01, constants, varying, images, categories: [] };
  if (surface.normal) {
    const folded = foldConstants(surface.normal.ir, ctx, new Map(), dep);
    material.normal = { ir: folded, type: surface.normal.type };
    if (folded.kind === 'op') {
      for (const r of collectImages(folded)) if (!images.some((o) => o.file === r.file && o.srgb === r.srgb && o.uaddress === r.uaddress && o.vaddress === r.vaddress)) images.push(r);
      for (const c of collectCategories(folded)) cats.add(c);
    }
  }
  material.categories = [...cats].sort();
  return material;
}

/** The authoring defaults of the shader category that differ from OpenPBR's. */
function applyDefaults(m: MtlxMaterial, out: MtlxParams): void {
  const table = m.category === 'standard_surface' ? STANDARD_SURFACE_DEFAULTS : m.category === 'gltf_pbr' ? GLTF_PBR_DEFAULTS : {};
  for (const [k, v] of Object.entries(table)) out[k as OpenPbrKey] = Array.isArray(v) ? ([v[0]!, v[1]!, v[2]!] as [number, number, number]) : (v as number);
}

/** A slot's value at a shading point as the JS value the parameter takes (scale, reduction and unit conversion applied). */
export function slotValue(m: MtlxMaterial, slot: MtlxSlot, ctx: MtlxTsContext): MtlxParamValue {
  const v = evalIr(slot.ir, ctx);
  const unit = slot.key === 'transmissionDepth' ? m.lengthScale : 1;
  const s = slot.scale * unit;
  if (slot.key === 'geometryThinWalled') return v[0] > 0.5;
  if (isColorKey(slot.key)) return [v[0] * s, v[1] * s, v[2] * s];
  if (slot.reduce === 'avg') return ((v[0] + v[1] + v[2]) / 3) * s;
  return v[0] * s;
}

/** The parameters at one shading point: constants and varying slots alike, plus the standard_surface defaults that differ from OpenPBR's and the derived thin-film weight. */
export function evalMtlxParams(m: MtlxMaterial, ctx: MtlxTsContext): MtlxParams {
  const out: MtlxParams = {};
  applyDefaults(m, out);
  for (const slot of [...Object.values(m.constants), ...Object.values(m.varying)]) out[slot.key] = slotValue(m, slot, ctx);
  if (m.category === 'standard_surface') out.thinFilmWeight = typeof out.thinFilmThickness === 'number' && out.thinFilmThickness > 0 ? 1 : 0;
  return out;
}

/** The parameters that do not depend on the shading point (what a constant material is). */
export function mtlxConstantParams(m: MtlxMaterial): MtlxParams {
  const out: MtlxParams = {};
  const ctx = constantContext();
  applyDefaults(m, out);
  for (const slot of Object.values(m.constants)) out[slot.key] = slotValue(m, slot, ctx);
  if (m.category === 'standard_surface') out.thinFilmWeight = typeof out.thinFilmThickness === 'number' && out.thinFilmThickness > 0 ? 1 : 0;
  return out;
}

export const slotIsGeometric = (slot: MtlxSlot): boolean => dependsOnGeometry(slot.ir);
