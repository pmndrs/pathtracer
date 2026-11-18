/**
 * Approach (a) of P5: bake the spatially varying inputs of a MaterialX material to textures over the uv square [0, 1]^2, so the existing
 * kernel renders it from the material record and the texture buckets, with no kernel change.
 *
 * What can be a texture is what the scene copy has a slot for: base colour (RGB, with the base weight folded in, and the opacity in alpha),
 * roughness and metalness (G and B of the metallic-roughness texture), emission colour and a tangent-space normal. Any other varying input
 * is replaced by its mean over the uv square and listed in `approximations`. Inputs that read the position or the normal cannot be baked in
 * uv; they are evaluated with the position at the origin (also listed). The resolution is the price: detail finer than a texel is lost
 * (approach (b), `wgsl.ts`, keeps it).
 */
import { coerceValue } from './nodes.ts';
import { collectCategories, constantContext, evalIr } from './eval.ts';
import { mtlxConstantParams, slotValue, type MtlxMaterial, type MtlxParams } from './material.ts';
import type { MtlxTsContext } from './nodes.ts';
import type { OpenPbrKey } from './surface.ts';

export interface BakedTexture {
  width: number;
  height: number;
  /** RGBA8, row 0 = v 0 (top). */
  data: Uint8Array;
  colorSpace: 'srgb' | 'linear';
}

export interface BakeOptions {
  /** Texture size (a square, default 256). */
  resolution?: number;
  /** Image lookups for `image` / `tiledimage` nodes (`imageSampler`). */
  sample?: MtlxTsContext['sample'];
  /** false: plan only. The result has the same parameters, notes and textures' sizes and colour spaces, with empty texel data (cheap; the scene description is built from it, the texels later). Default true. */
  texels?: boolean;
}

export interface BakedMaterial {
  /** The material's parameters: constants as authored, baked slots at their neutral factor (white, 1), approximated slots at their mean. */
  params: MtlxParams;
  baseColorTexture?: BakedTexture;
  metallicRoughnessTexture?: BakedTexture;
  emissiveTexture?: BakedTexture;
  normalTexture?: BakedTexture;
  /** `blend` when the opacity is below 1 anywhere. */
  alphaMode?: 'blend';
  /** What was averaged, clamped or dropped, one line each. */
  approximations: string[];
}

const SRGB_ENCODE = (() => {
  // Linear [0, 1] -> 8-bit sRGB code, by a 4096-entry table (error < 1 code) so a 256 x 256 x 3 bake is cheap.
  const n = 4096;
  const t = new Uint8Array(n + 1);
  for (let i = 0; i <= n; i++) {
    const x = i / n;
    const c = x <= 0.0031308 ? x * 12.92 : 1.055 * Math.pow(x, 1 / 2.4) - 0.055;
    t[i] = Math.min(255, Math.max(0, Math.round(c * 255)));
  }
  return t;
})();
const srgb8 = (x: number): number => SRGB_ENCODE[Math.round(Math.min(1, Math.max(0, x)) * 4096)]!;
const byte = (x: number): number => Math.min(255, Math.max(0, Math.round(x * 255)));

const TEXTURABLE: ReadonlySet<OpenPbrKey> = new Set<OpenPbrKey>(['baseColor', 'baseWeight', 'geometryOpacity', 'specularRoughness', 'baseMetalness', 'emissionColor']);

/** The context at texel `(x, y)` of an `n` x `n` grid. */
function texelContext(base: MtlxTsContext, x: number, y: number, n: number): MtlxTsContext {
  return { ...base, uv: [(x + 0.5) / n, (y + 0.5) / n] };
}

export function bakeMtlxMaterial(m: MtlxMaterial, options: BakeOptions = {}): BakedMaterial {
  const n = options.resolution ?? 256;
  const full = options.texels ?? true;
  if (!Number.isInteger(n) || n < 1) throw new RangeError(`bake: bad resolution ${n}`);
  const base = constantContext(options.sample);
  const notes: string[] = [];
  const params = mtlxConstantParams(m);
  const varying = m.varying;
  const has = (k: OpenPbrKey): boolean => k in varying;

  // Slots that read the position / normal / tangent: uv-only baking is an approximation.
  for (const slot of Object.values(varying)) {
    const cats = collectCategories(slot.ir);
    for (const c of ['position', 'normal', 'tangent']) if (cats.has(c)) notes.push(`${slot.key}: reads <${c}>; baked over uv with ${c === 'position' ? 'the position at the origin' : `the ${c} at its default`}`);
  }

  // Varying slots with no texture slot: their mean over a coarse grid.
  const coarse = 16;
  for (const slot of Object.values(varying)) {
    if (TEXTURABLE.has(slot.key)) continue;
    const acc: [number, number, number] = [0, 0, 0];
    for (let y = 0; y < coarse; y++) {
      for (let x = 0; x < coarse; x++) {
        const v = slotValue(m, slot, texelContext(base, x, y, coarse));
        if (Array.isArray(v)) for (let c = 0; c < 3; c++) acc[c]! += v[c]!;
        else acc[0]! += v === true ? 1 : v === false ? 0 : v;
      }
    }
    const cnt = coarse * coarse;
    const mean = Array.isArray(slotValue(m, slot, texelContext(base, 0, 0, coarse))) ? ([acc[0]! / cnt, acc[1]! / cnt, acc[2]! / cnt] as [number, number, number]) : acc[0]! / cnt;
    params[slot.key] = typeof mean === 'number' && slot.key === 'geometryThinWalled' ? mean > 0.5 : mean;
    notes.push(`${slot.key}: varies over uv but has no texture slot; replaced by its mean over the uv square`);
  }

  const out: BakedMaterial = { params, approximations: notes };
  const evalSlot = (k: OpenPbrKey, ctx: MtlxTsContext): number | [number, number, number] | undefined => {
    const slot = varying[k];
    if (!slot) return undefined;
    const v = slotValue(m, slot, ctx);
    return typeof v === 'boolean' ? (v ? 1 : 0) : v;
  };
  const constNum = (k: OpenPbrKey, d: number): number => (typeof params[k] === 'number' ? (params[k] as number) : d);
  const constRgb = (k: OpenPbrKey, d: [number, number, number]): [number, number, number] => (Array.isArray(params[k]) ? (params[k] as [number, number, number]) : d);
  let clamped = 0;
  const note01 = (_label: string, v: number): number => {
    if (v > 1 && Number.isFinite(v)) clamped++;
    return v;
  };

  // --- base colour (rgb sRGB, alpha = opacity) ---------------------------------------------------------------
  const baseVaries = has('baseColor') || has('baseWeight');
  const opacityVaries = has('geometryOpacity');
  if (baseVaries || opacityVaries) {
    const data = new Uint8Array(full ? n * n * 4 : 0);
    const cBase = constRgb('baseColor', [0.8, 0.8, 0.8]);
        const cOpacity = constNum('geometryOpacity', 1);
    for (let y = 0; y < (full ? n : 0); y++) {
      for (let x = 0; x < n; x++) {
        const ctx = texelContext(base, x, y, n);
        const color = (evalSlot('baseColor', ctx) as [number, number, number] | undefined) ?? cBase;
        const w = (evalSlot('baseWeight', ctx) as number | undefined) ?? 1; // a constant weight stays a factor of the record
        const op = (evalSlot('geometryOpacity', ctx) as number | undefined) ?? cOpacity;
        const o = (y * n + x) * 4;
        for (let c = 0; c < 3; c++) data[o + c] = srgb8(note01('baseColor', color[c]! * w));
        data[o + 3] = byte(op);
      }
    }
    out.baseColorTexture = { width: n, height: n, data, colorSpace: 'srgb' };
    params.baseColor = [1, 1, 1];
    if (has('baseWeight')) params.baseWeight = 1;
    if (opacityVaries) {
      params.geometryOpacity = 1;
      out.alphaMode = 'blend'; // varying opacity: blend (stochastic) whether or not this bake's texels dip below 1
    }
  }
  if (!opacityVaries && constNum('geometryOpacity', 1) < 1) out.alphaMode = 'blend';

  // --- roughness (G) and metalness (B) -----------------------------------------------------------------------
  if (has('specularRoughness') || has('baseMetalness')) {
    const data = new Uint8Array(full ? n * n * 4 : 0);
    const cRough = constNum('specularRoughness', 0.3);
    const cMetal = constNum('baseMetalness', 0);
    for (let y = 0; y < (full ? n : 0); y++) {
      for (let x = 0; x < n; x++) {
        const ctx = texelContext(base, x, y, n);
        const r = (evalSlot('specularRoughness', ctx) as number | undefined) ?? cRough;
        const mt = (evalSlot('baseMetalness', ctx) as number | undefined) ?? cMetal;
        const o = (y * n + x) * 4;
        data[o] = 0;
        data[o + 1] = byte(note01('roughness', r));
        data[o + 2] = byte(note01('metalness', mt));
        data[o + 3] = 255;
      }
    }
    out.metallicRoughnessTexture = { width: n, height: n, data, colorSpace: 'linear' };
    params.specularRoughness = 1;
    params.baseMetalness = 1;
  }

  // --- emission colour (rgb sRGB; the luminance stays a factor) ----------------------------------------------
  if (has('emissionColor')) {
    const data = new Uint8Array(full ? n * n * 4 : 0);
    for (let y = 0; y < (full ? n : 0); y++) {
      for (let x = 0; x < n; x++) {
        const c = evalSlot('emissionColor', texelContext(base, x, y, n)) as [number, number, number];
        const o = (y * n + x) * 4;
        for (let k = 0; k < 3; k++) data[o + k] = srgb8(note01('emissionColor', c[k]!));
        data[o + 3] = 255;
      }
    }
    out.emissiveTexture = { width: n, height: n, data, colorSpace: 'srgb' };
    params.emissionColor = [1, 1, 1];
  }
  if (clamped > 0) notes.push(`${clamped} baked value(s) above 1 were clamped to the texture range`);

  // --- normal map ----------------------------------------------------------------------------------------------
  if (m.normal && m.normal.ir.kind === 'op') {
    const nm = m.normal.ir;
    if (nm.category !== 'normalmap') {
      notes.push(`normal: <${nm.category}> is not a normalmap node; the geometric normal is used`);
    } else if ((nm.strings.space ?? 'tangent') !== 'tangent') {
      notes.push('normal: object-space normal maps are not baked; the geometric normal is used');
    } else if (!nm.args.in) {
      notes.push('normal: normalmap without an input; ignored');
    } else {
      const data = new Uint8Array(full ? n * n * 4 : 0);
      const inArg = nm.args.in, scaleArg = nm.args.scale;
      for (let y = 0; y < (full ? n : 0); y++) {
        for (let x = 0; x < n; x++) {
          const ctx = texelContext(base, x, y, n);
          const w = coerceValue(evalIr(inArg.ir, ctx), inArg.from, inArg.to);
          const s = scaleArg ? coerceValue(evalIr(scaleArg.ir, ctx), scaleArg.from, scaleArg.to)[0] : 1;
          // mx_normalmap.glsl: zero decodes to +Z, xy scaled, then normalised.
          let v: [number, number, number] = w[0] === 0 && w[1] === 0 && w[2] === 0 ? [0, 0, 1] : [w[0] * 2 - 1, w[1] * 2 - 1, w[2] * 2 - 1];
          v = [v[0] * s, v[1] * s, v[2]];
          const l = Math.hypot(v[0], v[1], v[2]);
          const o = (y * n + x) * 4;
          data[o] = byte((v[0] / l) * 0.5 + 0.5);
          data[o + 1] = byte((v[1] / l) * 0.5 + 0.5);
          data[o + 2] = byte((v[2] / l) * 0.5 + 0.5);
          data[o + 3] = 255;
        }
      }
      out.normalTexture = { width: n, height: n, data, colorSpace: 'linear' };
    }
  }
  return out;
}
