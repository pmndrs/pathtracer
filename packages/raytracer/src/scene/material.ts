/**
 * Material records: the OpenPBR subset of spec section 6 packed into 128 bytes (`MATERIAL_RECORD_BYTES`), plus the TypeScript
 * twin of the WGSL `rtMaterial` unpack. The id that indexes the table is bvh's 16-bit per-triangle / per-instance material.
 *
 * `RtMaterial` (here and in WGSL, same names) is the plain, decoded parameter set the BSDF code reads. It is deliberately
 * not a BSDF struct: the path tracer's `pt*` code maps it with a one-line adapter. Conventions:
 *  - `specularRoughness` and `coatRoughness` are perceptual (glTF / OpenPBR); GGX alpha = roughness^2 (`rtMaterialAlpha`).
 *  - Colours are linear RGB (Rec.709 primaries). `emissionColor` is radiance in W/sr/m^2, already multiplied by any
 *    intensity; `emissionLuminance` is its Rec.709 luminance (the light power uses it).
 *  - `conductorEtaK` selects the physical conductor (complex IOR `conductorEta` + i `conductorK`, the bench `conductor`
 *    type). Otherwise `metalness` blends a metal whose reflectance is `baseColor` (OpenPBR F82 tint); `specularColor` tints
 *    the dielectric specular lobe.
 *  - `transmissionDepth` is the Beer-Lambert depth in metres at which light is attenuated to `transmissionColor`; 0 = no
 *    absorption (the bench `dielectric`).
 *  - `specularWeight` 0 disables the dielectric specular lobe (a pure Lambertian: the bench `diffuse`).
 *  - `alphaMode`: 0 opaque, 1 mask (alpha test against `alphaCutoff`), 2 blend (stochastic).
 *
 * Storage, 32 words (8 vec4u). Full precision where an error would show in a reference comparison (colours, emission, IOR, depth);
 * half precision for conductor eta and k; unorm16 (error <= 7.7e-6, exact 0 and 1) for the weights, roughness and tints:
 *
 *     w0-2    baseColor f32 x3        w3  ior f32
 *     w4-6    emissionColor f32 x3    w7  emissionLuminance f32
 *     w8      transmissionDepth f32   w9  coatIor f32       w10 flags       w11 (baseColor, roughnessMetal) texture slots u16 x2
 *     w12     (normal, emission) texture slots u16 x2
 *     w13-15  conductorEta, conductorK as f16 x6
 *     w16-28  25 unorm16 values, two per word, low half first (`U16_SLOT`), then one spare half
 *     w29-31  reserved, written 0: subsurface, thin film, fuzz, dispersion will land here
 */
import { LUMA, TEXTURE_SLOT_NONE } from './constants.ts';

export type Rgb3 = [number, number, number];

export const MATERIAL_RECORD_VEC4S = 8;
export const MATERIAL_RECORD_WORDS = 32;
export const MATERIAL_RECORD_BYTES = 128;
/** First reserved word and how many (documented room for lobes not implemented yet). */
export const MATERIAL_RESERVED_WORDS = { first: 29, count: 3 } as const;

export const MATERIAL_WORD = {
  baseColor: 0,
  ior: 3,
  emissionColor: 4,
  emissionLuminance: 7,
  transmissionDepth: 8,
  coatIor: 9,
  flags: 10,
  texturesA: 11,
  texturesB: 12,
  /** f16 x6: eta r, g, b, k r, g, b. */
  conductor: 13,
  /** First of the unorm16 words. */
  unorm: 16,
} as const;

/** Slot (0-based) of each value among the 25 unorm16 values; word = MATERIAL_WORD.unorm + (slot >> 1), high half when odd. */
export const U16_SLOT = {
  baseWeight: 0,
  metalness: 1,
  specularRoughness: 2,
  specularWeight: 3,
  specularColor: 4, // 4, 5, 6
  specularAnisotropy: 7,
  anisotropyRotation: 8,
  transmissionWeight: 9,
  transmissionColor: 10, // 10, 11, 12
  coatWeight: 13,
  coatRoughness: 14,
  coatColor: 15, // 15, 16, 17
  sheenWeight: 18,
  sheenRoughness: 19,
  sheenColor: 20, // 20, 21, 22
  opacity: 23,
  alphaCutoff: 24,
} as const;

export const MATERIAL_FLAG = {
  /** Bits 0-1: alpha mode. */
  alphaModeMask: 0x3,
  thinWalled: 0x4,
  conductorEtaK: 0x8,
  emissionTwoSided: 0x10,
  doubleSided: 0x20,
} as const;

export const ALPHA_MODE = { opaque: 0, mask: 1, blend: 2 } as const;

/** Scale that turns a unorm16 integer into a float: `fround(1/65535)`. A multiply is correctly rounded on every GPU, a divide is not, and 65535 * it is exactly 1. */
export const UNORM16_SCALE = Math.fround(1 / 65535);

/** The decoded, plain parameter set (see the file comment). The same field names exist in WGSL as `RtMaterial`. */
export interface RtMaterial {
  baseColor: Rgb3;
  baseWeight: number;
  metalness: number;
  specularRoughness: number;
  specularWeight: number;
  specularColor: Rgb3;
  ior: number;
  specularAnisotropy: number;
  /** Turns, [0, 1): the anisotropy axis's rotation about the normal from the tangent toward the bitangent, times 2 pi (glTF `anisotropyRotation` is radians; `rotationTurns` in bench/scenes/to-scene-copy.ts converts). */
  anisotropyRotation: number;
  conductorEta: Rgb3;
  conductorK: Rgb3;
  transmissionWeight: number;
  transmissionColor: Rgb3;
  transmissionDepth: number;
  coatWeight: number;
  coatColor: Rgb3;
  coatRoughness: number;
  coatIor: number;
  sheenWeight: number;
  sheenColor: Rgb3;
  sheenRoughness: number;
  emissionColor: Rgb3;
  emissionLuminance: number;
  opacity: number;
  alphaCutoff: number;
  /** ALPHA_MODE value. */
  alphaMode: number;
  thinWalled: boolean;
  /** Physical conductor: use `conductorEta` and `conductorK` instead of `baseColor` + `metalness`. */
  conductorEtaK: boolean;
  /** Emit from both faces (default: the authored front face only). */
  emissionTwoSided: boolean;
  doubleSided: boolean;
  /** Texture slots (u16; TEXTURE_SLOT_NONE = none). Slots only: textures themselves are a later task. */
  baseColorTexture: number;
  roughnessMetalTexture: number;
  normalTexture: number;
  emissionTexture: number;
}

/** Every field optional; absent ones take the OpenPBR defaults (`defaultMaterial`). `emissionLuminance` defaults to the luminance of `emissionColor`. */
export type RtMaterialInput = Partial<RtMaterial>;

export function luminance(c: readonly number[]): number {
  return LUMA[0] * (c[0] as number) + LUMA[1] * (c[1] as number) + LUMA[2] * (c[2] as number);
}

/** OpenPBR defaults: a 0.8 grey dielectric with a 0.3 rough specular lobe, no emission, opaque. */
export function defaultMaterial(): RtMaterial {
  return {
    baseColor: [0.8, 0.8, 0.8],
    baseWeight: 1,
    metalness: 0,
    specularRoughness: 0.3,
    specularWeight: 1,
    specularColor: [1, 1, 1],
    ior: 1.5,
    specularAnisotropy: 0,
    anisotropyRotation: 0,
    conductorEta: [0.2, 0.9, 1.1],
    conductorK: [3.9, 2.4, 2.1],
    transmissionWeight: 0,
    transmissionColor: [1, 1, 1],
    transmissionDepth: 0,
    coatWeight: 0,
    coatColor: [1, 1, 1],
    coatRoughness: 0,
    coatIor: 1.5,
    sheenWeight: 0,
    sheenColor: [1, 1, 1],
    sheenRoughness: 0.3,
    emissionColor: [0, 0, 0],
    emissionLuminance: 0,
    opacity: 1,
    alphaCutoff: 0.5,
    alphaMode: ALPHA_MODE.opaque,
    thinWalled: false,
    conductorEtaK: false,
    emissionTwoSided: false,
    doubleSided: false,
    baseColorTexture: TEXTURE_SLOT_NONE,
    roughnessMetalTexture: TEXTURE_SLOT_NONE,
    normalTexture: TEXTURE_SLOT_NONE,
    emissionTexture: TEXTURE_SLOT_NONE,
  };
}

/** Fills defaults into a partial material (colours are copied). */
export function createMaterial(input: RtMaterialInput = {}): RtMaterial {
  const out = { ...defaultMaterial(), ...input } as RtMaterial;
  for (const key of ['baseColor', 'specularColor', 'conductorEta', 'conductorK', 'transmissionColor', 'coatColor', 'sheenColor', 'emissionColor'] as const) out[key] = [...out[key]] as Rgb3;
  if (input.emissionLuminance === undefined) out.emissionLuminance = Math.fround(luminance(out.emissionColor));
  return out;
}

// ---------------------------------------------------------------------------------------------------------------
// Scalar conversions
// ---------------------------------------------------------------------------------------------------------------

const f32Scratch = new Float32Array(1);
const u32Scratch = new Uint32Array(f32Scratch.buffer);

export function f32Bits(value: number): number {
  f32Scratch[0] = value;
  return u32Scratch[0] as number;
}
export function bitsF32(word: number): number {
  u32Scratch[0] = word;
  return f32Scratch[0] as number;
}

/** Round-to-nearest-even half float bits of a finite number; values past +-65504 clamp to the largest finite half. */
export function f32ToF16Bits(value: number): number {
  f32Scratch[0] = value;
  const x = u32Scratch[0] as number;
  const sign = (x >>> 16) & 0x8000;
  const exp = (x >>> 23) & 0xff;
  const mant = x & 0x7fffff;
  const e = exp - 127 + 15;
  if (e >= 31) return sign | 0x7bff;
  if (e <= 0) {
    if (e < -10) return sign;
    const full = mant | 0x800000;
    const shift = 14 - e;
    let half = full >>> shift;
    const rem = full & ((1 << shift) - 1);
    const halfway = 1 << (shift - 1);
    if (rem > halfway || (rem === halfway && (half & 1) === 1)) half++;
    return sign | half;
  }
  let half = (e << 10) | (mant >>> 13);
  const rem = mant & 0x1fff;
  if (rem > 0x1000 || (rem === 0x1000 && (half & 1) === 1)) half++;
  if (half >= 0x7c00) half = 0x7bff;
  return sign | half;
}

export function f16BitsToF32(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1;
  const exp = (bits >>> 10) & 0x1f;
  const mant = bits & 0x3ff;
  if (exp === 0) return sign * mant * 2 ** -24;
  if (exp === 31) return mant === 0 ? sign * Infinity : NaN;
  return sign * (1 + mant / 1024) * 2 ** (exp - 15);
}

/** unorm16 integer of a value clamped to [0, 1]. */
export function toUnorm16(value: number): number {
  return Math.round(Math.min(1, Math.max(0, value)) * 65535);
}
export function fromUnorm16(q: number): number {
  return Math.fround(q * UNORM16_SCALE);
}

function finite(value: number, name: string): number {
  if (!Number.isFinite(value)) throw new RangeError(`material: ${name} is not finite (${value})`);
  return value;
}

// ---------------------------------------------------------------------------------------------------------------
// Pack and unpack (the unpack mirrors WGSL `rtMaterialAt` word for word)
// ---------------------------------------------------------------------------------------------------------------

function slotsOf(m: RtMaterial): Record<number, number> {
  const s: Record<number, number> = {};
  const put = (slot: number, value: number, name: string) => {
    s[slot] = toUnorm16(finite(value, name));
  };
  const put3 = (slot: number, v: readonly number[], name: string) => v.forEach((c, k) => put(slot + k, c, `${name}[${k}]`));
  put(U16_SLOT.baseWeight, m.baseWeight, 'baseWeight');
  put(U16_SLOT.metalness, m.metalness, 'metalness');
  put(U16_SLOT.specularRoughness, m.specularRoughness, 'specularRoughness');
  put(U16_SLOT.specularWeight, m.specularWeight, 'specularWeight');
  put3(U16_SLOT.specularColor, m.specularColor, 'specularColor');
  put(U16_SLOT.specularAnisotropy, m.specularAnisotropy, 'specularAnisotropy');
  put(U16_SLOT.anisotropyRotation, m.anisotropyRotation, 'anisotropyRotation');
  put(U16_SLOT.transmissionWeight, m.transmissionWeight, 'transmissionWeight');
  put3(U16_SLOT.transmissionColor, m.transmissionColor, 'transmissionColor');
  put(U16_SLOT.coatWeight, m.coatWeight, 'coatWeight');
  put(U16_SLOT.coatRoughness, m.coatRoughness, 'coatRoughness');
  put3(U16_SLOT.coatColor, m.coatColor, 'coatColor');
  put(U16_SLOT.sheenWeight, m.sheenWeight, 'sheenWeight');
  put(U16_SLOT.sheenRoughness, m.sheenRoughness, 'sheenRoughness');
  put3(U16_SLOT.sheenColor, m.sheenColor, 'sheenColor');
  put(U16_SLOT.opacity, m.opacity, 'opacity');
  put(U16_SLOT.alphaCutoff, m.alphaCutoff, 'alphaCutoff');
  return s;
}

function textureSlot(value: number, name: string): number {
  if (!Number.isInteger(value) || value < 0 || value > TEXTURE_SLOT_NONE) throw new RangeError(`material: ${name} must be an integer 0-${TEXTURE_SLOT_NONE} (${TEXTURE_SLOT_NONE} = none), got ${value}`);
  return value;
}

/** Writes one 32-word record at `words[wordOffset]`. Non-finite numbers throw; out-of-range weights clamp. */
export function packMaterial(material: RtMaterial, words: Uint32Array, wordOffset = 0): void {
  const w = (i: number, v: number) => {
    words[wordOffset + i] = v >>> 0;
  };
  for (let k = 0; k < 3; k++) {
    w(MATERIAL_WORD.baseColor + k, f32Bits(finite(material.baseColor[k] as number, 'baseColor')));
    w(MATERIAL_WORD.emissionColor + k, f32Bits(finite(material.emissionColor[k] as number, 'emissionColor')));
  }
  w(MATERIAL_WORD.ior, f32Bits(finite(material.ior, 'ior')));
  w(MATERIAL_WORD.emissionLuminance, f32Bits(finite(material.emissionLuminance, 'emissionLuminance')));
  w(MATERIAL_WORD.transmissionDepth, f32Bits(finite(material.transmissionDepth, 'transmissionDepth')));
  w(MATERIAL_WORD.coatIor, f32Bits(finite(material.coatIor, 'coatIor')));
  if (!Number.isInteger(material.alphaMode) || material.alphaMode < 0 || material.alphaMode > 2) throw new RangeError(`material: alphaMode must be 0, 1 or 2, got ${material.alphaMode}`);
  w(
    MATERIAL_WORD.flags,
    material.alphaMode |
      (material.thinWalled ? MATERIAL_FLAG.thinWalled : 0) |
      (material.conductorEtaK ? MATERIAL_FLAG.conductorEtaK : 0) |
      (material.emissionTwoSided ? MATERIAL_FLAG.emissionTwoSided : 0) |
      (material.doubleSided ? MATERIAL_FLAG.doubleSided : 0),
  );
  w(MATERIAL_WORD.texturesA, textureSlot(material.baseColorTexture, 'baseColorTexture') | (textureSlot(material.roughnessMetalTexture, 'roughnessMetalTexture') << 16));
  w(MATERIAL_WORD.texturesB, textureSlot(material.normalTexture, 'normalTexture') | (textureSlot(material.emissionTexture, 'emissionTexture') << 16));
  const half = [...material.conductorEta, ...material.conductorK].map((v, i) => f32ToF16Bits(finite(v, i < 3 ? 'conductorEta' : 'conductorK')));
  for (let i = 0; i < 3; i++) w(MATERIAL_WORD.conductor + i, (half[i * 2] as number) | ((half[i * 2 + 1] as number) << 16));
  const slots = slotsOf(material);
  for (let i = 0; i < 13; i++) w(MATERIAL_WORD.unorm + i, (slots[i * 2] ?? 0) | ((slots[i * 2 + 1] ?? 0) << 16));
  for (let i = 0; i < MATERIAL_RESERVED_WORDS.count; i++) w(MATERIAL_RESERVED_WORDS.first + i, 0);
}

/** Reads one record. This is the TypeScript twin of the WGSL `rtMaterialAt`: the GPU decodes the same bits to the same values. */
export function unpackMaterial(words: ArrayLike<number>, wordOffset = 0): RtMaterial {
  const w = (i: number) => (words[wordOffset + i] as number) >>> 0;
  const f = (i: number) => bitsF32(w(i));
  const unorm = (slot: number) => fromUnorm16((w(MATERIAL_WORD.unorm + (slot >> 1)) >>> ((slot & 1) * 16)) & 0xffff);
  const unorm3 = (slot: number): Rgb3 => [unorm(slot), unorm(slot + 1), unorm(slot + 2)];
  const half = (i: number) => f16BitsToF32((w(MATERIAL_WORD.conductor + (i >> 1)) >>> ((i & 1) * 16)) & 0xffff);
  const flags = w(MATERIAL_WORD.flags);
  return {
    baseColor: [f(0), f(1), f(2)],
    baseWeight: unorm(U16_SLOT.baseWeight),
    metalness: unorm(U16_SLOT.metalness),
    specularRoughness: unorm(U16_SLOT.specularRoughness),
    specularWeight: unorm(U16_SLOT.specularWeight),
    specularColor: unorm3(U16_SLOT.specularColor),
    ior: f(MATERIAL_WORD.ior),
    specularAnisotropy: unorm(U16_SLOT.specularAnisotropy),
    anisotropyRotation: unorm(U16_SLOT.anisotropyRotation),
    conductorEta: [half(0), half(1), half(2)],
    conductorK: [half(3), half(4), half(5)],
    transmissionWeight: unorm(U16_SLOT.transmissionWeight),
    transmissionColor: unorm3(U16_SLOT.transmissionColor),
    transmissionDepth: f(MATERIAL_WORD.transmissionDepth),
    coatWeight: unorm(U16_SLOT.coatWeight),
    coatColor: unorm3(U16_SLOT.coatColor),
    coatRoughness: unorm(U16_SLOT.coatRoughness),
    coatIor: f(MATERIAL_WORD.coatIor),
    sheenWeight: unorm(U16_SLOT.sheenWeight),
    sheenColor: unorm3(U16_SLOT.sheenColor),
    sheenRoughness: unorm(U16_SLOT.sheenRoughness),
    emissionColor: [f(4), f(5), f(6)],
    emissionLuminance: f(MATERIAL_WORD.emissionLuminance),
    opacity: unorm(U16_SLOT.opacity),
    alphaCutoff: unorm(U16_SLOT.alphaCutoff),
    alphaMode: flags & MATERIAL_FLAG.alphaModeMask,
    thinWalled: (flags & MATERIAL_FLAG.thinWalled) !== 0,
    conductorEtaK: (flags & MATERIAL_FLAG.conductorEtaK) !== 0,
    emissionTwoSided: (flags & MATERIAL_FLAG.emissionTwoSided) !== 0,
    doubleSided: (flags & MATERIAL_FLAG.doubleSided) !== 0,
    baseColorTexture: w(MATERIAL_WORD.texturesA) & 0xffff,
    roughnessMetalTexture: w(MATERIAL_WORD.texturesA) >>> 16,
    normalTexture: w(MATERIAL_WORD.texturesB) & 0xffff,
    emissionTexture: w(MATERIAL_WORD.texturesB) >>> 16,
  };
}

/** GGX alpha of a material's specular lobe (contract `roughnessToAlpha`: roughness^2). */
export function materialAlpha(material: Pick<RtMaterial, 'specularRoughness'>): number {
  return Math.fround(material.specularRoughness * material.specularRoughness);
}

/** A material table: the caller's materials, then one default record (index `materials.length`) for ids out of range or `RT_NO_MATERIAL`. */
export function packMaterialTable(materials: readonly RtMaterial[], fallback: RtMaterial = defaultMaterial()): Uint32Array {
  const words = new Uint32Array((materials.length + 1) * MATERIAL_RECORD_WORDS);
  materials.forEach((m, i) => packMaterial(m, words, i * MATERIAL_RECORD_WORDS));
  packMaterial(fallback, words, materials.length * MATERIAL_RECORD_WORDS);
  return words;
}
