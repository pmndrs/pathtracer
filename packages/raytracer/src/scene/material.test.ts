import { describe, expect, it } from 'vitest';
import { roughnessToAlpha } from '../contract/conventions.ts';
import { TEXTURE_SLOT_NONE } from './constants.ts';
import {
  ALPHA_MODE,
  MATERIAL_RECORD_BYTES,
  MATERIAL_RECORD_VEC4S,
  MATERIAL_RECORD_WORDS,
  MATERIAL_RESERVED_WORDS,
  MATERIAL_WORD,
  U16_SLOT,
  UNORM16_SCALE,
  createMaterial,
  defaultMaterial,
  f16BitsToF32,
  f32ToF16Bits,
  fromUnorm16,
  luminance,
  materialAlpha,
  packMaterial,
  packMaterialTable,
  toUnorm16,
  unpackMaterial,
  type RtMaterial,
} from './material.ts';

const UNORM_HALF_STEP = 0.5 / 65535;

/** A material with a different, mid-range value in every field. */
function distinct(): RtMaterial {
  return createMaterial({
    baseColor: [0.123456789, 0.5, 0.987654321],
    baseWeight: 0.9,
    metalness: 0.25,
    specularRoughness: 0.37,
    specularWeight: 0.6,
    specularColor: [0.1, 0.2, 0.3],
    ior: 1.4567,
    specularAnisotropy: 0.45,
    anisotropyRotation: 0.65,
    conductorEta: [0.2, 0.924, 1.102],
    conductorK: [3.912, 2.452, 2.142],
    transmissionWeight: 0.75,
    transmissionColor: [0.4, 0.5, 0.6],
    transmissionDepth: 2.5,
    coatWeight: 0.33,
    coatColor: [0.7, 0.8, 0.9],
    coatRoughness: 0.12,
    coatIor: 1.6,
    sheenWeight: 0.21,
    sheenColor: [0.31, 0.41, 0.51],
    sheenRoughness: 0.81,
    emissionColor: [12.5, 3.25, 0.0625],
    opacity: 0.55,
    alphaCutoff: 0.35,
    alphaMode: ALPHA_MODE.mask,
    thinWalled: true,
    conductorEtaK: true,
    emissionTwoSided: true,
    doubleSided: true,
    baseColorTexture: 7,
    roughnessMetalTexture: 0x1234,
    normalTexture: 0,
    emissionTexture: 0xfffe,
  });
}

const f32 = Math.fround;
const UNORM_FIELDS = ['baseWeight', 'metalness', 'specularRoughness', 'specularWeight', 'specularAnisotropy', 'anisotropyRotation', 'transmissionWeight', 'coatWeight', 'coatRoughness', 'sheenWeight', 'sheenRoughness', 'opacity', 'alphaCutoff'] as const;
const UNORM_COLORS = ['specularColor', 'transmissionColor', 'coatColor', 'sheenColor'] as const;
const F32_FIELDS = ['ior', 'transmissionDepth', 'coatIor', 'emissionLuminance'] as const;
const F32_COLORS = ['baseColor', 'emissionColor'] as const;
const HALF_COLORS = ['conductorEta', 'conductorK'] as const;
const FLAG_FIELDS = ['thinWalled', 'conductorEtaK', 'emissionTwoSided', 'doubleSided'] as const;
const TEXTURE_FIELDS = ['baseColorTexture', 'roughnessMetalTexture', 'normalTexture', 'emissionTexture'] as const;

function roundTrip(m: RtMaterial): RtMaterial {
  const words = new Uint32Array(MATERIAL_RECORD_WORDS);
  packMaterial(m, words);
  return unpackMaterial(words);
}

describe('half floats', () => {
  it('every finite half decodes and re-encodes to itself', () => {
    for (let bits = 0; bits < 0x10000; bits++) {
      if (((bits >>> 10) & 0x1f) === 31) continue;
      const value = f16BitsToF32(bits);
      expect(f32ToF16Bits(value), `0x${bits.toString(16)}`).toBe(bits);
    }
  });

  it('rounds to nearest even, handles subnormals and clamps overflow', () => {
    expect(f32ToF16Bits(1)).toBe(0x3c00);
    expect(f32ToF16Bits(-2)).toBe(0xc000);
    expect(f32ToF16Bits(65504)).toBe(0x7bff);
    expect(f32ToF16Bits(1e9)).toBe(0x7bff);
    expect(f32ToF16Bits(-1e9)).toBe(0xfbff);
    expect(f32ToF16Bits(2 ** -24)).toBe(1);
    expect(f32ToF16Bits(2 ** -25)).toBe(0); // tie between 0 and the smallest subnormal rounds to even (0)
    expect(f32ToF16Bits(1.5 * 2 ** -24)).toBe(2); // tie between 1 and 2 rounds to even (2)
    expect(f32ToF16Bits(2 ** -26)).toBe(0);
    expect(f32ToF16Bits(1 + 2 ** -11)).toBe(0x3c00); // tie 1.0 / next: even is 1.0
    expect(f32ToF16Bits(1 + 3 * 2 ** -11)).toBe(0x3c02); // tie between odd mantissa 1 and 2: even is 2
    expect(f32ToF16Bits(-0)).toBe(0x8000);
  });

  it('half error is within 2^-11 relative for the eta/k range of real metals', () => {
    for (const v of [0.05, 0.2, 0.924, 1.102, 2.142, 3.912, 7.5, 50, 1000]) {
      expect(Math.abs(f16BitsToF32(f32ToF16Bits(v)) - v) / v).toBeLessThan(2 ** -11);
    }
  });
});

describe('unorm16', () => {
  it('0 and 1 are exact, the scale is the f32 of 1/65535, error is at most half a step', () => {
    expect(fromUnorm16(toUnorm16(0))).toBe(0);
    expect(fromUnorm16(toUnorm16(1))).toBe(1);
    expect(UNORM16_SCALE).toBe(f32(1 / 65535));
    expect(f32(65535 * UNORM16_SCALE)).toBe(1);
    for (let i = 0; i <= 1000; i++) {
      const x = i / 1000;
      expect(Math.abs(fromUnorm16(toUnorm16(x)) - x)).toBeLessThanOrEqual(UNORM_HALF_STEP + 1e-7);
    }
  });

  it('clamps out-of-range values', () => {
    expect(toUnorm16(-3)).toBe(0);
    expect(toUnorm16(7)).toBe(65535);
  });
});

describe('record layout', () => {
  it('is 128 bytes: 8 vec4u, 32 words', () => {
    expect(MATERIAL_RECORD_BYTES).toBe(128);
    expect(MATERIAL_RECORD_WORDS * 4).toBe(MATERIAL_RECORD_BYTES);
    expect(MATERIAL_RECORD_VEC4S * 16).toBe(MATERIAL_RECORD_BYTES);
  });

  it('the 25 unorm16 values plus one spare fill words 16-28 and the reserved words are 3', () => {
    expect(Math.max(...Object.values(U16_SLOT)) + 3).toBeLessThanOrEqual(25 + 2); // last colour slot + its two followers
    expect(U16_SLOT.alphaCutoff).toBe(24);
    expect(MATERIAL_WORD.unorm + Math.ceil(25 / 2)).toBe(MATERIAL_RESERVED_WORDS.first);
    expect(MATERIAL_RESERVED_WORDS.first + MATERIAL_RESERVED_WORDS.count).toBe(MATERIAL_RECORD_WORDS);
  });

  it('writes every field into a distinct place and leaves the reserved words zero', () => {
    const words = new Uint32Array(MATERIAL_RECORD_WORDS).fill(0xdeadbeef);
    packMaterial(distinct(), words);
    for (let i = 0; i < MATERIAL_RESERVED_WORDS.count; i++) expect(words[MATERIAL_RESERVED_WORDS.first + i]).toBe(0);
    // The spare unorm half (high half of word 28) is 0, not left from the fill.
    expect(words[28]! >>> 16).toBe(0);
    // Changing any one field changes the record (no two fields alias).
    const base = distinct();
    const reference = Array.from(words);
    const mutate = (patch: Partial<RtMaterial>): number[] => {
      const out = new Uint32Array(MATERIAL_RECORD_WORDS);
      packMaterial({ ...base, ...patch }, out);
      return Array.from(out);
    };
    const touched = new Map<string, number>();
    const changes: [string, Partial<RtMaterial>][] = [];
    for (const k of [...UNORM_FIELDS, 'ior', 'transmissionDepth', 'coatIor', 'emissionLuminance'] as const) changes.push([k, { [k]: (base[k] as number) > 0.5 ? 0.01 : 0.99 }]);
    for (const k of [...UNORM_COLORS, ...F32_COLORS, ...HALF_COLORS]) for (let c = 0; c < 3; c++) changes.push([`${k}[${c}]`, { [k]: (base[k] as number[]).map((v, j) => (j === c ? (v > 0.5 ? 0.01 : 0.99) : v)) }]);
    for (const k of FLAG_FIELDS) changes.push([k, { [k]: !base[k] }]);
    changes.push(['alphaMode', { alphaMode: ALPHA_MODE.blend }]);
    for (const k of TEXTURE_FIELDS) changes.push([k, { [k]: 99 }]);
    for (const [name, patch] of changes) {
      const next = mutate(patch);
      const differing = next.map((w, i) => (w !== reference[i] ? i : -1)).filter((i) => i >= 0);
      expect(differing.length, name).toBeGreaterThan(0);
      touched.set(name, differing.length);
    }
    expect(touched.size).toBe(changes.length);
  });
});

describe('pack / unpack round trip', () => {
  const m = distinct();
  const back = roundTrip(m);

  it('f32 fields are bit exact', () => {
    for (const k of F32_FIELDS) expect(back[k], k).toBe(f32(m[k]));
    for (const k of F32_COLORS) expect(back[k], k).toEqual(m[k].map(f32));
  });

  it('unorm16 fields are within half a step', () => {
    for (const k of UNORM_FIELDS) expect(Math.abs((back[k] as number) - (m[k] as number)), k).toBeLessThanOrEqual(UNORM_HALF_STEP + 1e-7);
    for (const k of UNORM_COLORS) back[k].forEach((v, i) => expect(Math.abs(v - m[k][i]!), `${k}[${i}]`).toBeLessThanOrEqual(UNORM_HALF_STEP + 1e-7));
  });

  it('half fields are within 2^-11 relative', () => {
    for (const k of HALF_COLORS) back[k].forEach((v, i) => expect(Math.abs(v - m[k][i]!) / m[k][i]!, `${k}[${i}]`).toBeLessThan(2 ** -11));
  });

  it('flags, alpha mode and texture slots are exact', () => {
    for (const k of FLAG_FIELDS) expect(back[k], k).toBe(m[k]);
    expect(back.alphaMode).toBe(ALPHA_MODE.mask);
    for (const k of TEXTURE_FIELDS) expect(back[k], k).toBe(m[k]);
  });

  it('all flag combinations, alpha modes and texture extremes survive', () => {
    for (let bits = 0; bits < 16; bits++)
      for (const alphaMode of [0, 1, 2]) {
        const mat = createMaterial({ thinWalled: !!(bits & 1), conductorEtaK: !!(bits & 2), emissionTwoSided: !!(bits & 4), doubleSided: !!(bits & 8), alphaMode });
        const out = roundTrip(mat);
        expect([out.thinWalled, out.conductorEtaK, out.emissionTwoSided, out.doubleSided, out.alphaMode]).toEqual([!!(bits & 1), !!(bits & 2), !!(bits & 4), !!(bits & 8), alphaMode]);
      }
    for (const slot of [0, 1, 0xfffe, TEXTURE_SLOT_NONE]) {
      const out = roundTrip(createMaterial({ baseColorTexture: slot, roughnessMetalTexture: TEXTURE_SLOT_NONE - slot, normalTexture: slot, emissionTexture: slot }));
      expect([out.baseColorTexture, out.roughnessMetalTexture, out.normalTexture, out.emissionTexture]).toEqual([slot, TEXTURE_SLOT_NONE - slot, slot, slot]);
    }
    expect(roundTrip(defaultMaterial()).baseColorTexture).toBe(TEXTURE_SLOT_NONE);
  });

  it('edge values: zeros, ones, tiny and huge', () => {
    const zero = roundTrip(
      createMaterial({ baseColor: [0, 0, 0], baseWeight: 0, specularRoughness: 0, metalness: 0, ior: 1, transmissionDepth: 0, conductorEta: [0, 0, 0], conductorK: [0, 0, 0], emissionColor: [0, 0, 0] }),
    );
    expect(zero.specularRoughness).toBe(0);
    expect(zero.baseWeight).toBe(0);
    expect(zero.conductorK).toEqual([0, 0, 0]);
    expect(zero.emissionLuminance).toBe(0);
    const one = roundTrip(createMaterial({ baseWeight: 1, metalness: 1, specularRoughness: 1, specularWeight: 1, transmissionWeight: 1, coatWeight: 1, sheenWeight: 1, opacity: 1, specularColor: [1, 1, 1] }));
    for (const k of ['baseWeight', 'metalness', 'specularRoughness', 'specularWeight', 'transmissionWeight', 'coatWeight', 'sheenWeight', 'opacity'] as const) expect(one[k], k).toBe(1);
    expect(one.specularColor).toEqual([1, 1, 1]);
    const big = roundTrip(createMaterial({ baseColor: [3e38, 1e-38, 0.5], emissionColor: [1e6, 65504, 1], ior: 3.4, transmissionDepth: 1e9, conductorK: [65504, 1e5, 6e-8] }));
    expect(big.baseColor).toEqual([f32(3e38), f32(1e-38), 0.5]);
    expect(big.emissionColor).toEqual([1e6, 65504, 1]);
    expect(big.transmissionDepth).toBe(f32(1e9));
    expect(big.conductorK[0]).toBe(65504);
    expect(big.conductorK[1]).toBe(65504); // clamped
    expect(big.conductorK[2]).toBeGreaterThan(0); // a half subnormal survives
    const clamped = roundTrip(createMaterial({ metalness: 4, baseWeight: -1 }));
    expect([clamped.metalness, clamped.baseWeight]).toEqual([1, 0]);
  });

  it('packing the decoded record gives back the same words (the format is a fixed point)', () => {
    const words = new Uint32Array(MATERIAL_RECORD_WORDS);
    let seed = 12345;
    const random = () => {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
      return seed / 2 ** 32;
    };
    for (let n = 0; n < 200; n++) {
      const rnd = () => random();
      const mat = createMaterial({
        baseColor: [rnd() * 4, rnd(), rnd()],
        baseWeight: rnd(),
        metalness: rnd(),
        specularRoughness: rnd(),
        specularColor: [rnd(), rnd(), rnd()],
        ior: 1 + rnd() * 2,
        conductorEta: [rnd() * 5, rnd() * 5, rnd() * 5],
        conductorK: [rnd() * 9, rnd() * 9, rnd() * 9],
        transmissionDepth: rnd() * 10,
        emissionColor: [rnd() * 20, rnd(), 0],
        alphaMode: Math.floor(rnd() * 3),
        thinWalled: rnd() > 0.5,
        baseColorTexture: Math.floor(rnd() * 65536),
      });
      packMaterial(mat, words);
      const first = Array.from(words);
      packMaterial(unpackMaterial(words), words);
      expect(Array.from(words)).toEqual(first);
    }
  });

  it('rejects non-finite numbers, bad alpha modes and bad texture slots', () => {
    expect(() => roundTrip(createMaterial({ baseColor: [NaN, 0, 0] }))).toThrow(RangeError);
    expect(() => roundTrip(createMaterial({ ior: Infinity }))).toThrow(RangeError);
    expect(() => roundTrip(createMaterial({ metalness: NaN }))).toThrow(RangeError);
    expect(() => roundTrip(createMaterial({ conductorK: [0, Infinity, 0] }))).toThrow(RangeError);
    expect(() => roundTrip(createMaterial({ alphaMode: 3 }))).toThrow(RangeError);
    expect(() => roundTrip(createMaterial({ normalTexture: 0x10000 }))).toThrow(RangeError);
    expect(() => roundTrip(createMaterial({ normalTexture: -1 }))).toThrow(RangeError);
  });
});

describe('createMaterial and the table', () => {
  it('fills OpenPBR defaults, copies colours and derives emission luminance', () => {
    const d = defaultMaterial();
    expect(d.baseColor).toEqual([0.8, 0.8, 0.8]);
    expect(d.ior).toBe(1.5);
    const a = createMaterial({ emissionColor: [1, 2, 3] });
    expect(a.emissionLuminance).toBe(f32(luminance([1, 2, 3])));
    expect(createMaterial({ emissionColor: [1, 2, 3], emissionLuminance: 9 }).emissionLuminance).toBe(9);
    const color: [number, number, number] = [0.1, 0.2, 0.3];
    const b = createMaterial({ baseColor: color });
    color[0] = 9;
    expect(b.baseColor[0]).toBe(0.1);
    expect(createMaterial({}).baseColor).not.toBe(defaultMaterial().baseColor);
  });

  it('the table holds the materials, then the default', () => {
    const words = packMaterialTable([createMaterial({ metalness: 1 }), createMaterial({ opacity: 0.25 })]);
    expect(words.length).toBe(3 * MATERIAL_RECORD_WORDS);
    expect(unpackMaterial(words, 0).metalness).toBe(1);
    expect(unpackMaterial(words, MATERIAL_RECORD_WORDS).opacity).toBeCloseTo(0.25, 4);
    expect(unpackMaterial(words, 2 * MATERIAL_RECORD_WORDS).baseColor).toEqual([f32(0.8), f32(0.8), f32(0.8)]);
  });

  it('alpha is perceptual roughness squared (contract)', () => {
    for (const r of [0, 0.1, 0.5, 1]) expect(materialAlpha({ specularRoughness: r })).toBe(f32(roughnessToAlpha(r)));
  });
});
