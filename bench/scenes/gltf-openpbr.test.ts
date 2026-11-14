import { describe, expect, it } from 'vitest';
import { GLTF_DEFAULTS, gltfMaterialToOpenPbr, gltfMaterialsToOpenPbr } from './gltf-openpbr.ts';
import type { GltfMaterialExtensions, GltfMaterialJson } from './gltf-openpbr.ts';
import {
  OPENPBR_DEFAULTS, classifyOpenPbr, gltfAnisotropyAlphas, gltfAnisotropyToOpenPbr, metresToOpenPbrCm, nmToMicrometres, openPbrAnisotropyAlphas,
  openPbrAnisotropyToGltf, openPbrCmToMetres, resolveOpenPbr,
} from './openpbr.ts';
import { materialFromDesc, materialUsesTexture } from './to-scene-copy.ts';

const map = (ext: GltfMaterialExtensions, base: Partial<GltfMaterialJson> = {}) => gltfMaterialToOpenPbr({ ...base, extensions: ext });
const dielectric = { pbrMetallicRoughness: { baseColorFactor: [0, 0, 0, 1], metallicFactor: 0, roughnessFactor: 0 } } satisfies Partial<GltfMaterialJson>;
const glass = { pbrMetallicRoughness: { metallicFactor: 0, roughnessFactor: 0 } } satisfies Partial<GltfMaterialJson>;

describe('OpenPBR defaults', () => {
  it('match OpenPBR 1.1.2', () => {
    const m = resolveOpenPbr({});
    expect(m).toMatchObject({
      baseWeight: 1, baseColor: [0.8, 0.8, 0.8], baseMetalness: 0, baseDiffuseRoughness: 0, specularWeight: 1, specularColor: [1, 1, 1], specularRoughness: 0.3,
      specularRoughnessAnisotropy: 0, specularIor: 1.5, transmissionWeight: 0, transmissionColor: [1, 1, 1], transmissionDepth: 0, transmissionDispersionScale: 0,
      transmissionDispersionAbbeNumber: 20, coatWeight: 0, coatColor: [1, 1, 1], coatRoughness: 0, coatIor: 1.6, fuzzWeight: 0, fuzzColor: [1, 1, 1], fuzzRoughness: 0.5,
      thinFilmWeight: 0, thinFilmThickness: 0.5, thinFilmIor: 1.4, emissionLuminance: 0, emissionColor: [1, 1, 1], geometryThinWalled: false, geometryOpacity: 1,
    });
  });
  it('resolve copies colours and keeps explicit values', () => {
    const c: [number, number, number] = [0.1, 0.2, 0.3];
    const m = resolveOpenPbr({ baseColor: c, specularRoughness: 0 });
    expect(m.baseColor).toEqual(c);
    expect(m.baseColor).not.toBe(c);
    expect(m.specularRoughness).toBe(0);
    expect(OPENPBR_DEFAULTS.baseColor).toEqual([0.8, 0.8, 0.8]);
  });
});

describe('glTF core', () => {
  it('uses glTF defaults (white, metallic 1, roughness 1, ior 1.5), not OpenPBR ones', () => {
    const r = gltfMaterialToOpenPbr({});
    expect(r.material).toEqual({ type: 'openpbr', baseColor: [1, 1, 1], baseMetalness: 1, specularRoughness: 1, specularIor: 1.5 });
    expect(r.class).toBe('A');
    expect(r.notes).toEqual([]);
    expect(GLTF_DEFAULTS.metallicFactor).toBe(1);
  });
  it('maps the factors, double sided and alpha modes', () => {
    const r = gltfMaterialToOpenPbr({ pbrMetallicRoughness: { baseColorFactor: [0.5, 0.25, 0.125, 0.5], metallicFactor: 0, roughnessFactor: 0.4 }, doubleSided: true, alphaMode: 'BLEND' });
    expect(r.material).toMatchObject({ baseColor: [0.5, 0.25, 0.125], baseMetalness: 0, specularRoughness: 0.4, doubleSided: true, alphaMode: 'blend', geometryOpacity: 0.5 });
    expect(r.class).toBe('C');
    const mask = gltfMaterialToOpenPbr({ alphaMode: 'MASK', alphaCutoff: 0.3, pbrMetallicRoughness: { baseColorFactor: [1, 1, 1, 0.9] } });
    expect(mask.material).toMatchObject({ alphaMode: 'mask', alphaCutoff: 0.3, geometryOpacity: 0.9 });
    const opaque = gltfMaterialToOpenPbr({ pbrMetallicRoughness: { baseColorFactor: [1, 1, 1, 0.5] } });
    expect(opaque.material.alphaMode).toBeUndefined();
    expect(opaque.notes[0]).toMatch(/alpha ignored/);
  });
  it('lists texture slots and unsupported extensions instead of dropping them silently', () => {
    const r = gltfMaterialToOpenPbr({
      pbrMetallicRoughness: { baseColorTexture: { index: 0 }, metallicRoughnessTexture: { index: 1 } }, normalTexture: { index: 2 }, emissiveTexture: { index: 3 },
      extensions: { KHR_materials_pbrSpecularGlossiness: {}, KHR_materials_clearcoat: { clearcoatNormalTexture: { index: 4 } } },
    });
    expect(r.textures).toEqual(['normalTexture', 'emissiveTexture', 'baseColorTexture', 'metallicRoughnessTexture', 'clearcoatNormalTexture']);
    expect(r.unsupported).toEqual(['KHR_materials_pbrSpecularGlossiness']);
    expect(r.notes.join('\n')).toMatch(/unsupported extensions: KHR_materials_pbrSpecularGlossiness/);
  });
  it('takes the extensions argument over material.extensions', () => {
    const r = gltfMaterialToOpenPbr({ extensions: { KHR_materials_ior: { ior: 2 } } }, { KHR_materials_ior: { ior: 1.2 } });
    expect(r.material.specularIor).toBe(1.2);
  });
});

describe('extension defaults', () => {
  it('KHR_materials_ior: default 1.5, clamps below 1', () => {
    expect(map({ KHR_materials_ior: {} }).material.specularIor).toBe(1.5);
    expect(map({ KHR_materials_ior: { ior: 2.42 } }).material.specularIor).toBe(2.42);
    const low = map({ KHR_materials_ior: { ior: 0 } });
    expect(low.material.specularIor).toBe(1);
    expect(low.notes[0]).toMatch(/clamped/);
  });
  it('KHR_materials_specular: factor 1, colour 1; a dielectric below that is class C, a metal is not', () => {
    const d = map({ KHR_materials_specular: {} }, { pbrMetallicRoughness: { metallicFactor: 0 } });
    expect(d.material).toMatchObject({ specularWeight: 1, specularColor: [1, 1, 1] });
    expect(d.notes).toEqual([]);
    const half = map({ KHR_materials_specular: { specularFactor: 0.5 } }, dielectric);
    expect(half.material.specularWeight).toBe(0.5);
    expect(half.class).toBe('C');
    expect(half.notes[0]).toMatch(/F0 only/);
    expect(map({ KHR_materials_specular: { specularColorFactor: [1, 0.5, 0.5] } }, dielectric).class).toBe('C');
    expect(map({ KHR_materials_specular: { specularFactor: 0.5 } }).class).toBe('A'); // metallic default 1
  });
  it('KHR_materials_transmission: factor 0 is no transmission; no volume means thin-walled', () => {
    const none = map({ KHR_materials_transmission: {} }, glass);
    expect(none.material.transmissionWeight).toBe(0);
    expect(none.material.geometryThinWalled).toBeUndefined();
    const thin = map({ KHR_materials_transmission: { transmissionFactor: 1 } }, glass);
    expect(thin.material).toMatchObject({ transmissionWeight: 1, geometryThinWalled: true, transmissionColor: [1, 1, 1] });
    expect(thin.class).toBe('A');
    expect(map({ KHR_materials_transmission: { transmissionFactor: 1 } }, { pbrMetallicRoughness: { metallicFactor: 0, roughnessFactor: 0.3 } }).class).toBe('C'); // thin-walled rough
    expect(map({ KHR_materials_transmission: { transmissionFactor: 0.5 } }, glass).class).toBe('B');
  });
  it('KHR_materials_volume: thickness 0 stays thin; attenuation is metres and OpenPBR centimetres are x100', () => {
    const thin = map({ KHR_materials_transmission: { transmissionFactor: 1 }, KHR_materials_volume: { attenuationDistance: 0.5, attenuationColor: [1, 0.5, 0.25] } }, glass);
    expect(thin.material.geometryThinWalled).toBe(true);
    expect(thin.material.transmissionDepth).toBeUndefined();
    expect(thin.notes.join()).toMatch(/thin-walled surface/);
    const thick = map({ KHR_materials_transmission: { transmissionFactor: 1 }, KHR_materials_volume: { thicknessFactor: 0.1, attenuationDistance: 0.5, attenuationColor: [1, 0.5, 0.25] } }, glass);
    expect(thick.material).toMatchObject({ transmissionDepth: 0.5, transmissionColor: [1, 0.5, 0.25] });
    expect(thick.material.geometryThinWalled).toBeUndefined();
    expect(thick.class).toBe('A');
    expect(metresToOpenPbrCm(thick.material.transmissionDepth!)).toBe(50);
    expect(openPbrCmToMetres(50)).toBe(0.5);
    // no attenuationDistance = infinite: no medium, colour is the base tint at depth 0
    const clear = map({ KHR_materials_transmission: { transmissionFactor: 1 }, KHR_materials_volume: { thicknessFactor: 1 } }, glass);
    expect(clear.material.transmissionDepth).toBeUndefined();
    expect(clear.class).toBe('A');
  });
  it('baseColor tint: a tint at depth 0 (class B), dropped next to a volume (class B + note)', () => {
    const tintBase = { pbrMetallicRoughness: { baseColorFactor: [1, 0.5, 0.5, 1], metallicFactor: 0, roughnessFactor: 0 } };
    const tint = map({ KHR_materials_transmission: { transmissionFactor: 1 } }, tintBase);
    expect(tint.material.transmissionColor).toEqual([1, 0.5, 0.5]);
    expect(tint.class).toBe('B');
    const both = map({ KHR_materials_transmission: { transmissionFactor: 1 }, KHR_materials_volume: { thicknessFactor: 1, attenuationDistance: 1, attenuationColor: [0.5, 1, 1] } }, tintBase);
    expect(both.material.transmissionColor).toEqual([0.5, 1, 1]);
    expect(both.class).toBe('B');
    expect(both.notes.join()).toMatch(/tint dropped/);
  });
  it('KHR_materials_clearcoat: factor 0, roughness 0, ior 1.5', () => {
    const r = map({ KHR_materials_clearcoat: {} });
    expect(r.material).toMatchObject({ coatWeight: 0, coatRoughness: 0, coatIor: 1.5 });
    expect(r.class).toBe('A');
    const on = map({ KHR_materials_clearcoat: { clearcoatFactor: 1, clearcoatRoughnessFactor: 0.2 } });
    expect(on.material).toMatchObject({ coatWeight: 1, coatRoughness: 0.2, coatIor: 1.5 });
    expect(on.class).toBe('B');
  });
  it('KHR_materials_sheen: black colour is no sheen; any colour is class C', () => {
    const off = map({ KHR_materials_sheen: {} });
    expect(off.material).toMatchObject({ fuzzWeight: 0, fuzzColor: [0, 0, 0], fuzzRoughness: 0 });
    expect(off.class).toBe('A');
    const on = map({ KHR_materials_sheen: { sheenColorFactor: [1, 0.8, 0.6], sheenRoughnessFactor: 0.4 } });
    expect(on.material).toMatchObject({ fuzzWeight: 1, fuzzColor: [1, 0.8, 0.6], fuzzRoughness: 0.4 });
    expect(on.class).toBe('C');
  });
  it('KHR_materials_iridescence: ior 1.3, thickness is the maximum in micrometres (nm / 1000)', () => {
    const d = map({ KHR_materials_iridescence: {} });
    expect(d.material).toMatchObject({ thinFilmWeight: 0, thinFilmIor: 1.3, thinFilmThickness: 0.4 });
    expect(d.class).toBe('A');
    const on = map({ KHR_materials_iridescence: { iridescenceFactor: 1, iridescenceIor: 1.8, iridescenceThicknessMinimum: 250, iridescenceThicknessMaximum: 1200 } });
    expect(on.material).toMatchObject({ thinFilmWeight: 1, thinFilmIor: 1.8, thinFilmThickness: 1.2 });
    expect(on.class).toBe('C');
    expect(nmToMicrometres(550)).toBe(0.55);
    const tex = map({ KHR_materials_iridescence: { iridescenceFactor: 1, iridescenceThicknessTexture: { index: 0 } } });
    expect(tex.textures).toContain('iridescenceThicknessTexture');
    expect(tex.notes.join()).toMatch(/100\.\.400 nm/);
  });
  it('KHR_materials_anisotropy: strength 0 changes nothing; otherwise roughness and anisotropy are converted', () => {
    const none = map({ KHR_materials_anisotropy: {} }, { pbrMetallicRoughness: { roughnessFactor: 0.5 } });
    expect(none.material.specularRoughness).toBe(0.5);
    expect(none.material.specularRoughnessAnisotropy).toBeUndefined();
    const s = map({ KHR_materials_anisotropy: { anisotropyStrength: 0.5, anisotropyRotation: 1.25 } }, { pbrMetallicRoughness: { roughnessFactor: 0.5 } });
    // glTF: alpha 0.25, alpha_t = 0.25 + 0.75 * 0.25 = 0.4375, alpha_b = 0.25 -> a = 1 - 0.25/0.4375 = 3/7, r = sqrt(0.4375 sqrt(65/98))
    expect(s.material.specularRoughnessAnisotropy).toBeCloseTo(3 / 7, 12);
    expect(s.material.specularRoughness).toBeCloseTo(0.59691274098, 10);
    expect(s.material.anisotropyRotation).toBe(1.25);
    expect(s.class).toBe('A');
    expect(map({ KHR_materials_anisotropy: { anisotropyStrength: 0.5 } }, { pbrMetallicRoughness: { metallicFactor: 0 } }).class).toBe('B');
  });
  it('KHR_materials_emissive_strength: default 1, multiplies the factor', () => {
    const e = { emissiveFactor: [0.5, 0.25, 0] };
    expect(gltfMaterialToOpenPbr(e).material).toMatchObject({ emissionColor: [0.5, 0.25, 0], emissionLuminance: 1 });
    expect(map({ KHR_materials_emissive_strength: {} }, e).material.emissionLuminance).toBe(1);
    const r = map({ KHR_materials_emissive_strength: { emissiveStrength: 4 } }, e);
    expect(r.material).toMatchObject({ emissionColor: [0.5, 0.25, 0], emissionLuminance: 4 });
    expect(materialFromDesc(r.material).emissionColor).toEqual([2, 1, 0]);
    expect(gltfMaterialToOpenPbr({}).material.emissionLuminance).toBeUndefined(); // factor 0: not emissive
    expect(map({ KHR_materials_emissive_strength: { emissiveStrength: 4 } }).material.emissionColor).toBeUndefined();
  });
  it('KHR_materials_dispersion: Abbe = 20 / dispersion, class B-prime', () => {
    expect(map({ KHR_materials_dispersion: {} }, glass).material.transmissionDispersionScale).toBeUndefined();
    const half = map({ KHR_materials_dispersion: { dispersion: 0.5 }, KHR_materials_transmission: { transmissionFactor: 1 } }, glass);
    expect(half.material).toMatchObject({ transmissionDispersionScale: 1, transmissionDispersionAbbeNumber: 40 });
    expect(half.class).toBe("B'");
    expect(map({ KHR_materials_dispersion: { dispersion: 2 }, KHR_materials_transmission: { transmissionFactor: 1 } }, glass).material.transmissionDispersionAbbeNumber).toBe(10);
    expect(map({ KHR_materials_dispersion: { dispersion: 1 } }, glass).notes.join()).toMatch(/without transmission/);
  });
  it('KHR_materials_diffuse_transmission: factor 0, colour 1; on = thin-walled subsurface, class B-prime', () => {
    expect(map({ KHR_materials_diffuse_transmission: {} }).material).toMatchObject({ subsurfaceWeight: 0, subsurfaceColor: [1, 1, 1] });
    const on = map({ KHR_materials_diffuse_transmission: { diffuseTransmissionFactor: 0.5, diffuseTransmissionColorFactor: [1, 0.5, 0.5] } }, { pbrMetallicRoughness: { metallicFactor: 0 } });
    expect(on.material).toMatchObject({ subsurfaceWeight: 0.5, subsurfaceColor: [1, 0.5, 0.5], geometryThinWalled: true });
    expect(on.class).toBe("B'");
  });
  it('KHR_materials_unlit: a black non-scattering emitter of the base colour', () => {
    const r = map({ KHR_materials_unlit: {} }, { pbrMetallicRoughness: { baseColorFactor: [0.2, 0.4, 0.6, 1], metallicFactor: 0 } });
    expect(r.material).toMatchObject({ baseWeight: 0, specularWeight: 0, emissionColor: [0.2, 0.4, 0.6], emissionLuminance: 1 });
    expect(r.class).toBe('A');
    const sc = materialFromDesc(r.material);
    expect(sc.baseWeight).toBe(0);
    expect(sc.emissionColor).toEqual([0.2, 0.4, 0.6]);
  });
});

describe('anisotropy conversion', () => {
  it('strength 0 is the identity', () => {
    const c = gltfAnisotropyToOpenPbr(0.5, 0);
    expect(c.anisotropy).toBe(0);
    expect(c.roughness).toBeCloseTo(0.5, 15);
  });
  it('hand-computed: roughness 0.5, strength 1 -> alpha_t 1, alpha_b 0.25, a = 0.75, r = sqrt(sqrt(0.53125))', () => {
    const c = gltfAnisotropyToOpenPbr(0.5, 1);
    expect(gltfAnisotropyAlphas(0.5, 1)).toEqual({ alphaT: 1, alphaB: 0.25 });
    expect(c.anisotropy).toBeCloseTo(0.75, 15);
    expect(c.roughness).toBeCloseTo(0.8537382425870722, 12);
  });
  it('roughness 0 with strength s: alpha_b 0, a = 1', () => {
    const c = gltfAnisotropyToOpenPbr(0, 0.5);
    expect(c.anisotropy).toBe(1);
    // alpha_t = 0.25 = r^2 sqrt(2 / (1 + 0)) -> r^2 = 0.25 / sqrt(2)
    expect(c.roughness).toBeCloseTo(Math.sqrt(0.25 / Math.SQRT2), 14);
  });
  it('both models give the same alphas and the round trip returns the glTF values', () => {
    for (const rough of [0, 0.05, 0.3, 0.5, 0.9, 1]) {
      for (const s of [0, 0.1, 0.5, 0.9, 1]) {
        const c = gltfAnisotropyToOpenPbr(rough, s);
        const g = gltfAnisotropyAlphas(rough, s);
        const o = openPbrAnisotropyAlphas(c.roughness, c.anisotropy);
        expect(o.alphaT).toBeCloseTo(g.alphaT, 12);
        expect(o.alphaB).toBeCloseTo(g.alphaB, 12);
        const back = openPbrAnisotropyToGltf(c.roughness, c.anisotropy);
        expect(back.roughness).toBeCloseTo(rough, 9);
        if (rough < 1) expect(back.strength).toBeCloseTo(s, 7); // roughness 1 has alpha_t = alpha_b = 1 whatever the strength: not invertible
      }
    }
  });
  it('OpenPBR formula at a = 0 is isotropic', () => {
    expect(openPbrAnisotropyAlphas(0.4, 0)).toEqual({ alphaT: 0.16000000000000003, alphaB: 0.16000000000000003 });
  });
});

describe('class assignment (tier1.md section 2)', () => {
  const cls = (p: Parameters<typeof classifyOpenPbr>[0]) => classifyOpenPbr(p).class;
  it('A: metals, black-base dielectric, smooth thick/thin transmission, plain Lambert', () => {
    expect(cls({ baseMetalness: 1, specularRoughness: 0.4 })).toBe('A');
    expect(cls({ baseMetalness: 1, specularRoughness: 0.4, specularRoughnessAnisotropy: 0.5 })).toBe('A');
    expect(cls({ baseMetalness: 0, baseColor: [0, 0, 0], specularRoughness: 0 })).toBe('A');
    expect(cls({ baseMetalness: 0, transmissionWeight: 1, specularRoughness: 0.2, transmissionDepth: 0.1, transmissionColor: [1, 0.5, 0.5] })).toBe('A');
    expect(cls({ baseMetalness: 0, transmissionWeight: 1, geometryThinWalled: true, specularRoughness: 0 })).toBe('A');
    expect(cls({ baseColor: [0.5, 0.5, 0.5], specularWeight: 0 })).toBe('A');
    expect(cls({ baseColor: [0, 0, 0], emissionColor: [1, 1, 1], emissionLuminance: 8 })).toBe('A');
  });
  it('B: dielectric over diffuse, clearcoat, coloured tint, partial transmission', () => {
    expect(cls({ baseColor: [0.8, 0.2, 0.2], specularRoughness: 0.4 })).toBe('B');
    expect(cls({ baseMetalness: 1, coatWeight: 1 })).toBe('B');
    expect(cls({ transmissionWeight: 1, transmissionColor: [1, 0.5, 0.5] })).toBe('B');
    expect(cls({ transmissionWeight: 0.5 })).toBe('B');
    expect(cls({ baseMetalness: 0.5, baseColor: [0, 0, 0] })).toBe('A'); // mix of two exact ends
    expect(cls({ baseMetalness: 0.5 })).toBe('B'); // metal + dielectric-over-diffuse
  });
  it("B': dispersion and diffuse transmission; C: sheen, thin film, specular scale, rough thin-walled, blend", () => {
    expect(cls({ transmissionWeight: 1, transmissionDispersionScale: 1, transmissionDispersionAbbeNumber: 40 })).toBe("B'");
    expect(cls({ subsurfaceWeight: 0.5, geometryThinWalled: true })).toBe("B'");
    expect(cls({ baseMetalness: 1, fuzzWeight: 1 })).toBe('C');
    expect(cls({ baseMetalness: 1, thinFilmWeight: 1 })).toBe('C');
    expect(cls({ baseColor: [0, 0, 0], specularWeight: 0.5 })).toBe('C');
    expect(cls({ transmissionWeight: 1, geometryThinWalled: true, specularRoughness: 0.3 })).toBe('C');
    expect(cls({ alphaMode: 'blend', geometryOpacity: 0.5 })).toBe('C');
    expect(cls({ baseMetalness: 1, alphaMode: 'blend', geometryOpacity: 1 })).toBe('A');
  });
  it('the worst feature wins and reasons are reported', () => {
    const r = classifyOpenPbr({ baseMetalness: 1, coatWeight: 1, fuzzWeight: 1, transmissionDispersionScale: 1 });
    expect(r.class).toBe('C');
    expect(r.reasons.map((x) => x[0])).toEqual(expect.arrayContaining(['C', 'B']));
  });
});

describe('scene copy mapping of the openpbr type', () => {
  it('maps every field the record has, scales emission by luminance, and uses OpenPBR defaults', () => {
    const m = materialFromDesc({ type: 'openpbr' });
    expect(m).toMatchObject({ baseColor: [0.8, 0.8, 0.8], baseWeight: 1, metalness: 0, specularRoughness: 0.3, ior: 1.5, coatIor: 1.6, transmissionWeight: 0, opacity: 1, alphaMode: 0, thinWalled: false });
    const t = materialFromDesc({ type: 'openpbr', baseColor: [1, 0, 0], baseMetalness: 1, specularRoughnessAnisotropy: 0.75, transmissionDepth: 0.5, geometryThinWalled: true, alphaMode: 'mask', alphaCutoff: 0.25, fuzzWeight: 1, emissionColor: [0.5, 0.5, 0], emissionLuminance: 4 });
    expect(t).toMatchObject({ baseColor: [1, 0, 0], metalness: 1, specularAnisotropy: 0.75, transmissionDepth: 0.5, thinWalled: true, alphaMode: 1, alphaCutoff: 0.25, sheenWeight: 1, emissionColor: [2, 2, 0] });
  });
  it('textures resolve through the slot callback and need uvs', () => {
    const d = { type: 'openpbr', baseColorTexture: 'a', normalTexture: 'b' } as const;
    expect(materialUsesTexture(d)).toBe(true);
    expect(materialUsesTexture({ type: 'openpbr' })).toBe(false);
    expect(materialFromDesc(d, (n) => (n === 'a' ? 3 : 4))).toMatchObject({ baseColorTexture: 3, normalTexture: 4 });
  });
});

describe('document helper', () => {
  it('converts every material in order with its name', () => {
    const rs = gltfMaterialsToOpenPbr({ materials: [{ name: 'a' }, { pbrMetallicRoughness: { metallicFactor: 0 } }] });
    expect(rs.map((r) => [r.index, r.name, r.class])).toEqual([[0, 'a', 'A'], [1, undefined, 'B']]);
    expect(gltfMaterialsToOpenPbr({})).toEqual([]);
  });
});
