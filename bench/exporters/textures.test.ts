import { describe, expect, it } from 'vitest';
import { readPFM } from '../metrics/pfm.ts';
import { decodeImage } from '../scenes/image-decode.ts';
import type { OpenPbrParams, SceneDesc } from '../scenes/schema.ts';
import { toGltf, toMitsuba, toPbrt } from './index.ts';
import { srgbToLinear, TextureSet } from './textures.ts';
import { reflectanceToEtaK } from './openpbr.ts';

const opts = { assetsDir: '', spp: 16 };
const text = (r: { files: { path: string; data: Buffer | string }[] }, path: string): string => r.files.find((f) => f.path === path)!.data.toString();
const file = (r: { files: { path: string; data: Buffer | string }[] }, path: string): Buffer => r.files.find((f) => f.path === path)!.data as Buffer;

/** 4 x 2 texels, the G byte of texel i is 10 * i + 5 (roughness), B = 255 - 20 i (metalness), R = 40 i, A = 255 on even texels and 100 on odd ones. */
const mrScene = (m: OpenPbrParams, extra: Partial<SceneDesc> = {}): SceneDesc => {
  return {
    version: 1, name: 'unit-tex', purpose: 'unit', tier: 1,
    camera: { type: 'perspective', position: [0, 0, 4], target: [0, 0, 0], up: [0, 1, 0], yFovDeg: 30 },
    film: { width: 8, height: 8, filter: 'box' }, integrator: { maxDepth: 4 },
    textures: { mr: { generator: 'image', path: 'unit.glb', image: 0, width: 4, height: 2, colorSpace: 'linear', wrap: 'clamp' }, col: { generator: 'image', path: 'unit.glb', image: 1, width: 4, height: 2, colorSpace: 'srgb', wrap: 'repeat' }, many: { generator: 'image', path: 'unit.glb', image: 2, width: 16, height: 1, colorSpace: 'linear', wrap: 'clamp' } },
    materials: { m: { type: 'openpbr', baseColor: [1, 1, 1], ...m } },
    meshes: [{ name: 'q', geometry: { type: 'quad', size: [1, 1] }, material: 'm', uv: {} }],
    environment: { type: 'uniform', radiance: [1, 1, 1] },
    ...extra,
  };
};

import { setImageTextureLoader } from '../scenes/textures.ts';
setImageTextureLoader((d) => {
  if (d.image === 2) {
    // 16 texels with 16 distinct roughness bytes: more than the 8 levels a roughplastic chain covers
    const many = new Uint8Array(16 * 4);
    for (let i = 0; i < 16; i++) many.set([0, 16 * i, 128, 255], i * 4);
    return { width: 16, height: 1, data: many };
  }
  const data = new Uint8Array(4 * 2 * 4);
  for (let i = 0; i < 8; i++) data.set([40 * i, 10 * i + 5, 255 - 20 * i, i % 2 === 0 ? 255 : 100], i * 4);
  return { width: 4, height: 2, data };
});

describe('TextureSet files', () => {
  const set = new TextureSet(mrScene({}));
  const t = set.get('mr');
  it('channel PNGs keep the byte of the channel exactly (pbrt reads a float texture from channel 0)', () => {
    const g = decodeImage(file({ files: [...(set.channel(t, 1), set.files.values())] }, 'tex-mr-g.png'));
    for (let i = 0; i < 8; i++) expect([g.data[i * 4], g.data[i * 4 + 1], g.data[i * 4 + 2]]).toEqual([10 * i + 5, 10 * i + 5, 10 * i + 5]);
  });
  it('colour PNG drops alpha (it travels as its own channel)', () => {
    set.image(t);
    const img = decodeImage(set.files.get('tex-mr.png')!.data as Buffer);
    expect(Array.from({ length: 8 }, (_, i) => img.data[i * 4 + 3])).toEqual(Array(8).fill(255));
    expect(img.data[4]).toBe(40);
  });
  it('rejects mirror wrap (pbrt-v4 has none)', () => {
    const s = mrScene({});
    (s.textures!['mr'] as { wrap: string }).wrap = 'mirror';
    expect(() => new TextureSet(s).get('mr')).toThrow(/mirror/);
  });
  it('mask bitmap: the bilinear alpha ramp 255 -> 100 crosses 0.5 at 82% of the way to the lower texel, on an 8x finer grid', () => {
    const { file: f, factor } = set.maskPng(set.get('col'), 1, 0.5);
    expect(factor).toBe(8);
    const img = decodeImage(set.files.get(f)!.data as Buffer);
    expect([img.width, img.height]).toEqual([32, 16]);
    // row 0 of the source holds alpha 255, 100, 255, 100 (texels 0..3, repeat); fine texel i centre = (i + 0.5) / 8 - 0.5 in source texels
    const row = (j: number) => Array.from({ length: 32 }, (_, i) => img.data[(j * 32 + i) * 4]);
    const expected = Array.from({ length: 32 }, (_, i) => {
      const s = (i + 0.5) / 8 - 0.5, x0 = Math.floor(s), fx = s - x0;
      const a = (x: number) => ((((x % 4) + 4) % 4) % 2 === 0 ? 255 : 100) / 255;
      return a(x0) * (1 - fx) + a(x0 + 1) * fx >= 0.5 ? 255 : 0;
    });
    // both source rows hold the same alpha pattern, so the vertical interpolation changes nothing
    expect(row(0)).toEqual(expected);
    // the cutoff sits where 255 - 155 f = 127.5: f = 0.823; texel 0 is opaque up to fine texel 3 (s = -0.0625 + ... ) and not at fine texel 7
    expect(expected.filter((v) => v === 255).length).toBeGreaterThan(8);
  });
});

describe('pbrt textured openpbr', () => {
  it('metal + dielectric with a metallic-roughness texture: float imagemaps, alpha = (factor G)^2 by a scale texture, metalness as the mix amount', () => {
    const r = toPbrt(mrScene({ baseMetalness: 1, specularRoughness: 0.5, metallicRoughnessTexture: 'mr', baseColorTexture: 'col' }), opts);
    const t = text(r, 'scene.pbrt');
    expect(t).toContain('Texture "m.g" "float" "imagemap" "string filename" "tex-mr-g.png" "string filter" "bilinear" "string wrap" "clamp" "string encoding" "linear" "float scale" 0.5');
    expect(t).toContain('Texture "m.alpharough" "float" "scale" "texture tex" "m.g" "texture scale" "m.g"');
    expect(t).toContain('Texture "m.metal" "float" "imagemap" "string filename" "tex-mr-b.png"');
    expect(t).toContain('Texture "m.base" "spectrum" "imagemap" "string filename" "tex-col.png" "string filter" "bilinear" "string wrap" "repeat" "string encoding" "sRGB"');
    expect(t).toMatch(/"string type" "conductor" "texture reflectance" "m\.base" "texture roughness" "m\.alpharough" "bool remaproughness" false/);
    expect(t).toMatch(/"string type" "mix" "string materials" \[ "m\.coateddiffuse0" "m\.conductor1" \] "texture amount" "m\.metal"/);
    expect(t.indexOf('Texture "m.metal"')).toBeLessThan(t.indexOf('MakeNamedMaterial'));
    for (const f of ['tex-mr-g.png', 'tex-mr-b.png', 'tex-col.png']) expect(r.files.some((x) => x.path === f)).toBe(true);
  });
  it('alpha mask: alpha = 0.5 + K (opacity a - cutoff) is >= 1 for opaque texels, <= 0 for holes, exactly 0.5 on the contour', () => {
    const r = toPbrt(mrScene({ baseColorTexture: 'col', specularWeight: 0, alphaMode: 'mask', alphaCutoff: 0.5, geometryOpacity: 0.8 }), opts);
    const t = text(r, 'scene.pbrt');
    expect(t).toContain('Shape "plymesh" "string filename" "00-q.ply" "texture alpha" "m.cut"');
    const gain = Number(/Texture "m\.a" .*"float scale" ([\d.]+)/.exec(t)![1]);
    const tex1 = Number(/"float tex1" (-?[\d.]+)/.exec(t)![1]);
    const alpha = (a: number): number => 2 * (0.5 * tex1 + 0.5 * gain * a); // mix then scale 2
    expect(gain).toBeCloseTo(800, 9); // K x opacity
    expect(alpha(1)).toBeGreaterThanOrEqual(1); // 0.8 x 1 >= 0.5
    expect(alpha(100 / 255)).toBeLessThanOrEqual(0); // 0.8 x 0.39 = 0.31 < 0.5
    expect(alpha(0.5 / 0.8)).toBeCloseTo(0.5, 9); // contour
    expect(r.files.some((f) => f.path === 'tex-col-a.png')).toBe(true);
  });
  it('a constant mask below its cutoff cuts the shape out; at or above it draws it', () => {
    expect(text(toPbrt(mrScene({ specularWeight: 0, alphaMode: 'mask', geometryOpacity: 0.3 }), opts), 'scene.pbrt')).toContain('"float alpha" 0');
    expect(text(toPbrt(mrScene({ specularWeight: 0, alphaMode: 'mask', geometryOpacity: 0.7 }), opts), 'scene.pbrt')).not.toContain('"float alpha"');
  });
  it('emissive texture: a linear PFM of texel x emission, read by the diffuse area light', () => {
    const r = toPbrt(mrScene({ baseColor: [0, 0, 0], specularWeight: 0, emissiveTexture: 'col', emissionLuminance: 3 }), opts);
    expect(text(r, 'scene.pbrt')).toContain('AreaLightSource "diffuse" "string filename" "tex-col-emis-m.pfm" "bool twosided" false');
    const img = readPFM(file(r, 'tex-col-emis-m.pfm'));
    expect(img.data[3]).toBeCloseTo(srgbToLinear(40) * 3, 6); // texel 1, R
  });
  it('refuses what it cannot draw: normal maps, a base colour factor, anisotropy with a roughness map, an unknown texture (coverage drops the normal map)', () => {
    expect(() => toPbrt(mrScene({ normalTexture: 'mr' }), opts)).toThrow(/texture slot 'normalTexture'/);
    const cov = toPbrt(mrScene({ normalTexture: 'mr' }), { ...opts, coverage: true });
    expect(cov.approximations.join()).toMatch(/coverage: .*normalTexture.* dropped/);
    expect(() => toPbrt(mrScene({ baseColorTexture: 'col', baseColor: [0.5, 1, 1] }), opts)).toThrow(/base colour texture needs baseColor/);
    expect(() => toPbrt(mrScene({ metallicRoughnessTexture: 'mr', specularRoughnessAnisotropy: 0.5 }), { ...opts, coverage: true })).toThrow(/anisotropy/);
    expect(() => toPbrt(mrScene({ baseColorTexture: 'nope' }), opts)).toThrow(/texture slot 'baseColorTexture'/);
    expect(() => toPbrt(mrScene({ baseColorTexture: 'mr' }), opts)).toThrow(/is linear/);
  });
});

describe('Mitsuba textured openpbr', () => {
  it('metallic-roughness: blendbsdf weight = factor x B, roughness alpha = (factor G)^2 as a raw float PFM, conductor k per texel from the base colour', () => {
    const r = toMitsuba(mrScene({ baseMetalness: 1, specularRoughness: 0.5, metallicRoughnessTexture: 'mr', baseColorTexture: 'col' }), opts);
    const x = text(r, 'scene.xml');
    expect(x).toContain('<bsdf type="blendbsdf"><texture type="bitmap" name="weight">');
    expect(x).toContain('tex-mr-metal-1.0.pfm');
    expect(x).toContain('tex-mr-alpha-0.5.pfm');
    expect(x).toContain('<texture type="bitmap" name="k"><string name="filename" value="tex-col-k.pfm"/>');
    expect(x).toContain('<spectrum name="eta" value="1"/>');
    const alpha = readPFM(file(r, 'tex-mr-alpha-0.5.pfm'));
    for (const i of [0, 3, 7]) expect(alpha.data[i * 3]).toBeCloseTo((0.5 * ((10 * i + 5) / 255)) ** 2, 7);
    const w = readPFM(file(r, 'tex-mr-metal-1.0.pfm'));
    expect(w.data[2 * 3]).toBeCloseTo((255 - 40) / 255, 7);
    const k = readPFM(file(r, 'tex-col-k.pfm'));
    const want = reflectanceToEtaK([srgbToLinear(40), srgbToLinear(15), srgbToLinear(235)]); // texel 1
    expect(k.data[3]).toBeCloseTo(want.k[0], 4);
    expect(k.data[5]).toBeCloseTo(want.k[2], 4);
  });
  it('dielectric over diffuse with a roughness texture: a blendbsdf chain of plastics over the texture\'s levels (roughplastic takes a float alpha), weights are 0/255 indicator bitmaps', () => {
    const r = toMitsuba(mrScene({ metallicRoughnessTexture: 'mr', baseColorTexture: 'col', baseMetalness: 0 }), opts);
    const x = text(r, 'scene.xml');
    expect(x.match(/<bsdf type="roughplastic">/g)).toHaveLength(8); // 8 distinct G bytes
    expect(x.match(/<bsdf type="blendbsdf">/g)).toHaveLength(7);
    expect(x).toContain('name="diffuse_reflectance"><string name="filename" value="tex-col.png"/>');
    expect(r.approximations.join()).toMatch(/8 roughness levels/);
    const w3 = decodeImage(file(r, 'tex-mr-lvl3-of8-5_15_25_35_45_55_65_75.png'));
    // indicator "nearest level > 3": G bytes 5 + 10 i for texels 0..7: true from texel 4 on
    expect(Array.from({ length: 8 }, (_, i) => w3.data[i * 4])).toEqual([0, 0, 0, 0, 255, 255, 255, 255]);
  });
  it('more than 8 roughness levels on a dielectric over diffuse has no exact form: throws, or writes the mean alpha with coverage', () => {
    const s = mrScene({ metallicRoughnessTexture: 'many', baseMetalness: 0 });
    expect(() => toMitsuba(s, opts)).toThrow(/16 levels/);
    const x = text(toMitsuba(s, { ...opts, coverage: true }), 'scene.xml');
    expect(x).toContain('<bsdf type="roughplastic">');
    expect(x).not.toContain('blendbsdf');
  });
  it('alpha mask: a mask BSDF over a 0/255 nearest bitmap, and volpath so shadow rays pass through the holes', () => {
    const r = toMitsuba(mrScene({ baseColorTexture: 'col', specularWeight: 0, alphaMode: 'mask' }), opts);
    const x = text(r, 'scene.xml');
    expect(x).toMatch(/<bsdf type="mask"><texture type="bitmap" name="opacity"><string name="filename" value="tex-col-mask-1-0p5\.png"\/><string name="filter_type" value="nearest"\/>/);
    expect(x).toContain('<integrator type="volpath">');
    expect(r.approximations.join()).toMatch(/volpath/);
    expect(toMitsuba(mrScene({ specularWeight: 0 }), opts).files.find((f) => f.path === 'scene.xml')!.data.toString()).toContain('<integrator type="path">');
  });
  it('emissive texture: an area emitter radiance bitmap (raw PFM)', () => {
    const x = text(toMitsuba(mrScene({ baseColor: [0, 0, 0], specularWeight: 0, emissiveTexture: 'col', emissionLuminance: 2 }), opts), 'scene.xml');
    expect(x).toContain('<emitter type="area"><texture type="bitmap" name="radiance"><string name="filename" value="tex-col-emis-m.pfm"/>');
  });
});

describe('glTF web export', () => {
  it('writes TEXCOORD_0 for meshes with uvs and no texture references', () => {
    const r = toGltf(mrScene({ baseColorTexture: 'col' }), opts);
    const glb = r.files[0]!.data as Buffer;
    const json = JSON.parse(glb.subarray(20, 20 + glb.readUInt32LE(12)).toString('utf8'));
    expect(json.meshes[0].primitives[0].attributes.TEXCOORD_0).toBeDefined();
    expect(json.accessors[json.meshes[0].primitives[0].attributes.TEXCOORD_0].type).toBe('VEC2');
    expect(json.textures).toBeUndefined();
  });
});
