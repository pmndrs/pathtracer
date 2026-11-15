import { describe, expect, it } from 'vitest';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { toMitsuba, toPbrt } from './index.ts';
import { cauchyFit, extinctionFromTransmission, LAMBDA_C, LAMBDA_D, LAMBDA_F, reflectanceToEtaK } from './openpbr.ts';
import type { OpenPbrParams, SceneDesc } from '../scenes/schema.ts';
import { openPbrCheckScenes, slabTransmittance, thinSheetTransmittance } from '../references/openpbr-check-scenes.ts';

const assetsDir = join(fileURLToPath(import.meta.url), '..', '..', 'assets');
const opts = { assetsDir, spp: 16 };

const sceneOf = (m: OpenPbrParams, mesh: Record<string, unknown> = {}): SceneDesc => ({
  version: 1,
  name: 'unit',
  purpose: 'unit',
  tier: 1,
  camera: { type: 'perspective', position: [0, 0, 4], target: [0, 0, 0], up: [0, 1, 0], yFovDeg: 30 },
  film: { width: 8, height: 8, filter: 'box' },
  integrator: { maxDepth: 8 },
  materials: { m: { type: 'openpbr', ...m } },
  meshes: [{ name: 'q', geometry: { type: 'box', size: [1, 1, 1] }, material: 'm', ...mesh }],
  environment: { type: 'uniform', radiance: [1, 1, 1] },
});
const pbrt = (m: OpenPbrParams, o: { coverage?: boolean; mesh?: Record<string, unknown> } = {}) => {
  const r = toPbrt(sceneOf(m, o.mesh), { ...opts, coverage: o.coverage });
  return { text: r.files.find((f) => f.path === 'scene.pbrt')!.data.toString(), notes: r.approximations };
};
const mitsuba = (m: OpenPbrParams, o: { coverage?: boolean; mesh?: Record<string, unknown> } = {}) => {
  const r = toMitsuba(sceneOf(m, o.mesh), { ...opts, coverage: o.coverage });
  return { text: r.files.find((f) => f.path === 'scene.xml')!.data.toString(), notes: r.approximations };
};

describe('helpers', () => {
  it('reflectanceToEtaK: eta 1 and k whose normal-incidence reflectance is r', () => {
    const { eta, k } = reflectanceToEtaK([0.9, 0.5, 0.04]);
    expect(eta).toEqual([1, 1, 1]);
    [0.9, 0.5, 0.04].forEach((r, i) => expect(k[i]! ** 2 / (4 + k[i]! ** 2)).toBeCloseTo(r, 9));
  });
  it('extinctionFromTransmission is Beer-Lambert in metres, white = 0', () => {
    const s = extinctionFromTransmission([0.5, 1, 0.25], 0.5);
    expect(Math.exp(-s[0] * 0.5)).toBeCloseTo(0.5, 12);
    expect(s[1]).toBe(0);
    expect(Math.exp(-s[2] * 0.5)).toBeCloseTo(0.25, 12);
  });
  it('cauchyFit hits n_d at the d line and the Abbe number', () => {
    const fit = cauchyFit(1.5, 20);
    expect(fit.eta(LAMBDA_D)).toBeCloseTo(1.5, 12);
    expect((1.5 - 1) / (fit.eta(LAMBDA_F) - fit.eta(LAMBDA_C))).toBeCloseTo(20, 9);
    expect(fit.eta(400)).toBeGreaterThan(fit.eta(700));
    expect(cauchyFit(1.5, Infinity).b).toBe(0);
  });
  it('closed forms used by the synthetic checks', () => {
    expect(thinSheetTransmittance(1.5)).toBeCloseTo(2 * 1.5 / (1.5 ** 2 + 1), 12);
    expect(slabTransmittance(1.5, 1)).toBeCloseTo(thinSheetTransmittance(1.5), 12); // lossless slab = thin sheet
    expect(slabTransmittance(1.5, 0)).toBe(0);
  });
});

describe('pbrt openpbr', () => {
  it('pure metal -> conductor with reflectance and alpha = roughness^2 (u/v when anisotropic)', () => {
    const { text } = pbrt({ baseColor: [0.9, 0.8, 0.7], baseMetalness: 1, specularRoughness: 0.5 });
    expect(text).toMatch(/MakeNamedMaterial "m\.conductor0" "string type" "conductor" "rgb reflectance" \[ 0\.9 0\.8 0\.7 \] "float roughness" 0\.25 "bool remaproughness" false/);
    expect(text).toContain('NamedMaterial "m.conductor0"');
    const a = pbrt({ baseMetalness: 1, specularRoughness: 0.5, specularRoughnessAnisotropy: 0.5 });
    expect(a.text).toMatch(/"float uroughness" 0\.31\d+ "float vroughness" 0\.15\d+/);
    expect(a.notes.join('\n')).toMatch(/dp\/du/);
  });
  it('transmission 1 -> dielectric (thindielectric when thin-walled)', () => {
    expect(pbrt({ transmissionWeight: 1, specularRoughness: 0, specularIor: 1.33 }).text).toMatch(/"string type" "dielectric" "float eta" 1\.33 "float roughness" 0\.0 "bool remaproughness" false/);
    const t = pbrt({ transmissionWeight: 1, specularRoughness: 0, geometryThinWalled: true }).text;
    expect(t).toMatch(/"string type" "thindielectric" "float eta" 1\.5/);
    expect(() => pbrt({ transmissionWeight: 1, specularRoughness: 0.3, geometryThinWalled: true })).toThrow(/class C/);
  });
  it('fractional metalness -> mix of the two ends with amount = metalness', () => {
    const { text } = pbrt({ baseMetalness: 0.25, transmissionWeight: 1, specularRoughness: 0.1 });
    expect(text).toMatch(/"string type" "mix" "string materials" \[ "m\.dielectric0" "m\.conductor1" \] "float amount" 0\.25/);
  });
  it('dielectric over diffuse -> coateddiffuse with raised maxdepth, nsamples and a transparent layer; both recorded', () => {
    const { text, notes } = pbrt({ baseColor: [0.7, 0.3, 0.2], specularRoughness: 0.3, specularIor: 1.5 });
    expect(text).toMatch(/"string type" "coateddiffuse" "rgb reflectance" \[ 0\.7 0\.3 0\.2 \] "float roughness" 0\.09\d* "float eta" 1\.5 "float thickness" 0\.000001 "integer maxdepth" 64 "integer nsamples" 1 "bool remaproughness" false/);
    expect(notes.join('\n')).toMatch(/maxdepth 64, nsamples 1/);
    const hi = pbrt({}, {}); // default scene keeps the default
    expect(hi.text).toContain('"integer maxdepth" 64');
    const r = toPbrt(sceneOf({}), { ...opts, layered: { maxDepth: 100, nSamples: 4 } });
    expect(r.files.find((f) => f.path === 'scene.pbrt')!.data.toString()).toContain('"integer maxdepth" 100 "integer nsamples" 4');
  });
  it('specular off -> plain diffuse; clearcoat on a metal -> coatedconductor; on a dielectric -> coateddiffuse with the coat IOR', () => {
    expect(pbrt({ specularWeight: 0, baseColor: [0.5, 0.5, 0.5] }).text).toMatch(/"string type" "diffuse" "rgb reflectance" \[ 0\.5 0\.5 0\.5 \]/);
    const cc = pbrt({ baseMetalness: 1, baseColor: [0.9, 0.9, 0.9], specularRoughness: 0.4, coatWeight: 1, coatRoughness: 0.1, coatIor: 1.5 }).text;
    expect(cc).toMatch(/"string type" "coatedconductor" "rgb reflectance" \[ 0\.9 0\.9 0\.9 \] "float conductor\.roughness" 0\.16\d* "float interface\.roughness" 0\.01\d* "float interface\.eta" 1\.5/);
    const cd = pbrt({ coatWeight: 1, coatIor: 1.6, coatRoughness: 0 }).text;
    expect(cd).toMatch(/"string type" "coateddiffuse".* "float eta" 1\.6 /);
  });
  it('thin-walled subsurface -> diffusetransmission (reflectance (1-w) base, transmittance w colour); specular drop recorded', () => {
    const { text, notes } = pbrt({ geometryThinWalled: true, subsurfaceWeight: 0.4, baseColor: [0.5, 0.5, 0.5], subsurfaceColor: [1, 0.5, 0.25], specularWeight: 0 });
    expect(text).toMatch(/"string type" "diffusetransmission" "rgb reflectance" \[ 0\.3 0\.3 0\.3 \] "rgb transmittance" \[ 0\.4 0\.2 0\.1 \]/);
    expect(notes.join('')).not.toMatch(/specular is dropped/);
    expect(pbrt({ geometryThinWalled: true, subsurfaceWeight: 0.4 }).notes.join('')).toMatch(/specular is dropped/);
  });
  it('absorbing glass -> named homogeneous medium, MediumInterface and volpath; plain scenes keep `path`', () => {
    const { text, notes } = pbrt({ transmissionWeight: 1, specularRoughness: 0, transmissionColor: [0.5, 0.25, 1], transmissionDepth: 0.5 });
    expect(text).toMatch(/^Integrator "volpath" "integer maxdepth" 8 "bool rr" false$/m);
    expect(text).toMatch(/MakeNamedMedium "med-m" "string type" "homogeneous" "rgb sigma_a" \[ 1\.386\d+ 2\.772\d+ 0\.0 \] "rgb sigma_s" \[ 0\.0 0\.0 0\.0 \] "float scale" 1\.0/);
    expect(text).toContain('MediumInterface "med-m" ""');
    expect(notes.join('\n')).toMatch(/volpath/);
    expect(pbrt({ transmissionWeight: 1 }).text).toMatch(/^Integrator "path"/m);
  });
  it('dispersion -> spectrum eta from a Cauchy fit', () => {
    const { text, notes } = pbrt({ transmissionWeight: 1, specularRoughness: 0, transmissionDispersionScale: 1 });
    const m = /"spectrum eta" \[ ([^\]]+) \]/.exec(text)!;
    const pairs = m[1]!.trim().split(/\s+/).map(Number);
    const eta = (nm: number): number => pairs[pairs.indexOf(nm) + 1]!;
    expect(eta(360)).toBeGreaterThan(eta(830));
    expect(eta(590)).toBeCloseTo(cauchyFit(1.5, 20).eta(590), 5);
    expect(notes.join('')).toMatch(/Cauchy/);
  });
  it('emission -> AreaLightSource L = luminance x colour, two-sided only on request; both mesh and material emission throw', () => {
    const t = pbrt({ emissionLuminance: 3, emissionColor: [1, 0.5, 0.25] }).text;
    expect(t).toContain('AreaLightSource "diffuse" "rgb L" [ 3.0 1.5 0.75 ] "bool twosided" false');
    expect(pbrt({ emissionLuminance: 1, emissionTwoSided: true }).text).toContain('"bool twosided" true');
    expect(() => pbrt({ emissionLuminance: 1 }, { mesh: { emission: [1, 1, 1] } })).toThrow(/both/);
  });
  it('class C throws with a clear message; coverage writes the closest approximation, flagged', () => {
    expect(() => pbrt({ fuzzWeight: 1 })).toThrow(/class C.*fuzz.*coverage/);
    const r = pbrt({ fuzzWeight: 1, thinFilmWeight: 1 }, { coverage: true });
    expect(r.notes.filter((n) => n.startsWith('coverage:')).length).toBeGreaterThanOrEqual(2);
    expect(() => pbrt({ baseColorTexture: 'x' })).toThrow(/texture slot/);
    const spec = pbrt({ specularWeight: 0.5 }, { coverage: true });
    expect(spec.notes.join('\n')).toMatch(/coverage:.*scale F0/);
  });
});

describe('mitsuba openpbr', () => {
  it('pure metal -> roughconductor eta 1 / k from reflectance, alpha_u/alpha_v when anisotropic', () => {
    const { text } = mitsuba({ baseColor: [0.9, 0.9, 0.9], baseMetalness: 1, specularRoughness: 0.5 });
    expect(text).toContain('<bsdf type="roughconductor"><string name="distribution" value="ggx"/><float name="alpha" value="0.25"/><spectrum name="eta" value="1"/><spectrum name="k"');
    const a = mitsuba({ baseMetalness: 1, specularRoughness: 0.5, specularRoughnessAnisotropy: 0.5 }).text;
    expect(a).toMatch(/alpha_u" value="0\.31\d+"\/><float name="alpha_v" value="0\.15\d+"/);
  });
  it('dielectric / roughdielectric / thindielectric', () => {
    expect(mitsuba({ transmissionWeight: 1, specularRoughness: 0 }).text).toContain('<bsdf type="dielectric"><float name="int_ior" value="1.5"/>');
    expect(mitsuba({ transmissionWeight: 1, specularRoughness: 0.3 }).text).toContain('<bsdf type="roughdielectric">');
    expect(mitsuba({ transmissionWeight: 1, specularRoughness: 0, geometryThinWalled: true }).text).toContain('<bsdf type="thindielectric">');
    expect(() => mitsuba({ transmissionWeight: 1, specularRoughness: 0.2, geometryThinWalled: true })).toThrow(/class C/);
  });
  it('fractional metalness -> blendbsdf with weight = metalness (children: dielectric end, metal end)', () => {
    const { text } = mitsuba({ baseMetalness: 0.25, transmissionWeight: 1, specularRoughness: 0.1 });
    expect(text).toMatch(/<bsdf type="blendbsdf"><float name="weight" value="0\.25"\/><bsdf type="roughdielectric">.*<\/bsdf><bsdf type="twosided"><bsdf type="roughconductor">/);
  });
  it('dielectric over diffuse -> roughplastic nonlinear, with an approximation note', () => {
    const { text, notes } = mitsuba({ baseColor: [0.7, 0.3, 0.2], specularRoughness: 0.3 });
    expect(text).toContain('<bsdf type="roughplastic">');
    expect(text).toContain('<boolean name="nonlinear" value="true"/>');
    expect(notes.join('')).toMatch(/roughplastic with nonlinear true/);
  });
  it('clearcoat on a metal needs coverage (principled), on a dielectric is roughplastic with the coat IOR', () => {
    const coated = { baseMetalness: 1, coatWeight: 1, specularRoughness: 0.4 };
    expect(() => mitsuba(coated)).toThrow(/coverage/);
    const r = mitsuba(coated, { coverage: true });
    expect(r.text).toContain('<bsdf type="principled">');
    expect(r.notes.join('\n')).toMatch(/coverage:.*clear coat on a metal/);
    expect(mitsuba({ coatWeight: 1, coatIor: 1.6, coatRoughness: 0.1 }).text).toMatch(/roughplastic.*int_ior" value="1\.6"/);
  });
  it('dispersion and diffuse transmission have no Mitsuba equivalent: throw, or a flagged constant-IOR / reflection-only coverage', () => {
    expect(() => mitsuba({ transmissionWeight: 1, specularRoughness: 0, transmissionDispersionScale: 1 })).toThrow(/dispersion/);
    const d = mitsuba({ transmissionWeight: 1, specularRoughness: 0, transmissionDispersionScale: 1 }, { coverage: true });
    expect(d.notes.join('\n')).toMatch(/coverage:.*dispersion/);
    expect(() => mitsuba({ geometryThinWalled: true, subsurfaceWeight: 0.5 })).toThrow(/diffuse transmission/);
  });
  it('absorbing glass -> homogeneous interior medium (scale x normalised rgb) and volpath', () => {
    const { text } = mitsuba({ transmissionWeight: 1, specularRoughness: 0, transmissionColor: [0.5, 0.25, 1], transmissionDepth: 0.5 });
    expect(text).toContain('<integrator type="volpath">');
    expect(text).toMatch(/<medium name="interior" type="homogeneous"><rgb name="sigma_t" value="0\.5, 1, 0"\/><rgb name="albedo" value="0, 0, 0"\/><float name="scale" value="2\.772\d+"\/><\/medium>/);
    expect(mitsuba({ transmissionWeight: 1 }).text).toContain('<integrator type="path">');
  });
  it('emission -> area emitter; class C throws', () => {
    expect(mitsuba({ emissionLuminance: 3, emissionColor: [1, 0.5, 0.25] }).text).toContain('<emitter type="area"><rgb name="radiance" value="3, 1.5, 0.75"/></emitter>');
    expect(() => mitsuba({ thinFilmWeight: 1 })).toThrow(/class C/);
  });
});

describe('synthetic check scenes', () => {
  it('every scene exports to both renderers (coverage for the principled one)', () => {
    for (const s of openPbrCheckScenes) {
      expect(() => toPbrt(s, { ...opts, coverage: true }), s.name).not.toThrow();
      expect(() => toMitsuba(s, { ...opts, coverage: true }), s.name).not.toThrow();
    }
  });
});
