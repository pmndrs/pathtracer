/**
 * glTF 2.0 material (core + `KHR_materials_*`) -> OpenPBR parameters (DIVERGES V53), and OpenPBR parameters -> the scene copy's material record.
 *
 * OpenPBR 1.1.2 has no glTF mapping section; this one is ours. Input is a material's JSON (and its `extensions` object, taken from
 * the material by default); output is a sparse OpenPBR parameter set (only fields the glTF material sets or that differ from OpenPBR's
 * defaults) and notes for everything that was dropped, approximated or needs a loader. Nothing is silently ignored: unknown extensions
 * and every texture slot are listed.
 *
 * Units: glTF is in metres and so is the scene, so `attenuationDistance` becomes `transmissionDepth` 1:1 (OpenPBR's default length unit is
 * the centimetre). Iridescence nm -> thin-film micrometres. `emissiveFactor * emissiveStrength` -> `emissionColor * emissionLuminance`
 * (same radiance; glTF nits and W/sr/m^2 are taken 1:1). Pure: no I/O.
 */
import { ALPHA_MODE, type RtMaterialInput } from '../scene/material.ts';
import { gltfAnisotropyToOpenPbr, nmToMicrometres, resolveOpenPbr } from './openpbr.ts';
import type { OpenPbrParams, OpenPbrRgb as Rgb } from './openpbr.ts';

// ---------------------------------------------------------------------------------------------------------------
// glTF input shapes (only the fields we read)
// ---------------------------------------------------------------------------------------------------------------

export interface GltfTextureInfo { index: number; texCoord?: number; scale?: number; strength?: number; extensions?: Record<string, unknown> }

export interface GltfMaterialExtensions {
  KHR_materials_ior?: { ior?: number };
  KHR_materials_specular?: { specularFactor?: number; specularColorFactor?: readonly number[]; specularTexture?: GltfTextureInfo; specularColorTexture?: GltfTextureInfo };
  KHR_materials_transmission?: { transmissionFactor?: number; transmissionTexture?: GltfTextureInfo };
  KHR_materials_volume?: { thicknessFactor?: number; thicknessTexture?: GltfTextureInfo; attenuationDistance?: number; attenuationColor?: readonly number[] };
  KHR_materials_clearcoat?: { clearcoatFactor?: number; clearcoatRoughnessFactor?: number; clearcoatTexture?: GltfTextureInfo; clearcoatRoughnessTexture?: GltfTextureInfo; clearcoatNormalTexture?: GltfTextureInfo };
  KHR_materials_sheen?: { sheenColorFactor?: readonly number[]; sheenRoughnessFactor?: number; sheenColorTexture?: GltfTextureInfo; sheenRoughnessTexture?: GltfTextureInfo };
  KHR_materials_iridescence?: { iridescenceFactor?: number; iridescenceIor?: number; iridescenceThicknessMinimum?: number; iridescenceThicknessMaximum?: number; iridescenceTexture?: GltfTextureInfo; iridescenceThicknessTexture?: GltfTextureInfo };
  KHR_materials_anisotropy?: { anisotropyStrength?: number; anisotropyRotation?: number; anisotropyTexture?: GltfTextureInfo };
  KHR_materials_emissive_strength?: { emissiveStrength?: number };
  KHR_materials_dispersion?: { dispersion?: number };
  KHR_materials_diffuse_transmission?: { diffuseTransmissionFactor?: number; diffuseTransmissionColorFactor?: readonly number[]; diffuseTransmissionTexture?: GltfTextureInfo; diffuseTransmissionColorTexture?: GltfTextureInfo };
  KHR_materials_scatter?: { scatterStrengthFactor?: number; scatterStrengthTexture?: GltfTextureInfo; multiscatterColorFactor?: readonly number[]; multiscatterColorTexture?: GltfTextureInfo; scatterAnisotropy?: number };
  KHR_materials_unlit?: Record<string, never>;
  [other: string]: unknown;
}

export interface GltfMaterialJson {
  name?: string;
  pbrMetallicRoughness?: { baseColorFactor?: readonly number[]; metallicFactor?: number; roughnessFactor?: number; baseColorTexture?: GltfTextureInfo; metallicRoughnessTexture?: GltfTextureInfo };
  normalTexture?: GltfTextureInfo;
  occlusionTexture?: GltfTextureInfo;
  emissiveTexture?: GltfTextureInfo;
  emissiveFactor?: readonly number[];
  alphaMode?: 'OPAQUE' | 'MASK' | 'BLEND';
  alphaCutoff?: number;
  doubleSided?: boolean;
  extensions?: GltfMaterialExtensions;
}

/** Extensions this function maps. Anything else on the material goes to `unsupported`. */
export const MAPPED_EXTENSIONS = [
  'KHR_materials_ior', 'KHR_materials_specular', 'KHR_materials_transmission', 'KHR_materials_volume', 'KHR_materials_clearcoat',
  'KHR_materials_sheen', 'KHR_materials_iridescence', 'KHR_materials_anisotropy', 'KHR_materials_emissive_strength',
  'KHR_materials_dispersion', 'KHR_materials_diffuse_transmission', 'KHR_materials_scatter', 'KHR_materials_unlit',
] as const;

/** glTF spec defaults, kept in one place so tests can pin them. */
export const GLTF_DEFAULTS = {
  baseColorFactor: [1, 1, 1, 1],
  metallicFactor: 1,
  roughnessFactor: 1,
  emissiveFactor: [0, 0, 0],
  alphaCutoff: 0.5,
  ior: 1.5,
  specularFactor: 1,
  specularColorFactor: [1, 1, 1],
  transmissionFactor: 0,
  thicknessFactor: 0,
  attenuationColor: [1, 1, 1],
  clearcoatFactor: 0,
  clearcoatRoughnessFactor: 0,
  sheenColorFactor: [0, 0, 0],
  sheenRoughnessFactor: 0,
  iridescenceFactor: 0,
  iridescenceIor: 1.3,
  iridescenceThicknessMinimum: 100,
  iridescenceThicknessMaximum: 400,
  anisotropyStrength: 0,
  anisotropyRotation: 0,
  emissiveStrength: 1,
  dispersion: 0,
  diffuseTransmissionFactor: 0,
  diffuseTransmissionColorFactor: [1, 1, 1],
} as const;

/** The clearcoat layer's IOR is fixed at 1.5 by KHR_materials_clearcoat. */
export const GLTF_CLEARCOAT_IOR = 1.5;

export interface GltfOpenPbrResult {
  /** The OpenPBR parameters, sparse: set fields only (`resolveOpenPbr` fills the rest). No texture keys: textures are the loader's job. */
  params: OpenPbrParams;
  /** Things dropped, approximated or lossy, one line each. */
  notes: string[];
  /** glTF texture slots the material has (not mapped here: textures are the loader's job). */
  textures: string[];
  /** Extensions on the material this function does not map. */
  unsupported: string[];
  /** A base-colour tint on a transmissive material was dropped because `transmissionColor` holds the volume's attenuation colour. */
  lostTransmissionTint: boolean;
}

const rgb = (v: readonly number[] | undefined, fallback: readonly number[]): Rgb => {
  const a = v ?? fallback;
  return [a[0] ?? 0, a[1] ?? 0, a[2] ?? 0];
};
const same = (a: readonly number[], b: readonly number[]): boolean => a.length === b.length && a.every((v, i) => v === b[i]);

const TEXTURE_FIELDS: [keyof GltfMaterialJson, string][] = [['normalTexture', 'normalTexture'], ['occlusionTexture', 'occlusionTexture'], ['emissiveTexture', 'emissiveTexture']];

/**
 * Maps one glTF material. `extensions` defaults to `material.extensions`; pass another object to override (tests, sidecar overrides).
 */
export function gltfMaterialToOpenPbr(material: GltfMaterialJson, extensions: GltfMaterialExtensions = material.extensions ?? {}): GltfOpenPbrResult {
  const D = GLTF_DEFAULTS;
  const notes: string[] = [];
  const textures: string[] = [];
  const pmr = material.pbrMetallicRoughness ?? {};
  const out: OpenPbrParams = {};

  for (const [field, name] of TEXTURE_FIELDS) if (material[field]) textures.push(name);
  if (pmr.baseColorTexture) textures.push('baseColorTexture');
  if (pmr.metallicRoughnessTexture) textures.push('metallicRoughnessTexture');

  // --- core metallic-roughness (glTF defaults differ from OpenPBR's: white, metallic 1, roughness 1) -------------
  const bc = pmr.baseColorFactor ?? D.baseColorFactor;
  const alpha = bc[3] ?? 1;
  out.baseColor = rgb(bc, D.baseColorFactor);
  out.baseMetalness = pmr.metallicFactor ?? D.metallicFactor;
  out.specularRoughness = pmr.roughnessFactor ?? D.roughnessFactor;

  const mode = material.alphaMode ?? 'OPAQUE';
  if (mode === 'BLEND') {
    out.alphaMode = 'blend';
    out.geometryOpacity = alpha;
  } else if (mode === 'MASK') {
    out.alphaMode = 'mask';
    out.geometryOpacity = alpha;
    out.alphaCutoff = material.alphaCutoff ?? D.alphaCutoff;
  } else if (alpha !== 1) {
    notes.push('baseColorFactor alpha ignored (alphaMode OPAQUE)');
  }
  if (material.doubleSided) out.doubleSided = true;

  // --- emission ----------------------------------------------------------------------------------------------
  const ef = rgb(material.emissiveFactor, D.emissiveFactor);
  const strengthExt = extensions.KHR_materials_emissive_strength;
  const strength = strengthExt ? (strengthExt.emissiveStrength ?? D.emissiveStrength) : 1;
  if (ef.some((v) => v > 0)) {
    out.emissionColor = ef;
    out.emissionLuminance = strength;
  }

  // --- IOR, specular ---------------------------------------------------------------------------------------------
  const iorExt = extensions.KHR_materials_ior;
  if (iorExt) {
    let ior = iorExt.ior ?? D.ior;
    if (ior < 1) {
      notes.push(`ior ${ior} < 1 clamped to 1`);
      ior = 1;
    }
    out.specularIor = ior;
  } else {
    out.specularIor = D.ior;
  }
  const spec = extensions.KHR_materials_specular;
  if (spec) {
    out.specularWeight = spec.specularFactor ?? D.specularFactor;
    out.specularColor = rgb(spec.specularColorFactor, D.specularColorFactor);
    if (spec.specularTexture) textures.push('specularTexture');
    if (spec.specularColorTexture) textures.push('specularColorTexture');
    if (out.specularWeight < 1 || !same(out.specularColor, [1, 1, 1])) notes.push('specularFactor/specularColor scale F0 only in glTF but F at both ends in OpenPBR (V54)');
  }

  // --- transmission, volume, diffuse transmission -------------------------------------------------------------------
  const tr = extensions.KHR_materials_transmission;
  const vol = extensions.KHR_materials_volume;
  const dt = extensions.KHR_materials_diffuse_transmission;
  const sc = extensions.KHR_materials_scatter;
  if (sc?.scatterStrengthTexture) textures.push('scatterStrengthTexture');
  if (sc?.multiscatterColorTexture) textures.push('multiscatterColorTexture');
  const transmission = tr ? (tr.transmissionFactor ?? D.transmissionFactor) : 0;
  if (tr?.transmissionTexture) textures.push('transmissionTexture');
  let lostTint = false;
  if (tr) {
    out.transmissionWeight = transmission;
    if (transmission > 0) {
      const thickness = vol ? (vol.thicknessFactor ?? D.thicknessFactor) : 0;
      if (vol?.thicknessTexture) textures.push('thicknessTexture');
      const distance = vol?.attenuationDistance;
      const finite = distance !== undefined && Number.isFinite(distance) && distance > 0 && distance < 1e30;
      const tint = rgb(pmr.baseColorFactor, D.baseColorFactor);
      if (thickness > 0) {
        // A closed, solid object: refraction through a medium.
        if (finite) {
          out.transmissionColor = rgb(vol?.attenuationColor, D.attenuationColor);
          out.transmissionDepth = distance;
          if (sc) {
            const strength = sc.scatterStrengthFactor ?? 0;
            // Volumetric mode of KHR_materials_scatter (draft, glTF PR 2579): sigma_s / sigma_a follow from the multi-scatter albedo; strength 0 = plain absorption.
            if (strength > 0) out.volumeScatter = { strength, multiscatterColor: rgb(sc.multiscatterColorFactor, [1, 1, 1]), anisotropy: sc.scatterAnisotropy ?? 0 };
          }
          if (!same(tint, [1, 1, 1])) {
            lostTint = true;
            notes.push('baseColor tint dropped: transmissionColor holds the volume attenuationColor');
          }
        } else {
          out.transmissionColor = tint;
        }
      } else {
        // glTF: no volume, or thickness 0, means a thin surface (no absorption, no refraction offset).
        out.geometryThinWalled = true;
        out.transmissionColor = tint;
        if (sc && (sc.scatterStrengthFactor ?? 0) > 0) notes.push('KHR_materials_scatter thin-walled mode (diffuse scatter BSDF) is not mapped');
        if (finite && !same(rgb(vol?.attenuationColor, D.attenuationColor), [1, 1, 1])) notes.push('attenuation ignored: thickness 0 means a thin-walled surface');
      }
      if (!same(out.transmissionColor ?? [1, 1, 1], [1, 1, 1]) && out.transmissionDepth === undefined) notes.push('transmissionColor is a non-physical tint (depth 0): baseColor on the transmission lobe');
    } else if (vol) {
      notes.push('KHR_materials_volume ignored (transmission 0)');
    }
  } else if (vol) {
    notes.push('KHR_materials_volume without transmission: no refraction, volume ignored');
  }
  if (dt) {
    const f = dt.diffuseTransmissionFactor ?? D.diffuseTransmissionFactor;
    out.subsurfaceWeight = f;
    out.subsurfaceColor = rgb(dt.diffuseTransmissionColorFactor, D.diffuseTransmissionColorFactor);
    if (dt.diffuseTransmissionTexture) textures.push('diffuseTransmissionTexture');
    if (dt.diffuseTransmissionColorTexture) textures.push('diffuseTransmissionColorTexture');
    if (f > 0) {
      if (transmission > 0 && !out.geometryThinWalled) notes.push('diffuse transmission on a thick transmissive material: OpenPBR subsurface models it thin-walled only');
      else out.geometryThinWalled = true;
    }
  }

  // --- coat, sheen, iridescence, anisotropy, dispersion -------------------------------------------------------------
  const cc = extensions.KHR_materials_clearcoat;
  if (cc) {
    out.coatWeight = cc.clearcoatFactor ?? D.clearcoatFactor;
    out.coatRoughness = cc.clearcoatRoughnessFactor ?? D.clearcoatRoughnessFactor;
    out.coatIor = GLTF_CLEARCOAT_IOR;
    for (const [k, n] of [['clearcoatTexture', 'clearcoatTexture'], ['clearcoatRoughnessTexture', 'clearcoatRoughnessTexture'], ['clearcoatNormalTexture', 'clearcoatNormalTexture']] as const) if (cc[k]) textures.push(n);
  }
  const sh = extensions.KHR_materials_sheen;
  if (sh) {
    const col = rgb(sh.sheenColorFactor, D.sheenColorFactor);
    out.fuzzWeight = col.some((v) => v > 0) ? 1 : 0;
    out.fuzzColor = col;
    out.fuzzRoughness = sh.sheenRoughnessFactor ?? D.sheenRoughnessFactor;
    if (sh.sheenColorTexture) textures.push('sheenColorTexture');
    if (sh.sheenRoughnessTexture) textures.push('sheenRoughnessTexture');
    notes.push('sheen: glTF Charlie lobe vs OpenPBR fuzz, close but not identical');
  }
  const ir = extensions.KHR_materials_iridescence;
  if (ir) {
    out.thinFilmWeight = ir.iridescenceFactor ?? D.iridescenceFactor;
    out.thinFilmIor = ir.iridescenceIor ?? D.iridescenceIor;
    const lo = ir.iridescenceThicknessMinimum ?? D.iridescenceThicknessMinimum;
    const hi = ir.iridescenceThicknessMaximum ?? D.iridescenceThicknessMaximum;
    // Without a thickness texture the thickness is the maximum (KHR_materials_iridescence).
    out.thinFilmThickness = nmToMicrometres(hi);
    if (ir.iridescenceTexture) textures.push('iridescenceTexture');
    if (ir.iridescenceThicknessTexture) {
      textures.push('iridescenceThicknessTexture');
      notes.push(`thin-film thickness is textured, mixing ${lo}..${hi} nm; the maximum is used`);
    }
  }
  const an = extensions.KHR_materials_anisotropy;
  if (an) {
    const s = an.anisotropyStrength ?? D.anisotropyStrength;
    const rot = an.anisotropyRotation ?? D.anisotropyRotation;
    if (an.anisotropyTexture) {
      textures.push('anisotropyTexture');
      notes.push('anisotropyTexture (strength and direction) not mapped');
    }
    if (s > 0) {
      const conv = gltfAnisotropyToOpenPbr(out.specularRoughness, s);
      out.specularRoughness = conv.roughness;
      out.specularRoughnessAnisotropy = conv.anisotropy;
      if (rot !== 0) out.anisotropyRotation = rot;
    } else if (rot !== 0) {
      out.anisotropyRotation = rot;
    }
  }
  const dp = extensions.KHR_materials_dispersion;
  if (dp) {
    const d = dp.dispersion ?? D.dispersion;
    if (d > 0) {
      out.transmissionDispersionScale = 1;
      out.transmissionDispersionAbbeNumber = 20 / d;
      if (transmission === 0) notes.push('dispersion has no effect without transmission');
    }
  }

  // --- unlit -----------------------------------------------------------------------------------------------------
  if (extensions.KHR_materials_unlit) {
    // The output is the base colour, no shading: a non-scattering emitter of that colour.
    out.emissionColor = rgb(pmr.baseColorFactor, D.baseColorFactor);
    out.emissionLuminance = 1;
    out.baseWeight = 0;
    out.specularWeight = 0;
    out.baseMetalness = 0;
    if (pmr.baseColorTexture) notes.push('unlit with a base colour texture: the texture is the emission');
    notes.push('KHR_materials_unlit: black non-scattering emitter of baseColor (display-referred value taken as radiance)');
  }

  const unsupported = Object.keys(extensions).filter((k) => !(MAPPED_EXTENSIONS as readonly string[]).includes(k));
  if (unsupported.length) notes.push(`unsupported extensions: ${unsupported.join(', ')}`);

  return { params: out, notes, textures, unsupported, lostTransmissionTint: lostTint };
}

// ---------------------------------------------------------------------------------------------------------------
// OpenPBR -> scene copy
// ---------------------------------------------------------------------------------------------------------------

/** An angle in radians (any sign) as the scene copy's unorm rotation field: turns in [0, 1). */
export function rotationTurns(radians: number): number {
  const t = radians / (2 * Math.PI);
  return t - Math.floor(t);
}

/**
 * The scene-copy material of an OpenPBR parameter set (`buildSceneCopy` input). `texture(key)` maps the parameter set's texture keys to
 * indices into `SceneCopyInput.textures`. The fields the scene copy has map one to one; subsurface, diffuse roughness and the metal's F82
 * tint have no record slot yet. Dispersion is KHR_materials_dispersion's 20 / Abbe number, scaled (only the spectral mode reads it).
 */
export function openPbrToRtMaterial(params: OpenPbrParams, texture: (key: string) => number = (key) => {
  throw new RangeError(`texture '${key}' is not defined`);
}): RtMaterialInput {
  const m = resolveOpenPbr(params);
  const tex = (name: string | undefined) => (name ? texture(name) : undefined);
  const baseColorTexture = tex(m.baseColorTexture), roughnessMetalTexture = tex(m.metallicRoughnessTexture), normalTexture = tex(m.normalTexture), emissionTexture = tex(m.emissiveTexture);
  return {
    baseColor: [...m.baseColor], baseWeight: m.baseWeight, metalness: m.baseMetalness,
    specularWeight: m.specularWeight, specularColor: [...m.specularColor], specularRoughness: m.specularRoughness, ior: m.specularIor,
    dispersion: m.transmissionDispersionScale > 0 && m.transmissionDispersionAbbeNumber > 0 ? (m.transmissionDispersionScale * 20) / m.transmissionDispersionAbbeNumber : 0,
    specularAnisotropy: m.specularRoughnessAnisotropy, anisotropyRotation: rotationTurns(m.anisotropyRotation),
    transmissionWeight: m.transmissionWeight, transmissionColor: [...m.transmissionColor], transmissionDepth: m.transmissionDepth,
    coatWeight: m.coatWeight, coatColor: [...m.coatColor], coatRoughness: m.coatRoughness, coatIor: m.coatIor,
    sheenWeight: m.fuzzWeight, sheenColor: [...m.fuzzColor], sheenRoughness: m.fuzzRoughness,
    thinFilmWeight: m.thinFilmWeight, thinFilmThickness: m.thinFilmThickness, thinFilmIor: m.thinFilmIor,
    emissionColor: [m.emissionColor[0] * m.emissionLuminance, m.emissionColor[1] * m.emissionLuminance, m.emissionColor[2] * m.emissionLuminance],
    opacity: m.geometryOpacity, alphaCutoff: m.alphaCutoff, alphaMode: ALPHA_MODE[m.alphaMode],
    thinWalled: m.geometryThinWalled, doubleSided: m.doubleSided, emissionTwoSided: m.emissionTwoSided,
    ...(baseColorTexture !== undefined ? { baseColorTexture } : {}), ...(roughnessMetalTexture !== undefined ? { roughnessMetalTexture } : {}),
    ...(normalTexture !== undefined ? { normalTexture } : {}), ...(emissionTexture !== undefined ? { emissionTexture } : {}),
  };
}
