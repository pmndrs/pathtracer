/**
 * `openpbr` material -> glTF 2.0 material (core + `KHR_materials_*`): the inverse of `gltfMaterialToOpenPbr` (bench/scenes/gltf-openpbr.ts),
 * for the web competitors (three-gpu-pathtracer, mutracer) that load glTF through three's GLTFLoader (P2.31, docs/design/tier1.md section 3).
 *
 * Only the extensions a material actually uses are written, and a field at its glTF default writes nothing. The mapping is the one of section 2
 * run backwards; what glTF cannot say is listed in `notes` (one line each). Textures are not written (the wave-1 scenes have none; a texture slot
 * is noted and dropped). Pure function, no I/O.
 */
import { openPbrAnisotropyToGltf } from '../scenes/openpbr.ts';
import { resolveOpenPbr } from '../scenes/openpbr.ts';
import type { OpenPbrParams, Rgb } from '../scenes/schema.ts';

export interface GltfMaterialOut {
  pbrMetallicRoughness: { baseColorFactor: number[]; metallicFactor: number; roughnessFactor: number };
  emissiveFactor?: number[];
  alphaMode?: 'MASK' | 'BLEND';
  alphaCutoff?: number;
  doubleSided: boolean;
  extensions?: Record<string, unknown>;
}

/**
 * Nominal `thicknessFactor` (metres) written for a thick transmissive material. glTF thickness is a rasterizer hint (the volume's depth along the view
 * ray); the path tracers integrate through the real closed mesh, so any positive value only marks the material as a solid.
 */
export const NOMINAL_THICKNESS = 1;

const isWhite = (c: readonly number[]): boolean => c.every((v) => v === 1);

export function openPbrToGltfMaterial(p: OpenPbrParams): { material: GltfMaterialOut; notes: string[] } {
  const m = resolveOpenPbr(p);
  const notes: string[] = [];
  const ext: Record<string, unknown> = {};
  const w = m.baseWeight;
  let base: Rgb = [m.baseColor[0] * w, m.baseColor[1] * w, m.baseColor[2] * w];
  if (w !== 1) notes.push('base_weight folded into baseColorFactor');
  let roughness = m.specularRoughness;

  // --- anisotropy -----------------------------------------------------------------------------------------------------
  if (m.specularRoughnessAnisotropy > 0) {
    const conv = openPbrAnisotropyToGltf(roughness, m.specularRoughnessAnisotropy);
    roughness = conv.roughness;
    ext['KHR_materials_anisotropy'] = { anisotropyStrength: conv.strength, ...(m.anisotropyRotation !== 0 ? { anisotropyRotation: m.anisotropyRotation } : {}) };
  }

  // --- ior, specular -------------------------------------------------------------------------------------------------------
  if (m.specularIor !== 1.5) ext['KHR_materials_ior'] = { ior: m.specularIor };
  if (m.specularWeight !== 1 || !isWhite(m.specularColor)) {
    ext['KHR_materials_specular'] = { specularFactor: m.specularWeight, specularColorFactor: [...m.specularColor] };
    if (m.specularWeight < 1) notes.push('specular_weight < 1 -> KHR_materials_specular specularFactor (glTF scales F0 only; OpenPBR scales F at both ends)');
  }

  // --- transmission / volume / diffuse transmission --------------------------------------------------------------------------
  if (m.transmissionWeight > 0) {
    ext['KHR_materials_transmission'] = { transmissionFactor: m.transmissionWeight };
    if (m.geometryThinWalled) {
      // glTF: transmission with no volume (thickness 0) is a thin surface; the tint is the base colour.
      base = [m.transmissionColor[0], m.transmissionColor[1], m.transmissionColor[2]];
    } else {
      const depth = m.transmissionDepth;
      const tinted = !isWhite(m.transmissionColor);
      ext['KHR_materials_volume'] = {
        thicknessFactor: NOMINAL_THICKNESS,
        ...(depth > 0 && tinted ? { attenuationDistance: depth, attenuationColor: [...m.transmissionColor] } : {}),
      };
      if (depth === 0 && tinted) {
        base = [m.transmissionColor[0], m.transmissionColor[1], m.transmissionColor[2]];
        notes.push('transmission_color at depth 0 is a tint of the refraction: written as baseColorFactor');
      }
      notes.push(`thick transmission -> KHR_materials_volume thicknessFactor ${NOMINAL_THICKNESS} (a nominal solid marker; the real depth comes from the closed mesh)`);
    }
  }
  if (m.subsurfaceWeight > 0) {
    ext['KHR_materials_diffuse_transmission'] = { diffuseTransmissionFactor: m.subsurfaceWeight, diffuseTransmissionColorFactor: [...m.subsurfaceColor] };
  }

  // --- coat, sheen, iridescence, dispersion --------------------------------------------------------------------------------------
  if (m.coatWeight > 0) {
    ext['KHR_materials_clearcoat'] = { clearcoatFactor: m.coatWeight, clearcoatRoughnessFactor: m.coatRoughness };
    if (m.coatIor !== 1.5) notes.push(`coat_ior ${m.coatIor} dropped (KHR_materials_clearcoat is fixed at 1.5)`);
  }
  if (m.fuzzWeight > 0) {
    ext['KHR_materials_sheen'] = { sheenColorFactor: m.fuzzColor.map((c) => c * m.fuzzWeight), sheenRoughnessFactor: m.fuzzRoughness };
  }
  if (m.thinFilmWeight > 0) {
    const nm = m.thinFilmThickness * 1000;
    ext['KHR_materials_iridescence'] = { iridescenceFactor: m.thinFilmWeight, iridescenceIor: m.thinFilmIor, iridescenceThicknessMinimum: nm, iridescenceThicknessMaximum: nm };
  }
  if (m.transmissionDispersionScale > 0) {
    ext['KHR_materials_dispersion'] = { dispersion: (20 / m.transmissionDispersionAbbeNumber) * m.transmissionDispersionScale };
  }

  // --- emission ---------------------------------------------------------------------------------------------------------------------
  let emissiveFactor: number[] | undefined;
  if (m.emissionLuminance > 0) {
    const peak = Math.max(...m.emissionColor, 1e-6);
    // emissiveFactor must be in [0, 1]; the excess goes to the strength.
    const scale = peak > 1 ? peak : 1;
    emissiveFactor = m.emissionColor.map((c) => c / scale);
    const strength = m.emissionLuminance * scale;
    if (strength !== 1) ext['KHR_materials_emissive_strength'] = { emissiveStrength: strength };
  }

  for (const f of ['baseColorTexture', 'metallicRoughnessTexture', 'normalTexture', 'emissiveTexture'] as const) if (m[f] !== undefined) notes.push(`texture slot '${f}' not written`);

  const out: GltfMaterialOut = {
    pbrMetallicRoughness: { baseColorFactor: [...base, m.alphaMode === 'blend' || m.alphaMode === 'mask' ? m.geometryOpacity : 1], metallicFactor: m.baseMetalness, roughnessFactor: roughness },
    // The shared exporter writes every material double sided (a closed transmissive mesh must be seen from inside; one-sided culling differs by consumer).
    doubleSided: true,
  };
  if (m.alphaMode === 'blend') out.alphaMode = 'BLEND';
  if (m.alphaMode === 'mask') {
    out.alphaMode = 'MASK';
    out.alphaCutoff = m.alphaCutoff;
  }
  if (emissiveFactor) out.emissiveFactor = emissiveFactor;
  if (Object.keys(ext).length) out.extensions = ext;
  return { material: out, notes };
}
