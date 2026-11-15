/**
 * Tier 1 feature ablations (P2.31, docs/design/tier1.md section 3): a scene with one feature stripped from every `openpbr` material.
 *
 * `<scene>--no-<tag>` is the base scene with `ABLATIONS[tag].strip` applied to each of its `openpbr` materials; `getScene` resolves the name, so
 * the same variant goes through every adapter and exporter unchanged (the sceneHash of the glb sidecar covers the stripped parameters).
 * A renderer supports a feature when its full render differs from its stripped render far above its own noise.
 * Browser-safe (no node imports): the bench pages import scenes through `scenes/index.ts`.
 */
import type { MaterialDesc, OpenPbrParams, SceneDesc } from '../schema.ts';

export interface Ablation {
  /** What the stripped variant removes, one phrase. */
  label: string;
  /** The stripped material, or null when the material does not use the feature (returned as is). */
  strip(p: OpenPbrParams): OpenPbrParams | null;
}

export const ABLATIONS: Readonly<Record<string, Ablation>> = {
  /** Metals become dielectrics (metalness 0). */
  metal: { label: 'baseMetalness forced to 0', strip: (p) => ((p.baseMetalness ?? 0) > 0 ? { ...p, baseMetalness: 0 } : null) },
  /** The dielectric specular lobe is removed (specularWeight 0): a pure diffuse base. */
  spec: { label: 'dielectric specular lobe removed (specularWeight 0)', strip: (p) => ((p.baseMetalness ?? 0) < 1 && (p.specularWeight ?? 1) > 0 ? { ...p, specularWeight: 0 } : null) },
  /** specularWeight below 1 forced to 1 (and specularColor white). */
  specfactor: { label: 'specularWeight < 1 forced to 1', strip: (p) => ((p.specularWeight ?? 1) < 1 ? { ...p, specularWeight: 1, specularColor: [1, 1, 1] } : null) },
  /** Transmission removed (an opaque base remains). */
  transmission: { label: 'transmissionWeight forced to 0', strip: (p) => ((p.transmissionWeight ?? 0) > 0 ? { ...p, transmissionWeight: 0 } : null) },
  /** Volume attenuation removed: the transmission stays, the medium is clear. */
  attenuation: {
    label: 'volume attenuation removed (transmissionDepth 0, transmissionColor white)',
    strip: (p) => ((p.transmissionDepth ?? 0) > 0 ? { ...p, transmissionDepth: 0, transmissionColor: [1, 1, 1] } : null),
  },
  /** Every IOR forced to the default 1.5. */
  ior: { label: 'specularIor forced to 1.5', strip: (p) => ((p.specularIor ?? 1.5) !== 1.5 ? { ...p, specularIor: 1.5 } : null) },
  coat: { label: 'coatWeight forced to 0', strip: (p) => ((p.coatWeight ?? 0) > 0 ? { ...p, coatWeight: 0 } : null) },
  fuzz: { label: 'fuzzWeight forced to 0 (sheen removed)', strip: (p) => ((p.fuzzWeight ?? 0) > 0 ? { ...p, fuzzWeight: 0 } : null) },
  /** The thin film removed (thinFilmWeight 0). */
  thinfilm: { label: 'thinFilmWeight forced to 0 (iridescence removed)', strip: (p) => ((p.thinFilmWeight ?? 0) > 0 ? { ...p, thinFilmWeight: 0 } : null) },
  /** emissive_strength removed: emissionLuminance 1 (the emissiveFactor stays). */
  strength: { label: 'emissive_strength forced to 1', strip: (p) => ((p.emissionLuminance ?? 0) > 0 && p.emissionLuminance !== 1 ? { ...p, emissionLuminance: 1 } : null) },
};

export const ABLATION_SEPARATOR = '--no-';

/** `{ base, tag }` of an ablation scene name, or null for an ordinary scene name. */
export function parseAblationName(name: string): { base: string; tag: string } | null {
  const i = name.lastIndexOf(ABLATION_SEPARATOR);
  if (i <= 0) return null;
  const tag = name.slice(i + ABLATION_SEPARATOR.length);
  return tag in ABLATIONS ? { base: name.slice(0, i), tag } : null;
}

export const ablationSceneName = (base: string, tag: string): string => `${base}${ABLATION_SEPARATOR}${tag}`;

/** The scene with the ablation applied to every `openpbr` material. Throws when no material uses the feature (the variant would equal the base). */
export function ablateScene(scene: SceneDesc, tag: string): SceneDesc {
  const ab = ABLATIONS[tag];
  if (!ab) throw new Error(`unknown ablation '${tag}' (have: ${Object.keys(ABLATIONS).join(', ')})`);
  let changed = 0;
  const materials: Record<string, MaterialDesc> = {};
  for (const [k, m] of Object.entries(scene.materials)) {
    if (m.type !== 'openpbr') {
      materials[k] = m;
      continue;
    }
    const { type: _t, ...params } = m;
    const stripped = ab.strip(params);
    if (stripped) changed++;
    materials[k] = stripped ? { type: 'openpbr', ...stripped } : m;
  }
  if (changed === 0) throw new Error(`ablation '${tag}' changes no material of scene '${scene.name}'`);
  return { ...scene, name: ablationSceneName(scene.name, tag), materials };
}
