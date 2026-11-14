/**
 * Tiny synthetic `openpbr` scenes (P2.20) that both pbrt-v4 and Mitsuba 3 can render: they prove the exported files are valid and that the
 * two renderers (and, where there is one, a closed form) agree on the class A mappings, catching unit and convention errors cheaply.
 * Rendered by `openpbr-check.ts` (128x128, 256 spp: well under a minute each on the CPU).
 *
 * `expected` holds the closed-form centre-pixel value where one exists; the checks use a small field of view so the incidence angle
 * stays under ~3 degrees (cos error < 1.4e-3).
 */
import type { MaterialDesc, MeshDesc, Rgb, SceneDesc } from '../scenes/schema.ts';
import { compose, rotationX, tessellateSphere, translation } from '../scenes/geometry.ts';

const FACE_CAMERA = rotationX(Math.PI / 2); // local +Y -> world +Z
const film = { width: 128, height: 128, filter: 'box' as const };
const black: MaterialDesc = { type: 'black' };

const sphereOf = (name: string, material: string, extra: Partial<MeshDesc> = {}): MeshDesc => ({ name, geometry: { type: 'sphere', radius: 1, segments: 64, rings: 32 }, material, ...extra });
const quadLight = (radiance: number): MeshDesc => ({
  name: 'light',
  geometry: { type: 'quad', size: [2, 2] },
  material: 'black',
  transform: compose(translation(0, 3, 1), rotationX(Math.PI)),
  emission: [radiance, radiance, radiance],
});

/** Emissive wall at z = -1 behind an object at z = 0, camera at z = 4 looking down -Z with a narrow field of view. */
function throughScene(name: string, purpose: string, materials: Record<string, MaterialDesc>, object: MeshDesc, expected: Rgb | undefined, extra: Partial<SceneDesc> = {}): SceneDesc {
  return {
    version: 1,
    name,
    purpose,
    tier: 1,
    camera: { type: 'perspective', position: [0, 0, 4], target: [0, 0, 0], up: [0, 1, 0], yFovDeg: 6 },
    film,
    integrator: { maxDepth: 64 },
    materials: { black, ...materials },
    meshes: [
      { name: 'wall', geometry: { type: 'quad', size: [8, 8] }, material: 'black', transform: compose(translation(0, 0, -1), FACE_CAMERA), emission: [1, 1, 1] },
      object,
    ],
    ...(expected ? { expected: { type: 'mean' as const, value: expected, tolerance: 0.01 } } : {}),
    ...extra,
  };
}

/** Fresnel reflectance at normal incidence of an interface of relative index n. */
export const fresnelNormal = (n: number): number => ((n - 1) / (n + 1)) ** 2;
/** Total transmittance of a thin-walled dielectric sheet with multiple internal reflections, normal incidence: (1 - R) / (1 + R). */
export const thinSheetTransmittance = (n: number): number => (1 - fresnelNormal(n)) / (1 + fresnelNormal(n));
/** Slab of absorbing glass: two interfaces, per-pass transmittance a, all internal reflections: (1 - R)^2 a / (1 - R^2 a^2). */
export const slabTransmittance = (n: number, a: number): number => ((1 - fresnelNormal(n)) ** 2 * a) / (1 - (fresnelNormal(n) * a) ** 2);

const ABSORB_COLORS: Record<string, Rgb> = { 'oc-absorb': [0.5, 0.25, 0.8], 'oc-absorb-grey': [0.3, 0.3, 0.3] };
const SLAB = 0.5;

const absorbScene = (name: string): SceneDesc => {
  const color = ABSORB_COLORS[name]!;
  return throughScene(
    name,
    `Absorbing glass slab (ior 1.5, ${SLAB} m, transmission_color ${color.join('/')} at depth ${SLAB} m) in front of an emissive wall: the centre pixel is the closed-form slab transmittance (1-R)^2 a / (1-R^2 a^2) per channel (Beer-Lambert through a MediumInterface/medium + volpath).`,
    { glass: { type: 'openpbr', baseColor: [1, 1, 1], specularRoughness: 0, specularIor: 1.5, transmissionWeight: 1, transmissionColor: color, transmissionDepth: SLAB } },
    { name: 'slab', geometry: { type: 'box', size: [6, 6, SLAB] }, material: 'glass' },
    color.map((a) => slabTransmittance(1.5, a)) as unknown as Rgb,
  );
};

const sphereScene = (name: string, purpose: string, material: MaterialDesc, mesh: Partial<MeshDesc> = {}, maxDepth = 8): SceneDesc => ({
  version: 1,
  name,
  purpose,
  tier: 1,
  camera: { type: 'perspective', position: [0, 0, 4], target: [0, 0, 0], up: [0, 1, 0], yFovDeg: 30 },
  film,
  integrator: { maxDepth },
  materials: { black, m: material },
  meshes: [sphereOf('sphere', 'm', mesh), quadLight(8)],
  environment: { type: 'uniform', radiance: [0.4, 0.4, 0.4] },
});

/** The closed-form (class A) scenes first, then sphere scenes that only compare the two renderers. */
export const openPbrCheckScenes: SceneDesc[] = [
  throughScene(
    'oc-thin',
    'Thin-walled smooth dielectric sheet (ior 1.5) in front of an emissive wall: the centre pixel is the sheet transmittance 2n/(n^2+1) = 0.9231 (multiple internal reflections included).',
    { sheet: { type: 'openpbr', baseColor: [1, 1, 1], specularRoughness: 0, specularIor: 1.5, transmissionWeight: 1, geometryThinWalled: true } },
    { name: 'sheet', geometry: { type: 'quad', size: [2, 2] }, material: 'sheet', transform: FACE_CAMERA },
    [thinSheetTransmittance(1.5), thinSheetTransmittance(1.5), thinSheetTransmittance(1.5)],
  ),
  absorbScene('oc-absorb'),
  absorbScene('oc-absorb-grey'),
  {
    version: 1,
    name: 'oc-emission',
    purpose: 'Emission of an openpbr material: a quad with emission_luminance 3 x emission_color (1, 0.5, 0.25) seen head on over a black environment reads 3, 1.5, 0.75 (the base reflects black).',
    tier: 1,
    camera: { type: 'perspective', position: [0, 0, 4], target: [0, 0, 0], up: [0, 1, 0], yFovDeg: 6 },
    film,
    integrator: { maxDepth: 4 },
    materials: { lamp: { type: 'openpbr', baseColor: [0.5, 0.5, 0.5], emissionLuminance: 3, emissionColor: [1, 0.5, 0.25] } },
    meshes: [{ name: 'lamp', geometry: { type: 'quad', size: [8, 8] }, material: 'lamp', transform: FACE_CAMERA }],
    expected: { type: 'mean', value: [3, 1.5, 0.75], tolerance: 0.01 },
  },
  sphereScene('oc-metal', 'Rough grey metal sphere (reflectance 0.9, roughness 0.4) under a uniform sky and a quad light: pbrt conductor vs Mitsuba roughconductor, both with eta = 1 and k from the reflectance.', { type: 'openpbr', baseColor: [0.9, 0.9, 0.9], baseMetalness: 1, specularRoughness: 0.4 }),
  sphereScene('oc-metal-copper', 'As oc-metal with a coloured reflectance (0.95, 0.64, 0.54): pbrt uplifts the RGB reflectance spectrally, Mitsuba (llvm_ad_rgb) works per channel; the difference is the spectral-uplift bias, informational.', { type: 'openpbr', baseColor: [0.95, 0.64, 0.54], baseMetalness: 1, specularRoughness: 0.4 }),
  sphereScene('oc-aniso', 'Anisotropic metal (roughness 0.4, anisotropy 0.8) on a uv sphere: tests that pbrt and Mitsuba take the same tangent axis (dp/du of the PLY uv); u-alpha is the tangent direction.', { type: 'openpbr', baseColor: [0.9, 0.9, 0.9], baseMetalness: 1, specularRoughness: 0.4, specularRoughnessAnisotropy: 0.8 }, { uv: {} }),
  sphereScene('oc-glass', 'Rough glass sphere (ior 1.5, roughness 0.3, no absorption): pbrt dielectric vs Mitsuba roughdielectric.', { type: 'openpbr', baseColor: [1, 1, 1], specularRoughness: 0.3, specularIor: 1.5, transmissionWeight: 1 }, {}, 16),
  sphereScene('oc-mix', 'Metalness 0.5 on a transmissive base: pbrt mix / Mitsuba blendbsdf of rough glass and rough grey metal (exact linear blend of two exact ends).', { type: 'openpbr', baseColor: [0.9, 0.9, 0.9], baseMetalness: 0.5, specularRoughness: 0.3, transmissionWeight: 1 }, {}, 16),
  sphereScene('oc-coated-clay', 'Dielectric over a clay diffuse base (class B): pbrt coateddiffuse (stochastic layers) vs Mitsuba roughplastic nonlinear; informational, they differ by a few percent.', { type: 'openpbr', baseColor: [0.7, 0.3, 0.2], specularRoughness: 0.3 }),
  sphereScene('oc-coat-metal', 'Clear coat over rough grey metal (class B, pbrt coatedconductor); Mitsuba has no equivalent (principled with the coverage option).', { type: 'openpbr', baseColor: [0.9, 0.9, 0.9], baseMetalness: 1, specularRoughness: 0.4, coatWeight: 1, coatRoughness: 0.1, coatIor: 1.5 }),
];

/**
 * White furnace for the dielectric-over-diffuse mapping: a flat-shaded sphere with a white base under a smooth clear coat (ior 1.5) in a
 * uniform white sky. Lossless, so every pixel must read 1; pbrt's `coateddiffuse` loses energy once `maxdepth` truncates the walk between
 * the layers (the P2.20 `maxdepth` sweep). `roughness` 0.3 and 0 are both used.
 */
const flatSphere = (() => {
  const m = tessellateSphere(1, 128, 64, { flat: true });
  return { type: 'triangles' as const, positions: Array.from(m.positions), indices: Array.from(m.indices), normals: Array.from(m.normals) };
})();

export function coatFurnaceScene(roughness: number): SceneDesc {
  return {
    version: 1,
    name: `oc-furnace-coat-r${String(roughness).replace('.', '')}`,
    purpose: `White furnace: white diffuse base under a clear coat (ior 1.5, roughness ${roughness}) inside a flat-shaded sphere, uniform L = 1; lossless, expects 1 everywhere.`,
    tier: 1,
    camera: { type: 'perspective', position: [0, 0, 2], target: [0, 0, 0], up: [0, 1, 0], yFovDeg: 40 },
    film,
    integrator: { maxDepth: 4 },
    materials: { m: { type: 'openpbr', baseColor: [1, 1, 1], specularRoughness: roughness, specularIor: 1.5 } },
    meshes: [{ name: 'sphere', geometry: flatSphere, material: 'm' }],
    environment: { type: 'uniform', radiance: [1, 1, 1] },
    expected: { type: 'uniform', value: [1, 1, 1], tolerance: 0.01 },
  };
}
