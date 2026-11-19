import { compose, rotationX, rotationZ, translation } from '../geometry.ts';
import type { MaterialDesc, MeshDesc, SceneDesc } from '../schema.ts';

const ROUGHNESS = [0.10, 0.17, 0.24, 0.31, 0.38, 0.45, 0.52];
const materials: Record<string, MaterialDesc> = {
  plastic: { type: 'openpbr', baseColor: [0.38, 0.25, 0.17], specularRoughness: 0.42, coatWeight: 1, coatRoughness: 0.19, coatIor: 1.6 },
  wall: { type: 'diffuse', reflectance: [0.55, 0.55, 0.55] },
  red: { type: 'diffuse', reflectance: [0.74, 0.13, 0.04] },
  blue: { type: 'diffuse', reflectance: [0.06, 0.24, 0.81] },
  light: { type: 'black' },
};

const metal: MeshDesc[] = ROUGHNESS.map((roughness, i) => {
  const name = `metal-${i}`;
  materials[name] = { type: 'openpbr', baseColor: [0.76, 0.83, 0.88], baseMetalness: 1, specularRoughness: roughness, normalTexture: 'bumps' };
  // Increase roughness along Z so the entire ramp is visible obliquely through the shot.
  return {
    name,
    material: name,
    geometry: { type: 'quad', size: [6, 1] },
    transform: translation(-3, 0, i - 3),
    uv: { scale: [2, 1] },
  };
});

export const nrcMaterials: SceneDesc = {
  version: 1,
  name: 'nrc-materials',
  purpose: 'Moving highlights on a normal-mapped conductor roughness ramp beside coated plastic.',
  tier: 2,
  camera: { type: 'perspective', position: [0, 2.2, 4.9], target: [0, 0.25, -1.3], up: [0, 1, 0], yFovDeg: 30 },
  film: { width: 1280, height: 720, filter: 'box' },
  integrator: { maxDepth: 6, rrStartDepth: 4 },
  environment: { type: 'uniform', radiance: [0.09, 0.09, 0.09] },
  textures: { bumps: { generator: 'bumps', width: 256, height: 128, colorSpace: 'linear', periods: 7, amplitude: 0.22 } },
  materials,
  meshes: [
    ...metal,
    { name: 'coated-plastic', material: 'plastic', geometry: { type: 'quad', size: [6, 7] }, transform: translation(3, 0, 0) },
    { name: 'back-wall', material: 'wall', geometry: { type: 'quad', size: [12, 5] }, transform: compose(translation(0, 2.5, -3.5), rotationX(Math.PI / 2)) },
    { name: 'red-panel', material: 'red', geometry: { type: 'quad', size: [2, 2] }, transform: compose(translation(-2.5, 1.1, -3.47), rotationX(Math.PI / 2)) },
    { name: 'blue-panel', material: 'blue', geometry: { type: 'quad', size: [2, 2] }, transform: compose(translation(2.5, 1.1, -3.47), rotationX(Math.PI / 2)) },
    { name: 'light', material: 'light', geometry: { type: 'quad', size: [4.5, 0.6] }, transform: compose(translation(0.6, 3.2, -0.65), rotationZ(0.15), rotationX(Math.PI)), emission: [16, 14, 12] },
  ],
};

/** Camera axis about 5.5 degrees above the metal floor. The lateral motion stays on its normal-mapped strips. */
export const nrcMaterialsGrazing: SceneDesc = {
  ...nrcMaterials, name: 'nrc-materials-grazing',
  camera: { ...nrcMaterials.camera, position: [-2.5, 0.45, 3.45], target: [-2.5, 0, -1.2], yFovDeg: 12 },
};

/** The blue card, its contact shadow, and its reflection in the coated floor share a single view. */
export const nrcMaterialsCard: SceneDesc = {
  ...nrcMaterials, name: 'nrc-materials-card',
  camera: { ...nrcMaterials.camera, position: [2.5, 1.05, 2], target: [2.5, 0.62, -3.45], yFovDeg: 32 },
};

/** Camera fitted to the four red-card corners in the shadow-flicker review. */
export const nrcMaterialsShadow: SceneDesc = {
  ...nrcMaterials, name: 'nrc-materials-shadow',
  camera: { ...nrcMaterials.camera, position: [2.51976, 1.00534, 2.04133], target: [1.73957, 1.10366, 1.42356], yFovDeg: 31.7051 },
};
