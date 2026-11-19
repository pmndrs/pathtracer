import { compose, rotationX, translation } from '../geometry.ts';
import type { MeshDesc, SceneDesc } from '../schema.ts';

const receiver: MeshDesc = { name: 'receiver', material: 'receiver', geometry: { type: 'quad', size: [20, 20] } };
const lamp: MeshDesc = {
  name: 'lamp', material: 'light', geometry: { type: 'quad', size: [2.8, 1.8] },
  transform: compose(translation(0, 2.3, -1.4), rotationX(Math.PI)), emission: [4, 3.4, 2.6],
};
const bounceWall: MeshDesc = {
  name: 'red-bounce-wall', material: 'red', geometry: { type: 'quad', size: [8, 3] },
  transform: compose(translation(0, 1.5, -1.7), rotationX(Math.PI / 2)),
};

export const transportScenes: SceneDesc[] = (['sky', 'area', 'bounce'] as const).map(kind => ({
  version: 1,
  name: kind,
  purpose: 'Diffuse consumer transport under a sky, an area emitter, or an area emitter with a colored bounce wall.',
  tier: 2,
  camera: { type: 'perspective', position: [0, 2.1, 2.2], target: [0, 0, 0], up: [0, 1, 0], yFovDeg: 12 },
  film: { width: 48, height: 48, filter: 'box' },
  integrator: { maxDepth: 5 },
  materials: {
    receiver: { type: 'diffuse', reflectance: [0.4, 0.5, 0.6] },
    red: { type: 'diffuse', reflectance: [0.82, 0.12, 0.04] },
    light: { type: 'black' },
  },
  meshes: [receiver, ...(kind === 'sky' ? [] : [lamp]), ...(kind === 'bounce' ? [bounceWall] : [])],
  ...(kind === 'sky' ? { environment: { type: 'uniform', radiance: [0.5, 0.3, 0.2] } } : {}),
}));
