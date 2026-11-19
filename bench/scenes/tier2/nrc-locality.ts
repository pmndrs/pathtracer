import { compose, rotationX, translation } from '../geometry.ts';
import type { CameraDesc, MeshDesc, SceneDesc } from '../schema.ts';

type Crop = readonly [left: number, top: number, right: number, bottom: number];
type Bounds = { min: [number, number, number]; max: [number, number, number] };

export interface LocalityView {
  tag: string;
  camera: CameraDesc;
  crop: Crop;
  /** Reference channel expected to dominate in this side's reflected light. */
  dominant?: 0 | 2;
}

interface LocalityFixture {
  scene: SceneDesc;
  bounds: Bounds;
  views: readonly LocalityView[];
  movedScene?: SceneDesc;
}

const camera = (position: CameraDesc['position'], target: CameraDesc['target'], yFovDeg: number): CameraDesc =>
  ({ type: 'perspective', position, target, up: [0, 1, 0], yFovDeg });

const floor: MeshDesc = { name: 'glossy-floor', material: 'metal', geometry: { type: 'quad', size: [8, 8] } };

const quad = (name: string, material: string, size: [number, number], x: number, y: number, z: number, pitch: number): MeshDesc =>
  ({ name, material, geometry: { type: 'quad', size }, transform: compose(translation(x, y, z), rotationX(pitch)) });

const base = {
  version: 1,
  tier: 2,
  integrator: { maxDepth: 6, rrStartDepth: 4 },
  environment: { type: 'uniform', radiance: [0.015, 0.015, 0.015] },
  materials: {
    wall: { type: 'diffuse', reflectance: [0.68, 0.68, 0.68] },
    metal: { type: 'conductor', eta: [0.2, 0.2, 0.2], k: [3.9, 3.9, 3.9], roughness: 0.3 },
    card: { type: 'diffuse', reflectance: [0.045, 0.24, 0.84] },
    light: { type: 'black' },
  },
} as const satisfies Omit<SceneDesc, 'name' | 'purpose' | 'camera' | 'film' | 'meshes'>;

export function thinWallFixture(width: number, height: number, thickness: number): LocalityFixture {
  // Same plane in screen coordinates from opposite sides. The floor reflects light
  // from the partition after the primary bounce, where the cache starts.
  const crop = [0.14, 0.64, 0.86, 0.93] as const;
  const views: LocalityView[] = [
    { tag: 'red', camera: camera([0, 1.32, 3.15], [0, 0.60, 0], 35), crop, dominant: 0 },
    { tag: 'blue', camera: camera([0, 1.32, -3.15], [0, 0.60, 0], 35), crop, dominant: 2 },
  ];
  return {
    scene: {
      ...base,
      name: 'nrc-thin-wall',
      purpose: 'Opposite faces of one thin partition reflect independent warm and cool illumination into the floor.',
      camera: views[0]!.camera,
      film: { width, height, filter: 'box' },
      meshes: [
        floor,
        { name: 'partition', material: 'wall', geometry: { type: 'box', size: [8, 3, thickness] }, transform: translation(0, 1.499, 0) },
        { ...quad('red-lamp', 'light', [5, 0.9], 0, 2.82, 1.8, Math.PI), emission: [16, 1.8, 0.18] },
        { ...quad('blue-lamp', 'light', [5, 0.9], 0, 2.82, -1.8, Math.PI), emission: [0.18, 1.8, 16] },
      ],
    },
    bounds: { min: [-4, -0.0011, -4], max: [4, 3, 4] },
    views,
  };
}

export function movingCardFixture(width: number, height: number): LocalityFixture {
  const view: LocalityView = {
    tag: 'card', camera: camera([0, 1.15, 3.3], [0, 0.52, -1.8], 32), crop: [0.16, 0.71, 0.84, 0.98],
  };
  const card: MeshDesc = {
    name: 'moving-card', material: 'card', geometry: { type: 'box', size: [0.78, 0.96, 0.075] }, transform: translation(-0.66, 0.49, -1.36),
  };
  const scene: SceneDesc = {
    ...base,
    name: 'nrc-moving-card',
    purpose: 'One opaque card moves across a glossy floor at a fixed camera and light, disoccluding the wall and changing its reflection.',
    camera: view.camera,
    film: { width, height, filter: 'box' },
    meshes: [
      floor,
      quad('back-wall', 'wall', [8, 3], 0, 1.5, -1.72, Math.PI / 2),
      { ...quad('lamp', 'light', [3.1, 0.65], -0.5, 2.35, -0.3, Math.PI), emission: [10, 8.9, 7.8] },
      card,
    ],
  };
  const movedCard = { ...card, transform: translation(0.66, 0.49, -1.36) };
  return {
    scene,
    movedScene: { ...scene, meshes: [...scene.meshes.slice(0, -1), movedCard] },
    bounds: { min: [-4, 0, -4], max: [4, 3, 4] },
    views: [view],
  };
}
