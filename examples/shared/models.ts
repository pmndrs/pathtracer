/**
 * Scenes the examples offer: the Cornell box built in code, or a Khronos sample model streamed from glTF-Sample-Assets at a pinned
 * commit (each model has its own licence; see the model's README in that repository). `loadScene` returns a scene-copy input with a
 * sky and a sun, and the camera to start from.
 */
import type { PtLookAtCamera } from '@pmndrs/pathtracer';
import { loadGltf, type SceneCopyInput, type SunInput } from '@pmndrs/raytracer';
import { cornellBox, CORNELL_CAMERA } from './cornell.ts';

const ASSETS = 'https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/8f492390ddc7a2ae221a1369ef560589cf8c70fb/Models/';

/** Label -> source: `cornell` or a model URL. */
export const SCENES: Record<string, string> = {
  'Cornell box': 'cornell',
  'Damaged Helmet': `${ASSETS}DamagedHelmet/glTF-Binary/DamagedHelmet.glb`,
  'Sheen Chair': `${ASSETS}SheenChair/glTF-Binary/SheenChair.glb`,
  'Toy Car': `${ASSETS}ToyCar/glTF-Binary/ToyCar.glb`,
  'Clear Coat Test': `${ASSETS}ClearCoatTest/glTF-Binary/ClearCoatTest.glb`,
  'Metal Rough Spheres': `${ASSETS}MetalRoughSpheres/glTF-Binary/MetalRoughSpheres.glb`,
  'Dragon Attenuation': `${ASSETS}DragonAttenuation/glTF-Binary/DragonAttenuation.glb`,
};

export interface LoadedScene {
  input: SceneCopyInput;
  camera: PtLookAtCamera;
  /** World bounds, for framing. */
  bounds: { min: number[]; max: number[] };
  name: string;
}

/** A sun from elevation / azimuth in degrees. `direction` is the way the light travels (down). */
export function sunFrom(elevationDeg: number, azimuthDeg: number, irradiance: number): SunInput {
  const el = (elevationDeg * Math.PI) / 180, az = (azimuthDeg * Math.PI) / 180;
  return { direction: [-Math.cos(el) * Math.sin(az), -Math.sin(el), -Math.cos(el) * Math.cos(az)], irradiance: [irradiance, irradiance * 0.95, irradiance * 0.88], angularRadius: 0.00465 };
}

/** Frames `bounds` from the front, slightly above. */
export function frameBounds(bounds: { min: readonly number[]; max: readonly number[] }, yFovDeg = 40): PtLookAtCamera {
  const c = [0, 1, 2].map((i) => (bounds.min[i]! + bounds.max[i]!) / 2) as [number, number, number];
  const r = Math.hypot(bounds.max[0]! - bounds.min[0]!, bounds.max[1]! - bounds.min[1]!, bounds.max[2]! - bounds.min[2]!) / 2 || 1;
  const d = (r / Math.sin((yFovDeg * Math.PI) / 360)) * 1.05;
  return { position: [c[0] + d * 0.34, c[1] + d * 0.26, c[2] + d * 0.9], target: c, up: [0, 1, 0], yFovDeg };
}

/** Loads a scene from `SCENES`. Models get a uniform sky and a sun; the Cornell box keeps its own ceiling light. */
export async function loadScene(source: string | ArrayBuffer, name: string, light: { sky: number; sun: SunInput | null } = { sky: 1, sun: sunFrom(50, 30, 3) }): Promise<LoadedScene> {
  if (source === 'cornell') {
    const input = cornellBox();
    return { input, camera: CORNELL_CAMERA, bounds: { min: [-1, 0, -1], max: [1, 2, 1] }, name };
  }
  const input = await loadGltf(source);
  input.environment = { radiance: [0.8 * light.sky, 0.9 * light.sky, light.sky] };
  input.sun = light.sun;
  if (input.notes.length) console.info(`glTF notes for ${name}:\n${input.notes.join('\n')}`);
  return { input, camera: input.cameras[0]?.camera ?? frameBounds(input.bounds), bounds: input.bounds, name };
}
