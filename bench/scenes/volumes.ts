/**
 * Bench scene media -> the path tracer's `PtVolumeOptions` (P5). Browser-safe (no node imports); the grid of a `grid` medium is the scene's own array.
 *
 * A medium's coefficients are pbrt's: `sigmaA`, `sigmaS` per metre times `scale`. A `grid` medium lives in the object space of the (single) mesh that has it as its
 * `interior`; its density is looked up through that mesh's transform. One grid medium per scene (one 3D texture per kernel).
 */
import { gridWorldToIndex, ptMediumFrom, type PtMedium, type PtMaterialVolume, type PtVolumeOptions } from '@pmndrs/pathtracer';
import type { DescSceneCopyInput } from './to-scene-copy.ts';
import type { SceneDesc } from './schema.ts';

export interface SceneVolumes {
  options: PtVolumeOptions;
  /** The density texture (x fastest) of the scene's one grid medium. */
  grid?: { resolution: readonly [number, number, number]; density: Float32Array };
}

export function sceneHasMedia(scene: SceneDesc): boolean {
  return Object.keys(scene.media ?? {}).length > 0 || scene.meshes.some((m) => m.medium) || scene.cameraMedium !== undefined;
}

export function sceneVolumes(scene: SceneDesc, input: Pick<DescSceneCopyInput, 'materialVolumes'>): SceneVolumes | null {
  if (!sceneHasMedia(scene)) return null;
  const names = Object.keys(scene.media ?? {});
  let grid: SceneVolumes['grid'];
  const media: PtMedium[] = names.map((name) => {
    const d = (scene.media as NonNullable<SceneDesc['media']>)[name]!;
    if (d.type === 'homogeneous') return ptMediumFrom(d.sigmaA, d.sigmaS, d.g ?? 0, d.scale ?? 1);
    if (grid) throw new RangeError(`scene '${scene.name}': more than one grid medium (one 3D texture per kernel)`);
    const owners = scene.meshes.filter((m) => m.medium?.interior === name);
    if (owners.length !== 1) throw new RangeError(`scene '${scene.name}': grid medium '${name}' must be the interior of exactly one mesh (found ${owners.length}): its transform places the grid`);
    const density = Float32Array.from(d.density as ArrayLike<number>);
    const [nx, ny, nz] = d.resolution;
    if (density.length !== nx * ny * nz) throw new RangeError(`scene '${scene.name}': grid medium '${name}' has ${density.length} values for ${nx} x ${ny} x ${nz}`);
    let max = 0;
    for (const v of density) max = Math.max(max, v);
    grid = { resolution: d.resolution, density };
    return ptMediumFrom(d.sigmaA, d.sigmaS, d.g ?? 0, d.scale ?? 1, { resolution: d.resolution, worldToIndex: gridWorldToIndex(d.resolution, d.bounds, owners[0]!.transform), densityMax: max });
  });
  const materials: PtMaterialVolume[] = input.materialVolumes.map((m) => ({ ...m }));
  const cam = scene.cameraMedium ? names.indexOf(scene.cameraMedium) + 1 : 0;
  if (scene.cameraMedium && cam === 0) throw new RangeError(`scene '${scene.name}': unknown cameraMedium '${scene.cameraMedium}'`);
  return { options: { media, materials, ...(cam ? { cameraMedium: cam } : {}) }, ...(grid ? { grid } : {}) };
}
