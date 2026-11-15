import type { MeshDesc, SceneDesc } from '../schema.ts';
import { translation } from '../geometry.ts';

/**
 * Many small area lights (P4 ReSTIR DI): a diffuse plane with two blockers under a `n x n` grid of one-sided quad lamps (0.1 m) hanging 1 m above it, four emission levels
 * (alias-table power picks the strong ones but the weak ones near a shading point matter). Most lamps are occluded or cosine-poor for a given point, so plain NEE with a power
 * pick is noisy and RIS / reuse has something to win. No pbrt/Mitsuba reference (the lamp count makes them slow): judge against our own high-spp NEE render.
 */
function manyLights(name: string, n: number): SceneDesc {
  const levels = [2, 8, 32, 128];
  const groups: Record<number, { x: number; z: number }[]> = {};
  const span = 7;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const level = (i * 7 + j * 13 + ((i * j) % 5)) % levels.length;
      (groups[level] ??= []).push({ x: -span / 2 + ((i + 0.5) * span) / n, z: -span / 2 + ((j + 0.5) * span) / n });
    }
  }
  const lampMeshes: MeshDesc[] = [];
  for (const [level, lamps] of Object.entries(groups)) {
    const e = levels[Number(level)] as number;
    // one triangle mesh per emission level (all its lamps); emission is per mesh
    const positions: number[] = [];
    const indices: number[] = [];
    const normals: number[] = [];
    const h = 0.05;
    for (const lamp of lamps) {
      const base = positions.length / 3;
      // quad facing down (-Y), CCW seen from below
      positions.push(lamp.x - h, 1, lamp.z - h, lamp.x + h, 1, lamp.z - h, lamp.x + h, 1, lamp.z + h, lamp.x - h, 1, lamp.z + h);
      for (let k = 0; k < 4; k++) normals.push(0, -1, 0);
      indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
    }
    lampMeshes.push({ name: `lamps-${e}`, geometry: { type: 'triangles', positions, indices, normals }, material: 'lightBack', emission: [e, e, e] });
  }
  return {
    version: 1,
    name,
    purpose: `${n * n} small one-sided quad lamps (four power levels) over a diffuse plane with two blockers, direct lighting only: many-light NEE / ReSTIR DI A/B (P4). No native reference: compare to our own high-spp NEE.`,
    tier: 1,
    camera: { type: 'perspective', position: [0, 2.2, 3.6], target: [0, 0, 0.2], up: [0, 1, 0], yFovDeg: 50 },
    film: { width: 256, height: 256, filter: 'box' },
    integrator: { maxDepth: 1 },
    materials: { floor: { type: 'diffuse', reflectance: [0.5, 0.5, 0.5] }, block: { type: 'diffuse', reflectance: [0.7, 0.45, 0.3] }, lightBack: { type: 'black' } },
    meshes: [
      { name: 'plane', geometry: { type: 'quad', size: [8, 8] }, material: 'floor' },
      { name: 'box', geometry: { type: 'box', size: [0.8, 0.6, 0.8] }, material: 'block', transform: translation(-0.9, 0.3, 0.2) },
      { name: 'ball', geometry: { type: 'sphere', radius: 0.45, segments: 48, rings: 24 }, material: 'block', transform: translation(0.9, 0.45, -0.2) },
      ...lampMeshes,
    ],
  };
}

export const manyLights256 = manyLights('many-lights-256', 16);
export const manyLights64 = manyLights('many-lights-64', 8);
export const manyLightsScenes: readonly SceneDesc[] = [manyLights256, manyLights64];
