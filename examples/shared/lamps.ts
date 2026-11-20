/**
 * The many-lights scene of the Lights and ReSTIR examples: 64 small coloured lamps over a floor with a few boxes, under a dim sky and a
 * low sun. `selection` is the scene copy's light-selection strategy (it lives in the light table).
 */
import type { LightSelection, RtMaterialInput, SceneCopyInput } from '@pmndrs/raytracer';
import { sunFrom } from './models.ts';

export const LAMPS_CAMERA = { position: [6.5, 4.5, 7] as [number, number, number], target: [0, 0.3, 0] as [number, number, number], up: [0, 1, 0] as [number, number, number], yFovDeg: 45 };

/** An axis-aligned box as quads facing outward. */
function box(cx: number, cz: number, w: number, h: number, d: number) {
  const x0 = cx - w / 2, x1 = cx + w / 2, z0 = cz - d / 2, z1 = cz + d / 2;
  const faces = [
    [[x0, h, z1], [x1, h, z1], [x1, h, z0], [x0, h, z0]],
    [[x0, 0, z1], [x1, 0, z1], [x1, h, z1], [x0, h, z1]],
    [[x1, 0, z0], [x0, 0, z0], [x0, h, z0], [x1, h, z0]],
    [[x1, 0, z1], [x1, 0, z0], [x1, h, z0], [x1, h, z1]],
    [[x0, 0, z0], [x0, 0, z1], [x0, h, z1], [x0, h, z0]],
  ];
  return { positions: new Float32Array(faces.flat(2)), indices: faces.flatMap((_, f) => [0, 1, 2, 0, 2, 3].map((k) => k + f * 4)) };
}

export function lampsScene(selection: LightSelection = 'pbrt'): SceneCopyInput {
  const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  const meshes = [
    { positions: new Float32Array([-6, 0, 6, 6, 0, 6, 6, 0, -6, -6, 0, -6]), indices: [0, 1, 2, 0, 2, 3] },
    box(-1.5, -0.5, 1, 1.2, 1), box(1.2, 0.8, 0.8, 0.6, 1.4), box(0.3, -2, 2.5, 0.4, 0.6),
  ];
  const materials: RtMaterialInput[] = [{ baseColor: [0.7, 0.7, 0.7], specularRoughness: 0.5 }];
  const instances = meshes.map((_, i) => ({ meshIndex: i, instanceIndex: i, objectToWorld: identity, material: 0 }));
  // 64 lamps: small downward-facing quads, each with its own colour (one material per lamp).
  let seed = 3;
  const rand = () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 4294967296);
  for (let i = 0; i < 64; i++) {
    const x = (rand() - 0.5) * 9, z = (rand() - 0.5) * 9, y = 0.4 + rand() * 1.4, s = 0.08;
    meshes.push({ positions: new Float32Array([x - s, y, z - s, x + s, y, z - s, x + s, y, z + s, x - s, y, z + s]), indices: [0, 1, 2, 0, 2, 3] });
    const h = rand() * 6;
    const rgb = [0, 2, 4].map((k) => Math.max(0, Math.min(1, Math.abs(((h + k) % 6) - 3) - 1))) as [number, number, number];
    materials.push({ baseColor: [0, 0, 0], specularWeight: 0, emissionColor: rgb.map((c) => 40 * (0.15 + c)) as [number, number, number] });
    instances.push({ meshIndex: meshes.length - 1, instanceIndex: meshes.length - 1, objectToWorld: identity, material: materials.length - 1 });
  }
  return { meshes, instances, materials, sun: sunFrom(12, -60, 0.6), environment: { radiance: [0.02, 0.025, 0.04] }, options: { lightSelection: selection } };
}

