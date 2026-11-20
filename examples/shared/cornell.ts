/**
 * A Cornell box as a scene-copy input, built from quads: a 2 m room (open towards +z), a ceiling light, a tall and a short box. Every
 * triangle is counter-clockwise seen from the side its normal points to (the front face; the light emits from its front face only).
 */
import type { PtLookAtCamera } from '@pmndrs/pathtracer';
import type { RtMaterialInput, SceneCopyInput, SceneCopyInstance, SceneCopyMesh } from '@pmndrs/raytracer';

type V3 = [number, number, number];

/** Quads a, b, c, d (counter-clockwise from the front) as two triangles with flat normals. */
export function mesh(quads: readonly (readonly V3[])[]): SceneCopyMesh {
  const positions: number[] = [], normals: number[] = [], indices: number[] = [];
  for (const [a, b, c, d] of quads as V3[][]) {
    const e1 = [b![0] - a![0], b![1] - a![1], b![2] - a![2]], e2 = [c![0] - a![0], c![1] - a![1], c![2] - a![2]];
    const n = [e1[1]! * e2[2]! - e1[2]! * e2[1]!, e1[2]! * e2[0]! - e1[0]! * e2[2]!, e1[0]! * e2[1]! - e1[1]! * e2[0]!];
    const l = Math.hypot(n[0]!, n[1]!, n[2]!);
    const base = positions.length / 3;
    for (const p of [a!, b!, c!, d!]) {
      positions.push(...p);
      normals.push(n[0]! / l, n[1]! / l, n[2]! / l);
    }
    indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
  }
  return { positions: new Float32Array(positions), normals: new Float32Array(normals), indices: new Uint32Array(indices) };
}

/** An outward-facing box: centre, half sizes, rotation about +y in degrees. */
export function box(center: V3, half: V3, rotYDeg: number): readonly V3[][] {
  const r = (rotYDeg * Math.PI) / 180, c = Math.cos(r), s = Math.sin(r);
  const p = (x: number, y: number, z: number): V3 => {
    const lx = x * half[0], ly = y * half[1], lz = z * half[2];
    return [center[0] + c * lx + s * lz, center[1] + ly, center[2] - s * lx + c * lz];
  };
  return [
    [p(-1, 1, 1), p(1, 1, 1), p(1, 1, -1), p(-1, 1, -1)],
    [p(-1, -1, -1), p(1, -1, -1), p(1, -1, 1), p(-1, -1, 1)],
    [p(-1, -1, 1), p(1, -1, 1), p(1, 1, 1), p(-1, 1, 1)],
    [p(1, -1, -1), p(-1, -1, -1), p(-1, 1, -1), p(1, 1, -1)],
    [p(1, -1, 1), p(1, -1, -1), p(1, 1, -1), p(1, 1, 1)],
    [p(-1, -1, -1), p(-1, -1, 1), p(-1, 1, 1), p(-1, 1, -1)],
  ];
}

export interface CornellOptions {
  /** Radiance of the ceiling light (W/sr/m^2). Default a warm (17, 12, 4). */
  light?: V3;
}

/** Diffuse (Lambert: `specularWeight` 0) unless said otherwise. */
const lambert = (baseColor: V3): RtMaterialInput => ({ baseColor, specularWeight: 0 });

export function cornellBox(options: CornellOptions = {}): SceneCopyInput {
  const materials: RtMaterialInput[] = [
    lambert([0.73, 0.73, 0.73]),
    lambert([0.63, 0.065, 0.05]),
    lambert([0.14, 0.45, 0.091]),
    { baseColor: [0.78, 0.78, 0.78], specularWeight: 0, emissionColor: options.light ?? [17, 12, 4] },
  ];
  const parts: { quads: readonly (readonly V3[])[]; material: number }[] = [
    { // floor, ceiling, back wall
      quads: [
        [[-1, 0, 1], [1, 0, 1], [1, 0, -1], [-1, 0, -1]],
        [[-1, 2, -1], [1, 2, -1], [1, 2, 1], [-1, 2, 1]],
        [[-1, 0, -1], [1, 0, -1], [1, 2, -1], [-1, 2, -1]],
      ],
      material: 0,
    },
    { quads: [[[-1, 0, 1], [-1, 0, -1], [-1, 2, -1], [-1, 2, 1]]], material: 1 },
    { quads: [[[1, 0, -1], [1, 0, 1], [1, 2, 1], [1, 2, -1]]], material: 2 },
    { quads: [[[-0.25, 1.99, -0.25], [0.25, 1.99, -0.25], [0.25, 1.99, 0.25], [-0.25, 1.99, 0.25]]], material: 3 },
    { quads: box([-0.33, 0.6, -0.3], [0.3, 0.6, 0.3], 18), material: 0 },
    { quads: box([0.35, 0.3, 0.35], [0.3, 0.3, 0.3], -17), material: 0 },
  ];
  const meshes = parts.map((p) => mesh(p.quads));
  const instances: SceneCopyInstance[] = parts.map((p, i) => ({ meshIndex: i, instanceIndex: i, objectToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], material: p.material }));
  return { meshes, instances, materials };
}

export const CORNELL_CAMERA: PtLookAtCamera = { position: [0, 1, 3.9], target: [0, 1, 0], up: [0, 1, 0], yFovDeg: 40 };
