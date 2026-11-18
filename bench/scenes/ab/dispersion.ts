import { compose, rotationX, translation } from '../geometry.ts';
import type { MaterialDesc, SceneDesc } from '../schema.ts';

/**
 * Dispersion (P5, P2.28): a closed glass prism (equilateral, 1.6 m edge, 3 m tall, n_d 1.62) standing in front of a white emitter with a picket of black bars, seen from 6 m.
 * A ray through the prism leaves at a wavelength-dependent angle, so the stripe edges seen through the glass get red / blue fringes (the colours of the
 * spectrum), while the path to the emitter is a short specular chain: little noise. Dispersion is OpenPBR's (`transmissionDispersionScale` 1, Abbe number 20 =
 * KHR_materials_dispersion 1.0, a dense flint: strong on purpose); the pbrt export writes `spectrum eta` from the same Cauchy law and pbrt terminates the
 * secondary wavelengths as we do, so it is the reference. Mitsuba has a scalar IOR: no reference there.
 *
 *  - `dispersion-prism`: dispersion on.
 *  - `dispersion-prism-off`: the same scene with `transmissionDispersionScale` 0 (a plain n = 1.62 prism): spectral mode must reproduce the RGB render in the mean.
 */
const SIDE = 1.6;
const HEIGHT = 3;
const R = SIDE / Math.sqrt(3);

/** Flat-shaded closed prism: per face its own vertices and normals, CCW from outside. */
function prismGeometry(): { positions: number[]; indices: number[]; normals: number[] } {
  const tri = [0, 1, 2].map((i) => [R * Math.sin((i * 2 * Math.PI) / 3), R * Math.cos((i * 2 * Math.PI) / 3)] as const); // (x, z)
  const positions: number[] = [];
  const normals: number[] = [];
  const indices: number[] = [];
  const quad = (p: number[][], n: number[]): void => {
    const base = positions.length / 3;
    for (const v of p) {
      positions.push(...(v as [number, number, number]));
      normals.push(...n);
    }
    // wind so the geometric normal agrees with n
    const e1 = [p[1]![0]! - p[0]![0]!, p[1]![1]! - p[0]![1]!, p[1]![2]! - p[0]![2]!];
    const e2 = [p[2]![0]! - p[0]![0]!, p[2]![1]! - p[0]![1]!, p[2]![2]! - p[0]![2]!];
    const g = [e1[1]! * e2[2]! - e1[2]! * e2[1]!, e1[2]! * e2[0]! - e1[0]! * e2[2]!, e1[0]! * e2[1]! - e1[1]! * e2[0]!];
    const flip = g[0]! * n[0]! + g[1]! * n[1]! + g[2]! * n[2]! < 0;
    if (p.length === 4) indices.push(...(flip ? [base, base + 2, base + 1, base, base + 3, base + 2] : [base, base + 1, base + 2, base, base + 2, base + 3]));
    else indices.push(...(flip ? [base, base + 2, base + 1] : [base, base + 1, base + 2]));
  };
  const y0 = -HEIGHT / 2;
  const y1 = HEIGHT / 2;
  for (let i = 0; i < 3; i++) {
    const a = tri[i]!;
    const b = tri[(i + 1) % 3]!;
    const mx = (a[0] + b[0]) / 2;
    const mz = (a[1] + b[1]) / 2;
    const len = Math.hypot(mx, mz);
    quad([[a[0], y0, a[1]], [b[0], y0, b[1]], [b[0], y1, b[1]], [a[0], y1, a[1]]], [mx / len, 0, mz / len]);
  }
  quad(tri.map((v) => [v[0], y1, v[1]]), [0, 1, 0]);
  quad(tri.map((v) => [v[0], y0, v[1]]), [0, -1, 0]);
  return { positions, indices, normals };
}

const facing = (z: number) => compose(translation(0, 0, z), rotationX(Math.PI / 2));

const glass = (dispersion: number): MaterialDesc => ({ type: 'openpbr', baseColor: [1, 1, 1], transmissionWeight: 1, specularRoughness: 0, specularIor: 1.62, transmissionDispersionScale: dispersion, transmissionDispersionAbbeNumber: 20 });

const scene = (name: string, dispersion: number, purpose: string): SceneDesc => ({
  version: 1,
  name,
  purpose,
  tier: 1,
  camera: { type: 'perspective', position: [0, 0, 6], target: [0, 0, 0], up: [0, 1, 0], yFovDeg: 34 },
  film: { width: 320, height: 180, filter: 'box' },
  integrator: { maxDepth: 12 },
  materials: { glass: glass(dispersion), back: { type: 'black' } },
  meshes: [
    { name: 'prism', geometry: { type: 'triangles', ...prismGeometry() }, material: 'glass' },
    { name: 'emitter', geometry: { type: 'quad', size: [16, 9] }, material: 'back', transform: facing(-3), emission: [1, 1, 1] },
    // a picket pattern: 25 bars 0.12 m wide every 0.4 m, so every ray through the glass meets an edge and the red / blue edge fringes (~0.14 m apart at the backdrop) show as colour
    ...Array.from({ length: 25 }, (_, i) => ({ name: `bar${i}`, geometry: { type: 'quad' as const, size: [0.12, 9] as const }, material: 'back', transform: compose(translation(-4.8 + 0.4 * i, 0, -2.99), rotationX(Math.PI / 2)) })),
  ],
});

export const dispersionPrism = scene('dispersion-prism', 1, 'Dispersive glass prism (n_d 1.62, Abbe 20) in front of a striped emitter: colour fringes at the stripe edges (P5). pbrt-v4 (spectral eta) is the reference.');
export const dispersionPrismOff = scene('dispersion-prism-off', 0, 'The same prism without dispersion: the spectral mode must match the RGB render in the mean (P5).');
export const dispersionScenes: readonly SceneDesc[] = [dispersionPrism, dispersionPrismOff];
