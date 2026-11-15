/**
 * R1: a coarse copy of the fuzz layer's directional albedo `E_fuzz(mu, alpha)` (the LTC table's R, `ptFuzzAlbedo`) that a reduced vertex can read without the fuzz table
 * binding or lobe code: 9 x 9 samples, mu on a sqrt axis (the albedo climbs steeply at grazing), alpha on a linear axis, bilinear. Only the albedo is needed: the reduced
 * material folds the fuzz into its lobe choice (`ptBsdfReduced`), it never evaluates the LTC lobe.
 */
import { ptFuzzAlbedo } from './fuzz.ts';

export const PT_RED_FUZZ_N = 9;

let cached: Float32Array | null = null;

/** Row-major (alpha, sqrt mu), rounded to the 4 decimals the WGSL const carries. */
export function ptRedFuzzTable(): Float32Array {
  if (cached) return cached;
  const n = PT_RED_FUZZ_N;
  const t = new Float32Array(n * n);
  for (let ia = 0; ia < n; ia++) {
    for (let im = 0; im < n; im++) {
      const mu = (im / (n - 1)) ** 2;
      t[ia * n + im] = Math.fround(Math.round(ptFuzzAlbedo(Math.min(mu, 0.999), ia / (n - 1)) * 1e4) / 1e4);
    }
  }
  cached = t;
  return t;
}

/** Bilinear lookup (the CPU twin of `ptRedFuzzAlbedo` in WGSL). */
export function ptRedFuzzAlbedo(mu: number, alpha: number): number {
  const n = PT_RED_FUZZ_N;
  const t = ptRedFuzzTable();
  const x = Math.sqrt(Math.min(Math.max(mu, 0), 1)) * (n - 1);
  const y = Math.min(Math.max(alpha, 0), 1) * (n - 1);
  const x0 = Math.min(Math.floor(x), n - 2);
  const y0 = Math.min(Math.floor(y), n - 2);
  const fx = x - x0;
  const fy = y - y0;
  const at = (iy: number, ix: number): number => t[iy * n + ix] as number;
  return (at(y0, x0) * (1 - fx) + at(y0, x0 + 1) * fx) * (1 - fy) + (at(y0 + 1, x0) * (1 - fx) + at(y0 + 1, x0 + 1) * fx) * fy;
}

/** The WGSL twin: the table and `ptRedFuzzAlbedo(mu, alpha)`. */
export function ptRedFuzzWgsl(): string {
  const n = PT_RED_FUZZ_N;
  const list = Array.from(ptRedFuzzTable())
    .map((v) => {
      const s = String(Number(v.toFixed(4)));
      return s.includes('.') ? s.replace(/^0\./, '.') : `${s}.0`;
    })
    .join(',');
  return /* wgsl */ `
var<private> PT_RED_FUZZ: array<f32, ${n * n}> = array<f32, ${n * n}>(${list});
fn ptRedFuzzAlbedo(mu: f32, alpha: f32) -> f32 {
    let x = sqrt(clamp(mu, 0.0, 1.0)) * ${n - 1}.0;
    let y = clamp(alpha, 0.0, 1.0) * ${n - 1}.0;
    let x0 = u32(min(floor(x), ${n - 2}.0));
    let y0 = u32(min(floor(y), ${n - 2}.0));
    let fx = x - f32(x0);
    let fy = y - f32(y0);
    let i00 = y0 * ${n}u + x0;
    return (PT_RED_FUZZ[i00] * (1.0 - fx) + PT_RED_FUZZ[i00 + 1u] * fx) * (1.0 - fy) + (PT_RED_FUZZ[i00 + ${n}u] * (1.0 - fx) + PT_RED_FUZZ[i00 + ${n + 1}u] * fx) * fy;
}
`;
}
