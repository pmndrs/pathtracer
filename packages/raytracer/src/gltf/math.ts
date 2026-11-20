/** Column-major 4x4 helpers for the glTF reader (copies of bench/scenes/geometry.ts, which the bench's tessellation keeps using). */
type Mat4 = readonly number[];
type Vec3 = readonly [number, number, number];

export const identity = (): number[] => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

/** a · b (apply b first, then a). */
export function multiply(a: Mat4, b: Mat4): number[] {
  const out = new Array<number>(16).fill(0);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      let sum = 0;
      for (let k = 0; k < 4; k++) sum += a[k * 4 + r]! * b[c * 4 + k]!;
      out[c * 4 + r] = sum;
    }
  }
  return out;
}

/** Compose left-to-right as written in code order: `compose(T, R, S)` = T·R·S (S applied first). */
export function compose(...ms: Mat4[]): number[] {
  return ms.reduce<number[]>((acc, m) => multiply(acc, m), identity());
}

export function transformPoint(m: Mat4, p: Vec3): [number, number, number] {
  const [x, y, z] = p;
  const w = m[3]! * x + m[7]! * y + m[11]! * z + m[15]!;
  const iw = w !== 0 && w !== 1 ? 1 / w : 1;
  return [
    (m[0]! * x + m[4]! * y + m[8]! * z + m[12]!) * iw,
    (m[1]! * x + m[5]! * y + m[9]! * z + m[13]!) * iw,
    (m[2]! * x + m[6]! * y + m[10]! * z + m[14]!) * iw,
  ];
}

/** Determinant of the upper-left 3×3: negative means the transform mirrors (flips triangle winding). */
export function determinant3(m: Mat4): number {
  return (
    m[0]! * (m[5]! * m[10]! - m[9]! * m[6]!) -
    m[4]! * (m[1]! * m[10]! - m[9]! * m[2]!) +
    m[8]! * (m[1]! * m[6]! - m[5]! * m[2]!)
  );
}

export function cross3(a: Vec3, b: Vec3): [number, number, number] {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
