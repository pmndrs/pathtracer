/**
 * Camera medium detection (V3.3). Which medium is a point (the camera) inside, given the scene's medium boundary meshes (closed, OUTWARD wound,
 * world space)? CPU, TypeScript; run once at scene build, never in a frame.
 *
 * Test: the generalised winding number of the triangle soup around the point (sum of signed solid angles / 4 pi, Van Oosterom-Strakhov): 1 inside a closed
 * outward-wound mesh, 0 outside, with no ray-edge degeneracies (a ray-parity test has them on axis-aligned boxes). A point is inside when it is above 0.5.
 *
 * Nesting rule: of the meshes containing the point, the INNERMOST wins, measured by the volume of the mesh's bounding box (smallest wins; a tie keeps the
 * later entry). A cheap proxy that is exact for properly nested shells; a boundary mesh that merely overlaps another is not a supported scene.
 */
export interface PtMediumBoundary {
  /** The medium inside this boundary: 1-based id (0 = vacuum, the boundary is ignored). */
  medium: number;
  /** World-space xyz per vertex. */
  positions: ArrayLike<number>;
  /** 3 indices per triangle, counter-clockwise seen from outside. */
  indices: ArrayLike<number>;
}

/** Winding number of a triangle soup around `p` (1 inside a closed outward-wound mesh, 0 outside). */
export function ptWindingNumber(p: readonly [number, number, number], positions: ArrayLike<number>, indices: ArrayLike<number>): number {
  let sum = 0;
  for (let t = 0; t + 2 < indices.length; t += 3) {
    const a = (indices[t] as number) * 3, b = (indices[t + 1] as number) * 3, c = (indices[t + 2] as number) * 3;
    const ax = (positions[a] as number) - p[0], ay = (positions[a + 1] as number) - p[1], az = (positions[a + 2] as number) - p[2];
    const bx = (positions[b] as number) - p[0], by = (positions[b + 1] as number) - p[1], bz = (positions[b + 2] as number) - p[2];
    const cx = (positions[c] as number) - p[0], cy = (positions[c + 1] as number) - p[1], cz = (positions[c + 2] as number) - p[2];
    const la = Math.hypot(ax, ay, az), lb = Math.hypot(bx, by, bz), lc = Math.hypot(cx, cy, cz);
    const det = ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx);
    const den = la * lb * lc + (ax * bx + ay * by + az * bz) * lc + (bx * cx + by * cy + bz * cz) * la + (cx * ax + cy * ay + cz * az) * lb;
    sum += 2 * Math.atan2(det, den);
  }
  return sum / (4 * Math.PI);
}

/** The 1-based id of the innermost medium whose boundary contains `point`; 0 when it is inside none. */
export function ptDetectCameraMedium(point: readonly [number, number, number], boundaries: readonly PtMediumBoundary[]): number {
  let best = 0;
  let bestVolume = Infinity;
  for (const m of boundaries) {
    if (m.medium <= 0 || m.positions.length < 9) continue;
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < m.positions.length; i += 3) for (let k = 0; k < 3; k++) {
      const v = m.positions[i + k] as number;
      if (v < lo[k]!) lo[k] = v;
      if (v > hi[k]!) hi[k] = v;
    }
    if ([0, 1, 2].some((k) => point[k]! < lo[k]! || point[k]! > hi[k]!)) continue; // outside the bounding box: cheap reject
    if (ptWindingNumber(point, m.positions, m.indices) <= 0.5) continue;
    const volume = (hi[0]! - lo[0]!) * (hi[1]! - lo[1]!) * (hi[2]! - lo[2]!);
    if (volume <= bestVolume) { best = m.medium; bestVolume = volume; }
  }
  return best;
}

/**
 * V6.2: ALL media containing `point`, OUTERMOST first (largest bounding box first; a tie keeps the entry order), for `volumes.cameraStack`. The same winding-number test as
 * `ptDetectCameraMedium` (whose result is the last entry). The same medium id listed twice (two boundary meshes of one medium) appears once.
 */
export function ptDetectCameraMediumStack(point: readonly [number, number, number], boundaries: readonly PtMediumBoundary[]): number[] {
  const found: { medium: number; volume: number; order: number }[] = [];
  boundaries.forEach((m, order) => {
    if (m.medium <= 0 || m.positions.length < 9) return;
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < m.positions.length; i += 3) for (let k = 0; k < 3; k++) {
      const v = m.positions[i + k] as number;
      if (v < lo[k]!) lo[k] = v;
      if (v > hi[k]!) hi[k] = v;
    }
    if ([0, 1, 2].some((k) => point[k]! < lo[k]! || point[k]! > hi[k]!)) return;
    if (ptWindingNumber(point, m.positions, m.indices) <= 0.5) return;
    found.push({ medium: m.medium, volume: (hi[0]! - lo[0]!) * (hi[1]! - lo[1]!) * (hi[2]! - lo[2]!), order });
  });
  found.sort((a, b) => b.volume - a.volume || a.order - b.order);
  return [...new Set(found.map((f) => f.medium))];
}
