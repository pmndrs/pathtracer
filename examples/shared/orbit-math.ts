/** Pure camera math behind `OrbitCamera` (no DOM, so it is unit tested). */
import type { PtLookAtCamera } from '@pmndrs/pathtracer';

export type V3 = [number, number, number];
export interface Bounds { min: readonly number[]; max: readonly number[] }

export function boundsCenter(b: Bounds): V3 {
  return [(b.min[0]! + b.max[0]!) / 2, (b.min[1]! + b.max[1]!) / 2, (b.min[2]! + b.max[2]!) / 2];
}

/** Half the diagonal of the box (never 0, so a flat or empty box still gives a usable scale). */
export function boundsRadius(b: Bounds): number {
  const r = Math.hypot(b.max[0]! - b.min[0]!, b.max[1]! - b.min[1]!, b.max[2]! - b.min[2]!) / 2;
  return Number.isFinite(r) && r > 0 ? r : 1;
}

/** Distance at which a sphere of `radius` fits the vertical field of view (with 5% margin). */
export function fitDistance(radius: number, yFovDeg: number): number {
  return (radius / Math.sin((yFovDeg * Math.PI) / 360)) * 1.05;
}

/**
 * Re-pivots a look-at camera on a subject. A glTF camera has no target, only a direction, and the reader puts `target` one unit in
 * front of it, which is nowhere near a model that is 0.1 m or 50 m across. Keeps the position and the viewing direction and moves the
 * target to the point of the view axis nearest the bounds centre (so the picture does not change and an orbit turns around the
 * model). When the centre is behind the camera, or on top of it, the target becomes the bounds centre instead.
 */
export function pivotOnSubject(camera: PtLookAtCamera, bounds: Bounds): PtLookAtCamera {
  const c = boundsCenter(bounds), p = camera.position;
  const d: V3 = [camera.target[0] - p[0], camera.target[1] - p[1], camera.target[2] - p[2]];
  const dl = Math.hypot(...d);
  const radius = boundsRadius(bounds);
  let target = c;
  if (dl > 0) {
    const dir: V3 = [d[0] / dl, d[1] / dl, d[2] / dl];
    const t = (c[0] - p[0]) * dir[0] + (c[1] - p[1]) * dir[1] + (c[2] - p[2]) * dir[2];
    if (t > radius * 0.05) target = [p[0] + dir[0] * t, p[1] + dir[1] * t, p[2] + dir[2] * t];
  }
  return { ...camera, target };
}

/** The zoom range for a subject of `radius`: close enough to see detail, far enough to see the context, never through the pivot. */
export function distanceRange(radius: number): { min: number; max: number } {
  return { min: radius * 0.02, max: radius * 50 };
}

export function clampDistance(distance: number, radius: number): number {
  const { min, max } = distanceRange(radius);
  return Math.max(min, Math.min(max, distance));
}

/** Keeps a pan target within 2 radii of the subject's centre, so a pan can always be undone. */
export function clampTarget(target: V3, center: V3, radius: number): V3 {
  const lim = radius * 2;
  return [0, 1, 2].map((i) => Math.max(center[i]! - lim, Math.min(center[i]! + lim, target[i]!))) as V3;
}
