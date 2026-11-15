/**
 * Pinhole camera per the bench schema (`CameraDesc`): look-at, +Y up, right-handed (view space looks down -Z, +X right in the
 * image), `yFovDeg` is the VERTICAL field of view. Image row 0 = top.
 *
 * The kernel gets four vectors: `origin`, and `forward`, `right`, `up` with right and up already scaled by
 * `tan(yFov/2)` (and `right` by the aspect ratio), so a pixel's ray direction is
 * `normalize(forward + right * ndc.x + up * ndc.y)` with ndc in [-1, 1], +x right, +y up.
 */
export type CameraVec3 = readonly [number, number, number];

export interface PtLookAtCamera {
  position: CameraVec3;
  target: CameraVec3;
  up: CameraVec3;
  /** Vertical field of view in degrees. */
  yFovDeg: number;
}

export interface PtCameraBasis {
  origin: [number, number, number];
  forward: [number, number, number];
  /** Unit right vector times `tan(yFov / 2) * aspect`. */
  right: [number, number, number];
  /** Unit up vector times `tan(yFov / 2)`. */
  up: [number, number, number];
}

const sub = (a: CameraVec3, b: CameraVec3): [number, number, number] => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a: CameraVec3, b: CameraVec3): [number, number, number] => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const normalize = (a: CameraVec3): [number, number, number] => {
  const l = Math.hypot(a[0], a[1], a[2]);
  if (!(l > 0)) throw new RangeError('camera: zero-length vector (position equals target, or up is parallel to the view direction)');
  return [a[0] / l, a[1] / l, a[2] / l];
};

export function ptCameraBasis(camera: PtLookAtCamera, width: number, height: number): PtCameraBasis {
  const forward = normalize(sub(camera.target, camera.position));
  const right = normalize(cross(forward, camera.up));
  const up = cross(right, forward);
  const tanHalf = Math.tan((camera.yFovDeg * Math.PI) / 360);
  const aspect = width / height;
  return {
    origin: [camera.position[0], camera.position[1], camera.position[2]],
    forward,
    right: [right[0] * tanHalf * aspect, right[1] * tanHalf * aspect, right[2] * tanHalf * aspect],
    up: [up[0] * tanHalf, up[1] * tanHalf, up[2] * tanHalf],
  };
}

/** Ray direction for continuous pixel coordinates (x right, y down, in pixels; pixel (i, j) covers [i, i+1) x [j, j+1)). TS twin of the kernel. */
export function ptCameraDirection(basis: PtCameraBasis, width: number, height: number, px: number, py: number): [number, number, number] {
  const nx = (px / width) * 2 - 1;
  const ny = 1 - (py / height) * 2;
  return normalize([
    basis.forward[0] + basis.right[0] * nx + basis.up[0] * ny,
    basis.forward[1] + basis.right[1] * nx + basis.up[1] * ny,
    basis.forward[2] + basis.right[2] * nx + basis.up[2] * ny,
  ]);
}
