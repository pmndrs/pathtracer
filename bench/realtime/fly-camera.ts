/**
 * Fly camera for the interactive real-time pages (nrc-pipeline, the Classroom showcase): drag to look, keys to move.
 *
 * Movement is integrated on a FIXED step (`FLY_STEP`) from real elapsed time, not once per rendered frame: a page calls `advance(seconds)` on
 * every animation frame, also while a render is still in flight, and renders with whatever pose the camera has when the GPU is free. So the
 * distance covered per second does not depend on the frame rate, and a slow frame shows a longer step instead of a slower camera. The
 * velocity eases toward the keys' target (`FLY_RESPONSE`) and snaps to exactly zero when it has decayed, so a released key leaves the pose
 * bit-identical from frame to frame (the pages restart their still-camera accumulation on any change of the camera).
 * No DOM and no GPU: unit tests import it (fly-camera.test.ts).
 */
import type { PtLookAtCamera } from '@pmndrs/pathtracer';

type V3 = [number, number, number];

/** Seconds per simulation step. */
export const FLY_STEP = 1 / 120;
/** Time constant (seconds) of the velocity's approach to its target: ~95% of full speed after 3 of these. */
export const FLY_RESPONSE = 0.07;
/** Real time one `advance` may add (a hidden tab or a stalled frame must not teleport the camera). */
export const FLY_MAX_BACKLOG = 0.25;
/** Below this fraction of `speed`, with no key held, the velocity is set to zero. */
const REST = 0.02;

/** Axes in [-1, 1] (opposite keys cancel); `boost` multiplies the speed by 3. */
export interface FlyInput {
  forward: number;
  right: number;
  up: number;
  boost?: boolean;
}

export const FLY_IDLE: FlyInput = { forward: 0, right: 0, up: 0 };

/** Key state -> axes. Uses `KeyboardEvent.code`, so WASD is the physical cluster on any layout; the arrows work too. */
export function flyInputFromCodes(codes: ReadonlySet<string>): FlyInput {
  const axis = (plus: string[], minus: string[]) => (plus.some((c) => codes.has(c)) ? 1 : 0) - (minus.some((c) => codes.has(c)) ? 1 : 0);
  return {
    forward: axis(['KeyW', 'ArrowUp'], ['KeyS', 'ArrowDown']),
    right: axis(['KeyD', 'ArrowRight'], ['KeyA', 'ArrowLeft']),
    up: axis(['KeyE'], ['KeyQ']),
    boost: codes.has('ShiftLeft') || codes.has('ShiftRight'),
  };
}

export class FlyCamera {
  position: V3 = [0, 0, 0];
  /** Radians; forward = (cos yaw, 0, sin yaw) at pitch 0. */
  yaw = 0;
  pitch = 0;
  yFovDeg = 60;
  /** Metres per second at full key. */
  speed = 1;
  velocity: V3 = [0, 0, 0];
  private backlog = 0;

  /** Takes the pose of a look-at camera (its `up` is taken as +y) and stops. */
  setPose(c: PtLookAtCamera): void {
    this.position = [...c.position] as V3;
    const d = [c.target[0] - c.position[0], c.target[1] - c.position[1], c.target[2] - c.position[2]];
    this.yaw = Math.atan2(d[2]!, d[0]!);
    this.pitch = Math.asin(Math.max(-1, Math.min(1, d[1]! / (Math.hypot(d[0]!, d[1]!, d[2]!) || 1))));
    this.yFovDeg = c.yFovDeg;
    this.stop();
  }

  stop(): void {
    this.velocity = [0, 0, 0];
    this.backlog = 0;
  }

  /** Pointer look, radians. */
  look(dYaw: number, dPitch: number): void {
    this.yaw += dYaw;
    this.pitch = Math.max(-1.5, Math.min(1.5, this.pitch + dPitch));
  }

  get moving(): boolean {
    return this.velocity.some((v) => v !== 0);
  }

  /** Adds `seconds` of real time and runs the whole steps it covers. Returns whether the position changed. */
  advance(seconds: number, input: FlyInput = FLY_IDLE): boolean {
    this.backlog = Math.min(FLY_MAX_BACKLOG, this.backlog + Math.max(0, seconds));
    let moved = false;
    while (this.backlog >= FLY_STEP) {
      this.backlog -= FLY_STEP;
      moved = this.step(input) || moved;
    }
    // Nothing to integrate: do not let idle time pile up into a jump on the next key press.
    if (!this.moving && input.forward === 0 && input.right === 0 && input.up === 0) this.backlog = 0;
    return moved;
  }

  private step(input: FlyInput): boolean {
    const f = [Math.cos(this.yaw), 0, Math.sin(this.yaw)];
    const r = [-Math.sin(this.yaw), 0, Math.cos(this.yaw)];
    const dir = [0, 1, 2].map((i) => f[i]! * input.forward + r[i]! * input.right + (i === 1 ? input.up : 0));
    const len = Math.hypot(dir[0]!, dir[1]!, dir[2]!);
    const idle = len === 0;
    // Normalised, so a diagonal is not faster than an axis.
    const top = idle ? 0 : (this.speed * (input.boost ? 3 : 1)) / len;
    const k = 1 - Math.exp(-FLY_STEP / FLY_RESPONSE);
    for (let i = 0; i < 3; i++) this.velocity[i] = this.velocity[i]! + (dir[i]! * top - this.velocity[i]!) * k;
    if (idle && Math.hypot(...this.velocity) < REST * this.speed) this.velocity = [0, 0, 0];
    if (!this.moving) return false;
    for (let i = 0; i < 3; i++) this.position[i] = this.position[i]! + this.velocity[i]! * FLY_STEP;
    return true;
  }

  get camera(): PtLookAtCamera {
    const d = [Math.cos(this.pitch) * Math.cos(this.yaw), Math.sin(this.pitch), Math.cos(this.pitch) * Math.sin(this.yaw)];
    const p = this.position;
    return { position: [...p], target: [p[0] + d[0]!, p[1] + d[1]!, p[2] + d[2]!], up: [0, 1, 0], yFovDeg: this.yFovDeg };
  }
}
