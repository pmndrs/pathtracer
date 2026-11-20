/**
 * Orbit camera for the examples: drag to orbit, shift-drag or right-drag to pan, wheel or pinch to zoom. Produces the path tracer's
 * look-at camera (`PtLookAtCamera`); `onChange` fires on every move so a progressive renderer can restart its image.
 */
import type { PtLookAtCamera } from '@pmndrs/pathtracer';

type V3 = [number, number, number];

export class OrbitCamera {
  target: V3;
  distance: number;
  /** Radians around +Y (0 looks down -Z). */
  yaw: number;
  /** Radians above the horizon. */
  pitch: number;
  yFovDeg: number;
  onChange: () => void = () => {};
  private readonly pointers = new Map<number, { x: number; y: number }>();
  private readonly element: HTMLElement;
  private readonly controller = new AbortController();

  constructor(element: HTMLElement, camera: PtLookAtCamera) {
    this.element = element;
    this.target = [0, 0, 0];
    this.distance = 1;
    this.yaw = 0;
    this.pitch = 0;
    this.yFovDeg = camera.yFovDeg;
    this.set(camera, false);
    const opts = { signal: this.controller.signal };
    element.addEventListener('pointerdown', (e) => this.down(e), opts);
    element.addEventListener('pointermove', (e) => this.move(e), opts);
    element.addEventListener('pointerup', (e) => this.up(e), opts);
    element.addEventListener('pointercancel', (e) => this.up(e), opts);
    element.addEventListener('contextmenu', (e) => e.preventDefault(), opts);
    element.addEventListener('wheel', (e) => this.wheel(e), { ...opts, passive: false });
  }

  /** Moves to a look-at camera (its `up` is taken as +y). */
  set(camera: PtLookAtCamera, notify = true): void {
    this.target = [...camera.target] as V3;
    this.yFovDeg = camera.yFovDeg;
    const d: V3 = [camera.position[0] - camera.target[0], camera.position[1] - camera.target[1], camera.position[2] - camera.target[2]];
    this.distance = Math.hypot(...d) || 1;
    this.yaw = Math.atan2(d[0], d[2]);
    this.pitch = Math.asin(Math.max(-1, Math.min(1, d[1] / this.distance)));
    if (notify) this.onChange();
  }

  /** Frames a bounding box (keeps the viewing direction). */
  frame(min: readonly number[], max: readonly number[]): void {
    this.target = [(min[0]! + max[0]!) / 2, (min[1]! + max[1]!) / 2, (min[2]! + max[2]!) / 2];
    const radius = Math.hypot(max[0]! - min[0]!, max[1]! - min[1]!, max[2]! - min[2]!) / 2 || 1;
    this.distance = (radius / Math.sin((this.yFovDeg * Math.PI) / 360)) * 1.05;
    this.onChange();
  }

  get camera(): PtLookAtCamera {
    const c = Math.cos(this.pitch);
    const position: V3 = [this.target[0] + this.distance * c * Math.sin(this.yaw), this.target[1] + this.distance * Math.sin(this.pitch), this.target[2] + this.distance * c * Math.cos(this.yaw)];
    return { position, target: [...this.target] as V3, up: [0, 1, 0], yFovDeg: this.yFovDeg };
  }

  dispose(): void {
    this.controller.abort();
  }

  private down(e: PointerEvent): void {
    this.element.setPointerCapture(e.pointerId);
    this.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
  }

  private up(e: PointerEvent): void {
    this.pointers.delete(e.pointerId);
  }

  private move(e: PointerEvent): void {
    const last = this.pointers.get(e.pointerId);
    if (!last) return;
    const dx = e.clientX - last.x, dy = e.clientY - last.y;
    if (this.pointers.size === 2) {
      // Pinch: the change of distance between the two pointers zooms.
      const [a, b] = [...this.pointers.values()];
      const before = Math.hypot(a!.x - b!.x, a!.y - b!.y);
      this.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      const [c, d] = [...this.pointers.values()];
      const after = Math.hypot(c!.x - d!.x, c!.y - d!.y);
      if (before > 0 && after > 0) this.zoom(before / after);
      return;
    }
    this.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (e.shiftKey || e.buttons === 2 || e.buttons === 4) this.pan(dx, dy);
    else {
      const k = Math.PI / this.element.clientHeight;
      this.yaw -= dx * k;
      this.pitch = Math.max(-1.55, Math.min(1.55, this.pitch + dy * k));
    }
    this.onChange();
  }

  private pan(dx: number, dy: number): void {
    const scale = (2 * this.distance * Math.tan((this.yFovDeg * Math.PI) / 360)) / this.element.clientHeight;
    const right: V3 = [Math.cos(this.yaw), 0, -Math.sin(this.yaw)];
    const s = Math.sin(this.pitch), c = Math.cos(this.pitch);
    const up: V3 = [-s * Math.sin(this.yaw), c, -s * Math.cos(this.yaw)];
    for (let i = 0; i < 3; i++) this.target[i] = this.target[i]! - right[i]! * dx * scale + up[i]! * dy * scale;
  }

  private zoom(factor: number): void {
    this.distance = Math.max(1e-3, this.distance * factor);
    this.onChange();
  }

  private wheel(e: WheelEvent): void {
    e.preventDefault();
    this.zoom(Math.exp(e.deltaY * 0.001));
  }
}
