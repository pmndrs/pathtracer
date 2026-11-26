/**
 * Orbit camera for the examples: drag to orbit, shift-drag or right-drag to pan, wheel or pinch to zoom,
 * double-click or F to go back to the framed view. Produces the path tracer's
 * look-at camera (`PtLookAtCamera`); `onChange` fires on every move so a progressive renderer can restart its image.
 */
import type { PtLookAtCamera } from '@pmndrs/pathtracer';
import { boundsCenter, boundsRadius, clampDistance, clampTarget, fitDistance, pivotOnSubject, type Bounds, type V3 } from './orbit-math.ts';

export class OrbitCamera {
  target: V3;
  distance: number;
  /** Radians around +Y (0 looks down -Z). */
  yaw: number;
  /** Radians above the horizon. */
  pitch: number;
  yFovDeg: number;
  onChange: () => void = () => {};
  /** The subject's bounds when known (`frame`, or `set` / the constructor with bounds): zoom and pan limits scale with it. */
  private bounds: Bounds | null = null;
  private home: PtLookAtCamera;
  private readonly pointers = new Map<number, { x: number; y: number }>();
  private readonly element: HTMLElement;
  private readonly controller = new AbortController();

  constructor(element: HTMLElement, camera: PtLookAtCamera, bounds?: Bounds) {
    this.element = element;
    this.target = [0, 0, 0];
    this.distance = 1;
    this.yaw = 0;
    this.pitch = 0;
    this.yFovDeg = camera.yFovDeg;
    this.home = camera;
    this.set(camera, false, bounds);
    const opts = { signal: this.controller.signal };
    element.addEventListener('pointerdown', (e) => this.down(e), opts);
    element.addEventListener('pointermove', (e) => this.move(e), opts);
    element.addEventListener('pointerup', (e) => this.up(e), opts);
    element.addEventListener('pointercancel', (e) => this.up(e), opts);
    element.addEventListener('contextmenu', (e) => e.preventDefault(), opts);
    element.addEventListener('dblclick', () => this.reset(), opts);
    window.addEventListener('keydown', (e) => {
      const t = e.target as HTMLElement | null;
      if ((e.key === 'f' || e.key === 'F') && !e.metaKey && !e.ctrlKey && !e.altKey && !(t && /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName))) this.reset();
    }, opts);
    element.addEventListener('wheel', (e) => this.wheel(e), { ...opts, passive: false });
  }

  /**
   * Moves to a look-at camera (its `up` is taken as +y). With the subject's `bounds` the pivot is moved onto the subject (see
   * `pivotOnSubject`: a glTF camera's `target` is only a direction) and the zoom and pan limits follow its size.
   */
  set(camera: PtLookAtCamera, notify = true, bounds?: Bounds): void {
    this.bounds = bounds ?? null;
    if (bounds) camera = pivotOnSubject(camera, bounds);
    this.home = camera;
    this.target = [...camera.target] as V3;
    this.yFovDeg = camera.yFovDeg;
    const d: V3 = [camera.position[0] - camera.target[0], camera.position[1] - camera.target[1], camera.position[2] - camera.target[2]];
    this.distance = Math.hypot(...d) || 1;
    this.scale = this.distance;
    this.yaw = Math.atan2(d[0], d[2]);
    this.pitch = Math.asin(Math.max(-1, Math.min(1, d[1] / this.distance)));
    if (notify) this.onChange();
  }

  /** Frames a bounding box (keeps the viewing direction). */
  frame(min: readonly number[], max: readonly number[]): void {
    this.bounds = { min, max };
    this.target = boundsCenter(this.bounds);
    this.distance = fitDistance(boundsRadius(this.bounds), this.yFovDeg);
    this.home = this.camera;
    this.onChange();
  }

  /** Back to the view the camera was last set or framed to (double-click, F). */
  reset(): void {
    if (this.bounds) this.frame(this.bounds.min, this.bounds.max);
    else this.set(this.home);
  }

  /** The subject's size: its bounds radius, else the distance the camera was set at. */
  private get radius(): number {
    return this.bounds ? boundsRadius(this.bounds) : this.scale;
  }
  private scale = 1;

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
    const t = [0, 1, 2].map((i) => this.target[i]! - right[i]! * dx * scale + up[i]! * dy * scale) as V3;
    this.target = this.bounds ? clampTarget(t, boundsCenter(this.bounds), this.radius) : t;
  }

  private zoom(factor: number): void {
    this.distance = clampDistance(this.distance * factor, this.radius);
    this.onChange();
  }

  private wheel(e: WheelEvent): void {
    e.preventDefault();
    this.zoom(Math.exp(e.deltaY * (e.deltaMode === 1 ? 0.033 : 0.001)));
  }
}
