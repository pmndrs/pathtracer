/**
 * Synthetic images for the MaterialX bench scenes (P5): a `file` of the form `gen:<kind><n>` reads one of these, so the scenes need no
 * downloads and run unchanged in a page. RGBA8, row 0 = top (uv v 0). Pure code, deterministic.
 *
 *  - `gen:checker<N>`: 128 x 128, N x N cells of (235, 200, 60) and (60, 90, 170), the top-left cell red (orientation marker). sRGB codes.
 *  - `gen:bumps<P>`: 128 x 128 tangent-space normal map, a sine bump pattern P periods across (the `bumps` generator of `textures.ts`), linear.
 *  - `gen:ramp`: 128 x 128 horizontal ramp, value x / 127 in R, G and B, linear data.
 *  - `gen:rings`: 128 x 128 concentric rings (data in [0.15, 0.95]), linear.
 */
import type { MtlxImage } from '@pmndrs/pathtracer';

const SIZE = 128;

const make = (fn: (x: number, y: number) => [number, number, number]): MtlxImage => {
  const data = new Uint8Array(SIZE * SIZE * 4);
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const [r, g, b] = fn(x, y);
      data.set([r, g, b, 255], (y * SIZE + x) * 4);
    }
  }
  return { width: SIZE, height: SIZE, data };
};
const byte = (v: number): number => Math.min(255, Math.max(0, Math.round(v * 255)));

export function syntheticImage(file: string): MtlxImage | undefined {
  const m = /^gen:([a-z]+)(\d*)$/.exec(file);
  if (!m) return undefined;
  const n = m[2] ? Number(m[2]) : 0;
  switch (m[1]) {
    case 'checker':
      return make((x, y) => {
        const cx = Math.floor((x * n) / SIZE), cy = Math.floor((y * n) / SIZE);
        if (cx === 0 && cy === 0) return [230, 40, 40];
        return (cx + cy) % 2 === 0 ? [235, 200, 60] : [60, 90, 170];
      });
    case 'bumps':
      return make((x, y) => {
        const u = (x + 0.5) / SIZE, v = (y + 0.5) / SIZE, w = 2 * Math.PI * n, amp = 0.6;
        const nx = -amp * Math.cos(w * u) * Math.sin(w * v), ny = amp * Math.sin(w * u) * Math.cos(w * v);
        const l = Math.hypot(nx, ny, 1);
        return [byte(0.5 + (0.5 * nx) / l), byte(0.5 + (0.5 * ny) / l), byte(0.5 + 0.5 / l)];
      });
    case 'ramp':
      return make((x) => {
        const c = Math.round((x * 255) / (SIZE - 1));
        return [c, c, c];
      });
    case 'rings':
      return make((x, y) => {
        const r = Math.hypot((x + 0.5) / SIZE - 0.5, (y + 0.5) / SIZE - 0.5);
        const c = byte(0.55 + 0.4 * Math.sin(r * 2 * Math.PI * 5));
        return [c, c, c];
      });
    default:
      return undefined;
  }
}
