/** Small colour ramps shared by the CLI previews and the viewer. t in [0,1] -> [r,g,b] in [0,1] (display-encoded). */
export type ColormapName = 'magma' | 'turbo';

const hex = (s: string): [number, number, number] => [
  parseInt(s.slice(1, 3), 16) / 255, parseInt(s.slice(3, 5), 16) / 255, parseInt(s.slice(5, 7), 16) / 255,
];

// Samples of matplotlib's magma (FLIP's own map is magma) and Google's turbo at t = 0, 1/8, ... 1.
export const STOPS: Record<ColormapName, Array<[number, number, number]>> = {
  magma: ['#000004', '#221150', '#51127c', '#832681', '#b73779', '#e75263', '#fc8961', '#fec98d', '#fcfdbf'].map(hex),
  turbo: ['#30123b', '#4662d7', '#36aaf9', '#1ae4b6', '#72fe5e', '#c7ef34', '#faba39', '#f66b19', '#7a0403'].map(hex),
};

export function colormap(name: ColormapName, t: number): [number, number, number] {
  const s = STOPS[name];
  const x = (t < 0 ? 0 : t > 1 ? 1 : Number.isNaN(t) ? 0 : t) * (s.length - 1);
  const i = Math.min(Math.floor(x), s.length - 2);
  const f = x - i;
  const a = s[i]!, b = s[i + 1]!;
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

/** 256-entry RGB8 LUT (768 bytes) for texture upload. */
export function colormapLUT(name: ColormapName): Uint8Array {
  const out = new Uint8Array(256 * 3);
  for (let i = 0; i < 256; i++) {
    const c = colormap(name, i / 255);
    out[i * 3] = Math.round(c[0] * 255); out[i * 3 + 1] = Math.round(c[1] * 255); out[i * 3 + 2] = Math.round(c[2] * 255);
  }
  return out;
}

/** Relative-error false colour: sqrt(relMSE pixel) on a log scale, 1e-3 (0.1 %) .. 1 (100 %). */
export const REL_ERR_LOG_MIN = -3;
export const REL_ERR_LOG_MAX = 0;
export function relErrToT(e: number): number {
  if (!(e > 0)) return 0;
  return (Math.log10(e) - REL_ERR_LOG_MIN) / (REL_ERR_LOG_MAX - REL_ERR_LOG_MIN);
}
