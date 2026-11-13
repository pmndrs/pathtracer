/**
 * Display transforms shared by metrics (PSNR, FLIP), PNG previews and the viewer (the viewer's GLSL repeats these
 * constants; keep them in sync). All map linear scene radiance (after an EV gain) to display-LINEAR [0,1]; apply
 * `srgbEncode` for 8-bit output.
 *
 *   clamp    min(x, 1)
 *   reinhard x / (1 + x), per channel
 *   aces     Narkowicz's ACES fit, per channel, no pre-exposure (exposure 0 EV maps 1.0 -> ~0.80)
 *   agx      the "minimal AgX" approximation (iolite / three.js: inset matrix, log2 encode, 6th-order contrast
 *            polynomial, outset matrix, gamma 2.2 undone), not the full OCIO AgX with looks
 */
import type { Image } from '../src/adapter.ts';

export type TonemapName = 'clamp' | 'reinhard' | 'aces' | 'agx';
export const TONEMAPS: readonly TonemapName[] = ['clamp', 'reinhard', 'aces', 'agx'];

const sat = (x: number) => (x < 0 ? 0 : x > 1 ? 1 : x);

const SRGB_TO_REC2020 = [0.6274, 0.3293, 0.0433, 0.0691, 0.9195, 0.0113, 0.0164, 0.088, 0.8956]; // row-major
const REC2020_TO_SRGB = [1.6605, -0.5876, -0.0728, -0.1246, 1.1329, -0.0083, -0.0182, -0.1006, 1.1187];
// Row-major; both rows sum to 1 so neutrals stay neutral through the whole curve.
export const AGX_INSET = [
  0.856627153315983, 0.0951212405381588, 0.0482516061458583,
  0.137318972929847, 0.761241990602591, 0.101439036467562,
  0.11189821299995, 0.0767994186031903, 0.811302368396859,
];
export const AGX_OUTSET = [
  1.1271005818144368, -0.1106066430966032, -0.016493938717834573,
  -0.14132976349843826, 1.157823702216272, -0.016493938717834573,
  -0.14132976349843826, -0.11060664309660323, 1.2519364065950405,
];
export const AGX_MIN_EV = -12.47393;
export const AGX_MAX_EV = 4.026069;
export const SRGB_TO_REC2020_MATRIX = SRGB_TO_REC2020;
export const REC2020_TO_SRGB_MATRIX = REC2020_TO_SRGB;

function mul3(m: number[], r: number, g: number, b: number, o: number[]) {
  o[0] = m[0]! * r + m[1]! * g + m[2]! * b;
  o[1] = m[3]! * r + m[4]! * g + m[5]! * b;
  o[2] = m[6]! * r + m[7]! * g + m[8]! * b;
}

const tmpA = [0, 0, 0];
const tmpB = [0, 0, 0];
const contrast = (x: number) => {
  const x2 = x * x, x4 = x2 * x2;
  return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
};

/** Tonemap one pixel (already multiplied by exposure). Writes display-linear [0,1] to out[o..o+2]. */
export function tonemapPixel(name: TonemapName, r: number, g: number, b: number, out: Float32Array | number[], o = 0): void {
  switch (name) {
    case 'clamp': out[o] = sat(r); out[o + 1] = sat(g); out[o + 2] = sat(b); return;
    case 'reinhard': {
      out[o] = r / (1 + r); out[o + 1] = g / (1 + g); out[o + 2] = b / (1 + b); return;
    }
    case 'aces': {
      const f = (x: number) => sat((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14));
      out[o] = f(r); out[o + 1] = f(g); out[o + 2] = f(b); return;
    }
    case 'agx': {
      mul3(SRGB_TO_REC2020, r, g, b, tmpA);
      mul3(AGX_INSET, tmpA[0]!, tmpA[1]!, tmpA[2]!, tmpB);
      for (let i = 0; i < 3; i++) {
        const v = Math.log2(Math.max(tmpB[i]!, 1e-10));
        tmpB[i] = contrast(sat((v - AGX_MIN_EV) / (AGX_MAX_EV - AGX_MIN_EV)));
      }
      mul3(AGX_OUTSET, tmpB[0]!, tmpB[1]!, tmpB[2]!, tmpA);
      for (let i = 0; i < 3; i++) tmpA[i] = Math.pow(Math.max(tmpA[i]!, 0), 2.2);
      mul3(REC2020_TO_SRGB, tmpA[0]!, tmpA[1]!, tmpA[2]!, tmpB);
      out[o] = sat(tmpB[0]!); out[o + 1] = sat(tmpB[1]!); out[o + 2] = sat(tmpB[2]!);
      return;
    }
  }
}

export function srgbEncode(v: number): number {
  const c = sat(v);
  return c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
}

/** Tonemap a whole image to display-linear RGB (3 floats/px, alpha dropped). NaN/Inf become 0. */
export function tonemapImage(img: Image, name: TonemapName = 'aces', ev = 0): Float32Array {
  const n = img.width * img.height;
  const out = new Float32Array(n * 3);
  const gain = Math.pow(2, ev);
  const ch = img.channels;
  const d = img.data;
  for (let i = 0; i < n; i++) {
    let r = d[i * ch]! * gain, g = d[i * ch + 1]! * gain, b = d[i * ch + 2]! * gain;
    if (!Number.isFinite(r)) r = 0;
    if (!Number.isFinite(g)) g = 0;
    if (!Number.isFinite(b)) b = 0;
    tonemapPixel(name, r, g, b, out, i * 3);
  }
  return out;
}
