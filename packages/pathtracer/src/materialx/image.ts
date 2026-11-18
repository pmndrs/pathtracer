/**
 * Image lookups for the TypeScript evaluator (P5): the CPU twin of what the GPU does with a scene texture (`rtTexSample` at level 0):
 * bilinear filtering with texel centres at half-integers, sRGB decoded before filtering, row 0 = uv v 0 (MaterialX and glTF agree: the
 * origin is the top-left texel and v grows downwards; V263).
 */
import type { MtlxImageRequest, MtlxTsContext, MtlxValue } from './nodes.ts';

export interface MtlxImage {
  width: number;
  height: number;
  /** RGBA8, `width * height * 4`, row 0 first. */
  data: Uint8Array;
}

const SRGB = (() => {
  const t = new Float64Array(256);
  for (let i = 0; i < 256; i++) {
    const c = i / 255;
    t[i] = c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
  }
  return t;
})();

/** Texel index under a MaterialX address mode (`periodic`, `clamp`, `mirror`; `constant` reads as clamp, V263). */
export function addressIndex(i: number, n: number, mode: string): number {
  if (mode === 'clamp' || mode === 'constant') return Math.min(n - 1, Math.max(0, i));
  if (mode === 'mirror') {
    const period = 2 * n;
    const m = ((i % period) + period) % period;
    return m < n ? m : period - 1 - m;
  }
  return ((i % n) + n) % n;
}

/** Bilinear lookup of `image`; colour channels linear (sRGB decoded when `srgb`), alpha 0..1. */
export function sampleImage(image: MtlxImage, srgb: boolean, uMode: string, vMode: string, u: number, v: number): MtlxValue {
  const x = u * image.width - 0.5;
  const y = v * image.height - 0.5;
  const x0 = Math.floor(x), y0 = Math.floor(y);
  const fx = x - x0, fy = y - y0;
  const out: MtlxValue = [0, 0, 0, 0];
  for (let j = 0; j < 2; j++) {
    for (let i = 0; i < 2; i++) {
      const w = (i === 0 ? 1 - fx : fx) * (j === 0 ? 1 - fy : fy);
      if (w === 0) continue;
      const px = addressIndex(x0 + i, image.width, uMode);
      const py = addressIndex(y0 + j, image.height, vMode);
      const o = (py * image.width + px) * 4;
      for (let c = 0; c < 3; c++) out[c]! += w * (srgb ? SRGB[image.data[o + c]!]! : image.data[o + c]! / 255);
      out[3]! += (w * image.data[o + 3]!) / 255;
    }
  }
  return out;
}

/** A `MtlxTsContext.sample` over a loader: `file` (prefix applied) -> image; a missing file reads as undefined (the node's default). */
export function imageSampler(load: (file: string) => MtlxImage | undefined): MtlxTsContext['sample'] {
  const cache = new Map<string, MtlxImage | undefined>();
  return (request: MtlxImageRequest, u: number, v: number) => {
    let img = cache.get(request.file);
    if (!cache.has(request.file)) {
      img = load(request.file);
      cache.set(request.file, img);
    }
    return img ? sampleImage(img, request.srgb, request.uaddress, request.vaddress, u, v) : undefined;
  };
}
