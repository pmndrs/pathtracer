/**
 * Texture files for the native exporters' `openpbr` materials (pbrt-v4, Mitsuba 3; P2.24).
 *
 * The decoded RGBA8 texels of a scene texture (`generateTexture`: PNG/JPEG of a glTF, or a procedural generator) are written as:
 *  - the image itself, an 8-bit PNG (`encodePng`), for colour textures that the renderer decodes from sRGB (base colour, pbrt `imagemap`
 *    `encoding sRGB`, Mitsuba `bitmap` `raw false`);
 *  - one 8-bit grey PNG per data channel (pbrt reads a float texture from channel 0 of the file, so roughness G and metalness B each get a
 *    file), value = the original byte, `encoding linear`;
 *  - float PFMs for values the renderers cannot derive from a byte (alpha = roughness^2, k of a conductor, linear emitted radiance).
 *
 * Every file name is `tex-<stem>-<role>.<ext>` and is written once per export. uv origin and v flip are the PLY's business (ply.ts).
 */
import { writePFM } from '../metrics/pfm.ts';
import type { SceneDesc } from '../scenes/schema.ts';
import { generateTexture } from '../scenes/textures.ts';
import type { GeneratedTexture } from '../scenes/textures.ts';
import { encodePng } from './png.ts';
import type { ExportedFile } from './types.ts';

const stem = (k: string): string => k.replace(/[^A-Za-z0-9_-]+/g, '_');

/** sRGB code (0-255) to linear. */
export const srgbToLinear = (b: number): number => {
  const c = b / 255;
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
};

/**
 * The distinct values of channel `c`, most frequent first, that cover at least `1 - tail` of the texels (the rest are antialiasing or
 * resampling stragglers and map to the nearest kept value), sorted ascending. Used where a renderer takes a constant where we have a texture.
 */
export function channelLevels(t: { width: number; height: number; data: Uint8Array }, c: 0 | 1 | 2 | 3, tail = 0.002): number[] {
  const count = new Map<number, number>();
  for (let i = 0; i < t.width * t.height; i++) count.set(t.data[i * 4 + c]!, (count.get(t.data[i * 4 + c]!) ?? 0) + 1);
  const byFreq = [...count.entries()].sort((a, b) => b[1] - a[1]);
  const kept: number[] = [];
  let covered = 0;
  for (const [v, n] of byFreq) {
    if (covered >= (1 - tail) * t.width * t.height) break;
    kept.push(v);
    covered += n;
  }
  return kept.sort((a, b) => a - b);
}

export interface NativeTexture extends GeneratedTexture {
  key: string;
  stem: string;
}

/** The textures one native export writes: resolves scene texture keys, de-duplicates files, and keeps them for the exporter's `files`. */
export class TextureSet {
  readonly files = new Map<string, ExportedFile>();
  private readonly cache = new Map<string, NativeTexture>();
  private readonly scene: SceneDesc;
  constructor(scene: SceneDesc) {
    this.scene = scene;
  }

  /** Scene texture `key`, decoded once. */
  get(key: string): NativeTexture {
    let t = this.cache.get(key);
    if (!t) {
      const desc = this.scene.textures?.[key];
      if (!desc) throw new Error(`unknown texture '${key}'`);
      if (desc.wrap === 'mirror') throw new Error(`texture '${key}': wrap 'mirror' does not exist in pbrt-v4 (repeat, clamp), so no native export is written`);
      t = { ...generateTexture(desc), key, stem: stem(key) };
      this.cache.set(key, t);
    }
    return t;
  }

  private put(path: string, make: () => Buffer | Uint8Array): string {
    if (!this.files.has(path)) this.files.set(path, { path, data: Buffer.from(make()) });
    return path;
  }

  /** The texture as an 8-bit RGB PNG (colour). Alpha is dropped: it travels as its own channel file (`channel`, `maskPng`), so no renderer premultiplies or composites it. */
  image(t: NativeTexture): string {
    return this.put(`tex-${t.stem}.png`, () => {
      const rgb = new Uint8Array(t.data);
      for (let i = 3; i < rgb.length; i += 4) rgb[i] = 255;
      return encodePng(t.width, t.height, rgb);
    });
  }

  /** One channel (0 R, 1 G, 2 B, 3 A) as a grey 8-bit PNG; the byte is unchanged. */
  channel(t: NativeTexture, c: 0 | 1 | 2 | 3): string {
    return this.put(`tex-${t.stem}-${'rgba'[c]}.png`, () => {
      const out = new Uint8Array(t.width * t.height * 4);
      for (let i = 0; i < t.width * t.height; i++) out.fill(t.data[i * 4 + c]!, i * 4, i * 4 + 3), (out[i * 4 + 3] = 255);
      return encodePng(t.width, t.height, out);
    });
  }

  /** A float PFM (grey, three equal channels) of `fn(texel rgba bytes)`, named `role`. */
  plane(t: NativeTexture, role: string, fn: (r: number, g: number, b: number, a: number) => number): string {
    return this.put(`tex-${t.stem}-${role}.pfm`, () => {
      const data = new Float32Array(t.width * t.height * 3);
      for (let i = 0; i < t.width * t.height; i++) data.fill(fn(t.data[i * 4]!, t.data[i * 4 + 1]!, t.data[i * 4 + 2]!, t.data[i * 4 + 3]!), i * 3, i * 3 + 3);
      return writePFM({ width: t.width, height: t.height, channels: 3, data });
    });
  }

  /** A 0/255 grey PNG of `fn(texel rgba bytes)` (read `raw`, so bilinear filtering gives exact weights in [0, 1]). */
  indicator(t: NativeTexture, role: string, fn: (r: number, g: number, b: number, a: number) => boolean): string {
    return this.put(`tex-${t.stem}-${role}.png`, () => {
      const out = new Uint8Array(t.width * t.height * 4);
      for (let i = 0; i < t.width * t.height; i++) out.fill(fn(t.data[i * 4]!, t.data[i * 4 + 1]!, t.data[i * 4 + 2]!, t.data[i * 4 + 3]!) ? 255 : 0, i * 4, i * 4 + 3), (out[i * 4 + 3] = 255);
      return encodePng(t.width, t.height, out);
    });
  }

  /** A float RGB PFM, `fn` returning the three channels of a texel. */
  rgb(t: NativeTexture, role: string, fn: (r: number, g: number, b: number, a: number) => readonly [number, number, number]): string {
    return this.put(`tex-${t.stem}-${role}.pfm`, () => {
      const data = new Float32Array(t.width * t.height * 3);
      for (let i = 0; i < t.width * t.height; i++) data.set(fn(t.data[i * 4]!, t.data[i * 4 + 1]!, t.data[i * 4 + 2]!, t.data[i * 4 + 3]!), i * 3);
      return writePFM({ width: t.width, height: t.height, channels: 3, data });
    });
  }

  /**
   * Hard-cutoff opacity for Mitsuba's `mask` BSDF (which has no threshold, only a stochastic opacity): `opacity * alpha >= cutoff` evaluated on an
   * `f` x finer grid with the same bilinear interpolation the renderer's lookup would use (wrap as the texture's), written as a 0/255 PNG to be read
   * with `filter_type nearest`. The contour is exact to 1 / (2 f) texel.
   */
  maskPng(t: NativeTexture, opacity: number, cutoff: number): { file: string; factor: number } {
    const factor = Math.max(1, Math.min(8, Math.floor(2048 / Math.max(t.width, t.height))));
    const w = t.width * factor, h = t.height * factor;
    const file = this.put(`tex-${t.stem}-mask-${String(opacity).replace(/\./g, 'p')}-${String(cutoff).replace(/\./g, 'p')}.png`, () => {
      const at = (x: number, y: number): number => {
        const wrapi = (v: number, n: number): number => (t.wrap === 'repeat' ? ((v % n) + n) % n : Math.min(n - 1, Math.max(0, v)));
        return t.data[(wrapi(y, t.height) * t.width + wrapi(x, t.width)) * 4 + 3]! / 255;
      };
      const out = new Uint8Array(w * h * 4);
      for (let j = 0; j < h; j++) {
        const sy = (j + 0.5) / factor - 0.5, y0 = Math.floor(sy), fy = sy - y0;
        for (let i = 0; i < w; i++) {
          const sx = (i + 0.5) / factor - 0.5, x0 = Math.floor(sx), fx = sx - x0;
          const a = (at(x0, y0) * (1 - fx) + at(x0 + 1, y0) * fx) * (1 - fy) + (at(x0, y0 + 1) * (1 - fx) + at(x0 + 1, y0 + 1) * fx) * fy;
          const v = opacity * a >= cutoff ? 255 : 0;
          out.set([v, v, v, 255], (j * w + i) * 4);
        }
      }
      return encodePng(w, h, out);
    });
    return { file, factor };
  }
}
