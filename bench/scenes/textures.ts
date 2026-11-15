/**
 * Procedural textures for bench scenes (`SceneDesc.textures`): generated in code, so nothing is downloaded and every renderer reads the same texels.
 * RGBA8, row 0 = top (glTF), values exactly as written to PNG for pbrt-v4 / Mitsuba (`bench/exporters/png.ts`).
 */
import type { TextureDesc } from './schema.ts';

export interface GeneratedTexture {
  width: number;
  height: number;
  /** RGBA8, `width * height * 4`. */
  data: Uint8Array;
  colorSpace: 'srgb' | 'linear';
  wrap: 'repeat' | 'clamp' | 'mirror';
  filter?: 'linear' | 'nearest';
}

/** Decodes a `generator: 'image'` texture (Node, `gltf-images.ts` registers it): RGBA8, row 0 = top. */
export type ImageTextureLoader = (desc: Extract<TextureDesc, { generator: 'image' }>) => { width: number; height: number; data: Uint8Array };
let imageLoader: ImageTextureLoader | undefined;
export function setImageTextureLoader(loader: ImageTextureLoader | undefined): void {
  imageLoader = loader;
}

const byte = (v: number): number => Math.min(255, Math.max(0, Math.round(v * 255)));

export function generateTexture(desc: TextureDesc): GeneratedTexture {
  const { width, height } = desc;
  if (desc.generator === 'image') {
    if (!imageLoader) throw new Error(`texture from '${desc.path}' image ${desc.image}: import bench/scenes/gltf-images.ts (Node) to decode image textures`);
    const img = imageLoader(desc);
    if (img.width !== width || img.height !== height) throw new RangeError(`texture '${desc.path}' image ${desc.image}: decoded ${img.width}x${img.height}, scene says ${width}x${height}`);
    return { width, height, data: img.data, colorSpace: desc.colorSpace, wrap: desc.wrap ?? 'repeat', ...(desc.filter ? { filter: desc.filter } : {}) };
  }
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1) throw new RangeError(`texture: bad size ${width}x${height}`);
  const data = new Uint8Array(width * height * 4);
  const put = (x: number, y: number, r: number, g: number, b: number, a = 255) => data.set([r, g, b, a], (y * width + x) * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      switch (desc.generator) {
        case 'checker': {
          const cx = Math.floor((x * desc.cells) / width);
          const cy = Math.floor((y * desc.cells) / height);
          const c = desc.marker && cx === 0 && cy === 0 ? desc.marker : (cx + cy) % 2 === 0 ? desc.a : desc.b;
          const isMarker = !!desc.marker && cx === 0 && cy === 0;
          put(x, y, c[0], c[1], c[2], isMarker ? 255 : (cx + cy) % 2 === 0 ? (desc.alphaA ?? 255) : (desc.alphaB ?? 255));
          break;
        }
        case 'solid':
          put(x, y, desc.color[0], desc.color[1], desc.color[2], desc.color[3] ?? 255);
          break;
        case 'bumps': {
          // Height h(u, v) = A / (2 pi P) * sin(2 pi P u) * sin(2 pi P v)-ish: use the analytic gradient of h = sin(2 pi P u) sin(2 pi P v) scaled so the slope peaks at `amplitude`.
          const u = (x + 0.5) / width;
          const v = (y + 0.5) / height;
          const w = 2 * Math.PI * desc.periods;
          const dhdu = Math.cos(w * u) * Math.sin(w * v);
          const dhdv = Math.sin(w * u) * Math.cos(w * v);
          // Tangent space +X = +u, +Y = image up (decreasing v): n = normalize(-A dh/du, +A dh/dv, 1).
          const nx = -desc.amplitude * dhdu;
          const ny = desc.amplitude * dhdv;
          const il = 1 / Math.hypot(nx, ny, 1);
          put(x, y, byte(0.5 + 0.5 * nx * il), byte(0.5 + 0.5 * ny * il), byte(0.5 + 0.5 / Math.hypot(nx, ny, 1)));
          break;
        }
        case 'stripes-mr': {
          const stripe = Math.floor((x * desc.periods * 2) / width) % 2 === 0 ? desc.a : desc.b;
          put(x, y, 0, byte(stripe[0]), byte(stripe[1]));
          break;
        }
      }
    }
  }
  return { width, height, data, colorSpace: desc.colorSpace, wrap: desc.wrap ?? 'repeat' };
}
