/**
 * The blue-noise textures the path tracer's `bluenoise` sampler needs, inside the package: `loadBlueNoise()` decodes one without
 * fetching anything, so real-time mode needs no hosted files. Each texture is its own module (base64 16-bit PNG, 27 KB / 105 KB),
 * loaded by dynamic import, so a bundle only carries the one it asks for. Other textures (`vc2d-*`, `st-*`) are in the repository's
 * `assets/bluenoise/` folder: serve them and use `fetchBlueNoise`.
 */
import { decodeBlueNoisePng, type BlueNoiseTexture } from './bluenoise.ts';

/** The textures shipped in the package: two-channel, one slice, power-of-two sides (what the `bluenoise` sampler takes). */
export type BuiltinBlueNoise = 'gf-vec2-64' | 'gf-vec2-128';

const cache = new Map<BuiltinBlueNoise, Promise<BlueNoiseTexture>>();

function base64Bytes(s: string): ArrayBuffer {
  const bin = atob(s);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out.buffer;
}

/** Decodes a blue-noise texture shipped in the package. Default `gf-vec2-64`, the one real-time mode uses. Cached per name. */
export function loadBlueNoise(name: BuiltinBlueNoise = 'gf-vec2-64'): Promise<BlueNoiseTexture> {
  let p = cache.get(name);
  if (!p) {
    const mod = name === 'gf-vec2-128' ? import('./builtin/gf-vec2-128.ts') : name === 'gf-vec2-64' ? import('./builtin/gf-vec2-64.ts') : null;
    if (!mod) return Promise.reject(new Error(`loadBlueNoise: '${name}' is not shipped in the package (gf-vec2-64, gf-vec2-128); use fetchBlueNoise`));
    p = mod.then((m) => decodeBlueNoisePng(base64Bytes(m.png), m.meta));
    cache.set(name, p);
    p.catch(() => cache.delete(name));
  }
  return p;
}
