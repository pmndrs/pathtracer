/** Environment loading for the web adapters: uniform radiance or the PFM equirect the scene points at. */
import { readPFM } from '../../metrics/pfm.ts';
import type { SceneDesc } from '../../scenes/schema.ts';

export type LoadedEnvironment =
  | { type: 'uniform'; radiance: readonly [number, number, number] }
  | {
      type: 'hdri';
      width: number;
      height: number;
      /** RGBA float32 in the scene's own layout: row 0 = +Y (zenith), u = 0 at -Z increasing toward +X, scale applied. */
      rgba: Float32Array;
      rotationYDeg: number;
    };

export async function loadEnvironment(scene: SceneDesc, assetsUrl: string): Promise<LoadedEnvironment | null> {
  const env = scene.environment;
  if (!env) return null;
  if (env.type === 'uniform') return { type: 'uniform', radiance: env.radiance };
  const response = await fetch(`${assetsUrl.replace(/\/$/, '')}/${env.path}`);
  if (!response.ok) throw new Error(`environment ${env.path}: HTTP ${response.status}. Run: node bench/scenes/tier0/generate-assets.ts`);
  const image = readPFM(await response.arrayBuffer());
  const rgba = new Float32Array(image.width * image.height * 4);
  for (let i = 0; i < image.width * image.height; i++) {
    for (let k = 0; k < 3; k++) rgba[i * 4 + k] = image.data[i * image.channels + k]! * env.scale;
    rgba[i * 4 + 3] = 1;
  }
  return { type: 'hdri', width: image.width, height: image.height, rgba, rotationYDeg: env.rotationYDeg };
}

/**
 * Re-lays the equirect for three.js's mapping, which both three-gpu-pathtracer and mutracer follow:
 *  - three's `DataTexture` with flipY = false has its first row at the BOTTOM (v = 0), ours is the zenith: rows reversed;
 *  - three's u = atan2(z, x) / 2pi + 0.5 puts +X at 0.5 and -Z at 0.25, ours puts +X at 0.25 and -Z at 0: columns
 *    shifted by a quarter turn (exact for widths divisible by 4).
 * `rotationYDeg` is left to the caller (an Euler rotation on the scene).
 */
export function equirectForThree(env: Extract<LoadedEnvironment, { type: 'hdri' }>): Float32Array {
  const { width, height, rgba } = env;
  if (width % 4 !== 0) throw new Error(`equirect width ${width} is not divisible by 4`);
  const shift = width / 4;
  const out = new Float32Array(rgba.length);
  for (let y = 0; y < height; y++) {
    const srcRow = height - 1 - y;
    for (let x = 0; x < width; x++) {
      const srcX = (x - shift + width) % width;
      out.set(rgba.subarray((srcRow * width + srcX) * 4, (srcRow * width + srcX) * 4 + 4), (y * width + x) * 4);
    }
  }
  return out;
}
