import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { decodeBlueNoisePng, type BlueNoiseMeta } from './bluenoise.ts';
import { loadBlueNoise } from './bluenoise-builtin.ts';

const dir = new URL('../../assets/bluenoise/', import.meta.url);
const manifest = JSON.parse(readFileSync(new URL('manifest.json', dir), 'utf8')) as { textures: BlueNoiseMeta[] };

describe('loadBlueNoise', () => {
  for (const name of ['gf-vec2-64', 'gf-vec2-128'] as const) {
    it(`${name} equals the asset file`, async () => {
      const meta = manifest.textures.find((t) => t.name === name)!;
      const png = readFileSync(new URL(meta.png, dir));
      const want = await decodeBlueNoisePng(png, meta);
      const got = await loadBlueNoise(name);
      expect(got.meta).toEqual(meta);
      expect(got.data).toEqual(want.data);
    });
  }
  it('defaults to gf-vec2-64 and refuses other names', async () => {
    expect((await loadBlueNoise()).meta.name).toBe('gf-vec2-64');
    await expect(loadBlueNoise('st-vec2-64x32' as never)).rejects.toThrow(/not shipped/);
  });
});
