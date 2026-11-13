import { describe, expect, it } from 'vitest';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { toGltf, toMitsuba, toPbrt } from './index.ts';
import { equalAreaSquareToSphere, equirectToOctahedral, sampleEquirect } from './envmap.ts';
import { shorterAxisFovDeg, xFovDeg } from './common.ts';
import { generateSkyImage, ENV_WIDTH, ENV_HEIGHT, SUN_DIR, skyRadiance } from '../scenes/tier0/env-sky.ts';
import { tier0Scenes, getTier0Scene } from '../scenes/tier0/index.ts';

const assetsDir = join(fileURLToPath(import.meta.url), '..', '..', 'assets');
const opts = { assetsDir, spp: 16 };
const text = (f: { data: Buffer | string }) => f.data.toString();

describe('pbrt exporter', () => {
  const scene = getTier0Scene('furnace-conductor');
  const r = toPbrt(scene, opts);
  const pbrt = text(r.files.find((f) => f.path === 'scene.pbrt')!);

  it('flips handedness, uses box filter 0.5, alpha = roughness² with remaproughness false, maxdepth verbatim', () => {
    expect(pbrt).toMatch(/^Scale -1 1 1$/m);
    expect(pbrt).toContain('PixelFilter "box" "float xradius" 0.5 "float yradius" 0.5');
    expect(pbrt).toContain('"float roughness" 0.48999999999999994');
    expect(pbrt).toContain('"bool remaproughness" false');
    expect(pbrt).toContain('"integer maxdepth" 16 "bool rr" false');
    expect(pbrt).toContain('LightSource "infinite" "rgb L" [ 1.0 1.0 1.0 ]');
  });

  it('emitters are one-sided unless twoSided; PLY meshes are emitted', () => {
    const q = text(toPbrt(getTier0Scene('quad-light'), opts).files.find((f) => f.path === 'scene.pbrt')!);
    expect(q).toContain('"bool twosided" false');
    expect(r.files.some((f) => f.path.endsWith('.ply'))).toBe(true);
  });

  it('keeps stock RR (and reports it) when rrStartDepth is given', () => {
    const rr = toPbrt({ ...scene, integrator: { maxDepth: 5, rrStartDepth: 3 } }, opts);
    expect(text(rr.files.find((f) => f.path === 'scene.pbrt')!)).not.toContain('"bool rr" false');
    expect(rr.approximations.join()).toMatch(/Russian roulette/);
  });
});

describe('fov helpers', () => {
  it('pbrt fov is the shorter axis: landscape = yFov, portrait = converted', () => {
    const cam = getTier0Scene('furnace-diffuse').camera;
    expect(shorterAxisFovDeg(cam, { width: 200, height: 100, filter: 'box' })).toBe(cam.yFovDeg);
    const portrait = shorterAxisFovDeg(cam, { width: 100, height: 200, filter: 'box' });
    expect(portrait).toBeLessThan(cam.yFovDeg);
    expect(xFovDeg(cam, { width: 100, height: 200, filter: 'box' })).toBeCloseTo(portrait, 9);
  });
});

describe('mitsuba exporter', () => {
  it('offsets max_depth by one, disables RR via rr_depth, sets fov_axis y, wraps diffuse in twosided', () => {
    const q = text(toMitsuba(getTier0Scene('quad-light'), opts).files.find((f) => f.path === 'scene.xml')!);
    expect(q).toContain('<integer name="max_depth" value="2"/>');
    expect(q).toContain('<integer name="rr_depth" value="1000000"/>');
    expect(q).toContain('<string name="fov_axis" value="y"/>');
    expect(q).toContain('<bsdf type="twosided">');
    expect(q).toContain('<rfilter type="box"/>');
    expect(q).toContain('<emitter type="area">');
  });
  it('two-sided emitters become two shapes and are reported', () => {
    const s = getTier0Scene('quad-light');
    const two = { ...s, meshes: s.meshes.map((m) => (m.emission ? { ...m, twoSided: true } : m)) };
    const r = toMitsuba(two, opts);
    expect(r.files.filter((f) => f.path.endsWith('.ply')).length).toBe(3);
    expect(r.approximations.join()).toMatch(/two-sided/);
  });
  it('rough conductor uses ggx alpha = roughness²', () => {
    const x = text(toMitsuba(getTier0Scene('furnace-conductor'), opts).files.find((f) => f.path === 'scene.xml')!);
    expect(x).toContain('type="roughconductor"');
    expect(x).toMatch(/name="alpha" value="0\.48999/);
  });
});

describe('glTF exporter', () => {
  it('produces a valid GLB container and lists approximations', () => {
    for (const s of tier0Scenes) {
      const r = toGltf(s, opts);
      const glb = r.files[0]!.data as Buffer;
      expect(glb.readUInt32LE(0)).toBe(0x46546c67);
      expect(glb.readUInt32LE(8)).toBe(glb.length);
      const jsonLen = glb.readUInt32LE(12);
      const json = JSON.parse(glb.subarray(20, 20 + jsonLen).toString('utf8'));
      expect(json.asset.version).toBe('2.0');
      expect(json.meshes.length).toBe(s.meshes.length);
      expect(json.cameras[0].perspective.yfov).toBeCloseTo((s.camera.yFovDeg * Math.PI) / 180, 9);
      expect(r.approximations.length).toBeGreaterThan(0);
    }
  });
  it('maps conductor to metallic F0 and dielectric to transmission+ior', () => {
    const c = toGltf(getTier0Scene('furnace-conductor'), opts);
    const d = toGltf(getTier0Scene('furnace-dielectric'), opts);
    const parse = (b: Buffer) => JSON.parse(b.subarray(20, 20 + b.readUInt32LE(12)).toString('utf8'));
    expect(parse(c.files[0]!.data as Buffer).materials[0].pbrMetallicRoughness.metallicFactor).toBe(1);
    expect(parse(d.files[0]!.data as Buffer).materials[0].extensions.KHR_materials_ior.ior).toBe(1.5);
  });
});

describe('octahedral env resampling', () => {
  it('equal-area mapping hits the poles and axes', () => {
    const top = equalAreaSquareToSphere(0.5, 0.5);
    expect(top[2]).toBeCloseTo(1, 6);
    const corner = equalAreaSquareToSphere(0, 0);
    expect(corner[2]).toBeCloseTo(-1, 6);
  });
  it('octahedral pixels reproduce the equirect radiance along their direction', () => {
    const img = { width: ENV_WIDTH, height: ENV_HEIGHT, channels: 3 as const, data: generateSkyImage() };
    const oct = equirectToOctahedral(img, 0, 1, 64);
    const i = 40, j = 20;
    const d = equalAreaSquareToSphere((i + 0.5) / 64, (j + 0.5) / 64);
    const direct = sampleEquirect(img, d, 0, 1);
    expect(oct.data[(j * 64 + i) * 3]).toBeCloseTo(direct[0], 5);
    // sun is a hot spot at its direction
    const sun = sampleEquirect(img, SUN_DIR, 0, 1);
    expect(sun[0]).toBeGreaterThan(skyRadiance([0, 1, 0])[0] * 10);
  });
  it('rotationYDeg rotates the environment', () => {
    const img = { width: ENV_WIDTH, height: ENV_HEIGHT, channels: 3 as const, data: generateSkyImage() };
    const rotated = sampleEquirect(img, [SUN_DIR[0] * Math.cos(0.5) + SUN_DIR[2] * Math.sin(0.5), SUN_DIR[1], -SUN_DIR[0] * Math.sin(0.5) + SUN_DIR[2] * Math.cos(0.5)], (0.5 * 180) / Math.PI, 1);
    const plain = sampleEquirect(img, SUN_DIR, 0, 1);
    expect(rotated[0]).toBeCloseTo(plain[0], 3);
  });
});
