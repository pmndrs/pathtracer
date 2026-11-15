import { describe, expect, it } from 'vitest';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { toGltf, toMitsuba, toPbrt } from './index.ts';
import { equalAreaSquareToSphere, equirectForMitsuba, equirectToOctahedral, sampleEquirect } from './envmap.ts';
import { shorterAxisFovDeg, xFovDeg } from './common.ts';
import { equirectDirection as equirectDirectionOf, generateSkyImage, ENV_WIDTH, ENV_HEIGHT, SUN_DIR, skyRadiance } from '../scenes/tier0/env-sky.ts';
import { tier0Scenes, getTier0Scene } from '../scenes/tier0/index.ts';
import { loadGltfTriangles } from '../scenes/gltf-loader.ts';

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

  it('maps rrStartDepth to the patched rrdepth = rrStartDepth - 1 (RR first fires at vertex rrStartDepth)', () => {
    const rr = toPbrt({ ...scene, integrator: { maxDepth: 5, rrStartDepth: 4 } }, opts);
    const t = text(rr.files.find((f) => f.path === 'scene.pbrt')!);
    expect(t).not.toContain('"bool rr" false');
    expect(t).toContain('"integer maxdepth" 5 "integer rrdepth" 3');
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

describe('Mitsuba envmap row convention', () => {
  it("rows resampled to v = j / (H - 1) make Mitsuba's lookup (row coordinate v * (H - 1)) equal our bilinear (v * H - 0.5)", () => {
    const img = { width: ENV_WIDTH, height: ENV_HEIGHT, channels: 3 as const, data: generateSkyImage() };
    const warped = equirectForMitsuba(img);
    const mitsubaRow = (v: number, i: number) => {
      const fy = v * (ENV_HEIGHT - 1);
      const y0 = Math.min(ENV_HEIGHT - 2, Math.floor(fy));
      const t = fy - y0;
      return (1 - t) * warped.data[(y0 * ENV_WIDTH + i) * 3 + 1]! + t * warped.data[((y0 + 1) * ENV_WIDTH + i) * 3 + 1]!;
    };
    // a smooth sky column at texel-centre u: far from the sun the re-interpolation error is tiny
    const i = 400;
    for (const v of [0.02, 0.3, 0.5, 0.7, 0.98]) {
      const ours = sampleEquirect(img, equirectDirectionOf((i + 0.5) / ENV_WIDTH, v), 0, 1)[1];
      expect(mitsubaRow(v, i)).toBeCloseTo(ours, 3);
    }
    // texel (i, 0) is the zenith row and stays unchanged
    expect(warped.data[400 * 3]).toBeCloseTo(img.data[400 * 3]!, 6);
  });
});

describe('file geometry (glb) end to end', () => {
  // Round trip: the glTF exporter's GLB of a tier-0 scene is loaded back by the Node loader and must give the same
  // triangles (flat-shaded, un-welded) that the pbrt/Mitsuba exporters write as PLY for a `file` mesh.
  it('glb exported by toGltf loads back with the same triangle count and area; file meshes export to PLY/GLB', async () => {
    const { mkdtempSync, writeFileSync, rmSync } = await import('node:fs');
    const { tmpdir } = await import('node:os');
    const { surfaceArea, meshToWorld } = await import('../scenes/geometry.ts');
    const { worldMeshes } = await import('./common.ts');
    const src = getTier0Scene('cornell');
    const glb = toGltf(src, opts).files[0]!.data as Buffer;
    const dir = mkdtempSync(join(tmpdir(), 'glb-'));
    try {
      writeFileSync(join(dir, 'scene.glb'), glb);
      const loaded = loadGltfTriangles(join(dir, 'scene.glb'));
      const expectTris = src.meshes.reduce((a, m) => a + meshToWorld(m).indices.length / 3, 0);
      expect(loaded.stats.triangles).toBe(expectTris);
      expect(surfaceArea(loaded.mesh)).toBeCloseTo(src.meshes.reduce((a, m) => a + surfaceArea(meshToWorld(m)), 0), 3);
      const scene = {
        ...src, name: 'file-test', meshes: [{ name: 'all', geometry: { type: 'file' as const, path: 'scene.glb', format: 'glb' as const }, material: Object.keys(src.materials)[0]! }],
      };
      const w = worldMeshes(scene, dir);
      expect(w[0]!.mesh.indices.length / 3).toBe(expectTris);
      for (const t of [toPbrt, toMitsuba, toGltf]) expect(t(scene, { assetsDir: dir, spp: 1 }).files.length).toBeGreaterThan(0);
      // sha256 pin is enforced
      expect(() => loadGltfTriangles(join(dir, 'scene.glb'), { sha256: '00' })).toThrow(/sha256 mismatch/);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});
