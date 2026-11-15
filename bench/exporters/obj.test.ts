import { describe, expect, it } from 'vitest';
import { existsSync } from 'node:fs';
import { join } from 'node:path';
import { ASSETS_DIR } from '../scenes/file-geometry.ts';
import { getTier0Scene } from '../scenes/tier0/index.ts';
import { getScene } from '../scenes/index.ts';
import { sponzaWhite, sponzaWhiteVknrc, SPONZA_GLTF } from '../scenes/tier2/sponza-white.ts';
import { worldMeshes } from './common.ts';
import { f32Text, toObj, vknrcLook, vknrcYawPitch, VKNRC_MAX_DEPTH, VKNRC_SKY_RADIANCE } from './obj.ts';

const text = (r: ReturnType<typeof toObj>, path: string): string => String(r.files.find((f) => f.path === path)!.data);

/** A minimal OBJ reader for our own output: `v` lines and triangle `f` lines (1-based), with the active `usemtl`. */
function parseObj(src: string): { v: number[][]; f: { idx: number[]; mtl: string }[]; mtllib: string | null } {
  const v: number[][] = [];
  const f: { idx: number[]; mtl: string }[] = [];
  let mtl = '';
  let mtllib: string | null = null;
  for (const line of src.split('\n')) {
    const [k, ...rest] = line.trim().split(/\s+/);
    if (k === 'v') v.push(rest.map(Number));
    else if (k === 'f') f.push({ idx: rest.map((s) => Number(s.split('/')[0])), mtl });
    else if (k === 'usemtl') mtl = rest[0]!;
    else if (k === 'mtllib') mtllib = rest[0]!;
  }
  return { v, f, mtllib };
}

describe('f32Text', () => {
  it('round-trips float32 values in at most 9 significant digits', () => {
    let seed = 12345;
    const rnd = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 2 ** 32 - 0.5) * 60;
    for (let i = 0; i < 2000; i++) {
      const x = Math.fround(rnd());
      const s = f32Text(x);
      expect(Math.fround(Number(s))).toBe(x);
      expect(s.replace(/^-|\.|e.*$/g, '').replace(/^0+/, '').length).toBeLessThanOrEqual(9);
    }
    expect(f32Text(0)).toBe('0');
    expect(f32Text(0.5)).toBe('0.5');
  });
});

describe('VkNRC camera pose', () => {
  it('yaw/pitch reproduce the look direction (yaw 0, pitch 0 looks down -Z)', () => {
    expect(vknrcLook(0, 0).map((x) => x + 0)).toEqual([0, 0, -1]);
    for (const d of [[1, 0, 0], [18, 0.9, 0], [0.3, -0.8, 0.4], [-2, 1, 5]] as [number, number, number][]) {
      const { yaw, pitch } = vknrcYawPitch(d);
      const l = Math.hypot(...d);
      const back = vknrcLook(yaw, pitch);
      for (let a = 0; a < 3; a++) expect(back[a]).toBeCloseTo(d[a]! / l, 12);
    }
  });
});

describe('OBJ exporter (VkNRC input, N6.2)', () => {
  const scene = getTier0Scene('cornell');
  const r = toObj(scene, { assetsDir: ASSETS_DIR });
  const obj = parseObj(text(r, 'scene.obj'));
  const mtl = text(r, 'scene.mtl');

  it('writes positions and triangles only, every face under a material, indices in range', () => {
    expect(obj.mtllib).toBe('scene.mtl');
    expect(text(r, 'scene.obj')).not.toMatch(/^vn /m);
    const tris = worldMeshes(scene, ASSETS_DIR).reduce((n, m) => n + m.mesh.indices.length / 3, 0);
    expect(obj.f.length).toBe(tris);
    expect(r.sidecar.triangles).toBe(tris);
    expect(obj.v.length).toBe(r.sidecar.vertices);
    expect(obj.v.length).toBeLessThan(3 * tris); // welded
    for (const face of obj.f) {
      expect(face.idx).toHaveLength(3);
      expect(face.mtl).not.toBe('');
      for (const i of face.idx) expect(i >= 1 && i <= obj.v.length).toBe(true);
    }
  });

  it('keeps every triangle bit-exact in float32, in scene order', () => {
    const want = worldMeshes(scene, ASSETS_DIR).flatMap(({ mesh }) => Array.from(mesh.indices, (i) => [0, 1, 2].map((a) => Math.fround(mesh.positions[i * 3 + a]!))));
    const got = obj.f.flatMap((face) => face.idx.map((i) => obj.v[i - 1]!.map(Math.fround)));
    expect(got).toEqual(want);
  });

  it('maps diffuse to Kd with Ks 0 (VkNRC Cook-Torrance -> Lambert) and emission to Ke', () => {
    expect(mtl).toMatch(/newmtl white\nKd [\d.]+ [\d.]+ [\d.]+\nKs 0 0 0\nKe 0 0 0/);
    const light = obj.f.find((face) => face.mtl.startsWith('black_e'))!.mtl;
    expect(mtl).toContain(`newmtl ${light}\nKd 0 0 0\nKs 0 0 0\nKe 18.4 14 6.75`);
    expect(r.approximations.some((a) => a.startsWith('emission:'))).toBe(true);
    expect(r.approximations.some((a) => a.startsWith('environment:'))).toBe(true);
  });

  it('sidecar: VkNRC normalisation puts the model in [-1, 1] with longest extent 2; camera pose in that frame', () => {
    const s = r.sidecar;
    const n = obj.v.map((p) => p.map((x, a) => (x - s.center[a]!) * s.scale));
    const lo = [0, 1, 2].map((a) => Math.min(...n.map((p) => p[a]!)));
    const hi = [0, 1, 2].map((a) => Math.max(...n.map((p) => p[a]!)));
    for (let a = 0; a < 3; a++) {
      expect(lo[a]).toBeGreaterThanOrEqual(-1 - 1e-6);
      expect(hi[a]).toBeLessThanOrEqual(1 + 1e-6);
    }
    expect(Math.max(...[0, 1, 2].map((a) => hi[a]! - lo[a]!))).toBeCloseTo(2, 5);
    const c = scene.camera;
    for (let a = 0; a < 3; a++) expect(s.camera.position[a]).toBeCloseTo((c.position[a]! - s.center[a]!) * s.scale, 6);
    const look = [0, 1, 2].map((a) => c.target[a]! - c.position[a]!);
    const l = Math.hypot(...look);
    const back = vknrcLook(s.camera.yaw, s.camera.pitch);
    for (let a = 0; a < 3; a++) expect(back[a]).toBeCloseTo(look[a]! / l, 9);
    expect(s.camera.fovY).toBeCloseTo((c.yFovDeg * Math.PI) / 180, 12);
  });

  it('throws on materials VkNRC cannot match', () => {
    expect(() => toObj(getTier0Scene('furnace-conductor'), { assetsDir: ASSETS_DIR })).toThrow(/conductor/);
  });
});

describe('sponza-white-vknrc', () => {
  it('is sponza-white with VkNRC lighting: sky 10, no sun, 7 scattering vertices, no RR', () => {
    expect(getScene('sponza-white-vknrc')).toBe(sponzaWhiteVknrc);
    expect(sponzaWhiteVknrc.meshes).toBe(sponzaWhite.meshes);
    expect(sponzaWhiteVknrc.materials).toBe(sponzaWhite.materials);
    expect(sponzaWhiteVknrc.camera).toBe(sponzaWhite.camera);
    expect(sponzaWhiteVknrc.lights).toEqual([]);
    expect(sponzaWhiteVknrc.environment).toEqual({ type: 'uniform', radiance: [VKNRC_SKY_RADIANCE, VKNRC_SKY_RADIANCE, VKNRC_SKY_RADIANCE] });
    expect(sponzaWhiteVknrc.integrator).toEqual({ maxDepth: VKNRC_MAX_DEPTH });
  });

  // Needs the git-ignored Sponza download (node bench/assets/fetch.ts).
  const have = existsSync(join(ASSETS_DIR, SPONZA_GLTF.path)) && existsSync(join(ASSETS_DIR, 'downloads/sponza/Sponza.bin'));
  it.skipIf(!have)('exports Sponza exactly: 262,266 triangles, no approximation besides none', () => {
    const s = toObj(sponzaWhiteVknrc, { assetsDir: ASSETS_DIR });
    expect(s.sidecar.triangles).toBe(262266);
    expect(s.approximations).toEqual([]);
  }, 120_000);
});
