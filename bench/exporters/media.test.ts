import { describe, expect, it } from 'vitest';
import { ASSETS_DIR } from '../scenes/tier0/generate-assets.ts';
import { getScene } from '../scenes/index.ts';
import { sceneDescToCopyInput } from '../scenes/to-scene-copy.ts';
import { sceneVolumes } from '../scenes/volumes.ts';
import { toMitsuba, toPbrt } from './index.ts';
import { mitsubaMedium, mitsubaVol } from './media.ts';

const text = (f: { data: string | Uint8Array } | undefined): string => (f ? (typeof f.data === 'string' ? f.data : Buffer.from(f.data).toString()) : '');
const opts = { assetsDir: ASSETS_DIR, spp: 16 };

describe('media export (P5)', () => {
  it('pbrt: volpath, a named medium per scene medium, MediumInterface on the meshes, the interface material', () => {
    const out = toPbrt(getScene('vol-cornell-smoke'), opts);
    const scene = text(out.files.find((f) => f.path === 'scene.pbrt'));
    expect(scene).toMatch(/Integrator "volpath" "integer maxdepth" 32 "bool rr" false/);
    expect(scene).toMatch(/MakeNamedMedium "smoke" "string type" "homogeneous" "rgb sigma_a" \[ 1\.0 1\.0 1\.0 \] "rgb sigma_s" \[ 9\.0 9\.0 9\.0 \] "float scale" 1\.0 "float g" 0\.3/);
    expect(scene).toContain('Material "interface"');
    expect(scene).toContain('MediumInterface "smoke" ""');
  });

  it('pbrt: a grid is a uniformgrid with its bounds and values, placed by the mesh transform', () => {
    const scene = text(toPbrt(getScene('vol-grid-slab'), opts).files.find((f) => f.path === 'scene.pbrt'));
    expect(scene).toMatch(/"string type" "uniformgrid".*"integer nx" 4 "integer ny" 4 "integer nz" 16 "point3 p0" \[ -4\.0 -4\.0 -0\.5 \] "point3 p1" \[ 4\.0 4\.0 0\.5 \] "float density" \[ /);
  });

  it('pbrt: the camera medium is declared before Camera and cleared after WorldBegin', () => {
    const scene = text(toPbrt(getScene('vol-camera-medium'), opts).files.find((f) => f.path === 'scene.pbrt'));
    expect(scene.indexOf('MediumInterface "" "fog"')).toBeGreaterThan(0);
    expect(scene.indexOf('MediumInterface "" "fog"')).toBeLessThan(scene.indexOf('Camera "perspective"'));
    expect(scene.indexOf('WorldBegin\nMediumInterface "" ""')).toBeGreaterThan(0);
  });

  it('mitsuba: volpath, scale x rgb media, interior refs, null BSDF; max_depth is ours + 1', () => {
    const out = toMitsuba(getScene('vol-cornell-smoke'), opts);
    const xml = text(out.files.find((f) => f.path === 'scene.xml'));
    expect(xml).toContain('<integrator type="volpath">');
    expect(xml).toContain('<integer name="max_depth" value="33"/>');
    expect(xml).toMatch(/<medium type="homogeneous" id="smoke"><rgb name="sigma_t" value="1\.0 1\.0 1\.0"\/><rgb name="albedo" value="0\.9 0\.9 0\.9"\/><float name="scale" value="10\.0"\/><phase type="hg"><float name="g" value="0\.3"\/><\/phase><\/medium>/);
    expect(xml).toContain('<ref name="interior" id="smoke"/>');
    expect(xml).toContain('<bsdf type="null"/>');
  });

  it('mitsuba: a grid is a heterogeneous medium with a .vol file (version 3, float32, one channel); a chromatic sigma_t throws', () => {
    const out = toMitsuba(getScene('vol-grid-slab'), opts);
    const vol = out.files.find((f) => f.path === 'layers.vol');
    expect(vol).toBeDefined();
    const buf = Buffer.from(vol!.data as Uint8Array);
    expect(buf.subarray(0, 3).toString()).toBe('VOL');
    expect(buf.readUInt8(3)).toBe(3);
    expect([buf.readInt32LE(4), buf.readInt32LE(8), buf.readInt32LE(12), buf.readInt32LE(16), buf.readInt32LE(20)]).toEqual([1, 4, 4, 16, 1]);
    expect(buf.length).toBe(48 + 4 * 4 * 16 * 4);
    const g = getScene('vol-grid-slab').media!.layers!;
    expect(() => mitsubaMedium('x', { ...g, sigmaA: [1, 2, 3] } as never, undefined, ASSETS_DIR)).toThrow(/chromatic/);
    expect(mitsubaVol([1, 1, 2], [1.5, 2.5], [[0, 0, 0], [1, 1, 1]]).readFloatLE(48 + 4)).toBeCloseTo(2.5, 6);
  });

  it('the scene copy gives each (material, medium) pair its own material id and the integrator its table', () => {
    const scene = getScene('vol-milk-sphere');
    const input = sceneDescToCopyInput(scene);
    const v = sceneVolumes(scene, input)!;
    expect(v.options.media).toHaveLength(1);
    expect(v.options.media[0]!.sigmaT[0]).toBeCloseTo(2.55, 6);
    expect(v.options.media[0]!.albedo[0]).toBeCloseTo(2.5 / 2.55, 6);
    expect(input.materialVolumes.some((m) => m.interior === 1 && !m.boundary)).toBe(true);
    expect(input.materialVolumes.some((m) => m.interior === 0 && m.exterior === 0)).toBe(true);
  });

  it('a densityFile grid needs its file at load time, takes it when given, and the exporters name the same box', () => {
    const scene = getScene('vol-vdb-smoke');
    const input = sceneDescToCopyInput(scene);
    expect(() => sceneVolumes(scene, input)).toThrow(/density file/);
    const [nx, ny, nz] = (scene.media!.plume as { resolution: [number, number, number] }).resolution;
    const files = new Map([['generated/volumes/vdb-smoke.f32', new Float32Array(nx * ny * nz).fill(0.25)]]);
    const v = sceneVolumes(scene, input, files)!;
    expect(v.grid!.resolution).toEqual([nx, ny, nz]);
    expect(v.options.media[0]!.grid!.densityMax).toBeCloseTo(0.25, 6);
    // the grid's world-to-index scale is (n / extent) / mesh scale: one voxel is extent / n in the mesh's object space, scaled by the 1.6 / 109.5 mesh transform
    expect(v.options.media[0]!.grid!.worldToIndex[0]).toBeCloseTo(nx / (2 * 28.571428571428573) / (1.6 / (2 * 54.76190476190476)), 4);
  });

  it('an interface mesh without a medium is an error, and so is a grid with no owner mesh', () => {
    const s = getScene('vol-absorb-slab');
    expect(() => sceneDescToCopyInput({ ...s, meshes: s.meshes.map((m) => ({ ...m, medium: undefined })) })).toThrow(/interface material but no/);
    const g = getScene('vol-grid-slab');
    const input = sceneDescToCopyInput(g);
    expect(() => sceneVolumes({ ...g, meshes: g.meshes.map((m) => ({ ...m, medium: undefined })) }, input)).toThrow(/exactly one mesh/);
  });
});
