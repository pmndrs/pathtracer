import { afterEach, describe, expect, it, vi } from 'vitest';
import { PT_VOLUME_COLLISION_CAP, ptGridWorldDiagonal, ptVolumeCapRisk, ptWarnVolumeCapRisk } from './cap-risk.ts';
import { gridWorldToIndex, ptMediumFrom } from './volume.ts';

const grid = (res: number, size: number, densityMax: number) => ({ resolution: [res, res, res] as [number, number, number], worldToIndex: gridWorldToIndex([res, res, res], [[0, 0, 0], [size, size, size]]), densityMax });
const opts = (m: ReturnType<typeof ptMediumFrom>, majorant?: 'brick') => ({ media: [m], materials: [], ...(majorant ? { majorant } : {}) });

afterEach(() => vi.restoreAllMocks());

describe('ptVolumeCapRisk', () => {
  it('recovers the world diagonal of a grid', () => {
    expect(ptGridWorldDiagonal([256, 256, 256], grid(256, 512, 1).worldToIndex)).toBeCloseTo(512 * Math.sqrt(3), 6);
  });
  it('cloud256-like numbers: 0.3 x 887 = 266 collisions, risk 0.065, no warning; brick with a lower max is lower still', () => {
    const o = opts(ptMediumFrom([0, 0, 0], [0.3, 0.3, 0.3], 0, 1, grid(256, 512, 1)));
    const [r] = ptVolumeCapRisk(o);
    expect(r?.expectedCollisions).toBeCloseTo(0.3 * 512 * Math.sqrt(3), 3);
    expect(r?.risk).toBeCloseTo(0.065, 3);
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    ptWarnVolumeCapRisk('t', o);
    expect(warn).not.toHaveBeenCalled();
    // brick majorant max of a cloud whose densest 8^3 brick is 0.4 of densityMax
    expect(ptVolumeCapRisk({ ...o, majorant: 'brick' }, undefined, 0.4)[0]?.risk).toBeCloseTo(0.026, 3);
  });
  it('a dense grid warns once, naming the medium and the remedy', () => {
    const o = opts(ptMediumFrom([0, 0, 0], [5, 5, 5], 0, 1, grid(64, 512, 1)));
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const [r] = ptWarnVolumeCapRisk('PtTest', o);
    expect(r?.risk).toBeGreaterThan(0.1);
    ptWarnVolumeCapRisk('PtTest', o);
    expect(warn).toHaveBeenCalledTimes(1);
    expect(String(warn.mock.calls[0]?.[0])).toMatch(/medium 0.*brick/);
  });
  it('a homogeneous medium needs the diagonal', () => {
    const o = opts(ptMediumFrom([0, 0, 0], [2, 1, 1]));
    expect(ptVolumeCapRisk(o)).toEqual([]);
    expect(ptVolumeCapRisk(o, 100)[0]).toMatchObject({ expectedCollisions: 200, cap: PT_VOLUME_COLLISION_CAP });
  });
});
