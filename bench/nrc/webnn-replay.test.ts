import { describe, expect, it } from 'vitest';
import { WEBNN_REPLAY_OPTIONS, webnnFoldShapes, webnnReplayOptions } from './webnn-replay-options.ts';

describe('WebNN replay hand-off variants (sync study)', () => {
  it('defaults to the November 19 sequence, so the old command still measures the same thing', () => {
    expect(webnnReplayOptions({})).toEqual({ ctx: 'type', tensors: 'shared', sync: 'drain', route: 'export', parity: 'strict', breakdown: false });
    for (const [key, values] of Object.entries(WEBNN_REPLAY_OPTIONS)) expect(webnnReplayOptions({})[key as keyof typeof WEBNN_REPLAY_OPTIONS]).toBe(values[0]);
  });

  it('accepts every documented value and rejects others', () => {
    for (const [key, values] of Object.entries(WEBNN_REPLAY_OPTIONS)) {
      for (const value of values) if (!(key === 'sync' && value === 'none')) expect(webnnReplayOptions({ [key]: value })).toMatchObject({ [key]: value });
      expect(() => webnnReplayOptions({ [key]: 'nope' })).toThrow(key);
    }
    expect(webnnReplayOptions({ breakdown: '1' }).breakdown).toBe(true);
  });

  it('refuses combinations that would measure nothing', () => {
    expect(() => webnnReplayOptions({ route: 'cpu', sync: 'none' })).toThrow('sync=none');
    expect(() => webnnReplayOptions({ tensors: 'chunk', publish: 'slab' })).toThrow('publish=slab');
    expect(webnnReplayOptions({ sync: 'none', parity: 'report' })).toMatchObject({ sync: 'none', route: 'export' });
  });

  it('folds a frame into a wider tensor without changing its bytes, and is off by default', () => {
    expect(webnnFoldShapes(4096, 1, 64)).toEqual({ slab: 4096, x: [4096, 64], y: [4096, 4], flatX: [4096, 64], flatY: [4096, 4] });
    const f = webnnFoldShapes(4096, 8, 64);
    expect(f).toEqual({ slab: 32768, x: [4096, 512], y: [4096, 32], flatX: [32768, 64], flatY: [32768, 4] });
    expect(f.x[0]! * f.x[1]!).toBe(f.flatX[0]! * f.flatX[1]!);
    expect(f.y[0]! * f.y[1]!).toBe(f.flatY[0]! * f.flatY[1]!);
    expect(f.x[0]).toBeLessThanOrEqual(8192);
    expect(() => webnnFoldShapes(4096, 3, 64)).toThrow('fold');
  });
});
