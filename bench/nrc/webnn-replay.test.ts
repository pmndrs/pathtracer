import { describe, expect, it } from 'vitest';
import { WEBNN_REPLAY_OPTIONS, webnnReplayOptions } from './webnn-replay-options.ts';

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
});
