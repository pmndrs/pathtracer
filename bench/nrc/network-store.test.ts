import { describe, expect, it } from 'vitest';
import { networkKey, usableNetwork, type SavedNetwork } from './network-store.ts';

const parts = { sceneHash: 'abc', network: 'mlp-64x3|lambert', maxDepth: 4 };
const entry = (over: Partial<SavedNetwork> = {}): SavedNetwork => ({ key: networkKey(parts), lighting: 'L', weights: new Float32Array([0.1, -0.2, 0.3]), trained: { frames: 50, steps: 200 }, savedAt: 1, ...over });

describe('saved network', () => {
  it('keys on the scene, the network and the path length', () => {
    const key = networkKey(parts);
    expect(networkKey({ ...parts })).toBe(key);
    expect(networkKey({ ...parts, sceneHash: 'abd' })).not.toBe(key);
    expect(networkKey({ ...parts, network: 'mlp-64x3|off' })).not.toBe(key);
    expect(networkKey({ ...parts, maxDepth: 5 })).not.toBe(key);
  });

  it('is usable only for the same key, lighting and weight count', () => {
    const key = networkKey(parts);
    expect(usableNetwork(entry(), key, 'L', 3)).toBe(true);
    expect(usableNetwork(undefined, key, 'L', 3)).toBe(false);
    expect(usableNetwork(entry(), key + 'x', 'L', 3)).toBe(false);
    expect(usableNetwork(entry(), key, 'lamps off', 3)).toBe(false);
    expect(usableNetwork(entry(), key, 'L', 4)).toBe(false);
  });

  it('refuses a network that never trained or holds a non-finite weight', () => {
    const key = networkKey(parts);
    expect(usableNetwork(entry({ trained: { frames: 0, steps: 0 } }), key, 'L', 3)).toBe(false);
    expect(usableNetwork(entry({ weights: new Float32Array([0.1, NaN, 0.3]) }), key, 'L', 3)).toBe(false);
  });
});
