import type { NrcSettings } from './cache.ts';

/** Shared controls for comparing the compact network with three hash representations. */
export function phase10Settings(arm: string): Partial<NrcSettings> {
  const shared = {
    network: 'compact', materialInputs: 'evaluated', factorization: 'off',
    training: 'inline', trainSelect: 'tile', updateBlock: 4, stepsPerFrame: 2,
    queryDepth: 1, trainDepth: 4, termination: 'fixed', fullSuffixEvery: 0,
  } as const;
  if (arm === 'mlp') return { ...shared, model: 'mlp', directionInput: 'reflected', positionBands: 4 };
  if (arm !== 'rgb' && arm !== 'bins' && arm !== 'sh') throw new Error('phase10 must be mlp, rgb, bins or sh');
  return {
    ...shared, model: 'hash', directionInput: 'view',
    hash: { memoryBudgetBytes: 10 * 1024 ** 2 + 84, dirBins: arm === 'bins' ? 4 : 0, directionalSh: arm === 'sh' },
  };
}
