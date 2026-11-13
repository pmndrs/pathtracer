/**
 * Pure analysis of resolved GPU timestamps (ns, as bigint), shared by bench/pages/timer-calibration.ts and its test.
 * Quantization shows as a large common step: Chrome rounds timestamp-query results to 100 us unless developer
 * features are on, so every timestamp is a multiple of 100000 ns and the step (gcd of all gaps) is 100000.
 * Real ticks give a step of a few ns (1 on Apple's 1 ns-resolution Metal counters).
 */

/** Anything coarser than this step counts as quantized. */
export const QUANTIZED_STEP_NS = 10_000n;

export function gcd(a: bigint, b: bigint): bigint {
  while (b !== 0n) [a, b] = [b, a % b];
  return a;
}

export interface TimestampAnalysis {
  /** gcd of all gaps between timestamps and the first one, ns. 0 if every timestamp is equal. */
  stepNs: bigint;
  quantized: boolean;
  /** Pass durations in ns (end - begin), in pass order. */
  durationsNs: number[];
  zeroShare: number;
  distinctDurations: number;
  minNonzeroNs: number;
}

/** `stamps` holds [begin0, end0, begin1, end1, ...]. */
export function analyzeTimestamps(stamps: ArrayLike<bigint>): TimestampAnalysis {
  const passes = stamps.length >> 1;
  const first = stamps[0] as bigint;
  let step = 0n;
  const durationsNs: number[] = [];
  for (let pass = 0; pass < passes; pass++) {
    const begin = stamps[pass * 2] as bigint;
    const end = stamps[pass * 2 + 1] as bigint;
    step = gcd(step, begin > first ? begin - first : first - begin);
    step = gcd(step, end > first ? end - first : first - end);
    durationsNs.push(Number(end - begin));
  }
  const nonzero = durationsNs.filter((value) => value > 0);
  return {
    stepNs: step,
    // A step of 0 means every timestamp was identical: a dead timer, which is not usable either.
    quantized: step === 0n || step >= QUANTIZED_STEP_NS,
    durationsNs,
    zeroShare: durationsNs.length === 0 ? 0 : durationsNs.filter((value) => value === 0).length / durationsNs.length,
    distinctDurations: new Set(durationsNs).size,
    minNonzeroNs: nonzero.length > 0 ? Math.min(...nonzero) : 0,
  };
}
