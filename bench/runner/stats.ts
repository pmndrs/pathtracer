/**
 * Run discipline helpers: median of N, and the interleaved ABBA ordering for A/B runs of two configs.
 * Same idea as ~/Developer/upscaler/scripts/measure-timer-overhead.mjs (A B B A legs per block, median over blocks):
 * interleaving cancels slow drift (thermal, background load) that would otherwise land on whichever config ran last.
 */

export function median(values: readonly number[]): number {
  if (values.length === 0) return NaN;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = sorted.length >> 1;
  return sorted.length % 2 === 1 ? (sorted[middle] as number) : ((sorted[middle - 1] as number) + (sorted[middle] as number)) / 2;
}

/**
 * ABBA order: `rounds` pairs, alternating A B, B A, A B, ... so two rounds give A B B A and every config runs
 * `rounds` times, leading half the time. Length is 2 * rounds.
 */
export function abbaOrder<T>(a: T, b: T, rounds: number): T[] {
  const order: T[] = [];
  for (let round = 0; round < rounds; round++) {
    if (round % 2 === 0) order.push(a, b);
    else order.push(b, a);
  }
  return order;
}

/** Median per metric over a list of runs; a metric missing from some runs uses the runs that have it. */
export function medianMetrics(runs: ReadonlyArray<Record<string, number>>): Record<string, number> {
  const names = new Set(runs.flatMap((run) => Object.keys(run)));
  const out: Record<string, number> = {};
  for (const name of names) {
    const values = runs.map((run) => run[name]).filter((value): value is number => typeof value === 'number' && Number.isFinite(value));
    if (values.length > 0) out[name] = median(values);
  }
  return out;
}

/** Paired ratio B/A of two medians, the form the repo compares timings in (never across machines). */
export function pairedRatio(a: number, b: number): number {
  return a === 0 ? NaN : b / a;
}
