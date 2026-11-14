/** Small deterministic RNG (PCG-style hash), shared by the oracle, tests and weight init so runs are reproducible. */

export function hashU32(x: number): number {
  // PCG output permutation (O'Neill), on 32 bits.
  let state = (Math.imul(x >>> 0, 747796405) + 2891336453) >>> 0;
  const word = Math.imul(((state >>> ((state >>> 28) + 4)) ^ state) >>> 0, 277803737) >>> 0;
  state = ((word >>> 22) ^ word) >>> 0;
  return state;
}

export class Rng {
  private state: number;
  constructor(seed: number) {
    this.state = hashU32(seed);
  }
  /** Uniform in [0, 1). */
  next(): number {
    this.state = hashU32(this.state + 0x9e3779b9);
    return this.state / 4294967296;
  }
  /** Standard normal (Box-Muller). */
  normal(): number {
    const u = Math.max(this.next(), 1e-12);
    const v = this.next();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
}
