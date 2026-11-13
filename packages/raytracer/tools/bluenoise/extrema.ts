/**
 * Incremental arg-max / arg-min over a mutable array, as a wide (B = 32) tournament tree.
 *
 * Void-and-cluster needs "the filled cell with the highest energy" (tightest cluster) and "the empty cell
 * with the lowest energy" (largest void) after every single change; the energy changes only inside a small
 * window, so a full scan per step (N^2 total) is the cost that matters at 128x128x64 (1M cells).
 * Here a change marks its leaf block dirty and `flush()` recomputes only dirty nodes bottom-up.
 */
const B = 32;

export class DualExtrema {
  /** key used for the arg-max over "ones" (-Infinity where the cell is empty). */
  readonly hi: Float64Array;
  /** key used for the arg-max over "zeros" (-Infinity where the cell is filled); we store -energy. */
  readonly lo: Float64Array;
  private readonly sizes: number[] = [];
  private readonly bestHi: Int32Array[] = [];
  private readonly bestLo: Int32Array[] = [];
  private readonly stamp: Int32Array[] = [];
  private readonly dirty: Int32Array[] = [];
  private readonly dirtyCount: number[] = [];
  private epoch = 1;

  readonly n: number;

  constructor(n: number) {
    this.n = n;
    this.hi = new Float64Array(n).fill(-Infinity);
    this.lo = new Float64Array(n).fill(-Infinity);
    let size = n;
    do {
      size = Math.ceil(size / B);
      this.sizes.push(size);
      this.bestHi.push(new Int32Array(size).fill(-1));
      this.bestLo.push(new Int32Array(size).fill(-1));
      this.stamp.push(new Int32Array(size));
      this.dirty.push(new Int32Array(size));
      this.dirtyCount.push(0);
    } while (size > 1);
  }

  touch(leaf: number): void {
    const b = (leaf / B) | 0;
    if (this.stamp[0]![b] !== this.epoch) {
      this.stamp[0]![b] = this.epoch;
      this.dirty[0]![this.dirtyCount[0]!++] = b;
    }
  }

  /** Recompute every block touched since the last flush, all the way to the root. */
  flush(): void {
    for (let level = 0; level < this.sizes.length; level++) {
      const count = this.dirtyCount[level]!;
      const dirty = this.dirty[level]!;
      const nextLevel = level + 1;
      const hasNext = nextLevel < this.sizes.length;
      for (let d = 0; d < count; d++) {
        const node = dirty[d]!;
        this.recompute(level, node);
        if (hasNext) {
          const parent = (node / B) | 0;
          if (this.stamp[nextLevel]![parent] !== this.epoch) {
            this.stamp[nextLevel]![parent] = this.epoch;
            this.dirty[nextLevel]![this.dirtyCount[nextLevel]!++] = parent;
          }
        }
      }
      this.dirtyCount[level] = 0;
    }
    this.epoch++;
  }

  /** Full rebuild (after bulk initialisation). */
  rebuild(): void {
    for (let level = 0; level < this.sizes.length; level++) {
      for (let node = 0; node < this.sizes[level]!; node++) this.recompute(level, node);
    }
  }

  private recompute(level: number, node: number): void {
    const hiKeys = this.hi;
    const loKeys = this.lo;
    let bh = -1;
    let bl = -1;
    let vh = -Infinity;
    let vl = -Infinity;
    if (level === 0) {
      const start = node * B;
      const end = Math.min(start + B, this.n);
      for (let i = start; i < end; i++) {
        const h = hiKeys[i]!;
        if (h > vh) { vh = h; bh = i; }
        const l = loKeys[i]!;
        if (l > vl) { vl = l; bl = i; }
      }
    } else {
      const childHi = this.bestHi[level - 1]!;
      const childLo = this.bestLo[level - 1]!;
      const start = node * B;
      const end = Math.min(start + B, this.sizes[level - 1]!);
      for (let c = start; c < end; c++) {
        const ih = childHi[c]!;
        if (ih >= 0 && hiKeys[ih]! > vh) { vh = hiKeys[ih]!; bh = ih; }
        const il = childLo[c]!;
        if (il >= 0 && loKeys[il]! > vl) { vl = loKeys[il]!; bl = il; }
      }
    }
    this.bestHi[level]![node] = bh;
    this.bestLo[level]![node] = bl;
  }

  /** Index of the filled cell with the highest energy, or -1 if none. */
  argMaxFilled(): number {
    return this.bestHi[this.sizes.length - 1]![0]!;
  }

  /** Index of the empty cell with the lowest energy, or -1 if none. */
  argMinEmpty(): number {
    return this.bestLo[this.sizes.length - 1]![0]!;
  }
}
