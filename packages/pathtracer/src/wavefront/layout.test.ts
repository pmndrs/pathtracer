import { describe, expect, it } from 'vitest';
import { PTWF_INIT, PTWF_STRIDE, PTWF_WORKGROUP, ptwfArgsStep, ptwfBufferBytes, ptwfDispatchesPerSample, ptwfGroups, ptwfPackDepth, ptwfQueueOffsets, ptwfRecordsPerSample, ptwfSlot, ptwfStateBytes, ptwfUnpackDepth } from './layout.ts';

describe('wavefront layout (P3.1)', () => {
  it('path state is 64 B plain and 80 B with a medium or textures, under the 96 B budget', () => {
    expect(ptwfStateBytes({ medium: false, textures: false })).toBe(64);
    expect(ptwfStateBytes({ medium: true, textures: false })).toBe(80);
    expect(ptwfStateBytes({ medium: false, textures: true })).toBe(80);
    expect(ptwfStateBytes({ medium: true, textures: true })).toBeLessThanOrEqual(96);
  });

  it('per-path buffers add up (720p, plain)', () => {
    const b = ptwfBufferBytes(1280 * 720, { medium: false, textures: false });
    expect(b.state).toBe(1280 * 720 * 64);
    expect(b.hit).toBe(1280 * 720 * 48);
    expect(b.shadow).toBe(1280 * 720 * 48);
    expect(b.queue).toBe(1280 * 720 * 12);
    expect(b.total).toBe(b.state + b.hit + b.shadow + b.queue + b.counters);
  });

  it('packs depth and the specular flag in one word and round-trips', () => {
    for (const depth of [0, 1, 7, 1000, 0x7fffffff]) {
      for (const specular of [false, true]) {
        const word = ptwfPackDepth(depth, specular);
        expect(word).toBeGreaterThanOrEqual(0);
        expect(word).toBeLessThan(2 ** 32);
        expect(ptwfUnpackDepth(word)).toEqual({ depth, specular });
      }
    }
    expect(ptwfPackDepth(5, true)).toBe(0x80000005);
    expect(() => ptwfPackDepth(0x80000000, false)).toThrow(RangeError);
    expect(() => ptwfPackDepth(-1, false)).toThrow(RangeError);
  });

  it('indirect groups cover the queue and fold into y past the stride', () => {
    expect(ptwfGroups(0)).toEqual([0, 0, 1]);
    expect(ptwfGroups(1)).toEqual([1, 1, 1]);
    expect(ptwfGroups(64)).toEqual([1, 1, 1]);
    expect(ptwfGroups(65)).toEqual([2, 1, 1]);
    for (const count of [1, 63, 64, 65, 4096 * 64, 4096 * 64 + 1, 921600, 8_294_400, 20_000_000]) {
      const [x, y] = ptwfGroups(count);
      expect(x).toBeLessThanOrEqual(PTWF_STRIDE);
      expect(y).toBeLessThanOrEqual(65535);
      // every queue slot < count has a thread, and the threads past it are fewer than one workgroup row's worth of slack
      let maxSlot = ptwfSlot([x - 1, y - 1], PTWF_WORKGROUP - 1);
      if (y > 1 || x * PTWF_WORKGROUP >= count) expect(maxSlot + 1).toBeGreaterThanOrEqual(count);
      expect(maxSlot + 1 - count).toBeLessThan(PTWF_STRIDE * PTWF_WORKGROUP);
    }
  });

  it('slot numbering is bijective over a dispatch', () => {
    const [x, y] = ptwfGroups(4096 * 64 * 2 + 70);
    const seen = new Set<number>();
    for (let wy = 0; wy < y; wy++) for (let wx = 0; wx < x; wx++) for (let li = 0; li < PTWF_WORKGROUP; li += 31) seen.add(ptwfSlot([wx, wy], li));
    expect(seen.size).toBe(x * y * Math.ceil(PTWF_WORKGROUP / 31));
  });

  it('queue regions do not overlap', () => {
    const o = ptwfQueueOffsets(1000);
    expect(o.extension).toEqual([0, 1000]);
    expect(o.shadow).toBe(2000);
  });

  it('dispatch and record counts', () => {
    expect(ptwfRecordsPerSample(5)).toBe(7);
    expect(ptwfDispatchesPerSample(0)).toBe(4); // camera, args, intersect, shade
    expect(ptwfDispatchesPerSample(5)).toBe(2 + 12 + 10);
  });

  /**
   * The counter protocol against a CPU model of the kernels: each iteration the shade step moves a random subset of the queued paths to
   * the next queue and a random subset to the shadow queue (atomics: counters are bumped), args() reads and clears. Checks that every
   * queued path is consumed exactly once per stage, that no counter holds a stale count when a stage starts appending, and that the
   * dispatch args always cover the queue.
   */
  it('counter ping-pong never loses or repeats a path (simulated)', () => {
    let seed = 12345;
    const rand = () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 2 ** 32);
    for (const paths of [1, 7, 64, 65, 1000, 5000]) {
      for (const maxDepth of [1, 2, 5, 9]) {
        const counters = new Uint32Array(4);
        const queues: number[][] = [[], []];
        const shadowQ: number[] = [];
        let args = ptwfArgsStep(counters, PTWF_INIT, paths);
        for (let i = 0; i < paths; i++) queues[0]!.push(i); // camera: Q0 = identity
        const seenAlive = new Set<number>();
        for (let d = 0; d <= maxDepth; d++) {
          const q = d & 1;
          const count = counters[q] as number;
          expect(count).toBe(queues[q]!.length);
          const cover = args.main[0] * args.main[1] * PTWF_WORKGROUP;
          expect(cover).toBeGreaterThanOrEqual(count);
          // shade: every queued path once; counters of the *next* queue and of this iteration's shadow queue must start at 0
          expect(counters[q ^ 1]).toBe(0);
          expect(counters[2 + q]).toBe(0);
          const read = [...queues[q]!];
          expect(new Set(read).size).toBe(read.length);
          const nextQ: number[] = [];
          const shadowNow: number[] = [];
          for (const path of read) {
            seenAlive.add(path);
            if (d < maxDepth && rand() < 0.9) shadowNow.push(path);
            if (d < maxDepth && rand() < 0.8) nextQ.push(path);
          }
          queues[q ^ 1]!.length = 0;
          queues[q ^ 1]!.push(...nextQ);
          counters[q ^ 1] = nextQ.length;
          shadowQ.length = 0;
          shadowQ.push(...shadowNow);
          counters[2 + q] = shadowNow.length;
          if (d === maxDepth) break;
          args = ptwfArgsStep(counters, d, paths);
          // args cleared the queue shade read, and the other shadow counter; the next queue count is untouched
          expect(counters[q]).toBe(0);
          queues[q]!.length = 0;
          expect(counters[2 + (q ^ 1)]).toBe(0);
          expect(args.shadow[0] * args.shadow[1] * PTWF_WORKGROUP).toBeGreaterThanOrEqual(shadowNow.length);
          expect(counters[2 + q]).toBe(shadowNow.length); // shadow(d) still reads it
        }
        expect(seenAlive.size).toBe(paths);
      }
    }
  });
});
