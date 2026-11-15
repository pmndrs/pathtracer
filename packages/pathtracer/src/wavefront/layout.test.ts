import { describe, expect, it } from 'vitest';
import { PTWF_INIT, PTWF_LIVE, PTWF_START_ITER, PTWF_STRIDE, PTWF_WORKGROUP, ptwfArgsStart, ptwfArgsStep, ptwfBlockSortPositions, ptwfBufferBytes, ptwfDispatchesPerSample, ptwfGroups, ptwfHasNext, ptwfHitSortKey, ptwfLodClass, ptwfOrderedAppend, ptwfPackDepth, ptwfQueueOffsets, ptwfRefillDecision, ptwfRecordsPerSample, ptwfRaySortKey, ptwfSampleOf, ptwfSlot, ptwfStateBytes, ptwfSubgroupAppend, ptwfUnpackDepth } from './layout.ts';

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
    for (const depth of [0, 1, 7, 1000, 0x1fffffff]) {
      for (const specular of [false, true]) {
        const word = ptwfPackDepth(depth, specular);
        expect(word).toBeGreaterThanOrEqual(0);
        expect(word).toBeLessThan(2 ** 32);
        expect(ptwfUnpackDepth(word)).toEqual({ depth, specular, live: true });
      }
    }
    expect(ptwfPackDepth(5, true)).toBe(0xa0000005);
    expect(ptwfPackDepth(5, false) & PTWF_LIVE).toBe(PTWF_LIVE); // a stored path is always live; a zero word is a dormant slot
    expect(ptwfUnpackDepth(0).live).toBe(false);
    expect(() => ptwfPackDepth(0x20000000, false)).toThrow(RangeError);
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

  /**
   * The refill protocol (P3.2) against a CPU model of the camera / shade / args kernels: slots = layers x pixels, each slot runs the samples
   * `layer, layer + K, ...` of its pixel one after the other (inline refill), the queue counters and slot state persist across encodes, and the
   * encode's iteration count need not finish anything. Whatever the path lengths, the chunking and the growing target, every pixel must end up
   * with every sample index 0..T-1 started exactly once and no slot may run two paths at a time.
   */
  it('refill: every pixel gets every sample exactly once, across encodes and chunks (simulated)', () => {
    let seed = 987654321;
    const rand = () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 2 ** 32);
    for (const [pixels, layers, batch] of [[1, 1, false], [7, 1, false], [7, 3, false], [64, 2, false], [130, 4, false], [33, 8, false], [1, 1, true], [7, 3, true], [64, 2, true], [130, 4, true], [33, 8, true]] as const) {
      const slots = pixels * layers;
      let idle = 0; // batch refill: slots that finished a path and still have a sample (the idle counter)
      const started = new Uint32Array(slots);
      const live = new Uint8Array(slots);
      const remaining = new Uint32Array(slots); // steps the current path of the slot still takes
      const ran: Set<number>[] = Array.from({ length: pixels }, () => new Set());
      const counters = new Uint32Array(4);
      const queue: number[][] = [[], []];
      let iterBase = 0;
      let target = 0;
      const pathLength = () => 1 + Math.floor(rand() * rand() * 40); // a heavy-tailed mix of short and long paths
      const start = (slot: number, q: number) => {
        const sample = ptwfSampleOf(started[slot] as number, Math.floor(slot / pixels), layers);
        const pixel = slot % pixels;
        expect(ran[pixel]!.has(sample), `sample ${sample} of pixel ${pixel} started twice`).toBe(false);
        ran[pixel]!.add(sample);
        started[slot]!++;
        live[slot] = 1;
        remaining[slot] = pathLength();
        queue[q]!.push(slot);
        counters[q]!++;
      };
      const encode = (passes: number, iterations: number, withStart: boolean) => {
        if (withStart) {
          target += passes;
          const cur = iterBase & 1;
          for (let slot = 0; slot < slots; slot++) if (!live[slot] && ptwfHasNext(started[slot] as number, Math.floor(slot / pixels), layers, target)) start(slot, cur);
          idle = 0; // the start pass restarted every idle slot (the args kernel zeroes the counter on a start record)
          expect(ptwfArgsStart(counters, PTWF_START_ITER + cur).main[0] * PTWF_WORKGROUP * ptwfArgsStart(counters, PTWF_START_ITER + cur).main[1]).toBeGreaterThanOrEqual(counters[cur] as number);
        }
        for (let i = 0; i < iterations; i++) {
          const d = iterBase + i;
          const q = d & 1;
          expect(counters[q]).toBe(queue[q]!.length);
          expect(counters[2 + q]).toBe(0); // shadow counter of this iteration was zeroed by the previous args
          const read = [...queue[q]!];
          expect(new Set(read).size).toBe(read.length); // one path per slot
          for (const slot of read) {
            expect(live[slot]).toBe(1);
            if (rand() < 0.5) counters[2 + q]!++; // a shadow ray
            remaining[slot]!--;
            if (remaining[slot] === 0) {
              live[slot] = 0;
              if (ptwfHasNext(started[slot] as number, Math.floor(slot / pixels), layers, target)) {
                if (batch) idle++;
                else start(slot, q ^ 1); // inline refill
              }
            } else {
              queue[q ^ 1]!.push(slot);
              counters[q ^ 1]!++;
            }
          }
          if (batch) {
            // args decides, the camera kernel scans the film in slot order and starts every slot that is not live and has a sample
            const decision = ptwfRefillDecision(counters[q ^ 1] as number, idle, slots, 0.25, pixels, 1, layers);
            let scanned = 0;
            if (decision.refill > 0) {
              expect(decision.refillGrid).toEqual([Math.ceil(pixels / 8), 1, layers]);
              for (let slot = 0; slot < slots; slot++) if (!live[slot] && ptwfHasNext(started[slot] as number, Math.floor(slot / pixels), layers, target)) { start(slot, q ^ 1); scanned++; }
              expect(scanned).toBe(idle); // the scan finds exactly the counted slots, so the main args (live + idle) cover the queue
              idle = 0;
            } else expect(decision.refillGrid).toEqual([0, 0, 0]);
            expect(decision.main[0] * decision.main[1] * PTWF_WORKGROUP).toBeGreaterThanOrEqual(counters[q ^ 1] as number);
          }
          const args = ptwfArgsStep(counters, d, 0);
          expect(args.main[0] * args.main[1] * PTWF_WORKGROUP).toBeGreaterThanOrEqual(counters[q ^ 1] as number);
          queue[q]!.length = 0; // args zeroed the queue shade read
          expect(counters[q]).toBe(0);
        }
        iterBase += iterations;
      };
      // first encode: far too few iterations to finish; later encodes grow the target while paths are still in flight
      encode(2, 3, true);
      encode(3, 5, true);
      encode(1, 0, true);
      expect(target).toBe(6);
      let guard = 0;
      while (counters[0]! + counters[1]! + idle > 0) {
        encode(0, 7, false); // finish(): chunks without a start pass
        expect(++guard).toBeLessThan(10000);
      }
      encode(4, 1, true); // more samples after a complete render: dormant slots restart
      while (counters[0]! + counters[1]! + idle > 0) encode(0, 9, false);
      for (let pixel = 0; pixel < pixels; pixel++) {
        expect([...ran[pixel]!].sort((a, b) => a - b)).toEqual(Array.from({ length: target }, (_, i) => i));
      }
      for (let slot = 0; slot < slots; slot++) expect(live[slot]).toBe(0);
    }
  });
});

describe('wavefront scheduling twins (P3.3, P3.4, P3.10, R1b)', () => {
  it('subgroup append: one reservation per subgroup, lanes keep their order, every wanting lane gets a distinct slot', () => {
    const wants = [true, false, true, true, false, false, true, false];
    const r = ptwfSubgroupAppend(wants, 100);
    expect(r.slots).toEqual([100, -1, 101, 102, -1, -1, 103, -1]);
    expect(r.counter).toBe(104);
    expect(ptwfSubgroupAppend(wants.map(() => false), 7)).toEqual({ slots: wants.map(() => -1), counter: 7 });
    // two subgroups append back to back: ranges are disjoint and dense
    const a = ptwfSubgroupAppend([true, true, false, true], 0);
    const b = ptwfSubgroupAppend([true, false, true, true], a.counter);
    expect([...a.slots, ...b.slots].filter((s) => s >= 0).sort((x, y) => x - y)).toEqual([0, 1, 2, 3, 4, 5]);
  });

  it('block sort: a stable permutation inside the block, ascending keys; the keys of the hit and ray sorts', () => {
    const keys = Array.from({ length: PTWF_WORKGROUP }, (_, i) => (i * 7 + (i >> 2)) % 5);
    const pos = ptwfBlockSortPositions(keys);
    expect([...pos].sort((a, b) => a - b)).toEqual(keys.map((_, i) => i)); // a permutation
    const sorted = new Array<number>(keys.length);
    const origin = new Array<number>(keys.length);
    pos.forEach((p, i) => {
      sorted[p] = keys[i] as number;
      origin[p] = i;
    });
    for (let i = 1; i < sorted.length; i++) {
      expect(sorted[i]).toBeGreaterThanOrEqual(sorted[i - 1] as number);
      if (sorted[i] === sorted[i - 1]) expect(origin[i]).toBeGreaterThan(origin[i - 1] as number); // stable
    }
    expect(ptwfBlockSortPositions([])).toEqual([]);
    expect(ptwfHitSortKey(-1, 7)).toBe(0);
    expect(ptwfHitSortKey(2, 0)).toBe(1);
    expect(ptwfHitSortKey(2, 15)).toBe(1);
    expect(ptwfHitSortKey(2, 14)).toBe(15);
    expect(ptwfRaySortKey([1, 1, 1])).toBe(0);
    expect(ptwfRaySortKey([-1, 1, -1])).toBe(5);
    expect(ptwfRaySortKey([-1, -1, -1])).toBe(7);
  });

  it('ordered restart: restarted slots join the queue in slot order, contiguous per block (inline appends scatter them)', () => {
    const flags = Array.from({ length: PTWF_WORKGROUP }, (_, i) => i % 3 === 0);
    const r = ptwfOrderedAppend(flags, 5, 10);
    expect(r.entries).toEqual(flags.map((f, i) => (f ? 5 * PTWF_WORKGROUP + i : -1)).filter((s) => s >= 0));
    expect(r.counter).toBe(10 + r.entries.length);
    expect([...r.entries]).toEqual([...r.entries].sort((a, b) => a - b));
    expect(ptwfOrderedAppend(flags.map(() => false), 0, 3)).toEqual({ entries: [], counter: 3 });
  });

  it('R1b: the two shade kernels partition the depths, so a path is shaded by exactly one of them per iteration', () => {
    for (let depth = 0; depth < 8; depth++) expect([ptwfLodClass(depth, 1), ptwfLodClass(depth, 3)]).toEqual([depth < 1 ? 'first' : 'second', depth < 3 ? 'first' : 'second']);
  });
});
