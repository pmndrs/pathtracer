import { describe, expect, it } from 'vitest';
import { abbaOrder, median, medianMetrics, pairedRatio } from './stats.ts';

describe('stats', () => {
  it('takes the median of odd and even counts, and of nothing', () => {
    expect(median([3, 1, 2])).toBe(2);
    expect(median([4, 1, 3, 2])).toBe(2.5);
    expect(median([7])).toBe(7);
    expect(median([])).toBeNaN();
  });

  it('orders A/B runs interleaved', () => {
    expect(abbaOrder('A', 'B', 1)).toEqual(['A', 'B']);
    expect(abbaOrder('A', 'B', 2)).toEqual(['A', 'B', 'B', 'A']);
    expect(abbaOrder('A', 'B', 4)).toEqual(['A', 'B', 'B', 'A', 'A', 'B', 'B', 'A']);
    const six = abbaOrder('A', 'B', 3);
    expect(six.filter((x) => x === 'A')).toHaveLength(3);
    expect(six.filter((x) => x === 'B')).toHaveLength(3);
  });

  it('takes the median per metric, ignoring runs that lack it', () => {
    expect(medianMetrics([{ a: 1, b: 10 }, { a: 3 }, { a: 2, b: 30 }])).toEqual({ a: 2, b: 20 });
  });

  it('makes paired ratios', () => {
    expect(pairedRatio(2, 3)).toBe(1.5);
    expect(pairedRatio(0, 3)).toBeNaN();
  });
});
