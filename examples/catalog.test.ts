import { describe, expect, it } from 'vitest';
import { EXAMPLES } from './catalog.ts';

// The three principles (catalog.ts header): every example says what it shows and what to try, and a performance or quality example
// names the "bad" version to compare against.
describe('example briefs', () => {
  for (const e of EXAMPLES) {
    it(`${e.slug} says what it shows, what to try and what to compare`, () => {
      expect(e.brief.shows.length).toBeGreaterThan(40);
      expect(e.brief.tryThis.length).toBeGreaterThanOrEqual(1);
      expect(e.brief.tryThis.length).toBeLessThanOrEqual(3);
      if (e.brief.kind === 'performance' || e.brief.kind === 'quality') expect(e.brief.compare, `${e.slug}: a ${e.brief.kind} example needs a baseline`).toBeTruthy();
    });
  }
  it('slugs and paths are unique', () => {
    expect(new Set(EXAMPLES.map((e) => e.slug)).size).toBe(EXAMPLES.length);
    expect(new Set(EXAMPLES.map((e) => e.path)).size).toBe(EXAMPLES.length);
  });
});
