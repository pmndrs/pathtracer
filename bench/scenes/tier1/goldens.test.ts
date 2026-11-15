import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { GOLDEN_DIR, goldenText, tier1Models } from './goldens.ts';

const models = tier1Models();

describe('Tier 1 golden OpenPBR tables', () => {
  it('lists every section 1.1 model', () => {
    expect(models.map((m) => m.name)).toHaveLength(18);
  });
  for (const model of models) {
    const path = join(GOLDEN_DIR, `${model.name}.json`);
    describe(model.name, () => {
      it('has a golden pinned to the manifest sha256, one line per material', () => {
        const text = readFileSync(path, 'utf8');
        const g = JSON.parse(text) as { header: { sourceSha256: string; materialCount: number; classCounts: Record<string, number> }; materials: { i: number; class: string; openpbr: { transmissionWeight?: number } }[] };
        expect(g.header.sourceSha256).toBe(model.sha256);
        expect(g.materials).toHaveLength(g.header.materialCount);
        expect(Object.values(g.header.classCounts).reduce((a, b) => a + b, 0)).toBe(g.materials.length);
        g.materials.forEach((m, i) => expect(m.i).toBe(i));
        expect(text.length).toBeLessThan(80_000);
      });
      // The downloads are git-ignored (node bench/assets/fetch.ts): the recomputation runs where they exist.
      it.skipIf(!existsSync(model.file))('equals the table computed from the downloaded model', () => {
        expect(goldenText(model)).toBe(readFileSync(path, 'utf8'));
      });
    });
  }
});
