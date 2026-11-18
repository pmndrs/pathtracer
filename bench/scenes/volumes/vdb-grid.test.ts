import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync, rmSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { getScene } from '../index.ts';
import { GENERATED_DIR } from './vdb-grid.ts';

const webvdb = process.env.WEBVDB_DIR ?? join(homedir(), 'Developer', 'WebVDB');
const smoke = join(webvdb, 'fixtures', 'vdb-samples', 'smoke.vdb');
const tools = join(webvdb, 'packages', 'vdb-web-tools', 'dist', 'index.js');
const have = existsSync(smoke) && existsSync(tools);

describe.skipIf(!have)("WebVDB's smoke.vdb as a dense density (P5.v7)", () => {
  it('converts (in a plain Node child: vitest cannot import the sibling checkout) to the grid the vol-vdb-smoke scene declares: peak 1, a zero border, the source mass', () => {
    const script = fileURLToPath(new URL('./vdb-grid.ts', import.meta.url));
    const meta = JSON.parse(execFileSync(process.execPath, [script, smoke, 'vdb-smoke-test'], { encoding: 'utf8' })) as { resolution: number[]; bounds: number[][]; peakBeforeNormalisation: number; source: { activeVoxels: number } };
    const raw = readFileSync(join(GENERATED_DIR, 'vdb-smoke-test.f32'));
    const density = new Float32Array(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
    rmSync(join(GENERATED_DIR, 'vdb-smoke-test.f32'));
    rmSync(join(GENERATED_DIR, 'vdb-smoke-test.json'));
    const med = getScene('vol-vdb-smoke').media!.plume!;
    expect(med.type).toBe('grid');
    if (med.type !== 'grid') return;
    expect(meta.resolution).toEqual([...med.resolution]);
    // the scene's box is the grid's bounds recentred on their centre (the mesh transform places it)
    for (let a = 0; a < 3; a++) expect((meta.bounds[1]![a]! - meta.bounds[0]![a]!) / 2).toBeCloseTo(med.bounds[1][a]!, 6);
    const [nx, ny, nz] = meta.resolution as [number, number, number];
    expect(density.length).toBe(nx * ny * nz);
    expect(density.reduce((a, v) => Math.max(a, v), 0)).toBe(1);
    let border = 0, sum = 0;
    for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const v = density[i + nx * (j + ny * k)]!;
      sum += v;
      if (Math.min(i, j, k, nx - 1 - i, ny - 1 - j, nz - 1 - k) < 2) border += v;
    }
    expect(border).toBe(0);
    expect(sum).toBeGreaterThan(0);
    expect(meta.source.activeVoxels).toBe(1049275);
    expect(meta.peakBeforeNormalisation).toBeCloseTo(5.38671875, 6);
  });
});
