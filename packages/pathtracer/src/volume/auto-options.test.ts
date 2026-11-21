import { describe, expect, it } from 'vitest';
import { ptVolumeAutoGrid, ptVolumeKernelBudgetNote } from './auto-options.ts';
import { ATLAS_EMPTY } from './volume.ts';

const MIB = 1024 * 1024;
const limits = { maxTextureDimension3D: 2048, maxBufferSize: 2 ** 31 };
const noFeat = new Set<string>();

function blob(res: [number, number, number], zFrac: number): Float32Array {
  const d = new Float32Array(res[0] * res[1] * res[2]);
  const zEnd = Math.floor(res[2] * zFrac);
  for (let z = 0; z < zEnd; z++) for (let y = 0; y < res[1]; y++) for (let x = 0; x < res[0]; x++) d[x + res[0] * (y + res[1] * z)] = 0.5 + 0.5 * Math.sin(x * 0.1) * Math.cos(y * 0.07);
  return d;
}

describe('ptVolumeAutoGrid', () => {
  it('small dense grid: r32 dense, manual, exact', () => {
    const res: [number, number, number] = [32, 32, 32];
    const d = blob(res, 1);
    const r = ptVolumeAutoGrid({ resolution: res, density: d, limits, features: noFeat });
    expect([r.format, r.layout, r.densityFilter, r.majorant]).toEqual(['r32float', 'dense', 'manual', 'brick']);
    expect(r.prepared!.texels).toBe(d);
    expect(r.prepared!.brick!.values.length).toBeGreaterThan(0);
  });

  it('504x352x616 at ~20% occupancy under 128 MiB: atlas r16', () => {
    const res: [number, number, number] = [504, 352, 616];
    const d = blob(res, 0.2);
    const r = ptVolumeAutoGrid({ resolution: res, density: d, limits, features: noFeat, budgetBytes: 128 * MIB, prepare: false });
    expect([r.format, r.layout]).toEqual(['r16float', 'atlas']);
    expect(r.bytes).toBeLessThanOrEqual(128 * MIB);
    console.log('big grid', r.dims, (r.bytes / MIB).toFixed(1), 'MiB, occupancy', r.occupancy.toFixed(3), r.reasons);
    // dense r32 / r16 were rejected for the budget
    expect(r.candidates.filter((c) => c.layout === 'dense' && c.format !== 'r8unorm').every((c) => !c.fits)).toBe(true);
  });

  it('prepared atlas r16 is consistent (small sparse grid, tight budget)', () => {
    const res: [number, number, number] = [64, 64, 64];
    const d = blob(res, 0.25);
    const r = ptVolumeAutoGrid({ resolution: res, density: d, limits, features: noFeat, budgetBytes: 500_000, accuracy: 'preview' });
    expect([r.format, r.layout]).toEqual(['r16float', 'atlas']);
    const p = r.prepared!;
    expect(p.texels).toBeInstanceOf(Uint16Array);
    expect(p.atlas!.indirection.some((v) => v === ATLAS_EMPTY)).toBe(true);
    expect(p.densityMax).toBeGreaterThan(0);
    expect(r.bytes).toBeLessThanOrEqual(500_000);
  });

  it('r8 is the last resort and says so', () => {
    const res: [number, number, number] = [64, 64, 64];
    const r = ptVolumeAutoGrid({ resolution: res, density: blob(res, 1), limits, features: noFeat, budgetBytes: 300_000 });
    expect(r.format).toBe('r8unorm');
    expect(r.reasons.join(' ')).toMatch(/quantises/);
    expect(r.prepared!.scale).toBeGreaterThan(0);
  });

  it('a grid over maxTextureDimension3D: atlas with a reason, or an error', () => {
    const res: [number, number, number] = [96, 16, 16];
    const d = blob(res, 1);
    const r = ptVolumeAutoGrid({ resolution: res, density: d, limits: { maxTextureDimension3D: 64 }, features: noFeat, prepare: false });
    expect(r.layout).toBe('atlas');
    expect(r.reasons.join(' ')).toMatch(/exceeds maxTextureDimension3D/);
    const dense = blob([200, 200, 8], 1);
    expect(() => ptVolumeAutoGrid({ resolution: [200, 200, 8], density: dense, limits: { maxTextureDimension3D: 16 }, features: noFeat, prepare: false })).toThrow(/maxTextureDimension3D/);
  });

  it('preview: r32 hardware filtering needs float32-filterable; r16 is hardware', () => {
    const res: [number, number, number] = [32, 32, 32];
    const d = blob(res, 1);
    const p = ptVolumeAutoGrid({ resolution: res, density: d, limits, features: noFeat, accuracy: 'preview', prepare: false });
    expect([p.format, p.densityFilter]).toEqual(['r16float', 'hardware']);
    // r32 only appears when memory allows nothing else: force it via the reference path and check the manual rule
    const ref = ptVolumeAutoGrid({ resolution: res, density: d, limits, features: new Set(['float32-filterable']), prepare: false });
    expect([ref.format, ref.densityFilter]).toEqual(['r32float', 'manual']);
  });

  it('kernel budget note', () => {
    expect(ptVolumeKernelBudgetNote(53_000, { backend: 'd3d12' })).toMatch(/wavefront/);
    expect(ptVolumeKernelBudgetNote(53_000, { description: 'NVIDIA GeForce RTX 3060, Direct3D 12' })).toMatch(/DXC/);
    expect(ptVolumeKernelBudgetNote(53_000, { vendor: 'apple', architecture: 'metal-3' })).toBeNull();
    expect(ptVolumeKernelBudgetNote(45_000, { backend: 'd3d12' })).toBeNull();
    expect(ptVolumeKernelBudgetNote(53_000, null)).toBeNull();
  });
});
