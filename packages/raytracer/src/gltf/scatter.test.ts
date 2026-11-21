import { describe, expect, it } from 'vitest';
import { gltfMaterialToOpenPbr } from './material.ts';

const solid = { transmissionFactor: 1 };
const volume = { thicknessFactor: 1, attenuationDistance: 0.5, attenuationColor: [0.9, 0.7, 0.5] };

describe('KHR_materials_scatter mapping', () => {
  it('volumetric mode carries strength, multi-scatter colour and anisotropy', () => {
    const r = gltfMaterialToOpenPbr({ extensions: { KHR_materials_transmission: solid, KHR_materials_volume: volume, KHR_materials_scatter: { scatterStrengthFactor: 1, multiscatterColorFactor: [0.81, 0.8, 0.5], scatterAnisotropy: 0.2 } } });
    expect(r.params.volumeScatter).toEqual({ strength: 1, multiscatterColor: [0.81, 0.8, 0.5], anisotropy: 0.2 });
    expect(r.unsupported).toEqual([]);
  });
  it('defaults: colour white, anisotropy 0; strength 0 or absent extension: no field', () => {
    const r = gltfMaterialToOpenPbr({ extensions: { KHR_materials_transmission: solid, KHR_materials_volume: volume, KHR_materials_scatter: { scatterStrengthFactor: 0.5 } } });
    expect(r.params.volumeScatter).toEqual({ strength: 0.5, multiscatterColor: [1, 1, 1], anisotropy: 0 });
    expect(gltfMaterialToOpenPbr({ extensions: { KHR_materials_transmission: solid, KHR_materials_volume: volume, KHR_materials_scatter: {} } }).params.volumeScatter).toBeUndefined();
    const plain = gltfMaterialToOpenPbr({ extensions: { KHR_materials_transmission: solid, KHR_materials_volume: volume } });
    expect('volumeScatter' in plain.params).toBe(false);
  });
  it('thin-walled mode is not mapped to a medium', () => {
    const r = gltfMaterialToOpenPbr({ extensions: { KHR_materials_transmission: solid, KHR_materials_scatter: { scatterStrengthFactor: 1 } } });
    expect(r.params.volumeScatter).toBeUndefined();
    expect(r.notes.some((n) => n.includes('thin-walled mode'))).toBe(true);
  });
});
