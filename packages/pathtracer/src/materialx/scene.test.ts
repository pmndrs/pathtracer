import { describe, expect, it } from 'vitest';
import { buildSceneCopy, type RtMaterialInput, type SceneCopyInput } from '@pmndrs/raytracer';
import { buildMaterialX, mtlxAddToScene, mtlxKernelOptions } from './scene.ts';
import { mtlxSyntheticImage } from './synthetic.ts';

const head = '<?xml version="1.0"?><materialx version="1.39" colorspace="lin_rec709">';
const material = '<surfacematerial name="M" type="material"><input name="surfaceshader" type="surfaceshader" nodename="S"/></surfacematerial>';

const CONSTANT = `${head}<open_pbr_surface name="S" type="surfaceshader">
  <input name="base_color" type="color3" value="0.7, 0.35, 0.15"/><input name="specular_roughness" type="float" value="0.35"/>
  <input name="coat_weight" type="float" value="0.6"/></open_pbr_surface>${material}</materialx>`;

/** uv-driven colour from an image, position-driven metalness. */
const VARYING = `${head}<nodegraph name="NG">
  <texcoord name="uv" type="vector2"/>
  <image name="col" type="color3"><input name="file" type="filename" value="gen:checker4" colorspace="srgb_texture"/><input name="texcoord" type="vector2" nodename="uv"/></image>
  <position name="p" type="vector3"/>
  <separate3 name="ps" type="multioutput"><input name="in" type="vector3" nodename="p"/></separate3>
  <smoothstep name="m" type="float"><input name="in" type="float" nodename="ps" output="outx"/><input name="low" type="float" value="-0.1"/><input name="high" type="float" value="0.1"/></smoothstep>
  <output name="out_color" type="color3" nodename="col"/><output name="out_metal" type="float" nodename="m"/>
</nodegraph>
<open_pbr_surface name="S" type="surfaceshader"><input name="base_color" type="color3" nodegraph="NG" output="out_color"/><input name="base_metalness" type="float" nodegraph="NG" output="out_metal"/></open_pbr_surface>
${material}</materialx>`;

const quad = (): SceneCopyInput['meshes'][number] => ({ positions: new Float32Array([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0]), indices: [0, 1, 2, 0, 2, 3], uvs: [0, 1, 1, 1, 1, 0, 0, 0] });

describe('buildMaterialX', () => {
  it('folds a constant document into one record with no textures', () => {
    const built = buildMaterialX(CONSTANT);
    expect(built.mode).toBe('constants');
    expect(built.textures).toEqual([]);
    expect(built.material.baseColor?.map((v) => Math.fround(v))).toEqual([0.7, 0.35, 0.15].map(Math.fround));
    expect(built.material.coatWeight).toBeCloseTo(0.6, 6);
    expect(mtlxKernelOptions([{ material: 0, textureOffset: 0, built }]).integrator).toBeUndefined();
  });

  it('refuses constants for a varying document and picks codegen by default', () => {
    expect(() => buildMaterialX(VARYING, { mode: 'constants' })).toThrow(/varies/);
    expect(buildMaterialX(VARYING, { images: mtlxSyntheticImage }).mode).toBe('codegen');
  });

  it('bakes uv-driven inputs into textures the record points at', () => {
    const built = buildMaterialX(VARYING, { mode: 'bake', resolution: 32, images: mtlxSyntheticImage });
    expect(built.textures.length).toBeGreaterThan(0);
    expect(built.textures[built.material.baseColorTexture!]).toMatchObject({ width: 32, height: 32, colorSpace: 'srgb' });
    expect(built.notes.some((n) => /position|geometr/i.test(n))).toBe(true); // the position-driven metalness can only be approximated
  });

  it('places a codegen material in a scene with offset texture slots and generated kernel options', () => {
    const built = buildMaterialX(VARYING, { mode: 'codegen', images: mtlxSyntheticImage });
    expect(built.images).toHaveLength(1);
    const ground = { width: 1, height: 1, data: new Uint8Array(4).fill(200), colorSpace: 'srgb' as const };
    const input: SceneCopyInput & { materials: RtMaterialInput[] } = {
      meshes: [quad(), quad()],
      instances: [0, 1].map((i) => ({ meshIndex: i, instanceIndex: i, objectToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], material: i })),
      materials: [{ baseColorTexture: 0 }],
      textures: [ground],
    };
    const use = mtlxAddToScene(input, built);
    expect(use).toMatchObject({ material: 1, textureOffset: 1 });
    expect(input.textures).toHaveLength(2);
    const k = mtlxKernelOptions([use]);
    expect(k.integrator?.materialx.wgsl).toContain('ptMtlxApply');
    // The program can switch metalness between 0 and 1: the kernel needs the metal / dielectric mix.
    expect(k.materialProbes).toContainEqual({ metalness: 0.5 });
    expect(() => buildSceneCopy(input)).not.toThrow();
  });
});
