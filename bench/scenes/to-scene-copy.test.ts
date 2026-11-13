import { describe, expect, it } from 'vitest';
import { readLight, readLightIndexOfHit, readMaterial, SCENE_NONE, type RtSunLight, type RtTriangleLight } from '@pmndrs/raytracer';
import { meshToWorld, surfaceArea, tessellateQuad } from './geometry.ts';
import { sponzaWhite } from './tier2/sponza-white.ts';
import { cornell, quadLight, tier0Scenes } from './tier0/index.ts';
import { INSTANCE_MASK_EMITTER, INSTANCE_MASK_PLAIN, buildSceneCopyFromDesc, materialFromDesc, sceneDescToCopyInput } from './to-scene-copy.ts';

const Y = (r: number, g: number, b: number) => 0.2126 * r + 0.7152 * g + 0.0722 * b;

describe('SceneDesc -> scene copy', () => {
  it('builds for every Tier 0 scene, with consistent counts', () => {
    for (const scene of tier0Scenes) {
      const { copy, input } = buildSceneCopyFromDesc(scene);
      const what = scene.name;
      expect(input.meshes, what).toHaveLength(scene.meshes.length);
      expect(input.instances, what).toHaveLength(scene.meshes.length);
      expect(input.materialNames, what).toHaveLength(input.materials.length);
      // Triangle lights = triangles of the emissive meshes, nothing else.
      const emissiveTriangles = scene.meshes.reduce((n, mesh, i) => n + (mesh.emission ? (input.meshes[i]!.indices!.length / 3) : 0), 0);
      expect(copy.triangleLightCount, what).toBe(emissiveTriangles);
      expect(copy.sunLight !== SCENE_NONE, what).toBe(!!scene.lights?.length);
      expect(copy.environmentLight !== SCENE_NONE, what).toBe(!!scene.environment);
      const total = copy.lights.length;
      expect(total, what).toBe(emissiveTriangles + (scene.lights?.length ?? 0) + (scene.environment ? 1 : 0));
      if (total > 0) {
        let sum = 0;
        for (let i = 0; i < total; i++) sum += readLight(copy, i).pmf;
        expect(sum, what).toBeCloseTo(1, 5);
      }
      // Every instance's material resolves to the right bench material and every mesh's emission is a light only when asked for.
      scene.meshes.forEach((mesh, i) => {
        const id = input.instances[i]!.material!;
        expect(input.materialNames[id], what).toContain(mesh.material);
        const material = readMaterial(copy, id);
        expect(material.emissionLuminance > 0, `${what}/${mesh.name}`).toBe(!!mesh.emission);
        expect(input.instances[i]!.mask).toBe(mesh.emission ? INSTANCE_MASK_EMITTER : INSTANCE_MASK_PLAIN);
      });
      expect(copy.byteLength, what).toBeGreaterThan(0);
      expect(copy.fingerprint, what).toMatch(/^[0-9a-f]{16}$/);
    }
  });

  it('maps each bench material type to its OpenPBR-subset record', () => {
    const diffuse = materialFromDesc({ type: 'diffuse', reflectance: [0.2, 0.4, 0.6] });
    expect(diffuse).toMatchObject({ baseColor: [0.2, 0.4, 0.6], baseWeight: 1, metalness: 0, specularWeight: 0 });
    const conductor = materialFromDesc({ type: 'conductor', eta: [0.2, 0.924, 1.102], k: [3.912, 2.452, 2.142], roughness: 0.3 });
    expect(conductor).toMatchObject({ conductorEtaK: true, conductorEta: [0.2, 0.924, 1.102], conductorK: [3.912, 2.452, 2.142], metalness: 1, specularRoughness: 0.3 });
    const dielectric = materialFromDesc({ type: 'dielectric', ior: 1.5, roughness: 0 });
    expect(dielectric).toMatchObject({ transmissionWeight: 1, transmissionDepth: 0, baseWeight: 0, ior: 1.5, specularRoughness: 0, specularWeight: 1 });
    expect(materialFromDesc({ type: 'black' })).toMatchObject({ baseWeight: 0, specularWeight: 0 });
  });

  it('copper (env-sphere) survives the packed record: eta and k within half precision', () => {
    const scene = tier0Scenes.find((s) => s.name === 'env-sphere')!;
    const { copy, input } = buildSceneCopyFromDesc(scene);
    const id = input.materialNames.indexOf('copper');
    const m = readMaterial(copy, id);
    expect(m.conductorEtaK).toBe(true);
    [0.2, 0.924, 1.102].forEach((v, k) => expect(Math.abs(m.conductorEta[k]! - v) / v).toBeLessThan(2 ** -11));
    [3.912, 2.452, 2.142].forEach((v, k) => expect(Math.abs(m.conductorK[k]! - v) / v).toBeLessThan(2 ** -11));
    expect(m.specularRoughness).toBeCloseTo(0.3, 4);
    // hdri: the map slot is reserved, the constant radiance is the (black) default.
    const env = copy.lights[copy.environmentLight]!;
    expect(env).toMatchObject({ kind: 'environment', mapSlot: 0, radiance: [0, 0, 0] });
  });

  it('cornell light: 0.13 x 0.105 m facing down, power pi * area * Y(L), the only light', () => {
    const { copy } = buildSceneCopyFromDesc(cornell);
    const lights = copy.lights as RtTriangleLight[];
    expect(lights).toHaveLength(2);
    const area = 0.13 * 0.105;
    expect(lights[0]!.area + lights[1]!.area).toBeCloseTo(area, 6);
    expect(surfaceArea(meshToWorld(cornell.meshes.find((m) => m.emission)!))).toBeCloseTo(area, 6);
    for (const l of lights) {
      expect(l.normal[1]).toBeCloseTo(-1, 6);
      expect(l.p0[1]).toBeCloseTo(0.5487, 6);
      expect(l.radiance).toEqual([Math.fround(18.4), 14, 6.75]);
      expect(l.twoSided).toBe(false);
    }
    const y = Y(18.4, 14, 6.75);
    expect(y).toBeCloseTo(14.41199, 5);
    expect(lights[0]!.power + lights[1]!.power).toBeCloseTo(Math.PI * area * y, 6);
    expect(copy.totalPower).toBeCloseTo(Math.PI * area * y, 6);
    expect(readLight(copy, 0).pmf + readLight(copy, 1).pmf).toBeCloseTo(1, 6);
    // The emissive mesh is instance 5; both of its triangles are lights, nothing else is.
    expect([0, 1].map((t) => readLightIndexOfHit(copy, 5, t))).toEqual([0, 1]);
    for (const instance of [0, 1, 2, 3, 4]) expect(readLightIndexOfHit(copy, instance, 0)).toBe(SCENE_NONE);
    // Scene radius: the room is 0.556 x 0.5488 x 0.5592 m.
    expect(copy.bounds.radius).toBeCloseTo(0.5 * Math.hypot(0.556, 0.5488, 0.5592), 4);
  });

  it('quad-light: a 1 m^2 emitter at y = 1 facing down, radiance 10, power 10 pi', () => {
    const { copy } = buildSceneCopyFromDesc(quadLight);
    const lights = copy.lights as RtTriangleLight[];
    expect(lights).toHaveLength(2);
    expect(lights[0]!.area).toBeCloseTo(0.5, 6);
    expect(lights[0]!.area + lights[1]!.area).toBeCloseTo(surfaceArea(tessellateQuad([1, 1])), 6);
    for (const l of lights) {
      expect(l.normal[0]).toBeCloseTo(0, 6);
      expect(l.normal[1]).toBeCloseTo(-1, 6);
      expect(l.normal[2]).toBeCloseTo(0, 6);
      expect(l.p0[1]).toBeCloseTo(1, 6);
      expect(l.radiance).toEqual([10, 10, 10]);
    }
    expect(copy.totalPower).toBeCloseTo(10 * Math.PI, 5);
    expect(lights[0]!.power).toBeCloseTo(5 * Math.PI, 5);
    expect(readLight(copy, 0).pmf).toBeCloseTo(0.5, 6);
  });

  it('furnace scenes: the environment is the only light (probability 1), the sphere is not emissive', () => {
    for (const scene of tier0Scenes.filter((s) => s.name.startsWith('furnace'))) {
      const { copy } = buildSceneCopyFromDesc(scene);
      expect(copy.lights).toHaveLength(1);
      expect(copy.lights[0]).toMatchObject({ kind: 'environment', radiance: [1, 1, 1] });
      expect(readLight(copy, 0).pmf).toBe(1);
    }
  });

  it('sponza-white (file geometry through resolveFile): sun toward the sun, uniform sky', () => {
    expect(() => sceneDescToCopyInput(sponzaWhite)).toThrow(/resolveFile/);
    const { copy } = buildSceneCopyFromDesc(sponzaWhite, { resolveFile: () => tessellateQuad([20, 20]) });
    expect(copy.triangleLightCount).toBe(0);
    const sun = copy.lights[copy.sunLight] as RtSunLight;
    const d = sponzaWhite.lights![0]!.direction;
    d.forEach((v, k) => expect(sun.direction[k]!).toBeCloseTo(-v, 6));
    expect(sun.irradiance).toEqual([50, 46, 40]);
    expect(sun.angularRadius).toBeCloseTo(0.00465, 8);
    expect(sun.solidAngle).toBeCloseTo(2 * Math.PI * (1 - Math.cos(0.00465)), 9);
    expect(copy.lights[copy.environmentLight]).toMatchObject({ radiance: [2, Math.fround(2.3), 3] });
    // Sun and environment share the probability by power: 50/46/40 over a 20 x 20 plane versus the sky's 4 pi pi R^2 weight.
    expect(readLight(copy, copy.sunLight).pmf + readLight(copy, copy.environmentLight).pmf).toBeCloseTo(1, 6);
  });

  it('rejects unknown materials and a second sun', () => {
    expect(() => sceneDescToCopyInput({ ...cornell, meshes: [{ ...cornell.meshes[0]!, material: 'nope' }] })).toThrow(RangeError);
    expect(() => sceneDescToCopyInput({ ...sponzaWhite, lights: [sponzaWhite.lights![0]!, sponzaWhite.lights![0]!] }, { resolveFile: () => tessellateQuad([1, 1]) })).toThrow(RangeError);
  });
});
