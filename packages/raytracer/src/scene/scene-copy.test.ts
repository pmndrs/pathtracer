import { buildTraceScene, createCpuTracer, createMeshBvhBuilder } from '@pmndrs/bvh';
import { describe, expect, it } from 'vitest';
import { SURFACE, SURFACE_TABLE, buildSurfaceTable, readSurfaceFromTable, referenceSurface, type SurfaceHitInput } from '../trace/index.ts';
import { INSTANCE_LIGHT, LIGHT_MAP_MODE, SCENE_COPY_STORAGE_BUFFERS, SCENE_HEADER, SCENE_HEADER_VEC4S, SCENE_NONE } from './constants.ts';
import { aliasProbabilityOf, lightSelectionWeights, trianglePower, type RtSunLight, type RtTriangleLight } from './lights.ts';
import { createMaterial, luminance, unpackMaterial, MATERIAL_RECORD_WORDS } from './material.ts';
import { ContentHasher, buildSceneCopy, readLight, readLightIndexOfHit, readLightPmfOfHit, readMaterial, referenceTriangleArea, type SceneCopyInput } from './scene-copy.ts';

const f32 = Math.fround;

type M16 = number[];
const identity = (): M16 => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
/** Column-major: scale, then rotate about Y, then translate (apply right to left). */
function trs(t: [number, number, number], angle: number, s: [number, number, number]): M16 {
  const c = Math.cos(angle);
  const sn = Math.sin(angle);
  // R_y * S, column-major.
  return [c * s[0], 0, -sn * s[0], 0, 0, s[1], 0, 0, sn * s[2], 0, c * s[2], 0, t[0], t[1], t[2], 1];
}

/** Unit quad in the XZ plane, two triangles, authored normal -Y ((p1 - p0) x (p2 - p0) = (0, -1, 0)). */
const quad = () => ({
  positions: new Float32Array([0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1]),
  indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
  normals: new Float32Array([0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0]),
  uvs: new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]),
});

/** Four triangles in a strip along +X; triangles 0 and 2 are emissive (material 1), 1 and 3 fall back to the instance's (0). */
const strip = () => ({
  positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 2, 0, 0, 2, 0, 1, 3, 0, 0, 3, 0, 1]),
  indices: new Uint32Array([0, 1, 2, 2, 1, 3, 1, 4, 3, 3, 4, 5]),
  triangleMaterials: [1, 0xffff, 1, 0xffff],
});

function fixture(): SceneCopyInput {
  return {
    meshes: [quad(), strip()],
    instances: [
      { meshIndex: 0, instanceIndex: 0, objectToWorld: trs([0, 3, 0], 0.4, [2, 1, 3]), material: 1 }, // emitter, dense, area 6
      { meshIndex: 0, instanceIndex: 1, objectToWorld: trs([5, 3, 0], -0.7, [-1.5, 1, 1]), material: 2 }, // mirrored, two-sided emitter, dense, area 1.5
      { meshIndex: 1, instanceIndex: 3, objectToWorld: trs([0, 0, 5], 0.2, [1, 1, 1]), material: 0 }, // mixed: map
      { meshIndex: 0, instanceIndex: 4, objectToWorld: identity(), material: 1, visible: false }, // hidden emitter
      { meshIndex: 0, instanceIndex: 6, objectToWorld: identity(), material: 0 }, // plain
      { meshIndex: 0, instanceIndex: 7, objectToWorld: identity() }, // no material at all
    ],
    materials: [{ baseColor: [0.5, 0.5, 0.5] }, { emissionColor: [2, 4, 6] }, { emissionColor: [1, 1, 1], emissionTwoSided: true }],
    sun: { direction: [0.3, -0.8, 0.4], irradiance: [50, 46, 40], angularRadius: 0.00465 },
    environment: { radiance: [2, 2.3, 3] },
  };
}

describe('scene copy: sections, header, lights', () => {
  const input = fixture();
  const copy = buildSceneCopy(input);

  it('header counts and offsets agree with the sections', () => {
    const h = (word: number) => copy.table[word] as number;
    expect(SCENE_COPY_STORAGE_BUFFERS).toBe(2);
    expect(copy.storageBuffers).toBe(2);
    expect(h(SCENE_HEADER.instanceSlots)).toBe(8); // largest instanceIndex 7 + 1
    expect(h(SCENE_HEADER.materialCount)).toBe(3);
    expect(h(SCENE_HEADER.triangleLightCount)).toBe(6); // instances 0 and 1 (two triangles each) and the strip's two; the hidden, plain and material-less ones add none
    expect(h(SCENE_HEADER.lightCount)).toBe(6 + 2);
    expect(h(SCENE_HEADER.sunLight)).toBe(6);
    expect(h(SCENE_HEADER.envLight)).toBe(7);
    expect(h(SCENE_HEADER.materialBase)).toBe(copy.sections.materials);
    expect(h(SCENE_HEADER.lightBase)).toBe(copy.sections.lights);
    expect(copy.sections.meshes).toBe(SCENE_HEADER_VEC4S + 8 * SURFACE_TABLE.INSTANCE_VEC4S);
    expect(copy.sections.vec4s * 16).toBe(copy.table.byteLength);
    expect(f32(0)).toBe(0);
    expect(new Float32Array(new Uint32Array([h(SCENE_HEADER.totalPower)]).buffer)[0]).toBe(f32(copy.totalPower));
    expect(new Float32Array(new Uint32Array([h(SCENE_HEADER.sceneRadius)]).buffer)[0]).toBe(f32(copy.bounds.radius));
    expect(copy.byteLength).toBe(copy.table.byteLength + copy.data.byteLength);
  });

  it('materials are the quantised table plus a default record', () => {
    expect(copy.materials).toHaveLength(3);
    expect(readMaterial(copy, 1).emissionColor).toEqual([2, 4, 6]);
    expect(readMaterial(copy, 1).emissionLuminance).toBe(f32(luminance([2, 4, 6])));
    expect(readMaterial(copy, 2).emissionTwoSided).toBe(true);
    expect(readMaterial(copy, 3)).toEqual(copy.defaultMaterial);
    expect(readMaterial(copy, 0xffff)).toEqual(copy.defaultMaterial);
    expect(readMaterial(copy, SCENE_NONE)).toEqual(copy.defaultMaterial);
    expect(copy.defaultMaterial.emissionLuminance).toBe(0);
    for (let i = 0; i < 3; i++) expect(readMaterial(copy, i)).toEqual(copy.materials[i]);
    expect(unpackMaterial(copy.table, copy.sections.materials * 4 + 0 * MATERIAL_RECORD_WORDS)).toEqual(copy.materials[0]);
  });

  it('emissive triangles: instance order, triangle order, hidden and plain instances excluded', () => {
    const tri = copy.lights.slice(0, copy.triangleLightCount) as RtTriangleLight[];
    expect(tri.map((l) => [l.instance, l.triangle, l.material])).toEqual([
      [0, 0, 1],
      [0, 1, 1],
      [1, 0, 2],
      [1, 1, 2],
      [3, 0, 1],
      [3, 2, 1],
    ]);
    expect(copy.triangleLightCount).toBe(6);
  });

  it('area, radiance and power match hand-computed values', () => {
    const tri = copy.lights.slice(0, 6) as RtTriangleLight[];
    // Instance 0: a unit quad scaled 2 x 3 (rotation keeps area): 6 m^2 over 2 triangles. Y(2, 4, 6) = 0.4252 + 2.8608 + 0.4332 = 3.7192.
    expect(tri[0]!.area + tri[1]!.area).toBeCloseTo(6, 5);
    expect(tri[0]!.area).toBeCloseTo(3, 5);
    expect(luminance([2, 4, 6])).toBeCloseTo(3.7192, 10);
    expect(tri[0]!.power + tri[1]!.power).toBeCloseTo(Math.PI * 6 * 3.7192, 4);
    expect(tri[0]!.radiance).toEqual([2, 4, 6]);
    expect(tri[0]!.twoSided).toBe(false);
    // Instance 1: scale 1.5 x 1 (mirror): 1.5 m^2, two-sided white (Y = 1): pi * 1.5 * 2.
    expect(tri[2]!.area + tri[3]!.area).toBeCloseTo(1.5, 5);
    expect(tri[2]!.twoSided).toBe(true);
    expect(tri[2]!.power + tri[3]!.power).toBeCloseTo(Math.PI * 1.5 * 2, 4);
    // Strip triangles: 0.5 m^2 each (right triangles with legs 1).
    expect(tri[4]!.area).toBeCloseTo(0.5, 6);
    expect(tri[5]!.area).toBeCloseTo(0.5, 6);
    // Against an independent area routine.
    for (const l of tri) expect(l.area).toBeCloseTo(referenceTriangleArea(input.meshes[input.instances.find((i) => i.instanceIndex === l.instance)!.meshIndex]!, input.instances.find((i) => i.instanceIndex === l.instance)!, l.triangle), 5);
    for (const l of tri) expect(l.power).toBeCloseTo(trianglePower(l.area, l.radiance, l.twoSided), 10);
  });

  it('the sun points toward the sun; sun and environment power use the scene bounds', () => {
    const sun = copy.lights[copy.sunLight] as RtSunLight;
    const d = [0.3, -0.8, 0.4];
    const len = Math.hypot(...d);
    expect(sun.direction).toEqual(d.map((v) => f32(-v / len)));
    expect(sun.irradiance).toEqual([50, 46, 40]);
    expect(sun.power / (luminance([50, 46, 40]) * Math.PI * copy.bounds.radius ** 2)).toBeCloseTo(1, 6);
    const env = copy.lights[copy.environmentLight]!;
    expect(env.power / (luminance([2, 2.3, 3]) * 4 * Math.PI * Math.PI * copy.bounds.radius ** 2)).toBeCloseTo(1, 6);
    expect(copy.bounds.radius).toBeGreaterThan(4);
  });

  it('light selection probabilities follow the default strategy (pbrt), sum to 1, and the alias table realises them', () => {
    const total = copy.lights.reduce((s, l) => s + l.power, 0);
    expect(copy.totalPower).toBeCloseTo(total, 6);
    const w = lightSelectionWeights(copy.lights, 'pbrt');
    let sum = 0;
    copy.lights.forEach((_light, i) => {
      const read = readLight(copy, i);
      sum += read.pmf;
      expect(read.pmf).toBe(f32(w[i]!));
      expect(aliasProbabilityOf({ prob: Float32Array.of(...copy.lights.map((_, j) => readLight(copy, j).aliasProb)), alias: Uint32Array.of(...copy.lights.map((_, j) => readLight(copy, j).alias)) }, i)).toBeCloseTo(w[i]!, 6);
    });
    expect(sum).toBeCloseTo(1, 6);
  });

  it('lightSelection changes the stored probabilities (and only those): pmf = the strategy weights, total power stays the real total', () => {
    const base = fixture();
    for (const selection of ['power', 'kind', { capped: 0.25 }] as const) {
      const other = buildSceneCopy({ ...base, options: { ...base.options, lightSelection: selection } });
      const w = lightSelectionWeights(other.lights, selection);
      other.lights.forEach((_, i) => expect(readLight(other, i).pmf).toBe(f32(w[i]!)));
      expect(other.totalPower).toBeCloseTo(copy.totalPower, 6);
    }
    const kind = buildSceneCopy({ ...base, options: { ...base.options, lightSelection: 'kind' } });
    // Triangles, sun and environment each take a third.
    expect(readLight(kind, copy.sunLight).pmf).toBeCloseTo(1 / 3, 6);
    expect(readLight(kind, copy.environmentLight).pmf).toBeCloseTo(1 / 3, 6);
  });

  it('light records read back what the builder made', () => {
    copy.lights.forEach((light, i) => {
      const read = readLight(copy, i).light;
      expect(read.kind).toBe(light.kind);
      if (light.kind === 'triangle') expect(read).toMatchObject({ instance: light.instance, triangle: light.triangle, p0: light.p0, edge1: light.edge1, edge2: light.edge2, normal: light.normal, area: light.area, twoSided: light.twoSided });
    });
    expect(() => readLight(copy, 10)).toThrow(RangeError);
  });

  it('triangle -> light map: dense instances, a mixed instance, plain and unknown ones', () => {
    const meta = (instance: number) => {
      const at = (SCENE_HEADER_VEC4S + instance * SURFACE_TABLE.INSTANCE_VEC4S + SURFACE_TABLE.INSTANCE_META) * 4;
      return [copy.table[at + INSTANCE_LIGHT.metaLightMode], copy.table[at + INSTANCE_LIGHT.metaLightBase]];
    };
    expect(meta(0)).toEqual([LIGHT_MAP_MODE.dense, 0]);
    expect(meta(1)).toEqual([LIGHT_MAP_MODE.dense, 2]);
    expect(meta(3)![0]).toBe(LIGHT_MAP_MODE.map);
    expect(meta(4)).toEqual([LIGHT_MAP_MODE.none, SCENE_NONE]);
    expect(meta(6)![0]).toBe(LIGHT_MAP_MODE.none);
    expect(meta(2)![0]).toBe(LIGHT_MAP_MODE.none); // an unused slot

    expect([0, 1].map((t) => readLightIndexOfHit(copy, 0, t))).toEqual([0, 1]);
    expect([0, 1].map((t) => readLightIndexOfHit(copy, 1, t))).toEqual([2, 3]);
    expect([0, 1, 2, 3].map((t) => readLightIndexOfHit(copy, 3, t))).toEqual([4, SCENE_NONE, 5, SCENE_NONE]);
    for (const instance of [2, 4, 6, 7, 8, 100]) expect(readLightIndexOfHit(copy, instance, 0)).toBe(SCENE_NONE);
    expect(readLightPmfOfHit(copy, 3, 1)).toBe(0);
    expect(readLightPmfOfHit(copy, 3, 2)).toBe(readLight(copy, 5).pmf);
    // Every light is found again from the (instance, triangle) a hit reports.
    copy.lights.slice(0, copy.triangleLightCount).forEach((light, i) => {
      const t = light as RtTriangleLight;
      expect(readLightIndexOfHit(copy, t.instance, t.triangle)).toBe(i);
    });
  });
});

describe('emission side matches what bvh reports on a hit (mirrors included)', () => {
  it('the light normal equals the hit normal and the ray from the emitting side sees a front face', async () => {
    const builder = createMeshBvhBuilder({ worker: 'never' });
    const input = fixture();
    const copy = buildSceneCopy(input);
    const meshes = input.meshes.map((m) => ({ positions: m.positions, indices: m.indices as Uint32Array, triangleMaterials: (m as { triangleMaterials?: number[] }).triangleMaterials }));
    const scene = await buildTraceScene(meshes as never, input.instances as never, { build: (geometry) => builder.build(geometry) });
    const tracer = createCpuTracer(scene);
    const lights = copy.lights.slice(0, copy.triangleLightCount) as RtTriangleLight[];
    for (const light of lights) {
      const centroid = [0, 1, 2].map((k) => light.p0[k]! + (light.edge1[k]! + light.edge2[k]!) / 3);
      const from = centroid.map((c, k) => c + light.normal[k]! * 0.75) as [number, number, number];
      const hit = tracer.closest({ origin: from, tMin: 0, direction: light.normal.map((v) => -v) as [number, number, number], tMax: 10 });
      expect(hit, `light ${light.instance}/${light.triangle}`).not.toBeNull();
      expect(hit!.instanceIndex).toBe(light.instance);
      expect(hit!.triangleIndex).toBe(light.triangle);
      expect(hit!.frontFace).toBe(true);
      expect(hit!.material).toBe(light.material);
      light.normal.forEach((v, k) => expect(hit!.normal[k]!, `normal ${k}`).toBeCloseTo(v, 5));
    }
    // From the other side the same triangles are back faces (one-sided emitters emit nothing there).
    const one = lights[0]!;
    const centroid = [0, 1, 2].map((k) => one.p0[k]! + (one.edge1[k]! + one.edge2[k]!) / 3);
    const behind = centroid.map((c, k) => c - one.normal[k]! * 0.75) as [number, number, number];
    const back = tracer.closest({ origin: behind, tMin: 0, direction: one.normal as unknown as [number, number, number], tMax: 10 });
    expect(back!.frontFace).toBe(false);
  });
});

describe('surface section', () => {
  const input = fixture();
  const copy = buildSceneCopy(input);
  const standalone = buildSurfaceTable(input.meshes, input.instances, {});

  it('decodes the same as the slice-1 table, hit for hit', () => {
    let seed = 7;
    const rnd = () => {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
      return seed / 2 ** 32;
    };
    let checked = 0;
    for (const instance of input.instances) {
      const mesh = input.meshes[instance.meshIndex]!;
      const triangles = mesh.indices!.length / 3;
      for (let t = 0; t < triangles; t++) {
        const u = rnd() * 0.5;
        const v = rnd() * 0.5;
        const hit: SurfaceHitInput = { t: 2, instance: instance.instanceIndex, triangle: t, barycentrics: [u, v], frontFace: true, material: instance.material ?? SCENE_NONE, normal: [0, -1, 0] };
        const ray = { origin: [0.1, 5, 0.2] as [number, number, number], direction: [0, -1, 0] as [number, number, number] };
        expect(readSurfaceFromTable(copy.surface, hit, ray, SURFACE.ALL)).toEqual(readSurfaceFromTable(standalone, hit, ray, SURFACE.ALL));
        const reference = referenceSurface(input.meshes, input.instances, hit, ray);
        const got = readSurfaceFromTable(copy.surface, hit, ray, SURFACE.ALL);
        got.uv.forEach((c, k) => expect(c).toBeCloseTo(reference.uv[k]!, 5));
        got.shadingNormal.forEach((c, k) => expect(c).toBeCloseTo(reference.shadingNormal[k]!, 5));
        checked++;
      }
    }
    expect(checked).toBeGreaterThan(8);
  });

  it('with positions: the barycentric world position', () => {
    const withPositions = buildSceneCopy({ ...input, options: { surface: { positions: true } } });
    const hit: SurfaceHitInput = { t: 2, instance: 0, triangle: 1, barycentrics: [0.25, 0.5], frontFace: true, material: 1, normal: [0, -1, 0] };
    const ray = { origin: [0, 5, 0] as [number, number, number], direction: [0, -1, 0] as [number, number, number] };
    const got = readSurfaceFromTable(withPositions.surface, hit, ray, SURFACE.POSITION);
    const reference = referenceSurface(input.meshes, input.instances, hit, ray, { positionFromVertices: true });
    got.position.forEach((c, k) => expect(c).toBeCloseTo(reference.position[k]!, 4));
  });

  it('vertex data sits behind the indices in `data`, and the header says where', () => {
    expect(copy.table[SCENE_HEADER.vertexWordBase]).toBe(copy.surface.indices.length);
    expect(copy.surface.vertices.buffer).toBe(copy.data.buffer);
    expect(copy.data.length).toBeGreaterThanOrEqual(copy.surface.indices.length + copy.surface.vertices.length);
  });
});

describe('scene copy: inputs and fingerprint', () => {
  it('fingerprints are stable and change with materials, lights and geometry', () => {
    const a = buildSceneCopy(fixture());
    const b = buildSceneCopy(fixture());
    expect(a.fingerprint).toBe(b.fingerprint);
    expect(a.fingerprint).toMatch(/^[0-9a-f]{16}$/);
    const other = fixture();
    (other.materials as unknown[])[0] = createMaterial({ baseColor: [0.5, 0.5, 0.51] });
    expect(buildSceneCopy(other).fingerprint).not.toBe(a.fingerprint);
    const sunless = fixture();
    sunless.sun = null;
    expect(buildSceneCopy(sunless).fingerprint).not.toBe(a.fingerprint);
    const moved = fixture();
    moved.instances = moved.instances.map((i, k) => (k === 2 ? { ...i, objectToWorld: trs([0, 0, 6], 0.2, [1, 1, 1]) } : i));
    expect(buildSceneCopy(moved).fingerprint).not.toBe(a.fingerprint);
    expect(new ContentHasher().words(Uint32Array.of(1, 2)).digest()).not.toBe(new ContentHasher().words(Uint32Array.of(2, 1)).digest());
  });

  it('an empty scene still builds, with no lights', () => {
    const empty = buildSceneCopy({ meshes: [], instances: [], materials: [] });
    expect(empty.lights).toHaveLength(0);
    expect(empty.table[SCENE_HEADER.lightCount]).toBe(0);
    expect(empty.bounds.radius).toBe(0);
    expect(readMaterial(empty, 0)).toEqual(empty.defaultMaterial);
  });

  it('no emitters but a sun gives exactly one light with probability 1', () => {
    const only = buildSceneCopy({ ...fixture(), materials: [{}, {}, {}], environment: null });
    expect(only.lights).toHaveLength(1);
    expect(readLight(only, 0).pmf).toBe(1);
  });

  it('rejects duplicate instance indices, too many materials and unknown meshes', () => {
    const dup = fixture();
    dup.instances = [dup.instances[0]!, { ...dup.instances[1]!, instanceIndex: 0 }];
    expect(() => buildSceneCopy(dup)).toThrow(RangeError);
    expect(() => buildSceneCopy({ meshes: [], instances: [], materials: new Array(65536).fill({}) })).toThrow(RangeError);
    expect(() => buildSceneCopy({ meshes: [quad()], instances: [{ meshIndex: 3, instanceIndex: 0, objectToWorld: identity() }], materials: [] })).toThrow(RangeError);
  });

  it('per-triangle materials outside the table are not lights', () => {
    const copy = buildSceneCopy({ meshes: [{ ...quad(), triangleMaterials: [9, 0xffff] }], instances: [{ meshIndex: 0, instanceIndex: 0, objectToWorld: identity(), material: 0 }], materials: [{ emissionColor: [1, 1, 1] }] });
    expect(copy.triangleLightCount).toBe(1); // triangle 1 falls back to the instance's emissive material 0; triangle 0 names material 9 (not in the table)
    expect((copy.lights[0] as RtTriangleLight).triangle).toBe(1);
    expect(readLightIndexOfHit(copy, 0, 0)).toBe(SCENE_NONE);
    expect(readLightIndexOfHit(copy, 0, 1)).toBe(0);
  });
});

describe('scene copy: textures (P1.4c)', () => {
  const tex = (w: number, h: number, colorSpace: 'srgb' | 'linear', rgba: [number, number, number, number] = [200, 100, 50, 255]) => ({ width: w, height: h, colorSpace, data: Uint8Array.from({ length: w * h * 4 }, (_, i) => rgba[i & 3] as number), label: `${colorSpace}${w}` });
  const textured = (): SceneCopyInput => {
    const base = fixture();
    return {
      ...base,
      materials: [{ baseColor: [0.5, 0.5, 0.5], baseColorTexture: 0, roughnessMetalTexture: 1, normalTexture: 1, emissionTexture: 2 }, ...(base.materials as []).slice(1)],
      textures: [tex(16, 16, 'srgb'), tex(32, 32, 'linear'), tex(16, 16, 'srgb')],
    };
  };

  it('without textures nothing changes: header words 14 and 15 are 0, no texture section, same table length and fingerprint as before', () => {
    const copy = buildSceneCopy(fixture());
    expect(copy.textures).toBeNull();
    expect(copy.textureLayout).toBeNull();
    expect(copy.table[SCENE_HEADER.textureBase]).toBe(0);
    expect(copy.table[SCENE_HEADER.textureCount]).toBe(0);
    expect(copy.table.length).toBe(copy.sections.vec4s * 4);
    expect(copy.sections.vec4s).toBe(copy.sections.lights + copy.lights.length * 6);
  });

  it('writes the slot records after the lights and points the header at them', () => {
    const copy = buildSceneCopy(textured());
    expect(copy.table[SCENE_HEADER.textureCount]).toBe(3);
    const base = copy.table[SCENE_HEADER.textureBase] as number;
    expect(base).toBe(copy.sections.lights + copy.lights.length * 6);
    expect(copy.sections.vec4s).toBe(base + 3);
    expect(copy.textureLayout!.buckets.map((b) => `${b.width}/${b.colorSpace}/${b.layers}`)).toEqual(['16/srgb/2', '32/linear/1']);
    // slot 0 and 2 share the 16x16 sRGB array (layers 0, 1); slot 1 is the 32x32 linear one.
    const record = (slot: number) => Array.from(copy.table.subarray((base + slot) * 4, (base + slot) * 4 + 4));
    expect(record(0)).toEqual([0, 0, 16 | (16 << 16), 5]);
    expect(record(1)).toEqual([1, 0, 32 | (32 << 16), 6]);
    expect(record(2)).toEqual([0, 1, 16 | (16 << 16), 5]);
  });

  it('forces vertex positions into the surface data (the texture frame needs the edges)', () => {
    const plain = buildSceneCopy(fixture());
    expect(plain.layout.positions).toBe(false);
    const copy = buildSceneCopy(textured());
    expect(copy.layout.positions).toBe(true);
    expect(copy.layout.uvs).toBe(true);
  });

  it('a textured emitter is not a light: NEE never sees constant radiance for a varying emitter', () => {
    const base = fixture();
    const reference = buildSceneCopy(base);
    const lit = buildSceneCopy({ ...base, materials: [base.materials[0]!, { emissionColor: [2, 4, 6], emissionTexture: 0 }, base.materials[2]!], textures: [tex(16, 16, 'srgb')] });
    // Material 1 lights instance 0 (dense, 2 triangles) and the strip's emissive triangles; with a texture they are BSDF-found only.
    expect(lit.triangleLightCount).toBeLessThan(reference.triangleLightCount);
    expect(reference.triangleLightCount - lit.triangleLightCount).toBe(2 + 2);
  });

  it('rejects a slot with no texture and a texture of the wrong colour space', () => {
    expect(() => buildSceneCopy({ ...textured(), textures: [tex(16, 16, 'srgb')] })).toThrow(/there are 1 textures/);
    expect(() => buildSceneCopy({ ...textured(), textures: [tex(16, 16, 'linear'), tex(32, 32, 'linear'), tex(16, 16, 'srgb')] })).toThrow(/baseColorTexture.*linear; this field needs srgb/);
    expect(() => buildSceneCopy({ ...textured(), textures: [tex(16, 16, 'srgb'), tex(32, 32, 'srgb'), tex(16, 16, 'srgb')] })).toThrow(/roughnessMetalTexture.*needs linear/);
  });

  it('textured scenes fingerprint their texels', () => {
    const a = buildSceneCopy(textured());
    const input = textured();
    input.textures = [tex(16, 16, 'srgb', [201, 100, 50, 255]), input.textures![1]!, input.textures![2]!];
    expect(buildSceneCopy(input).fingerprint).not.toBe(a.fingerprint);
    expect(buildSceneCopy(textured()).fingerprint).toBe(a.fingerprint);
    expect(a.fingerprint).not.toBe(buildSceneCopy(fixture()).fingerprint);
  });
});
