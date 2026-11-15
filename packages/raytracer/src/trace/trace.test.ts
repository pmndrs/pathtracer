import { describe, expect, it } from 'vitest';
import {
  DEFAULT_NODE_FORMAT,
  HIT_RECORD_BYTES,
  HIT_RECORD_COMPACT_BYTES,
  HIT_RECORD_WORD,
  MAX_MATERIAL,
  NO_MATERIAL,
  RAY_BYTES as BVH_RAY_BYTES,
  TRACER_GPU_LAYOUT_VERSION,
  traceModuleWgsl,
} from '@pmndrs/bvh';
import { RESERVED } from '../wgsl-names.testutil.ts';
import { rtPruneWgsl, rtShortenWgsl } from './modules.ts';
import { buildCaps } from '../caps/index.ts';
import { BVH_LAYOUT_VERSION, BVH_MAX_MATERIAL, HIT_BYTES, HIT_COMPACT_BYTES, RAY_BYTES, RAY_CONE_BYTES } from '../contract/ray.ts';
import {
  BATCH_PARAMS_BYTES,
  BATCH_WORKGROUP_SIZES,
  HIT_WORD,
  RT_NO_MATERIAL,
  SURFACE,
  SURFACE_TABLE,
  aosToSoa,
  buildSurfaceTable,
  dispatchGrid,
  packBatchParams,
  packCones,
  packRays,
  packRaysSoa,
  readHits,
  readSurfaceFromTable,
  readVisibility,
  referenceSurface,
  resolveBatchOptions,
  rtBatchArgsKernelWgsl,
  rtBatchEntries,
  rtBatchKernelWgsl,
  rtModuleStorageBuffers,
  rtModuleWgsl,
  sceneFormatOf,
  unpackRay,
  type BatchKernelOptions,
  type SurfaceInstanceInput,
  type SurfaceMeshInput,
} from './index.ts';

const noFeatures = buildCaps({ features: new Set(), limits: {} }, new Set());
const allFeatures = buildCaps({ features: new Set(['subgroups', 'shader-f16', 'timestamp-query']), limits: {} }, new Set());
/** Windows DXC compile time blows up with shader size (CLAUDE.md traps); every generated kernel stays under this. */
const KERNEL_BYTE_LIMIT = 50_000;

// ---------------------------------------------------------------------------------------------------------------
// A small WGSL struct layout calculator (host-shareable scalars and vectors only), to check generated structs against
// the TypeScript constants without a GPU.
// ---------------------------------------------------------------------------------------------------------------
const TYPE_LAYOUT: Record<string, { size: number; align: number }> = {
  f32: { size: 4, align: 4 },
  u32: { size: 4, align: 4 },
  i32: { size: 4, align: 4 },
  vec2f: { size: 8, align: 8 },
  vec2u: { size: 8, align: 8 },
  vec3f: { size: 12, align: 16 },
  vec3u: { size: 12, align: 16 },
  vec4f: { size: 16, align: 16 },
  vec4u: { size: 16, align: 16 },
};

function structLayout(wgsl: string, name: string): { size: number; offsets: Record<string, number> } {
  const match = new RegExp(`struct\\s+${name}\\s*\\{([^}]*)\\}`).exec(wgsl);
  if (!match) throw new Error(`struct ${name} not found`);
  const offsets: Record<string, number> = {};
  let offset = 0;
  let align = 1;
  for (const raw of (match[1] as string).split('\n')) {
    const line = raw.replace(/\/\/.*$/, '').trim();
    if (!line) continue;
    for (const field of line.split(',').map((part) => part.trim()).filter(Boolean)) {
      const [fieldName, type] = field.split(':').map((part) => part.trim()) as [string, string];
      const layout = TYPE_LAYOUT[type];
      if (!layout) throw new Error(`structLayout: type ${type} of ${name}.${fieldName} not supported`);
      offset = Math.ceil(offset / layout.align) * layout.align;
      offsets[fieldName] = offset;
      offset += layout.size;
      align = Math.max(align, layout.align);
    }
  }
  return { size: Math.ceil(offset / align) * align, offsets };
}

/** Names declared at module scope (struct, fn, const, var, alias, override). */
function declaredNames(wgsl: string): Set<string> {
  const names = new Set<string>();
  for (const m of wgsl.matchAll(/^(?:@[^\n]*?\s)?(?:struct|fn|const|alias|override|var(?:<[^>]*>)?)\s+([A-Za-z_]\w*)/gm)) names.add(m[1] as string);
  return names;
}

function bindingsOf(wgsl: string, group: number): number[] {
  return [...wgsl.matchAll(new RegExp(`@group\\(${group}\\)\\s*@binding\\((\\d+)\\)`, 'g'))].map((m) => Number(m[1])).sort((a, b) => a - b);
}

function batchCombos(): BatchKernelOptions[] {
  const out: BatchKernelOptions[] = [];
  for (const mode of ['closest', 'visible'] as const)
    for (const workgroupSize of BATCH_WORKGROUP_SIZES)
      for (const rayLayout of ['aos', 'soa'] as const)
        for (const indirect of [false, true])
          for (const cones of mode === 'closest' ? [false, true] : [false])
            for (const hitRecord of ['full', 'compact'] as const)
              for (const rayMasks of [true, false])
                for (const nodeFormat of ['f32', 'quantized'] as const)
                  for (const triangleLeaves of ['records', 'indexed'] as const)
                    out.push({ mode, workgroupSize, rayLayout, indirect, cones, hitRecord, rayMasks, nodeFormat, triangleLeaves });
  return out;
}

describe('contract against bvh layout v2', () => {
  it('ray and hit sizes, versions and limits agree with @pmndrs/bvh', () => {
    expect(BVH_LAYOUT_VERSION).toBe(TRACER_GPU_LAYOUT_VERSION);
    expect(RAY_BYTES).toBe(BVH_RAY_BYTES);
    expect(HIT_BYTES).toBe(HIT_RECORD_BYTES);
    expect(HIT_COMPACT_BYTES).toBe(HIT_RECORD_COMPACT_BYTES);
    expect(BVH_MAX_MATERIAL).toBe(MAX_MATERIAL);
    expect(RT_NO_MATERIAL).toBe(NO_MATERIAL);
    expect(HIT_WORD).toEqual(HIT_RECORD_WORD);
  });

  it('scene format defaults are bvh defaults', () => {
    expect(sceneFormatOf({})).toEqual({ nodeFormat: DEFAULT_NODE_FORMAT, triangleLeaves: 'records', positionLayout: 'vec4' });
  });
});

describe('generated WGSL', () => {
  it('every batch kernel combination preprocesses cleanly, stays under the size cap and binds what rtBatchEntries declares', () => {
    let largest = 0;
    for (const combo of batchCombos()) {
      for (const caps of [noFeatures, allFeatures]) {
        const code = rtBatchKernelWgsl(caps, combo);
        const what = JSON.stringify(combo);
        expect(code, what).not.toMatch(/^\s*#(if|elif|else|endif)\b/m);
        expect(code, what).toContain('fn rtBatch(');
        expect(code, what).toContain(`@workgroup_size(${combo.workgroupSize})`);
        largest = Math.max(largest, code.length);
        const declared = bindingsOf(code, 1);
        expect(declared, what).toEqual(rtBatchEntries(resolveBatchOptions(combo)).map((entry) => entry.binding));
        expect(bindingsOf(code, 0).length, what).toBe(combo.triangleLeaves === 'indexed' ? 4 : 3);
      }
    }
    expect(largest).toBeLessThan(KERNEL_BYTE_LIMIT);
  });

  it('the inline module (with and without rtSurface) stays under the size cap and has the public functions', () => {
    for (const normals of [false, true])
      for (const uvs of [false, true])
        for (const positions of [false, true])
          for (const triangleLeaves of ['records', 'indexed'] as const) {
            const scene = { slot: { group: 2, firstBinding: 0 }, layout: { normals, uvs, positions }, materials: false, lights: false };
            const code = rtModuleWgsl(allFeatures, { triangleLeaves, scene });
            expect(code.length).toBeLessThan(KERNEL_BYTE_LIMIT);
            for (const fn of ['rtVisible', 'rtVisibleSegment', 'rtClosest', 'rtClosestCone', 'rtSurface', 'rtSegment', 'rtConePropagate']) expect(code).toContain(`fn ${fn}(`);
            expect(code).not.toMatch(/^\s*#(if|elif|else|endif)\b/m);
            expect(bindingsOf(code, 2)).toEqual([0, 1]);
          }
    const plain = rtModuleWgsl(noFeatures);
    expect(plain).not.toContain('fn rtSurface(');
    expect(plain).toContain('bvhTraceAnyMasked');
    expect(rtModuleWgsl(noFeatures, { rayMasks: false })).not.toContain('bvhTraceAnyMasked');
  });

  it('declares nothing outside the rt prefix beyond what the bvh walk declares', () => {
    const walk = declaredNames(traceModuleWgsl({}));
    const scene = { slot: { group: 2, firstBinding: 0 }, layout: { normals: true, uvs: true, positions: true } };
    const ours = new Set([...declaredNames(rtModuleWgsl(noFeatures, { scene })), ...declaredNames(rtBatchKernelWgsl(noFeatures, { mode: 'closest', cones: true, indirect: true, rayLayout: 'soa' })), ...declaredNames(rtBatchArgsKernelWgsl(noFeatures))]);
    const strays = [...ours].filter((name) => !walk.has(name) && !/^(rt|Rt|RT_)/.test(name));
    expect(strays).toEqual([]);
    expect(ours.has('rtVisible')).toBe(true);
  });

  it('caps enable lines come first (subgroups, f16)', () => {
    const code = rtBatchKernelWgsl(allFeatures, { mode: 'visible' });
    expect(code.startsWith('enable f16;\nenable subgroups;')).toBe(true);
    expect(rtBatchKernelWgsl(noFeatures, { mode: 'visible' })).not.toContain('enable ');
  });
});

describe('struct layouts against the TypeScript constants', () => {
  const full = rtBatchKernelWgsl(noFeatures, { mode: 'closest', hitRecord: 'full', cones: true });
  const compact = rtBatchKernelWgsl(noFeatures, { mode: 'closest', hitRecord: 'compact' });

  it('RtHitRecord (full) is the 40-byte contract Hit with bvh word positions', () => {
    const layout = structLayout(full, 'RtHitRecord');
    expect(layout.size).toBe(HIT_BYTES);
    expect(layout.offsets).toEqual({
      t: HIT_WORD.t * 4,
      instance: HIT_WORD.instance * 4,
      triangle: HIT_WORD.triangle * 4,
      frontFace: HIT_WORD.frontFace * 4,
      bary: HIT_WORD.barycentrics * 4,
      material: HIT_WORD.material * 4,
      normalX: HIT_WORD.normal * 4,
      normalY: HIT_WORD.normal * 4 + 4,
      normalZ: HIT_WORD.normal * 4 + 8,
    });
  });

  it('RtHitRecord (compact) is 24 bytes', () => {
    const layout = structLayout(compact, 'RtHitRecord');
    expect(layout.size).toBe(HIT_COMPACT_BYTES);
    expect(layout.offsets.bary).toBe(HIT_WORD.barycentrics * 4);
  });

  it('RtRay matches the contract Ray / BvhRay (32 B), RtCone the 8 B cone, RtBatchParams the 16 B packer', () => {
    expect(structLayout(full, 'RtRay')).toEqual({ size: RAY_BYTES, offsets: { origin: 0, tMin: 12, direction: 16, tMax: 28 } });
    expect(structLayout(full, 'BvhRay')).toEqual({ size: RAY_BYTES, offsets: { origin: 0, tMin: 12, direction: 16, tMax: 28 } });
    expect(structLayout(full, 'RtCone')).toEqual({ size: RAY_CONE_BYTES, offsets: { width: 0, spread: 4 } });
    expect(structLayout(full, 'RtBatchParams')).toEqual({ size: BATCH_PARAMS_BYTES, offsets: { count: 0, mask: 4, capacity: 8, countWord: 12 } });
    expect(Array.from(packBatchParams({ count: 5, mask: 0xffffffff, capacity: 9, countWord: 2 }))).toEqual([5, 0xffffffff, 9, 2]);
  });

  it('rejects cones on a visibility batch', () => {
    expect(() => resolveBatchOptions({ mode: 'visible', cones: true })).toThrow(RangeError);
  });

  it('binding budget: walk 3 (4 indexed), scene copy 2', () => {
    expect(rtModuleStorageBuffers()).toBe(3);
    expect(rtModuleStorageBuffers({ triangleLeaves: 'indexed', scene: { slot: { group: 1, firstBinding: 0 }, layout: { normals: true, uvs: false, positions: false } } })).toBe(6);
  });
});

describe('records', () => {
  const rays = [
    { origin: [1, 2, 3], tMin: 0.001, direction: [0, 1, 0], tMax: 100 },
    { origin: [-4, 5.5, 6], tMin: 0, direction: [0.6, 0, -0.8], tMax: 1e30 },
  ] as const;

  it('packs AoS and SoA rays bit-identically and round-trips', () => {
    const aos = packRays(rays);
    expect(aos.length).toBe((rays.length * RAY_BYTES) / 4);
    expect(unpackRay(aos, 1)).toEqual({ origin: [-4, 5.5, 6], tMin: 0, direction: [Math.fround(0.6), 0, Math.fround(-0.8)], tMax: Math.fround(1e30) });
    const soa = packRaysSoa(rays);
    const converted = aosToSoa(aos);
    expect(Array.from(converted.a)).toEqual(Array.from(soa.a));
    expect(Array.from(converted.b)).toEqual(Array.from(soa.b));
    expect(Array.from(packCones([{ width: 0.5, spread: 0.25 }]))).toEqual([0.5, 0.25]);
  });

  it('reads full and compact hit records and visibility words', () => {
    const buffer = new ArrayBuffer(HIT_BYTES * 2);
    const f = new Float32Array(buffer);
    const u = new Uint32Array(buffer);
    f[0] = 2.5;
    u[1] = 7;
    u[2] = 11;
    u[3] = 1;
    f[4] = 0.25;
    f[5] = 0.5;
    u[6] = 3;
    f.set([0, 0, 1], 7);
    f[10] = -1;
    u[16] = NO_MATERIAL;
    const hits = readHits(buffer, 2);
    expect(hits[0]).toEqual({ t: 2.5, instance: 7, triangle: 11, frontFace: true, barycentrics: [0.25, 0.5], material: 3, normal: [0, 0, 1] });
    expect(hits[1]?.t).toBe(-1);
    expect(hits[1]?.material).toBe(NO_MATERIAL);
    const compact = readHits(new Float32Array([1, 0, 0, 0, 0.5, 0.5]), 1, 'compact');
    expect(compact[0]?.material).toBe(0xffffffff);
    expect(readVisibility(new Uint32Array([1, 0, 1]), 3)).toEqual([true, false, true]);
  });

  it('dispatch grid covers every ray within the per-dimension limit, with at most one partial row', () => {
    for (const workgroupSize of BATCH_WORKGROUP_SIZES)
      for (const count of [0, 1, 63, 64, 65, 1_000_003, 4_194_304, 10_000_000, 67_108_864]) {
        const [x, y, z] = dispatchGrid(count, workgroupSize, 65535);
        expect(z).toBe(1);
        expect(x).toBeLessThanOrEqual(65535);
        if (count === 0) {
          expect(x).toBe(0);
          continue;
        }
        const groups = Math.ceil(count / workgroupSize);
        expect(x * y).toBeGreaterThanOrEqual(groups);
        expect(x * (y - 1)).toBeLessThan(groups);
      }
  });
});

describe('surface table', () => {
  // Deterministic LCG.
  let seed = 12345;
  const random = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);

  const meshes: SurfaceMeshInput[] = [
    {
      positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0.5]),
      indices: [0, 1, 2, 2, 1, 3],
      normals: [0, 0, 1, 0.3, 0, 1, 0, 0.3, 1, 0.2, 0.2, 1],
      uvs: [0, 0, 1, 0, 0, 1, 1, 1],
    },
    // Unindexed, no attributes.
    { positions: new Float32Array([0, 0, 0, 0, 2, 0, 0, 0, 2]) },
    // Normals but no UVs.
    { positions: new Float32Array([1, 1, 1, 2, 1, 1, 1, 2, 1]), indices: [0, 2, 1], normals: [0, 0, -1, 0, 0, -1, 0.1, 0, -1] },
  ];
  // Column-major: rotation about y, non-uniform scale, translation; one mirrored.
  const affine = (angle: number, scale: [number, number, number], move: [number, number, number]) => {
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    return [c * scale[0], 0, -s * scale[0], 0, 0, scale[1], 0, 0, s * scale[2], 0, c * scale[2], 0, ...move, 1];
  };
  const instances: SurfaceInstanceInput[] = [
    { instanceIndex: 0, meshIndex: 0, objectToWorld: affine(0.3, [1, 2, 0.5], [1, 2, 3]) },
    { instanceIndex: 3, meshIndex: 0, objectToWorld: affine(-1.1, [-1, 1, 1], [0, 0, -2]) },
    { instanceIndex: 1, meshIndex: 1, objectToWorld: affine(0, [1, 1, 1], [0, 0, 0]) },
    { instanceIndex: 2, meshIndex: 2, objectToWorld: affine(2, [3, 3, 3], [5, 0, 0]) },
  ];

  it('header and record sizes', () => {
    const table = buildSurfaceTable(meshes, instances);
    expect(SURFACE_TABLE.INSTANCE_VEC4S * 16).toBe(SURFACE_TABLE.INSTANCE_BYTES);
    expect(table.instanceSlots).toBe(4);
    expect(table.table[0]).toBe(4);
    expect(table.table[1]).toBe(SURFACE_TABLE.HEADER_VEC4S + 4 * SURFACE_TABLE.INSTANCE_VEC4S);
    expect(table.layout).toEqual({ normals: true, uvs: true, positions: false });
    expect(table.vertexStride).toBe(5);
    expect(table.indices.length).toBe(6 + 3 + 3);
  });

  for (const positions of [false, true]) {
    it(`packed decode equals the double-precision oracle (positions ${positions ? 'from vertices' : 'from the ray'})`, () => {
      const table = buildSurfaceTable(meshes, instances, { positions });
      for (let k = 0; k < 400; k++) {
        const instance = instances[k % instances.length] as SurfaceInstanceInput;
        const mesh = meshes[instance.meshIndex] as SurfaceMeshInput;
        const triangles = (mesh.indices ? mesh.indices.length : mesh.positions.length / 3) / 3;
        const u = random();
        const v = random() * (1 - u);
        const hit = {
          t: random() * 10,
          instance: instance.instanceIndex,
          triangle: Math.floor(random() * triangles),
          barycentrics: [u, v] as const,
          frontFace: random() < 0.5,
          material: k,
          normal: [0, 0, random() < 0.5 ? 1 : -1] as const,
        };
        const ray = { origin: [random(), random(), random()] as const, direction: [0, 0, 1] as const };
        const packed = readSurfaceFromTable(table, hit, ray, SURFACE.ALL);
        const oracle = referenceSurface(meshes, instances, hit, ray, { positionFromVertices: positions });
        for (let c = 0; c < 3; c++) {
          expect(packed.position[c]).toBeCloseTo(oracle.position[c] as number, 4);
          expect(packed.shadingNormal[c]).toBeCloseTo(oracle.shadingNormal[c] as number, 5);
        }
        expect(packed.uv[0]).toBeCloseTo(oracle.uv[0], 5);
        expect(packed.uv[1]).toBeCloseTo(oracle.uv[1], 5);
        expect(packed.material).toBe(k);
        expect(packed.frontFace).toBe(hit.frontFace);
      }
    });
  }

  it('asks for nothing: no vertex data is read and nothing but the hit fields is filled', () => {
    const table = buildSurfaceTable(meshes, instances);
    const hit = { t: 1, instance: 0, triangle: 0, barycentrics: [0.2, 0.3] as const, frontFace: true, material: 4, normal: [0, 1, 0] as const };
    const out = readSurfaceFromTable(table, hit, { origin: [0, 0, 0], direction: [0, 0, 1] }, SURFACE.MATERIAL);
    expect(out.uv).toEqual([0, 0]);
    expect(out.position).toEqual([0, 0, 0]);
    expect(out.shadingNormal).toEqual([0, 1, 0]);
  });
});

describe('layout version check', () => {
  it('throws on a bvh walk of another layout version', async () => {
    const { assertBvhLayout } = await import('./modules.ts');
    expect(() => assertBvhLayout(BVH_LAYOUT_VERSION)).not.toThrow();
    expect(() => assertBvhLayout(BVH_LAYOUT_VERSION + 1)).toThrow(/layout v/);
  });
});

describe('rtPruneWgsl / rtShortenWgsl (P2.10)', () => {
  const src = [
    'enable f16;',
    'struct Used { a: f32, }',
    'struct Dead { b: f32, }',
    'const KEEP: f32 = 1.0;',
    'const DEAD: f32 = 2.0;',
    'fn helperUsed(x: f32) -> f32 {',
    '  let longLocalName = x * KEEP;',
    '  return longLocalName;',
    '}',
    'fn helperDead() -> f32 { return DEAD; }',
    'fn onlyFromDead() -> f32 { return helperDead(); }',
    '@group(0) @binding(0) var<storage, read_write> outBuf: array<f32>;',
    '@compute @workgroup_size(1)',
    'fn mainEntry() {',
    '  var u: Used;',
    '  u.a = helperUsed(1.0);',
    '  outBuf[0] = u.a;',
    '}',
  ].join('\n');

  it('prunes what no entry point or binding reaches, keeps entries, bindings and the lone attribute line', () => {
    const { code, dropped } = rtPruneWgsl(src);
    expect(dropped.sort()).toEqual(['DEAD', 'Dead', 'helperDead', 'onlyFromDead']);
    expect(code).toContain('@compute @workgroup_size(1)\nfn mainEntry');
    expect(code).toContain('outBuf');
    expect(code).toContain('enable f16;');
    expect(code).not.toContain('helperDead');
  });

  it('shortens top-level names consistently and leaves entries, bindings, members and locals alone', () => {
    const { code, names } = rtShortenWgsl(rtPruneWgsl(src).code, ['mainEntry']);
    expect(names.has('helperUsed')).toBe(true);
    expect(names.has('mainEntry')).toBe(false);
    expect(names.has('outBuf')).toBe(false);
    expect(code).toContain('outBuf');
    expect(code).toContain('mainEntry');
    expect(code).toContain('u.a');
    expect(code).toContain('longLocalName');
    expect(code).not.toContain('helperUsed');
    expect(code.split('\n').length).toBe(rtPruneWgsl(src).code.split('\n').length);
    for (const n of names.values()) expect(n.length).toBeLessThan(4);
  });

  it('does not rename a top-level name that is also a member, a parameter or a local elsewhere', () => {
    const clash = 'const Shared: f32 = 1.0;\nstruct S { Shared: f32, }\nfn f(Shared2: f32) -> f32 { return Shared2; }\nfn g() -> f32 { var s: S; return s.Shared + Shared; }';
    expect(rtShortenWgsl(clash).names.has('Shared')).toBe(false);
  });

  it('never picks a name already in the source or a WGSL reserved word', () => {
    const many = Array.from({ length: 80 }, (_, i) => `fn someLongFunctionName${i}() -> f32 { return A + ${i}.0; }`).join('\n');
    const { code, names } = rtShortenWgsl(`const A: f32 = 1.0;\n${many}`);
    expect([...names.values()]).not.toContain('A');
    expect(new Set(names.values()).size).toBe(names.size);
    for (const n of names.values()) expect(RESERVED.has(n)).toBe(false);
    expect(code).toContain('const A');
  });
});
