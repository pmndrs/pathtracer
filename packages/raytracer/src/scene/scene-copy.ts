/**
 * The scene copy (D4): what the GPU needs beyond bvh's trees to shade a hit. Built from the same inputs as bvh's
 * `buildTraceScene` (meshes with optional per-triangle materials, instances with a material and transform) plus a material
 * list, a sun and an environment. Two storage buffers (`SCENE_COPY_STORAGE_BUFFERS`): the table and the data (layout in
 * constants.ts).
 *
 * Contents:
 *  - the surface section `rtSurface` reads (instance transforms, mesh records, indices, vertex attributes): the layout of
 *    the slice-1 `SurfaceTable` (trace/surface-table.ts) moved into the scene copy, so the old stand-in and this share one
 *    CPU decode (`readSurfaceFromTable`) and one WGSL body;
 *  - the material table (material.ts);
 *  - the light table with its alias table (lights.ts), the sun and the environment, and the triangle -> light map.
 *
 * Provenance: the table-with-header + fixed-size records design, the "none" sentinel, and the content fingerprint follow
 * webgpu-ddgi's platform scene copy (`packages/ddgi-core/src/platform/scene/`, MIT, same author); `ContentHasher` is adapted
 * from its `scene-fingerprint.ts`. The records themselves are new (OpenPBR subset, emissive-triangle lights with alias
 * sampling); ddgi's change log, GPU mesh packing and texture store are not taken (full rebuild only, see TASKS follow-ups).
 *
 * Everything is CPU code run in Node: the unit tests read the packed words back with the same twins the GPU check uses.
 */
import { buildSurfaceTable, type SurfaceHitInput, type SurfaceInstanceInput, type SurfaceLayout, type SurfaceMeshInput, type SurfaceTable, type SurfaceTableOptions } from '../trace/surface-table.ts';
import { SURFACE_TABLE } from '../trace/constants.ts';
import { BVH_MAX_MATERIAL } from '../contract/ray.ts';
import {
  INSTANCE_LIGHT,
  LIGHT_MAP_MODE,
  LIGHT_RECORD_VEC4S,
  SCENE_COPY_STORAGE_BUFFERS,
  SCENE_COPY_VERSION,
  SCENE_FLAG,
  SCENE_HEADER,
  SCENE_HEADER_VEC4S,
  SCENE_NONE,
} from './constants.ts';
import {
  buildAliasTable,
  lightSelectionWeights,
  type LightSelection,
  buildTriangleLights,
  makeEnvironment,
  makeSun,
  packLight,
  unpackLight,
  worldTriangle,
  type AliasTable,
  type EnvironmentInput,
  type RtEnvironmentLight,
  type RtLight,
  type SunInput,
  type UnpackedLight,
} from './lights.ts';
import { buildEnvMapTable, envMapMean } from './env-map.ts';
import { MATERIAL_RECORD_VEC4S, MATERIAL_RECORD_WORDS, createMaterial, defaultMaterial, f32Bits, packMaterialTable, unpackMaterial, type RtMaterial, type RtMaterialInput } from './material.ts';

export interface SceneCopyMesh extends SurfaceMeshInput {
  /** bvh's per-triangle materials (by original triangle number; 0xffff = none, then the instance's applies). */
  triangleMaterials?: ArrayLike<number> | null;
}

/** The part of bvh's `TraceSceneInstance` the scene copy reads. */
export interface SceneCopyInstance extends SurfaceInstanceInput {
  /** bvh material (0-65534) of the instance; omit for none. */
  material?: number;
  /** Default true. */
  visible?: boolean;
  /** The bvh instance mask (0-255). Not used by the scene copy; carried so one instance list can feed both bvh and the copy. */
  mask?: number;
}

export interface SceneCopyOptions {
  /** Surface data options: `normals`, `uvs`, `positions` (see `SurfaceTableOptions`). */
  surface?: SurfaceTableOptions;
  /** Multiplies the sun's power in the light-selection weights. Default 1 (see lights.ts for the power formulas). */
  sunPowerScale?: number;
  /** Multiplies the environment's power in the light-selection weights. Default 1. */
  envPowerScale?: number;
  /** Which weights the light alias table is built from (P1.4f A/B): `power` (default), `kind`, `{ capped: share }`. Changes noise only. */
  lightSelection?: LightSelection;
}

export interface SceneCopyInput {
  meshes: readonly SceneCopyMesh[];
  instances: readonly SceneCopyInstance[];
  /** Material table: bvh's material id is an index into this list. Partial materials take the OpenPBR defaults. */
  materials: readonly RtMaterialInput[];
  sun?: SunInput | null;
  environment?: EnvironmentInput | null;
  options?: SceneCopyOptions;
}

export interface SceneBounds {
  min: [number, number, number];
  max: [number, number, number];
  center: [number, number, number];
  /** Radius of the bounding sphere (half the box diagonal). */
  radius: number;
}

export interface SceneCopy {
  /** `array<vec4u>`: header, instances, meshes, materials, lights. */
  table: Uint32Array;
  /** `array<u32>`: indices, vertex attributes, light maps. */
  data: Uint32Array;
  layout: SurfaceLayout;
  /** The surface section as a `SurfaceTable` view over `table` and `data` (no copy): feed it to `readSurfaceFromTable` / `referenceSurface` checks. */
  surface: SurfaceTable;
  /** Decoded (quantised) materials as the GPU sees them; `defaultMaterial` is the record at index `materials.length`. */
  materials: RtMaterial[];
  defaultMaterial: RtMaterial;
  /** Emissive triangles first, then the sun, then the environment (when present). */
  lights: RtLight[];
  triangleLightCount: number;
  sunLight: number;
  environmentLight: number;
  alias: AliasTable;
  totalPower: number;
  bounds: SceneBounds;
  /** vec4u offsets of the sections inside `table`. */
  sections: { meshes: number; materials: number; lights: number; vec4s: number };
  byteLength: number;
  /** Storage buffers this adds to a kernel. */
  storageBuffers: number;
  /** Content hash (hex) of everything the shaders read; equal scene copies give equal fingerprints. */
  fingerprint: string;
}

/** Adapted from webgpu-ddgi `scene-fingerprint.ts` (MIT, same author): two 32-bit multiply-xorshift lanes over u32 words. */
export class ContentHasher {
  private lane1 = 0x9e3779b9;
  private lane2 = 0x85ebca6b;
  private count = 0;

  word(value: number): this {
    this.lane1 = Math.imul(((this.lane1 ^ value) << 13) | ((this.lane1 ^ value) >>> 19), 0xcc9e2d51) >>> 0;
    this.lane2 = Math.imul(((this.lane2 ^ value) << 17) | ((this.lane2 ^ value) >>> 15), 0x1b873593) >>> 0;
    this.lane1 = (this.lane1 + this.lane2) >>> 0;
    this.count++;
    return this;
  }

  words(words: Uint32Array): this {
    this.word(words.length);
    let lane1 = this.lane1;
    let lane2 = this.lane2;
    for (let index = 0; index < words.length; index++) {
      const value = words[index] as number;
      let mixed1 = lane1 ^ value;
      let mixed2 = lane2 ^ value;
      mixed1 = (mixed1 << 13) | (mixed1 >>> 19);
      mixed2 = (mixed2 << 17) | (mixed2 >>> 15);
      lane2 = Math.imul(mixed2, 0x1b873593) >>> 0;
      lane1 = (Math.imul(mixed1, 0xcc9e2d51) + lane2) >>> 0;
    }
    this.lane1 = lane1;
    this.lane2 = lane2;
    this.count += words.length;
    return this;
  }

  digest(): string {
    return hex32(finalise(this.lane1 ^ this.count)) + hex32(finalise(this.lane2 ^ Math.imul(this.count, 0x27d4eb2d)));
  }
}

function finalise(value: number): number {
  let x = value >>> 0;
  x ^= x >>> 16;
  x = Math.imul(x, 0x85ebca6b) >>> 0;
  x ^= x >>> 13;
  x = Math.imul(x, 0xc2b2ae35) >>> 0;
  x ^= x >>> 16;
  return x >>> 0;
}
function hex32(value: number): string {
  return (value >>> 0).toString(16).padStart(8, '0');
}

function sceneBounds(meshes: readonly SceneCopyMesh[], instances: readonly SceneCopyInstance[]): SceneBounds {
  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  for (const instance of instances) {
    if (instance.visible === false) continue;
    const mesh = meshes[instance.meshIndex] as SceneCopyMesh;
    const m = instance.objectToWorld;
    for (let v = 0; v < mesh.positions.length; v += 3) {
      const x = mesh.positions[v] as number;
      const y = mesh.positions[v + 1] as number;
      const z = mesh.positions[v + 2] as number;
      for (let k = 0; k < 3; k++) {
        const p = (m[k] as number) * x + (m[4 + k] as number) * y + (m[8 + k] as number) * z + (m[12 + k] as number);
        if (p < (min[k] as number)) min[k] = p;
        if (p > (max[k] as number)) max[k] = p;
      }
    }
  }
  if (!Number.isFinite(min[0])) return { min: [0, 0, 0], max: [0, 0, 0], center: [0, 0, 0], radius: 0 };
  const center: [number, number, number] = [0, 1, 2].map((k) => 0.5 * ((min[k] as number) + (max[k] as number))) as [number, number, number];
  return { min, max, center, radius: 0.5 * Math.hypot((max[0] as number) - (min[0] as number), (max[1] as number) - (min[1] as number), (max[2] as number) - (min[2] as number)) };
}

/** Builds the scene copy. A full rebuild: incremental updates are a follow-up. */
export function buildSceneCopy(input: SceneCopyInput): SceneCopy {
  const { meshes, instances } = input;
  const options = input.options ?? {};
  if (input.materials.length > BVH_MAX_MATERIAL + 1) throw new RangeError(`scene copy: ${input.materials.length} materials; bvh's 16-bit id allows ${BVH_MAX_MATERIAL + 1}`);
  const seen = new Set<number>();
  for (const instance of instances) {
    if (seen.has(instance.instanceIndex)) throw new RangeError(`scene copy: instanceIndex ${instance.instanceIndex} is used twice`);
    seen.add(instance.instanceIndex);
  }

  // Materials: pack, then keep the decoded (quantised) values so light radiance equals what the GPU decodes.
  const requested = input.materials.map((m) => createMaterial(m));
  const fallback = defaultMaterial();
  const materialWords = packMaterialTable(requested, fallback);
  const materials = requested.map((_, i) => unpackMaterial(materialWords, i * MATERIAL_RECORD_WORDS));
  const decodedDefault = unpackMaterial(materialWords, requested.length * MATERIAL_RECORD_WORDS);

  // Surface section: the slice-1 builder, re-laid under the bigger header.
  const surface = buildSurfaceTable(meshes, instances, options.surface);
  const bounds = sceneBounds(meshes, instances);

  // Lights.
  const built = buildTriangleLights(meshes, instances, materials);
  const lights: RtLight[] = [...built.lights];
  const triangleLightCount = built.lights.length;
  let sunLight = SCENE_NONE;
  let environmentLight = SCENE_NONE;
  if (input.sun) {
    sunLight = lights.length;
    lights.push(makeSun(input.sun, bounds.radius, options.sunPowerScale ?? 1));
  }
  // Environment map: its words go after the light maps in `data`; the base is known once the light maps are laid out.
  const envMapTable = input.environment?.map ? buildEnvMapTable(input.environment.map) : null;
  const envMapMeanRadiance = input.environment?.map ? envMapMean(input.environment.map) : null;
  const envLightAt = lights.length;
  if (input.environment) {
    environmentLight = lights.length;
    lights.push(makeEnvironment(input.environment, bounds.radius, options.envPowerScale ?? 1, envMapTable && envMapMeanRadiance ? { base: 0, width: envMapTable.width, height: envMapTable.height, pdfNorm: envMapTable.pdfNorm, mean: envMapMeanRadiance } : undefined));
  }
  const alias = buildAliasTable(lightSelectionWeights(lights, options.lightSelection ?? 'power'));
  // The real total power, whatever the selection weights are (the alias table's own total is 1 for a non-power selection).
  const totalPower = lights.reduce((sum, l) => sum + l.power, 0);

  // Layout.
  const surfaceBodyVec4s = surface.table.length / 4 - SURFACE_TABLE.HEADER_VEC4S;
  const meshBase = (surface.table[1] as number) - SURFACE_TABLE.HEADER_VEC4S + SCENE_HEADER_VEC4S;
  const materialBase = SCENE_HEADER_VEC4S + surfaceBodyVec4s;
  const lightBase = materialBase + (materials.length + 1) * MATERIAL_RECORD_VEC4S;
  const totalVec4s = lightBase + lights.length * LIGHT_RECORD_VEC4S;
  const table = new Uint32Array(Math.max(totalVec4s, 1) * 4);
  table.set(surface.table.subarray(SURFACE_TABLE.HEADER_VEC4S * 4), SCENE_HEADER_VEC4S * 4);

  // Light maps (instance -> light index) and the data buffer.
  const mapWords: number[] = [];
  const indexWords = surface.indices.length;
  const vertexWords = surface.vertices.length;
  const mapBaseOfData = indexWords + vertexWords;
  const lightMode = new Map<number, { mode: number; base: number; map: number }>();
  for (const per of built.perInstance) {
    if (per.count === 0) continue;
    if (per.count === per.triangleCount) {
      lightMode.set(per.instanceIndex, { mode: LIGHT_MAP_MODE.dense, base: per.firstLight, map: 0 });
    } else {
      const base = mapBaseOfData + mapWords.length;
      for (let t = 0; t < per.triangleCount; t++) mapWords.push(SCENE_NONE);
      per.triangles.forEach((triangle, k) => {
        mapWords[base - mapBaseOfData + triangle] = per.firstLight + k;
      });
      lightMode.set(per.instanceIndex, { mode: LIGHT_MAP_MODE.map, base: per.firstLight, map: base });
    }
  }
  for (const instance of instances) {
    const record = SCENE_HEADER_VEC4S + instance.instanceIndex * SURFACE_TABLE.INSTANCE_VEC4S;
    const entry = lightMode.get(instance.instanceIndex);
    const meta = (record + SURFACE_TABLE.INSTANCE_META) * 4;
    table[meta + INSTANCE_LIGHT.metaLightMode] = entry ? entry.mode : LIGHT_MAP_MODE.none;
    table[meta + INSTANCE_LIGHT.metaLightBase] = entry ? entry.base : SCENE_NONE;
    table[(record + INSTANCE_LIGHT.spareVec4) * 4 + INSTANCE_LIGHT.mapBaseWord] = entry ? entry.map : 0;
  }
  const envMapBase = indexWords + vertexWords + mapWords.length;
  if (envMapTable) (lights[envLightAt] as RtEnvironmentLight).mapBase = envMapBase;
  const data = new Uint32Array(Math.max(envMapBase + (envMapTable ? envMapTable.words.length : 0), 1));
  data.set(surface.indices, 0);
  data.set(new Uint32Array(surface.vertices.buffer, surface.vertices.byteOffset, vertexWords), indexWords);
  data.set(mapWords, mapBaseOfData);
  if (envMapTable) data.set(envMapTable.words, envMapBase);

  table.set(materialWords, materialBase * 4);
  lights.forEach((light, i) => packLight(light, alias.pmf[i] as number, alias.prob[i] as number, alias.alias[i] as number, table, (lightBase + i * LIGHT_RECORD_VEC4S) * 4));

  // Header.
  const h = (word: number, value: number) => {
    table[word] = value >>> 0;
  };
  h(SCENE_HEADER.instanceSlots, surface.instanceSlots);
  h(SCENE_HEADER.meshBase, meshBase);
  h(SCENE_HEADER.vertexStride, surface.vertexStride);
  h(SCENE_HEADER.vertexWordBase, indexWords);
  h(SCENE_HEADER.materialCount, materials.length);
  h(SCENE_HEADER.materialBase, materialBase);
  h(SCENE_HEADER.lightCount, lights.length);
  h(SCENE_HEADER.lightBase, lightBase);
  h(SCENE_HEADER.triangleLightCount, triangleLightCount);
  h(SCENE_HEADER.sunLight, sunLight);
  h(SCENE_HEADER.envLight, environmentLight);
  h(SCENE_HEADER.flags, surface.layout.positions ? SCENE_FLAG.positions : 0);
  h(SCENE_HEADER.totalPower, f32Bits(totalPower));
  h(SCENE_HEADER.sceneRadius, f32Bits(bounds.radius));

  const surfaceView: SurfaceTable = {
    ...surface,
    table,
    indices: data.subarray(0, indexWords),
    vertices: new Float32Array(data.buffer, indexWords * 4, vertexWords),
    headerVec4s: SCENE_HEADER_VEC4S,
    byteLength: table.byteLength + data.byteLength,
  };

  const fingerprint = new ContentHasher().word(SCENE_COPY_VERSION).words(table).words(data).digest();
  return {
    table,
    data,
    layout: surface.layout,
    surface: surfaceView,
    materials,
    defaultMaterial: decodedDefault,
    lights,
    triangleLightCount,
    sunLight,
    environmentLight,
    alias,
    totalPower,
    bounds,
    sections: { meshes: meshBase, materials: materialBase, lights: lightBase, vec4s: totalVec4s },
    byteLength: table.byteLength + data.byteLength,
    storageBuffers: SCENE_COPY_STORAGE_BUFFERS,
    fingerprint,
  };
}

// ---------------------------------------------------------------------------------------------------------------
// CPU twins of the WGSL lookups: they read the packed words, not the objects the builder made.
// ---------------------------------------------------------------------------------------------------------------

/** Twin of WGSL `rtMaterial(id)`: ids at or past the table (including `RT_NO_MATERIAL`) read the default record. */
export function readMaterial(copy: SceneCopy, id: number): RtMaterial {
  const count = copy.table[SCENE_HEADER.materialCount] as number;
  const slot = id >>> 0 < count ? id : count;
  return unpackMaterial(copy.table, ((copy.table[SCENE_HEADER.materialBase] as number) + slot * MATERIAL_RECORD_VEC4S) * 4);
}

/** Twin of WGSL `rtLight(index)`; throws for an index past the table. */
export function readLight(copy: SceneCopy, index: number): UnpackedLight {
  const count = copy.table[SCENE_HEADER.lightCount] as number;
  if (!(index >= 0 && index < count)) throw new RangeError(`readLight: ${index} of ${count}`);
  return unpackLight(copy.table, ((copy.table[SCENE_HEADER.lightBase] as number) + index * LIGHT_RECORD_VEC4S) * 4);
}

/** Twin of WGSL `rtLightIndexOfHit(instance, triangle)`: the light index of an emissive triangle, `SCENE_NONE` otherwise. */
export function readLightIndexOfHit(copy: SceneCopy, instance: number, triangle: number): number {
  if (instance >= (copy.table[SCENE_HEADER.instanceSlots] as number)) return SCENE_NONE;
  const record = (SCENE_HEADER_VEC4S + instance * SURFACE_TABLE.INSTANCE_VEC4S) * 4;
  const meta = record + SURFACE_TABLE.INSTANCE_META * 4;
  const mode = copy.table[meta + INSTANCE_LIGHT.metaLightMode] as number;
  if (mode === LIGHT_MAP_MODE.dense) return ((copy.table[meta + INSTANCE_LIGHT.metaLightBase] as number) + triangle) >>> 0;
  if (mode === LIGHT_MAP_MODE.map) return copy.data[(copy.table[record + INSTANCE_LIGHT.spareVec4 * 4 + INSTANCE_LIGHT.mapBaseWord] as number) + triangle] as number;
  return SCENE_NONE;
}

/** Twin of WGSL `rtLightPmfOfHit`: the probability `rtSampleLight` picks the emissive triangle a hit landed on (0 for a plain triangle). */
export function readLightPmfOfHit(copy: SceneCopy, instance: number, triangle: number): number {
  const index = readLightIndexOfHit(copy, instance, triangle);
  return index === SCENE_NONE ? 0 : readLight(copy, index).pmf;
}

export type SceneSurfaceHit = SurfaceHitInput;

/** World-space area of one triangle of one instance, in doubles (an oracle for the light records). */
export function referenceTriangleArea(mesh: SceneCopyMesh, instance: SceneCopyInstance, triangle: number): number {
  const [a, b, c] = worldTriangle(mesh, instance.objectToWorld, triangle);
  const e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
  const e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
  return 0.5 * Math.hypot((e1[1] as number) * (e2[2] as number) - (e1[2] as number) * (e2[1] as number), (e1[2] as number) * (e2[0] as number) - (e1[0] as number) * (e2[2] as number), (e1[0] as number) * (e2[1] as number) - (e1[1] as number) * (e2[0] as number));
}

