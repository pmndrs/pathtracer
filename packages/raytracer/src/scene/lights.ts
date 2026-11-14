/**
 * Lights: emissive triangles (world space, built on the CPU from the meshes the trace scene was built from), one
 * directional sun and one environment, with an alias table over light power for sampling (Vose, O(1) per sample).
 *
 * Power (luminance-weighted, Rec.709 Y):
 *  - one-sided emissive triangle: pi * area * Y(L)   (radiant exitance of a diffuse emitter is pi * L); two-sided doubles it;
 *  - sun: Y(E) * pi * R^2, the flux through the scene's bounding-sphere cross section (R = sceneRadius);
 *  - uniform environment: Y(L) * 4 pi * pi * R^2 (pbrt's convention for infinite lights).
 * The sun and environment power scale are settings (`sunPowerScale`, `envPowerScale`): power-proportional selection
 * can over- or under-sample infinite lights next to small emitters, and the right weights are a P2 A/B.
 *
 * Everything here is plain CPU code (Node tests). The WGSL twin is `rtSampleLight` in scene-wgsl.ts.
 */
import { LIGHT_FLAG, LIGHT_KIND, LIGHT_RECORD_VEC4S, LIGHT_WORD, SCENE_NONE } from './constants.ts';
import { bitsF32, f32Bits, luminance, type Rgb3, type RtMaterial } from './material.ts';

export type Vec3Tuple = [number, number, number];

/** An emissive triangle in world space. Radiance and side come from the triangle's material. */
export interface RtTriangleLight {
  kind: 'triangle';
  instance: number;
  /** Original triangle index inside the instance's mesh (what a hit reports). */
  triangle: number;
  material: number;
  p0: Vec3Tuple;
  /** p1 - p0 and p2 - p0, world space. */
  edge1: Vec3Tuple;
  edge2: Vec3Tuple;
  /** Unit normal of the emitting side: the authored front side, through mirroring transforms (what a hit's `normal` is). */
  normal: Vec3Tuple;
  area: number;
  /** Emitted radiance, W/sr/m^2. */
  radiance: Rgb3;
  twoSided: boolean;
  /** Luminance-weighted power. */
  power: number;
}

/** The directional sun. `direction` points from the scene toward the sun (the opposite of the way light travels). */
export interface RtSunLight {
  kind: 'sun';
  direction: Vec3Tuple;
  /** Radians; 0 = a delta direction. */
  angularRadius: number;
  /** `2 pi (1 - cos(angularRadius))`, steradians; 0 for a delta sun. */
  solidAngle: number;
  /** Irradiance on a surface facing the sun, W/m^2 (radiance-equivalent: radiance = irradiance / solidAngle). */
  irradiance: Rgb3;
  power: number;
}

/**
 * The environment: constant `radiance`, or an equirect map (`mapSlot` 0, env-map.ts; SCENE_NONE = none). With a map,
 * `radiance` is the solid-angle-weighted mean of the map (the constant that carries the same power: light selection and the
 * constant-radiance fallback use it) and the map's data sits at `mapBase` in the scene copy's `data` buffer.
 */
export interface RtEnvironmentLight {
  kind: 'environment';
  radiance: Rgb3;
  /** Word index of the map in `data`, and its size; 0 without a map. */
  mapBase: number;
  mapWidth: number;
  mapHeight: number;
  /** f32 pdf normaliser of the map's sampling density. */
  pdfNorm: number;
  scale: number;
  /** Radians about +Y; used with a map. */
  rotationY: number;
  mapSlot: number;
  power: number;
}

export type RtLight = RtTriangleLight | RtSunLight | RtEnvironmentLight;

export interface SunInput {
  /** The way the light travels (the bench `LightDesc.direction`); normalised here. */
  direction: readonly [number, number, number];
  irradiance: readonly [number, number, number];
  angularRadius?: number;
}
export interface EnvironmentInput {
  /** Constant radiance. With a `map` it defaults to the map's mean. */
  radiance?: readonly [number, number, number];
  /** An equirect HDR map (`buildSceneCopy` copies it into the scene copy; env-map.ts). */
  map?: { width: number; height: number; channels: number; data: ArrayLike<number> };
  scale?: number;
  rotationY?: number;
  mapSlot?: number;
}

// ---------------------------------------------------------------------------------------------------------------
// Alias table (Vose 1991)
// ---------------------------------------------------------------------------------------------------------------

export interface AliasTable {
  /** Probability (f32-rounded) of keeping slot i when it is drawn; otherwise `alias[i]`. */
  prob: Float32Array;
  alias: Uint32Array;
  /** Selection probability of each index, `weight / sum`, rounded to f32. */
  pmf: Float32Array;
  total: number;
}

/** Builds the table for non-negative weights. All-zero weights give a uniform table. */
export function buildAliasTable(weights: ArrayLike<number>): AliasTable {
  const n = weights.length;
  let total = 0;
  for (let i = 0; i < n; i++) {
    const w = weights[i] as number;
    if (!(w >= 0) || !Number.isFinite(w)) throw new RangeError(`alias table: weight ${i} is ${w}`);
    total += w;
  }
  const uniform = !(total > 0);
  const scaled = new Float64Array(n);
  for (let i = 0; i < n; i++) scaled[i] = uniform ? 1 : ((weights[i] as number) * n) / total;
  const prob = new Float64Array(n);
  const alias = new Uint32Array(n);
  const small: number[] = [];
  const large: number[] = [];
  for (let i = 0; i < n; i++) {
    if ((scaled[i] as number) < 1) small.push(i);
    else large.push(i);
  }
  while (small.length > 0 && large.length > 0) {
    const s = small.pop() as number;
    const l = large.pop() as number;
    prob[s] = scaled[s] as number;
    alias[s] = l;
    scaled[l] = (scaled[l] as number) + (scaled[s] as number) - 1;
    if ((scaled[l] as number) < 1) small.push(l);
    else large.push(l);
  }
  for (const i of [...large, ...small]) {
    prob[i] = 1;
    alias[i] = i;
  }
  const pmf = new Float32Array(n);
  for (let i = 0; i < n; i++) pmf[i] = uniform ? 1 / n : (weights[i] as number) / total;
  return { prob: Float32Array.from(prob), alias, pmf, total: uniform ? 0 : total };
}

// ---------------------------------------------------------------------------------------------------------------
// Light selection strategies (P1.4f): which weights go into the alias table. Selection only moves noise, never the mean.
// ---------------------------------------------------------------------------------------------------------------

/** The default light selection: pbrt-v4's rule (D22). */
export const DEFAULT_LIGHT_SELECTION: LightSelection = 'pbrt';

/**
 * How NEE picks a light:
 *  - `pbrt` (the default, D22): pbrt-v4's default `bvh` light sampler without the tree. Each infinite light (sun,
 *    environment) gets an equal share 1/(nInfinite + 1), and the area lights together get the remaining 1/(nInfinite + 1),
 *    split by power (pbrt splits it with its light BVH; power is our stand-in until we have one). With no area power the
 *    infinite lights share everything. Mitsuba 3 does the same in spirit: each emitter has sampling_weight 1;
 *  - `power`: probability proportional to power (sun and environment power use the bounding-sphere conventions above);
 *  - `kind`: each kind that has power (emissive triangles, sun, environment) gets an equal share, and within a kind the
 *    share follows power (so triangles stay area/radiance weighted);
 *  - `{ capped: c }`: power-proportional, except the infinite lights (sun + environment together) are capped at share `c`
 *    (their weights are scaled down when their power share exceeds `c`; never scaled up). With no area power they take everything.
 * A light with no power gets weight 0 under every strategy.
 */
export type LightSelection = 'pbrt' | 'power' | 'kind' | { capped: number };

/** Parses the bench/page spelling: `pbrt`, `power`, `kind`, `cap:0.25`. */
export function parseLightSelection(text: string): LightSelection {
  if (text === 'pbrt' || text === 'power' || text === 'kind') return text;
  const m = /^cap:(\d*\.?\d+)$/.exec(text);
  if (m) return { capped: Number(m[1]) };
  throw new RangeError(`light selection '${text}': expected pbrt, power, kind or cap:<share>`);
}

export function lightSelectionName(selection: LightSelection): string {
  return typeof selection === 'string' ? selection : `cap:${selection.capped}`;
}

/** Selection probabilities (sum to 1 when any light has power, else all 0) of each light under `selection`. The alias table is built from these. */
export function lightSelectionWeights(lights: readonly RtLight[], selection: LightSelection = DEFAULT_LIGHT_SELECTION): number[] {
  const power = lights.map((l) => (l.power > 0 ? l.power : 0));
  const total = power.reduce((a, b) => a + b, 0);
  if (!(total > 0)) return power;
  const group = (pred: (l: RtLight) => boolean) => power.reduce((a, p, i) => a + (pred(lights[i] as RtLight) ? p : 0), 0);
  const scaleGroup = (pred: (l: RtLight) => boolean, share: number, groupTotal: number) =>
    power.map((p, i) => (pred(lights[i] as RtLight) && groupTotal > 0 ? (p / groupTotal) * share : 0));
  if (selection === 'power') return power.map((p) => p / total);
  if (selection === 'pbrt') {
    const isInfinite = (l: RtLight) => l.kind !== 'triangle';
    const nInfinite = lights.filter((l, i) => isInfinite(l) && (power[i] as number) > 0).length;
    const pArea = group((l) => !isInfinite(l));
    const slots = nInfinite + (pArea > 0 ? 1 : 0);
    return power.map((p, i) => {
      if (!(p > 0)) return 0;
      return isInfinite(lights[i] as RtLight) ? 1 / slots : p / pArea / slots;
    });
  }
  if (selection === 'kind') {
    const kinds = ['triangle', 'sun', 'environment'] as const;
    const present = kinds.filter((k) => group((l) => l.kind === k) > 0);
    const out = new Array<number>(lights.length).fill(0);
    for (const k of present) {
      const w = scaleGroup((l) => l.kind === k, 1 / present.length, group((l) => l.kind === k));
      w.forEach((x, i) => (out[i] = (out[i] as number) + x));
    }
    return out;
  }
  const cap = selection.capped;
  if (!(cap >= 0 && cap <= 1)) throw new RangeError(`light selection: cap ${cap} must be in [0, 1]`);
  const infinite = (l: RtLight) => l.kind !== 'triangle';
  const pInf = group(infinite);
  const pArea = total - pInf;
  const share = pInf / total;
  if (pArea <= 0 || pInf <= 0 || share <= cap) return power.map((p) => p / total);
  const a = scaleGroup((l) => !infinite(l), 1 - cap, pArea);
  const b = scaleGroup(infinite, cap, pInf);
  return a.map((x, i) => x + (b[i] as number));
}

export interface LightSample {
  /** Light index, `SCENE_NONE` when there are no lights. */
  index: number;
  /** Probability that this index was chosen (f32 pmf stored in the light record). */
  pmf: number;
  /** The leftover of `u` rescaled to [0, 1): reusable for the next dimension. */
  u: number;
}

const BELOW_ONE = 1 - 2 ** -24;

/** TypeScript twin of WGSL `rtSampleLight`: same f32 operations in the same order. */
export function sampleAlias(table: Pick<AliasTable, 'prob' | 'alias' | 'pmf'>, u: number): LightSample {
  const n = table.prob.length;
  if (n === 0) return { index: SCENE_NONE, pmf: 0, u };
  const x = Math.fround(u * n);
  const i = Math.min(Math.floor(x), n - 1);
  const frac = Math.min(Math.max(Math.fround(x - i), 0), BELOW_ONE);
  const prob = table.prob[i] as number;
  const keep = frac < prob;
  const index = keep ? i : (table.alias[i] as number);
  const remap = keep ? Math.fround(frac / prob) : Math.fround(Math.fround(frac - prob) / Math.fround(1 - prob));
  return { index, pmf: table.pmf[index] as number, u: Math.min(remap, BELOW_ONE) };
}

/** The probability a table really gives index `i` (from `prob` and `alias`, not the stored pmf): used to check the table itself. */
export function aliasProbabilityOf(table: Pick<AliasTable, 'prob' | 'alias'>, i: number): number {
  const n = table.prob.length;
  let p = table.prob[i] as number;
  for (let j = 0; j < n; j++) if (j !== i && table.alias[j] === i) p += 1 - (table.prob[j] as number);
  return p / n;
}

// ---------------------------------------------------------------------------------------------------------------
// Building triangle lights
// ---------------------------------------------------------------------------------------------------------------

/** What the light builder needs of a mesh (a subset of `SurfaceMeshInput` plus bvh's per-triangle materials). */
export interface LightMeshInput {
  positions: Float32Array;
  indices?: ArrayLike<number> | null;
  /** Per original triangle: a material id (0xffff = none, falls back to the instance's). */
  triangleMaterials?: ArrayLike<number> | null;
}
export interface LightInstanceInput {
  instanceIndex: number;
  meshIndex: number;
  /** Column-major 4x4. */
  objectToWorld: ArrayLike<number>;
  /** bvh material (0-65534); `undefined` or `0xffffffff` = none. */
  material?: number;
  /** Default true; a hidden instance emits nothing. */
  visible?: boolean;
}

const BVH_TRIANGLE_NO_MATERIAL = 0xffff;

/** Material of one triangle as the trace walk reports it: the triangle's own, else the instance's, else none. */
export function triangleMaterialOf(mesh: LightMeshInput, instance: LightInstanceInput, triangle: number): number {
  const own = mesh.triangleMaterials ? (mesh.triangleMaterials[triangle] as number) : BVH_TRIANGLE_NO_MATERIAL;
  if (own !== undefined && own !== BVH_TRIANGLE_NO_MATERIAL && own !== SCENE_NONE) return own;
  return instance.material === undefined ? SCENE_NONE : instance.material;
}

function det3(m: ArrayLike<number>): number {
  const a = (r: number, c: number) => m[c * 4 + r] as number;
  return a(0, 0) * (a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1)) - a(0, 1) * (a(1, 0) * a(2, 2) - a(1, 2) * a(2, 0)) + a(0, 2) * (a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0));
}

/** World-space triangle in doubles from the f32 mesh and the f32 matrix, the way a reference would compute it. */
export function worldTriangle(mesh: LightMeshInput, matrix: ArrayLike<number>, triangle: number): [Vec3Tuple, Vec3Tuple, Vec3Tuple] {
  const out: Vec3Tuple[] = [];
  for (let k = 0; k < 3; k++) {
    const v = mesh.indices ? (mesh.indices[triangle * 3 + k] as number) : triangle * 3 + k;
    const x = mesh.positions[v * 3] as number;
    const y = mesh.positions[v * 3 + 1] as number;
    const z = mesh.positions[v * 3 + 2] as number;
    out.push([
      (matrix[0] as number) * x + (matrix[4] as number) * y + (matrix[8] as number) * z + (matrix[12] as number),
      (matrix[1] as number) * x + (matrix[5] as number) * y + (matrix[9] as number) * z + (matrix[13] as number),
      (matrix[2] as number) * x + (matrix[6] as number) * y + (matrix[10] as number) * z + (matrix[14] as number),
    ]);
  }
  return out as [Vec3Tuple, Vec3Tuple, Vec3Tuple];
}

export function trianglePower(area: number, radiance: readonly number[], twoSided: boolean): number {
  return Math.PI * area * luminance(radiance) * (twoSided ? 2 : 1);
}

export function triangleCountOfMesh(mesh: LightMeshInput): number {
  return Math.floor((mesh.indices ? mesh.indices.length : mesh.positions.length / 3) / 3);
}

export interface TriangleLightResult {
  lights: RtTriangleLight[];
  /** Per input instance (same order as `instances`): how its triangles map to lights. */
  perInstance: { instanceIndex: number; firstLight: number; count: number; triangleCount: number; triangles: number[] }[];
}

/**
 * Emissive triangles of every visible instance, in instance order then triangle order. A triangle is emissive when its
 * resolved material has emission luminance above zero; a degenerate triangle (zero or non-finite area) is not a light.
 */
export function buildTriangleLights(meshes: readonly LightMeshInput[], instances: readonly LightInstanceInput[], materials: readonly RtMaterial[]): TriangleLightResult {
  const lights: RtTriangleLight[] = [];
  const perInstance: TriangleLightResult['perInstance'] = [];
  for (const instance of instances) {
    const mesh = meshes[instance.meshIndex];
    if (!mesh) throw new RangeError(`lights: instance ${instance.instanceIndex} uses mesh ${instance.meshIndex}, which does not exist`);
    const first = lights.length;
    const triangleIds: number[] = [];
    const triangleCount = triangleCountOfMesh(mesh);
    if (instance.visible !== false) {
      const sign = det3(instance.objectToWorld) < 0 ? -1 : 1;
      for (let t = 0; t < triangleCount; t++) {
        const id = triangleMaterialOf(mesh, instance, t);
        const material = id < materials.length ? (materials[id] as RtMaterial) : undefined;
        if (!material || !(material.emissionLuminance > 0)) continue;
        const [a, b, c] = worldTriangle(mesh, instance.objectToWorld, t);
        const e1: Vec3Tuple = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        const e2: Vec3Tuple = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        const cross: Vec3Tuple = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
        const len = Math.hypot(cross[0], cross[1], cross[2]);
        if (!(len > 0) || !Number.isFinite(len)) continue;
        const area = 0.5 * len;
        const f = Math.fround;
        // Cross of the world edges flips with a mirror; the authored front side does not (it moves with the normal matrix).
        const normal: Vec3Tuple = [f((sign * cross[0]) / len), f((sign * cross[1]) / len), f((sign * cross[2]) / len)];
        const radiance = material.emissionColor;
        lights.push({
          kind: 'triangle',
          instance: instance.instanceIndex,
          triangle: t,
          material: id,
          p0: [f(a[0]), f(a[1]), f(a[2])],
          edge1: [f(e1[0]), f(e1[1]), f(e1[2])],
          edge2: [f(e2[0]), f(e2[1]), f(e2[2])],
          normal,
          area: f(area),
          radiance: [radiance[0], radiance[1], radiance[2]],
          twoSided: material.emissionTwoSided,
          power: trianglePower(f(area), radiance, material.emissionTwoSided),
        });
        triangleIds.push(t);
      }
    }
    perInstance.push({ instanceIndex: instance.instanceIndex, firstLight: first, count: lights.length - first, triangleCount, triangles: triangleIds });
  }
  return { lights, perInstance };
}

// ---------------------------------------------------------------------------------------------------------------
// Sun and environment, record pack / unpack
// ---------------------------------------------------------------------------------------------------------------

export function makeSun(input: SunInput, sceneRadius: number, powerScale = 1): RtSunLight {
  const d = input.direction;
  const len = Math.hypot(d[0], d[1], d[2]);
  if (!(len > 0)) throw new RangeError('sun: direction has zero length');
  const angularRadius = input.angularRadius ?? 0;
  if (!(angularRadius >= 0 && angularRadius < Math.PI / 2)) throw new RangeError(`sun: angularRadius ${angularRadius} must be in [0, pi/2)`);
  const f = Math.fround;
  const irradiance: Rgb3 = [f(input.irradiance[0]), f(input.irradiance[1]), f(input.irradiance[2])];
  return {
    kind: 'sun',
    direction: [f(-d[0] / len) + 0, f(-d[1] / len) + 0, f(-d[2] / len) + 0],
    angularRadius: f(angularRadius),
    solidAngle: f(2 * Math.PI * (1 - Math.cos(angularRadius))),
    irradiance,
    power: luminance(irradiance) * Math.PI * sceneRadius * sceneRadius * powerScale,
  };
}

export function makeEnvironment(input: EnvironmentInput, sceneRadius: number, powerScale = 1, map?: { base: number; width: number; height: number; pdfNorm: number; mean: Rgb3 }): RtEnvironmentLight {
  const f = Math.fround;
  const scale = input.scale ?? 1;
  const given = input.radiance ?? map?.mean ?? [0, 0, 0];
  const radiance: Rgb3 = [f(given[0]), f(given[1]), f(given[2])];
  return {
    kind: 'environment',
    radiance,
    mapBase: map ? map.base : 0,
    mapWidth: map ? map.width : 0,
    mapHeight: map ? map.height : 0,
    pdfNorm: map ? f(map.pdfNorm) : 0,
    scale: f(scale),
    rotationY: f(input.rotationY ?? 0),
    mapSlot: map ? 0 : (input.mapSlot ?? SCENE_NONE),
    power: luminance(radiance) * scale * 4 * Math.PI * Math.PI * sceneRadius * sceneRadius * powerScale,
  };
}

/** Writes one 6-vec4 light record (24 words). `pmf`, `aliasProb`, `alias` come from the alias table. */
export function packLight(light: RtLight, pmf: number, aliasProb: number, alias: number, words: Uint32Array, wordOffset: number): void {
  const w = (i: number, v: number) => {
    words[wordOffset + i] = v >>> 0;
  };
  const put3 = (at: number, v: readonly number[]) => v.forEach((c, k) => w(at + k, f32Bits(c)));
  for (let i = 0; i < LIGHT_RECORD_VEC4S * 4; i++) w(i, 0);
  w(LIGHT_WORD.pmf, f32Bits(pmf));
  w(LIGHT_WORD.aliasProb, f32Bits(aliasProb));
  w(LIGHT_WORD.alias, alias);
  if (light.kind === 'triangle') {
    w(LIGHT_WORD.kind, LIGHT_KIND.triangle);
    put3(LIGHT_WORD.position, light.p0);
    put3(LIGHT_WORD.edge1, light.edge1);
    w(LIGHT_WORD.area, f32Bits(light.area));
    put3(LIGHT_WORD.edge2, light.edge2);
    put3(LIGHT_WORD.normal, light.normal);
    w(LIGHT_WORD.material, light.material);
    put3(LIGHT_WORD.radiance, light.radiance);
    w(LIGHT_WORD.instance, light.instance);
    w(LIGHT_WORD.triangle, light.triangle);
    w(LIGHT_WORD.flags, light.twoSided ? LIGHT_FLAG.twoSided : 0);
  } else if (light.kind === 'sun') {
    w(LIGHT_WORD.kind, LIGHT_KIND.sun);
    put3(LIGHT_WORD.position, light.direction);
    w(LIGHT_WORD.sunAngularRadius, f32Bits(light.angularRadius));
    w(LIGHT_WORD.sunCosRadius, f32Bits(Math.cos(light.angularRadius)));
    w(LIGHT_WORD.sunSolidAngle, f32Bits(light.solidAngle));
    w(LIGHT_WORD.material, SCENE_NONE);
    put3(LIGHT_WORD.radiance, light.irradiance);
    w(LIGHT_WORD.instance, SCENE_NONE);
    w(LIGHT_WORD.triangle, SCENE_NONE);
  } else {
    w(LIGHT_WORD.kind, LIGHT_KIND.environment);
    w(LIGHT_WORD.envScale, f32Bits(light.scale));
    w(LIGHT_WORD.envRotation, f32Bits(light.rotationY));
    w(LIGHT_WORD.envMapSlot, light.mapSlot);
    w(LIGHT_WORD.envMapBase, light.mapBase);
    w(LIGHT_WORD.envMapSize, (light.mapWidth | (light.mapHeight << 16)) >>> 0);
    w(LIGHT_WORD.envPdfNorm, f32Bits(light.pdfNorm));
    put3(LIGHT_WORD.radiance, light.radiance);
    w(LIGHT_WORD.instance, SCENE_NONE);
    w(LIGHT_WORD.triangle, SCENE_NONE);
  }
}

/** A decoded light record: the TypeScript twin of WGSL `rtLight` / `rtSun` / `rtEnvironment`. */
export interface UnpackedLight {
  light: RtLight;
  pmf: number;
  aliasProb: number;
  alias: number;
}

export function unpackLight(words: ArrayLike<number>, wordOffset = 0): UnpackedLight {
  const w = (i: number) => (words[wordOffset + i] as number) >>> 0;
  const f = (i: number) => bitsF32(w(i));
  const f3 = (i: number): Vec3Tuple => [f(i), f(i + 1), f(i + 2)];
  const kind = w(LIGHT_WORD.kind);
  const common = { pmf: f(LIGHT_WORD.pmf), aliasProb: f(LIGHT_WORD.aliasProb), alias: w(LIGHT_WORD.alias) };
  if (kind === LIGHT_KIND.triangle) {
    const flags = w(LIGHT_WORD.flags);
    const radiance = f3(LIGHT_WORD.radiance);
    return {
      ...common,
      light: {
        kind: 'triangle',
        instance: w(LIGHT_WORD.instance),
        triangle: w(LIGHT_WORD.triangle),
        material: w(LIGHT_WORD.material),
        p0: f3(LIGHT_WORD.position),
        edge1: f3(LIGHT_WORD.edge1),
        edge2: f3(LIGHT_WORD.edge2),
        normal: f3(LIGHT_WORD.normal),
        area: f(LIGHT_WORD.area),
        radiance,
        twoSided: (flags & LIGHT_FLAG.twoSided) !== 0,
        power: trianglePower(f(LIGHT_WORD.area), radiance, (flags & LIGHT_FLAG.twoSided) !== 0),
      },
    };
  }
  if (kind === LIGHT_KIND.sun) {
    return {
      ...common,
      light: { kind: 'sun', direction: f3(LIGHT_WORD.position), angularRadius: f(LIGHT_WORD.sunAngularRadius), solidAngle: f(LIGHT_WORD.sunSolidAngle), irradiance: f3(LIGHT_WORD.radiance), power: 0 },
    };
  }
  if (kind === LIGHT_KIND.environment) {
    const size = w(LIGHT_WORD.envMapSize);
    return {
      ...common,
      light: {
        kind: 'environment',
        radiance: f3(LIGHT_WORD.radiance),
        mapBase: w(LIGHT_WORD.envMapBase),
        mapWidth: size & 0xffff,
        mapHeight: size >>> 16,
        pdfNorm: f(LIGHT_WORD.envPdfNorm),
        scale: f(LIGHT_WORD.envScale),
        rotationY: f(LIGHT_WORD.envRotation),
        mapSlot: w(LIGHT_WORD.envMapSlot),
        power: 0,
      },
    };
  }
  throw new RangeError(`light record: unknown kind ${kind}`);
}

