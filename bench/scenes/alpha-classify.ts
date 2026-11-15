/**
 * R2b: alpha MASK card analysis on the CPU, from the scene copy's inputs (no GPU, no kernel code).
 *
 *  - `alphaCoverageOf(input)`: per material, the fraction of the texture that is solid (alpha x opacity >= cutoff); 1 for a material without a cutout.
 *    The `coverage` / `stochastic` shadow modes read it (`integrator.alphaCoverage`).
 *  - `classifyAlphaCards(input)`: an opacity-micromap-like 2-state classification at triangle granularity. The kernel decides a hit by the bilinear alpha of the
 *    STORED level-0 texture (`buildSceneTextures` may resample a non-power-of-two texture, so the stored texels, not the input ones, are used). For every triangle
 *    the extrema of that bilinear function over the triangle are computed exactly (the function is bilinear per patch of four texel centres, so its extrema over a
 *    convex polygon lie on the polygon's boundary, where it is a quadratic): all of it at or above the cutoff -> `solid` (always a hit: moved to an opaque-twin
 *    instance), all below -> `transparent` (never a hit: dropped), otherwise `mixed` (stays a cutout card, instance mask `mixedMask`). The mixed cards are the only
 *    geometry the exact cutout loop has to visit (`integrator.alphaShadow: 'classified'`). A triangle that touches the cutoff contour only along an edge counts
 *    as pure (zero-measure contact; the GPU's 8-bit filter weights move the contour by about 1/256 texel anyway). Level-0 lookups only (`textureLod: 0`).
 */
import { ALPHA_MODE, buildSceneTextures, createMaterial, type RtMaterialInput, type SceneCopyMesh, type SceneTextures, type TextureWrap } from '@pmndrs/raytracer';
import type { DescSceneCopyInput } from './to-scene-copy.ts';

/** Instance mask bit of mixed cards (`integrator.alphaMixedMask`). Solid geometry keeps `INSTANCE_MASK_PLAIN`. */
export const INSTANCE_MASK_MIXED = 0x04;

interface Tex {
  width: number;
  height: number;
  data: ArrayLike<number>;
}

/** One cutout material's texture slot and thresholds, or null when the material is not a textured cutout. */
function cutoutOf(m: RtMaterialInput, textures: readonly Tex[] | undefined): { slot: number; tex: Tex; opacity: number; cutoff: number } | null {
  const mat = createMaterial(m);
  if (mat.alphaMode !== ALPHA_MODE.mask || mat.emissionLuminance > 0) return null;
  const slot = mat.baseColorTexture;
  const tex = textures?.[slot];
  if (!tex) return null;
  return { slot, tex, opacity: mat.opacity, cutoff: mat.alphaCutoff };
}

/** Fraction of the texels of each material's cutout texture that are solid (1 for every material that is not a textured cutout). */
export function alphaCoverageOf(input: Pick<DescSceneCopyInput, 'materials' | 'textures'>): number[] {
  return input.materials.map((m) => {
    const c = cutoutOf(m, input.textures);
    if (!c) return 1;
    let solid = 0;
    const n = c.tex.width * c.tex.height;
    for (let i = 0; i < n; i++) if (((c.tex.data[i * 4 + 3] as number) / 255) * c.opacity >= c.cutoff) solid++;
    return solid / n;
  });
}

export type AlphaClass = 'transparent' | 'solid' | 'mixed';

export interface AlphaClassStats {
  mesh: string;
  material: string;
  triangles: number;
  transparent: number;
  solid: number;
  mixed: number;
}

/** The stored level-0 alpha of a slot (what the kernel filters), unorm in [0, 1], with the bucket's address mode. */
export interface AlphaLevel {
  width: number;
  height: number;
  wrap: TextureWrap;
  at(i: number, j: number): number;
}

function wrapIndex(i: number, n: number, wrap: TextureWrap): number {
  if (wrap === 'clamp') return Math.min(n - 1, Math.max(0, i));
  const m = ((i % (2 * n)) + 2 * n) % (2 * n);
  if (wrap === 'mirror') return m < n ? m : 2 * n - 1 - m;
  return m % n;
}

export function alphaLevelOf(textures: SceneTextures, slot: number): AlphaLevel {
  const s = textures.slots[slot];
  if (!s) throw new RangeError(`alphaLevelOf: no texture slot ${slot}`);
  const bucket = textures.buckets[s.bucket]!;
  const base = s.layer * s.width * s.height * 4;
  const mip = bucket.mips[0]!;
  return { width: s.width, height: s.height, wrap: bucket.wrap, at: (i, j) => (mip[base + (wrapIndex(j, s.height, bucket.wrap) * s.width + wrapIndex(i, s.width, bucket.wrap)) * 4 + 3] as number) / 255 };
}

type Pt = readonly [number, number];

/** Sutherland-Hodgman against the box [x0, x1] x [y0, y1]. */
function clipToBox(poly: readonly Pt[], x0: number, y0: number, x1: number, y1: number): Pt[] {
  let out: Pt[] = [...poly];
  const planes: Array<[(p: Pt) => number]> = [[(p) => p[0] - x0], [(p) => x1 - p[0]], [(p) => p[1] - y0], [(p) => y1 - p[1]]];
  for (const [d] of planes) {
    const input = out;
    out = [];
    for (let k = 0; k < input.length; k++) {
      const a = input[k] as Pt;
      const b = input[(k + 1) % input.length] as Pt;
      const da = d(a);
      const db = d(b);
      if (da >= 0) out.push(a);
      if ((da >= 0) !== (db >= 0)) {
        const t = da / (da - db);
        out.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]);
      }
    }
    if (out.length === 0) break;
  }
  return out;
}

function polygonArea(p: readonly Pt[]): number {
  let a = 0;
  for (let k = 0; k < p.length; k++) {
    const u = p[k] as Pt;
    const v = p[(k + 1) % p.length] as Pt;
    a += u[0] * v[1] - v[0] * u[1];
  }
  return Math.abs(a) / 2;
}

/** Min and max of `f` (bilinear: no interior extremum) over a convex polygon: vertices plus the stationary point of the quadratic along every edge. */
function extrema(poly: readonly Pt[], f: (x: number, y: number) => number): [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  const see = (v: number): void => {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  };
  for (let k = 0; k < poly.length; k++) {
    const p = poly[k] as Pt;
    const q = poly[(k + 1) % poly.length] as Pt;
    const g = (t: number): number => f(p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]));
    const g0 = g(0);
    const g1 = g(1);
    see(g0);
    see(g1);
    const a = 2 * g1 + 2 * g0 - 4 * g(0.5);
    if (Math.abs(a) > 1e-12) {
      const t = -(g1 - g0 - a) / (2 * a);
      if (t > 0 && t < 1) see(g(t));
    }
  }
  return [lo, hi];
}

/** Zero-measure contact tolerance of the classification (alpha units). */
const TOUCH = 1e-6;

/** Classify one triangle (uv, v down, in texture units 0..1) against a level-0 alpha; `scale` is the material opacity. */
export function classifyTriangle(uv: readonly number[], level: AlphaLevel, opacity: number, cutoff: number): AlphaClass {
  const W = level.width;
  const H = level.height;
  const tri: Pt[] = [[uv[0]! * W, uv[1]! * H], [uv[2]! * W, uv[3]! * H], [uv[4]! * W, uv[5]! * H]];
  const minX = Math.min(tri[0]![0], tri[1]![0], tri[2]![0]);
  const maxX = Math.max(tri[0]![0], tri[1]![0], tri[2]![0]);
  const minY = Math.min(tri[0]![1], tri[1]![1], tri[2]![1]);
  const maxY = Math.max(tri[0]![1], tri[1]![1], tri[2]![1]);
  const i0 = Math.floor(minX - 0.5);
  const i1 = Math.floor(maxX - 0.5);
  const j0 = Math.floor(minY - 0.5);
  const j1 = Math.floor(maxY - 0.5);
  if ((i1 - i0 + 1) * (j1 - j0 + 1) > 4 * W * H) return 'mixed'; // wraps the texture several times: not worth the test
  let lo = Infinity;
  let hi = -Infinity;
  for (let j = j0; j <= j1; j++) {
    for (let i = i0; i <= i1; i++) {
      const poly = clipToBox(tri, i + 0.5, j + 0.5, i + 1.5, j + 1.5);
      if (poly.length < 3 || polygonArea(poly) < 1e-9) continue;
      const A = level.at(i, j) * opacity;
      const B = level.at(i + 1, j) * opacity;
      const C = level.at(i, j + 1) * opacity;
      const D = level.at(i + 1, j + 1) * opacity;
      const f = (x: number, y: number): number => {
        const s = x - (i + 0.5);
        const t = y - (j + 0.5);
        return (1 - s) * (1 - t) * A + s * (1 - t) * B + (1 - s) * t * C + s * t * D;
      };
      const [a, b] = extrema(poly, f);
      if (a < lo) lo = a;
      if (b > hi) hi = b;
      if (lo < cutoff - TOUCH && hi > cutoff + TOUCH) return 'mixed';
    }
  }
  if (lo >= cutoff - TOUCH) return 'solid';
  if (hi <= cutoff + TOUCH) return 'transparent';
  return 'mixed';
}

export interface ClassifyOptions {
  /** Instance mask bit of the mixed cards. Default `INSTANCE_MASK_MIXED`. */
  mixedMask?: number;
  /** Drop the transparent triangles (they can never be hit at level 0). Default true. */
  dropTransparent?: boolean;
}

/**
 * Split every cutout instance into a mixed part (the original MASK material, instance mask `mixedMask`) and a solid part (an opaque-twin material: same
 * everything but `alphaMode` opaque, default instance mask), and drop the transparent triangles. Returns a new input; the argument is not modified.
 */
export function classifyAlphaCards(input: DescSceneCopyInput, options: ClassifyOptions = {}): { input: DescSceneCopyInput; stats: AlphaClassStats[] } {
  const mixedMask = options.mixedMask ?? INSTANCE_MASK_MIXED;
  const drop = options.dropTransparent ?? true;
  const meshes: SceneCopyMesh[] = [...input.meshes];
  const instances = input.instances.map((i) => ({ ...i }));
  const materials: RtMaterialInput[] = [...input.materials];
  const materialNames = [...input.materialNames];
  const meshNames = [...input.meshNames];
  const twin = new Map<number, number>();
  const stats: AlphaClassStats[] = [];
  const cache = new Map<string, { mixed: number[]; solid: number[]; transparent: number[] }>();
  const stored: SceneTextures | null = input.textures && input.textures.length > 0 ? buildSceneTextures(input.textures, input.options?.textures) : null;

  input.instances.forEach((inst, k) => {
    const material = inst.material;
    if (material === undefined || !stored) return;
    const cut = cutoutOf(input.materials[material] as RtMaterialInput, input.textures);
    const mesh = input.meshes[inst.meshIndex] as SceneCopyMesh;
    if (!cut || !mesh.uvs) return;
    const key = `${inst.meshIndex}/${material}`;
    let split = cache.get(key);
    if (!split) {
      split = { mixed: [], solid: [], transparent: [] };
      const level = alphaLevelOf(stored, cut.slot);
      const count = mesh.indices ? mesh.indices.length / 3 : mesh.positions.length / 9;
      const uvs = mesh.uvs;
      const vi = (t: number, c: number): number => (mesh.indices ? (mesh.indices[t * 3 + c] as number) : t * 3 + c);
      for (let t = 0; t < count; t++) {
        const uv: number[] = [];
        for (let c = 0; c < 3; c++) uv.push(uvs[vi(t, c) * 2] as number, uvs[vi(t, c) * 2 + 1] as number);
        split[classifyTriangle(uv, level, cut.opacity, cut.cutoff)].push(t);
      }
      cache.set(key, split);
      stats.push({ mesh: input.meshNames[inst.meshIndex] ?? String(inst.meshIndex), material: input.materialNames[material] ?? String(material), triangles: count, transparent: split.transparent.length, solid: split.solid.length, mixed: split.mixed.length });
    }
    const subMesh = (tris: readonly number[]): SceneCopyMesh => {
      const idx = new Uint32Array(tris.length * 3);
      tris.forEach((t, n) => {
        for (let c = 0; c < 3; c++) idx[n * 3 + c] = mesh.indices ? (mesh.indices[t * 3 + c] as number) : t * 3 + c;
      });
      return { ...mesh, indices: idx };
    };
    const mixedTris = split.mixed;
    const solidTris = [...split.solid, ...(drop ? [] : split.transparent)].sort((a, b) => a - b);
    const slot = instances[k] as (typeof instances)[number];
    // the original instance slot keeps the first non-empty part; the other becomes a new instance
    const parts: { tris: readonly number[]; solid: boolean }[] = [];
    if (mixedTris.length > 0) parts.push({ tris: mixedTris, solid: false });
    if (solidTris.length > 0) parts.push({ tris: solidTris, solid: true });
    if (parts.length === 0) {
      slot.visible = false;
      return;
    }
    const twinOf = (): number => {
      let tIdx = twin.get(material);
      if (tIdx === undefined) {
        tIdx = materials.length;
        twin.set(material, tIdx);
        materials.push({ ...(input.materials[material] as RtMaterialInput), alphaMode: ALPHA_MODE.opaque });
        materialNames.push(`${input.materialNames[material] ?? material}+solid`);
      }
      return tIdx;
    };
    parts.forEach((part, n) => {
      const meshIndex = meshes.length;
      meshes.push(subMesh(part.tris));
      meshNames.push(`${input.meshNames[inst.meshIndex] ?? inst.meshIndex}${part.solid ? '+solid' : '+mixed'}`);
      const base = { ...inst, meshIndex, material: part.solid ? twinOf() : material, mask: part.solid ? inst.mask : mixedMask };
      if (n === 0) Object.assign(slot, base);
      else instances.push({ ...base, instanceIndex: instances.length });
    });
  });
  return { input: { ...input, meshes, instances, materials, materialNames, meshNames }, stats };
}
