/**
 * Scenes and rays for the trace-core pages (trace-core, trace-timing). Not a page itself (no .html).
 *
 * Scenes are meshes in object space plus instances, so the walk's instance transforms (non-uniform scale, rotation,
 * mirror) are exercised, not only identity:
 *  - `cornell`, `quad-light`: the Tier 0 scenes, tessellated by bench/scenes/geometry.ts, plus a mirrored copy of one
 *    mesh placed outside the room;
 *  - `sponza-white`: the Tier 2 Sponza from the web glb export (`node bench/adapters/export-scenes.ts`), plus a
 *    mirrored, scaled second instance above the atrium;
 *  - `edges`: a tessellated quad (bvh's `tessellatedQuad`) under three transforms and a closed tessellated box. Rays
 *    are aimed exactly at shared edges and vertices: the watertight trap (CLAUDE.md, Metal fma contraction).
 *
 * Ray sets: primary (camera), incoherent (random origins and directions, half with random [tMin, tMax] intervals),
 * shadow (segments from primary hits to emitters / the sun / other surface points), edges (aimed at triangle edges and
 * vertices), and for `edges` `inside` (from inside the closed box: every one must hit).
 */
import { createRandom, randomUnitVector, sharedEdgeRays, tessellatedQuad, type MeshGeometry, type Random, type TraceSceneInstance } from '@pmndrs/bvh';
import { packRays, type Ray } from '@pmndrs/raytracer';
import { fetchSceneGlb } from '../adapters/shared/glb.ts';
import { compose, rotationX, rotationY, scaling, tessellate, tessellateBox, translation } from '../scenes/geometry.ts';
import type { CameraDesc, SceneDesc } from '../scenes/schema.ts';
import { cornell } from '../scenes/tier0/cornell.ts';
import { quadLight } from '../scenes/tier0/quad-light.ts';
import { sponzaWhite } from '../scenes/tier2/sponza-white.ts';

export type Vec3 = [number, number, number];

/** Column-major object-to-world times a point (bvh's `Mat4` is any `ArrayLike<number>`). */
export function transformPoint(m: ArrayLike<number>, p: readonly number[]): Vec3 {
  const [x, y, z] = p as [number, number, number];
  const e = (i: number) => m[i] as number;
  return [e(0) * x + e(4) * y + e(8) * z + e(12), e(1) * x + e(5) * y + e(9) * z + e(13), e(2) * x + e(6) * y + e(10) * z + e(14)];
}

export interface TraceMesh extends MeshGeometry {
  readonly positions: Float32Array;
  readonly indices: Uint32Array;
  /** Object-space shading normals (for rtSurface). */
  readonly normals: Float32Array;
  /** Synthetic UVs (for rtSurface): a fixed linear function of the object-space position. */
  readonly uvs: Float32Array;
}

export interface TraceTestScene {
  readonly name: string;
  readonly meshes: readonly TraceMesh[];
  readonly instances: readonly TraceSceneInstance[];
  readonly camera: CameraDesc | null;
  readonly aspect: number;
  /** World bounds `[minX, minY, minZ, maxX, maxY, maxZ]` over every instance. */
  readonly bounds: readonly number[];
  /** Diagonal of the bounds (metres): offsets and tolerances scale with it. */
  readonly scale: number;
  /** Instances whose triangles are shadow-ray targets (emitters). */
  readonly emitters: readonly number[];
  /** Direction light travels (sun), or null. */
  readonly sun: Vec3 | null;
  /** `edges` only: instances of the closed box (rays from inside must hit). */
  readonly closed: readonly number[];
}

/** Instance mask of emitters; a ray mask of `MASK_NO_EMITTERS` skips them. */
export const EMITTER_MASK = 0x02;
export const MASK_NO_EMITTERS = 0x01;

function withUvs(positions: Float32Array, indices: Uint32Array, normals: Float32Array, triangleMaterials?: Uint32Array): TraceMesh {
  const uvs = new Float32Array((positions.length / 3) * 2);
  for (let v = 0; v < positions.length / 3; v++) {
    const [x, y, z] = [positions[v * 3] as number, positions[v * 3 + 1] as number, positions[v * 3 + 2] as number];
    uvs[v * 2] = 0.37 * x + 0.11 * y - 0.23 * z + 0.5;
    uvs[v * 2 + 1] = -0.19 * x + 0.41 * y + 0.07 * z + 0.25;
  }
  return triangleMaterials ? { positions, indices, normals, uvs, triangleMaterials } : { positions, indices, normals, uvs };
}

function boundsOf(meshes: readonly TraceMesh[], instances: readonly TraceSceneInstance[]): number[] {
  const b = [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];
  for (const instance of instances) {
    const mesh = meshes[instance.meshIndex] as TraceMesh;
    for (let v = 0; v < mesh.positions.length; v += 3) {
      const p = transformPoint(instance.objectToWorld, [mesh.positions[v] as number, mesh.positions[v + 1] as number, mesh.positions[v + 2] as number]);
      for (let k = 0; k < 3; k++) {
        b[k] = Math.min(b[k] as number, p[k] as number);
        b[k + 3] = Math.max(b[k + 3] as number, p[k] as number);
      }
    }
  }
  return b;
}

function finish(partial: Omit<TraceTestScene, 'bounds' | 'scale'>): TraceTestScene {
  const bounds = boundsOf(partial.meshes, partial.instances);
  const scale = Math.hypot((bounds[3] as number) - (bounds[0] as number), (bounds[4] as number) - (bounds[1] as number), (bounds[5] as number) - (bounds[2] as number));
  return { ...partial, bounds, scale };
}

/** A Tier 0 scene: one instance per scene mesh (its transform), plus a mirrored, scaled copy of mesh `extra` outside. */
function tier0(scene: SceneDesc, extra: { mesh: number; transform: number[] }): TraceTestScene {
  const materials = Object.keys(scene.materials);
  const meshes: TraceMesh[] = [];
  const instances: TraceSceneInstance[] = [];
  const emitters: number[] = [];
  scene.meshes.forEach((desc, i) => {
    const local = tessellate(desc.geometry);
    meshes.push(withUvs(local.positions, local.indices, local.normals));
    const emitter = !!desc.emission;
    if (emitter) emitters.push(i);
    instances.push({
      meshIndex: i,
      instanceIndex: i,
      objectToWorld: desc.transform ? [...desc.transform] : compose(),
      material: materials.indexOf(desc.material),
      mask: emitter ? EMITTER_MASK : 0x01,
      static: true,
    });
  });
  instances.push({ meshIndex: extra.mesh, instanceIndex: instances.length, objectToWorld: extra.transform, material: 0, mask: 0x01, static: true });
  return finish({ name: scene.name, meshes, instances, camera: scene.camera, aspect: scene.film.width / scene.film.height, emitters, sun: null, closed: [] });
}

/** Mirror on x (negative determinant), non-uniform scale, rotation, translation. */
function mirrored(move: Vec3, scale: Vec3, angle: number): number[] {
  return compose(translation(...move), rotationY(angle), scaling(-scale[0], scale[1], scale[2]));
}

export function cornellScene(): TraceTestScene {
  // Copy of the tall block (mesh 4), mirrored, behind the back wall.
  return tier0(cornell, { mesh: 4, transform: mirrored([0.6, 0, 1.3], [1, 0.8, 1.2], 0.4) });
}

export function quadLightScene(): TraceTestScene {
  // Copy of the light quad (mesh 1), mirrored and tilted, off to the side.
  return tier0(quadLight, { mesh: 1, transform: compose(translation(3, 0.5, -2), rotationX(0.7), scaling(-2, 1, 0.5)) });
}

/** Minimal GLB reader for the bench's own export (bench/exporters/gltf.ts): float VEC3 POSITION/NORMAL, u32 indices. */
function readGlbMeshes(glb: ArrayBuffer): { positions: Float32Array; normals: Float32Array; indices: Uint32Array }[] {
  const view = new DataView(glb);
  if (view.getUint32(0, true) !== 0x46546c67) throw new Error('not a GLB');
  let offset = 12;
  let json: { accessors: { bufferView: number; byteOffset?: number; componentType: number; count: number; type: string }[]; bufferViews: { byteOffset?: number; byteLength: number; byteStride?: number }[]; meshes: { primitives: { attributes: Record<string, number>; indices?: number }[] }[]; nodes: { mesh?: number; matrix?: number[] }[] } | null = null;
  let bin: Uint8Array | null = null;
  while (offset + 8 <= glb.byteLength) {
    const length = view.getUint32(offset, true);
    const type = view.getUint32(offset + 4, true);
    const data = new Uint8Array(glb, offset + 8, length);
    if (type === 0x4e4f534a) json = JSON.parse(new TextDecoder().decode(data));
    else if (type === 0x004e4942) bin = data;
    offset += 8 + length;
  }
  if (!json || !bin) throw new Error('GLB without JSON or BIN chunk');
  const binary = bin;
  const doc = json;
  const read = (index: number, kind: 'f32' | 'u32') => {
    const accessor = doc.accessors[index];
    if (!accessor) throw new Error(`accessor ${index} missing`);
    const bufferView = doc.bufferViews[accessor.bufferView];
    if (!bufferView || bufferView.byteStride) throw new Error('strided views are not supported');
    const components = accessor.type === 'VEC3' ? 3 : 1;
    const byteOffset = binary.byteOffset + (bufferView.byteOffset ?? 0) + (accessor.byteOffset ?? 0);
    const bytes = binary.buffer.slice(byteOffset, byteOffset + accessor.count * components * 4);
    if (kind === 'f32' && accessor.componentType !== 5126) throw new Error('positions must be float');
    if (kind === 'u32' && accessor.componentType !== 5125) throw new Error('indices must be u32');
    return kind === 'f32' ? new Float32Array(bytes) : new Uint32Array(bytes);
  };
  const out: { positions: Float32Array; normals: Float32Array; indices: Uint32Array }[] = [];
  for (const node of doc.nodes) {
    if (node.mesh === undefined) continue;
    if (node.matrix) throw new Error('mesh nodes with a matrix are not expected in the bench export');
    for (const primitive of doc.meshes[node.mesh]?.primitives ?? []) {
      out.push({
        positions: read(primitive.attributes.POSITION as number, 'f32') as Float32Array,
        normals: read(primitive.attributes.NORMAL as number, 'f32') as Float32Array,
        indices: read(primitive.indices as number, 'u32') as Uint32Array,
      });
    }
  }
  return out;
}

export async function sponzaScene(): Promise<TraceTestScene> {
  const { glb } = await fetchSceneGlb(sponzaWhite);
  const meshes = readGlbMeshes(glb).map((mesh) => {
    // Per-triangle materials 0..6 so the walk's material path (triangle code, not instance code) is exercised.
    const materials = new Uint32Array(mesh.indices.length / 3);
    for (let t = 0; t < materials.length; t++) materials[t] = t % 7;
    return withUvs(mesh.positions, mesh.indices, mesh.normals, materials);
  });
  const instances: TraceSceneInstance[] = meshes.map((_, i) => ({ meshIndex: i, instanceIndex: i, objectToWorld: compose(), material: 9, mask: 0x01, static: true }));
  instances.push({ meshIndex: 0, instanceIndex: instances.length, objectToWorld: mirrored([0, 25, 0], [0.5, 0.4, 0.6], 0.3), material: 8, mask: 0x01, static: true });
  const sun = sponzaWhite.lights?.[0]?.direction as Vec3;
  return finish({ name: sponzaWhite.name, meshes, instances, camera: sponzaWhite.camera, aspect: sponzaWhite.film.width / sponzaWhite.film.height, emitters: [], sun: [...sun] as Vec3, closed: [] });
}

export function edgesScene(): TraceTestScene {
  const quad = tessellatedQuad(8, 2); // XY plane, z = 0, cells 0.5 m: grid lines are exact in f32.
  const quadPositions = quad.positions;
  const quadIndices = Uint32Array.from(quad.indices ?? []);
  const quadNormals = new Float32Array(quadPositions.length);
  for (let v = 0; v < quadPositions.length / 3; v++) quadNormals[v * 3 + 2] = 1;
  const box = tessellateBox([1, 1, 1]);
  const meshes = [withUvs(quadPositions, quadIndices, quadNormals), withUvs(box.positions, box.indices, box.normals)];
  const instances: TraceSceneInstance[] = [
    { meshIndex: 0, instanceIndex: 0, objectToWorld: compose(), material: 0, mask: 0x01, static: true },
    { meshIndex: 0, instanceIndex: 1, objectToWorld: compose(translation(6, 0.3, -1), rotationY(0.6), rotationX(-0.4), scaling(1.5, 0.7, 2)), material: 1, mask: 0x01, static: true },
    { meshIndex: 0, instanceIndex: 2, objectToWorld: mirrored([-6, -0.5, 1], [1.2, 0.9, 1], -0.8), material: 2, mask: 0x01, static: true },
    { meshIndex: 1, instanceIndex: 3, objectToWorld: compose(translation(0, 4, 0), rotationY(0.3), scaling(2, 1, 1.5)), material: 3, mask: 0x01, static: true },
  ];
  return finish({ name: 'edges', meshes, instances, camera: null, aspect: 1, emitters: [], sun: null, closed: [3] });
}

export const TRACE_SCENES = ['cornell', 'quad-light', 'sponza-white', 'edges'] as const;
export type TraceSceneName = (typeof TRACE_SCENES)[number];

export async function loadTraceScene(name: string): Promise<TraceTestScene> {
  if (name === 'cornell') return cornellScene();
  if (name === 'quad-light') return quadLightScene();
  if (name === 'sponza-white') return sponzaScene();
  if (name === 'edges') return edgesScene();
  throw new Error(`unknown trace scene '${name}' (have ${TRACE_SCENES.join(', ')})`);
}

// ---------------------------------------------------------------------------------------------------------------
// Rays
// ---------------------------------------------------------------------------------------------------------------

const sub = (a: readonly number[], b: readonly number[]): Vec3 => [(a[0] as number) - (b[0] as number), (a[1] as number) - (b[1] as number), (a[2] as number) - (b[2] as number)];
const add = (a: readonly number[], b: readonly number[]): Vec3 => [(a[0] as number) + (b[0] as number), (a[1] as number) + (b[1] as number), (a[2] as number) + (b[2] as number)];
const mul = (a: readonly number[], s: number): Vec3 => [(a[0] as number) * s, (a[1] as number) * s, (a[2] as number) * s];
const dot = (a: readonly number[], b: readonly number[]) => (a[0] as number) * (b[0] as number) + (a[1] as number) * (b[1] as number) + (a[2] as number) * (b[2] as number);
const cross = (a: readonly number[], b: readonly number[]): Vec3 => [(a[1] as number) * (b[2] as number) - (a[2] as number) * (b[1] as number), (a[2] as number) * (b[0] as number) - (a[0] as number) * (b[2] as number), (a[0] as number) * (b[1] as number) - (a[1] as number) * (b[0] as number)];
const norm = (a: readonly number[]): Vec3 => mul(a, 1 / Math.sqrt(dot(a, a)));
const lerp = (a: readonly number[], b: readonly number[], s: number): Vec3 => add(a, mul(sub(b, a), s));

/** A far tMax that stays finite in f32. */
export const FAR = 1e30;

export function cameraRays(scene: TraceTestScene, count: number, random: Random): Ray[] {
  const camera = scene.camera;
  if (!camera) return [];
  const forward = norm(sub(camera.target, camera.position));
  const right = norm(cross(forward, camera.up));
  const up = cross(right, forward);
  const tanY = Math.tan((camera.yFovDeg * Math.PI) / 360);
  const width = Math.round(Math.sqrt(count * scene.aspect));
  const height = Math.ceil(count / width);
  const rays: Ray[] = [];
  for (let i = 0; i < count; i++) {
    const px = (i % width) + random();
    const py = Math.floor(i / width) + random();
    const x = ((2 * px) / width - 1) * tanY * scene.aspect;
    const y = (1 - (2 * py) / height) * tanY;
    rays.push({ origin: [...camera.position] as Vec3, tMin: 0, direction: norm(add(forward, add(mul(right, x), mul(up, y)))), tMax: FAR });
  }
  return rays;
}

export function randomPointIn(bounds: readonly number[], random: Random, shrink = 0.98): Vec3 {
  return [0, 1, 2].map((k) => {
    const lo = bounds[k] as number;
    const hi = bounds[k + 3] as number;
    const mid = (lo + hi) / 2;
    return mid + (random() - 0.5) * (hi - lo) * shrink;
  }) as Vec3;
}

export function incoherentRays(scene: TraceTestScene, count: number, random: Random): Ray[] {
  const rays: Ray[] = [];
  for (let i = 0; i < count; i++) {
    const origin = randomPointIn(scene.bounds, random);
    const direction = randomUnitVector(random);
    if (i % 2 === 0) {
      rays.push({ origin, tMin: 0, direction, tMax: FAR });
    } else {
      // A random interval: exercises tMin > 0 and finite tMax (L1 is "closest hit over [tMin, tMax]").
      const tMin = random() * 0.3 * scene.scale;
      rays.push({ origin, tMin, direction, tMax: tMin + random() * 0.5 * scene.scale });
    }
  }
  return rays;
}

/** World vertex of instance `instance`, triangle `t`, corner `k`. */
function worldCorner(scene: TraceTestScene, instance: TraceSceneInstance, t: number, k: number): Vec3 {
  const mesh = scene.meshes[instance.meshIndex] as TraceMesh;
  const v = mesh.indices[t * 3 + k] as number;
  return transformPoint(instance.objectToWorld, [mesh.positions[v * 3] as number, mesh.positions[v * 3 + 1] as number, mesh.positions[v * 3 + 2] as number]);
}

function randomInstanceTriangle(scene: TraceTestScene, random: Random, which?: readonly number[]): { instance: TraceSceneInstance; triangle: number } {
  const pool = which && which.length ? which.map((index) => scene.instances[index] as TraceSceneInstance) : scene.instances;
  const instance = pool[Math.floor(random() * pool.length)] as TraceSceneInstance;
  const mesh = scene.meshes[instance.meshIndex] as TraceMesh;
  return { instance, triangle: Math.floor(random() * (mesh.indices.length / 3)) };
}

/** Rays aimed at a point on a triangle edge (or exactly at a vertex, one in eight) from a random point nearby. */
export function edgeRays(scene: TraceTestScene, count: number, random: Random): Ray[] {
  const rays: Ray[] = [];
  for (let i = 0; i < count; i++) {
    const { instance, triangle } = randomInstanceTriangle(scene, random);
    const a = Math.floor(random() * 3);
    const b = (a + 1) % 3;
    const s = i % 8 === 0 ? 0 : random();
    const target = lerp(worldCorner(scene, instance, triangle, a), worldCorner(scene, instance, triangle, b), s);
    const distance = (0.02 + random() * 0.3) * scene.scale;
    const origin = add(target, mul(randomUnitVector(random), distance));
    rays.push({ origin, tMin: 0, direction: norm(sub(target, origin)), tMax: FAR });
  }
  if (scene.name === 'edges') {
    // Shared-edge rays from bvh (`sharedEdgeRays`, the generator its GPU oracle page uses), aimed at the seams of the
    // quad (`tessellatedQuad(8, 2)`) under each of its three instance transforms: the plain one (seed 555, with exact
    // vertical rays on grid lines: exact in f32), then the transformed and the mirrored one (seed 777, no vertical
    // rays: through a transform a ray is no longer exactly along the line). A quarter of the set, half of that plain.
    const quarter = Math.floor(count / 4);
    const plain = Math.ceil(quarter / 2);
    const transformed = Math.floor((quarter - plain) / 2);
    const mirroredCount = quarter - plain - transformed;
    const seams = [
      sharedEdgeRays(plain, { cells: 8, half: 2, seed: 555 }).rays,
      sharedEdgeRays(transformed, { cells: 8, half: 2, objectToWorld: Array.from((scene.instances[1] as TraceSceneInstance).objectToWorld), seed: 777, verticalEvery: 0 }).rays,
      sharedEdgeRays(mirroredCount, { cells: 8, half: 2, objectToWorld: Array.from((scene.instances[2] as TraceSceneInstance).objectToWorld), seed: 777, verticalEvery: 0 }).rays,
    ].flat();
    seams.forEach((ray, i) => {
      rays[i] = { origin: [...ray.origin], tMin: ray.tMin, direction: [...ray.direction], tMax: ray.tMax };
    });
  }
  return rays;
}

/** `edges` only: rays from inside the closed box, a third aimed at its edges and corners. Every one must hit. */
export function insideRays(scene: TraceTestScene, count: number, random: Random): Ray[] {
  const rays: Ray[] = [];
  for (const index of scene.closed) {
    const instance = scene.instances[index] as TraceSceneInstance;
    for (let i = 0; i < count / scene.closed.length; i++) {
      const origin = transformPoint(instance.objectToWorld, [(random() - 0.5) * 0.8, (random() - 0.5) * 0.8, (random() - 0.5) * 0.8]);
      let direction: Vec3;
      if (i % 3 === 0) {
        const { triangle } = randomInstanceTriangle(scene, random, [index]);
        const k = Math.floor(random() * 3);
        const target = lerp(worldCorner(scene, instance, triangle, k), worldCorner(scene, instance, triangle, (k + 1) % 3), i % 6 === 0 ? 0 : random());
        direction = norm(sub(target, origin));
      } else {
        direction = randomUnitVector(random);
      }
      rays.push({ origin, tMin: 0, direction, tMax: FAR });
    }
  }
  return rays;
}

/** What shadow-ray generation needs from a closest hit (the CPU oracle's). */
export interface SurfacePoint {
  readonly position: Vec3;
  /** Geometric normal flipped to the side the ray came from. */
  readonly normal: Vec3;
}

export function surfacePoints(rays: readonly Ray[], hits: readonly ({ t: number; normal: readonly number[] } | null)[]): SurfacePoint[] {
  const out: SurfacePoint[] = [];
  hits.forEach((hit, i) => {
    if (!hit) return;
    const ray = rays[i] as Ray;
    const n = dot(hit.normal, ray.direction) > 0 ? mul(hit.normal, -1) : ([...hit.normal] as Vec3);
    out.push({ position: add(ray.origin, mul(ray.direction, hit.t)), normal: n });
  });
  return out;
}

/**
 * Shadow segments from surface points: to a random point on an emitter triangle, toward the sun, or to another surface
 * point. Both ends are pulled in by `eps` (scene-scaled), as `rtSegment` does.
 */
export function shadowRays(scene: TraceTestScene, points: readonly SurfacePoint[], count: number, random: Random): Ray[] {
  const rays: Ray[] = [];
  if (points.length === 0) return rays;
  const eps = 1e-4 * scene.scale;
  for (let i = 0; i < count; i++) {
    const from = points[Math.floor(random() * points.length)] as SurfacePoint;
    const start = add(from.position, mul(from.normal, eps));
    let end: Vec3;
    const pick = random();
    if (scene.emitters.length && pick < 0.6) {
      const { instance, triangle } = randomInstanceTriangle(scene, random, scene.emitters);
      let u = random();
      let v = random();
      if (u + v > 1) [u, v] = [1 - u, 1 - v];
      const c0 = worldCorner(scene, instance, triangle, 0);
      end = add(c0, add(mul(sub(worldCorner(scene, instance, triangle, 1), c0), u), mul(sub(worldCorner(scene, instance, triangle, 2), c0), v)));
    } else if (scene.sun && pick < 0.6) {
      end = add(start, mul(scene.sun, -2 * scene.scale));
    } else {
      end = (points[Math.floor(random() * points.length)] as SurfacePoint).position;
    }
    const d = sub(end, start);
    const length = Math.sqrt(dot(d, d));
    if (!(length > 2 * eps)) continue;
    rays.push({ origin: start, tMin: 0, direction: mul(d, 1 / length), tMax: length - eps });
  }
  return rays;
}

export { createRandom, packRays };
