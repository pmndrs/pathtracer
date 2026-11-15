/**
 * Scene -> Wavefront OBJ + MTL for VkNRC (track N, N6.2 / WN6), the native NRC baseline (docs/design/nrc.md §9).
 *
 * Our own code. What VkNRC's loader needs was read from its source at `a667859` (no licence, D28/D29: read, never copied) and
 * is restated here in words:
 * - It loads one OBJ through tinyobjloader, triangles only, and refuses a file with no material: every face needs a `usemtl`.
 * - It reads positions, optional texture coordinates and the MTL's `Kd`, `Ks`, `Ke` (plus textures, metallic, roughness, ior).
 *   It does NOT read normals: shading uses the flat geometric normal of each triangle, flipped toward the ray (two-sided).
 *   So we write no `vn` lines, and welding bit-identical positions loses nothing.
 * - Its BRDF is Cook-Torrance: `Kd / pi` plus a Beckmann lobe weighted by `Ks`. `Ks 0 0 0` makes it pure Lambert, which is what
 *   our `diffuse` material is.
 * - It rescales the model to the box [-1, 1] (centre of the vertex AABB to the origin, longest extent to 2). The sidecar
 *   (`vknrc.json`) gives that transform and the scene camera in VkNRC's own terms (normalised position, yaw, pitch, vertical fov), for
 *   the fixed-camera patch of WN7.
 * - Lighting is fixed in VkNRC: a constant white sky of radiance 10 and no sun or NEE; paths end after 8 vertices without Russian roulette.
 *   Scene lights and the environment are therefore not exported; `sponza-white-vknrc` is our scene with the same lighting.
 */
import type { MaterialDesc, SceneDesc } from '../scenes/schema.ts';
import { worldMeshes } from './common.ts';
import type { ExportOptions, ExportResult } from './types.ts';

/** VkNRC's fixed lighting and path length (its path tracer, `a667859`): constant sky radiance, no sun, at most this many surface vertices scatter toward the sky. */
export const VKNRC_SKY_RADIANCE = 10;
/** Our `maxDepth` equivalent of VkNRC's 8-vertex loop: vertices 1..7 sample a direction that can reach the sky, vertex 8 only adds its emission. */
export const VKNRC_MAX_DEPTH = 7;

/** Shortest decimal that parses back to the same float32 (tinyobjloader reads floats). */
export function f32Text(v: number): string {
  const f = Math.fround(v);
  if (f === 0) return '0';
  for (let p = 6; p < 10; p++) {
    const s = Number(f.toPrecision(p)).toString();
    if (Math.fround(Number(s)) === f) return s;
  }
  return f.toString();
}

export interface VknrcSidecar {
  scene: string;
  /** VkNRC's load transform: p' = (p - center) * scale (centre of the vertex AABB, scale = 2 / longest extent). */
  center: [number, number, number];
  scale: number;
  aabb: { min: [number, number, number]; max: [number, number, number] };
  /** The scene camera in VkNRC's normalised frame: look = (-sin yaw cos pitch, sin pitch, -cos yaw cos pitch), no roll, vertical fov in radians. */
  camera: { position: [number, number, number]; yaw: number; pitch: number; fovY: number; width: number; height: number };
  vertices: number;
  triangles: number;
  approximations: string[];
}

/** Camera look direction (unit) -> VkNRC yaw/pitch (its camera looks along -Z at yaw = pitch = 0, yaw about +Y, pitch about +X). */
export function vknrcYawPitch(look: readonly [number, number, number]): { yaw: number; pitch: number } {
  const l = Math.hypot(look[0], look[1], look[2]);
  const d = [look[0] / l, look[1] / l, look[2] / l] as const;
  return { yaw: Math.atan2(-d[0], -d[2]), pitch: Math.asin(Math.max(-1, Math.min(1, d[1]))) };
}

/** Inverse of `vknrcYawPitch` (VkNRC's camera basis, for tests and for reading a pose back). */
export function vknrcLook(yaw: number, pitch: number): [number, number, number] {
  return [-Math.sin(yaw) * Math.cos(pitch), Math.sin(pitch), -Math.cos(yaw) * Math.cos(pitch)];
}

function mtlFor(name: string, m: MaterialDesc, emission: readonly number[] | undefined, approx: Set<string>): string {
  let kd: readonly number[];
  switch (m.type) {
    case 'diffuse':
      if (m.reflectanceTexture) throw new Error(`obj: material '${name}' has a reflectance texture; the VkNRC export writes constant Kd only`);
      kd = m.reflectance;
      break;
    case 'black':
      kd = [0, 0, 0];
      break;
    default:
      throw new Error(`obj: material '${name}' is type ${m.type}; the VkNRC export maps only diffuse and black (VkNRC's Cook-Torrance with Ks 0 is Lambert)`);
  }
  if (emission) approx.add('emission: written as Ke; VkNRC emits from both faces and finds emitters only by BSDF sampling (no NEE).');
  const ke = emission ?? [0, 0, 0];
  // `Pr 1`: the specular lobe has weight 0 anyway; roughness 1 keeps its Beckmann term far from VkNRC's 0.001 clamp (no inf x 0).
  return `newmtl ${name}\nKd ${kd.map(f32Text).join(' ')}\nKs 0 0 0\nKe ${ke.map(f32Text).join(' ')}\nPr 1\nillum 1\n`;
}

/**
 * The OBJ export. `scene.obj` (positions and faces, bit-identical positions welded, one `usemtl` per mesh), `scene.mtl` and `vknrc.json`.
 * Only `diffuse` (untextured) and `black` materials; anything else throws, so VkNRC never renders a scene we cannot match.
 */
export function toObj(scene: SceneDesc, opts: Pick<ExportOptions, 'assetsDir'> & Partial<ExportOptions>): ExportResult & { sidecar: VknrcSidecar } {
  const approx = new Set<string>();
  const meshes = worldMeshes(scene, opts.assetsDir);
  const mtl: string[] = [`# ${scene.name}: pathtracer bench export for VkNRC (bench/exporters/obj.ts)\n`];
  const mtlNames = new Map<string, string>();
  const materialOf = (desc: (typeof meshes)[number]['desc']): string => {
    const key = `${desc.material}|${desc.emission ? desc.emission.join(',') : ''}`;
    let name = mtlNames.get(key);
    if (!name) {
      const m = scene.materials[desc.material];
      if (!m) throw new Error(`obj: unknown material '${desc.material}'`);
      name = `${desc.material.replace(/[^A-Za-z0-9_-]+/g, '_')}${desc.emission ? `_e${mtlNames.size}` : ''}`;
      mtl.push(mtlFor(name, m, desc.emission, approx));
      mtlNames.set(key, name);
    }
    return name;
  };

  // Weld bit-identical float32 positions (VkNRC reads no normals, so this loses nothing).
  const index = new Map<string, number>();
  const verts: number[] = [];
  const f32 = new Float32Array(3);
  const u32 = new Uint32Array(f32.buffer);
  const vertexOf = (x: number, y: number, z: number): number => {
    f32[0] = x; f32[1] = y; f32[2] = z;
    const key = `${u32[0]},${u32[1]},${u32[2]}`;
    let i = index.get(key);
    if (i === undefined) {
      i = verts.length / 3;
      verts.push(f32[0]!, f32[1]!, f32[2]!);
      index.set(key, i);
    }
    return i;
  };
  const faceLines: string[] = [];
  let triangles = 0;
  for (const { desc, mesh } of meshes) {
    faceLines.push(`o ${desc.name.replace(/\s+/g, '_')}\nusemtl ${materialOf(desc)}\n`);
    const p = mesh.positions;
    const idx = mesh.indices;
    for (let t = 0; t < idx.length; t += 3) {
      const c = [0, 1, 2].map((k) => {
        const v = idx[t + k]! * 3;
        return vertexOf(p[v]!, p[v + 1]!, p[v + 2]!) + 1;
      });
      faceLines.push(`f ${c[0]} ${c[1]} ${c[2]}\n`);
      triangles++;
    }
    if (desc.twoSided === false && desc.emission) approx.add('emission: one-sided emitters become two-sided in VkNRC.');
  }
  if (verts.length === 0) throw new Error(`obj: scene '${scene.name}' has no triangles`);

  // VkNRC's normalisation, in float32 like its loader.
  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < verts.length; i += 3) {
    for (let a = 0; a < 3; a++) {
      min[a] = Math.min(min[a]!, verts[i + a]!);
      max[a] = Math.max(max[a]!, verts[i + a]!);
    }
  }
  const center = [0, 1, 2].map((a) => Math.fround((min[a]! + max[a]!) * 0.5)) as [number, number, number];
  const scale = Math.fround(2 / Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2]));

  const cam = scene.camera;
  if (cam.up[0] !== 0 || cam.up[1] <= 0 || cam.up[2] !== 0) approx.add('camera: up is not +Y; VkNRC has no roll, the pose drops it.');
  const look: [number, number, number] = [cam.target[0] - cam.position[0], cam.target[1] - cam.position[1], cam.target[2] - cam.position[2]];
  const { yaw, pitch } = vknrcYawPitch(look);
  const position = [0, 1, 2].map((a) => (cam.position[a]! - center[a]!) * scale) as [number, number, number];

  if ((scene.lights ?? []).length > 0) approx.add(`lights: ${scene.lights!.length} light(s) not exported (VkNRC has no lights besides its sky); render sponza-white-vknrc for the matching image.`);
  const env = scene.environment;
  if (!env || env.type !== 'uniform' || env.radiance.some((c) => c !== VKNRC_SKY_RADIANCE)) {
    approx.add(`environment: VkNRC's sky is a constant ${VKNRC_SKY_RADIANCE}; the scene's environment is not exported.`);
  }
  if (scene.integrator.maxDepth !== VKNRC_MAX_DEPTH || scene.integrator.rrStartDepth !== undefined) {
    approx.add(`integrator: VkNRC scatters at up to ${VKNRC_MAX_DEPTH} vertices (8 hits), no Russian roulette; the scene asks maxDepth ${scene.integrator.maxDepth}${scene.integrator.rrStartDepth !== undefined ? `, RR from ${scene.integrator.rrStartDepth}` : ''}.`);
  }
  const approximations = [...approx];

  const vertexLines: string[] = [];
  for (let i = 0; i < verts.length; i += 3) vertexLines.push(`v ${f32Text(verts[i]!)} ${f32Text(verts[i + 1]!)} ${f32Text(verts[i + 2]!)}\n`);
  const obj =
    `# ${scene.name}: pathtracer bench export for VkNRC (bench/exporters/obj.ts). Positions only (VkNRC uses flat geometric normals).\n` +
    `# ${verts.length / 3} vertices (bit-identical positions welded), ${triangles} triangles\n` +
    `mtllib scene.mtl\n` + vertexLines.join('') + faceLines.join('');

  const sidecar: VknrcSidecar = {
    scene: scene.name,
    center,
    scale,
    aabb: { min, max },
    camera: { position, yaw, pitch, fovY: (cam.yFovDeg * Math.PI) / 180, width: scene.film.width, height: scene.film.height },
    vertices: verts.length / 3,
    triangles,
    approximations,
  };
  return {
    sceneFile: 'scene.obj',
    files: [
      { path: 'scene.obj', data: obj },
      { path: 'scene.mtl', data: mtl.join('\n') },
      { path: 'vknrc.json', data: `${JSON.stringify(sidecar, null, 2)}\n` },
    ],
    approximations,
    sidecar,
  };
}
