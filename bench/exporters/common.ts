import type { CameraDesc, FilmDesc, MeshDesc, SceneDesc } from '../scenes/schema.ts';
import { meshToWorld, transformMesh } from '../scenes/geometry.ts';
import { ASSETS_DIR, loadFileGeometry } from '../scenes/file-geometry.ts';
import type { TriangleMesh } from '../scenes/geometry.ts';
import { roughnessToAlpha } from '../../packages/raytracer/src/contract/conventions.ts';

export { roughnessToAlpha };

export const num = (v: number): string => (Number.isInteger(v) ? v.toFixed(1) : String(v));
export const vec = (v: readonly number[]): string => v.map(num).join(' ');

/** Field of view (degrees) along the SHORTER image axis, as pbrt's `fov` is defined. */
export function shorterAxisFovDeg(camera: CameraDesc, film: FilmDesc): number {
  if (film.width >= film.height) return camera.yFovDeg;
  const ty = Math.tan((camera.yFovDeg * Math.PI) / 360);
  return (2 * Math.atan(ty * (film.width / film.height)) * 180) / Math.PI;
}

/** Horizontal FOV in degrees (Mitsuba `fov_axis` x). */
export function xFovDeg(camera: CameraDesc, film: FilmDesc): number {
  const ty = Math.tan((camera.yFovDeg * Math.PI) / 360);
  return (2 * Math.atan(ty * (film.width / film.height)) * 180) / Math.PI;
}

export interface WorldMesh {
  desc: MeshDesc;
  /** Safe file stem: `<index>-<name>`. */
  stem: string;
  mesh: TriangleMesh;
}

/**
 * Tessellate every mesh in world space, the single source of triangles for all exporters. `file` geometry is loaded
 * from `assetsDir` (default bench/assets) by the Node glTF loader; analytic shapes by scenes/geometry.ts.
 */
export function worldMeshes(scene: SceneDesc, assetsDir: string = ASSETS_DIR): WorldMesh[] {
  return scene.meshes.map((desc, i) => ({
    desc,
    stem: `${String(i).padStart(2, '0')}-${desc.name.replace(/[^A-Za-z0-9_-]+/g, '_')}`,
    mesh: desc.geometry.type === 'file' ? transformMesh(loadFileGeometry(desc.geometry, assetsDir).mesh, desc.transform) : meshToWorld(desc),
  }));
}

/** Normal-incidence Fresnel reflectance of a conductor: ((n-1)^2+k^2)/((n+1)^2+k^2). */
export function conductorF0(eta: number, k: number): number {
  return ((eta - 1) ** 2 + k * k) / ((eta + 1) ** 2 + k * k);
}
