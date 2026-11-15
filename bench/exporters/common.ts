import type { CameraDesc, FilmDesc, MeshDesc, SceneDesc } from '../scenes/schema.ts';
import { generateTexture } from '../scenes/textures.ts';
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

export interface UsedTexture {
  name: string;
  /** File written next to the scene (PNG). */
  file: string;
  width: number;
  height: number;
  data: Uint8Array;
  colorSpace: 'srgb' | 'linear';
  wrap: 'repeat' | 'clamp' | 'mirror';
}

/**
 * The reflectance textures a native-renderer export needs (pbrt-v4, Mitsuba): only `diffuse.reflectanceTexture` is expressible there. A textured
 * diffuse must have white `reflectance` (neither renderer multiplies an image texture by a colour without extra nodes); `pbr` materials and
 * the other texture slots are ours only and throw, so a reference is never rendered for something it did not draw.
 */
export function usedTextures(scene: SceneDesc, renderer: string): Map<string, UsedTexture> {
  const out = new Map<string, UsedTexture>();
  for (const [materialName, m] of Object.entries(scene.materials)) {
    if (m.type === 'pbr') throw new Error(`${renderer}: material '${materialName}' is type pbr (glTF metallic-roughness with textures), which only our renderer expresses exactly`);
    if (m.type !== 'diffuse' || !m.reflectanceTexture) continue;
    if (m.reflectance.some((c) => c !== 1)) throw new Error(`${renderer}: textured diffuse '${materialName}' needs reflectance [1, 1, 1] (the image is the whole reflectance)`);
    const desc = scene.textures?.[m.reflectanceTexture];
    if (!desc) throw new Error(`material '${materialName}': unknown texture '${m.reflectanceTexture}'`);
    if (desc.colorSpace !== 'srgb') throw new Error(`texture '${m.reflectanceTexture}' is linear data; a reflectance texture is sRGB`);
    if (!out.has(m.reflectanceTexture)) {
      const t = generateTexture(desc);
      out.set(m.reflectanceTexture, { name: m.reflectanceTexture, file: `tex-${m.reflectanceTexture.replace(/[^A-Za-z0-9_-]+/g, '_')}.png`, ...t });
    }
  }
  return out;
}
