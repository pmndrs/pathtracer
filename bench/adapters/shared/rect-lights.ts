/**
 * One-sided quad emitters -> three.js `RectAreaLight`s (area lights sampled by next-event estimation, emitting from the
 * front face only: exactly the schema's default emitter). Used by three-gpu-pathtracer, which only finds emissive
 * *meshes* by BSDF sampling. The emissive mesh is dropped; the black double-sided twin that `three-scene.ts` put just behind it stays as
 * the opaque black back (an analytic rect light is not geometry; the twin sits BACK_OFFSET behind the light so
 * shadow rays toward the light never reach it).
 *
 * Quad = a mesh with exactly 4 distinct vertices (two triangles) in world space; radiance = emission (W sr^-1 m^-2).
 */
import { RectAreaLight, type Mesh, type Scene } from 'three';
import { analyseQuad } from './quad.ts';
import type { SceneDesc } from '../../scenes/schema.ts';

export interface RectLightConversion {
  converted: string[];
  /** Emissive meshes that were not 4-vertex quads and stay as meshes. */
  kept: string[];
}

export function convertQuadEmittersToRectLights(scene: Scene, desc: SceneDesc): RectLightConversion {
  const emitters = new Map(desc.meshes.filter((m) => m.emission && !m.twoSided).map((m) => [m.name, m.emission!] as const));
  const result: RectLightConversion = { converted: [], kept: [] };
  const targets: Array<{ mesh: Mesh; emission: readonly [number, number, number] }> = [];
  scene.traverse((object) => {
    const mesh = object as Mesh;
    const emission = mesh.isMesh && !object.name.endsWith('-black-back') ? emitters.get(object.name || (object.parent?.name ?? '')) : undefined;
    if (emission) targets.push({ mesh, emission });
  });

  for (const { mesh, emission } of targets) {
    const quad = analyseQuad(mesh.geometry, mesh.matrixWorld);
    if (!quad) {
      result.kept.push(mesh.name);
      continue;
    }
    const peak = Math.max(...emission, 1e-6);
    const light = new RectAreaLight(0xffffff, peak, quad.width, quad.height);
    light.color.setRGB(emission[0] / peak, emission[1] / peak, emission[2] / peak, 'srgb-linear');
    light.position.copy(quad.center);
    light.quaternion.copy(quad.rotation);
    light.name = `${mesh.name}-rect-light`;
    // three-gpu-pathtracer hides area lights from camera rays unless asked (the schema's emitters are visible).
    (light as RectAreaLight & { visibleToCameraRays?: boolean }).visibleToCameraRays = true;
    mesh.removeFromParent();
    scene.add(light);
    result.converted.push(mesh.name);
  }
  scene.updateMatrixWorld(true);
  return result;
}

