/** Convert the local Blender exports to the bench's world-space meshes and decoded material textures. */
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { exportScene, WEB_DIR } from '../adapters/export-scenes.ts';
import { ASSETS_DIR } from '../scenes/assets-dir.ts';
import { gltfSceneDescFromFile } from '../scenes/gltf-scene-desc.ts';
import { sponzaWhite } from '../scenes/tier2/sponza-white.ts';
import type { CameraDesc, LightDesc, Rgb } from '../scenes/schema.ts';

for (const kind of process.argv.length > 2 ? process.argv.slice(2) : ['classroom', 'sponza']) {
  if (kind !== 'classroom' && kind !== 'sponza') throw new Error('Expected classroom or sponza');
  const path = 'generated/hero-source/' + kind + '.glb';
  const sha256 = createHash('sha256').update(readFileSync(join(ASSETS_DIR, path))).digest('hex');
  const classroom = kind === 'classroom';
  const shot: { camera: CameraDesc; lights?: LightDesc[]; emitters?: { name: string; positions: number[]; emission: Rgb }[] } = classroom
    ? JSON.parse(readFileSync(join(ASSETS_DIR, 'generated/hero-source/classroom-camera.json'), 'utf8'))
    : sponzaWhite;
  const { scene, loaded } = gltfSceneDescFromFile({ path, format: 'glb', sha256 }, {
    name: 'hero-' + kind, purpose: 'Color interior for the live NRC showcase.', tier: 2,
    camera: shot.camera, lights: classroom ? [{
      // Afternoon sunlight toward the blackboard; window frames and slats cast actual visibility shadows.
      type: 'directional', direction: [-0.694, -0.375, -0.615], irradiance: [12, 10.2, 7.8], angularRadius: 0.00465,
    }] : shot.lights ?? [], normals: 'shading',
    film: { width: 640, height: 360, filter: 'box' }, integrator: { maxDepth: 6, rrStartDepth: 4 },
    environment: classroom ? { type: 'uniform', radiance: [0.11, 0.23, 0.55] } : sponzaWhite.environment!,
  });
  // A Cycles sampling portal is an opening, not a surface. An emissive proxy here
  // hides the sky and occludes the directional light sampled by shadow rays.
  if (classroom) scene.meshes = scene.meshes.filter(mesh => !mesh.material.endsWith('hero-dayLight_portal'));
  if (shot.emitters?.length) {
    scene.materials.lamp = { type: 'black' };
    for (const emitter of shot.emitters.filter(light => light.name !== 'exterior_fillLight')) scene.meshes.push({
      name: emitter.name, material: 'lamp', emission: emitter.emission,
      geometry: { type: 'triangles', positions: emitter.positions, indices: [0, 1, 2, 0, 2, 3] },
    });
  }
  exportScene(scene);
  writeFileSync(join(WEB_DIR, scene.name + '.scene.json'), JSON.stringify(scene));
  console.log(scene.name, scene.meshes.length, 'material groups,', Object.keys(scene.textures ?? {}).length, 'texture slots', loaded.notes);
}
