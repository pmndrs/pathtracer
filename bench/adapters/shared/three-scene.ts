/**
 * SceneDesc -> three.js scene for the three-based renderers (three-gpu-pathtracer, mutracer's THREESync).
 * Geometry and materials come from the glTF the shared exporter writes; the camera is built directly from the
 * scene's lookAt (exact transform, exact vertical fov); the environment is set as a float equirect `DataTexture`.
 */
import {
  DataTexture,
  EquirectangularReflectionMapping,
  Euler,
  FloatType,
  DoubleSide,
  FrontSide,
  LinearFilter,
  PerspectiveCamera,
  RGBAFormat,
  Scene,
  type Material,
  type Mesh,
  type MeshStandardMaterial,
  type Texture,
} from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { lookAtCameraToWorld } from '../../scenes/geometry.ts';
import type { SceneDesc } from '../../scenes/schema.ts';
import { analyseQuad } from './quad.ts';
import { equirectForThree, type LoadedEnvironment } from './env.ts';
import { fetchSceneGlb } from './glb.ts';

/** Metres. Tier 0's Cornell light sits 0.1 mm below the ceiling, so this must stay well under that. */
export const BACK_OFFSET = 2e-5;

export interface ThreeScene {
  scene: Scene;
  camera: PerspectiveCamera;
  environmentTexture: Texture | null;
  approximations: string[];
}

function floatEquirect(data: Float32Array, width: number, height: number): DataTexture {
  const texture = new DataTexture(data, width, height, RGBAFormat, FloatType);
  texture.mapping = EquirectangularReflectionMapping;
  texture.magFilter = LinearFilter;
  texture.minFilter = LinearFilter;
  texture.generateMipmaps = false;
  texture.flipY = false;
  texture.needsUpdate = true;
  return texture;
}

export async function buildThreeScene(desc: SceneDesc, env: LoadedEnvironment | null): Promise<ThreeScene> {
  const { glb, approximations } = await fetchSceneGlb(desc);
  const gltf = await new GLTFLoader().parseAsync(glb, '');
  const scene = new Scene();
  // The glTF carries a camera node and (for directional lights) light nodes; the camera is replaced below.
  for (const child of [...gltf.scene.children]) if ((child as { isCamera?: boolean }).isCamera) gltf.scene.remove(child);
  scene.add(gltf.scene);

  // One-sided emitters (the schema default, as in pbrt/Mitsuba): glTF cannot say "emit from the front only". The emissive
  // mesh is restricted to its front face (back-face culling) and a black, non-emitting, double-sided twin of it is added
  // BACK_OFFSET metres behind it, so the back is an opaque black occluder like in the references. (Coplanar twins with
  // back-face-only culling were tried: shadow rays ignore culling in both libraries and the twin blocked the light.)
  const oneSidedEmitters = new Set(desc.meshes.filter((m) => m.emission && !m.twoSided).map((m) => m.name));
  const emitterMeshes: Mesh[] = [];
  gltf.scene.traverse((object) => {
    const mesh = object as Mesh;
    if (mesh.isMesh && oneSidedEmitters.has(object.name || (object.parent?.name ?? ''))) emitterMeshes.push(mesh);
  });
  for (const mesh of emitterMeshes) {
    const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
    const quad = analyseQuad(mesh.geometry, mesh.matrixWorld);
    for (const material of materials) (material as Material).side = FrontSide;
    if (!quad) continue; // not a quad: front-face emission only, the back stays see-through
    const blackBack = mesh.clone();
    blackBack.name = `${mesh.name}-black-back`;
    blackBack.position.addScaledVector(quad.normal, -BACK_OFFSET);
    blackBack.material = materials.map((material) => {
      const black = material.clone() as MeshStandardMaterial;
      black.emissive.setRGB(0, 0, 0);
      black.emissiveIntensity = 0;
      black.side = DoubleSide;
      return black;
    })[0]!;
    mesh.parent?.add(blackBack);
  }

  const cam = desc.camera;
  const camera = new PerspectiveCamera(cam.yFovDeg, desc.film.width / desc.film.height, 0.01, 1000);
  camera.matrixAutoUpdate = false;
  camera.matrix.fromArray(lookAtCameraToWorld(cam.position, cam.target, cam.up));
  camera.matrix.decompose(camera.position, camera.quaternion, camera.scale);
  camera.matrixWorld.copy(camera.matrix);
  camera.matrixWorldInverse.copy(camera.matrixWorld).invert();
  camera.updateProjectionMatrix();

  let environmentTexture: Texture | null = null;
  if (env?.type === 'uniform') {
    const [r, g, b] = env.radiance;
    // 4x2 constant map: enough texels for the importance-sampling tables, every one equal.
    const data = new Float32Array(4 * 2 * 4);
    for (let i = 0; i < 8; i++) data.set([r, g, b, 1], i * 4);
    environmentTexture = floatEquirect(data, 4, 2);
  } else if (env?.type === 'hdri') {
    environmentTexture = floatEquirect(equirectForThree(env), env.width, env.height);
    // Environment appears rotated by +rotationYDeg about +Y (schema), which is an Euler y rotation of the environment.
    const rotation = new Euler(0, (env.rotationYDeg * Math.PI) / 180, 0);
    scene.environmentRotation.copy(rotation);
    scene.backgroundRotation.copy(rotation);
  }
  if (environmentTexture) {
    scene.environment = environmentTexture;
    scene.background = environmentTexture;
    scene.environmentIntensity = 1;
    scene.backgroundIntensity = 1;
  }
  scene.updateMatrixWorld(true);
  return { scene, camera, environmentTexture, approximations };
}
