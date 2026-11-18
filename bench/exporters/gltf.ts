/**
 * Scene -> binary glTF 2.0 (.glb) for the web renderers (three-gpu-pathtracer, mutracer, ours via loaders).
 * glTF cannot express conductor eta/k, exact dielectrics, a uniform/HDRI environment, one-sided emission or radiometric
 * light units, so this is an *approximation*; every mapping is listed in `approximations`.
 */
import type { MaterialDesc, SceneDesc } from '../scenes/schema.ts';
import { conductorF0, worldMeshes } from './common.ts';
import { openPbrToGltfMaterial } from './gltf-openpbr.ts';
import type { Exporter } from './types.ts';
import { lookAtCameraToWorld } from '../scenes/geometry.ts';

interface GltfMaterial {
  name: string;
  pbrMetallicRoughness: { baseColorFactor: number[]; metallicFactor: number; roughnessFactor: number };
  emissiveFactor?: number[];
  alphaMode?: 'MASK' | 'BLEND';
  alphaCutoff?: number;
  doubleSided?: boolean;
  extensions?: Record<string, unknown>;
}

function pad4(buf: Buffer, fill: number): Buffer {
  const rem = buf.length % 4;
  return rem === 0 ? buf : Buffer.concat([buf, Buffer.alloc(4 - rem, fill)]);
}

export const toGltf: Exporter = (scene: SceneDesc, opts) => {
  const approx = new Set<string>();
  const used = new Set<string>();
  const materials: GltfMaterial[] = [];
  const materialIndex = new Map<string, number>();

  const makeMaterial = (key: string, m: MaterialDesc, emission: readonly number[] | undefined, twoSided: boolean | undefined): number => {
    const id = `${key}|${emission ? emission.join(',') : ''}|${twoSided ? 1 : 0}`;
    const hit = materialIndex.get(id);
    if (hit !== undefined) return hit;
    const g: GltfMaterial = {
      name: key,
      pbrMetallicRoughness: { baseColorFactor: [0, 0, 0, 1], metallicFactor: 0, roughnessFactor: 1 },
      doubleSided: true,
    };
    const ext: Record<string, unknown> = {};
    switch (m.type) {
      case 'diffuse':
        g.pbrMetallicRoughness.baseColorFactor = [...m.reflectance, 1];
        if (m.reflectanceTexture) approx.add('diffuse: reflectanceTexture is dropped (the web renderers skip textured scenes).');
        ext['KHR_materials_specular'] = { specularFactor: 0 };
        approx.add('diffuse: baseColor = reflectance, metallic 0, roughness 1, KHR_materials_specular specularFactor 0 to remove the default 4% dielectric lobe (a viewer without that extension adds a weak specular lobe).');
        break;
      case 'pbr':
        g.pbrMetallicRoughness.baseColorFactor = [...m.baseColor, 1];
        g.pbrMetallicRoughness.metallicFactor = m.metalness;
        g.pbrMetallicRoughness.roughnessFactor = m.roughness;
        if (m.emissive) g.emissiveFactor = [...m.emissive];
        approx.add('pbr: the glb carries the factors only; texture references are dropped (the web renderers skip textured scenes).');
        break;
      case 'black':
        ext['KHR_materials_specular'] = { specularFactor: 0 };
        break;
      case 'interface':
        throw new Error('gltf: interface (medium boundary) materials are not expressible; the web competitors skip media scenes');
      case 'openpbr': {
        const { material, notes } = openPbrToGltfMaterial(m);
        g.pbrMetallicRoughness = material.pbrMetallicRoughness;
        if (material.emissiveFactor) g.emissiveFactor = material.emissiveFactor;
        if (material.alphaMode) g.alphaMode = material.alphaMode;
        if (material.alphaCutoff !== undefined) g.alphaCutoff = material.alphaCutoff;
        Object.assign(ext, material.extensions ?? {});
        approx.add('openpbr: written as core metallic-roughness + the KHR_materials_* extensions the material uses (inverse of gltfMaterialToOpenPbr, bench/exporters/gltf-openpbr.ts); the consumer decides which extensions it honours.');
        for (const n of notes) approx.add(`openpbr: ${n}`);
        break;
      }
      case 'conductor': {
        g.pbrMetallicRoughness.baseColorFactor = [conductorF0(m.eta[0], m.k[0]), conductorF0(m.eta[1], m.k[1]), conductorF0(m.eta[2], m.k[2]), 1];
        g.pbrMetallicRoughness.metallicFactor = 1;
        g.pbrMetallicRoughness.roughnessFactor = m.roughness;
        approx.add('conductor: eta/k cannot be expressed; baseColor = normal-incidence Fresnel F0 per channel computed from eta/k, metallic 1, perceptual roughness unchanged (glTF alpha = roughness² matches). Grazing-angle Fresnel becomes Schlick-like instead of the exact conductor curve.');
        break;
      }
      case 'dielectric':
        g.pbrMetallicRoughness.baseColorFactor = [1, 1, 1, 1];
        g.pbrMetallicRoughness.roughnessFactor = m.roughness;
        ext['KHR_materials_transmission'] = { transmissionFactor: 1 };
        ext['KHR_materials_ior'] = { ior: m.ior };
        approx.add('dielectric: KHR_materials_transmission 1 + KHR_materials_ior; transmission is implemented as thin/approximate refraction by many viewers (a closed solid with a volume interior is only exact in renderers that trace refraction through the closed mesh, e.g. three-gpu-pathtracer, ours).');
        break;
    }
    if (emission) {
      const strength = Math.max(...emission, 1e-6);
      g.emissiveFactor = emission.map((v) => Math.min(1, v / strength));
      if (strength > 1) {
        ext['KHR_materials_emissive_strength'] = { emissiveStrength: strength };
        used.add('KHR_materials_emissive_strength');
      }
      approx.add('emission: emissiveFactor × KHR_materials_emissive_strength; glTF has no one-sided emitter, so emission faces are decided by the consumer (typically both sides of a doubleSided material). Radiance units are taken as W·sr⁻¹·m⁻² = glTF emissive nits scale.');
      if (!twoSided) approx.add('emission: one-sided emitters (default) become emissive on both faces of a doubleSided material where the consumer does not cull by facing.');
    }
    if (Object.keys(ext).length) {
      g.extensions = ext;
      for (const k of Object.keys(ext)) used.add(k);
    }
    materials.push(g);
    materialIndex.set(id, materials.length - 1);
    return materials.length - 1;
  };

  // Binary buffer layout: per mesh [positions | normals | indices], each 4-byte aligned.
  const chunks: Buffer[] = [];
  const bufferViews: unknown[] = [];
  const accessors: unknown[] = [];
  const gltfMeshes: unknown[] = [];
  const nodes: unknown[] = [];
  let offset = 0;
  const addView = (data: Buffer, target: number): number => {
    const padded = pad4(data, 0);
    bufferViews.push({ buffer: 0, byteOffset: offset, byteLength: data.length, target });
    chunks.push(padded);
    offset += padded.length;
    return bufferViews.length - 1;
  };

  for (const w of worldMeshes(scene, opts?.assetsDir)) {
    const m = scene.materials[w.desc.material];
    if (!m) throw new Error(`mesh '${w.desc.name}': unknown material '${w.desc.material}'`);
    const pos = w.mesh.positions;
    const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < pos.length; i += 3) for (let k = 0; k < 3; k++) {
      min[k] = Math.min(min[k]!, pos[i + k]!);
      max[k] = Math.max(max[k]!, pos[i + k]!);
    }
    const pv = addView(Buffer.from(pos.buffer, pos.byteOffset, pos.byteLength), 34962);
    const nv = addView(Buffer.from(w.mesh.normals.buffer, w.mesh.normals.byteOffset, w.mesh.normals.byteLength), 34962);
    const iv = addView(Buffer.from(w.mesh.indices.buffer, w.mesh.indices.byteOffset, w.mesh.indices.byteLength), 34963);
    const pa = accessors.push({ bufferView: pv, componentType: 5126, count: pos.length / 3, type: 'VEC3', min, max }) - 1;
    const na = accessors.push({ bufferView: nv, componentType: 5126, count: pos.length / 3, type: 'VEC3' }) - 1;
    const ia = accessors.push({ bufferView: iv, componentType: 5125, count: w.mesh.indices.length, type: 'SCALAR' }) - 1;
    const attributes: Record<string, number> = { POSITION: pa, NORMAL: na };
    const uvs = w.mesh.uvs;
    if (uvs) {
      // TEXCOORD_0 as float VEC2, glTF uv convention (origin top-left): our own web adapter reads it back (readGlbMeshes); the texture images travel beside the glb (export-scenes.ts)
      const uv = addView(Buffer.from(uvs.buffer, uvs.byteOffset, uvs.byteLength), 34962);
      attributes['TEXCOORD_0'] = accessors.push({ bufferView: uv, componentType: 5126, count: uvs.length / 2, type: 'VEC2' }) - 1;
      approx.add('uvs: TEXCOORD_0 is written for meshes whose material reads a texture; the glb carries no texture references or images (our web adapter reads the images from the sidecar files).');
    }
    const tangents = w.mesh.tangents;
    if (tangents) {
      // TANGENT (xyzw, the loader's world-space glTF tangent) for anisotropic meshes (P2.26); our web adapter reads it back (readGlbMeshes)
      const tv = addView(Buffer.from(tangents.buffer, tangents.byteOffset, tangents.byteLength), 34962);
      attributes['TANGENT'] = accessors.push({ bufferView: tv, componentType: 5126, count: tangents.length / 4, type: 'VEC4' }) - 1;
    }
    const mat = makeMaterial(w.desc.material, m, w.desc.emission, w.desc.twoSided);
    gltfMeshes.push({ name: w.desc.name, primitives: [{ attributes, indices: ia, material: mat }] });
    nodes.push({ name: w.desc.name, mesh: gltfMeshes.length - 1 });
  }

  const cam = scene.camera;
  const c2w = lookAtCameraToWorld(cam.position, cam.target, cam.up);
  nodes.push({ name: 'camera', camera: 0, matrix: c2w });
  const aspect = scene.film.width / scene.film.height;
  const cameras = [{ type: 'perspective', perspective: { yfov: (cam.yFovDeg * Math.PI) / 180, aspectRatio: aspect, znear: 0.01, zfar: 1000 } }];

  const extensions: Record<string, unknown> = {};
  for (const l of scene.lights ?? []) {
    used.add('KHR_lights_punctual');
    const lights = ((extensions['KHR_lights_punctual'] as { lights: unknown[] } | undefined) ?? { lights: [] });
    lights.lights.push({ type: 'directional', color: l.irradiance.map((v) => v / Math.max(...l.irradiance, 1e-6)), intensity: Math.max(...l.irradiance) });
    extensions['KHR_lights_punctual'] = lights;
    // direction: node -Z axis points along the light direction
    const f = l.direction;
    nodes.push({ name: 'sun', extensions: { KHR_lights_punctual: { light: lights.lights.length - 1 } }, matrix: lookAtCameraToWorld([0, 0, 0], f, Math.abs(f[1]) > 0.99 ? [1, 0, 0] : [0, 1, 0]) });
    approx.add('directional light: irradiance mapped to KHR_lights_punctual intensity (lux) 1:1; angularRadius dropped (delta light).');
  }

  const env = scene.environment;
  const extras: Record<string, unknown> = { sceneName: scene.name, film: scene.film, maxDepth: scene.integrator.maxDepth };
  if (env) {
    extras['environment'] = env;
    approx.add(`environment (${env.type}) is not representable in glTF: stored in asset.extras / scene extras only; the web adapter must set it up (uniform colour or HDRI file) separately.`);
  }
  approx.add('camera: yfov/aspect map exactly; film size, box filter, maxDepth and spp are renderer settings carried only in extras.');

  const json = {
    asset: { version: '2.0', generator: 'pathtracer bench exporter' },
    scene: 0,
    scenes: [{ name: scene.name, nodes: nodes.map((_, i) => i), extras }],
    nodes,
    meshes: gltfMeshes,
    materials,
    cameras,
    accessors,
    bufferViews,
    buffers: [{ byteLength: offset }],
    extensionsUsed: [...used],
    ...(Object.keys(extensions).length ? { extensions } : {}),
  };

  const jsonBuf = pad4(Buffer.from(JSON.stringify(json), 'utf8'), 0x20);
  const bin = Buffer.concat(chunks);
  const total = 12 + 8 + jsonBuf.length + 8 + bin.length;
  const head = Buffer.alloc(12);
  head.writeUInt32LE(0x46546c67, 0);
  head.writeUInt32LE(2, 4);
  head.writeUInt32LE(total, 8);
  const jh = Buffer.alloc(8);
  jh.writeUInt32LE(jsonBuf.length, 0);
  jh.writeUInt32LE(0x4e4f534a, 4);
  const bh = Buffer.alloc(8);
  bh.writeUInt32LE(bin.length, 0);
  bh.writeUInt32LE(0x004e4942, 4);
  const glb = Buffer.concat([head, jh, jsonBuf, bh, bin]);
  return { sceneFile: 'scene.glb', files: [{ path: 'scene.glb', data: glb }], approximations: [...approx] };
};
