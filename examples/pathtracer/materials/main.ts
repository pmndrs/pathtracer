/**
 * Material explorer: every OpenPBR lobe of one material, on a sphere under a studio environment.
 *
 * A material is one record in the scene copy (`RtMaterialInput`, OpenPBR names: base, specular, transmission, coat, fuzz, thin film,
 * emission). The kernel compiles only the lobes its scene uses, so this page compiles it once with `materialProbes` that switch every
 * lobe on; after that a slider only rebuilds the scene copy on the CPU and rewrites it in place (`buildSceneCopy` + `writeSceneCopy`):
 * no BVH rebuild, no recompile. The environment is an equirect HDR map made here: a sky gradient and three soft boxes.
 */
import { buildGpuScene, buildSceneCopy, writeSceneCopy, type GpuScene, type RtMaterialInput, type SceneCopyInput, type SceneCopyMesh } from '@pmndrs/raytracer';
import type { BindingParams } from 'tweakpane';
import { defineExample } from '../../shared/example.ts';
import { OrbitCamera } from '../../shared/orbit.ts';
import { ProgressiveRenderer } from '../../shared/progressive.ts';

type Rgb = { r: number; g: number; b: number };
const rgb = (c: Rgb): [number, number, number] => [c.r, c.g, c.b];
const col = (r: number, g: number, b: number): Rgb => ({ r, g, b });

/** The explorer's parameters, in OpenPBR's groups. Colours are linear. */
interface Params {
  baseColor: Rgb; baseWeight: number; metalness: number;
  specularWeight: number; specularColor: Rgb; specularRoughness: number; ior: number; specularAnisotropy: number; anisotropyRotation: number;
  transmissionWeight: number; transmissionColor: Rgb; transmissionDepth: number; thinWalled: boolean;
  coatWeight: number; coatColor: Rgb; coatRoughness: number; coatIor: number;
  sheenWeight: number; sheenColor: Rgb; sheenRoughness: number;
  thinFilmWeight: number; thinFilmThickness: number; thinFilmIor: number;
  emissionColor: Rgb; emissionLuminance: number;
}

const BASE: Params = {
  baseColor: col(0.8, 0.8, 0.8), baseWeight: 1, metalness: 0,
  specularWeight: 1, specularColor: col(1, 1, 1), specularRoughness: 0.3, ior: 1.5, specularAnisotropy: 0, anisotropyRotation: 0,
  transmissionWeight: 0, transmissionColor: col(1, 1, 1), transmissionDepth: 0, thinWalled: false,
  coatWeight: 0, coatColor: col(1, 1, 1), coatRoughness: 0, coatIor: 1.6,
  sheenWeight: 0, sheenColor: col(1, 1, 1), sheenRoughness: 0.3,
  thinFilmWeight: 0, thinFilmThickness: 0.5, thinFilmIor: 1.4,
  emissionColor: col(1, 1, 1), emissionLuminance: 0,
};

const PRESETS: Record<string, Partial<Params>> = {
  'Plastic (red)': { baseColor: col(0.7, 0.05, 0.04), specularRoughness: 0.25 },
  Gold: { baseColor: col(1, 0.71, 0.29), metalness: 1, specularRoughness: 0.2 },
  'Brushed aluminium': { baseColor: col(0.91, 0.92, 0.92), metalness: 1, specularRoughness: 0.45, specularAnisotropy: 0.8 },
  Glass: { transmissionWeight: 1, specularRoughness: 0, ior: 1.5 },
  'Green glass (absorbing)': { transmissionWeight: 1, transmissionColor: col(0.3, 0.75, 0.4), transmissionDepth: 0.4, specularRoughness: 0 },
  'Frosted glass': { transmissionWeight: 1, specularRoughness: 0.35 },
  'Soap bubble': { transmissionWeight: 1, thinWalled: true, specularRoughness: 0, ior: 1.33, thinFilmWeight: 1, thinFilmThickness: 0.45, thinFilmIor: 1.33 },
  'Car paint': { baseColor: col(0.05, 0.12, 0.45), specularRoughness: 0.45, coatWeight: 1, coatRoughness: 0.02 },
  'Coated copper': { baseColor: col(0.95, 0.64, 0.54), metalness: 1, specularRoughness: 0.4, coatWeight: 1, coatRoughness: 0.05 },
  Velvet: { baseColor: col(0.25, 0.02, 0.08), specularWeight: 0, sheenWeight: 1, sheenColor: col(0.9, 0.4, 0.6), sheenRoughness: 0.4 },
  'Oil slick (thin film)': { baseColor: col(0.02, 0.02, 0.02), specularRoughness: 0.05, thinFilmWeight: 1, thinFilmThickness: 0.6, thinFilmIor: 1.45 },
  Lamp: { baseColor: col(0.9, 0.9, 0.9), emissionColor: col(1, 0.75, 0.45), emissionLuminance: 4 },
};

function material(p: Params): RtMaterialInput {
  return {
    baseColor: rgb(p.baseColor), baseWeight: p.baseWeight, metalness: p.metalness,
    specularWeight: p.specularWeight, specularColor: rgb(p.specularColor), specularRoughness: p.specularRoughness, ior: p.ior,
    specularAnisotropy: p.specularAnisotropy, anisotropyRotation: p.anisotropyRotation,
    transmissionWeight: p.transmissionWeight, transmissionColor: rgb(p.transmissionColor), transmissionDepth: p.transmissionDepth, thinWalled: p.thinWalled,
    coatWeight: p.coatWeight, coatColor: rgb(p.coatColor), coatRoughness: p.coatRoughness, coatIor: p.coatIor,
    sheenWeight: p.sheenWeight, sheenColor: rgb(p.sheenColor), sheenRoughness: p.sheenRoughness,
    thinFilmWeight: p.thinFilmWeight, thinFilmThickness: p.thinFilmThickness, thinFilmIor: p.thinFilmIor,
    emissionColor: rgb(p.emissionColor), emissionLuminance: p.emissionLuminance,
  };
}

/** Records the kernel must handle whatever the sliders do: every lobe the explorer can switch on. */
const PROBES: RtMaterialInput[] = [
  { metalness: 0.5, specularAnisotropy: 0.5 },
  { metalness: 1, coatWeight: 1, coatIor: 1.5 },
  { coatWeight: 1, coatIor: 1.5 },
  { transmissionWeight: 1, thinWalled: true },
  { transmissionWeight: 1, transmissionDepth: 1 },
  { sheenWeight: 1, sheenColor: [1, 1, 1] },
  { thinFilmWeight: 1, thinFilmThickness: 0.5 },
];

/** A uv sphere with smooth normals, uvs and tangents (u around the vertical axis: the anisotropy follows the tangent, dp/du). */
function sphere(cy: number, r: number, seg = 96, rings = 48): SceneCopyMesh {
  const positions: number[] = [], normals: number[] = [], uvs: number[] = [], tangents: number[] = [], indices: number[] = [];
  for (let j = 0; j <= rings; j++) {
    const v = j / rings, th = v * Math.PI;
    for (let i = 0; i <= seg; i++) {
      const u = i / seg, ph = u * 2 * Math.PI;
      const n = [Math.sin(th) * Math.sin(ph), Math.cos(th), Math.sin(th) * Math.cos(ph)];
      positions.push(r * n[0]!, cy + r * n[1]!, r * n[2]!);
      normals.push(...n);
      uvs.push(u, v);
      tangents.push(Math.cos(ph), 0, -Math.sin(ph), 1);
    }
  }
  for (let j = 0; j < rings; j++) {
    for (let i = 0; i < seg; i++) {
      const a = j * (seg + 1) + i, b = a + seg + 1;
      if (j > 0) indices.push(a, b, a + 1);
      if (j < rings - 1) indices.push(a + 1, b, b + 1);
    }
  }
  return { positions: new Float32Array(positions), normals: new Float32Array(normals), uvs: new Float32Array(uvs), tangents: new Float32Array(tangents), indices: new Uint32Array(indices) };
}

/** An equirect studio map (row 0 = up): a dim sky gradient, a darker floor, and three soft boxes. */
function studio(width = 512, height = 256): { width: number; height: number; channels: number; data: Float32Array } {
  const data = new Float32Array(width * height * 3);
  const boxes = [
    { phi: 0.6, theta: 0.32, w: 0.35, h: 0.18, L: [14, 13, 12] },
    { phi: 3.2, theta: 0.4, w: 0.5, h: 0.12, L: [6, 7, 9] },
    { phi: 4.9, theta: 0.2, w: 0.2, h: 0.12, L: [10, 9, 8] },
  ];
  for (let y = 0; y < height; y++) {
    const theta = ((y + 0.5) / height) * Math.PI; // 0 = up
    const up = Math.cos(theta);
    for (let x = 0; x < width; x++) {
      const phi = ((x + 0.5) / width) * 2 * Math.PI;
      let c = up > 0 ? [0.18 + 0.12 * up, 0.2 + 0.14 * up, 0.25 + 0.2 * up] : [0.05, 0.045, 0.04];
      for (const b of boxes) {
        const dphi = Math.abs(((phi - b.phi + 3 * Math.PI) % (2 * Math.PI)) - Math.PI);
        if (dphi < b.w && Math.abs(theta - b.theta) < b.h) c = b.L;
      }
      data.set(c, (y * width + x) * 3);
    }
  }
  return { width, height, channels: 3, data };
}

function sceneInput(p: Params): SceneCopyInput & { materials: RtMaterialInput[] } {
  const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  const meshes: SceneCopyMesh[] = [
    { positions: new Float32Array([-3, 0, 3, 3, 0, 3, 3, 0, -3, -3, 0, -3]), indices: [0, 1, 2, 0, 2, 3] },
    sphere(0.75, 0.7),
  ];
  return {
    meshes,
    instances: meshes.map((_, i) => ({ meshIndex: i, instanceIndex: i, objectToWorld: identity, material: i })),
    materials: [{ baseColor: [0.35, 0.35, 0.36], specularRoughness: 0.6 }, material(p)],
    environment: { map: studio() },
  };
}

defineExample(async (ctx) => {
  const { device, pane, canvas, stats } = ctx;
  const search = new URLSearchParams(location.search);
  const params: Params & { preset: string } = { ...structuredClone(BASE), preset: search.get('preset') ?? 'Gold' };
  Object.assign(params, structuredClone(PRESETS[params.preset] ?? {}));
  const orbit = new OrbitCamera(canvas, { position: [0, 1.3, 3.2], target: [0, 0.7, 0], up: [0, 1, 0], yFovDeg: 35 });
  ctx.progress('Building the scene…');
  const input = sceneInput(params);
  const scene: GpuScene = await buildGpuScene(device, input, { label: 'materials' });
  ctx.progress('Compiling (every lobe, once)…');
  let renderer: ProgressiveRenderer;
  try {
    renderer = await ProgressiveRenderer.create(ctx, scene, orbit.camera, { materialProbes: PROBES });
  } catch (error) {
    ctx.error('The kernel does not compile', error);
    return;
  }
  renderer.settings.maxDepth = 12;
  ctx.progress(null);
  orbit.onChange = () => renderer.setCamera(orbit.camera);

  // A slider: rebuild the scene copy on the CPU (materials, light table, environment) and rewrite the GPU buffer in place.
  let pending = false;
  const update = () => {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => {
      pending = false;
      input.materials[1] = material(params);
      const copy = buildSceneCopy(input);
      if (!writeSceneCopy(device, scene.buffers, copy)) return ctx.error('The scene copy grew past its buffer');
      renderer.restart();
    });
  };

  pane.addBinding(params, 'preset', { options: Object.fromEntries(Object.keys(PRESETS).map((k) => [k, k])) }).on('change', () => {
    Object.assign(params, structuredClone(BASE), structuredClone(PRESETS[params.preset]!));
    pane.refresh();
    update();
  });
  const color = { color: { type: 'float' as const } };
  const groups: [string, [keyof Params, BindingParams][]][] = [
    ['Base', [['baseColor', color], ['baseWeight', { min: 0, max: 1 }], ['metalness', { min: 0, max: 1 }]]],
    ['Specular', [['specularWeight', { min: 0, max: 1 }], ['specularColor', color], ['specularRoughness', { min: 0, max: 1 }], ['ior', { min: 1, max: 3 }], ['specularAnisotropy', { min: 0, max: 1 }], ['anisotropyRotation', { min: 0, max: 0.999 }]]],
    ['Transmission', [['transmissionWeight', { min: 0, max: 1 }], ['transmissionColor', color], ['transmissionDepth', { min: 0, max: 2 }], ['thinWalled', {}]]],
    ['Coat', [['coatWeight', { min: 0, max: 1 }], ['coatColor', color], ['coatRoughness', { min: 0, max: 1 }], ['coatIor', { min: 1, max: 3 }]]],
    ['Fuzz', [['sheenWeight', { min: 0, max: 1 }], ['sheenColor', color], ['sheenRoughness', { min: 0, max: 1 }]]],
    ['Thin film', [['thinFilmWeight', { min: 0, max: 1 }], ['thinFilmThickness', { min: 0, max: 1.5 }], ['thinFilmIor', { min: 1, max: 3 }]]],
    ['Emission', [['emissionColor', color], ['emissionLuminance', { min: 0, max: 20 }]]],
  ];
  for (const [title, fields] of groups) {
    const folder = pane.addFolder({ title, expanded: title === 'Base' || title === 'Specular' });
    for (const [key, opts] of fields) folder.addBinding(params, key, opts).on('change', update);
  }
  stats.set('lobes', 'all compiled in');

  ctx.loop((encoder) => renderer.frame(encoder));
});
