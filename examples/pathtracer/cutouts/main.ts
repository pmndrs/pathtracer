/**
 * Alpha cutouts: leaf cards (quads with a texture whose alpha holes are the gaps between leaves) over a ground plane, lit by a low sun.
 *
 * A material with `alphaMode: ALPHA_MODE.mask` is cut out where its base colour alpha times `opacity` is below `alphaCutoff`.
 * `ptSceneOptions` sees such a material and compiles the cutout test (`integrator.alphaMask`). Camera and bounce rays step through holes;
 * shadow rays are the expensive part, so there are settings for them:
 *   exact          every shadow ray tests the texture at every card it crosses (reference)
 *   opaqueShadow   shadow rays treat cards as solid: card-shaped shadows, fastest, BIASED (too dark)
 *   roulette       solid first; a blocked ray is re-tested exactly with probability q and counts 1/q: UNBIASED, noisier
 *   transmit       a ray blocked only by cards passes a constant fraction: soft uniform shadows, BIASED
 */
import { ALPHA_MODE, buildGpuScene, type GpuScene, type SceneCopyInput, type TextureInput } from '@pmndrs/raytracer';
import type { PtIntegratorOptions } from '@pmndrs/pathtracer';
import { defineExample } from '../../shared/example.ts';
import { OrbitCamera } from '../../shared/orbit.ts';
import { ProgressiveRenderer } from '../../shared/progressive.ts';
import { sunFrom } from '../../shared/models.ts';

/** Instance mask of the cutout cards (the `transmit` shadow mode tells cards from the rest by it). */
const CARD_MASK = 0x04;

/** 256^2 RGBA: green leaves (discs) on a transparent background, sRGB colour, alpha 0 or 255. */
function leafTexture(): TextureInput {
  const size = 256;
  const data = new Uint8Array(size * size * 4);
  let seed = 7;
  const rand = () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 4294967296);
  const leaves = Array.from({ length: 42 }, () => ({ x: rand() * size, y: rand() * size, r: 10 + rand() * 18, shade: 0.55 + rand() * 0.45 }));
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = (y * size + x) * 4;
      // Wrap around so the texture tiles.
      const leaf = leaves.find((l) => Math.hypot(((x - l.x + size * 1.5) % size) - size / 2, ((y - l.y + size * 1.5) % size) - size / 2) < l.r);
      if (leaf) data.set([Math.round(60 * leaf.shade), Math.round(150 * leaf.shade), Math.round(40 * leaf.shade), 255], i);
    }
  }
  return { width: size, height: size, data, colorSpace: 'srgb', wrap: 'repeat', label: 'leaves' };
}

/** A quad from corner `o` along edges `u` and `v`, with uvs repeating `tiles` times. */
function quad(o: number[], u: number[], v: number[], tiles = 1) {
  const p = [o, [o[0]! + u[0]!, o[1]! + u[1]!, o[2]! + u[2]!], [o[0]! + u[0]! + v[0]!, o[1]! + u[1]! + v[1]!, o[2]! + u[2]! + v[2]!], [o[0]! + v[0]!, o[1]! + v[1]!, o[2]! + v[2]!]];
  return { positions: new Float32Array(p.flat()), indices: [0, 1, 2, 0, 2, 3], uvs: [0, tiles, tiles, tiles, tiles, 0, 0, 0] };
}

function foliageScene(sunElevation: number): SceneCopyInput {
  const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  const meshes = [quad([-6, 0, 6], [12, 0, 0], [0, 0, -12])];
  // Upright cards in a row, and a canopy of flat cards above the ground: both cast dappled shadows.
  for (let i = 0; i < 5; i++) meshes.push(quad([-3 + i * 1.4, 0, -1 + (i % 2) * 0.6], [1.1, 0, -0.3], [0, 2.2, 0]));
  for (let i = 0; i < 3; i++) meshes.push(quad([-2.5 + i * 1.8, 2.6, 1.5], [1.6, 0, 0], [0, 0.15, -1.6], 2));
  return {
    meshes,
    instances: meshes.map((_, i) => ({ meshIndex: i, instanceIndex: i, objectToWorld: identity, material: i === 0 ? 0 : 1, mask: i === 0 ? 0x01 : CARD_MASK })),
    materials: [
      { baseColor: [0.75, 0.72, 0.65], specularWeight: 0 },
      { baseColor: [1, 1, 1], baseColorTexture: 0, alphaMode: ALPHA_MODE.mask, alphaCutoff: 0.5, doubleSided: true, specularRoughness: 0.6 },
    ],
    textures: [leafTexture()],
    sun: sunFrom(sunElevation, 20, 4),
    environment: { radiance: [0.35, 0.45, 0.6] },
  };
}

const MODES: Record<string, PtIntegratorOptions> = {
  'Exact (reference)': { alphaPolicy: 'exact' },
  'Opaque shadows (fast, biased)': { alphaPolicy: 'opaqueShadow' },
  'Roulette (unbiased, noisier)': { alphaShadow: 'roulette', alphaRoulette: 0.25 },
  'Transmit (fast, biased)': { alphaShadow: 'transmit', alphaTransmit: 0.5, alphaMixedMask: CARD_MASK },
};

defineExample(async (ctx) => {
  const { device, pane, canvas } = ctx;
  const params = { mode: 'Exact (reference)', sunElevation: 35 };
  const camera = { position: [5.5, 3.2, 6.5] as [number, number, number], target: [0, 0.9, 0] as [number, number, number], up: [0, 1, 0] as [number, number, number], yFovDeg: 45 };
  let scene: GpuScene = await buildGpuScene(device, foliageScene(params.sunElevation), { label: 'foliage' });
  let renderer: ProgressiveRenderer | null = null;
  const orbit = new OrbitCamera(canvas, camera);

  const rebuild = async () => {
    ctx.progress('Compiling…');
    const next = await ProgressiveRenderer.create(ctx, scene, orbit.camera, { integrator: MODES[params.mode] });
    if (renderer) Object.assign(next.settings, renderer.settings);
    const old = renderer;
    renderer = next;
    ctx.progress(null);
    await device.queue.onSubmittedWorkDone();
    old?.destroy();
  };
  orbit.onChange = () => renderer?.setCamera(orbit.camera);

  const shadows = pane.addFolder({ title: 'Cutout shadows' });
  shadows.addBinding(params, 'mode', { label: 'shadow rays', options: Object.fromEntries(Object.keys(MODES).map((k) => [k, k])) }).on('change', () => void rebuild());
  const light = pane.addFolder({ title: 'Light' });
  light.addBinding(params, 'sunElevation', { label: 'sun elevation', min: 5, max: 90, step: 1 }).on('change', async (e) => {
    if (!e.last || !renderer) return;
    const old = scene;
    scene = await buildGpuScene(device, foliageScene(params.sunElevation), { label: 'foliage' });
    renderer.setScene(scene);
    await device.queue.onSubmittedWorkDone();
    old.destroy();
  });

  await rebuild();
  ctx.loop((encoder) => renderer?.frame(encoder) ?? false);
});
