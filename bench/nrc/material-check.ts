import { encodeNrc, layerOffset, recordLoss, type MlpNetwork, type Precision } from '@pathtracer/mlp';
import type { PtLookAtCamera, PtMegakernel } from '@pmndrs/pathtracer';
import { PtMegakernelAdapter } from '../adapters/pt-megakernel/index.ts';
import type { AdapterSummary } from '../runner/contract.ts';
import { drainDevice, watchDevice } from '../runner/page-client.ts';
import type { SceneDesc } from '../scenes/schema.ts';
import { translation } from '../scenes/geometry.ts';
import { reconstructionError } from './factor-check.ts';
import { NrcCache, type NrcFactorization, type NrcDirectionInput } from './cache.ts';

const SIZE = 16;
const METAL_COLOR = [0.36, 0.49, 0.64] as const;
const PLASTIC_COLOR = [0.2, 0.3, 0.4] as const;
const DIFFUSE_COLOR = [0.3, 0.4, 0.5] as const;
const NORMAL_TEXEL = [179, 204, 230] as const;
const ROUGHNESS_TEXEL = 128;

function pointAtPanel(x: number): PtLookAtCamera {
  return { position: [x, 3, 0.5], target: [x, 0, 0], up: [0, 1, 0], yFovDeg: 35 };
}

const scene: SceneDesc = {
  version: 1,
  name: 'nrc-material-inputs',
  purpose: 'Verify cache inputs for a normal-mapped conductor with a roughness texture, coated plastic, and Lambertian diffuse.',
  tier: 0,
  camera: { type: 'perspective', ...pointAtPanel(0) },
  film: { width: SIZE, height: SIZE, filter: 'box' },
  integrator: { maxDepth: 1 },
  textures: {
    normal: { generator: 'solid', width: 8, height: 8, colorSpace: 'linear', color: NORMAL_TEXEL },
    roughness: { generator: 'solid', width: 8, height: 8, colorSpace: 'linear', color: [0, ROUGHNESS_TEXEL, 255] },
  },
  materials: {
    metal: { type: 'openpbr', baseColor: METAL_COLOR, baseMetalness: 1, specularRoughness: 0.8, normalTexture: 'normal', metallicRoughnessTexture: 'roughness' },
    plastic: { type: 'openpbr', baseColor: PLASTIC_COLOR, specularRoughness: 0.8, coatWeight: 1, coatRoughness: 0.35, coatIor: 1.6 },
    diffuse: { type: 'diffuse', reflectance: DIFFUSE_COLOR },
  },
  meshes: ['metal', 'plastic', 'diffuse'].map((material, index) => ({
    name: material,
    material,
    geometry: { type: 'quad', size: [5, 5] },
    uv: { scale: [1, 1] },
    transform: translation(index * 8, 0, 0),
  })),
};

export async function checkMaterialInputs(
  precision: Precision,
  infer: 'perRecord' | 'tiled',
  setDeviceDiagnostics: (summary: AdapterSummary, errors: string[]) => void,
  factorization: NrcFactorization = 'off', network: MlpNetwork = 'default', checkDirection?: NrcDirectionInput,
): Promise<Record<string, number>> {
  const adapter = new PtMegakernelAdapter({ nee: false, rrStartDepth: 0, frame: true, scenePacking: 'packed' });
  await adapter.load(scene);
  const { device, summary } = adapter.gpuContext!;
  const errors = watchDevice(device);
  setDeviceDiagnostics(summary, errors);
  const { caps, trace, sceneBuffers } = adapter.debug;
  const cache = await NrcCache.create(device, caps, adapter.renderer as PtMegakernel, [trace, sceneBuffers], { width: SIZE, height: SIZE, camera: pointAtPanel(0), maxDepth: 1 }, { min: [-3, -1, -3], max: [19, 1, 3] }, {
    materialInputs: 'evaluated', directionInput: checkDirection ?? 'view', factorization, network, precision, infer, queryDepth: 0, ema: 'off', trainDepth: 1,
    updateBlock: 1, batch: SIZE * SIZE, stepsPerFrame: 1, learningRate: 0,
  });
  try {
    // Route features through independent neurons so display and target loss both expose the stored record.
    const lanes = [[44, 45, 46, 47], [48, 49, 50, 51], [52, 53, 54, 55], [56, 57, 58], [59, 60, 61], [62]];
    if (checkDirection) lanes.push([36, 37, 38, 39], [40, 41, 42, 43]);
    const shape = cache.mlp!.shape;
    const weights = new Float32Array(cache.mlp!.weightCount);
    for (let row = 0; row < lanes.length; row++) {
      const features = lanes[row]!;
      for (let i = 0; i < features.length; i++) weights[row * shape.inputs + features[i]!] = (i + 1) / 4;
      for (let layer = 1; layer < shape.hidden; layer++) weights[layerOffset(shape, layer) + row * shape.width + row] = 1;
      weights[layerOffset(shape, shape.hidden) + (row % 3) * shape.width + row] = 1;
    }
    cache.mlp!.uploadWeights(weights);
    // Tangent +X, bitangent -Z, geometric normal +Y, matching the UV axes of the quad.
    const [tx, ty, tz] = NORMAL_TEXEL.map((v) => 2 * v / 255 - 1) as [number, number, number];
    const normal: [number, number] = [Math.acos(tz / Math.hypot(tx, ty, tz)) / Math.PI, Math.atan2(-ty, tx) / (2 * Math.PI) + 0.5];
    const coatF0 = ((1.6 - 1) / (1.6 + 1)) ** 2;
    const flatNormal: [number, number] = [0, 0.5];
    const panels = [
      { name: 'metal', lobe: 1, normal, roughness: 0.8 * ROUGHNESS_TEXEL / 255, diffuse: [0, 0, 0], specular: METAL_COLOR },
      { name: 'plastic', lobe: 5, normal: flatNormal, roughness: 0.35, diffuse: PLASTIC_COLOR, specular: [coatF0, coatF0, coatF0] },
      { name: 'diffuse', lobe: 0, normal: flatNormal, roughness: 1, diffuse: DIFFUSE_COLOR, specular: [0, 0, 0] },
    ] as const;
    const metrics: Record<string, number> = {};
    for (const [panelIndex, panel] of panels.entries()) {
      // A narrow camera cone keeps every ray within 1e-6 radians of its axis for the angular oracle.
      const view = pointAtPanel(panelIndex * 8);
      if (checkDirection) view.yFovDeg = 0.00005;
      const n = panel.name === 'metal' ? [tx, tz, -ty].map(v => v / Math.hypot(tx, ty, tz)) : [0, 1, 0];
      const wo = [0, 3, 0.5].map(v => v / Math.hypot(3, 0.5));
      const dot = wo.reduce((sum, v, axis) => sum + v * n[axis]!, 0);
      const d = checkDirection === 'reflected' ? n.map((v, axis) => 2 * dot * v - wo[axis]!) : wo;
      const angles: [number, number] = [Math.acos(d[1]!) / Math.PI, Math.atan2(d[2]!, d[0]!) / (2 * Math.PI) + 0.5];
      const features = encodeNrc({ position: [0, 0, 0], direction: angles, normal: panel.normal, roughness: panel.roughness, diffuse: [...panel.diffuse], specular: [...panel.specular] });
      features[62] = 1 + 0.2 * panel.lobe;
      const expected = [0, 0, 0];
      for (const [row, indices] of lanes.entries()) {
        for (const [i, lane] of indices.entries()) expected[row % 3]! += features[lane]! * (i + 1) / 4;
      }
      const encoder = device.createCommandEncoder();
      cache.setCamera(view);
      cache.reset(encoder);
      cache.encodeFrame(encoder);
      device.queue.submit([encoder.finish()]);
      const [frame, queries, { batch, loss }] = await Promise.all([cache.render.readFrame(), cache.readQueryCount(), cache.readLoss()]);
      const planes = reconstructionError(frame);
      metrics[panel.name + '_planes_max_error'] = planes;
      const { radiance: data } = frame;
      const scale = panel.name === 'diffuse' && factorization !== 'off' ? DIFFUSE_COLOR
        : panel.name === 'metal' && factorization === 'reflectance' ? METAL_COLOR.map(f => Math.max(f, 1 / 16)) : [1, 1, 1];
      let worst = 0;
      for (let i = 0; i < data.length; i++) {
        if (i % 4 !== 3) worst = Math.max(worst, Math.abs(data[i]! - expected[i % 4]! * scale[i % 4]!));
      }
      metrics[panel.name + '_max_error'] = worst;
      if (!(planes < 2e-5) || queries !== SIZE * SIZE || !(worst < (precision === 'f32' ? 2e-5 : 0.002))) {
        throw new Error(`nrc evaluated materials: ${panel.name} had ${queries} queries, max error ${worst}, first pixel ${Array.from(data.subarray(0, 3))}, expected ${expected}`);
      }
      // The scene has zero suffix radiance. Comparing the first training step's loss checks
      // that training reads the same evaluated features that display inference used above.
      const lossError = Math.abs(loss - recordLoss(expected, [0, 0, 0]));
      metrics[panel.name + '_loss_error'] = lossError;
      if (batch !== SIZE * SIZE || !(lossError < (precision === 'f32' ? 2e-4 : 0.02))) {
        throw new Error(`nrc evaluated materials: ${panel.name} had ${batch} training samples, loss error ${lossError}`);
      }
    }
    await drainDevice(device);
    if (errors.length) throw new Error(errors.join('; '));
    return metrics;
  } finally {
    cache.destroy();
    adapter.dispose();
  }
}
