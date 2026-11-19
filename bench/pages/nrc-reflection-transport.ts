/** Phase 12 gate: a full direct-light estimate plus a strictly indirect suffix through the Reflections provider. */
import { PtMegakernel, ptCameraBasis } from '@pmndrs/pathtracer';
import { detectCaps, requestDeviceForCaps, sceneFormatOf } from '@pmndrs/raytracer';
import { compareImages } from '../metrics/metrics.ts';
import { makeReflectionProvider, reflectionProviderModules } from '../nrc/reflection-provider.ts';
import { REFLECTION_TRANSPORT_WGSL } from '../nrc/reflection-transport.ts';
import type { AdapterSummary } from '../runner/contract.ts';
import { drainDevice, getArgs, numberArg, reportError, reportResult, summarizeAdapter, uploadImage, watchDevice } from '../runner/page-client.ts';
import { transportScenes } from '../scenes/tier2/nrc-reflection-transport.ts';
import { sceneDescToCopyInput } from '../scenes/to-scene-copy.ts';
import type { Image } from '../src/adapter.ts';

const SHADER = /* wgsl */ `
struct RcParams { origin: vec4f, right: vec4f, up: vec4f, forward: vec4f, dims: vec4u }
@group(0) @binding(0) var<uniform> params: RcParams;
@group(0) @binding(1) var<storage, read_write> sums: array<vec4f>;

@compute @workgroup_size(8, 8)
fn transport(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= params.dims.xy)) { return; }
  let index = id.y * params.dims.x + id.x;
  let seed = rcHash(index * 2654435761u + params.dims.z * 0x68bc21ebu);
  let jitter = vec2f(rcRandom(seed, 0u), rcRandom(seed, 1u));
  let ndc = 2.0 * (vec2f(id.xy) + jitter) / vec2f(params.dims.xy) - 1.0;
  let dir = normalize(params.forward.xyz + params.right.xyz * ndc.x - params.up.xyz * ndc.y);
  let ray = RtRay(params.origin.xyz, 0.0, dir, 1e30);
  let hit = traceFirstHit(ray.origin, ray.direction, ray.tMax);
  if (hit.t < 0.0) { return; }
  let p = rtShadePoint(rtpRtHit(hit), ray);
  let direct = rcDirect(p, seed, 2u);
  let suffix = rcSuffix(p, seed, params.dims.w, true);
  sums[index * 3u] += vec4f(rtShadeEmission(p) + direct.direct, 1.0);
  sums[index * 3u + 1u] += vec4f(suffix.indirect, 0.0);
  sums[index * 3u + 2u] += vec4f(suffix.directComplement, 0.0);
}
`;

let adapterSummary: AdapterSummary | null = null;
let deviceErrors: string[] = [];

async function main() {
  const args = getArgs();
  if (!args.reflections) throw new Error('Pass reflections=<checkout> and set REFLECTIONS_REPO to the same checkout in the runner environment');
  const spp = numberArg(args, 'spp', 4096);
  const refSpp = numberArg(args, 'refSpp', 4096);
  const size = numberArg(args, 'size', 48);
  const upload = args.upload === '1';
  if (![size, spp, refSpp].every(n => Number.isSafeInteger(n) && n > 0)) throw new Error('size, spp and refSpp must be positive integers');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('No usable GPU');
  adapterSummary = summarizeAdapter(adapter);
  const { device, caps } = await requestDeviceForCaps(adapter, detectCaps(adapter), { label: 'nrc-reflection-transport' });
  deviceErrors = watchDevice(device);
  const refl = await reflectionProviderModules(args.reflections);
  const metrics: Record<string, number> = {};

  for (const scene of transportScenes) {
    const name = scene.name;
    const provider = await makeReflectionProvider(refl, device, sceneDescToCopyInput(scene));
    const code = [refl.HIT_WGSL, provider.code({ group: 1, firstBinding: 0 }), REFLECTION_TRANSPORT_WGSL, SHADER].join('\n');
    const shader = device.createShaderModule({ label: name + ' strict indirect', code });
    const info = await shader.getCompilationInfo();
    if (info.messages.some(m => m.type === 'error')) {
      throw new Error(info.messages.map(m => `${m.lineNum}:${m.linePos} ${m.message}`).join('\n'));
    }
    const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: shader, entryPoint: 'transport' } });
    const basis = ptCameraBasis(scene.camera, size, size);
    const cameraData = new ArrayBuffer(80);
    new Float32Array(cameraData, 0, 16).set([...basis.origin, 0, ...basis.right, 0, ...basis.up, 0, ...basis.forward, 0]);
    const dims = new Uint32Array(cameraData, 64, 4);
    dims.set([size, size, 0, scene.integrator.maxDepth]);
    const camera = device.createBuffer({ size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const sizeBytes = size * size * 48;
    const sums = device.createBuffer({ size: sizeBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const groups = [
      device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: camera } }, { binding: 1, resource: { buffer: sums } }],
      }),
      device.createBindGroup({
        layout: pipeline.getBindGroupLayout(1),
        entries: provider.resources().map((resource, binding) => ({ resource, binding })),
      }),
    ];
    for (let n = 0; n < spp; n++) {
      dims[2] = n;
      device.queue.writeBuffer(camera, 0, cameraData);
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      groups.forEach((g, i) => pass.setBindGroup(i, g));
      pass.dispatchWorkgroups(Math.ceil(size / 8), Math.ceil(size / 8));
      pass.end();
      device.queue.submit([encoder.finish()]);
      if ((n + 1) % 64 === 0) await device.queue.onSubmittedWorkDone();
    }
    const staging = device.createBuffer({ size: sizeBytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const readEncoder = device.createCommandEncoder();
    readEncoder.copyBufferToBuffer(sums, 0, staging, 0, sizeBytes);
    device.queue.submit([readEncoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const values = new Float32Array(staging.getMappedRange());
    const direct = new Float32Array(size * size * 4);
    const indirect = new Float32Array(direct.length);
    const complement = new Float32Array(direct.length);
    for (let i = 0; i < size * size; i++) {
      if (values[i * 12 + 3] !== spp) throw new Error('Transport receiver does not cover every camera ray');
      for (let c = 0; c < 3; c++) {
        direct[i * 4 + c] = values[i * 12 + c]! / spp;
        indirect[i * 4 + c] = values[i * 12 + 4 + c]! / spp;
        complement[i * 4 + c] = values[i * 12 + 8 + c]! / spp;
      }
      direct[i * 4 + 3] = indirect[i * 4 + 3] = complement[i * 4 + 3] = 1;
    }
    staging.unmap();

    const pt = await PtMegakernel.create(device, caps, {
      format: sceneFormatOf(provider.traceBuffers), layout: provider.sceneCopy.layout, scenePacking: provider.sceneCopyBuffers.packing,
    });
    pt.setScene(provider.traceBuffers, provider.sceneCopyBuffers);
    pt.setFilm(size, size);
    pt.setCamera(scene.camera);
    pt.setLimits(scene.integrator);
    pt.setSeed(5723);
    for (let n = 0; n < refSpp; n += 64) {
      const encoder = device.createCommandEncoder();
      if (n === 0) pt.reset(encoder);
      pt.encode(encoder, Math.min(64, refSpp - n));
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
    }
    const image = (data: Float32Array): Image => ({ width: size, height: size, channels: 4, data });
    const ref = image((await pt.readImage()).data);
    const correct = image(direct.map((x, i) => i % 4 === 3 ? x : x + indirect[i]!));
    const naive = image(correct.data.map((x, i) => i % 4 === 3 ? x : x + complement[i]!));
    for (const [arm, sample] of [['direct', image(direct)], ['strict', correct], ['naive', naive]] as const) {
      const m = compareImages(sample, ref);
      if (m.nonFiniteTest || m.nonFiniteRef) throw new Error('Transport produced non-finite RGB');
      metrics[`${name}_${arm}_relmse`] = m.relMse;
      metrics[`${name}_${arm}_relbias`] = m.relBias;
      if (upload) await uploadImage(`${name}-${arm}`, sample);
    }
    for (const [part, data] of [['direct', direct], ['indirect', indirect], ['mis_complement', complement]] as const) {
      for (let c = 0; c < 3; c++) {
        let sum = 0;
        for (let i = c; i < data.length; i += 4) sum += data[i]!;
        metrics[`${name}_${part}_mean_${'rgb'[c]}`] = sum / (size * size);
      }
    }
    const correctBias = metrics[`${name}_strict_relbias`]!;
    const naiveBias = metrics[`${name}_naive_relbias`]!;
    if (Math.abs(correctBias) > 0.045 || naiveBias - correctBias < 0.005) {
      throw new Error(`${name}: incorrect transport (strict bias ${correctBias}, double-counted bias ${naiveBias})`);
    }
    if (name === 'sky') {
      const analytic = [0.2, 0.15, 0.12];
      for (let c = 0; c < 3; c++) {
        if (Math.abs(metrics[`sky_direct_mean_${'rgb'[c]}`]! - analytic[c]!) > 0.0001
            || metrics[`sky_indirect_mean_${'rgb'[c]}`] !== 0) throw new Error('Isolated Lambert sky does not match its closed form');
      }
    }
    if (name === 'area' && metrics.area_indirect_mean_r !== 0) throw new Error('Isolated area light produced an indirect term');
    if (name === 'bounce' && !(metrics.bounce_indirect_mean_r! > 4 * metrics.bounce_indirect_mean_b!)) {
      throw new Error('Expected a red indirect response from the bounce wall');
    }
    if (upload) await uploadImage(`${name}-reference`, ref);
    for (const b of [camera, sums, staging]) b.destroy();
    pt.destroy();
    provider.destroy();
  }
  await drainDevice(device);
  if (deviceErrors.length) throw new Error(deviceErrors.join(' | '));
  reportResult({ scene: 'nrc-reflection-transport', renderer: 'reflections-provider', config: { size, spp, refSpp }, metrics, valid: true }, { adapter: adapterSummary, deviceErrors });
}

main().catch(error => reportError(error, { adapter: adapterSummary, deviceErrors }));
