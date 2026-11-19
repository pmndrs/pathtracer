/** Phase 12 bench-only integration with the Reflections TraceProvider and hit-shading seam. */
import { PtMegakernel, ptCameraBasis, ptFrComplex } from '@pmndrs/pathtracer';
import { detectCaps, requestDeviceForCaps, sceneFormatOf } from '@pmndrs/raytracer';
import { compareImages, percentile } from '../metrics/metrics.ts';
import { HashGrid } from '../nrc/hash-grid.ts';
import { reflectionConsumerWgsl } from '../nrc/reflection-consumer-wgsl.ts';
import { makeReflectionProvider, reflectionProviderModules } from '../nrc/reflection-provider.ts';
import { REFLECTION_TRANSPORT_WGSL } from '../nrc/reflection-transport.ts';
import { drainDevice, getArgs, numberArg, reportError, reportResult, summarizeAdapter, uploadImage, watchDevice } from '../runner/page-client.ts';
import type { AdapterSummary } from '../runner/contract.ts';
import { movingCardFixture, thinWallFixture, type LocalityView } from '../scenes/tier2/nrc-locality.ts';
import { compose, identity, rotationX, translation } from '../scenes/geometry.ts';
import { sceneDescToCopyInput } from '../scenes/to-scene-copy.ts';
import type { SceneDesc } from '../scenes/schema.ts';
import type { Image } from '../src/adapter.ts';

type ConsumerArm = 'hash' | 'traced' | 'environment' | 'direct';

// Five vec4s from the shader: radiance and mirror sample count, hit metadata,
// ray counts, cache reach, and reflected miss/emitter counts.
const OUTPUT_WORDS = 20;
const bytes = (words: number) => words * 4;
let adapterSummary: AdapterSummary | null = null;
let deviceErrors: string[] = [];

function mirrorScene(scene: SceneDesc): SceneDesc {
  const metal = scene.materials.metal;
  if (metal?.type !== 'conductor') throw new Error('Reflection fixture requires a conductor');
  return { ...scene, integrator: { maxDepth: scene.integrator.maxDepth }, materials: { ...scene.materials, metal: { ...metal, roughness: 0 } } };
}

function crop(image: Image, view: LocalityView): Image {
  const [x0, y0, x1, y1] = view.crop;
  const x = Math.floor(x0 * image.width), y = Math.floor(y0 * image.height);
  const width = Math.floor(x1 * image.width) - x, height = Math.floor(y1 * image.height) - y;
  const data = new Float32Array(width * height * image.channels);
  for (let i = 0; i < height; i++) {
    const start = ((y + i) * image.width + x) * image.channels;
    data.set(image.data.subarray(start, start + width * image.channels), i * width * image.channels);
  }
  return { width, height, channels: image.channels, data };
}

async function main() {
  const args = getArgs();
  if (!args.reflections) throw new Error('Pass reflections=<checkout> and set REFLECTIONS_REPO to the same checkout in the runner environment');
  const name = args.fixture ?? 'mirror';
  if (!['mirror', 'thin', 'moving', 'cut', 'emitter', 'delta', 'miss'].includes(name)) {
    throw new Error('fixture must be mirror, thin, moving, cut, emitter, delta or miss');
  }
  const [width, height] = (args.res ?? '128x72').split('x').map(Number) as [number, number];
  const spp = numberArg(args, 'spp', 128), refSpp = numberArg(args, 'refSpp', 4096), warm = numberArg(args, 'warm', 128);
  const upload = args.upload === '1';
  if (![width, height, spp, refSpp, warm].every(n => Number.isSafeInteger(n) && n > 0) || Math.min(width, height) < 16) {
    throw new Error('Expected positive integer sample counts and a resolution of at least 16x16');
  }
  const thin = name === 'thin' || name === 'cut';
  const fixture = thin ? thinWallFixture(width, height, 0.008) : movingCardFixture(width, height);
  const scene = mirrorScene(fixture.scene);
  const card = name === 'mirror' || name === 'emitter' || name === 'delta';
  const miss = name === 'miss';
  const cardInstance = scene.meshes.findIndex(mesh => mesh.name === 'moving-card');
  if (miss) scene.meshes = scene.meshes.filter(mesh => mesh.name === 'glossy-floor');
  if (name === 'emitter') {
    scene.meshes = scene.meshes.map((mesh, i) => i === cardInstance ? { ...mesh, material: 'light', emission: [3, 2, 1], twoSided: false } : mesh);
  }
  if (name === 'delta') {
    scene.meshes = scene.meshes.map((mesh, i) => i === cardInstance ? { ...mesh, material: 'metal' } : mesh);
    scene.meshes.push({
      name: 'two-mirror-emitter', material: 'light', emission: [3, 2, 1], geometry: { type: 'quad', size: [0.42, 0.44] },
      transform: compose(translation(-0.78, 0.75, -0.54), rotationX(-Math.PI / 2)),
    });
  }
  const views: readonly LocalityView[] = card || miss ? [{
    tag: 'offscreen',
    camera: { ...scene.camera, target: [-0.66, -0.46, -1.36], yFovDeg: 7 },
    crop: [0.12, 0.12, 0.88, 0.88],
  }] : thin ? fixture.views.map(view => ({ ...view, crop: [0.16, 0.82, 0.84, 0.97] })) : fixture.views;
  const pixels = width * height;

  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('No usable GPU');
  adapterSummary = summarizeAdapter(adapter);
  const { device, caps } = await requestDeviceForCaps(adapter, detectCaps(adapter), { label: 'nrc-reflection-consumer' });
  deviceErrors = watchDevice(device);
  const refl = await reflectionProviderModules(args.reflections);
  const input = sceneDescToCopyInput(scene);
  const provider = await makeReflectionProvider(refl, device, input);
  const sky = refl.environmentHitShading(device, 'equirect');
  const skyTexture = device.createTexture({ size: [1, 1], format: 'rgba32float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  const skyRadiance = input.environment?.radiance ?? [0, 0, 0];
  device.queue.writeTexture({ texture: skyTexture }, new Float32Array([...skyRadiance, 1]), { bytesPerRow: 16 }, [1, 1]);
  sky.setLighting(1, {});

  const grid = await HashGrid.create(device, { log2Entries: 15, minSamples: 16, cellPixels: 4, fallbackLevels: 2, pyramid: true }, {
    yFovDeg: views[0]!.camera.yFovDeg, filmHeight: height,
  });
  const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const frameData = new ArrayBuffer(96);
  const frameFloats = new Float32Array(frameData, 0, 16);
  const frameInts = new Uint32Array(frameData, 64);
  const params = device.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const output = device.createBuffer({ size: bytes(pixels * OUTPUT_WORDS), usage: storage });
  const records = device.createBuffer({ size: bytes(pixels * 16), usage: storage });
  const count = device.createBuffer({ size: 16, usage: storage });
  const staging = device.createBuffer({ size: output.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  grid.bindRecords(records, count, pixels);

  const hook = grid.displayHook;
  const C = GPUShaderStage.COMPUTE;
  const ioLayout = device.createBindGroupLayout({
    entries: [params, output, records, count].map((_, binding) => ({ binding, visibility: C, buffer: { type: binding === 0 ? 'uniform' : 'storage' } })),
  });
  const providerLayout = device.createBindGroupLayout({ entries: provider.entries.map((entry, binding) => ({ ...entry, binding })) });
  const emptyLayout = device.createBindGroupLayout({ entries: [] });
  const cacheLayout = device.createBindGroupLayout({ entries: hook.bindings });
  const ioGroup = device.createBindGroup({ layout: ioLayout, entries: [params, output, records, count].map((buffer, binding) => ({ binding, resource: { buffer } })) });
  const providerGroup = () => device.createBindGroup({ layout: providerLayout, entries: provider.resources().map((resource, binding) => ({ binding, resource })) });
  let sceneGroup = providerGroup();
  const cacheGroup = device.createBindGroup({ layout: cacheLayout, entries: hook.entries });
  const envGroup = sky.createBindGroup(skyTexture.createView());
  const renderer = async (environment: boolean) => {
    const code = [refl.HIT_WGSL, provider.code({ group: 1, firstBinding: 0 }), REFLECTION_TRANSPORT_WGSL,
      environment ? sky.wgsl : hook.wgsl, reflectionConsumerWgsl(environment)].join('\n');
    const module = device.createShaderModule({ code, label: environment ? 'environment hit shading' : 'strict indirect hit shading' });
    const diagnostics = (await module.getCompilationInfo()).messages.filter(message => message.type === 'error');
    if (diagnostics.length) throw new Error(diagnostics.map(d => `${d.lineNum}:${d.linePos} ${d.message}`).join('\n'));
    return device.createComputePipelineAsync({
      layout: device.createPipelineLayout({ bindGroupLayouts: [ioLayout, providerLayout, emptyLayout, environment ? sky.layout : cacheLayout] }),
      compute: { module, entryPoint: 'renderConsumer' },
    });
  };
  const [cachePipeline, envPipeline] = await Promise.all([renderer(false), renderer(true)]);
  const pt = await PtMegakernel.create(device, caps, {
    format: sceneFormatOf(provider.traceBuffers), layout: provider.sceneCopy.layout, scenePacking: provider.sceneCopyBuffers.packing,
    integrator: { counters: true },
  });
  pt.setFilm(width, height);
  pt.setLimits(scene.integrator);

  function dispatch(view: LocalityView, mode: number, seed: number, clear: boolean, environment = false): void {
    const basis = ptCameraBasis(view.camera, width, height);
    frameFloats.set([...basis.origin, 0, ...basis.right, 0, ...basis.up, 0, ...basis.forward, 0]);
    frameInts.set([width, height, seed, scene.integrator.maxDepth, mode, cardInstance >>> 0, 0, 0]);
    device.queue.writeBuffer(params, 0, frameData);
    const encoder = device.createCommandEncoder();
    if (clear) encoder.clearBuffer(output);
    if (mode === 0) encoder.clearBuffer(count);
    const pass = encoder.beginComputePass();
    pass.setPipeline(environment ? envPipeline : cachePipeline);
    pass.setBindGroup(0, ioGroup);
    pass.setBindGroup(1, sceneGroup);
    pass.setBindGroup(3, environment ? envGroup : cacheGroup);
    pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
    pass.end();
    if (mode === 0) grid.encodeUpdate(encoder);
    device.queue.submit([encoder.finish()]);
  }

  const metrics: Record<string, number> = { cache_bytes: grid.bytes };
  let updates = 0;
  const train = async (trainingViews: readonly LocalityView[], frames: number) => {
    const times: number[] = [];
    for (let i = 0; i < frames; i++) {
      const view = trainingViews[i % trainingViews.length]!;
      grid.setCameraPosition(view.camera.position);
      const start = performance.now();
      dispatch(view, 0, 7371 + updates++, false);
      await device.queue.onSubmittedWorkDone();
      times.push(performance.now() - start);
    }
    metrics.training_last_p50_ms = percentile(times, 50);
    metrics.training_last_p95_ms = percentile(times, 95);
  };
  const trace = async (view: LocalityView, samples: number, seed: number, timed = false) => {
    pt.setScene(provider.traceBuffers, provider.sceneCopyBuffers);
    pt.setCamera(view.camera);
    pt.setSeed(seed);
    const times: number[] = [];
    for (let n = 0; n < samples;) {
      const start = performance.now();
      const encoder = device.createCommandEncoder();
      if (n === 0) pt.reset(encoder);
      const batch = timed ? 1 : Math.min(64, samples - n);
      pt.encode(encoder, batch);
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      times.push(performance.now() - start);
      n += batch;
    }
    const image = (await pt.readImage()).data;
    return { image: { width, height, channels: 4, data: image } satisfies Image, counters: await pt.readCounters(), times };
  };

  const run = async (view: LocalityView, arm: ConsumerArm, countSpp: number, centre = false) => {
    grid.setCameraPosition(view.camera.position);
    const mode = centre ? 3 : { hash: 1, traced: 2, environment: 2, direct: 4 }[arm];
    const times: number[] = [];
    for (let n = 0; n < countSpp; n++) {
      const start = performance.now();
      dispatch(view, mode, 277 + n, n === 0, arm === 'environment');
      await device.queue.onSubmittedWorkDone();
      times.push(performance.now() - start);
    }
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(output, 0, staging, 0, output.size);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const data = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    const rgb = new Float32Array(pixels * 4);
    for (let pixel = 0; pixel < pixels; pixel++) {
      const i = pixel * OUTPUT_WORDS;
      for (let c = 0; c < 3; c++) rgb[pixel * 4 + c] = data[i + c]! / countSpp;
      rgb[pixel * 4 + 3] = 1;
    }
    return { data, image: { width, height, channels: 4, data: rgb } satisfies Image, times };
  };

  const evaluate = async (key: string, view: LocalityView, ref?: Image) => {
    const reference = ref ?? (await trace(view, refSpp, 5731)).image;
    const ptArm = await trace(view, spp, 6771, true);
    const ptError = compareImages(crop(ptArm.image, view), crop(reference, view));
    metrics[`${key}_pt_crop_relmse`] = ptError.relMse;
    metrics[`${key}_pt_wall_p50_ms`] = percentile(ptArm.times, 50);
    metrics[`${key}_pt_wall_p95_ms`] = percentile(ptArm.times, 95);
    metrics[`${key}_pt_rays_per_pixel`] = ptArm.counters.summary.extensionRays + ptArm.counters.summary.shadowRays;
    let prior: Float32Array<ArrayBufferLike> | undefined;
    for (const arm of ['environment', 'direct', 'traced', 'hash'] as const) {
      const rendered = await run(view, arm, spp);
      const tag = `${key}_${arm}`;
      const sample = crop(rendered.image, view), target = crop(reference, view);
      const m = compareImages(sample, target);
      if (m.nonFiniteTest || m.nonFiniteRef) throw new Error('Invalid reflection radiance');
      metrics[`${tag}_crop_relmse`] = m.relMse;
      metrics[`${tag}_crop_relbias`] = m.relBias;
      const sums = new Float64Array(OUTPUT_WORDS);
      for (let i = 0; i < rendered.data.length; i += OUTPUT_WORDS) {
        for (let n = 3; n < OUTPUT_WORDS; n++) sums[n]! += rendered.data[i + n]!;
        if (prior && (prior[i + 3] !== rendered.data[i + 3]
            || prior[i + 4] !== rendered.data[i + 4] || prior[i + 5] !== rendered.data[i + 5])) {
          throw new Error('Hit distance or geometry confidence changed with the shading arm');
        }
      }
      prior = rendered.data;
      if (!sums[3] || (miss ? sums[5] !== 0 : !sums[5])) throw new Error('Unexpected mirror hit count');
      if (card && sums[3] !== pixels * spp) throw new Error('The offscreen object fixture must show only the floor directly');
      metrics[`${tag}_primary_rays_per_pixel`] = sums[8]! / (pixels * spp);
      metrics[`${tag}_reflection_rays_per_pixel`] = sums[9]! / (pixels * spp);
      metrics[`${tag}_tail_rays_per_pixel`] = sums[10]! / (pixels * spp);
      metrics[`${tag}_shadow_rays_per_pixel`] = sums[11]! / (pixels * spp);
      metrics[`${tag}_hit_confidence`] = sums[5]! / sums[3]!;
      metrics[`${tag}_hit_mean_distance`] = sums[5] ? sums[4]! / sums[5] : -1;
      metrics[`${tag}_card_hit_fraction`] = sums[6]! / sums[3]!;
      if (sums[5]! + sums[16]! !== sums[3]!) throw new Error('Mirror hits plus misses must match launched reflection rays');
      metrics[`${tag}_emitter_hit_fraction`] = sums[17]! / sums[3]!;
      if (arm === 'hash' && sums[12]) {
        metrics[`${tag}_cache_hit_fraction`] = sums[13]! / sums[12]!;
        metrics[`${tag}_cache_coarse_fraction`] = sums[14]! / sums[12]!;
      }
      metrics[`${tag}_wall_p50_ms`] = percentile(rendered.times, 50);
      metrics[`${tag}_wall_p95_ms`] = percentile(rendered.times, 95);
      if (upload) await uploadImage(tag, sample);
    }
    if (upload) await uploadImage(`${key}_reference`, crop(reference, view));
    return reference;
  };

  // The cold hash should trace its whole suffix whenever it lacks support.
  if (name === 'mirror') {
    const cold = await run(views[0]!, 'hash', 1);
    const explicit = await run(views[0]!, 'traced', 1);
    if (cold.image.data.some((v, i) => v !== explicit.image.data[i])) throw new Error('Cold hash diverges from traced fallback');
  }
  if (!miss) await train(name === 'cut' ? views.slice(0, 1) : views, warm);
  for (const view of views) await evaluate(view.tag, view);

  if (card || miss) {
    const view = views[0]!;
    const center = await run(view, 'traced', 1, true);
    const pixel = Math.floor(width / 2) + Math.floor(height / 2) * width;
    const i = pixel * OUTPUT_WORDS;
    if (card && !center.data[i + 6]) throw new Error('Offscreen card missed by the center mirror ray');
    if (name === 'emitter' && !center.data[i + 17]) throw new Error('Expected a front-face emissive hit after one mirror');
    if (name === 'delta' && center.data[i + 17]) throw new Error('Two-mirror fixture measured a one-mirror emissive hit');
    const basis = ptCameraBasis(view.camera, width, height);
    const x = 2 * (Math.floor(width / 2) + 0.5) / width - 1;
    const y = 2 * (Math.floor(height / 2) + 0.5) / height - 1;
    const dir = basis.forward.map((v, a) => v + basis.right[a]! * x - basis.up[a]! * y);
    const normalized = dir.map(v => v / Math.hypot(...dir));
    if (name === 'emitter' || name === 'delta' || miss) {
      const material = scene.materials.metal;
      if (material?.type !== 'conductor') throw new Error('Missing mirror material');
      const errors = (miss ? skyRadiance : [3, 2, 1]).map((radiance, c) => {
        const fresnel = (axis: number) => ptFrComplex(Math.abs(normalized[axis]!), material.eta[c]!, material.k[c]!);
        const expected = radiance * fresnel(1) * (name === 'delta' ? fresnel(2) : 1);
        return Math.abs(center.image.data[pixel * 4 + c]! - expected);
      });
      metrics.offscreen_center_radiance_max_abs_error = Math.max(...errors);
      if (metrics.offscreen_center_radiance_max_abs_error > (miss ? 0.00002 : 0.002)) throw new Error('Wrong sky or emitter radiance after Fresnel reflection');
    }
    if (card) {
      const floorT = -basis.origin[1] / normalized[1]!;
      const floorZ = basis.origin[2] + normalized[2]! * floorT;
      const expected = (-1.36 + 0.075 / 2 - floorZ) / normalized[2]!;
      const distanceError = Math.abs(center.data[i + 4]! - expected);
      if (distanceError > 0.001) throw new Error(`Offscreen mirror hit distance error: ${distanceError}`);
      metrics.offscreen_center_distance_error = distanceError;
    }
  }
  if (name === 'moving' && fixture.movedScene) {
    provider.setTransforms(fixture.movedScene.meshes.map(mesh => mesh.transform ?? identity()));
    sceneGroup = providerGroup();
    const view = views[0]!;
    const moved = await evaluate('moved_stale', view);
    await train(views, 64);
    await evaluate('moved_f64', view, moved);
  }
  if (name === 'cut') {
    const view = views[1]!;
    await train([view], 64);
    await evaluate('cut_f64', view);
  }
  metrics.train_frames = updates;
  await drainDevice(device);
  if (deviceErrors.length) throw new Error(deviceErrors.join(' | '));
  reportResult({
    scene: name, renderer: 'nrc-reflection-consumer', valid: true, metrics,
    config: { width, height, spp, refSpp, warm },
    notes: { provider: args.reflections, views, timing: 'CPU wall per dispatch and queue drain, includes counters; provisional' },
  }, { adapter: adapterSummary, deviceErrors });
  for (const buffer of [params, output, records, count, staging]) buffer.destroy();
  pt.destroy();
  grid.destroy();
  sky.destroy();
  skyTexture.destroy();
  provider.destroy();
}

main().catch(error => reportError(error, { adapter: adapterSummary, deviceErrors }));
