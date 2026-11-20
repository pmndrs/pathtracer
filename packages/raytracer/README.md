# @pmndrs/raytracer

[![npm](https://img.shields.io/npm/v/@pmndrs/raytracer?style=flat&colorA=000000&colorB=000000)](https://www.npmjs.com/package/@pmndrs/raytracer)
[![Docs](https://img.shields.io/badge/docs-pmndrs.github.io-000000?style=flat&colorA=000000)](https://pmndrs.github.io/pathtracer/raytracer/overview)

Ray tracing on WebGPU from your own WGSL: rays in, hits out. A BVH from [`@pmndrs/bvh`](https://www.npmjs.com/package/@pmndrs/bvh), a scene copy of materials, lights and textures, and WGSL to trace, read surfaces and sample lights inside your own compute shaders. It is the core of [`@pmndrs/pathtracer`](https://www.npmjs.com/package/@pmndrs/pathtracer); use it on its own for reflections, shadows, AO, GI probes or baking.

```sh
npm install @pmndrs/raytracer
```

```ts
import { buildGpuScene, rtModuleWgsl, sceneFormatOf } from '@pmndrs/raytracer';

const scene = await buildGpuScene(device, input); // BVH + scene copy on the GPU
const wgsl = rtModuleWgsl(caps, {
  ...sceneFormatOf(scene.trace),
  traceSlot: { group: 0, firstBinding: 0 },
  scene: { slot: { group: 1, firstBinding: 0 }, layout: scene.copy.layout, textures: scene.copy.textureLayout, packing: scene.packing },
}) + myKernel; // your kernel calls rtClosest(ray), rtVisible(ray), rtSurface(hit), ...
```

- [Documentation](https://pmndrs.github.io/pathtracer/raytracer/overview): inline WGSL, the batch tracer, interval queries, the scene copy, shading, API reference
- [Examples](https://pmndrs.github.io/pathtracer/examples/): hit inspector, ambient occlusion, one-bounce shading, batch tracing
- [Repository](https://github.com/pmndrs/pathtracer)

MIT licence.
