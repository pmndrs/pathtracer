# @pmndrs/pathtracer

[![npm](https://img.shields.io/npm/v/@pmndrs/pathtracer?style=flat&colorA=000000&colorB=000000)](https://www.npmjs.com/package/@pmndrs/pathtracer)
[![Docs](https://img.shields.io/badge/docs-pmndrs.github.io-000000?style=flat&colorA=000000)](https://pmndrs.github.io/pathtracer/)

A physically based path tracer on WebGPU, built on [`@pmndrs/raytracer`](https://www.npmjs.com/package/@pmndrs/raytracer): OpenPBR materials, area lights, sun and HDR environments, textures and glTF, megakernel and wavefront schedulers, and modes from converged reference images to one sample per pixel per frame for a denoiser.

```sh
npm install @pmndrs/pathtracer @pmndrs/raytracer
```

```ts
import { buildGpuScene, loadGltf } from '@pmndrs/raytracer';
import { PtMegakernel, ptSceneOptions } from '@pmndrs/pathtracer';

const scene = await buildGpuScene(device, await loadGltf('/model.glb'));
const pt = await PtMegakernel.create(device, caps, ptSceneOptions(scene));
pt.setScene(scene.trace, scene.buffers);
pt.setFilm(1280, 720);
pt.setCamera({ position: [0, 1, 4], target: [0, 0.5, 0], up: [0, 1, 0], yFovDeg: 45 });

const encoder = device.createCommandEncoder();
pt.encode(encoder, 16); // 16 samples per pixel
device.queue.submit([encoder.finish()]);
const image = await pt.readImage(); // linear HDR RGBA float32
```

- [Documentation](https://pmndrs.github.io/pathtracer/): getting started, concepts, megakernel and wavefront, integrator options, real time, API reference
- [Examples](https://pmndrs.github.io/pathtracer/examples/) and the [Classroom showcase](https://pmndrs.github.io/pathtracer/examples/showcase/classroom/)
- [Repository](https://github.com/pmndrs/pathtracer)

MIT licence.
