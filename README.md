# pathtracer

[![npm @pmndrs/pathtracer](https://img.shields.io/npm/v/@pmndrs/pathtracer?label=%40pmndrs%2Fpathtracer&style=flat&colorA=000000&colorB=000000)](https://www.npmjs.com/package/@pmndrs/pathtracer)
[![npm @pmndrs/raytracer](https://img.shields.io/npm/v/@pmndrs/raytracer?label=%40pmndrs%2Fraytracer&style=flat&colorA=000000&colorB=000000)](https://www.npmjs.com/package/@pmndrs/raytracer)
[![Docs](https://img.shields.io/badge/docs-pmndrs.github.io-000000?style=flat&colorA=000000)](https://pmndrs.github.io/pathtracer/)
[![Examples](https://img.shields.io/badge/examples-live-000000?style=flat&colorA=000000)](https://pmndrs.github.io/pathtracer/examples/)
[![Discord](https://img.shields.io/discord/740090768164651008?style=flat&colorA=000000&colorB=000000&label=discord&logo=discord&logoColor=ffffff)](https://discord.gg/poimandres)

Ray tracing and path tracing on WebGPU, in the browser. Two packages:

- **[`@pmndrs/raytracer`](packages/raytracer)** traces rays against a BVH from your own WGSL: closest hit, any hit, interval queries, surface data, materials and lights. Use it for reflections, shadows, AO, GI probes or baking inside your own renderer.
- **[`@pmndrs/pathtracer`](packages/pathtracer)** is a physically based path tracer built on it: OpenPBR materials, many lights, textures and glTF, two GPU schedulers, and modes from converged reference images to one sample per pixel per frame.

**[Documentation](https://pmndrs.github.io/pathtracer/)** · **[Examples](https://pmndrs.github.io/pathtracer/examples/)** · **[Classroom showcase](https://pmndrs.github.io/pathtracer/examples/showcase/classroom/)** · [npm](https://www.npmjs.com/package/@pmndrs/pathtracer)

## Install

```sh
npm install @pmndrs/pathtracer @pmndrs/raytracer
```

You need a browser with WebGPU (Chrome or Edge 113+, Safari 26, Firefox 141+ on Windows) and an ES module bundler. With Vite, add `optimizeDeps: { exclude: ['@pmndrs/bvh'] }` so the BVH build workers load. See [Installation](https://pmndrs.github.io/pathtracer/getting-started/installation) and [Browser and GPU support](https://pmndrs.github.io/pathtracer/getting-started/browser-support).

## A first render

```ts
import { buildGpuScene, detectCaps, requestDeviceForCaps, type SceneCopyInput } from '@pmndrs/raytracer';
import { PtMegakernel, ptSceneOptions } from '@pmndrs/pathtracer';

const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
const { device, caps } = await requestDeviceForCaps(adapter!, detectCaps(adapter!));

// A floor and a lamp above it (OpenPBR materials; anything left out takes its OpenPBR default).
const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const input: SceneCopyInput = {
  meshes: [
    { positions: new Float32Array([-3, 0, 3, 3, 0, 3, 3, 0, -3, -3, 0, -3]), indices: [0, 1, 2, 0, 2, 3] },
    { positions: new Float32Array([-0.5, 2, -0.5, 0.5, 2, -0.5, 0.5, 2, 0.5, -0.5, 2, 0.5]), indices: [0, 1, 2, 0, 2, 3] },
  ],
  instances: [
    { meshIndex: 0, instanceIndex: 0, objectToWorld: identity, material: 0 },
    { meshIndex: 1, instanceIndex: 1, objectToWorld: identity, material: 1 },
  ],
  materials: [
    { baseColor: [0.8, 0.8, 0.8], specularRoughness: 0.4 },
    { baseColor: [0, 0, 0], specularWeight: 0, emissionColor: [20, 18, 15] },
  ],
  environment: { radiance: [0.15, 0.2, 0.3] },
};

const scene = await buildGpuScene(device, input);                     // BVH + scene copy, uploaded
const pt = await PtMegakernel.create(device, caps, ptSceneOptions(scene)); // a kernel compiled for this scene
pt.setScene(scene.trace, scene.buffers);
pt.setFilm(640, 360);
pt.setCamera({ position: [0, 1.5, 5], target: [0, 0.5, 0], up: [0, 1, 0], yFovDeg: 45 });

for (let done = 0; done < 256; done += 16) {
  const encoder = device.createCommandEncoder();
  pt.encode(encoder, 16); // 16 samples per pixel
  device.queue.submit([encoder.finish()]);
}
const image = await pt.readImage(); // linear HDR RGBA float32, the mean of all samples
```

The [full walkthrough](https://pmndrs.github.io/pathtracer/getting-started/your-first-render) shows the image on a canvas. [Loading glTF](https://pmndrs.github.io/pathtracer/getting-started/loading-gltf) loads a model in one call (`loadGltf`).

## Features

| | |
| --- | --- |
| **Trace core** | BVH from [`@pmndrs/bvh`](https://www.npmjs.com/package/@pmndrs/bvh) (binary, BVH4, CWBVH8), watertight triangle tests, inline WGSL or a batch tracer, interval queries, ray cones |
| **Materials** | OpenPBR: metal, dielectric, coat, fuzz, thin film, transmission, thin-walled, anisotropy; glTF PBR and the common `KHR_materials_*` extensions; textures with GPU mips, normal maps, alpha cutouts |
| **Lights** | Area lights, sun, sky colour, HDR environment maps with importance sampling, light selection by power |
| **Integrator** | Next-event estimation with MIS, Owen-Sobol, PCG or blue-noise sampling, Russian roulette |
| **Schedulers** | Megakernel or wavefront (queues, refill, sorting), the same images per pixel |
| **Real time** | One sample per frame, demodulated frame planes and motion vectors for a denoiser and upscaler, ReSTIR DI |
| **Experimental** | Volumes (fog, density grids), spectral rendering and dispersion, ReSTIR GI, MaterialX graphs |

Every feature is checked against pbrt-v4 and Mitsuba 3 references, analytic expectations or a CPU twin before it ships.

## Examples

The [gallery](https://pmndrs.github.io/pathtracer/examples/) runs in the browser. Run it locally with:

```sh
pnpm install
pnpm examples
```

- Getting started: [Hello, path tracer](https://pmndrs.github.io/pathtracer/examples/start/hello/) · [glTF models](https://pmndrs.github.io/pathtracer/examples/start/gltf/)
- Ray tracer: [Hit inspector](https://pmndrs.github.io/pathtracer/examples/raytracer/hits/) · [Ambient occlusion](https://pmndrs.github.io/pathtracer/examples/raytracer/ao/) · [One-bounce shading](https://pmndrs.github.io/pathtracer/examples/raytracer/shade/) · [Batch tracing](https://pmndrs.github.io/pathtracer/examples/raytracer/batch/)
- Path tracer: [Megakernel or wavefront](https://pmndrs.github.io/pathtracer/examples/pathtracer/schedulers/) · [Alpha cutouts](https://pmndrs.github.io/pathtracer/examples/pathtracer/cutouts/) · [Many lights](https://pmndrs.github.io/pathtracer/examples/pathtracer/lights/) · [Fog and clouds](https://pmndrs.github.io/pathtracer/examples/pathtracer/volumes/) · [Spectral](https://pmndrs.github.io/pathtracer/examples/pathtracer/spectral/) · [MaterialX](https://pmndrs.github.io/pathtracer/examples/pathtracer/materialx/)

## Status

Version 0.1. The APIs can still change between minor versions; pages marked **Experimental** in the docs change more often. Where the project stands: [STATUS.md](STATUS.md). Open work: [TASKS.md](TASKS.md).

## Development

pnpm workspace, TypeScript (strict, ESM), Node 22+.

```sh
pnpm install
pnpm test        # unit tests, no GPU
pnpm typecheck
pnpm examples    # the examples gallery (Vite)
node scripts/build-site.mjs --docs   # docs + examples site into dist/site
```

Contributor notes: [CLAUDE.md](CLAUDE.md) (rules, conventions and known traps), [DECISIONS.md](DECISIONS.md), [DIVERGES.md](DIVERGES.md) (where we depart from a paper or renderer, and why). Research records: [research/BRIEF.md](research/BRIEF.md), [research/notes/](research/notes/), [spec/SPEC-draft.md](spec/SPEC-draft.md). Design notes: [docs/internal/design/](docs/internal/design/). The bench (`bench/`) holds the reference renders, competitor comparisons and GPU check pages.

## License

MIT, see [LICENSE](LICENSE). Ported code and third-party assets: [THIRD_PARTY.md](THIRD_PARTY.md).
