/**
 * Every example, once: the gallery index, the nav every example page gets, the Vite build inputs and the docs' examples page
 * (`node scripts/build-site.mjs` writes docs/getting-started/examples.mdx from it) all read this list.
 *
 * Adding an example: a folder `examples/<category>/<slug>/` with `index.html` + `main.ts` (copy `start/hello`), and an entry here.
 * Pure data: no DOM, no imports (vite.config.ts and Node scripts load it).
 */

export type CategoryId = 'showcase' | 'start' | 'raytracer' | 'pathtracer' | 'realtime' | 'lab';

/**
 * - `stable`: built on the published API with its current defaults.
 * - `experimental`: works and is checked, but the API or its defaults can still change.
 * - `research`: a study from one of our research tracks, not a supported feature; it may move or go away.
 */
export type ExampleStatus = 'stable' | 'experimental' | 'research';

export interface Category {
  id: CategoryId;
  title: string;
  blurb: string;
}

export interface ExampleEntry {
  /** Unique, also the anchor on the index. */
  slug: string;
  /** Folder under examples/ (and URL under the gallery root), with a trailing slash. */
  path: string;
  title: string;
  /** One line for the card. */
  summary: string;
  /** What to look for, shown in the page's info panel. */
  description: string;
  category: CategoryId;
  status: ExampleStatus;
  /** Features, shown as tags (`gltf`, `openpbr`, `nee`, ...). */
  tags: string[];
  /** Docs page path below the docs root (no leading slash), linked from the page's nav. */
  docs?: string;
  /** Repository path of the source, linked from the page's nav. */
  source: string;
  /** Optional WebGPU features the example needs; the index marks it when the browser lacks one. */
  requires?: GPUFeatureName[];
  /** Streams assets from another host at run time. */
  network?: boolean;
  /** Heavy on the GPU (a laptop fan will notice). */
  heavy?: boolean;
  /** Card image under examples/ (16:9, e.g. `start/hello/thumb.webp`); a generated pattern when absent. */
  thumbnail?: string;
  /** Built by another build (not this Vite app); the gallery links it only in the deployed site. `dev` says how to run it locally. */
  external?: { dev: string };
}

export const CATEGORIES: readonly Category[] = [
  { id: 'showcase', title: 'Showcases', blurb: 'Bigger scenes built to be looked at: what the libraries do together.' },
  { id: 'start', title: 'Getting started', blurb: 'The smallest complete programs: a first image, a glTF model.' },
  { id: 'raytracer', title: 'Ray tracer', blurb: '@pmndrs/raytracer on its own: rays in, hits out, in your own shaders.' },
  { id: 'pathtracer', title: 'Path tracer', blurb: '@pmndrs/pathtracer features one at a time.' },
  { id: 'realtime', title: 'Real time', blurb: 'One sample per pixel per frame, denoised and upscaled.' },
  { id: 'lab', title: 'Lab', blurb: 'Tools for checking images and GPUs.' },
];

export const STATUS_LABELS: Record<ExampleStatus, { label: string; blurb: string }> = {
  stable: { label: 'Stable', blurb: 'Built on the published API with its current defaults.' },
  experimental: { label: 'Experimental', blurb: 'Works and is checked, but the API or its defaults can still change.' },
  research: { label: 'Research', blurb: 'A study from one of our research tracks, not a supported feature. It may move or go away.' },
};

export const REPOSITORY = 'https://github.com/pmndrs/pathtracer';
/** The deployed site (docs at the root, the gallery under `examples/`). */
export const SITE_URL = 'https://pmndrs.github.io/pathtracer/';

export const EXAMPLES: readonly ExampleEntry[] = [
  {
    slug: 'classroom',
    path: 'showcase/classroom/',
    title: 'Classroom',
    summary: 'An interior lit by the sun through the windows, with a neural radiance cache, SVGF and temporal upscaling.',
    description:
      'Blender’s Classroom scene path traced at 640 × 360, one bounce traced before a neural radiance cache that trains while you look around, ' +
      'denoised with SVGF and upscaled to 1280 × 720. Split the view to compare plain path tracing with the cache, change the daylight, or step through the pipeline stages.',
    category: 'showcase',
    status: 'research',
    tags: ['nrc', 'svgf', 'upscaler', 'daylight'],
    docs: 'getting-started/introduction',
    source: 'bench/hero',
    heavy: true,
    external: { dev: 'python3 bench/hero/prepare.py classroom && node bench/live.ts nrc-hero' },
  },
  {
    slug: 'hello',
    path: 'start/hello/',
    title: 'Hello, path tracer',
    summary: 'A Cornell box built in code and path traced progressively.',
    description:
      'The smallest complete program: build a scene from triangles, compile a path tracer for it and accumulate samples into the canvas. ' +
      'Drag to orbit, scroll to zoom. The image restarts when the camera moves and converges while it is still.',
    category: 'start',
    status: 'stable',
    tags: ['progressive', 'nee', 'area light'],
    docs: 'getting-started/your-first-render',
    source: 'examples/start/hello',
  },
  {
    slug: 'gltf',
    path: 'start/gltf/',
    title: 'glTF models',
    summary: 'Load a glTF 2.0 model with its OpenPBR materials, under a sky and a sun.',
    description:
      '`loadGltf` reads the model, its textures and its cameras; `buildGpuScene` builds the trees and the scene copy; `ptSceneOptions` compiles a kernel for exactly those materials. ' +
      'Pick one of the Khronos sample models or drop your own .glb onto the page.',
    category: 'start',
    status: 'stable',
    tags: ['gltf', 'openpbr', 'textures', 'sun'],
    docs: 'getting-started/loading-gltf',
    source: 'examples/start/gltf',
    network: true,
  },
  {
    slug: 'hits',
    path: 'raytracer/hits/',
    title: 'Hit inspector',
    summary: 'rtClosest and rtSurface in your own compute shader: normals, uvs, materials and distances of the first hit.',
    description:
      'One camera ray per pixel per frame through `rtClosest`, then `rtSurface` decodes what was hit. Pick the property to show: shading or geometric normal, ' +
      'front or back face, distance bands, texture coordinates, barycentrics, material and instance ids, or the base colour with its texture. No path tracer is involved.',
    category: 'raytracer',
    status: 'stable',
    tags: ['rtClosest', 'rtSurface', 'inline WGSL'],
    docs: 'raytracer/inline-wgsl',
    source: 'examples/raytracer/hits',
    network: true,
  },
  {
    slug: 'ao',
    path: 'raytracer/ao/',
    title: 'Ambient occlusion',
    summary: 'rtVisible, the cheapest query: cosine rays over the hemisphere ask only whether anything is within a radius.',
    description:
      'For the first hit of each pixel, a few cosine-distributed rays per frame ask `rtVisible` whether anything lies within the radius: an any-hit walk that stops at the first blocker. ' +
      'The fraction that escapes converges to the ambient occlusion. Change the radius to see contact shadows turn into broad darkening.',
    category: 'raytracer',
    status: 'stable',
    tags: ['rtVisible', 'any hit', 'inline WGSL'],
    docs: 'raytracer/inline-wgsl',
    source: 'examples/raytracer/ao',
    network: true,
  },
  {
    slug: 'shade',
    path: 'raytracer/shade/',
    title: 'One-bounce shading',
    summary: 'rtShade at the first hit: emission and one light sample with its shadow ray, the lighting a reflection or GI effect needs.',
    description:
      '`rtShade(hit, ray, mode, u, mask)` lights a hit with one sample of the scene\u2019s lights (sun, sky, emissive triangles) and one shadow ray, and returns the environment for a miss. ' +
      'It is not a path tracer: there is no indirect light unless you plug your own cache into `rtShadeIndirect`. Compare the full, Lambert-only and flat modes.',
    category: 'raytracer',
    status: 'stable',
    tags: ['rtShade', 'light sampling', 'inline WGSL'],
    docs: 'raytracer/shading',
    source: 'examples/raytracer/shade',
    network: true,
  },
  {
    slug: 'batch',
    path: 'raytracer/batch/',
    title: 'Batch tracing',
    summary: 'RtBatchTracer: camera rays and shadow rays from GPU buffers, hits and visibility back into GPU buffers, five passes and no readback.',
    description:
      'Each frame: our kernel writes camera rays into a buffer, `RtBatchTracer` (closest) turns them into hit records, our kernel builds shadow rays towards the sun, ' +
      '`RtBatchTracer` (visible) answers them, and a last kernel composes base colour, sun and sky. The trace core never appears in our shaders\u2019 text: the batch tracer owns the walk.',
    category: 'raytracer',
    status: 'stable',
    tags: ['RtBatchTracer', 'shadow rays', 'GPU buffers'],
    docs: 'raytracer/batch-tracer',
    source: 'examples/raytracer/batch',
    network: true,
  },
  {
    slug: 'schedulers',
    path: 'pathtracer/schedulers/',
    title: 'Megakernel or wavefront',
    summary: 'The same estimator scheduled two ways: one kernel per sample, or kernels connected by GPU queues. Same image, different time.',
    description:
      '`PtMegakernel` traces a whole path per thread; `PtWavefront` splits a sample into camera, intersect, shade and shadow kernels joined by queues and indirect dispatch, ' +
      'and can refill a pixel as soon as its path ends. The image is the same per pixel. Switch schedulers and their options and watch the frame time; ' +
      'the stats show what `ptwfAutoPolicy` would pick for this film and path length. Frame times here are wall-clock readings, not benchmarks.',
    category: 'pathtracer',
    status: 'stable',
    tags: ['PtMegakernel', 'PtWavefront', 'ptwfAutoPolicy'],
    docs: 'concepts/schedulers',
    source: 'examples/pathtracer/schedulers',
    network: true,
  },
  {
    slug: 'cutouts',
    path: 'pathtracer/cutouts/',
    title: 'Alpha cutouts',
    summary: 'Leaf cards with alpha-masked textures: exact cutout shadows, and the faster settings that trade bias or noise for speed.',
    description:
      'Cards with an `alphaMode` MASK material are cut out where their texture alpha is below the cutoff: camera and bounce rays step through the holes. ' +
      'Shadow rays are the expensive part. Compare the exact test with opaque shadows (card-shaped, too dark), roulette (unbiased, noisier) and transmit (uniform partial shadow).',
    category: 'pathtracer',
    status: 'stable',
    tags: ['alpha mask', 'textures', 'shadows'],
    docs: 'pathtracer/integrator-options',
    source: 'examples/pathtracer/cutouts',
  },
  {
    slug: 'lights',
    path: 'pathtracer/lights/',
    title: 'Many lights',
    summary: '64 coloured lamps, a sun and a sky: light selection strategies for next-event estimation, and ReSTIR DI.',
    description:
      'Next-event estimation picks one light per vertex; how it picks decides the noise. Switch the light table\u2019s selection rule (pbrt, power, kind, capped), ' +
      'or turn on ReSTIR DI (experimental), which resamples several candidates per pixel and reuses good picks over time and between neighbours.',
    category: 'pathtracer',
    status: 'stable',
    tags: ['NEE', 'light selection', 'ReSTIR DI'],
    docs: 'concepts/lights',
    source: 'examples/pathtracer/lights',
  },
  {
    slug: 'volumes',
    path: 'pathtracer/volumes/',
    title: 'Fog and clouds',
    summary: 'Participating media: homogeneous fog filling the Cornell box, or a cloud read from a 3D density grid.',
    description:
      'A medium lives inside a closed mesh whose material is a medium boundary: rays pass through it and switch the medium they travel in. ' +
      'Set the extinction, the scattering albedo and the Henyey-Greenstein anisotropy, or switch to a cloud whose density comes from a procedural 3D grid.',
    category: 'pathtracer',
    status: 'experimental',
    tags: ['volumes', 'media', 'density grid'],
    docs: 'pathtracer/volumes',
    source: 'examples/pathtracer/volumes',
    heavy: true,
  },
  {
    slug: 'spectral',
    path: 'pathtracer/spectral/',
    title: 'Spectral rendering and dispersion',
    summary: 'Paths that carry wavelengths: glass whose index of refraction depends on the wavelength splits white bars into colour fringes.',
    description:
      'With `integrator.spectral` every path carries three wavelengths: colours are lifted to spectra and converted back at the end. A dispersive glass prism and slab stand in front of bright bars, ' +
      'so the split shows as colour fringes in the refracted view. Turn spectral off to see the same scene in RGB, where dispersion is ignored.',
    category: 'pathtracer',
    status: 'experimental',
    tags: ['spectral', 'dispersion', 'glass'],
    docs: 'pathtracer/spectral',
    source: 'examples/pathtracer/spectral',
  },
];

export const exampleBySlug = (slug: string): ExampleEntry | undefined => EXAMPLES.find((e) => e.slug === slug);
