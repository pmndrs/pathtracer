/**
 * Every example, once: the gallery index, the nav every example page gets, the Vite build inputs and the docs' examples page
 * (`node scripts/build-site.mjs` writes docs/getting-started/examples.mdx from it) all read this list.
 *
 * Adding an example: a folder `examples/<category>/<slug>/` with `index.html` + `main.ts` (copy `start/hello`), and an entry here.
 *
 * Every example follows three principles, and its `brief` answers them on the page (the info panel opens on a first visit):
 *  1. What is the point? A feature, a performance or quality trade-off, a lesson, a debugging aid or information (`brief.kind`, `brief.shows`).
 *  2. Can a visitor see it? The page has the controls that make the point (`brief.tryThis`) and shows things as they would really be used.
 *  3. A performance or quality page also shows the "bad" version, so the difference can be perceived (`brief.compare`, required for those kinds).
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

/** What kind of point an example makes (principle 1). */
export type ExampleKind = 'showcase' | 'feature' | 'performance' | 'quality' | 'educational' | 'debugging' | 'informative';

/** The example's brief, shown at the top of its info panel: what it shows, what to try, and what to compare it against. */
export interface ExampleBrief {
  kind: ExampleKind;
  /** What this shows and why it matters, in one or two sentences a newcomer can follow. */
  shows: string;
  /** The one to three things to do on the page that make the point. */
  tryThis: readonly string[];
  /** The baseline to compare against: the control that shows the slower, noisier or plainer way. Required for `performance` and `quality`. */
  compare?: string;
}

export interface ExampleEntry {
  /** Unique, also the anchor on the index. */
  slug: string;
  /** Folder under examples/ (and URL under the gallery root), with a trailing slash. */
  path: string;
  title: string;
  /** One line for the card. */
  summary: string;
  /** How it works, shown in the page's info panel below the brief. */
  description: string;
  /** What it shows, what to try and what to compare against (the three principles above). */
  brief: ExampleBrief;
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
  /** Card image under examples/ when it is not `<path>thumb.webp` (16:9, 640 x 360 webp); a generated pattern when there is none. */
  thumbnail?: string;
  /** Built by another build (not this Vite app); the gallery links it only in the deployed site. `dev` says how to run it locally. */
  external?: { dev: string };
  /**
   * `scripts/examples-smoke.mjs`: extra query for the headless check (`?offscreen=1` is always added; the check prefers scenes that
   * need no network), the samples or frames to wait for, and `image: false` for a page that draws nothing to its canvas. `thumbQuery` /
   * `thumbSamples` are used instead when capturing the card image (`--thumbs`): a nicer scene, more samples.
   */
  check?: { query?: string; samples?: number; image?: boolean; thumbQuery?: string; thumbSamples?: number };
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
    brief: {
      kind: "showcase",
      shows: "A full real-time pipeline on a real interior: path tracing at 640 × 360 with a neural radiance cache for the light after the first bounce, SVGF denoising, and temporal upscaling to 1280 × 720.",
      tryThis: ["Set Pipeline stage to Raw radiance to see what the path tracer gives in one frame, then Denoiser output, then Final output", "Choose Plain PT / NRC to split the view: the left half has no cache", "Open Render controls to move the sun"],
      compare: "Comparison: Plain path tracing (no cache) and Pipeline stage: Raw radiance (no denoiser, no upscaler)",
    },
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
    brief: {
      kind: "educational",
      shows: "The smallest complete path tracer program: a scene built from triangles, a kernel compiled for it, and samples added into the canvas until the noise is gone.",
      tryThis: ["Drag to orbit: the image restarts noisy and converges in a second or two", "Set Samples › stop at to 1 to see one sample per pixel, the raw path tracer output"],
    },
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
    brief: {
      kind: "feature",
      shows: "Loading a glTF 2.0 model in one call (`loadGltf`) and path tracing its materials and textures as OpenPBR, under a sky and a sun.",
      tryThis: ["Pick another Khronos model", "Drop your own .glb onto the page", "Move the sun"],
    },
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
    title: 'Rays from your own shader',
    summary: 'What ray tracing adds to a raster pass: shadow, reflection and AO rays traced from every hit in your own compute shader. Click a pixel to see its rays.',
    description:
      'Each pixel shoots one camera ray (`rtClosest`) and decodes its hit (`rtSurface`); that first hit is exactly what a rasteriser draws. From the hit the shader traces a shadow ray to the sun, a mirror reflection ray and a few AO rays against the whole scene, ' +
      'including what is off screen. Clicking a pixel runs a second entry point that traces its rays once more; the page draws them over the image with each hit. The G-buffer views show what the first hit carries.',
    brief: {
      kind: "educational",
      shows: "What ray tracing adds to a raster pass. The first hit of each camera ray is exactly what a rasteriser draws, so that view alone looks like raster. The rays traced from each hit (shadows, reflections, ambient occlusion) are what raster cannot do without extra passes or screen-space tricks.",
      tryThis: ["Turn on Shadow rays, Reflection rays and AO rays one at a time", "Click a pixel: its rays are drawn in the scene with each hit's distance and surface", "Switch Show to a G-buffer layer (normals, uv, ids) to see what the first hit carries"],
      compare: "Rays from each hit: none (primary hits only), which is what a raster G-buffer gives you",
    },
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
    brief: {
      kind: "performance",
      shows: "Ambient occlusion with `rtVisible`, the any-hit query: it only asks whether anything lies within the radius, so the walk stops at the first blocker instead of searching for the nearest one.",
      tryThis: ["Change the radius: small gives contact shadows, large gives soft room-scale darkening", "Switch Query to rtClosest and compare the frame time: the image is the same"],
      compare: "Query: rtClosest (closest hit) traces the same rays and gives the same image, slower",
    },
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
    brief: {
      kind: "feature",
      shows: "`rtShade`: one bounce of direct light (area lights, sun and sky, with a shadow ray each) computed in your own kernel, without the path tracer.",
      tryThis: ["Switch Mode between Flat, Lambert and Full to see what each layer of shading adds", "Pick another model"],
      compare: "Mode: Flat (albedo only, no rays)",
    },
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
      'Every pixel is ray traced, nothing is rasterised: per frame, one camera ray (`RtBatchTracer` closest hit) and one shadow ray toward the sun (`RtBatchTracer` visible), each a buffer of rays in and a buffer of results out. ' +
      'There is no denoiser. With a point sun and a constant sky only the anti-aliasing jitter is random, so the image is clean within a second; a larger **sun radius** (the default is the real sun\'s quarter degree) makes every shadow ray a random sample of a wide disc: soft, grainy penumbrae that converge, and **stop at** 1 or **pixel jitter** off shows the jagged single-ray edges.',
    brief: {
      kind: "performance",
      shows: "`RtBatchTracer`: rays are written to a GPU buffer, traced in a separate pass, and the hits come back in another buffer, with no readback. Use it when your rays come from many places or you want tracing separate from shading.",
      tryThis: ["Switch Tracing to Inline: the same rays traced inside one kernel give the same image; compare the frame time", "Raise the sun radius for soft, sampled shadows"],
      compare: "Tracing: Inline (the rays traced inside one kernel with rtClosest / rtVisible)",
    },
    category: 'raytracer',
    status: 'stable',
    tags: ['RtBatchTracer', 'shadow rays', 'GPU buffers'],
    docs: 'raytracer/batch-tracer',
    source: 'examples/raytracer/batch',
    network: true,
  },
  {
    slug: 'intervals',
    path: 'raytracer/intervals/',
    title: 'Interval queries',
    summary: 'rtClosestInterval splits each ray into radiance-cascade intervals with no gap and no overlap.',
    description:
      'Cascade n owns the distances (t0 · growth^(n-1), t0 · growth^n]. The final view lights each surface by the sky it can see, tracing every sky ray interval by interval with `rtClosestInterval`; the reference traces the same rays whole with `rtClosest`, and the two match exactly. ' +
      'The broken view shortens each interval at both ends, the classic off-by-epsilon bug exaggerated: whatever lies in a gap is missed and light leaks in.',
    brief: {
      kind: "feature",
      shows: "Interval queries split a ray into consecutive distance ranges with no gap and no overlap, the building block of radiance cascades. The final image is sky light gathered cascade by cascade, and it matches one full-length ray exactly.",
      tryThis: ["Compare Final with Reference (one full-length ray): the same image, pixel for pixel", "Switch Show to Broken: naive intervals with gaps let the sky leak onto the back wall and into the corners", "Switch to the Pieces views: which cascade each ray hit in, or one cascade alone"],
      compare: "Naive intervals with a small gap between cascades, and one full-length ray as the reference",
    },
    category: 'raytracer',
    status: 'experimental',
    tags: ['rtClosestInterval', 'radiance cascades', 'inline WGSL'],
    docs: 'raytracer/interval-queries',
    source: 'examples/raytracer/intervals',
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
    brief: {
      kind: "performance",
      shows: "The same path tracer compiled two ways: one megakernel, or a wavefront of queues with separate intersect and shade passes. The images match per pixel; which is faster depends on the scene and the GPU.",
      tryThis: ["Switch Scheduler and compare the frame time", "On the wavefront, try refill and sorting on a bigger model"],
      compare: "Scheduler: Megakernel against Wavefront",
    },
    category: 'pathtracer',
    status: 'stable',
    tags: ['PtMegakernel', 'PtWavefront', 'ptwfAutoPolicy'],
    docs: 'concepts/schedulers',
    source: 'examples/pathtracer/schedulers',
    network: true,
  },
  {
    slug: 'materials',
    path: 'pathtracer/materials/',
    title: 'Material explorer',
    summary: 'Every OpenPBR lobe of one material on a sphere: base, specular, transmission, coat, fuzz, thin film, emission.',
    description:
      'Start from a preset (gold, glass, car paint, velvet, a soap bubble) and change any parameter. The kernel is compiled once with every lobe switched on, ' +
      'so a slider only rewrites the scene copy (`buildSceneCopy` + `writeSceneCopy`) and the image restarts at once. The studio environment is an HDR map generated on the page.',
    brief: {
      kind: "feature",
      shows: "Every OpenPBR lobe of one material, live: edit a parameter and the image restarts at once, because the kernel is compiled once and only the scene copy changes.",
      tryThis: ["Pick a preset: gold, glass, car paint, velvet, a soap bubble", "Raise the coat or fuzz weight", "Set specular anisotropy on Brushed aluminium"],
    },
    category: 'pathtracer',
    status: 'stable',
    tags: ['openpbr', 'coat', 'fuzz', 'thin film', 'transmission', 'env map'],
    docs: 'concepts/materials',
    source: 'examples/pathtracer/materials',
  },
  {
    slug: 'bvh',
    path: 'pathtracer/bvh/',
    title: 'BVH formats',
    summary: 'One scene in four @pmndrs/bvh node formats: build time, memory and a 64 spp benchmark per format on your GPU.',
    description:
      '`buildGpuScene(device, input, { nodeFormat })` builds a quantized or f32 binary tree, or a 4-wide or compressed 8-wide one (both experimental); the kernel compiles the matching walk. ' +
      'The images agree; build time, memory and speed differ by GPU. Benchmark each format here and compare them on this machine.',
    brief: {
      kind: "performance",
      shows: "One scene built into four BVH node formats, with build time, GPU memory and path tracing speed measured on your GPU.",
      tryThis: ["Switch Node format and read the build time and memory", "Press Benchmark for each format and compare the ms per sample"],
      compare: "Node format: Quantized (the default) against f32, BVH4 and CWBVH8",
    },
    category: 'pathtracer',
    status: 'stable',
    tags: ['bvh', 'bvh4', 'cwbvh8', 'benchmark'],
    docs: 'concepts/scenes',
    source: 'examples/pathtracer/bvh',
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
    brief: {
      kind: "quality",
      shows: "Leaves and fences made of textured cards with alpha cutouts. The exact shadow rays test the texture at every card they cross; the fast policies skip that work and get the shadows slightly wrong.",
      tryThis: ["Switch Shadow rays between Exact and the fast policies and look at the shadows on the ground", "Move the sun to stretch the shadows"],
      compare: "Shadow rays: Opaque shadows (fast, biased) against Exact",
    },
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
    brief: {
      kind: "quality",
      shows: "With many lights, which light each shadow ray goes to decides the noise. The light-selection strategy changes the noise at the same sample count; ReSTIR DI goes further by reusing good picks.",
      tryThis: ["Set Samples › stop at to 16, then change Light selection and compare the noise", "Turn on ReSTIR DI"],
      compare: "Light selection: power or kind against pbrt, and ReSTIR DI off against on",
    },
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
    brief: {
      kind: "feature",
      shows: "Participating media: light scattering inside fog and a cloud, with soft beams and shadows through the volume.",
      tryThis: ["Switch between Homogeneous fog and Cloud", "Raise the extinction for thicker fog, or the anisotropy g for forward-scattering haze"],
    },
    category: 'pathtracer',
    status: 'experimental',
    tags: ['volumes', 'media', 'density grid'],
    docs: 'pathtracer/volumes',
    source: 'examples/pathtracer/volumes',
    heavy: true,
  },
  {
    slug: 'cloud',
    path: 'pathtracer/cloud/',
    title: 'Cloud',
    summary: 'A cumulus cloud path traced from a 128\u00b3 density grid made on the page: a bright forward-scattering medium under a sun and a sky.',
    description:
      'A closed box with a medium-boundary material bounds a procedural density grid (soft blobs times fractal noise). Rays are tracked through it by delta tracking against 8\u00b3 brick majorants. ' +
      'The medium is grey with albedo 0.99, so light scatters many times (the core optical depth is about 40; 128 scattering events per path are plenty), and the phase is strongly forward: ' +
      'one Henyey-Greenstein lobe, or the water-droplet Mie fit of `ptPhaseLobes`. Move the sun, thicken the cloud, change the droplet size. Extinction, albedo and phase are compiled into the kernel, so a change rebuilds it.',
    brief: {
      kind: 'feature',
      shows: 'A dense, bright cloud from a density grid: light scatters dozens of times inside it, and the forward-peaked droplet phase makes the silver lining when the sun is behind it.',
      tryThis: ['Lower the sun elevation behind the cloud to see the silver lining', 'Switch the phase between Mie droplets and Henyey-Greenstein', 'Lower the density scale to see the cloud thin out'],
    },
    category: 'pathtracer',
    status: 'experimental',
    tags: ['volumes', 'cloud', 'density grid', 'Mie phase', 'brick majorant'],
    docs: 'pathtracer/volumes',
    source: 'examples/pathtracer/cloud',
    heavy: true,
    check: { query: 'scale=0.25&res=64', samples: 2 },
  },
  {
    slug: 'fog',
    path: 'pathtracer/fog/',
    title: 'Fog and light shafts',
    summary: 'Sunlit thin fog in a room with slits in its wall: light shafts that forced in-scatter and equiangular sampling turn from noise into a smooth image.',
    description:
      'A closed box with a medium-boundary material fills a dark room with thin homogeneous fog (optical depth about 0.4); a low sun shines through narrow slits. The view is split, each half with its own estimator at the same sample count. In thin fog most rays never collide, so the plain estimator, which samples light only at real collisions, leaves the shafts very noisy. ' +
      '`forcedNee` adds one light sample per free-flight segment at a distance drawn on the segment; `equiangular` draws that distance around the lamp, so a lamp inside the fog no longer gives 1/r\u00b2 noise. All three estimators converge to the same image. The estimator is compiled into the kernel, so a change rebuilds it.',
    brief: {
      kind: 'quality',
      shows: 'Light shafts in thin fog: the plain estimator leaves them as noise, forced in-scatter NEE (and equiangular sampling for a lamp) makes them smooth at the same sample count.',
      tryThis: ['Compare the two halves of the split view at a few samples', 'Turn the lamp on and set one half to equiangular'],
      compare: 'Comparison: Estimators, left half plain (light only at real collisions)',
    },
    category: 'pathtracer',
    status: 'experimental',
    tags: ['volumes', 'fog', 'god rays', 'forced NEE', 'equiangular sampling'],
    docs: 'pathtracer/volumes',
    source: 'examples/pathtracer/fog',
    heavy: true,
    check: { query: 'scale=0.25', samples: 2 },
  },
  {
    slug: 'spectral',
    path: 'pathtracer/spectral/',
    title: 'Spectral rendering and dispersion',
    summary: 'Paths that carry wavelengths: glass whose index of refraction depends on the wavelength splits white bars into colour fringes.',
    description:
      'With `integrator.spectral` every path carries three wavelengths: colours are lifted to spectra and converted back at the end. A dispersive glass prism and slab stand in front of bright bars, ' +
      'so the split shows as colour fringes in the refracted view. Turn spectral off to see the same scene in RGB, where dispersion is ignored.',
    brief: {
      kind: "quality",
      shows: "Spectral rendering: paths carry wavelengths of light instead of RGB, so a dispersive glass splits white light into colours. RGB rendering cannot.",
      tryThis: ["Turn spectral off: the prism's colours disappear", "Raise the dispersion"],
      compare: "Spectral: off (RGB transport)",
    },
    category: 'pathtracer',
    status: 'experimental',
    tags: ['spectral', 'dispersion', 'glass'],
    docs: 'pathtracer/spectral',
    source: 'examples/pathtracer/spectral',
  },
  {
    slug: 'materialx',
    path: 'pathtracer/materialx/',
    title: 'MaterialX',
    summary: 'MaterialX documents as path tracer materials: graphs compiled to WGSL per hit, or baked into textures.',
    description:
      '`buildMaterialX` reads a `.mtlx` document. Codegen turns its graph into WGSL that runs at every hit, so anything it reads (uv, position, images) is exact; ' +
      'bake renders the uv-driven inputs into textures and needs no kernel change, but a position-driven graph only averages. Compare both on a texture graph and on procedural stripes.',
    brief: {
      kind: "feature",
      shows: "MaterialX documents as path tracer materials, compiled into the kernel (exact per hit) or baked into textures (no kernel change).",
      tryThis: ["Pick the procedural stripes document and switch Approach between codegen and bake: baking cannot follow a position-driven pattern", "Pick the texture graph"],
      compare: "Approach: Bake (textures over uv) against Codegen (WGSL per hit)",
    },
    category: 'pathtracer',
    status: 'experimental',
    tags: ['MaterialX', 'codegen', 'bake'],
    docs: 'pathtracer/materialx',
    source: 'examples/pathtracer/materialx',
  },
  {
    slug: 'realtime-frame',
    path: 'realtime/frame/',
    title: 'Real-time frames',
    summary: 'One sample per pixel per frame with the planes a denoiser takes: demodulated diffuse and specular, albedo, normals, depth, motion.',
    description:
      '`mode: \'realtime\'` renders one blue-noise sample per pixel per frame at two bounces. Step through the frame planes, check that diffuse x albedo + specular x factor + emission ' +
      'gives back the beauty, average the frames while the camera is still, and compare blue noise with Owen-Sobol on a single frame.',
    brief: {
      kind: "educational",
      shows: "What one real-time frame contains: one sample per pixel, plus the planes a denoiser and an upscaler read (lighting split from surface colour, normals, depth, motion).",
      tryThis: ["Step through the views", "Turn Average while still off to see the single noisy frame a denoiser starts from", "Switch the sampler to compare blue noise with Owen-Sobol on one frame"],
      compare: "Sampler: Owen-Sobol or PCG against Blue noise (one frame each)",
    },
    category: 'realtime',
    status: 'stable',
    tags: ['realtime', 'frame planes', 'blue noise', 'motion vectors'],
    docs: 'pathtracer/realtime',
    source: 'examples/realtime/frame',
    network: true,
    check: { samples: 8 },
  },
  {
    slug: 'realtime-chain',
    path: 'realtime/chain/',
    title: 'Denoised real-time chain',
    summary: 'One path-traced sample per pixel per frame at half resolution, through a denoiser and a temporal upscaler: the raw frame beside the final image.',
    description:
      'The path tracer renders one sample per pixel per frame at a reduced size with its frame planes. The planes are packed into textures, filtered by the SVGF denoiser of `denoiser/realtime` ' +
      '(diffuse and specular lighting demodulated from the surface colour, so texture detail is not blurred, then remodulated) and brought to display size by the temporal path of `@pmndrs/upscaler`. ' +
      'The split view shows the raw frame the path tracer gives on the left and the final image on the right.',
    brief: {
      kind: 'quality',
      shows: 'What a denoiser and an upscaler make of one path-traced sample per pixel per frame: the raw frame is mostly noise, the final image is usable in real time. The chain is a library recipe, not raster.',
      tryThis: ['Orbit the camera: the denoiser follows the motion vectors, so the image stays stable instead of restarting', 'Turn the denoiser off to see what the upscaler gets without it', 'Switch the upscaler to Spatial or Bilinear to see what accumulating frames adds'],
      compare: 'The left half of the split view: the raw 1 spp frame, bilinear-stretched to display size; or the denoiser off, or the bilinear upscaler',
    },
    category: 'realtime',
    status: 'experimental',
    tags: ['realtime', 'denoiser', 'SVGF', 'upscaler', 'frame planes'],
    docs: 'pathtracer/realtime',
    source: 'examples/realtime/chain',
    check: { samples: 8 },
  },
  {
    slug: 'realtime-restir',
    path: 'realtime/restir/',
    title: 'ReSTIR DI in real time',
    summary: 'Plain light sampling and ReSTIR DI side by side at one sample per pixel over 64 lamps, with the error of each single frame measured live.',
    description:
      'Both halves render the same frame with one sample per pixel. Plain next-event estimation picks one lamp per pixel at random; ReSTIR DI keeps the best of several candidates and reuses picks from neighbours and the previous frame. ' +
      'While the camera is still, the page averages plain NEE frames into a reference and reads each method\'s single-frame error against it (about 15x lower with ReSTIR). Average while still shows that both converge to the same image.',
    brief: {
      kind: "quality",
      shows: "At the same one sample per pixel, ReSTIR DI gives a far less noisy frame than plain light sampling with many lights, and both converge to the same image: less noise, not a different answer. That is what makes many-light real time feasible behind a denoiser.",
      tryThis: ["Hold the camera still for a few seconds and read the error per frame for each half", "Turn on Average while still: both halves converge to the same picture", "Turn temporal or spatial reuse off to see what each part of ReSTIR adds"],
      compare: "The left half: plain NEE (one light per pixel), same frame, same sample count",
    },
    category: 'realtime',
    status: 'experimental',
    tags: ['ReSTIR DI', 'realtime', 'many lights'],
    docs: 'pathtracer/restir',
    source: 'examples/realtime/restir',
    check: { samples: 8 },
  },
  {
    slug: 'convergence',
    path: 'lab/convergence/',
    title: 'Convergence',
    summary: 'Error against a reference as samples accumulate, log-log: Owen-Sobol against independent random numbers, and the 1/spp line.',
    description:
      'The page renders a reference, then renders the same view again from scratch and measures the relMSE against the reference at 1, 2, 4, ... samples per pixel. ' +
      'Monte Carlo error falls as 1/spp; Owen-scrambled Sobol points are stratified and fall faster at low counts. Near the reference\'s own sample count the curve flattens: its noise is the floor.',
    brief: {
      kind: "educational",
      shows: "How fast a path traced image approaches the right answer: the error against a reference at 1, 2, 4 … samples, on log-log axes. Monte Carlo error falls as 1/spp; stratified Owen-Sobol samples fall faster at low counts.",
      tryThis: ["Run it and compare the two curves with the dashed 1/spp line", "Switch to 64 lamps, where the lights make the noise"],
      compare: "PCG (independent random numbers) against Owen-Sobol, with the 1/spp guide",
    },
    category: 'lab',
    status: 'stable',
    tags: ['relMSE', 'sampling', 'owen-sobol', 'pcg'],
    docs: 'concepts/sampling',
    source: 'examples/lab/convergence',
    check: { query: 'ref=16', samples: 4 },
  },
  {
    slug: 'caps',
    path: 'lab/caps/',
    title: 'GPU report',
    summary: 'What this browser and GPU offer the libraries: tier, optional features, limits, and which examples run here.',
    description:
      '`detectCaps(adapter)` as the libraries see it: the adapter, the optional WebGPU features and what each is used for, the device limits against the adapter\'s maximum, ' +
      'and which examples this machine can run. Copy the report into a bug report.',
    brief: {
      kind: "informative",
      shows: "What this browser and GPU offer the libraries: the capability tier, optional features and what each is used for, the device limits, and which examples run here.",
      tryThis: ["Copy the report into a bug report"],
    },
    category: 'lab',
    status: 'stable',
    tags: ['caps', 'webgpu features', 'limits'],
    docs: 'raytracer/capabilities',
    source: 'examples/lab/caps',
    check: { image: false },
  },
];

export const exampleBySlug = (slug: string): ExampleEntry | undefined => EXAMPLES.find((e) => e.slug === slug);
