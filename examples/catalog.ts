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
];

export const exampleBySlug = (slug: string): ExampleEntry | undefined => EXAMPLES.find((e) => e.slug === slug);
