import type { SceneDesc } from '../scenes/schema.ts';

/** A file produced by an exporter; `path` is relative to the export directory. */
export interface ExportedFile {
  path: string;
  data: Buffer | string;
}

export interface ExportResult {
  /** Entry point relative to the export directory (.pbrt, .xml, .glb). */
  sceneFile: string;
  files: ExportedFile[];
  /** Where each tool should write its image, when the exporter decides (informational). */
  /** Things the target format could not express exactly, with the mapping used. Empty = exact. */
  approximations: string[];
}

export interface ExportOptions {
  /** Directory holding assets referenced by `hdri`/`file` (default: bench/assets). */
  assetsDir: string;
  /** Samples per pixel written into the scene file. */
  spp: number;
  /**
   * `openpbr` materials only: write the closest approximation, flagged `coverage:` in `approximations`, where no native renderer
   * expresses the material (reference class C, Mitsuba-only gaps); without it the exporter throws.
   */
  coverage?: boolean;
  /** pbrt `coateddiffuse` / `coatedconductor` `maxdepth` and `nsamples` (default `LAYERED_DEFAULTS` in openpbr.ts). */
  layered?: { maxDepth?: number; nSamples?: number };
}

export type Exporter = (scene: SceneDesc, opts: ExportOptions) => ExportResult;
