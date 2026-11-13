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
}

export type Exporter = (scene: SceneDesc, opts: ExportOptions) => ExportResult;
