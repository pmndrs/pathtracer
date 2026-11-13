/**
 * Run-report manifest for the A/B viewer (`?report=<url of report.json>`; schema in the header of
 * bench/viewer/src/report.ts). The runner writes one next to the uploaded images for every invocation that produced a cell
 * (an uploaded image, or a skipped scene): `bench/results/incoming/<run-id>/report.json`, with every path relative to that file.
 *
 * Grid: scene (rows) x renderer (columns). A run's scene and renderer come from its `BenchResult`
 * (`scene`, `renderer`; a page without them shows up as scene "-" / renderer = page name). The cell image is the
 * upload named by `result.notes.image`, else the first upload of the run. Cells get `spp`, `wallMs`, `msPerSpp` from
 * the result's metrics (`spp`, `wall_ms`, `ms_per_spp`) and, when the scene has a cached reference of the same size,
 * `relMse`, `relBias`, `flip` and `psnr` against it. A run whose result says `skipped` becomes a greyed cell.
 * References: bench/references/<scene>/<renderer>-<spp>spp.pfm; Mitsuba `llvm_ad_rgb` (D15's headline reference) is
 * preferred, then pbrt-v4, then any other .pfm there; the highest spp wins.
 */
import { readdirSync, readFileSync, existsSync } from 'node:fs';
import { join, posix, relative, sep } from 'node:path';
import type { Image } from '../src/adapter.ts';
import { flipLdr } from '../metrics/flip.ts';
import { compareImages } from '../metrics/metrics.ts';
import { readPFM } from '../metrics/pfm.ts';
import type { ReportCell, ReportManifest } from '../viewer/src/report.ts';
import type { RunRecord, UploadRecord } from './contract.ts';

/** Upload slot of the `index`-th run (1-based) of a variant: where `server.setSlot` puts its images. */
export const slotFor = (index: number, variant: string): string => `${index}-${variant === '-' ? 'run' : variant}`;

const REFERENCE_ORDER = ['mitsuba3-llvm_ad_rgb', 'pbrt-v4-cpu'];

/** Picks the reference PFM for a scene among file names (no directories), or null. Highest spp of the preferred renderer. */
export function pickReference(fileNames: readonly string[]): string | null {
  const pfms = fileNames.filter((name) => name.endsWith('.pfm'));
  const spp = (name: string): number => Number(/-(\d+)spp\.pfm$/.exec(name)?.[1] ?? 0);
  for (const prefix of REFERENCE_ORDER) {
    const matches = pfms.filter((name) => name.startsWith(`${prefix}-`)).sort((a, b) => spp(b) - spp(a));
    if (matches[0]) return matches[0];
  }
  return [...pfms].sort((a, b) => spp(b) - spp(a))[0] ?? null;
}

export interface ReportIo {
  /** File names inside bench/references/<scene>/, or [] when there is no such directory. */
  listReferences(scene: string): string[];
  /** Reads a PFM; paths are repo-relative. */
  readImage(repoPath: string): Image;
}

export interface ReportInput {
  runId: string;
  page: string;
  runs: readonly RunRecord[];
  uploads: readonly UploadRecord[];
  /** Directory of the manifest, repo-relative (`bench/results/incoming/<run-id>`). */
  manifestDir: string;
  /** References directory, repo-relative (`bench/references`). */
  referencesDir: string;
  /** Skip FLIP (it takes about a second per megapixel). */
  flip?: boolean;
}

const toUrl = (path: string): string => path.split(sep).join('/');
const relUrl = (fromDir: string, repoPath: string): string => toUrl(relative(fromDir, repoPath));

function finite(value: number): number | undefined {
  return Number.isFinite(value) ? value : undefined;
}

/** Builds the manifest, or null when no run produced a cell (no upload and no skipped scene). Never throws on a bad cell: it drops the metrics. */
export function buildReportManifest(input: ReportInput, io: ReportIo): ReportManifest | null {
  const scenes = new Map<string, { id: string; name: string; reference?: string }>();
  const renderers = new Map<string, { id: string; name: string }>();
  const cells: ReportCell[] = [];
  const seen = new Set<string>();
  const imageCache = new Map<string, Image | null>();
  const load = (repoPath: string): Image | null => {
    if (!imageCache.has(repoPath)) {
      try {
        imageCache.set(repoPath, io.readImage(repoPath));
      } catch {
        imageCache.set(repoPath, null);
      }
    }
    return imageCache.get(repoPath) ?? null;
  };

  for (const run of input.runs) {
    if (run.outcome !== 'pass' || !run.result) continue;
    const result = run.result;
    const slot = slotFor(run.index, run.variant);
    const mine = input.uploads.filter((upload) => upload.slot === slot);
    const scene = result.scene ?? '-';
    let renderer = result.renderer ?? input.page;
    const named = typeof result.notes?.image === 'string' ? mine.find((upload) => upload.name === result.notes?.image) : undefined;
    const primary = named ?? mine[0];
    const skippedReason = typeof result.notes?.skipped === 'string' ? result.notes.skipped : result.metrics.skipped ? 'skipped' : undefined;
    if (!primary && !skippedReason) continue;

    if (seen.has(`${scene}\u0000${renderer}`)) {
      // A repeat of the same cell keeps the first; an A/B variant with another config becomes its own column.
      if (run.variant === '-') continue;
      renderer = `${renderer} (${run.variant})`;
      if (seen.has(`${scene}\u0000${renderer}`)) continue;
    }
    seen.add(`${scene}\u0000${renderer}`);
    if (!renderers.has(renderer)) {
      const base = result.renderer ?? input.page;
      renderers.set(renderer, { id: renderer, name: renderer === base ? base : renderer });
    }

    if (!scenes.has(scene)) {
      const refName = pickReference(io.listReferences(scene));
      const refPath = refName ? posix.join(toUrl(input.referencesDir), scene, refName) : undefined;
      scenes.set(scene, { id: scene, name: scene, ...(refPath ? { reference: relUrl(input.manifestDir, refPath) } : {}) });
    }
    const sceneEntry = scenes.get(scene)!;
    const refRepoPath = sceneEntry.reference ? posix.normalize(posix.join(toUrl(input.manifestDir), sceneEntry.reference)) : undefined;

    const metrics: Record<string, number> = {};
    const m = result.metrics;
    if (typeof m.spp === 'number') metrics.spp = m.spp;
    if (typeof m.wall_ms === 'number') metrics.wallMs = m.wall_ms;
    if (typeof m.ms_per_spp === 'number') metrics.msPerSpp = m.ms_per_spp;

    if (skippedReason) {
      cells.push({ scene, renderer, image: sceneEntry.reference ?? '', skipped: skippedReason });
      continue;
    }
    const image = relUrl(input.manifestDir, primary!.pfm);
    if (refRepoPath) {
      const test = load(primary!.pfm);
      const ref = load(refRepoPath);
      if (test && ref && test.width === ref.width && test.height === ref.height) {
        const pair = compareImages(test, ref);
        for (const [key, value] of Object.entries({ relMse: pair.relMse, relBias: pair.relBias, psnr: pair.psnr })) {
          const v = finite(value);
          if (v !== undefined) metrics[key] = v;
        }
        if (input.flip !== false) {
          try {
            metrics.flip = flipLdr(test, ref).mean;
          } catch {
            // FLIP is an extra; the cell is still useful without it.
          }
        }
      }
    }
    cells.push({ scene, renderer, image, metrics });
  }
  if (cells.length === 0) return null;
  return { run: input.runId, scenes: [...scenes.values()], renderers: [...renderers.values()], cells };
}

/** The real filesystem, for run.ts. */
export function fileSystemIo(repoRoot: string, referencesDir: string): ReportIo {
  return {
    listReferences: (scene) => {
      const dir = join(repoRoot, referencesDir, scene);
      return existsSync(dir) ? readdirSync(dir) : [];
    },
    readImage: (repoPath) => readPFM(readFileSync(join(repoRoot, repoPath))),
  };
}
