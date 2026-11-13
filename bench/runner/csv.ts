/**
 * Result CSV: bench/results/<suite>.csv, appended to and never rewritten. Columns follow
 * ~/Developer/webgpu-ddgi/bench/results (date, commit, device, browser, scene, metric, value) plus what an A/B
 * bench needs: renderer, config_hash, valid, run, note.
 */

import { createHash } from 'node:crypto';
import { appendFileSync, existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';

export const CSV_COLUMNS = ['date', 'commit', 'device', 'browser', 'scene', 'renderer', 'config_hash', 'metric', 'value', 'valid', 'run', 'note'] as const;

export interface CsvRow {
  /** UTC date, YYYY-MM-DD. */
  date: string;
  /** Short git hash, `-dirty` when a measured file is modified. */
  commit: string;
  device: string;
  browser: string;
  scene: string;
  renderer: string;
  config_hash: string;
  /** Unit in the name: `gpu_ms_total`. */
  metric: string;
  value: number | string;
  /** 1 when the run counts (valid timing verdict, page says valid), else 0. */
  valid: 0 | 1;
  /** `1`..`N` for single runs, `median` for the median of them. */
  run: string;
  /** Short free text (verdict reason, variant). */
  note: string;
}

/** Quotes a field only when it needs it (comma, quote, newline). */
export function csvField(value: string | number): string {
  const text = typeof value === 'number' ? (Number.isFinite(value) ? String(value) : '') : value;
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

export function csvLine(row: CsvRow): string {
  return CSV_COLUMNS.map((column) => csvField(row[column])).join(',');
}

export const CSV_HEADER = CSV_COLUMNS.join(',');

/** The text to append: the header only when the file is new, then one line per row. */
export function csvText(rows: readonly CsvRow[], writeHeader: boolean): string {
  const lines = rows.map(csvLine);
  return (writeHeader ? [CSV_HEADER, ...lines] : lines).map((line) => `${line}\n`).join('');
}

/** Appends rows to `path`, creating it (with header) when missing. Returns the number of rows written. */
export function appendCsv(path: string, rows: readonly CsvRow[]): number {
  if (rows.length === 0) return 0;
  mkdirSync(dirname(path), { recursive: true });
  const fresh = !existsSync(path);
  if (fresh) writeFileSync(path, '');
  appendFileSync(path, csvText(rows, fresh));
  return rows.length;
}

/** JSON with sorted keys, so equal configs hash equal whatever the key order. */
export function canonicalJson(value: unknown): string {
  if (value === null || typeof value !== 'object') return JSON.stringify(value) ?? 'null';
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
  const entries = Object.entries(value as Record<string, unknown>)
    .filter(([, item]) => item !== undefined)
    .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
  return `{${entries.map(([key, item]) => `${JSON.stringify(key)}:${canonicalJson(item)}`).join(',')}}`;
}

/** 8 hex characters of SHA-1 over the canonical JSON of what defines a configuration. */
export function configHash(config: unknown): string {
  return createHash('sha1').update(canonicalJson(config)).digest('hex').slice(0, 8);
}
