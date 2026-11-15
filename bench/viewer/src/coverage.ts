/**
 * Coverage tab: ?coverage=<url of coverage.json> (default /results/reports/tier1/coverage.json), written by bench/adapters/tier1-matrix.ts (P2.31).
 * One row per Tier 1 feature, one column per renderer. A cell shows its symbol and the ablation ratio; click it to open the comparison behind it
 * (full vs feature-stripped render of that renderer, the reference as R), hover for the numbers and notes.
 */
import type { ReportPick } from './report.ts';

interface CovCell {
  status: string; symbol: string; notes: string[]; reportRef: string | null;
  relMSE_pbrt?: number; relMSE_mitsuba?: number; floor?: number; ablationRelMSE?: number; noise?: number; ratio?: number;
}
interface CovRow { id: string; label: string; extension: string; refClass: string; scene: string | null; stripped: string | null; noScene: boolean; note?: string; cells: Record<string, CovCell> }
interface CovJson { generated: string; report: string; spp: number; halfSpp: number; thresholds: { sensitivity: number; agreeA: number; agreeB: number }; rows: CovRow[] }

export const COVERAGE_URL = '/results/reports/tier1/coverage.json';
const COLUMNS: [string, string][] = [['ours', 'ours'], ['tgpt', 'three-gpu-pathtracer'], ['mutracer', 'mutracer'], ['pbrt', 'pbrt-v4'], ['mitsuba', 'Mitsuba 3']];
const sci = (v: number | undefined): string => (v === undefined ? '-' : v.toExponential(1));

export function parseRef(ref: string): ReportPick {
  const q = new URLSearchParams(ref);
  return { a: q.get('a')!, b: q.get('b')!, ...(q.get('r') ? { r: q.get('r')! } : {}) };
}

export async function showCoverage(root: HTMLElement, url: string, onPick: (p: ReportPick) => void): Promise<void> {
  root.innerHTML = '<p class="muted">loading coverage...</p>';
  const j = (await (await fetch(url)).json()) as CovJson;
  root.innerHTML = '';
  const bar = document.createElement('div');
  bar.className = 'report-bar';
  bar.innerHTML = `<h2>Tier 1 coverage (${j.generated})</h2><span class="muted">report ${j.report}, ${j.spp} spp. ● supported  ◐ approximate / off reference  ○ missing  – no scene yet. Number: ablation relMSE / own noise (>= ${j.thresholds.sensitivity} to count). Click a cell for its comparison (A full, B stripped, R reference); pbrt / Mitsuba cells compare the two references.</span>`;
  root.append(bar);
  const table = document.createElement('table');
  table.className = 'report-grid coverage';
  const head = table.createTHead().insertRow();
  head.insertCell().textContent = 'feature';
  head.insertCell().textContent = 'class';
  for (const [, title] of COLUMNS) head.insertCell().textContent = title;
  const body = table.createTBody();
  for (const r of j.rows) {
    const tr = body.insertRow();
    const label = tr.insertCell();
    label.textContent = r.label + (r.noScene ? ' (no scene yet)' : '');
    label.title = [r.extension, r.scene ? `scene ${r.scene}, stripped ${r.stripped}` : 'no scene yet', r.note ?? ''].filter(Boolean).join('\n');
    tr.insertCell().textContent = r.refClass;
    for (const [id] of COLUMNS) {
      const c = r.cells[id];
      const td = tr.insertCell();
      if (!c) { td.textContent = '-'; continue; }
      td.className = `ccell ${c.status}`;
      td.textContent = `${c.symbol}${c.ratio !== undefined && Number.isFinite(c.ratio) ? ` ${c.ratio.toFixed(0)}x` : ''}`;
      td.title = [
        `${c.status}`,
        c.ablationRelMSE !== undefined ? `ablation relMSE ${sci(c.ablationRelMSE)}, noise ${sci(c.noise)}` : '',
        c.relMSE_pbrt !== undefined ? `vs pbrt ${sci(c.relMSE_pbrt)}, vs Mitsuba ${sci(c.relMSE_mitsuba)}, floor ${sci(c.floor)}` : '',
        ...c.notes,
      ].filter(Boolean).join('\n');
      if (c.reportRef) {
        td.classList.add('link');
        td.onclick = () => onPick(parseRef(c.reportRef!));
      }
    }
  }
  root.append(table);
}
