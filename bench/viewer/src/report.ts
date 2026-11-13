/**
 * Run-report mode: ?report=<url of report.json>. Shows a grid of scene (rows) x renderer (columns) thumbnails with
 * one metric per cell; click one cell then another to A/B them (the scene's reference, if any, becomes R).
 *
 * Manifest (all paths relative to the manifest's own URL):
 * {
 *   "run": "2026-11-13-baseline",            // optional title
 *   "scenes":    [{ "id": "furnace", "name": "Furnace", "reference": "refs/furnace.pfm" }],
 *   "renderers": [{ "id": "ours-mega", "name": "ours (megakernel)" }],
 *   "cells": [{ "scene": "furnace", "renderer": "ours-mega", "image": "ours-mega/furnace.pfm",
 *               "thumb": "ours-mega/furnace.png",            // optional; else the PFM is fetched and downsampled
 *               "reference": "refs/furnace.pfm",             // optional per-cell override
 *               "metrics": { "relMse": 0.012, "flip": 0.08, "wallMs": 1234, "spp": 256 },
 *               "skipped": "feature unsupported" }]          // optional: cell is shown greyed out
 * }
 */
import { readPFM } from '../../metrics/pfm.ts';
import { tonemapImage, srgbEncode } from '../../metrics/tonemap.ts';

export interface ReportCell {
  scene: string; renderer: string; image: string; thumb?: string; reference?: string;
  metrics?: Record<string, number>; skipped?: string;
}
export interface ReportManifest {
  run?: string;
  scenes: Array<{ id: string; name?: string; reference?: string }>;
  renderers: Array<{ id: string; name?: string }>;
  cells: ReportCell[];
}
export interface ReportPick { a: string; b: string; r?: string }

const abs = (base: string, rel: string | undefined) => {
  if (!rel) return undefined;
  const u = new URL(rel, new URL(base, location.href));
  return u.pathname + u.search;
};

async function thumbFromPFM(url: string, maxW = 240): Promise<HTMLCanvasElement> {
  const img = readPFM(await (await fetch(url)).arrayBuffer());
  const f = Math.max(1, Math.floor(img.width / maxW));
  const w = Math.floor(img.width / f), h = Math.floor(img.height / f);
  const small = new Float32Array(w * h * 3);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      for (let c = 0; c < 3; c++) {
        let s = 0, n = 0;
        for (let yy = 0; yy < f; yy++) for (let xx = 0; xx < f; xx++) {
          const v = img.data[((y * f + yy) * img.width + (x * f + xx)) * img.channels + c]!;
          if (Number.isFinite(v)) { s += v; n++; }
        }
        small[(y * w + x) * 3 + c] = n ? s / n : 0;
      }
    }
  }
  const lin = tonemapImage({ width: w, height: h, channels: 3, data: small }, 'aces', 0);
  const canvas = document.createElement('canvas');
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext('2d')!;
  const out = ctx.createImageData(w, h);
  for (let i = 0; i < w * h; i++) {
    for (let c = 0; c < 3; c++) out.data[i * 4 + c] = Math.round(srgbEncode(lin[i * 3 + c]!) * 255);
    out.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(out, 0, 0);
  return canvas;
}

let queue: Promise<unknown> = Promise.resolve();
const slots = { n: 0 };
function schedule<T>(fn: () => Promise<T>): Promise<T> {
  // at most 2 concurrent PFM thumbnail loads
  const run = async () => { while (slots.n >= 2) await new Promise((r) => setTimeout(r, 30)); slots.n++; try { return await fn(); } finally { slots.n--; } };
  const p = queue.then(run, run);
  queue = p.catch(() => undefined);
  return p;
}

const fmt = (v: number | undefined) => (v === undefined ? '-' : !Number.isFinite(v) ? String(v) : Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 0.01 ? v.toPrecision(3) : v.toExponential(2));

export async function showReport(root: HTMLElement, manifestUrl: string, onPick: (p: ReportPick) => void): Promise<void> {
  root.innerHTML = '<p class="muted">loading report...</p>';
  const m = (await (await fetch(manifestUrl)).json()) as ReportManifest;
  const keys = [...new Set(m.cells.flatMap((c) => Object.keys(c.metrics ?? {})))];
  let metric = keys.includes('relMse') ? 'relMse' : (keys[0] ?? '');
  let picked: { url: string; ref?: string; el: HTMLElement } | undefined;

  const cellFor = (s: string, r: string) => m.cells.find((c) => c.scene === s && c.renderer === r);
  const sceneRef = (s: string, c?: ReportCell) => abs(manifestUrl, c?.reference ?? m.scenes.find((x) => x.id === s)?.reference);

  const render = () => {
    root.innerHTML = '';
    const bar = document.createElement('div');
    bar.className = 'report-bar';
    bar.innerHTML = `<h2>${m.run ?? 'run report'}</h2><span class="muted">click a cell, then another to A/B them (reference becomes R). Shift-click: cell vs its reference.</span>`;
    if (keys.length) {
      const sel = document.createElement('select');
      for (const k of keys) { const o = document.createElement('option'); o.value = k; o.textContent = k; o.selected = k === metric; sel.append(o); }
      sel.onchange = () => { metric = sel.value; picked = undefined; render(); };
      bar.append(' metric: ', sel);
    }
    root.append(bar);
    const table = document.createElement('table');
    table.className = 'report-grid';
    const head = table.createTHead().insertRow();
    head.insertCell().textContent = 'scene';
    const hasRefs = m.scenes.some((s) => s.reference);
    if (hasRefs) head.insertCell().textContent = 'reference';
    for (const r of m.renderers) head.insertCell().textContent = r.name ?? r.id;
    const body = table.createTBody();
    for (const s of m.scenes) {
      const row = body.insertRow();
      row.insertCell().textContent = s.name ?? s.id;
      const values = m.renderers.map((r) => cellFor(s.id, r.id)?.metrics?.[metric]).filter((v): v is number => v !== undefined && Number.isFinite(v));
      const best = values.length > 1 ? Math.min(...values) : undefined;
      const refUrl = sceneRef(s.id);
      const addCell = (url: string | undefined, thumb: string | undefined, label: string, cls: string, ref?: string, skipped?: string) => {
        const td = row.insertCell();
        td.className = `rcell ${cls}`;
        if (!url) { td.innerHTML = '<span class="muted">-</span>'; return; }
        const holder = document.createElement('div');
        holder.className = 'thumb';
        td.append(holder);
        const lab = document.createElement('div');
        lab.className = 'rlabel';
        lab.textContent = skipped ? `skipped: ${skipped}` : label;
        td.append(lab);
        if (skipped) td.classList.add('skipped');
        const io = new IntersectionObserver((es) => {
          if (!es[0]!.isIntersecting) return;
          io.disconnect();
          if (thumb) { const im = new Image(); im.src = thumb; holder.append(im); return; }
          schedule(() => thumbFromPFM(url)).then((cv) => holder.append(cv), () => { holder.textContent = 'load failed'; });
        });
        io.observe(td);
        td.onclick = (e) => {
          if (e.shiftKey && ref) { onPick({ a: url, b: ref }); return; }
          if (!picked) { picked = { url, ref, el: td }; td.classList.add('picked'); return; }
          if (picked.url === url) { td.classList.remove('picked'); picked = undefined; return; }
          const p = { a: picked.url, b: url, r: picked.ref ?? ref };
          picked = undefined;
          onPick(p);
        };
      };
      if (hasRefs) addCell(refUrl, undefined, 'reference', 'ref');
      for (const r of m.renderers) {
        const c = cellFor(s.id, r.id);
        const v = c?.metrics?.[metric];
        const extras = ['spp', 'wallMs'].filter((k) => k !== metric && c?.metrics?.[k] !== undefined).map((k) => `${k} ${fmt(c!.metrics![k])}`).join('  ');
        addCell(c ? abs(manifestUrl, c.image) : undefined, c?.thumb ? abs(manifestUrl, c.thumb) : undefined,
          `${metric} ${fmt(v)}${extras ? `  ${extras}` : ''}`, best !== undefined && v === best ? 'best' : '', sceneRef(s.id, c), c?.skipped);
      }
    }
    root.append(table);
  };
  render();
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && picked) { picked.el.classList.remove('picked'); picked = undefined; } });
}
