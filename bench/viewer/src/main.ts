/**
 * A/B image viewer. Slots: A, B and an optional reference R (PFM, linear float, row 0 = top).
 * Views (keys 1-5): image, abs diff, relative error, FLIP map, bias. Layouts (S/F/D): swipe, flip (A/B/R keys,
 * space), side by side. With R loaded, the error views show A-vs-R and B-vs-R; without it, A-vs-B.
 * Metrics (MSE, relMSE, SMAPE, PSNR, bias, firefly %, FLIP) run in a worker with the same code as the CLI, at a
 * FIXED ACES / 0 EV display so numbers match `node bench/metrics/compare.ts`. The EV slider and tonemap picker only
 * change what you see.
 */
import { STOPS } from '../../metrics/colormap.ts';
import type { PairMetrics } from '../../metrics/metrics.ts';
import { readPFM } from '../../metrics/pfm.ts';
import type { Image } from '../../src/adapter.ts';
import { GLView, LAYOUT, type ViewState } from './renderer.ts';
import { showReport, type ReportPick } from './report.ts';
import type { FromWorker, ToWorker } from './worker.ts';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const canvas = $<HTMLCanvasElement>('gl');
const stage = $('stage');

interface Loaded { name: string; url?: string; img: Image; gen: number }
const loaded: Array<Loaded | undefined> = [undefined, undefined, undefined];
const SLOT_NAME = ['A', 'B', 'R'];
const PAIRS: Array<[number, number]> = [[0, 1], [0, 2], [1, 2]]; // (test, ref)
const PAIR_LABEL = ['A vs B', 'A vs R', 'B vs R'];
let gen = 0;

const CONTENT_KEYS = ['image', 'absdiff', 'relerr', 'flip', 'bias'] as const;
const LAYOUT_KEYS = ['swipe', 'flip', 'side'] as const;
const TONEMAP_KEYS = ['clamp', 'reinhard', 'aces', 'agx'] as const;

const st = {
  content: 0, layout: 0, toggle: 0, showR: false, abDiff: false, split: 0.5, ev: 0, tonemap: 2,
  cx: 0, cy: 0, zoom: 1, // zoom in CSS px per image px
  fitted: true,
};

let gl: GLView;
try { gl = new GLView(canvas); } catch (e) { document.body.textContent = `Viewer needs WebGL2 with float textures: ${(e as Error).message}`; throw e; }
const dpr = () => window.devicePixelRatio || 1;

// ---- worker + metrics -------------------------------------------------------------------------------------------
const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
const send = (m: ToWorker, transfer: Transferable[] = []) => worker.postMessage(m, transfer);
interface PairState { key: string; metrics?: PairMetrics; flip?: number; flipMap?: Float32Array; error?: string }
const pairs: Array<PairState | undefined> = [undefined, undefined, undefined];
worker.onmessage = (e: MessageEvent<FromWorker>) => {
  const m = e.data;
  const p = pairs.findIndex((x) => x?.key === m.key);
  if (p < 0) return; // stale
  const ps = pairs[p]!;
  if (m.type === 'metrics') ps.metrics = m.metrics;
  else if (m.type === 'flip') { ps.flip = m.mean; ps.flipMap = m.map; gl.setFlipMap(p, m.map, m.width, m.height); }
  else ps.error = m.message;
  renderMetrics();
  requestDraw();
};

function requestMetrics() {
  const order = loaded[2] ? [1, 2, 0] : [0];
  for (const p of order) {
    const [t, r] = PAIRS[p]!;
    const lt = loaded[t], lr = loaded[r];
    if (!lt || !lr) { if (pairs[p]) { pairs[p] = undefined; gl.setFlipMap(p, null); } continue; }
    const key = `${p}:${lt.gen}:${lr.gen}`;
    if (pairs[p]?.key === key) continue;
    gl.setFlipMap(p, null);
    pairs[p] = { key };
    send({ type: 'pair', key, test: t, ref: r });
  }
}

const f = (v: number, d = 4) => (!Number.isFinite(v) ? String(v) : Math.abs(v) < 1e-3 && v !== 0 ? v.toExponential(2) : v.toFixed(d));
let hideMetrics = false;
function renderMetrics() {
  const el = $('metrics');
  const rows = [0, 1, 2].filter((p) => pairs[p]);
  if (hideMetrics || !rows.length || document.body.dataset.screen !== 'viewer') { el.hidden = true; return; }
  el.hidden = false;
  const cols: Array<[string, (m: PairMetrics) => string]> = [
    ['MSE', (m) => f(m.mse)], ['relMSE', (m) => f(m.relMse)], ['SMAPE', (m) => f(m.smape)], ['PSNR dB', (m) => f(m.psnr, 2)],
    ['bias', (m) => f(m.bias)], ['firefly %', (m) => f(m.fireflyPct, 3)],
  ];
  let html = `<table><tr><th>test vs ref</th>${cols.map((c) => `<th>${c[0]}</th>`).join('')}<th>FLIP</th></tr>`;
  for (const p of rows) {
    const s = pairs[p]!;
    html += `<tr><td>${PAIR_LABEL[p]}</td>`;
    if (s.error) html += `<td colspan="7">${s.error}</td></tr>`;
    else {
      html += s.metrics ? cols.map((c) => `<td>${c[1](s.metrics!)}</td>`).join('') : `<td class="pending" colspan="6">computing...</td>`;
      html += `<td class="${s.flip === undefined ? 'pending' : ''}">${s.flip === undefined ? '...' : f(s.flip)}</td></tr>`;
    }
  }
  const nf = rows.reduce((n, p) => n + (pairs[p]!.metrics ? pairs[p]!.metrics!.nonFiniteTest + pairs[p]!.metrics!.nonFiniteRef : 0), 0);
  html += `</table><div class="foot">PSNR and FLIP: ACES, 0 EV, 67 ppd. relMSE eps 1e-2. Second image is the reference.${nf ? ` <span style="color:var(--warn)">${nf} non-finite pixel(s) excluded (magenta in the view).</span>` : ''}</div>`;
  el.innerHTML = html;
}

// ---- slots ------------------------------------------------------------------------------------------------------
function toast(msg: string) {
  const t = $('toast');
  t.textContent = msg; t.hidden = false;
  clearTimeout((toast as unknown as { h?: number }).h);
  (toast as unknown as { h?: number }).h = window.setTimeout(() => { t.hidden = true; }, 5000);
}

function imgSize(): [number, number] | undefined {
  const l = loaded.find(Boolean);
  return l ? [l.img.width, l.img.height] : undefined;
}

function assign(slot: number, name: string, img: Image, url?: string): boolean {
  const sz = imgSize();
  const others = loaded.some((l, i) => l && i !== slot);
  if (sz && others && (sz[0] !== img.width || sz[1] !== img.height)) {
    toast(`${name} is ${img.width}x${img.height} but the loaded images are ${sz[0]}x${sz[1]}; clear them first`);
    return false;
  }
  try { gl.setImage(slot, img); } catch (e) { toast((e as Error).message); return false; }
  const wasEmpty = !loaded.some(Boolean);
  loaded[slot] = { name, url, img, gen: ++gen };
  gl.size = [img.width, img.height];
  send({ type: 'image', slot, width: img.width, height: img.height, data: img.data.slice() });
  document.body.dataset.screen = 'viewer';
  if (wasEmpty || st.fitted) fit();
  refreshUI();
  requestMetrics();
  return true;
}

function clearSlot(slot: number) {
  loaded[slot] = undefined;
  gl.setImage(slot, null);
  send({ type: 'clear', slot });
  for (let p = 0; p < 3; p++) if (PAIRS[p]!.includes(slot)) { pairs[p] = undefined; gl.setFlipMap(p, null); }
  if (!loaded.some(Boolean)) document.body.dataset.screen = location.search.includes('report=') ? 'report' : 'start';
  refreshUI();
  requestMetrics();
}

async function loadUrl(slot: number, url: string) {
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    const img = readPFM(await res.arrayBuffer());
    assign(slot, decodeURIComponent(url.split('?')[0]!.split('/').pop() ?? url), img, url);
  } catch (e) { toast(`${url}: ${(e as Error).message}`); }
}

async function loadFile(slot: number, file: File) {
  try { assign(slot, file.name, readPFM(await file.arrayBuffer())); } catch (e) { toast(`${file.name}: ${(e as Error).message}`); }
}

// ---- view state / derived ---------------------------------------------------------------------------------------
function derived(): { img: [number, number]; pair: [number, number]; effContent: number; two: boolean; labels: [string, string] } {
  const has = loaded.map(Boolean);
  const nm = (i: number) => loaded[i]?.name ?? '';
  const ids = [0, 1, 2].filter((i) => has[i]);
  let pair: [number, number] = [-1, -1];
  if (has[0] && has[1] && has[2] && !st.abDiff) pair = [1, 2];
  else if (has[0] && has[1]) pair = [0, -1];
  else if (has[0] && has[2]) pair = [1, -1];
  else if (has[1] && has[2]) pair = [2, -1];
  let content = st.content;
  if (content !== 0 && pair[0] < 0) content = 0;
  let img: [number, number] = [ids[0] ?? 0, ids[1] ?? -1];
  if (has[0] && has[1]) img = [0, 1];
  if (st.showR && has[2] && content === 0) img = [2, 2];
  const lab = (p: number) => `${SLOT_NAME[PAIRS[p]![0]]} − ${SLOT_NAME[PAIRS[p]![1]]}`;
  const labels: [string, string] = content === 0
    ? [`${SLOT_NAME[img[0]]} · ${nm(img[0])}`, img[1] >= 0 ? `${SLOT_NAME[img[1]]} · ${nm(img[1])}` : '']
    : [lab(pair[0]), pair[1] >= 0 ? lab(pair[1]) : ''];
  if (content === 0 && st.showR && has[2]) labels[0] = labels[1] = `R · ${nm(2)}`;
  const two = (content === 0 ? img[1] : pair[1]) >= 0;
  return { img, pair, effContent: content, two, labels };
}

function viewState(): ViewState {
  const d = derived();
  return {
    layout: st.layout, content: d.effContent, toggle: st.toggle, split: st.split, ev: st.ev, tonemap: st.tonemap,
    cx: st.cx, cy: st.cy, zoom: st.zoom * dpr(), img: d.img, pair: d.pair,
  };
}

let rafPending = false;
function requestDraw() {
  if (rafPending) return;
  rafPending = true;
  requestAnimationFrame(() => { rafPending = false; syncSize(); gl.draw(viewState()); });
}

/** Keep the drawing buffer at CSS size x devicePixelRatio (the ratio changes with browser zoom or display moves). */
function syncSize() {
  const r = stageRect();
  const w = Math.max(1, Math.round(r.width * dpr())), h = Math.max(1, Math.round(r.height * dpr()));
  if (canvas.width === w && canvas.height === h) return;
  canvas.width = w; canvas.height = h;
  if (st.fitted) fit();
}

function stageRect() { return stage.getBoundingClientRect(); }
function panelFor(x: number, W: number, two: boolean): { ox: number; w: number } {
  if (!(two && st.layout === LAYOUT.side)) return { ox: 0, w: W };
  const h = Math.floor(W / 2);
  return x < h ? { ox: 0, w: h } : { ox: h, w: W - h };
}

function fit() {
  const sz = imgSize();
  if (!sz) return;
  const r = stageRect();
  const d = derived();
  const w = d.two && st.layout === LAYOUT.side ? r.width / 2 : r.width;
  st.zoom = Math.min(w / sz[0], r.height / sz[1]) * 0.98;
  st.cx = sz[0] / 2; st.cy = sz[1] / 2;
  st.fitted = true;
  requestDraw();
}

/** CSS-px cursor (relative to the stage) -> image pixel coordinates. */
function toImage(px: number, py: number): { x: number; y: number } {
  const r = stageRect();
  const p = panelFor(px, r.width, derived().two);
  return { x: st.cx + (px - p.ox - p.w / 2) / st.zoom, y: st.cy + (py - r.height / 2) / st.zoom };
}

// ---- UI sync ----------------------------------------------------------------------------------------------------
function refreshUI() {
  const d = derived();
  document.querySelectorAll<HTMLButtonElement>('.slot').forEach((b) => {
    const i = Number(b.dataset.slot);
    const l = loaded[i];
    b.classList.toggle('filled', !!l);
    b.querySelector('.nm')!.textContent = l ? `${l.name}  ${l.img.width}x${l.img.height}` : (i === 2 ? 'reference (optional)' : 'drop / pick');
    b.title = l?.url ?? l?.name ?? 'click to pick a file';
  });
  const hasPair = d.pair[0] >= 0;
  document.querySelectorAll<HTMLButtonElement>('#seg-content button').forEach((b, i) => {
    b.classList.toggle('on', i === d.effContent);
    b.disabled = i > 0 && !hasPair;
  });
  document.querySelectorAll<HTMLButtonElement>('#seg-layout button').forEach((b, i) => { b.classList.toggle('on', i === st.layout); b.disabled = !d.two; });
  $('labL').textContent = d.two && st.layout === LAYOUT.flip ? d.labels[st.toggle]! : d.labels[0];
  $('labR').textContent = d.two && st.layout !== LAYOUT.flip ? d.labels[1] : '';
  const lg = $('legend');
  const c = d.effContent;
  lg.hidden = !(c >= 2) || document.body.dataset.screen !== 'viewer';
  if (c >= 2) {
    const stops = c === 3 ? STOPS.magma : c === 2 ? STOPS.turbo : null;
    const css = stops
      ? `linear-gradient(90deg, ${stops.map((s, i) => `rgb(${s.map((v) => Math.round(v * 255)).join(',')}) ${(i / (stops.length - 1)) * 100}%`).join(',')})`
      : 'linear-gradient(90deg, #0d40ff, #fff, #ff1a0d)';
    lg.querySelector<HTMLElement>('.bar')!.style.background = css;
    lg.querySelector('.ticks')!.innerHTML = c === 2 ? '<span>0.1%</span><span>1%</span><span>10%</span><span>100%</span>'
      : c === 3 ? '<span>0</span><span>0.5</span><span>1 FLIP</span>' : '<span>−1</span><span>0</span><span>+1 (× EV gain)</span>';
  }
  renderMetrics();
  updateURL();
  requestDraw();
}


function updateURL() {
  const q = new URLSearchParams(location.search);
  for (const [k, i] of [['a', 0], ['b', 1], ['r', 2]] as const) {
    const u = loaded[i]?.url;
    if (u) q.set(k, u); else q.delete(k);
  }
  q.set('view', CONTENT_KEYS[st.content]!);
  q.set('layout', LAYOUT_KEYS[st.layout]!);
  q.set('ev', String(st.ev));
  q.set('tm', TONEMAP_KEYS[st.tonemap]!);
  history.replaceState(null, '', `${location.pathname}?${decodeURIComponent(q.toString()).replace(/ /g, '%20')}`);
}

function setContent(i: number) { st.content = i; refreshUI(); }
function setLayout(i: number) { st.layout = i; if (st.fitted) fit(); refreshUI(); }
function setEV(v: number) {
  st.ev = Math.max(-10, Math.min(10, v));
  $<HTMLInputElement>('ev').value = String(st.ev);
  $('ev-out').textContent = (st.ev > 0 ? '+' : '') + st.ev.toFixed(2).replace(/\.?0+$/, '');
  updateURL(); requestDraw();
}
function setTonemap(i: number) { st.tonemap = i; $<HTMLSelectElement>('tm').selectedIndex = i; updateURL(); requestDraw(); }

document.querySelectorAll<HTMLButtonElement>('#seg-content button').forEach((b, i) => b.addEventListener('click', () => setContent(i)));
document.querySelectorAll<HTMLButtonElement>('#seg-layout button').forEach((b, i) => b.addEventListener('click', () => setLayout(i)));
$('ev').addEventListener('input', (e) => setEV(Number((e.target as HTMLInputElement).value)));
$('ev').addEventListener('dblclick', () => setEV(0));
$('tm').addEventListener('change', (e) => setTonemap((e.target as HTMLSelectElement).selectedIndex));
$('help-btn').addEventListener('click', () => { $('help').hidden = !$('help').hidden; });
$('back').addEventListener('click', () => { document.body.dataset.screen = 'report'; $('report').hidden = false; $('back').hidden = true; $('metrics').hidden = true; $('legend').hidden = true; });

// ---- file picking & drag-drop -----------------------------------------------------------------------------------
const fileInput = $<HTMLInputElement>('file');
let pickSlot = 0;
document.querySelectorAll<HTMLButtonElement>('.slot').forEach((b) => b.addEventListener('click', (e) => {
  const slot = Number(b.dataset.slot);
  if ((e.target as HTMLElement).classList.contains('x')) { clearSlot(slot); return; }
  pickSlot = slot; fileInput.value = ''; fileInput.click();
}));
fileInput.addEventListener('change', async () => {
  const files = [...(fileInput.files ?? [])];
  for (let i = 0; i < files.length && pickSlot + i < 3; i++) await loadFile(pickSlot + i, files[i]!);
});

let dragDepth = 0;
const zones = $('dropzones');
window.addEventListener('dragenter', (e) => { if (e.dataTransfer?.types.includes('Files')) { dragDepth++; zones.hidden = false; } });
window.addEventListener('dragleave', () => { if (--dragDepth <= 0) { dragDepth = 0; zones.hidden = true; } });
window.addEventListener('dragover', (e) => {
  e.preventDefault();
  zones.querySelectorAll('div').forEach((z) => z.classList.toggle('over', z === e.target));
});
window.addEventListener('drop', async (e) => {
  e.preventDefault();
  dragDepth = 0; zones.hidden = true;
  const files = [...(e.dataTransfer?.files ?? [])];
  const z = (e.target as HTMLElement).closest<HTMLElement>('#dropzones > div');
  let slot = z ? Number(z.dataset.slot) : loaded.findIndex((l) => !l);
  if (slot < 0) slot = 0;
  for (const file of files) { if (slot > 2) break; await loadFile(slot++, file); }
});

// ---- mouse / keyboard -------------------------------------------------------------------------------------------
let drag: { x: number; y: number; mode: 'pan' | 'split' } | undefined;
canvas.addEventListener('pointerdown', (e) => {
  const r = stageRect();
  const x = e.clientX - r.left;
  const d = derived();
  const mode = d.two && st.layout === LAYOUT.swipe && Math.abs(x - st.split * r.width) < 10 ? 'split' : 'pan';
  drag = { x: e.clientX, y: e.clientY, mode };
  canvas.setPointerCapture(e.pointerId);
  canvas.classList.add('drag');
});
canvas.addEventListener('pointerup', () => { drag = undefined; canvas.classList.remove('drag'); });
canvas.addEventListener('pointermove', (e) => {
  const r = stageRect();
  const x = e.clientX - r.left, y = e.clientY - r.top;
  if (drag) {
    if (drag.mode === 'split') st.split = Math.max(0, Math.min(1, x / r.width));
    else { st.cx -= (e.clientX - drag.x) / st.zoom; st.cy -= (e.clientY - drag.y) / st.zoom; st.fitted = false; }
    drag.x = e.clientX; drag.y = e.clientY;
    requestDraw();
  }
  inspect(x, y);
});
canvas.addEventListener('pointerleave', () => { $('inspector').hidden = true; });
canvas.addEventListener('dblclick', () => fit());
canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  const r = stageRect();
  const x = e.clientX - r.left, y = e.clientY - r.top;
  const before = toImage(x, y);
  const k = Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0015));
  st.zoom = Math.max(0.02 / dpr(), Math.min(256 / dpr(), st.zoom * k));
  const p = panelFor(x, r.width, derived().two);
  st.cx = before.x - (x - p.ox - p.w / 2) / st.zoom;
  st.cy = before.y - (y - r.height / 2) / st.zoom;
  st.fitted = false;
  requestDraw();
  inspect(x, y);
}, { passive: false });
new ResizeObserver(() => { syncSize(); if (st.fitted) fit(); requestDraw(); }).observe(stage);
window.addEventListener('resize', () => requestDraw());

function inspect(px: number, py: number) {
  const el = $('inspector');
  const sz = imgSize();
  if (!sz || document.body.dataset.screen !== 'viewer') { el.hidden = true; return; }
  const p = toImage(px, py);
  const x = Math.floor(p.x), y = Math.floor(p.y);
  if (x < 0 || y < 0 || x >= sz[0] || y >= sz[1]) { el.hidden = true; return; }
  el.hidden = false;
  const idx = y * sz[0] + x;
  const px3 = (i: number): [number, number, number] | undefined => {
    const l = loaded[i];
    if (!l) return undefined;
    const o = idx * l.img.channels;
    return [l.img.data[o]!, l.img.data[o + 1]!, l.img.data[o + 2]!];
  };
  const rowOf = (label: string, v: [number, number, number] | undefined, extra = '') =>
    v ? `<tr><td>${label}</td><td>${f(v[0], 5)}</td><td>${f(v[1], 5)}</td><td>${f(v[2], 5)}</td><td>${extra}</td></tr>` : '';
  const lum = (v: [number, number, number]) => f(0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2], 5);
  let html = `<div class="muted">x ${x}  y ${y}   zoom ${(st.zoom * dpr() * 100).toFixed(0)}%</div><table><tr><th></th><th>R</th><th>G</th><th>B</th><th>luma</th></tr>`;
  const vals = [0, 1, 2].map(px3);
  for (let i = 0; i < 3; i++) {
    const v = vals[i];
    if (!v) continue;
    const sw = `<span class="sw" style="background:rgb(${v.map((c) => Math.round(255 * Math.min(1, Math.max(0, c)) ** (1 / 2.2))).join(',')})"></span>`;
    html += rowOf(`${sw}${SLOT_NAME[i]}`, v, lum(v));
  }
  const diff = (a: [number, number, number] | undefined, b: [number, number, number] | undefined, label: string) =>
    a && b ? rowOf(label, [a[0] - b[0], a[1] - b[1], a[2] - b[2]], '') : '';
  html += diff(vals[0], vals[1], 'A−B') + diff(vals[0], vals[2], 'A−R') + diff(vals[1], vals[2], 'B−R');
  html += '</table>';
  const flips = [0, 1, 2].filter((i) => pairs[i]?.flipMap).map((i) => `${PAIR_LABEL[i]} ${f(pairs[i]!.flipMap![idx]!, 4)}`);
  if (flips.length) html += `<div class="muted">FLIP: ${flips.join('   ')}</div>`;
  el.innerHTML = html;
}

window.addEventListener('keydown', (e) => {
  const tag = (e.target as HTMLElement).tagName;
  if (tag === 'INPUT' && (e.target as HTMLInputElement).type === 'text') return;
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const k = e.key.toLowerCase();
  const d = derived();
  if (k >= '1' && k <= '5') { if (Number(k) === 1 || d.pair[0] >= 0) setContent(Number(k) - 1); }
  else if (k === 's') setLayout(LAYOUT.swipe);
  else if (k === 'd') setLayout(LAYOUT.side);
  else if (k === 'f') { st.layout = LAYOUT.flip; st.toggle = 1 - st.toggle; st.showR = false; refreshUI(); }
  else if (k === 'a' || k === 'b') { st.layout = LAYOUT.flip; st.toggle = k === 'a' ? 0 : 1; st.showR = false; refreshUI(); }
  else if (k === 'r') { if (loaded[2]) { st.layout = LAYOUT.flip; st.showR = true; refreshUI(); } }
  else if (k === ' ') { e.preventDefault(); st.layout = LAYOUT.flip; st.toggle = 1 - st.toggle; st.showR = false; refreshUI(); }
  else if (k === '[') setEV(st.ev - 0.5);
  else if (k === ']') setEV(st.ev + 0.5);
  else if (k === '\\') setEV(0);
  else if (k === 't') setTonemap((st.tonemap + 1) % 4);
  else if (k === '0') fit();
  else if (k === 'z') { if (st.fitted) { const sz = imgSize(); if (sz) { st.zoom = 1 / dpr(); st.fitted = false; requestDraw(); } } else fit(); }
  else if (k === 'm') { hideMetrics = !hideMetrics; renderMetrics(); }
  else if (k === '?') $('help').hidden = !$('help').hidden;
  else return;
});

// ---- start screen, report, boot ---------------------------------------------------------------------------------
async function openPick(p: ReportPick) {
  for (let i = 0; i < 3; i++) if (loaded[i]) clearSlot(i);
  document.body.dataset.screen = 'viewer';
  $('report').hidden = true; $('back').hidden = false;
  await Promise.all([loadUrl(0, p.a), loadUrl(1, p.b), p.r ? loadUrl(2, p.r) : Promise.resolve()]);
  fit();
}

async function fillStart() {
  const el = $('files');
  try {
    const list = (await (await fetch('/__files')).json()) as Array<{ path: string; size: number; mtime: number }>;
    list.sort((a, b) => b.mtime - a.mtime);
    const reports = list.filter((x) => x.path.endsWith('report.json'));
    const pfms = list.filter((x) => x.path.endsWith('.pfm'));
    let html = '';
    if (reports.length) html += `<h3>Run reports</h3>${reports.map((r) => `<div class="frow"><span class="p">${r.path}</span><a href="?report=${encodeURIComponent(r.path)}"><button>open report</button></a></div>`).join('')}`;
    html += `<h3>Images (${pfms.length})</h3>`;
    html += pfms.map((x) => `<div class="frow"><span class="p" title="${x.path}">${x.path}</span><span class="muted">${(x.size / 1e6).toFixed(1)} MB</span>${['A', 'B', 'R'].map((s, i) => `<button data-load="${i}" data-url="${x.path}">${s}</button>`).join('')}</div>`).join('');
    if (!list.length) html += '<p class="muted">No files under bench/results, bench/references or .cache/viewer-samples. Try <code>node bench/viewer/make-samples.ts</code>.</p>';
    el.innerHTML = html;
    el.querySelectorAll<HTMLButtonElement>('button[data-load]').forEach((b) => b.addEventListener('click', () => loadUrl(Number(b.dataset.load), b.dataset.url!)));
  } catch { el.innerHTML = '<p class="muted">(file index only exists on the dev server)</p>'; }
}

async function boot() {
  const q = new URLSearchParams(location.search);
  const li = LAYOUT_KEYS.indexOf(q.get('layout') as never); if (li >= 0) st.layout = li;
  const ci = CONTENT_KEYS.indexOf(q.get('view') as never); if (ci >= 0) st.content = ci;
  const ti = TONEMAP_KEYS.indexOf(q.get('tm') as never); if (ti >= 0) setTonemap(ti);
  if (q.has('ev')) setEV(Number(q.get('ev')));
  else setEV(0);
  const rep = q.get('report');
  if (rep) {
    document.body.dataset.screen = 'report';
    $('start').hidden = true;
    $('report').hidden = false;
    try { await showReport($('report'), rep, (p) => { void openPick(p); }); } catch (e) { $('report').textContent = `report failed: ${(e as Error).message}`; }
    return;
  }
  const urls: Array<[string, number]> = [['a', 0], ['b', 1], ['r', 2]];
  const wanted = urls.filter(([k]) => q.has(k));
  if (wanted.length) await Promise.all(wanted.map(([k, i]) => loadUrl(i, q.get(k)!)));
  else void fillStart();
  refreshUI();
}
// "start" panel visibility follows the screen attribute
new MutationObserver(() => { $('start').hidden = document.body.dataset.screen !== 'start'; }).observe(document.body, { attributes: true, attributeFilter: ['data-screen'] });
void boot();
