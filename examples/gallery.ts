/** The gallery index: the catalog grouped by category, filterable, with the browser's WebGPU support checked against each example. */
import { CATEGORIES, EXAMPLES, SITE_URL, STATUS_LABELS, type CategoryId, type ExampleEntry } from './catalog.ts';

const sections = document.getElementById('sections')!;
const chips = document.getElementById('chips')!;
const showExperimental = document.getElementById('show-experimental') as HTMLInputElement;
const gpuBox = document.getElementById('gpu')!;

// Docs live next to the gallery in the deployed site; in dev they are not served, so link the deployed ones.
if (import.meta.env.DEV) (document.getElementById('docs-link') as HTMLAnchorElement).href = SITE_URL;

// Card images: `examples/<category>/<slug>/thumb.webp` (written by `node scripts/examples-smoke.mjs --thumbs`), bundled with hashed URLs.
const THUMBS = import.meta.glob<string>('./*/*/thumb.webp', { eager: true, query: '?url', import: 'default' });

let selected: CategoryId | 'all' = 'all';
let features: ReadonlySet<string> | null = null;

function hash(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
  return h >>> 0;
}

/** Placeholder art until an example has a captured thumbnail: a few light paths bouncing in a box, seeded by the slug. */
function placeholder(entry: ExampleEntry): string {
  let seed = hash(entry.slug);
  const rand = () => ((seed = Math.imul(seed ^ (seed >>> 15), 2246822507) ^ Math.imul(seed ^ (seed >>> 13), 3266489909)) >>> 0) / 4294967296;
  const hue = (hash(entry.category) % 360 + 70) % 360;
  const paths: string[] = [];
  for (let k = 0; k < 7; k++) {
    let x = 160 + (rand() - 0.5) * 40, y = 30;
    const pts = [`${x.toFixed(1)},${y}`];
    for (let b = 0; b < 4; b++) {
      x = 20 + rand() * 280;
      y = 20 + rand() * 140;
      pts.push(`${x.toFixed(1)},${y.toFixed(1)}`);
    }
    paths.push(`<polyline points="${pts.join(' ')}" stroke="hsl(${(hue + k * 9) % 360} 85% ${62 - k * 3}%)" stroke-opacity="${(0.75 - k * 0.07).toFixed(2)}" />`);
  }
  return `<svg viewBox="0 0 320 180" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
    <defs><radialGradient id="g${entry.slug}" cx="50%" cy="10%" r="90%"><stop offset="0" stop-color="hsl(${hue} 40% 22%)"/><stop offset="1" stop-color="#121210"/></radialGradient></defs>
    <rect width="320" height="180" fill="url(#g${entry.slug})"/>
    <rect x="130" y="14" width="60" height="8" rx="2" fill="hsl(${hue} 90% 80%)"/>
    <g fill="none" stroke-width="1.4" stroke-linejoin="round">${paths.join('')}</g>
  </svg>`;
}

function card(entry: ExampleEntry): HTMLElement {
  const deployedOnly = !!entry.external && import.meta.env.DEV;
  const missing = (entry.requires ?? []).filter((f) => features && !features.has(f));
  const node = document.createElement(deployedOnly ? 'div' : 'a');
  node.className = `card${deployedOnly || missing.length ? ' unavailable' : ''}`;
  node.id = entry.slug;
  if (node instanceof HTMLAnchorElement) node.href = entry.path;

  const thumb = document.createElement('div');
  thumb.className = 'thumb';
  const thumbUrl = THUMBS[`./${entry.path}thumb.webp`] ?? (entry.thumbnail ? THUMBS[`./${entry.thumbnail}`] : undefined);
  if (thumbUrl) {
    const img = document.createElement('img');
    img.src = thumbUrl;
    img.alt = '';
    img.loading = 'lazy';
    thumb.append(img);
  } else thumb.innerHTML = placeholder(entry);
  if (entry.status !== 'stable') {
    const badge = document.createElement('span');
    badge.className = `badge status ${entry.status}`;
    badge.textContent = STATUS_LABELS[entry.status].label;
    badge.title = STATUS_LABELS[entry.status].blurb;
    thumb.append(badge);
  }

  const body = document.createElement('div');
  body.className = 'body';
  const h = document.createElement('h3');
  h.textContent = entry.title;
  const p = document.createElement('p');
  p.textContent = entry.summary;
  body.append(h, p);
  if (missing.length) {
    const warn = document.createElement('div');
    warn.className = 'warn';
    warn.textContent = `Needs ${missing.join(', ')}, which this browser does not offer.`;
    body.append(warn);
  }
  if (deployedOnly) {
    const dev = document.createElement('div');
    dev.className = 'dev';
    dev.textContent = `Built by the site workflow. Locally: ${entry.external!.dev}`;
    body.append(dev);
  }
  const tags = document.createElement('div');
  tags.className = 'tags';
  for (const t of [...entry.tags, ...(entry.network ? ['streams assets'] : []), ...(entry.heavy ? ['heavy'] : [])]) {
    const span = document.createElement('span');
    span.textContent = t;
    tags.append(span);
  }
  body.append(tags);
  node.append(thumb, body);
  return node;
}

function render(): void {
  sections.replaceChildren();
  const visible = EXAMPLES.filter((e) => (selected === 'all' || e.category === selected) && (showExperimental.checked || e.status === 'stable'));
  for (const category of CATEGORIES) {
    const entries = visible.filter((e) => e.category === category.id);
    if (!entries.length) continue;
    const section = document.createElement('section');
    section.className = 'section';
    section.id = `category-${category.id}`;
    const h = document.createElement('h2');
    h.textContent = category.title;
    const p = document.createElement('p');
    p.textContent = category.blurb;
    const grid = document.createElement('div');
    grid.className = 'grid';
    grid.append(...entries.map(card));
    section.append(h, p, grid);
    sections.append(section);
  }
  if (!sections.childElementCount) {
    const empty = document.createElement('p');
    empty.className = 'empty';
    empty.textContent = 'No examples match these filters yet.';
    sections.append(empty);
  }
}

function renderChips(): void {
  const used = CATEGORIES.filter((c) => EXAMPLES.some((e) => e.category === c.id));
  const all: { id: CategoryId | 'all'; title: string }[] = [{ id: 'all', title: 'All' }, ...used];
  chips.replaceChildren(
    ...all.map((c) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.textContent = c.title;
      b.setAttribute('aria-pressed', String(selected === c.id));
      b.addEventListener('click', () => {
        selected = c.id;
        renderChips();
        render();
      });
      return b;
    }),
  );
}

async function checkGpu(): Promise<void> {
  const OPTIONAL = ['subgroups', 'shader-f16', 'timestamp-query'] as const;
  if (!('gpu' in navigator)) {
    gpuBox.className = 'gpu bad';
    gpuBox.innerHTML = '<b>No WebGPU in this browser.</b> The examples need Chrome or Edge 113+, Safari 26+, or Firefox 141+ (Windows).';
    return;
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' }).catch(() => null);
  if (!adapter) {
    gpuBox.className = 'gpu bad';
    gpuBox.innerHTML = '<b>WebGPU is present but no GPU adapter is available</b> (blocklisted driver or disabled hardware acceleration).';
    return;
  }
  features = new Set([...adapter.features].map(String));
  const info = adapter.info;
  const name = [info?.vendor, info?.architecture].filter(Boolean).join(' ') || 'GPU';
  gpuBox.innerHTML = `<b>WebGPU ready</b> <span></span>${OPTIONAL.map((f) => `<span class="feat ${features!.has(f) ? 'on' : 'off'}">${f}</span>`).join('')}`;
  gpuBox.querySelector('span')!.textContent = name;
  render();
}

showExperimental.addEventListener('change', render);
renderChips();
render();
void checkGpu();
