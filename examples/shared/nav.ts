/**
 * The frame every example page gets (injected by vite.config.ts, and by bench/hero/build-pages.mjs for the Classroom showcase): a pill in
 * the top-left corner with the pmndrs logo (back to the gallery), the breadcrumb, the status badge of an experimental or research example,
 * an info button (the catalog description), and links to the docs page and the source. Examples never import it.
 *
 * A page whose top-left corner is taken moves the pill: `<meta name="example-nav-offset" content="64px 16px">` (top, left), or puts it in the
 * bottom-left corner with `<meta name="example-nav-placement" content="bottom">`.
 */
import { CATEGORIES, EXAMPLES, REPOSITORY, SITE_URL, STATUS_LABELS, type ExampleEntry } from '../catalog.ts';
import logo from './pmndrs-logo.svg?raw';

const STYLE = `
.ex-nav { position: fixed; z-index: 100000; display: flex; align-items: center; gap: 2px; padding: 4px; max-width: calc(100vw - 24px);
  background: rgba(14, 14, 12, 0.82); border: 1px solid rgba(255, 255, 240, 0.12); border-radius: 999px; backdrop-filter: blur(8px);
  font: 13px/1 ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif; color: #ecebe4; user-select: none; }
.ex-nav a, .ex-nav button, .ex-nav span.crumb { display: inline-flex; align-items: center; gap: 6px; height: 30px; padding: 0 10px;
  border: 0; border-radius: 999px; background: none; color: inherit; font: inherit; text-decoration: none; cursor: pointer; white-space: nowrap; }
.ex-nav a:hover, .ex-nav button:hover, .ex-nav a:focus-visible, .ex-nav button:focus-visible { background: rgba(202, 245, 67, 0.14); color: #caf543; outline: none; }
.ex-nav .logo { padding: 0 6px 0 4px; }
.ex-nav .logo svg { width: 24px; height: 24px; border-radius: 6px; display: block; }
.ex-nav .crumb { color: #a3a194; cursor: default; overflow: hidden; text-overflow: ellipsis; }
.ex-nav .crumb b { color: #ecebe4; font-weight: 600; }
.ex-nav .sep { width: 1px; height: 18px; background: rgba(255, 255, 240, 0.12); margin: 0 2px; }
.ex-nav .status { margin: 0 4px; padding: 1px 8px; border: 1px solid currentColor; border-radius: 999px; font-size: 11px; font-weight: 600; line-height: 18px; }
.ex-nav .status.experimental { color: #ffc35a; }
.ex-nav .status.research { color: #c9a7ff; }
.ex-info { position: fixed; z-index: 100000; width: min(420px, calc(100vw - 24px)); padding: 14px 16px; border-radius: 12px;
  background: rgba(14, 14, 12, 0.92); border: 1px solid rgba(255, 255, 240, 0.12); backdrop-filter: blur(8px); color: #ecebe4;
  font: 13px/1.55 ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif; }
.ex-info h2 { margin: 0 0 6px; font-size: 15px; }
.ex-info p { margin: 0 0 8px; color: #cfcdc2; }
.ex-info .note { color: #a3a194; font-size: 12px; }
.ex-info code { font: 12px ui-monospace, Menlo, monospace; color: #caf543; }
.ex-info[hidden] { display: none; }
@media (max-width: 1100px) { .ex-nav .label { display: none; } }
@media (max-width: 900px) { .ex-nav .crumb .cat { display: none; } }
@media (max-width: 520px) { .ex-nav .crumb { display: none; } }
`;

const icon = (path: string): string => `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${path}</svg>`;
const ICONS = {
  info: icon('<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/>'),
  docs: icon('<path d="M4 5a2 2 0 0 1 2-2h12v16H6a2 2 0 0 0-2 2z"/><path d="M4 19V5"/>'),
  source: icon('<path d="m8 8-4 4 4 4M16 8l4 4-4 4"/>'),
  fullscreen: icon('<path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/>'),
};

function el<K extends keyof HTMLElementTagNameMap>(tag: K, props: Partial<HTMLElementTagNameMap[K]> & { className?: string } = {}, html?: string): HTMLElementTagNameMap[K] {
  const node = Object.assign(document.createElement(tag), props);
  if (html !== undefined) node.innerHTML = html;
  return node;
}

/** The gallery root, from this page's place in the catalog (works under any deploy base, and for pages built elsewhere). */
function galleryRoot(entry: ExampleEntry): URL {
  return new URL('../'.repeat(entry.path.split('/').filter(Boolean).length), location.href);
}

function docsUrl(entry: ExampleEntry, root: URL): string | undefined {
  if (!entry.docs) return undefined;
  // In dev the docs are not served next to the gallery: link the deployed site.
  return import.meta.env.DEV ? new URL(entry.docs, SITE_URL).href : new URL(`../${entry.docs}`, root).href;
}

function mount(): void {
  const slug = document.querySelector<HTMLMetaElement>('meta[name="example"]')?.content;
  const entry = slug ? EXAMPLES.find((e) => e.slug === slug) : undefined;
  if (!entry) return;
  const root = galleryRoot(entry);
  const [top, left] = (document.querySelector<HTMLMetaElement>('meta[name="example-nav-offset"]')?.content ?? '12px 12px').split(/\s+/);
  const bottom = document.querySelector<HTMLMetaElement>('meta[name="example-nav-placement"]')?.content === 'bottom';

  document.head.append(el('style', {}, STYLE));
  const nav = el('nav', { className: 'ex-nav', ariaLabel: 'Example navigation' });
  if (bottom) nav.style.bottom = '12px';
  else nav.style.top = top ?? '12px';
  nav.style.left = left ?? '12px';

  const home = el('a', { className: 'logo', href: root.href, title: 'All examples', ariaLabel: 'All examples' }, logo);
  const category = CATEGORIES.find((c) => c.id === entry.category)?.title ?? '';
  const crumb = el('span', { className: 'crumb' }, `<span class="cat">${category} /</span> <b></b>`);
  crumb.querySelector('b')!.textContent = entry.title;
  nav.append(home, crumb);
  if (entry.status !== 'stable') {
    const status = el('span', { className: `status ${entry.status}`, title: STATUS_LABELS[entry.status].blurb });
    status.textContent = STATUS_LABELS[entry.status].label;
    nav.append(status);
  }
  nav.append(el('span', { className: 'sep' }));

  const info = el('aside', { className: 'ex-info', hidden: true });
  if (bottom) info.style.bottom = '60px';
  else info.style.top = `calc(${top ?? '12px'} + 48px)`;
  info.style.left = left ?? '12px';
  const h = el('h2');
  h.textContent = entry.title;
  const p = el('p');
  p.innerHTML = inlineMarkdown(entry.description);
  info.append(h, p);
  if (entry.status !== 'stable') {
    const note = el('p', { className: 'note' });
    note.textContent = `${STATUS_LABELS[entry.status].label}: ${STATUS_LABELS[entry.status].blurb}`;
    info.append(note);
  }
  const infoButton = el('button', { type: 'button', title: 'About this example', ariaExpanded: 'false' }, `${ICONS.info}<span class="label">Info</span>`);
  infoButton.addEventListener('click', () => {
    info.hidden = !info.hidden;
    infoButton.ariaExpanded = String(!info.hidden);
  });
  nav.append(infoButton);

  const docs = docsUrl(entry, root);
  if (docs) nav.append(el('a', { href: docs, title: 'Documentation' }, `${ICONS.docs}<span class="label">Docs</span>`));
  nav.append(el('a', { href: `${REPOSITORY}/tree/main/${entry.source}`, target: '_blank', rel: 'noopener', title: 'Source on GitHub' }, `${ICONS.source}<span class="label">Source</span>`));
  if (document.fullscreenEnabled) {
    const full = el('button', { type: 'button', title: 'Fullscreen' }, ICONS.fullscreen);
    full.addEventListener('click', () => (document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen()));
    nav.append(full);
  }
  document.body.append(nav, info);
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', mount);
else mount();

/** The catalog's descriptions use `code` and **bold**: escape the text, then render those two. */
function inlineMarkdown(text: string): string {
  const escaped = text.replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
  return escaped.replace(/`([^`]+)`/g, '<code>$1</code>').replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
}
