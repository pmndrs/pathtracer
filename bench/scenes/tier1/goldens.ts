/**
 * Golden OpenPBR material tables for the Tier 1 models (docs/internal/design/tier1.md section 1.1, TASKS P2.17/P2.18): the output of
 * `gltfMaterialToOpenPbr` over each downloaded model's material list, read straight from the model JSON (GLB JSON chunk or `.gltf`),
 * textures skipped (only the slot names are listed).
 *
 *   node bench/scenes/tier1/goldens.ts [--check] [--only <Model>]
 *
 * Needs `node bench/assets/fetch.ts` first. Without `--check` it (re)writes `bench/scenes/tier1/golden/<Model>.json`; with it, it fails
 * when a golden differs. The loader (P2.19) must produce the same table.
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gltfMaterialsToOpenPbr, readGltfJson } from '../gltf-openpbr.ts';
import type { ReferenceClass } from '../openpbr.ts';

const HERE = dirname(fileURLToPath(import.meta.url));
const ASSETS = join(HERE, '..', '..', 'assets');
export const GOLDEN_DIR = join(HERE, 'golden');

interface ManifestAsset { id: string; files: { path: string; sha256: string; bytes: number }[] }

/** Tier 1 model names in the manifest, with the path of the file that holds the JSON and its sha256. */
export function tier1Models(): { name: string; file: string; sha256: string }[] {
  const manifest = JSON.parse(readFileSync(join(ASSETS, 'MANIFEST.json'), 'utf8')) as { assets: ManifestAsset[] };
  const out: { name: string; file: string; sha256: string }[] = [];
  for (const a of manifest.assets) {
    if (!a.id.startsWith('tier1-')) continue;
    const f = a.files.find((x) => /\.(glb|gltf)$/.test(x.path));
    if (f) out.push({ name: a.id.slice('tier1-'.length), file: join(ASSETS, f.path), sha256: f.sha256 });
  }
  return out;
}

/** Round to 7 significant digits so a golden does not depend on the last bits of a derived value. */
const round = (v: unknown): unknown => {
  if (typeof v === 'number') return Number.isInteger(v) ? v : Number(v.toPrecision(7));
  if (Array.isArray(v)) return v.map(round);
  if (v && typeof v === 'object') return Object.fromEntries(Object.entries(v).map(([k, x]) => [k, round(x)]));
  return v;
};

/** The golden text of one model: a small header and one material per line. */
export function goldenText(model: { name: string; file: string; sha256: string }): string {
  const json = readGltfJson(model.file);
  const mats = gltfMaterialsToOpenPbr(json);
  const counts: Record<ReferenceClass, number> = { A: 0, B: 0, "B'": 0, C: 0 };
  for (const m of mats) counts[m.class]++;
  const header = { model: model.name, sourceSha256: model.sha256, extensionsUsed: json.extensionsUsed ?? [], materialCount: mats.length, classCounts: counts };
  const lines = mats.map((m) => JSON.stringify(round({
    i: m.index, name: m.name, class: m.class, openpbr: { ...m.material, type: undefined },
    ...(m.notes.length ? { notes: m.notes } : {}), ...(m.textures.length ? { textures: m.textures } : {}), ...(m.unsupported.length ? { unsupported: m.unsupported } : {}),
  })));
  return `{\n"header": ${JSON.stringify(header)},\n"materials": [\n${lines.join(',\n')}\n]\n}\n`;
}

export function main(argv: string[]): number {
  const check = argv.includes('--check');
  const only = argv.includes('--only') ? argv[argv.indexOf('--only') + 1] : undefined;
  let bad = 0;
  mkdirSync(GOLDEN_DIR, { recursive: true });
  for (const model of tier1Models()) {
    if (only && model.name !== only) continue;
    if (!existsSync(model.file)) { console.log(`missing  ${model.name} (run node bench/assets/fetch.ts)`); bad++; continue; }
    const text = goldenText(model);
    const path = join(GOLDEN_DIR, `${model.name}.json`);
    if (check) {
      const same = existsSync(path) && readFileSync(path, 'utf8') === text;
      console.log(`${same ? 'ok      ' : 'DIFFERS '} ${model.name}`);
      if (!same) bad++;
    } else {
      writeFileSync(path, text);
      console.log(`wrote    ${model.name} (${(text.length / 1024).toFixed(1)} KB)`);
    }
  }
  return bad ? 1 : 0;
}

if (process.argv[1] && fileURLToPath(import.meta.url) === resolve(process.argv[1])) process.exit(main(process.argv.slice(2)));
