/**
 * Writes synthetic PFMs (ref, a, b, per-scene cells) and a report.json manifest to .cache/viewer-samples (git-ignored,
 * served by the viewer dev server at /samples/). Usage: node bench/viewer/make-samples.ts [outDir]
 */
import { writeFileSync, mkdirSync } from 'node:fs';
import { makeImage, writePFM } from '../metrics/pfm.ts';
const out = process.argv[2] ?? new URL('../../.cache/viewer-samples', import.meta.url).pathname; mkdirSync(out, { recursive: true });
const W = 512, H = 288;
let s = 12345; const rnd = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
const base = (x: number, y: number): [number, number, number] => {
  const u = x / W, v = y / H;
  const d = Math.hypot(u - 0.5, v - 0.5);
  const sun = d < 0.08 ? 50 : 0;
  const checker = ((Math.floor(u * 16) + Math.floor(v * 9)) & 1) ? 0.6 : 0.15;
  return [checker * (0.5 + u) + sun, checker * (1.2 - u) + sun, checker * (0.4 + v) + sun];
};
const ref = makeImage(W, H, base);
const noisy = (amp: number, bias: number, fire: number) => makeImage(W, H, (x, y) => {
  const [r, g, b] = base(x, y); const n = () => (rnd() - 0.5) * amp;
  const f = rnd() < fire ? 40 : 0;
  return [Math.max(0, r * bias + n() * (1 + r) + f), Math.max(0, g * bias + n() * (1 + g) + f), Math.max(0, b * bias + n() * (1 + b) + f)];
});
const w = (n: string, i: Parameters<typeof writePFM>[0]) => writeFileSync(`${out}/${n}.pfm`, writePFM(i));
w('ref', ref); w('a', noisy(0.4, 1, 0.0005)); w('b', noisy(0.15, 1.04, 0));
// run report
const scenes = ['furnace', 'checker'], rends = ['ours', 'other'];
const cells: unknown[] = [];
for (const sc of scenes) for (const r of rends) {
  const n = r === 'ours' ? 'a' : 'b'; const f = `${sc}_${r}.pfm`;
  w(`${sc}_${r}`, n === 'a' ? noisy(0.4, 1, 0.0005) : noisy(0.15, 1.04, 0));
  cells.push({ scene: sc, renderer: r, image: f, metrics: { relMse: rnd() * 0.1, flip: rnd() * 0.3, wallMs: 100 + rnd() * 400 } });
}
console.log(`wrote samples to ${out}`);
writeFileSync(`${out}/report.json`, JSON.stringify({ run: 'synthetic', scenes: scenes.map((id) => ({ id, name: id, reference: 'ref.pfm' })), renderers: rends.map((id) => ({ id, name: id })), cells }, null, 2));
