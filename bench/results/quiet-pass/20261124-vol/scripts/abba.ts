// Paired ABBA timing of competitor-page variants (quiet pass 2026-11-24).
// node abba.ts <out.json> <name> <scene> <spp> <rounds> <variant>=<k=v;k=v> ...   (variant args may include renderer=...)
// Each round runs every variant once through bench/runner/run.ts (one page run each, --no-csv), order reversed every other
// round (A B C, C B A, ...). Ratios are variant / first variant within the same round.
import { spawnSync } from 'node:child_process';
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';

const REPO = '/Users/dex/Developer/pathtracer';
const [outArg, name, scene, sppS, roundsS, ...specs] = process.argv.slice(2);
const out = resolve(outArg);
const spp = Number(sppS), rounds = Number(roundsS);
const variants = specs.map((s) => {
  const i = s.indexOf('=');
  const key = s.slice(0, i);
  const args = s.slice(i + 1).split(';').filter(Boolean);
  if (!args.some((a) => a.startsWith('renderer='))) args.unshift('renderer=pt-megakernel');
  return { key, args };
});
type Row = { round: number; variant: string; ok: boolean; ms_per_spp?: number; wall_ms?: number; compile_ms?: number; lum?: number; nonFinite?: number; load1?: number; loaded?: boolean; err?: string };
const rows: Row[] = [];
const tmp = join(dirname(out), 'runs');
mkdirSync(tmp, { recursive: true });
for (let r = 0; r < rounds; r++) {
  const order = r % 2 === 0 ? variants : [...variants].reverse();
  for (const v of order) {
    const json = join(tmp, `${name}-${v.key}-r${r}.json`);
    const args = [`scene=${scene}`, `spp=${spp}`, ...v.args];
    const p = spawnSync(process.execPath, [join(REPO, 'bench/runner/run.ts'), '--page', 'competitor', '--suite', 'competitor', '--no-csv', '--timeout', '900000', '--json', json, '--args', ...args], { cwd: REPO, encoding: 'utf8' });
    const row: Row = { round: r, variant: v.key, ok: false };
    try {
      const j = JSON.parse(readFileSync(json, 'utf8'));
      const run = j.runs?.[0];
      const m = run?.result?.metrics ?? {};
      row.ok = !!j.ok && typeof m.ms_per_spp === 'number';
      Object.assign(row, { ms_per_spp: m.ms_per_spp, wall_ms: m.wall_ms, compile_ms: m.compile_ms, lum: 0.2126 * m.mean_r + 0.7152 * m.mean_g + 0.0722 * m.mean_b, nonFinite: m.non_finite_pixels, load1: run?.load?.load1, loaded: run?.load?.loaded });
      if (!row.ok) row.err = j.error ?? run?.reasons?.join('; ');
    } catch (e) {
      row.err = `no json: ${(p.stdout + p.stderr).split('\n').filter((l) => /FAIL|Error|error/.test(l)).slice(0, 3).join(' | ').slice(0, 400)}`;
    }
    rows.push(row);
    console.log(`${name} r${r} ${v.key}: ${row.ok ? row.ms_per_spp!.toFixed(3) + ' ms/spp' : 'FAILED ' + row.err} load ${row.load1?.toFixed(2)}`);
  }
}
const q = (xs: number[], f: number) => { const s = [...xs].sort((a, b) => a - b); return s[Math.min(s.length - 1, Math.floor(f * (s.length - 1) + 0.5))]; };
const summary = variants.map((v) => {
  const mine = rows.filter((x) => x.variant === v.key && x.ok);
  const ratios: number[] = [];
  for (let r = 0; r < rounds; r++) {
    const a = rows.find((x) => x.round === r && x.variant === variants[0].key && x.ok);
    const b = rows.find((x) => x.round === r && x.variant === v.key && x.ok);
    if (a && b) ratios.push(b.ms_per_spp! / a.ms_per_spp!);
  }
  const ms = mine.map((x) => x.ms_per_spp!);
  return { variant: v.key, args: v.args.join(' '), n: mine.length, ms_min: Math.min(...ms), ms_med: ms.length ? q(ms, 0.5) : NaN, ms_max: Math.max(...ms), ratio_min: Math.min(...ratios), ratio_med: ratios.length ? q(ratios, 0.5) : NaN, ratio_max: Math.max(...ratios), lum: mine.length ? mine.reduce((s, x) => s + x.lum!, 0) / mine.length : NaN, nonFinite: Math.max(0, ...mine.map((x) => x.nonFinite ?? 0)), compile_ms: mine.length ? q(mine.map((x) => x.compile_ms!), 0.5) : NaN };
});
const loads = rows.map((x) => x.load1).filter((x): x is number => typeof x === 'number');
const doc = { name, scene, spp, rounds, rows, summary, load: { min: Math.min(...loads), max: Math.max(...loads) }, loadedRuns: rows.filter((x) => x.loaded).length, failed: rows.filter((x) => !x.ok).length };
const all = existsSync(out) ? JSON.parse(readFileSync(out, 'utf8')) : [];
all.push(doc);
writeFileSync(out, JSON.stringify(all, null, 1));
console.log(`== ${name} (${scene}, ${spp} spp, ${rounds} rounds, load ${doc.load.min.toFixed(2)}..${doc.load.max.toFixed(2)}, loaded ${doc.loadedRuns}, failed ${doc.failed})`);
for (const s of summary) console.log(`  ${s.variant.padEnd(22)} ms/spp ${s.ms_min.toFixed(3)}..${s.ms_max.toFixed(3)} (med ${s.ms_med.toFixed(3)})  ratio ${s.ratio_min.toFixed(3)}..${s.ratio_max.toFixed(3)} (med ${s.ratio_med.toFixed(3)})  lum ${s.lum.toPrecision(5)}  compile ${s.compile_ms?.toFixed(0)} ms`);
