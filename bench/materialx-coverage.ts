// MaterialX coverage report (P5): how many documents of a directory tree load (every node and shader supported) and why the rest do not.
//   node bench/materialx-coverage.ts [dir ...]     default: the maintainer's sibling checkouts (materialx-fidelity samples, MaterialX resources)
import { readdirSync, readFileSync, statSync, existsSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { loadMtlxMaterial } from '@pmndrs/pathtracer';

function walk(dir: string, out: string[] = []): string[] {
  for (const f of readdirSync(dir)) {
    const p = join(dir, f);
    if (statSync(p).isDirectory()) walk(p, out);
    else if (f.endsWith('.mtlx')) out.push(p);
  }
  return out;
}

const defaults = [join(homedir(), 'Developer/materialx-fidelity/third_party/materialx-samples/materials'), join(homedir(), 'Developer/MaterialX/resources/Materials/Examples/StandardSurface')];
const roots = (process.argv.length > 2 ? process.argv.slice(2) : defaults).filter((d) => existsSync(d));
let ok = 0;
let varying = 0;
const reasons = new Map<string, number>();
const categories = new Map<string, number>();
for (const root of roots) {
  for (const path of walk(root)) {
    try {
      const m = loadMtlxMaterial(readFileSync(path, 'utf8'));
      ok++;
      if (Object.keys(m.varying).length > 0 || m.normal) varying++;
      for (const c of m.categories) categories.set(c, (categories.get(c) ?? 0) + 1);
    } catch (e) {
      const msg = (e as Error).message.replace(/'[^']*'/g, "'_'").slice(0, 100);
      reasons.set(msg, (reasons.get(msg) ?? 0) + 1);
    }
  }
}
const total = ok + [...reasons.values()].reduce((a, b) => a + b, 0);
console.log(`${ok} of ${total} documents load (${varying} with spatially varying inputs)`);
console.log('nodes used by the loadable varying ones:', [...categories].sort((a, b) => b[1] - a[1]).map(([c, n]) => `${c} ${n}`).join(', '));
console.log('why the rest do not:');
for (const [msg, n] of [...reasons].sort((a, b) => b[1] - a[1]).slice(0, 25)) console.log(`  ${String(n).padStart(3)}  ${msg}`);
