// Prints where the megakernel's minified WGSL bytes go, per top-level declaration. Usage:
//   node bench/size-breakdown.ts [--env] [--top 40]
import { buildCaps } from '@pmndrs/raytracer';
import { minifyWgsl, ptMegakernelWgsl } from '@pmndrs/pathtracer';

const args = process.argv.slice(2);
const env = args.includes('--env');
const top = Number(args[args.indexOf('--top') + 1] ?? 40) || 40;
const caps = buildCaps({ features: new Set(['subgroups', 'shader-f16', 'timestamp-query']), limits: {} }, new Set());
const layout = { normals: true, uvs: false, positions: true };
const integrator = env ? { envMap: true, envSampling: 'importance' as const } : {};
// Readable names (so the groups can be told apart); the shipped total is with shortened names.
const plain = ptMegakernelWgsl(caps, { format: {}, layout, integrator, minify: false, shortenNames: false });
const total = ptMegakernelWgsl(caps, { format: {}, layout, integrator }).length;
const unshortened = ptMegakernelWgsl(caps, { format: {}, layout, integrator, shortenNames: false }).length;
const lines = plain.split('\n');
const decl = /^(?:@[^\n]*?\s)?(?:struct|fn|const|alias|override|var(?:<[^>]*>)?)\s+([A-Za-z_]\w*)/;
const chunks: { name: string; text: string }[] = [];
let cur = { name: '(prelude)', text: '' };
for (const l of lines) {
  const m = decl.exec(l);
  if (m) { chunks.push(cur); cur = { name: m[1] as string, text: '' }; }
  cur.text += `${l}\n`;
}
chunks.push(cur);
const groups = new Map<string, number>();
const per: [string, number][] = [];
for (const c of chunks) {
  const n = minifyWgsl(c.text).length;
  per.push([c.name, n]);
  const g = /^(bvh|rt[A-Z]|RT_|Rt|pt[A-Z]|PT_|Pt)/.exec(c.name)?.[1]?.replace(/[A-Z]$/, '') ?? 'other';
  groups.set(g, (groups.get(g) ?? 0) + n);
}
console.log(`shipped ${total} bytes (pruned, minified, shortened names); ${unshortened} without shortened names; chunks below are the readable pruned source, sum ${per.reduce((a, b) => a + b[1], 0)}`);
console.log([...groups].sort((a, b) => b[1] - a[1]).map(([g, n]) => `${g}: ${n}`).join('  '));
for (const [n, b] of per.sort((a, b) => b[1] - a[1]).slice(0, top)) console.log(String(b).padStart(6), n);
