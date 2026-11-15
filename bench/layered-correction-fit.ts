// Fits the albedo-layering correction table (P2.32b, `albedoCorrection`) against the stochastic walk's sampler albedo.
//   node bench/layered-correction-fit.ts gen --ior 1.5 [--spp 100000] --out data-1.5.json   (Monte Carlo, ~3 min per ior at 100k)
//   node bench/layered-correction-fit.ts fit [--target mid|sample] data-1.33.json data-1.5.json data-2.json   (writes packages/pathtracer/src/bsdf/layered-correction-data.ts)
// Ground truth per cell (perceptual roughness r, outside cosine mu, grey base rho, ior): the walk's directional albedo from `ptLayeredSample`
// (thickness 1e-6, maxDepth 64, the Monte Carlo mean of f |cos| / pdf), minus the exact top reflection (quadrature of a black base). Model:
// the `albedoBase: 'internal'` albedo path's own base energy B_m (quadrature of `ptLayeredEval`). Correction c(r, mu, x = rho Ri(ior)) = B_walk / B_m,
// stored as a quadratic in x per (sqrt r, sqrt mu) node: c = a + b x + c x^2, weighted least squares on B_walk ~ B_m c (weights B_m^2, ridge to c = 1).
import { readFileSync, writeFileSync } from 'node:fs';
import { ptInternalReflectance, ptLayered } from '../packages/pathtracer/src/bsdf/layered.ts';
import { evalAlbedo, layeredQuadrature, sampleAlbedo } from '../packages/pathtracer/src/bsdf/layered.testutil.ts';

const NR = 8;
const NM = 8;
const RHOS = [0.2, 0.4, 0.6, 0.8, 1.0];
const arg = (name: string, d?: string): string | undefined => {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1] : d;
};

/** ws = the walk's sampler albedo, we = the albedo of its pbrt-exit-MIS `f` (cosine sampled; pbrt's own f and Sample_f disagree by up to +13%, V58), int = the internal base's albedo. */
interface Row { ir: number; im: number; rho: number; ior: number; spec: number; ws: number; we: number; int: number }

function gen(ior: number, spp: number): Row[] {
  const rows: Row[] = [];
  for (let ir = 2; ir < NR; ir++) {
    const r = (ir / (NR - 1)) ** 2;
    for (let im = 0; im < NM; im++) {
      const mu = Math.max((im / (NM - 1)) ** 2, 0.03);
      const wo: [number, number, number] = [Math.sqrt(1 - mu * mu), 0, mu];
      const spec = layeredQuadrature(ptLayered([0, 0, 0], ior, r, { thickness: 1e-6 }), wo, { layering: 'albedo' }, 300, 240).albedo[0];
      for (const rho of RHOS) {
        const p = ptLayered([rho, rho, rho], ior, r, { thickness: 1e-6 });
        const ws = sampleAlbedo(p, wo, spp, 1 + ir * 100 + im, { layering: 'stochastic', maxDepth: 64 }).mean[0];
        const we = evalAlbedo(p, wo, spp, 2 + ir * 100 + im, { layering: 'stochastic', maxDepth: 64, exitMis: 'pbrt' }).mean[0];
        const int = layeredQuadrature(p, wo, { layering: 'albedo', albedoBase: 'internal' }, 200, 180).albedo[0];
        rows.push({ ir, im, rho, ior, spec, ws, we, int });
      }
    }
  }
  return rows;
}

function solve(A: number[][], y: number[]): number[] {
  const n = y.length;
  const M = A.map((r, i) => [...r, y[i] as number]);
  for (let i = 0; i < n; i++) {
    let p = i;
    for (let k = i + 1; k < n; k++) if (Math.abs((M[k] as number[])[i] as number) > Math.abs((M[p] as number[])[i] as number)) p = k;
    [M[i], M[p]] = [M[p] as number[], M[i] as number[]];
    for (let k = i + 1; k < n; k++) {
      const f = ((M[k] as number[])[i] as number) / ((M[i] as number[])[i] as number);
      for (let j = i; j <= n; j++) (M[k] as number[])[j] = ((M[k] as number[])[j] as number) - f * ((M[i] as number[])[j] as number);
    }
  }
  const x = new Array<number>(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let s = (M[i] as number[])[n] as number;
    for (let j = i + 1; j < n; j++) s -= ((M[i] as number[])[j] as number) * (x[j] as number);
    x[i] = s / (((M[i] as number[])[i] as number));
  }
  return x;
}

/**
 * The energy the correction aims at. 'sample': the walk's sampler albedo (energy-consistent). 'mid': sampler + `lambda` (eval - sampler), lambda 0.5, but only from r >= 0.33
 * (cosine-sampled eval of a narrow lobe is too noisy below; its excess over the sampler is ~1 there anyway): what a reference-mode render reads, since next-event
 * estimates use pbrt's `f` and the bounce uses the sampler.
 */
function target(x: Row, kind: 'sample' | 'mid'): number {
  return kind === 'mid' && x.ir >= 4 ? x.ws + 0.5 * (x.we - x.ws) : x.ws;
}

/** [ir][im] -> [a, b, c]. Rows ir 0 and 1 (smooth interface: alpha below the smooth threshold) are the identity. */
export function fit(rows: Row[], kind: 'sample' | 'mid' = 'sample', lambda = 0.02): number[][][] {
  const out: number[][][] = [];
  for (let ir = 0; ir < NR; ir++) {
    out.push([]);
    for (let im = 0; im < NM; im++) {
      if (ir <= 1) { (out[ir] as number[][]).push([1, 0, 0]); continue; }
      const cells = rows.filter((x) => x.ir === ir && x.im === im);
      const A = [[0, 0, 0], [0, 0, 0], [0, 0, 0]] as number[][];
      const y = [0, 0, 0];
      let wm = 0;
      for (const x of cells) {
        const t = x.rho * ptInternalReflectance(x.ior);
        const Bm = x.int - x.spec;
        const Bw = target(x, kind) - x.spec;
        const f = [Bm, Bm * t, Bm * t * t];
        for (let i = 0; i < 3; i++) {
          for (let j = 0; j < 3; j++) (A[i] as number[])[j] = ((A[i] as number[])[j] as number) + (f[i] as number) * (f[j] as number);
          y[i] = (y[i] as number) + (f[i] as number) * Bw;
        }
        wm += Bm * Bm;
      }
      const reg = lambda * wm;
      for (let i = 0; i < 3; i++) { (A[i] as number[])[i] = ((A[i] as number[])[i] as number) + reg; y[i] = (y[i] as number) + (i === 0 ? reg : 0); }
      (out[ir] as number[][]).push(solve(A, y));
    }
  }
  return out;
}

const mode = process.argv[2];
if (mode === 'gen') {
  const ior = Number(arg('ior', '1.5'));
  writeFileSync(arg('out', `data-${ior}.json`) as string, JSON.stringify(gen(ior, Number(arg('spp', '100000')))));
} else if (mode === 'fit') {
  const files = process.argv.slice(3).filter((f) => f.endsWith('.json'));
  const rows = files.flatMap((f) => JSON.parse(readFileSync(f, 'utf8')) as Row[]);
  const kind = (arg('target', 'mid') as 'sample' | 'mid');
  const t = fit(rows, kind);
  let e = 0;
  let n = 0;
  let worst = 0;
  for (const x of rows) {
    const c = (t[x.ir] as number[][])[x.im] as number[];
    const s = x.rho * ptInternalReflectance(x.ior);
    const model = x.spec + (x.int - x.spec) * ((c[0] as number) + (c[1] as number) * s + (c[2] as number) * s * s);
    const r = (model - target(x, kind)) / target(x, kind);
    e += r * r; n++; if (Math.abs(r) > Math.abs(worst)) worst = r;
  }
  console.log(`cells ${n}, rms relative albedo error of the corrected model ${Math.sqrt(e / n).toFixed(4)}, worst ${worst.toFixed(3)}`);
  const flat = t.flatMap((row) => row.flatMap((c) => c.map((v) => Math.round(v * 1000) / 1000)));
  const lines: string[] = [];
  for (let i = 0; i < flat.length; i += NM * 3) lines.push(`  ${flat.slice(i, i + NM * 3).map((v) => (Object.is(v, -0) ? 0 : v)).join(', ')},`);
  writeFileSync(
    new URL('../packages/pathtracer/src/bsdf/layered-correction-data.ts', import.meta.url),
    `// GENERATED by bench/layered-correction-fit.ts (P2.32b): do not edit. Fit data: ${files.map((f) => f.split('/').pop()).join(', ')}; target '${kind}'.\n` +
      `// Rows = roughness nodes (sqrt r uniform, ${NR}), columns = mu nodes (sqrt mu uniform, ${NM}), then the quadratic (a, b, c) in x = base * internalReflectance.\n` +
      `export const PT_LAYER_CORR_DATA: readonly number[] = [\n${lines.join('\n')}\n];\n`,
  );
  console.log('wrote layered-correction-data.ts');
}
