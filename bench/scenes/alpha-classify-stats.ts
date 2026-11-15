/**
 * R2b: how many triangles of a scene's cutout cards are transparent / solid / mixed (the opacity-micromap-style 2-state classification of `alpha-classify.ts`),
 * and the per-material mean coverage.
 *
 *   node bench/scenes/alpha-classify-stats.ts [scene ...]      (default: tex-mask foliage-mask foliage-mask-dense foliage-mask-grid16 foliage-mask-grid12)
 */
import { alphaCoverageOf, classifyAlphaCards } from './alpha-classify.ts';
import { getScene } from './index.ts';
import { sceneDescToCopyInput } from './to-scene-copy.ts';

const names = process.argv.slice(2);
for (const name of names.length > 0 ? names : ['tex-mask', 'foliage-mask', 'foliage-mask-dense', 'foliage-mask-grid16', 'foliage-mask-grid12']) {
  const scene = getScene(name);
  const input = sceneDescToCopyInput(scene);
  const { stats } = classifyAlphaCards(input);
  const total = stats.reduce((a, s) => ({ t: a.t + s.triangles, tr: a.tr + s.transparent, so: a.so + s.solid, mx: a.mx + s.mixed }), { t: 0, tr: 0, so: 0, mx: 0 });
  const cov = alphaCoverageOf(input);
  console.log(`${name}: ${stats.length} cutout (mesh, material) pairs, ${total.t} triangles: transparent ${total.tr} (${((100 * total.tr) / Math.max(1, total.t)).toFixed(1)}%), solid ${total.so} (${((100 * total.so) / Math.max(1, total.t)).toFixed(1)}%), mixed ${total.mx} (${((100 * total.mx) / Math.max(1, total.t)).toFixed(1)}%); coverage by material: ${input.materialNames.map((n, i) => `${n}=${(cov[i] as number).toFixed(3)}`).join(' ')}`);
}
