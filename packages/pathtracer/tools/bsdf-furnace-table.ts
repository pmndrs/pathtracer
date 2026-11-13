/** Prints the furnace tables (brute-force integral of eval vs MC from sampling). `node packages/pathtracer/tools/bsdf-furnace-table.ts` */
import { ptConductor, ptDielectric } from '../src/bsdf/materials.ts';
import { bruteForceAlbedo, chiSquareTest, dirFromAngles, mcAlbedo } from '../src/bsdf/bsdf.testutil.ts';

const f = (x: number) => x.toFixed(4);
console.log('white conductor (eta 1e-4, k 1e4): albedo = integral f |cos|, brute force / MC standard / MC cap');
for (const th of [0, 30, 60, 80]) {
  const row = [`th ${th}`];
  for (const r of [0.05, 0.1, 0.25, 0.5, 1]) {
    const p = ptConductor([1e-4, 1e-4, 1e-4], [1e4, 1e4, 1e4], r);
    const wo = dirFromAngles(th, 25);
    const bf = bruteForceAlbedo(p, wo).refl[0];
    const a = mcAlbedo(p, wo, 200_000, 1, { vndf: 'standard' }).refl[0];
    const b = mcAlbedo(p, wo, 200_000, 1, { vndf: 'cap' }).refl[0];
    row.push(`r${r}: ${f(bf)}/${f(a)}/${f(b)}`);
  }
  console.log(row.join('  '));
}
console.log('rough dielectric ior 1.5, importance transport: R+T brute force (MC standard)');
for (const th of [0, 30, 60, 80, 130, 160]) {
  const row = [`th ${th}`];
  for (const r of [0.05, 0.1, 0.25, 0.5, 1]) {
    const p = ptDielectric(1.5, r);
    const wo = dirFromAngles(th, 25);
    const bf = bruteForceAlbedo(p, wo, { transport: 'importance' });
    const mc = mcAlbedo(p, wo, 200_000, 1, { transport: 'importance' });
    row.push(`r${r}: ${f(bf.refl[0] + bf.trans[0])} (${f(mc.refl[0] + mc.trans[0])})`);
  }
  console.log(row.join('  '));
}
console.log('rough dielectric ior 1.5, radiance transport: R + T/etap^2');
for (const th of [0, 60, 130]) {
  const row = [`th ${th}`];
  for (const r of [0.1, 0.5, 1]) {
    const p = ptDielectric(1.5, r);
    const wo = dirFromAngles(th, 25);
    const bf = bruteForceAlbedo(p, wo, { transport: 'radiance' });
    row.push(`r${r}: ${f(bf.refl[0] + bf.trans[0])}`);
  }
  console.log(row.join('  '));
}
console.log('chi-square (400k samples): stat / dof / p');
for (const vndf of ['standard', 'cap'] as const) {
  const r = chiSquareTest(ptConductor([0.143, 0.375, 1.442], [3.983, 2.386, 1.603], 0.5), dirFromAngles(40, 30), 400_000, 1234, { vndf });
  const d = chiSquareTest(ptDielectric(1.5, 0.5), dirFromAngles(150, 30), 400_000, 1234, { vndf });
  console.log(vndf, 'conductor', r.stat.toFixed(0), r.dof, r.p.toFixed(3), '| dielectric inside', d.stat.toFixed(0), d.dof, d.p.toFixed(3));
}
