/**
 * Scene -> Mitsuba 3 XML + PLY meshes (+ env PFM).
 *
 * Camera: right-handed `lookat`, `fov_axis` set explicitly to y so `yFovDeg` is used verbatim. Box filter radius 0.5.
 * maxDepth: Mitsuba's `path` integrator counts a camera ray that hits an emitter as depth 1, so direct lighting needs
 * max_depth 2: max_depth = maxDepth + 1 (verified with the quad-light scene, see bench/references/README.md).
 * Russian roulette: with no `rrStartDepth` we set `rr_depth` far above max_depth so RR never fires. Otherwise
 * rr_depth = rrStartDepth: RR first fires at path vertex number rr_depth (1 = first surface hit; rr_depth must be > 0),
 * verified by variance with rr_depth 1/2/3 against pbrt's `rrdepth` 0/1/2 (pbrt rrdepth = Mitsuba rr_depth - 1). The
 * earlier "+ 1" here was unverified and made RR start one vertex late (only noise differed: RR is unbiased).
 * Mitsuba BSDFs are one-sided, so diffuse/conductor are wrapped in `twosided` to match pbrt (two-sided BSDFs).
 * Area emitters are one-sided (front = +normal); `twoSided` emission is emulated with a second, flipped copy.
 */
import type { MaterialDesc } from '../scenes/schema.ts';
import { encodePly, flipMesh } from './ply.ts';
import { loadEnvImage } from './envmap.ts';
import { writePFM } from '../metrics/pfm.ts';
import { num, roughnessToAlpha, worldMeshes } from './common.ts';
import type { Exporter, ExportedFile } from './types.ts';

const rgb = (name: string, v: readonly number[]) => `<rgb name="${name}" value="${v.join(', ')}"/>`;

/**
 * eta/k for conductors. A grey RGB triple is exported as a constant `spectrum`: Mitsuba's spectral variants uplift
 * `rgb` through a sigmoid polynomial and k = 1000 collapses to a dim, wrong spectrum (furnace read 0.15 instead of
 * 0.69), while a constant spectrum is exact in every variant. Coloured eta/k (copper) stays `rgb` and is only an
 * approximation in the spectral variants.
 */
const optical = (name: string, v: readonly number[]) =>
  v[0] === v[1] && v[1] === v[2] ? `<spectrum name="${name}" value="${v[0]}"/>` : rgb(name, v);

function bsdf(m: MaterialDesc, indent: string): string {
  const i2 = indent + '  ';
  const wrap = (inner: string) => `${indent}<bsdf type="twosided">\n${i2}${inner}\n${indent}</bsdf>`;
  switch (m.type) {
    case 'diffuse':
      return wrap(`<bsdf type="diffuse">${rgb('reflectance', m.reflectance)}</bsdf>`);
    case 'black':
      return wrap(`<bsdf type="diffuse">${rgb('reflectance', [0, 0, 0])}</bsdf>`);
    case 'conductor': {
      const a = roughnessToAlpha(m.roughness);
      if (a === 0) return wrap(`<bsdf type="conductor">${optical('eta', m.eta)}${optical('k', m.k)}</bsdf>`);
      return wrap(
        `<bsdf type="roughconductor"><string name="distribution" value="ggx"/><float name="alpha" value="${num(a)}"/>${optical('eta', m.eta)}${optical('k', m.k)}</bsdf>`,
      );
    }
    case 'dielectric': {
      const a = roughnessToAlpha(m.roughness);
      const ior = `<float name="int_ior" value="${num(m.ior)}"/><float name="ext_ior" value="1.0"/>`;
      if (a === 0) return `${indent}<bsdf type="dielectric">${ior}</bsdf>`;
      return `${indent}<bsdf type="roughdielectric"><string name="distribution" value="ggx"/><float name="alpha" value="${num(a)}"/>${ior}</bsdf>`;
    }
  }
}

export const toMitsuba: Exporter = (scene, opts) => {
  const approximations: string[] = [];
  const files: ExportedFile[] = [];
  const c = scene.camera;
  const noRr = scene.integrator.rrStartDepth === undefined;
  const maxDepth = scene.integrator.maxDepth + 1;
  const rrDepth = noRr ? 1_000_000 : Math.max(1, scene.integrator.rrStartDepth!);
  const x: string[] = [];
  x.push(`<scene version="3.0.0">`);
  x.push(`  <!-- ${scene.name}: ${scene.purpose.replace(/--/g, '-')} -->`);
  x.push(`  <integrator type="path">`);
  x.push(`    <integer name="max_depth" value="${maxDepth}"/>`);
  x.push(`    <integer name="rr_depth" value="${rrDepth}"/>`);
  x.push(`  </integrator>`);
  x.push(`  <sensor type="perspective">`);
  x.push(`    <float name="fov" value="${num(c.yFovDeg)}"/>`);
  x.push(`    <string name="fov_axis" value="y"/>`);
  x.push(`    <float name="near_clip" value="0.001"/>`);
  x.push(`    <float name="far_clip" value="10000"/>`);
  x.push(`    <transform name="to_world"><lookat origin="${c.position.join(', ')}" target="${c.target.join(', ')}" up="${c.up.join(', ')}"/></transform>`);
  x.push(`    <sampler type="multijitter"><integer name="sample_count" value="${opts.spp}"/></sampler>`);
  x.push(`    <film type="hdrfilm"><integer name="width" value="${scene.film.width}"/><integer name="height" value="${scene.film.height}"/><string name="pixel_format" value="rgb"/><string name="component_format" value="float32"/><rfilter type="box"/></film>`);
  x.push(`  </sensor>`);

  const env = scene.environment;
  if (env?.type === 'uniform') {
    x.push(`  <emitter type="constant">${rgb('radiance', env.radiance)}</emitter>`);
  } else if (env?.type === 'hdri') {
    const img = loadEnvImage(opts.assetsDir, env.path);
    files.push({ path: 'env.pfm', data: Buffer.from(writePFM(img)) });
    x.push(`  <emitter type="envmap">`);
    x.push(`    <string name="filename" value="env.pfm"/>`);
    x.push(`    <float name="scale" value="${num(env.scale)}"/>`);
    x.push(`    <transform name="to_world"><rotate y="1" angle="${num(env.rotationYDeg)}"/></transform>`);
    x.push(`  </emitter>`);
  }
  for (const l of scene.lights ?? []) {
    x.push(`  <emitter type="directional"><vector name="direction" value="${l.direction.join(', ')}"/>${rgb('irradiance', l.irradiance)}</emitter>`);
    approximations.push('mitsuba: directional light angularRadius is ignored (delta direction).');
  }

  for (const w of worldMeshes(scene, opts.assetsDir)) {
    const m = scene.materials[w.desc.material];
    if (!m) throw new Error(`mesh '${w.desc.name}': unknown material '${w.desc.material}'`);
    files.push({ path: `${w.stem}.ply`, data: encodePly(w.mesh) });
    const emitter = (rad: readonly number[]) => `    <emitter type="area">${rgb('radiance', rad)}</emitter>`;
    x.push(`  <shape type="ply">`);
    x.push(`    <string name="filename" value="${w.stem}.ply"/>`);
    x.push(bsdf(m, '    '));
    if (w.desc.emission) x.push(emitter(w.desc.emission));
    x.push(`  </shape>`);
    if (w.desc.emission && w.desc.twoSided) {
      files.push({ path: `${w.stem}-back.ply`, data: encodePly(flipMesh(w.mesh)) });
      x.push(`  <shape type="ply">`);
      x.push(`    <string name="filename" value="${w.stem}-back.ply"/>`);
      x.push(bsdf(m, '    '));
      x.push(emitter(w.desc.emission));
      x.push(`  </shape>`);
      approximations.push(`mitsuba: two-sided emitter '${w.desc.name}' exported as two coincident one-sided emitters (front + flipped copy); light sampling sees 2x the emitter area.`);
    }
  }
  x.push(`</scene>`);
  files.push({ path: 'scene.xml', data: x.join('\n') + '\n' });
  return { sceneFile: 'scene.xml', files, approximations };
};
