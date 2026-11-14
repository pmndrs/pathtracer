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
 * `openpbr` materials go through `mitsubaOpenPbr`; a scene with an absorbing interior medium uses `volpath` (same depth rules).
 */
import type { MaterialDesc } from '../scenes/schema.ts';
import { encodePly, flipMesh } from './ply.ts';
import { equirectForMitsuba, loadEnvImage } from './envmap.ts';
import { writePFM } from '../metrics/pfm.ts';
import { num, roughnessToAlpha, usedTextures, worldMeshes } from './common.ts';
import { encodePng } from './png.ts';
import { deriveOpenPbr, reflectanceToEtaK } from './openpbr.ts';
import type { HomogeneousAbsorber, NativeCtx } from './openpbr.ts';
import type { Exporter, ExportedFile } from './types.ts';
import type { OpenPbrParams, Rgb } from '../scenes/schema.ts';

/** Does Mitsuba's bitmap lookup want v = 0 at the bottom of the image (like pbrt)? Verified with an asymmetric texture, see bench/references/README.md. */
const MITSUBA_FLIP_V = false;

const rgb = (name: string, v: readonly number[]) => `<rgb name="${name}" value="${v.join(', ')}"/>`;

/**
 * eta/k for conductors. A grey RGB triple is exported as a constant `spectrum`: Mitsuba's spectral variants uplift
 * `rgb` through a sigmoid polynomial and k = 1000 collapses to a dim, wrong spectrum (furnace read 0.15 instead of
 * 0.69), while a constant spectrum is exact in every variant. Coloured eta/k (copper) stays `rgb` and is only an
 * approximation in the spectral variants.
 */
const optical = (name: string, v: readonly number[]) =>
  v[0] === v[1] && v[1] === v[2] ? `<spectrum name="${name}" value="${v[0]}"/>` : rgb(name, v);

function bsdf(m: MaterialDesc, indent: string, textureFile: (name: string) => string): string {
  const i2 = indent + '  ';
  const wrap = (inner: string) => `${indent}<bsdf type="twosided">\n${i2}${inner}\n${indent}</bsdf>`;
  switch (m.type) {
    case 'pbr':
      throw new Error('mitsuba: pbr materials are not expressible');
    case 'openpbr':
      throw new Error('mitsuba: openpbr materials go through mitsubaOpenPbr');
    case 'diffuse':
      if (m.reflectanceTexture) {
        return wrap(`<bsdf type="diffuse"><texture type="bitmap" name="reflectance"><string name="filename" value="${textureFile(m.reflectanceTexture)}"/><string name="filter_type" value="bilinear"/><string name="wrap_mode" value="repeat"/><boolean name="raw" value="false"/></texture></bsdf>`);
      }
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

export interface MitsubaOpenPbr {
  /** `<bsdf>` XML (already wrapped in `twosided` where the leaf is one-sided). */
  xml: string;
  medium?: HomogeneousAbsorber;
  emission?: Rgb;
}

/**
 * `openpbr` -> Mitsuba 3 BSDF XML (docs/design/tier1.md section 2):
 *  - metalness 1: `conductor` / `roughconductor` (eta 1, k from the reflectance; anisotropic `alpha_u`/`alpha_v`);
 *  - metalness 0: transmission 1 -> `dielectric` / `roughdielectric`, thin-walled smooth -> `thindielectric`; transmission 0 and a base
 *    colour -> `plastic` / `roughplastic` with `nonlinear true` (approximate: Mitsuba's closed-form layering, not pbrt's stochastic one);
 *  - fractional metalness / transmission / coat weight -> `blendbsdf` (`weight` = fraction of the second child);
 *  - a clear coat on a dielectric is `roughplastic` with the coat IOR/roughness; a coat on a metal has no Mitsuba equivalent
 *    (`principled` clearcoat, Disney, only with `coverage`); thin-walled diffuse transmission and dispersion do not exist
 *    (pbrt only): throw, or with `coverage` write the closest approximation;
 *  - `transmission_depth` > 0 with a transmission colour -> an absorption-only homogeneous interior medium (`volpath`).
 */
export function mitsubaOpenPbr(matName: string, p: OpenPbrParams, ctx: NativeCtx): MitsubaOpenPbr {
  const d = deriveOpenPbr(matName, p, ctx);
  const { m } = d;
  const tag = `mitsuba: material '${matName}'`;
  const note = (s: string): void => void ctx.notes.add(`${tag}: ${s}`);
  const need = (what: string, fallback: string): void => {
    if (!ctx.coverage) throw new Error(`${tag}: ${what} has no Mitsuba equivalent; pass the 'coverage' option to write ${fallback} (flagged, not a reference)`);
    ctx.notes.add(`coverage: ${tag}: ${what} has no Mitsuba equivalent; wrote ${fallback}`);
  };
  const two = (inner: string): string => `<bsdf type="twosided">${inner}</bsdf>`;
  const ior = (n: number): string => `<float name="int_ior" value="${num(n)}"/><float name="ext_ior" value="1.0"/>`;
  const ggx = (a: number): string => `<string name="distribution" value="ggx"/><float name="alpha" value="${num(a)}"/>`;
  const blend = (a: string, b: string, w: number): string => (w <= 0 ? a : w >= 1 ? b : `<bsdf type="blendbsdf"><float name="weight" value="${num(w)}"/>${a}${b}</bsdf>`);

  const metalLeaf = (): string => {
    if (m.specularColor.some((c) => c !== 1)) note('specular_color != 1 on a metal (F82 tint) is ignored: reflectance = base_color is the F0');
    const { eta, k } = reflectanceToEtaK(d.base);
    if (m.coatWeight > 0) {
      need('a clear coat on a metal (pbrt coatedconductor)', "`principled` with clearcoat (Disney lobes, metallic 1)");
      return two(`<bsdf type="principled">${rgb('base_color', d.base)}<float name="metallic" value="1.0"/><float name="roughness" value="${num(m.specularRoughness)}"/><float name="clearcoat" value="${num(m.coatWeight)}"/><float name="clearcoat_gloss" value="${num(1 - m.coatRoughness)}"/></bsdf>`);
    }
    if (d.alphaU === 0 && d.alphaV === 0) return two(`<bsdf type="conductor">${optical('eta', eta)}${optical('k', k)}</bsdf>`);
    const alpha = d.anisotropic ? `<float name="alpha_u" value="${num(d.alphaU)}"/><float name="alpha_v" value="${num(d.alphaV)}"/>` : `<float name="alpha" value="${num(d.alphaU)}"/>`;
    return two(`<bsdf type="roughconductor"><string name="distribution" value="ggx"/>${alpha}${optical('eta', eta)}${optical('k', k)}</bsdf>`);
  };
  const plastic = (n: number, alpha: number): string =>
    two(alpha === 0 ? `<bsdf type="plastic">${ior(n)}${rgb('diffuse_reflectance', d.base)}<boolean name="nonlinear" value="true"/></bsdf>` : `<bsdf type="roughplastic">${ggx(alpha)}${ior(n)}${rgb('diffuse_reflectance', d.base)}<boolean name="nonlinear" value="true"/></bsdf>`);
  const diffuseLeaf = (): string => two(`<bsdf type="diffuse">${rgb('reflectance', d.base)}</bsdf>`);
  const opaqueLeaf = (): string => {
    if (m.subsurfaceWeight > 0) {
      need('thin-walled diffuse transmission (subsurface_weight > 0)', 'the diffuse reflection only (no transmitted light)');
      return diffuseLeaf();
    }
    if (m.coatWeight > 0) {
      note('clear coat on a dielectric: roughplastic (nonlinear) with the coat IOR/roughness; the base dielectric specular lobe is not added');
      const c = plastic(m.coatIor, d.coatAlpha);
      if (m.coatWeight >= 1) return c;
      note(`coat_weight ${m.coatWeight} < 1: linear blend of the uncoated dielectric and the coated one`);
      return blend(m.specularWeight > 0 && d.ior !== 1 ? plastic(d.ior, d.alphaU) : diffuseLeaf(), c, m.coatWeight);
    }
    if (m.specularWeight <= 0 || d.ior === 1) return diffuseLeaf();
    if (d.anisotropic) note('anisotropic dielectric over diffuse: roughplastic has an isotropic alpha (u alpha used)');
    note('dielectric over diffuse is roughplastic with nonlinear true (closed-form internal scattering); pbrt coateddiffuse is the stochastic layered reference and differs by a few percent');
    return plastic(d.ior, d.alphaU);
  };
  const glassLeaf = (): string => {
    if (m.transmissionDispersionScale > 0) need('dispersion (transmission_dispersion_scale > 0)', `a constant IOR of ${d.ior} (no dispersion)`);
    if (d.thin) {
      if (d.alphaU > 0 || d.alphaV > 0) ctx.notes.add(`coverage: ${tag}: rough thin-walled transmission has no reference: smooth thindielectric written`);
      return `<bsdf type="thindielectric">${ior(d.ior)}</bsdf>`;
    }
    if (d.alphaU === 0 && d.alphaV === 0) return `<bsdf type="dielectric">${ior(d.ior)}</bsdf>`;
    const alpha = d.anisotropic ? `<float name="alpha_u" value="${num(d.alphaU)}"/><float name="alpha_v" value="${num(d.alphaV)}"/>` : `<float name="alpha" value="${num(d.alphaU)}"/>`;
    return `<bsdf type="roughdielectric"><string name="distribution" value="ggx"/>${alpha}${ior(d.ior)}</bsdf>`;
  };
  const dielectric = (): string => {
    if (!d.transmissive) return opaqueLeaf();
    const g = glassLeaf();
    if (m.transmissionWeight >= 1) return g;
    note(`transmission_weight ${m.transmissionWeight} < 1: linear blend of the opaque dielectric and the glass (OpenPBR layers them)`);
    return blend(opaqueLeaf(), g, m.transmissionWeight);
  };
  const xml = d.metal >= 1 ? metalLeaf() : d.metal <= 0 ? dielectric() : blend(dielectric(), metalLeaf(), d.metal);
  return { xml, medium: d.metal < 1 ? d.medium : undefined, emission: d.emission };
}

export const toMitsuba: Exporter = (scene, opts) => {
  const approximations: string[] = [];
  const files: ExportedFile[] = [];
  const c = scene.camera;
  const ctx: NativeCtx = { renderer: 'mitsuba', coverage: opts.coverage === true, layered: { maxDepth: 0, nSamples: 0 }, notes: new Set() };
  const openpbr = new Map<string, MitsubaOpenPbr>();
  for (const w of scene.meshes) {
    const m = scene.materials[w.material];
    if (m?.type === 'openpbr' && !openpbr.has(w.material)) openpbr.set(w.material, mitsubaOpenPbr(w.material, m, ctx));
  }
  const volumetric = [...openpbr.values()].some((o) => o.medium);
  const noRr = scene.integrator.rrStartDepth === undefined;
  const maxDepth = scene.integrator.maxDepth + 1;
  const rrDepth = noRr ? 1_000_000 : Math.max(1, scene.integrator.rrStartDepth!);
  const x: string[] = [];
  x.push(`<scene version="3.0.0">`);
  x.push(`  <!-- ${scene.name}: ${scene.purpose.replace(/--/g, '-')} -->`);
  if (volumetric) approximations.push('mitsuba: absorption-only interior media use the `volpath` integrator (same max_depth and rr_depth rules as `path`); homogeneous medium with sigma_t = -ln(T)/depth per RGB channel (written as scale = max sigma and rgb = sigma / max sigma, because an `rgb` medium value must lie in [0, 1]), albedo 0.');
  x.push(`  <integrator type="${volumetric ? 'volpath' : 'path'}">`);
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
    const img = equirectForMitsuba(loadEnvImage(opts.assetsDir, env.path));
    files.push({ path: 'env.pfm', data: Buffer.from(writePFM(img)) });
    approximations.push('mitsuba: envmap rows resampled to v = j / (H - 1) because Mitsuba looks the map up at v * (H - 1) (corner-aligned rows; unfixed it reads the sun 0.15 degrees off and +0.4% bright on env-sphere).');
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

  const textures = usedTextures(scene, 'mitsuba');
  for (const t of textures.values()) {
    if (t.wrap !== 'repeat') throw new Error(`mitsuba: texture '${t.name}' wrap '${t.wrap}' is not exported (the bench textures repeat)`);
    files.push({ path: t.file, data: encodePng(t.width, t.height, t.data) });
  }
  if (textures.size > 0) approximations.push('mitsuba: bitmap textures with filter_type bilinear (level 0 only, no mip maps), wrap repeat, raw false (sRGB decoded to linear before filtering).');
  const textureFile = (name: string) => (textures.get(name) as { file: string }).file;
  for (const w of worldMeshes(scene, opts.assetsDir)) {
    const m = scene.materials[w.desc.material];
    if (!m) throw new Error(`mesh '${w.desc.name}': unknown material '${w.desc.material}'`);
    files.push({ path: `${w.stem}.ply`, data: encodePly(w.mesh, { flipV: MITSUBA_FLIP_V }) });
    const emitter = (rad: readonly number[]) => `    <emitter type="area">${rgb('radiance', rad)}</emitter>`;
    x.push(`  <shape type="ply">`);
    x.push(`    <string name="filename" value="${w.stem}.ply"/>`);
    const o = m.type === 'openpbr' ? openpbr.get(w.desc.material)! : undefined;
    let emission = w.desc.emission;
    let twoSided = w.desc.twoSided === true;
    if (o?.emission) {
      if (emission) throw new Error(`mesh '${w.desc.name}': both the mesh and its openpbr material emit; use one`);
      emission = o.emission;
      twoSided = m.type === 'openpbr' && m.emissionTwoSided === true;
    }
    const shapeBsdf = o ? `    ${o.xml}` : bsdf(m, '    ', textureFile);
    // `rgb` on a spectrum-typed medium parameter is read as a reflectance and must lie in [0, 1]: sigma_t = scale * rgb with rgb = sigma / max(sigma)
    const sMax = o?.medium ? Math.max(...o.medium.sigma) : 0;
    const medium = o?.medium ? `    <medium name="interior" type="homogeneous">${rgb('sigma_t', o.medium.sigma.map((v) => Number((v / sMax).toPrecision(9))))}${rgb('albedo', [0, 0, 0])}<float name="scale" value="${sMax}"/></medium>` : undefined;
    x.push(shapeBsdf);
    if (medium) x.push(medium);
    if (emission) x.push(emitter(emission));
    x.push(`  </shape>`);
    if (emission && twoSided) {
      files.push({ path: `${w.stem}-back.ply`, data: encodePly(flipMesh(w.mesh)) });
      x.push(`  <shape type="ply">`);
      x.push(`    <string name="filename" value="${w.stem}-back.ply"/>`);
      x.push(shapeBsdf);
      if (medium) x.push(medium);
      x.push(emitter(emission));
      x.push(`  </shape>`);
      approximations.push(`mitsuba: two-sided emitter '${w.desc.name}' exported as two coincident one-sided emitters (front + flipped copy); light sampling sees 2x the emitter area.`);
    }
  }
  x.push(`</scene>`);
  approximations.push(...ctx.notes);
  files.push({ path: 'scene.xml', data: x.join('\n') + '\n' });
  return { sceneFile: 'scene.xml', files, approximations };
};
