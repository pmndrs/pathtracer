/**
 * Scene -> pbrt-v4 (.pbrt + binary PLY meshes + octahedral env image).
 *
 * Handedness: pbrt is left-handed. We keep world coordinates unchanged and prepend `Scale -1 1 1` to the camera
 * transform, which un-mirrors the image (left-handed LookAt puts +x on the image's left for a right-handed scene).
 * `fov` in pbrt is the field of view of the SHORTER image axis. Pixel filter is a box with radius 0.5.
 * Roughness: material roughness is perceptual, alpha = roughness²; we write alpha with `remaproughness false`.
 * Area lights are one-sided unless `twoSided`. Russian roulette is disabled by the patched `bool rr false`
 * (bench/references/pbrt-no-rr.patch; stock pbrt-v4 cannot disable RR) when `rrStartDepth` is absent. Otherwise the
 * patch's `"integer rrdepth"` is set to rrStartDepth - 1: pbrt applies RR when `depth > rrdepth`, so RR first fires at
 * path vertex number rrStartDepth (1 = first surface hit); verified by variance with rrdepth 0/1/2 (stock = 1). The patch covers `volpath` too.
 * `openpbr` materials become named materials (`pbrtOpenPbr`); a scene with an absorbing interior uses `volpath` + `MakeNamedMedium`.
 */
import type { MaterialDesc } from '../scenes/schema.ts';
import { encodePly } from './ply.ts';
import { equirectToOctahedral, loadEnvImage } from './envmap.ts';
import { writePFM } from '../metrics/pfm.ts';
import { num, roughnessToAlpha, shorterAxisFovDeg, usedTextures, vec, worldMeshes } from './common.ts';
import { encodePng } from './png.ts';
import { pbrtMedium } from './media.ts';
import { srgbToLinear, TextureSet } from './textures.ts';
import { cauchyFit, deriveOpenPbr, LAYERED_DEFAULTS, LAYERED_THICKNESS } from './openpbr.ts';
import type { HomogeneousAbsorber, NativeCtx } from './openpbr.ts';
import type { Exporter, ExportedFile } from './types.ts';
import type { OpenPbrParams, Rgb } from '../scenes/schema.ts';

/** pbrt octahedral environment resolution (square). 512² covers the 512×256 equirect with margin. */
const OCTA_SIZE = 512;

function material(m: MaterialDesc, textureName: (name: string) => string): string {
  switch (m.type) {
    case 'pbr':
      throw new Error('pbrt: pbr materials are not expressible');
    case 'openpbr':
      throw new Error('pbrt: openpbr materials go through pbrtOpenPbr (named materials)');
    case 'diffuse':
      if (m.reflectanceTexture) return `Material "diffuse" "texture reflectance" "${textureName(m.reflectanceTexture)}"`;
      return `Material "diffuse" "rgb reflectance" [ ${vec(m.reflectance)} ]`;
    case 'black':
      return `Material "diffuse" "rgb reflectance" [ 0 0 0 ]`;
    case 'interface':
      return `Material "interface"`;
    case 'conductor':
      return `Material "conductor" "rgb eta" [ ${vec(m.eta)} ] "rgb k" [ ${vec(m.k)} ] "float roughness" ${num(roughnessToAlpha(m.roughness))} "bool remaproughness" false`;
    case 'dielectric':
      return `Material "dielectric" "float eta" ${num(m.ior)} "float roughness" ${num(roughnessToAlpha(m.roughness))} "bool remaproughness" false`;
  }
}

/** One pbrt material node: a leaf (`type` + parameters) or a `mix` of two named nodes (prob. `amount` of the second). */
type Node = { type: string; params: string } | { type: 'mix'; a: string; b: string; amount: number | string };

export interface PbrtOpenPbr {
  /** `MakeNamedMaterial` lines, dependencies first. */
  decls: string[];
  /** Name of the top node (use as `NamedMaterial "<use>"`). */
  use: string;
  medium?: HomogeneousAbsorber;
  emission?: Rgb;
  /** `Texture` directives the named materials and the shape refer to (emit before `decls`). */
  textures: string[];
  /** Extra `Shape` parameters (the alpha cutout), with a leading space. */
  shapeParams: string;
  /** Emission from an image (linear RGB PFM, radiance already scaled): `AreaLightSource "diffuse" "string filename"` instead of `rgb L`. */
  emissionFile?: string;
}

/** Alpha mask gain: pbrt's alpha is stochastic between 0 and 1, so the interpolated alpha is amplified until the transition is narrower than an 8-bit step. */
const MASK_GAIN = 1000;

const f = (name: string, v: number): string => `"float ${name}" ${num(v)}`;
const rgbP = (name: string, v: readonly number[]): string => `"rgb ${name}" [ ${vec(v)} ]`;

/**
 * `openpbr` -> pbrt-v4 named materials (docs/design/tier1.md section 2):
 *  - metalness 1: `conductor` (`reflectance` = base colour, eta 1 + complex k; anisotropic alphas via u/v roughness), with `coatedconductor`
 *    when `coat_weight` > 0 (interface eta = coat IOR);
 *  - metalness 0: transmission 1 -> `dielectric` (or `thindielectric` when thin-walled and smooth); transmission 0 -> `coateddiffuse`
 *    (specular lobe = the coat interface; `diffuse` when the specular is off); thin-walled `subsurface` -> `diffusetransmission`;
 *  - fractional metalness / transmission / coat weight -> `mix` (exact linear blend of the two ends);
 *  - dispersion -> `spectrum eta` from a Cauchy fit of (specular_ior, Abbe number / scale);
 *  - `transmission_depth` > 0 with a transmission colour -> an absorption-only interior medium (the caller emits `MakeNamedMedium`, a
 *    `MediumInterface` and switches to `volpath`).
 */
export function pbrtOpenPbr(matName: string, p: OpenPbrParams, ctx: NativeCtx): PbrtOpenPbr {
  const d = deriveOpenPbr(matName, p, ctx);
  const { m } = d;
  const note = (s: string): void => void ctx.notes.add(`pbrt: material '${matName}': ${s}`);
  const cov = (s: string): void => void ctx.notes.add(`coverage: pbrt: material '${matName}': ${s}`);
  const stem = matName.replace(/[^A-Za-z0-9_-]+/g, '_');
  const nodes: Array<[string, Node]> = [];
  let counter = 0;
  const add = (label: string, n: Node): string => {
    const name = `${stem}.${label}${counter++}`;
    nodes.push([name, n]);
    return name;
  };
  const mix = (a: string, b: string, amount: number | string): string => (typeof amount === 'number' && amount <= 0 ? a : typeof amount === 'number' && amount >= 1 ? b : add('mix', { type: 'mix', a, b, amount }));
  const rough = (prefix: string, uAlpha: number, vAlpha: number): string =>
    roughTex ? `"texture ${prefix}roughness" "${roughTex}"` : uAlpha === vAlpha ? `${f(`${prefix}roughness`, uAlpha)}` : `${f(`${prefix}uroughness`, uAlpha)} ${f(`${prefix}vroughness`, vAlpha)}`;
  const layer = `"integer maxdepth" ${ctx.layered.maxDepth} "integer nsamples" ${ctx.layered.nSamples}`;

  // ----- textures (P2.24). Colour: spectrum imagemap decoded from sRGB. Data: one float imagemap per channel file (pbrt reads float textures from channel 0),
  // linear. Roughness alpha = (factor x G)^2 is a `scale` texture of the G map with itself; the metalness `mix` amount is the B map x factor.
  const tx = d.tex;
  const textures: string[] = [];
  const imagemap = (name: string, kind: 'spectrum' | 'float', file: string, wrap: string, encoding: string, scale?: number): void => {
    textures.push(`Texture "${name}" "${kind}" "imagemap" "string filename" "${file}" "string filter" "bilinear" "string wrap" "${wrap}" "string encoding" "${encoding}"${scale !== undefined && scale !== 1 ? ` ${f('scale', scale)}` : ''}`);
  };
  const baseName = tx.base ? `${stem}.base` : undefined;
  if (tx.base) imagemap(baseName!, 'spectrum', ctx.textures!.image(tx.base), tx.base.wrap, 'sRGB');
  let roughTex: string | undefined, metalTex: string | undefined;
  if (tx.mr) {
    const file = (c: 1 | 2): string => ctx.textures!.channel(tx.mr!, c);
    imagemap(`${stem}.g`, 'float', file(1), tx.mr.wrap, 'linear', m.specularRoughness);
    textures.push(`Texture "${stem}.alpharough" "float" "scale" "texture tex" "${stem}.g" "texture scale" "${stem}.g"`);
    roughTex = `${stem}.alpharough`;
    if (d.metal > 0) {
      imagemap(`${stem}.metal`, 'float', file(2), tx.mr.wrap, 'linear', m.baseMetalness);
      metalTex = `${stem}.metal`;
    }
    note(`metallic-roughness texture: roughness alpha = (${m.specularRoughness} x G)^2 as a scale texture, metalness ${m.baseMetalness} x B as the mix amount (float imagemaps, 8-bit linear, bilinear, wrap ${tx.mr.wrap}; the mix picks a lobe stochastically, exact in expectation)`);
  }
  if (tx.base) note(`base colour texture '${tx.base.key}' (${tx.base.width}x${tx.base.height}, sRGB, wrap ${tx.base.wrap}): imagemap bilinear, which still selects a mip level from the scaled camera ray differentials`);
  const reflectance = (): string => (baseName ? `"texture reflectance" "${baseName}"` : rgbP('reflectance', d.base));
  let shapeParams = '';
  if (d.mask) {
    const { opacity, cutoff, texture } = d.mask;
    if (!texture) {
      if (opacity < cutoff) shapeParams = ` "float alpha" 0`;
    } else {
      // opaque iff opacity x a >= cutoff, a = the bilinearly filtered alpha: alpha' = 0.5 + K (opacity a - cutoff), K large; pbrt keeps a hit with probability alpha' (>= 1 always, <= 0 never)
      imagemap(`${stem}.a`, 'float', ctx.textures!.channel(texture, 3), texture.wrap, 'linear', MASK_GAIN * opacity);
      textures.push(`Texture "${stem}.amix" "float" "mix" "float tex1" ${num(0.5 - MASK_GAIN * cutoff)} "texture tex2" "${stem}.a" "float amount" 0.5`);
      textures.push(`Texture "${stem}.cut" "float" "scale" "texture tex" "${stem}.amix" "float scale" 2`);
      shapeParams = ` "texture alpha" "${stem}.cut"`;
      note(`alpha mask: hard cutoff at ${cutoff} on the bilinear alpha through a gain-${MASK_GAIN} alpha texture (stochastic only inside a band of ${(1 / MASK_GAIN).toPrecision(2)} alpha units)`);
    }
  }

  // ----- metal end
  const metalNode = (): string => {
    if (m.specularColor.some((c) => c !== 1)) note('specular_color != 1 on a metal (F82 tint) is ignored: reflectance = base_color is the F0');
    if (m.coatWeight > 0) {
      const coated = add('coatedconductor', {
        type: 'coatedconductor',
        params: `${reflectance()} ${rough('conductor.', d.alphaU, d.alphaV)} ${f('interface.roughness', d.coatAlpha)} ${f('interface.eta', m.coatIor)} "float thickness" ${LAYERED_THICKNESS} ${layer} "bool remaproughness" false`,
      });
      if (m.coatWeight >= 1) return coated;
      note(`coat_weight ${m.coatWeight} < 1: linear mix of the bare conductor and the coated one (OpenPBR scales the coat Fresnel instead)`);
      return mix(add('conductor', { type: 'conductor', params: `${rgbP('reflectance', d.base)} ${rough('', d.alphaU, d.alphaV)} "bool remaproughness" false` }), coated, m.coatWeight);
    }
    return add('conductor', { type: 'conductor', params: `${reflectance()} ${rough('', d.alphaU, d.alphaV)} "bool remaproughness" false` });
  };

  // ----- dielectric end
  const etaParam = (): string => {
    if (m.transmissionDispersionScale <= 0) return f('eta', d.ior);
    const fit = cauchyFit(d.ior, m.transmissionDispersionAbbeNumber / m.transmissionDispersionScale);
    const pts: string[] = [];
    for (let nm = 360; nm <= 830; nm += 10) pts.push(`${nm} ${num(Number(fit.eta(nm).toFixed(6)))}`);
    note(`dispersion: spectrum eta from a Cauchy fit n = A + B/lambda^2 (A = ${fit.a.toFixed(5)}, B = ${fit.b.toFixed(6)} um^2) of n_d = ${d.ior} and Abbe ${(m.transmissionDispersionAbbeNumber / m.transmissionDispersionScale).toFixed(3)}; pbrt terminates the secondary wavelengths at each refraction`);
    return `"spectrum eta" [ ${pts.join(' ')} ]`;
  };
  const opaqueNode = (): string => {
    if (m.subsurfaceWeight > 0) {
      if (!d.thin) note('subsurface_weight > 0 on a non-thin-walled surface: exported as diffuse transmission anyway (OpenPBR would use a volume)');
      const w = m.subsurfaceWeight;
      const refl = d.base.map((b) => b * (1 - w)) as unknown as Rgb;
      const trans = m.subsurfaceColor.map((c) => c * w) as unknown as Rgb;
      if (m.specularWeight > 0) note('diffusetransmission has no specular lobe: exact only with specular_weight 0; the dielectric specular is dropped here (class B\')');
      return add('diffusetransmission', { type: 'diffusetransmission', params: `${rgbP('reflectance', refl)} ${rgbP('transmittance', trans)}` });
    }
    const diffuse = (): string => add('diffuse', { type: 'diffuse', params: reflectance() });
    const coated = (ior: number, uA: number, vA: number): string =>
      add('coateddiffuse', { type: 'coateddiffuse', params: `${reflectance()} ${rough('', uA, vA)} ${f('eta', ior)} "float thickness" ${LAYERED_THICKNESS} ${layer} "bool remaproughness" false` });
    if (m.coatWeight > 0) {
      note('clear coat on a dielectric: coateddiffuse with the coat IOR/roughness as the interface; the base dielectric specular lobe is not added');
      const c = coated(m.coatIor, d.coatAlpha, d.coatAlpha);
      if (m.coatWeight >= 1) return c;
      note(`coat_weight ${m.coatWeight} < 1: linear mix of the uncoated dielectric and the coated one`);
      return mix(m.specularWeight > 0 && d.ior !== 1 ? coated(d.ior, d.alphaU, d.alphaV) : diffuse(), c, m.coatWeight);
    }
    if (m.specularWeight <= 0 || d.ior === 1) return diffuse();
    if (d.anisotropic) note('anisotropic dielectric over diffuse: coateddiffuse takes u/v roughness on the interface');
    note(`dielectric over diffuse is pbrt's stochastic LayeredBxDF (maxdepth ${ctx.layered.maxDepth}, nsamples ${ctx.layered.nSamples}); OpenPBR approximates the layering with albedo scaling`);
    return coated(d.ior, d.alphaU, d.alphaV);
  };
  const glassNode = (): string => {
    if (d.thin) {
      if (d.alphaU > 0 || d.alphaV > 0) cov('rough thin-walled transmission has no reference: smooth thindielectric written');
      return add('thindielectric', { type: 'thindielectric', params: etaParam() });
    }
    return add('dielectric', { type: 'dielectric', params: `${etaParam()} ${rough('', d.alphaU, d.alphaV)} "bool remaproughness" false` });
  };
  const dielectricNode = (): string => {
    if (!d.transmissive) return opaqueNode();
    const glass = glassNode();
    if (m.transmissionWeight >= 1) return glass;
    note(`transmission_weight ${m.transmissionWeight} < 1: linear mix of the opaque dielectric and the glass (OpenPBR layers them)`);
    return mix(opaqueNode(), glass, m.transmissionWeight);
  };

  let top: string;
  if (tx.mr) top = metalTex ? mix(dielectricNode(), metalNode(), metalTex) : dielectricNode();
  else if (d.metal >= 1) top = metalNode();
  else if (d.metal <= 0) top = dielectricNode();
  else top = mix(dielectricNode(), metalNode(), d.metal);

  const decls = nodes.map(([name, n]) =>
    'a' in n
      ? `MakeNamedMaterial "${name}" "string type" "mix" "string materials" [ "${n.a}" "${n.b}" ] ${typeof n.amount === 'string' ? `"texture amount" "${n.amount}"` : f('amount', n.amount)}`
      : `MakeNamedMaterial "${name}" "string type" "${n.type}" ${n.params}`,
  );
  // emissive texture: linear radiance (texel decoded from sRGB x emission_color x emission_luminance) as a PFM for `AreaLightSource "diffuse" "string filename"`
  let emissionFile: string | undefined;
  if (tx.emissive && d.emission) {
    const e = d.emission;
    emissionFile = ctx.textures!.rgb(tx.emissive, `emis-${stem}`, (r, g, b) => [srgbToLinear(r) * e[0], srgbToLinear(g) * e[1], srgbToLinear(b) * e[2]]);
    note(`emissive texture '${tx.emissive.key}': a float PFM of the decoded texels x ${e.join(' ')}; DiffuseAreaLight reads it bilinearly at the shape uv`);
  }
  return { decls, use: top, medium: d.metal < 1 ? d.medium : undefined, ...(emissionFile ? {} : { emission: d.emission }), textures, shapeParams, ...(emissionFile ? { emissionFile } : {}) };
}

export const toPbrt: Exporter = (scene, opts) => {
  const approximations: string[] = [];
  const files: ExportedFile[] = [];
  const c = scene.camera;
  const lines: string[] = [];
  lines.push(`# ${scene.name}: ${scene.purpose}`);
  lines.push(`# generated by bench/exporters/pbrt.ts; do not edit`);
  lines.push(`Scale -1 1 1`);
  lines.push(`LookAt ${vec(c.position)}  ${vec(c.target)}  ${vec(c.up)}`);
  if (scene.cameraMedium) lines.push(`MediumInterface "" "${scene.cameraMedium}"`);
  lines.push(`Camera "perspective" "float fov" ${num(shorterAxisFovDeg(c, scene.film))}`);
  lines.push(`Film "rgb" "integer xresolution" ${scene.film.width} "integer yresolution" ${scene.film.height}`);
  lines.push(`PixelFilter "box" "float xradius" 0.5 "float yradius" 0.5`);
  lines.push(`Sampler "zsobol" "integer pixelsamples" ${opts.spp}`);
  // openpbr materials: named materials (and absorbers) first, so the integrator line knows whether media are in play
  const ctx: NativeCtx = { renderer: 'pbrt', coverage: opts.coverage === true, layered: { ...LAYERED_DEFAULTS, ...opts.layered }, notes: new Set(), textures: new TextureSet(scene) };
  const openpbr = new Map<string, PbrtOpenPbr>();
  for (const w of scene.meshes) {
    const m = scene.materials[w.material];
    if (m?.type === 'openpbr' && !openpbr.has(w.material)) openpbr.set(w.material, pbrtOpenPbr(w.material, m, ctx));
  }
  const hasMedia = Object.keys(scene.media ?? {}).length > 0;
  const volumetric = hasMedia || [...openpbr.values()].some((o) => o.medium);
  const rr = scene.integrator.rrStartDepth === undefined;
  const rrArgs = rr ? ' "bool rr" false' : ` "integer rrdepth" ${Math.max(0, scene.integrator.rrStartDepth! - 1)}`;
  lines.push(`Integrator "${volumetric ? 'volpath' : 'path'}" "integer maxdepth" ${scene.integrator.maxDepth}${rrArgs}`);
  lines.push(`WorldBegin`);
  if (scene.cameraMedium) lines.push(`MediumInterface "" ""`);
  for (const [name, o] of openpbr) {
    lines.push(...o.textures, ...o.decls);
    if (o.medium) lines.push(`MakeNamedMedium "med-${name}" "string type" "homogeneous" "rgb sigma_a" [ ${vec(o.medium.sigma)} ] "rgb sigma_s" [ 0.0 0.0 0.0 ] "float scale" 1.0`);
  }
  for (const [name, medium] of Object.entries(scene.media ?? {})) lines.push(...pbrtMedium(name, medium, medium.type === 'grid' ? scene.meshes.find((m) => m.medium?.interior === name)?.transform : undefined, opts.assetsDir));
  if (hasMedia) approximations.push('pbrt: participating media use `volpath` (patched no-RR, V59); RGB sigma_a / sigma_s are uplifted to unbounded spectra (grey is exact, saturated colours are off per channel, V62); grids are `uniformgrid` media (dense density, no NanoVDB converter).');
  if (volumetric && !hasMedia) approximations.push('pbrt: absorption-only interior media use `volpath` (patched no-RR, bench/references/pbrt-no-rr.patch) and RGB `sigma_a` that pbrt uplifts to unbounded spectra (a coloured sigma is slightly off per channel: measured by bench/references/openpbr-check.ts).');

  const env = scene.environment;
  if (env?.type === 'uniform') {
    lines.push(`LightSource "infinite" "rgb L" [ ${vec(env.radiance)} ]`);
  } else if (env?.type === 'hdri') {
    const equirect = loadEnvImage(opts.assetsDir, env.path);
    const octa = equirectToOctahedral(equirect, env.rotationYDeg, env.scale, OCTA_SIZE);
    files.push({ path: 'env-octahedral.pfm', data: Buffer.from(writePFM(octa)) });
    lines.push(`LightSource "infinite" "string filename" "env-octahedral.pfm"`);
    approximations.push(`pbrt: equirect env resampled to a ${OCTA_SIZE}x${OCTA_SIZE} equal-area octahedral image (pbrt-v4 only accepts that layout); scale and rotationYDeg baked in; pbrt looks up nearest-pixel while Mitsuba interpolates bilinearly.`);
  }
  for (const l of scene.lights ?? []) {
    // pbrt distant light: `illuminance` in lux-like units; w/m² irradiance maps 1:1 for an RGB illuminant.
    lines.push(`LightSource "distant" "point3 from" [ ${vec(l.direction.map((v) => -v))} ] "point3 to" [ 0 0 0 ] "rgb L" [ ${vec(l.irradiance)} ]`);
    approximations.push('pbrt: directional light angularRadius is ignored (pbrt distant lights are delta lights); `L` is used as irradiance.');
  }

  // Image textures: pbrt-v4 `imagemap` with the `bilinear` filter. It does pick a mip level from the camera ray differentials (scaled by
  // max(1/8, 1/sqrt(spp)), so at the reference spp the footprint is an eighth of a pixel and a texture at most a few texels per pixel stays
  // on level 0); Mitsuba's `bitmap` bilinear never uses mips. pbrt flips t (image row 0 = t 1), so the PLY v is written as 1 - v.
  const textures = usedTextures(scene, 'pbrt');
  for (const t of textures.values()) {
    if (t.wrap === 'mirror') throw new Error(`pbrt: texture '${t.name}' wrap 'mirror' does not exist in pbrt-v4 (repeat, clamp, black, octahedralsphere)`);
    files.push({ path: t.file, data: encodePng(t.width, t.height, t.data) });
    lines.push(`Texture "${t.name}" "spectrum" "imagemap" "string filename" "${t.file}" "string filter" "bilinear" "string wrap" "${t.wrap}" "string encoding" "sRGB"`);
  }
  if (textures.size > 0) approximations.push('pbrt: image textures use the `bilinear` imagemap filter, which still selects a mip level from the (scaled) camera ray differentials; Mitsuba bilinear is level 0 only. Spectral uplift of the texel RGB as for any reflectance.');
  for (const w of worldMeshes(scene, opts.assetsDir)) {
    const m = scene.materials[w.desc.material];
    if (!m) throw new Error(`mesh '${w.desc.name}': unknown material '${w.desc.material}'`);
    files.push({ path: `${w.stem}.ply`, data: encodePly(w.mesh, { flipV: true }) });
    if (m.type === 'openpbr' && (m.specularRoughnessAnisotropy ?? 0) > 0 && !w.mesh.uvs) approximations.push(`pbrt: mesh '${w.desc.name}' is anisotropic but has no uvs: the GGX axis is pbrt's default uv frame, not a tangent the other renderers share (P2.26)`);
    lines.push(`AttributeBegin`);
    let emission = w.desc.emission;
    let emissionFile: string | undefined;
    let twoSided = w.desc.twoSided === true;
    if (m.type === 'openpbr') {
      const o = openpbr.get(w.desc.material)!;
      lines.push(`  NamedMaterial "${o.use}"`);
      if (o.medium) lines.push(`  MediumInterface "med-${w.desc.material}" ""`);
      if (o.emission || o.emissionFile) {
        if (emission) throw new Error(`mesh '${w.desc.name}': both the mesh and its openpbr material emit; use one`);
        twoSided = m.emissionTwoSided === true;
        if (o.emissionFile) emissionFile = o.emissionFile;
        else emission = o.emission;
      }
    } else {
      lines.push(`  ${material(m, (name) => name)}`);
    }
    if (w.desc.medium) lines.push(`  MediumInterface "${w.desc.medium.interior ?? ''}" "${w.desc.medium.exterior ?? ''}"`);
    if (emission) {
      lines.push(`  AreaLightSource "diffuse" "rgb L" [ ${vec(emission)} ] "bool twosided" ${twoSided ? 'true' : 'false'}`);
    } else if (emissionFile) {
      lines.push(`  AreaLightSource "diffuse" "string filename" "${emissionFile}" "bool twosided" ${twoSided ? 'true' : 'false'}`);
    }
    lines.push(`  Shape "plymesh" "string filename" "${w.stem}.ply"${m.type === 'openpbr' ? openpbr.get(w.desc.material)!.shapeParams : ''}`);
    lines.push(`AttributeEnd`);
  }
  approximations.push(...ctx.notes);
  files.push(...ctx.textures!.files.values());
  files.push({ path: 'scene.pbrt', data: lines.join('\n') + '\n' });
  return { sceneFile: 'scene.pbrt', files, approximations };
};
