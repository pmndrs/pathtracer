import { compose, rotationX, translation } from '../geometry.ts';
import { buildMaterialX, type MaterialXMode } from '../materialx.ts';
import type { MaterialDesc, SceneDesc, TextureDesc } from '../schema.ts';

/**
 * MaterialX scenes (P5, docs/internal/design/materialx.md). Experiment scenes without pbrt / Mitsuba references (neither reads MaterialX graphs):
 * the checks compare our own two approaches and the equivalent native `openpbr` description.
 *
 *  - `mtlx-const-open-pbr` / `mtlx-const-open-pbr-ref`: an `open_pbr_surface` of constants, and the `openpbr` MaterialDesc with the same numbers. Same record, same image.
 *  - `mtlx-const-std` / `mtlx-const-std-ref`: a `standard_surface` of constants through the documented mapping (V261), and its hand-mapped OpenPBR.
 *  - `mtlx-uv-bake` / `mtlx-uv-codegen`: one graph (image + place2d + ramp + normalmap, all uv-driven) rendered by approach (a) and (b). The bake is 1024^2.
 *  - `mtlx-uv-bake-256`: the same at 256^2 (the resolution cost).
 *  - `mtlx-proc-codegen`: a procedural graph driven by the hit position (stripes by `sin`), which only approach (b) can draw.
 *
 * The meshes carry uv in [0, 1]^2 (sphere and an upright panel) because the bake covers exactly that square.
 */
const header = (v = '1.39') => `<?xml version="1.0"?><materialx version="${v}" colorspace="lin_rec709">`;

export const DOC_CONST_OPEN_PBR = `${header()}
  <surfacematerial name="M" type="material"><input name="surfaceshader" type="surfaceshader" nodename="S"/></surfacematerial>
  <open_pbr_surface name="S" type="surfaceshader">
    <input name="base_color" type="color3" value="0.7, 0.35, 0.15"/>
    <input name="specular_roughness" type="float" value="0.35"/>
    <input name="coat_weight" type="float" value="0.6"/>
    <input name="coat_roughness" type="float" value="0.1"/>
  </open_pbr_surface></materialx>`;

export const DOC_CONST_STD = `${header('1.38')}
  <standard_surface name="S" type="surfaceshader">
    <input name="base" type="float" value="0.9"/>
    <input name="base_color" type="color3" value="0.2, 0.5, 0.8"/>
    <input name="specular_roughness" type="float" value="0.25"/>
    <input name="coat" type="float" value="0.4"/>
    <input name="coat_roughness" type="float" value="0.15"/>
    <input name="coat_IOR" type="float" value="1.5"/>
  </standard_surface>
  <surfacematerial name="M" type="material"><input name="surfaceshader" type="surfaceshader" nodename="S"/></surfacematerial></materialx>`;

/** uv-driven: a rotated, offset checker for colour, rings for roughness, a ramp for metalness, a bump normal map. */
export const DOC_UV = `${header('1.38')}
  <nodegraph name="NG">
    <texcoord name="uv" type="vector2"/>
    <place2d name="pl" type="vector2"><input name="texcoord" type="vector2" nodename="uv"/><input name="rotate" type="float" value="15"/><input name="offset" type="vector2" value="0.1, 0"/></place2d>
    <tiledimage name="col" type="color3"><input name="file" type="filename" value="gen:checker8" colorspace="srgb_texture"/><input name="texcoord" type="vector2" nodename="pl"/><input name="uvtiling" type="vector2" value="1, 1"/></tiledimage>
    <multiply name="tint" type="color3"><input name="in1" type="color3" nodename="col"/><input name="in2" type="color3" value="1, 0.9, 0.8"/></multiply>
    <image name="rings" type="float"><input name="file" type="filename" value="gen:rings"/><input name="texcoord" type="vector2" nodename="pl"/></image>
    <multiply name="r1" type="float"><input name="in1" type="float" nodename="rings"/><input name="in2" type="float" value="0.7"/></multiply>
    <add name="rough" type="float"><input name="in1" type="float" nodename="r1"/><input name="in2" type="float" value="0.12"/></add>
    <ramplr name="metal" type="float"><input name="valuel" type="float" value="0"/><input name="valuer" type="float" value="1"/><input name="texcoord" type="vector2" nodename="uv"/></ramplr>
    <smoothstep name="metal2" type="float"><input name="in" type="float" nodename="metal"/><input name="low" type="float" value="0.55"/><input name="high" type="float" value="0.75"/></smoothstep>
    <image name="bump" type="vector3"><input name="file" type="filename" value="gen:bumps4"/><input name="texcoord" type="vector2" nodename="uv"/></image>
    <normalmap name="nm" type="vector3"><input name="in" type="vector3" nodename="bump"/><input name="scale" type="float" value="1"/></normalmap>
    <output name="out_color" type="color3" nodename="tint"/>
    <output name="out_rough" type="float" nodename="rough"/>
    <output name="out_metal" type="float" nodename="metal2"/>
    <output name="out_normal" type="vector3" nodename="nm"/>
  </nodegraph>
  <standard_surface name="S" type="surfaceshader">
    <input name="base_color" type="color3" nodegraph="NG" output="out_color"/>
    <input name="metalness" type="float" nodegraph="NG" output="out_metal"/>
    <input name="specular_roughness" type="float" nodegraph="NG" output="out_rough"/>
    <input name="normal" type="vector3" nodegraph="NG" output="out_normal"/>
  </standard_surface>
  <surfacematerial name="M" type="material"><input name="surfaceshader" type="surfaceshader" nodename="S"/></surfacematerial></materialx>`;

/** Position-driven: stripes of two colours along a tilted axis, roughness varying with x, a metallic band. */
export const DOC_PROC = `${header()}
  <nodegraph name="NG">
    <position name="p" type="vector3"/>
    <separate3 name="ps" type="multioutput"><input name="in" type="vector3" nodename="p"/></separate3>
    <multiply name="py" type="float"><input name="in1" type="float" nodename="ps" output="outy"/><input name="in2" type="float" value="22"/></multiply>
    <multiply name="px" type="float"><input name="in1" type="float" nodename="ps" output="outx"/><input name="in2" type="float" value="7"/></multiply>
    <add name="arg" type="float"><input name="in1" type="float" nodename="py"/><input name="in2" type="float" nodename="px"/></add>
    <sin name="s" type="float"><input name="in" type="float" nodename="arg"/></sin>
    <multiply name="s2" type="float"><input name="in1" type="float" nodename="s"/><input name="in2" type="float" value="0.5"/></multiply>
    <add name="s3" type="float"><input name="in1" type="float" nodename="s2"/><input name="in2" type="float" value="0.5"/></add>
    <smoothstep name="stripe" type="float"><input name="in" type="float" nodename="s3"/><input name="low" type="float" value="0.45"/><input name="high" type="float" value="0.55"/></smoothstep>
    <mix name="color" type="color3"><input name="fg" type="color3" value="0.85, 0.75, 0.2"/><input name="bg" type="color3" value="0.1, 0.25, 0.6"/><input name="mix" type="float" nodename="stripe"/></mix>
    <remap name="rough" type="float"><input name="in" type="float" nodename="ps" output="outx"/><input name="inlow" type="float" value="-0.7"/><input name="inhigh" type="float" value="0.7"/><input name="outlow" type="float" value="0.1"/><input name="outhigh" type="float" value="0.7"/></remap>
    <output name="out_color" type="color3" nodename="color"/>
    <output name="out_rough" type="float" nodename="rough"/>
    <output name="out_metal" type="float" nodename="stripe"/>
  </nodegraph>
  <open_pbr_surface name="S" type="surfaceshader">
    <input name="base_color" type="color3" nodegraph="NG" output="out_color"/>
    <input name="base_metalness" type="float" nodegraph="NG" output="out_metal"/>
    <input name="specular_roughness" type="float" nodegraph="NG" output="out_rough"/>
  </open_pbr_surface>
  <surfacematerial name="M" type="material"><input name="surfaceshader" type="surfaceshader" nodename="S"/></surfacematerial></materialx>`;

function stage(name: string, purpose: string, material: MaterialDesc, textures: Record<string, TextureDesc>): SceneDesc {
  const facing = (x: number, y: number, z: number) => compose(translation(x, y, z), rotationX(Math.PI / 2));
  return {
    version: 1,
    name,
    purpose,
    tier: 1,
    camera: { type: 'perspective', position: [0, 1.0, 3.8], target: [0, 0.65, 0], up: [0, 1, 0], yFovDeg: 38 },
    film: { width: 320, height: 240, filter: 'box' },
    integrator: { maxDepth: 4 },
    ...(Object.keys(textures).length > 0 ? { textures } : {}),
    materials: { ground: { type: 'diffuse', reflectance: [0.55, 0.55, 0.55] }, subject: material },
    meshes: [
      { name: 'ground', geometry: { type: 'quad', size: [8, 8] }, material: 'ground' },
      { name: 'ball', geometry: { type: 'sphere', radius: 0.6, segments: 64, rings: 32 }, material: 'subject', transform: translation(-0.75, 0.6, 0), uv: {} },
      { name: 'panel', geometry: { type: 'quad', size: [1.2, 1.2] }, material: 'subject', transform: facing(0.8, 0.7, 0), uv: {} },
    ],
    lights: [{ type: 'directional', direction: [-0.5, -0.75, -0.4], irradiance: [3.2, 3.0, 2.7] }],
    environment: { type: 'uniform', radiance: [0.45, 0.5, 0.6] },
  };
}

function fromDoc(name: string, purpose: string, key: string, source: string, mode: MaterialXMode, resolution?: number): SceneDesc {
  const { material, textures } = buildMaterialX({ key, source, mode, ...(resolution ? { resolution } : {}) });
  return stage(name, purpose, material, textures);
}

export const mtlxConstOpenPbr = fromDoc('mtlx-const-open-pbr', 'open_pbr_surface of constants through MaterialX (mode constants): must equal mtlx-const-open-pbr-ref per pixel.', 'const-open-pbr', DOC_CONST_OPEN_PBR, 'constants');
export const mtlxConstOpenPbrRef = stage('mtlx-const-open-pbr-ref', 'The native openpbr description of the numbers in mtlx-const-open-pbr.', { type: 'openpbr', baseColor: [0.7, 0.35, 0.15], specularRoughness: 0.35, coatWeight: 0.6, coatRoughness: 0.1 }, {});
export const mtlxConstStd = fromDoc('mtlx-const-std', 'standard_surface of constants through MaterialX and the V261 mapping: must equal mtlx-const-std-ref per pixel.', 'const-std', DOC_CONST_STD, 'constants');
export const mtlxConstStdRef = stage('mtlx-const-std-ref', 'The hand-mapped OpenPBR of the numbers in mtlx-const-std (standard_surface base 0.9 -> base_weight, coat -> coat_weight, ...; standard_surface defaults sheen_roughness 0.3 and thin_film_IOR 1.5).', { type: 'openpbr', baseWeight: 0.9, baseColor: [0.2, 0.5, 0.8], specularRoughness: 0.25, coatWeight: 0.4, coatRoughness: 0.15, coatIor: 1.5, fuzzRoughness: 0.3, thinFilmIor: 1.5 }, {});
export const mtlxUvBake = fromDoc('mtlx-uv-bake', 'uv-driven MaterialX graph (checker, rings, ramp, bump normal map) baked to 1024^2 textures (approach a).', 'uv-bake', DOC_UV, 'bake', 1024);
export const mtlxUvBake256 = fromDoc('mtlx-uv-bake-256', 'The same graph baked to 256^2: the cost of the bake resolution.', 'uv-bake-256', DOC_UV, 'bake', 256);
export const mtlxUvCodegen = fromDoc('mtlx-uv-codegen', 'The same uv-driven graph as generated WGSL evaluated at the hit (approach b).', 'uv-codegen', DOC_UV, 'codegen');
export const mtlxProcCodegen = fromDoc('mtlx-proc-codegen', 'Position-driven procedural MaterialX graph (sin stripes, remap, smoothstep) as generated WGSL: not expressible as a uv bake.', 'proc-codegen', DOC_PROC, 'codegen');

export const materialxScenes: readonly SceneDesc[] = [mtlxConstOpenPbr, mtlxConstOpenPbrRef, mtlxConstStd, mtlxConstStdRef, mtlxUvBake, mtlxUvBake256, mtlxUvCodegen, mtlxProcCodegen];
