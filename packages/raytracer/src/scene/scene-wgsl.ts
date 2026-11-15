/**
 * WGSL for the scene copy (prefix `rt`): the two bindings, `rtMaterial`, the light functions and the data accessors
 * `rtSurface` uses. Generated text carries no comments (the module is near the ~50 KB kernel cap; see modules.ts
 * `stripComments`): the documentation is here and in constants.ts / material.ts / lights.ts.
 *
 * Bindings (group/binding from the caller, two consecutive): `rtSceneTable: array<vec4u>`, `rtSceneData: array<u32>`.
 * With `packing: 'packed'` (P3.11) one binding instead: `rtScene: array<vec4u>` = [header | table | data] (constants.ts `SceneCopyPacking`), and the
 * generated text reaches the table as `rtScene[i + 1u]` and the data as `rtSceneWord(i)`: `rtPackSceneAccess` rewrites every `rtSceneTable[..]` /
 * `rtSceneData[..]` of a finished source, so the split text of every generator stays byte-identical to what it was before packing existed.
 *
 * Public WGSL (decoded structs have the TypeScript twins `RtMaterial`, `RtLight`, ...):
 *
 *     fn rtMaterial(id: u32) -> RtMaterial          decoded material; id >= materialCount (incl. RT_NO_MATERIAL) = default record
 *     fn rtMaterialAlpha(m: RtMaterial) -> f32       GGX alpha = specularRoughness^2
 *     fn rtLightCount() -> u32
 *     fn rtLight(index: u32) -> RtLight               emissive triangle record (kind RT_LIGHT_TRIANGLE); sun/env fields are zero
 *     fn rtSun() -> RtSun, rtHasSun() -> bool         the directional light (rtSun().index == RT_NONE when absent)
 *     fn rtEnvironment() -> RtEnvironment, rtHasEnvironment()   (mapW / mapH / mapBase / pdfNorm: the equirect map, mapW 0 = none; env-map.ts)
 *     fn rtSampleLight(u: f32) -> RtLightSample       alias-table pick over power: { index, pmf, u (the leftover, rescaled) }
 *     fn rtLightPmf(index: u32) -> f32                probability rtSampleLight picks `index`
 *     fn rtLightIndexOfHit(instance, triangle) -> u32 light index of an emissive triangle hit, RT_NONE otherwise
 *     fn rtLightPmfOfHit(instance, triangle) -> f32   the pmf of that light (0 for a plain triangle): the MIS term for a BSDF-sampled hit
 *     fn rtSceneInfo() -> RtSceneInfo                 counts, total power, scene radius
 *     (with `sampling: true`) rtSampleLightLi(index, p, u) and the sun / environment helpers: see light-sample.ts
 *     (with `textures`, texture.ts)
 *       fn rtTexSample(slot, uv, lod) -> vec4f                  trilinear lookup, linear-light rgb (sRGB decoded before filtering) + alpha
 *       fn rtTexLodOf(record, uvFootprint) -> f32               level for a filter width in uv units (twin: textureLod)
 *       fn rtApplyMaterialTextures(m, uv, uvFootprint) -> RtMaterial   glTF semantics: baseColor * rgb, opacity * a, roughness * g, metalness * b, emission * rgb
 *       fn rtApplyNormalMap(n, tangent, texel) -> vec3f         tangent-space normal map (glTF: +Y up) around the shading normal
 */
import { INSTANCE_LIGHT, LIGHT_FLAG, LIGHT_KIND, LIGHT_MAP_MODE, LIGHT_RECORD_VEC4S, PACKED_HEADER, PACKED_TABLE_BASE_VEC4S, SCENE_HEADER_VEC4S, SCENE_NONE, DEFAULT_SCENE_PACKING, type SceneCopyPacking } from './constants.ts';
import { MATERIAL_FLAG, MATERIAL_RECORD_VEC4S } from './material.ts';
import { rtLightSampleWgsl } from './light-sample.ts';
import { SURFACE_TABLE } from '../trace/constants.ts';
import type { TextureLayout } from './texture.ts';

export interface SceneWgslOptions {
  group: number;
  firstBinding: number;
  /** Emit `rtMaterial`. Default true. */
  materials?: boolean;
  /** Emit the light functions. Default true. */
  lights?: boolean;
  /** Also emit the light point sampling (`light-sample.ts`: `rtSampleLightLi` and friends). Needs `lights`. Default false (kernel size). */
  sampling?: boolean;
  /** Emit the texture arrays and lookups (needs `materials`). Bindings follow the buffers (two, or one packed): textures, then samplers (`rtSceneCopyLayoutEntries`). */
  textures?: TextureLayout | null;
  /** `split` (default `DEFAULT_SCENE_PACKING`): `rtSceneTable` + `rtSceneData`; `packed`: one `rtScene` binding. */
  packing?: SceneCopyPacking;
}

/** Bindings the scene copy's buffers take (2 split, 1 packed): textures start after them. */
export function rtSceneBufferBindings(packing: SceneCopyPacking = DEFAULT_SCENE_PACKING): number {
  return packing === 'packed' ? 1 : 2;
}

/**
 * Rewrites a WGSL source written against the split bindings so it reads the packed buffer: `rtSceneTable[e]` -> `rtScene[e + 1u]` (the table starts
 * after the header vec4), `rtSceneData[e]` -> `rtSceneWord(e)`. Nested brackets are handled; an index with an operator that binds weaker than `+` is
 * parenthesised. Idempotent (the output has neither name). Does not touch the binding declarations: `rtSceneBindingsWgsl` emits those per packing.
 */
export function rtPackSceneAccess(source: string): string {
  const re = /rtScene(Table|Data)\[/g;
  const convert = (text: string): string => {
    let out = '';
    let last = 0;
    re.lastIndex = 0;
    for (let m = re.exec(text); m; m = re.exec(text)) {
      const open = m.index + m[0].length;
      let depth = 1;
      let close = open;
      for (; close < text.length && depth > 0; close++) {
        const c = text[close];
        if (c === '[') depth++;
        else if (c === ']') depth--;
      }
      if (depth !== 0) throw new Error('rtPackSceneAccess: unbalanced bracket after ' + m[0]);
      const inner = convert(text.slice(open, close - 1));
      out += text.slice(last, m.index);
      if (m[1] === 'Data') out += `rtSceneWord(${inner})`;
      else if (/^\d+u?$/.test(inner.trim())) out += `rtScene[${parseInt(inner, 10) + PACKED_TABLE_BASE_VEC4S}u]`;
      else out += `rtScene[${weakerThanAdd(inner) ? `(${inner})` : inner} + ${PACKED_TABLE_BASE_VEC4S}u]`;
      last = close;
      re.lastIndex = close;
    }
    return out + text.slice(last);
  };
  return convert(source);
}

/** True when the expression has a top-level (outside parentheses and brackets) operator that binds weaker than `+`. */
function weakerThanAdd(expr: string): boolean {
  let depth = 0;
  for (let i = 0; i < expr.length; i++) {
    const c = expr[i] as string;
    if (c === '(' || c === '[') depth++;
    else if (c === ')' || c === ']') depth--;
    else if (depth === 0 && (c === '&' || c === '|' || c === '^' || c === '<' || c === '>' || c === '=' || c === '!' || c === '?')) return true;
  }
  return false;
}

const hex = (value: number): string => `0x${value.toString(16)}u`;

/** The bindings and the generic accessors (always emitted). */
export function rtSceneBindingsWgsl(o: SceneWgslOptions): string {
  const packed = (o.packing ?? DEFAULT_SCENE_PACKING) === 'packed';
  const declarations = packed
    ? `@group(${o.group}) @binding(${o.firstBinding}) var<storage, read> rtScene: array<vec4u>;
fn rtSceneWord(i: u32) -> u32 {
    return rtScene[rtScene[0][${PACKED_HEADER.dataBase}] + (i >> 2u)][i & 3u];
}`
    : `@group(${o.group}) @binding(${o.firstBinding}) var<storage, read> rtSceneTable: array<vec4u>;
@group(${o.group}) @binding(${o.firstBinding + 1}) var<storage, read> rtSceneData: array<u32>;`;
  return /* wgsl */ `
const RT_NONE: u32 = ${hex(SCENE_NONE)};
${declarations}
struct RtSceneInfo {
    instanceSlots: u32,
    materialCount: u32,
    lightCount: u32,
    triangleLightCount: u32,
    totalPower: f32,
    sceneRadius: f32,
}
fn rtSceneInfo() -> RtSceneInfo {
    let h1 = rtSceneTable[1];
    let h2 = rtSceneTable[2];
    let h3 = rtSceneTable[3];
    return RtSceneInfo(rtSceneTable[0].x, h1.x, h1.z, h2.x, bitcast<f32>(h3.x), bitcast<f32>(h3.y));
}
`;
}

export function rtMaterialWgsl(): string {
  const f = (v: string) => `bitcast<f32>(${v})`;
  return /* wgsl */ `
const RT_UNORM16_SCALE: f32 = ${Math.fround(1 / 65535)};
const RT_MATERIAL_THIN_WALLED: u32 = ${hex(MATERIAL_FLAG.thinWalled)};
const RT_MATERIAL_CONDUCTOR_ETA_K: u32 = ${hex(MATERIAL_FLAG.conductorEtaK)};
const RT_MATERIAL_EMISSION_TWO_SIDED: u32 = ${hex(MATERIAL_FLAG.emissionTwoSided)};
const RT_MATERIAL_DOUBLE_SIDED: u32 = ${hex(MATERIAL_FLAG.doubleSided)};
const RT_ALPHA_OPAQUE: u32 = 0u;
const RT_ALPHA_MASK: u32 = 1u;
const RT_ALPHA_BLEND: u32 = 2u;
const RT_TEXTURE_NONE: u32 = 0xffffu;

struct RtMaterial {
    baseColor: vec3f,
    baseWeight: f32,
    metalness: f32,
    specularRoughness: f32,
    specularWeight: f32,
    specularColor: vec3f,
    ior: f32,
    specularAnisotropy: f32,
    anisotropyRotation: f32,
    conductorEta: vec3f,
    conductorK: vec3f,
    transmissionWeight: f32,
    transmissionColor: vec3f,
    transmissionDepth: f32,
    coatWeight: f32,
    coatColor: vec3f,
    coatRoughness: f32,
    coatIor: f32,
    sheenWeight: f32,
    sheenColor: vec3f,
    sheenRoughness: f32,
    emissionColor: vec3f,
    emissionLuminance: f32,
    opacity: f32,
    alphaCutoff: f32,
    alphaMode: u32,
    thinWalled: bool,
    conductorEtaK: bool,
    emissionTwoSided: bool,
    doubleSided: bool,
    baseColorTexture: u32,
    roughnessMetalTexture: u32,
    normalTexture: u32,
    emissionTexture: u32,
}

fn rtLo(w: u32) -> f32 {
    return f32(w & 0xffffu) * RT_UNORM16_SCALE;
}
fn rtHi(w: u32) -> f32 {
    return f32(w >> 16u) * RT_UNORM16_SCALE;
}

fn rtMaterialAt(base: u32) -> RtMaterial {
    let r0 = rtSceneTable[base];
    let r1 = rtSceneTable[base + 1u];
    let r2 = rtSceneTable[base + 2u];
    let r3 = rtSceneTable[base + 3u];
    let r4 = rtSceneTable[base + 4u];
    let r5 = rtSceneTable[base + 5u];
    let r6 = rtSceneTable[base + 6u];
    let r7 = rtSceneTable[base + 7u];
    var m: RtMaterial;
    m.baseColor = bitcast<vec3f>(r0.xyz);
    m.ior = ${f('r0.w')};
    m.emissionColor = bitcast<vec3f>(r1.xyz);
    m.emissionLuminance = ${f('r1.w')};
    m.transmissionDepth = ${f('r2.x')};
    m.coatIor = ${f('r2.y')};
    let flags = r2.z;
    m.alphaMode = flags & 3u;
    m.thinWalled = (flags & RT_MATERIAL_THIN_WALLED) != 0u;
    m.conductorEtaK = (flags & RT_MATERIAL_CONDUCTOR_ETA_K) != 0u;
    m.emissionTwoSided = (flags & RT_MATERIAL_EMISSION_TWO_SIDED) != 0u;
    m.doubleSided = (flags & RT_MATERIAL_DOUBLE_SIDED) != 0u;
    m.baseColorTexture = r2.w & 0xffffu;
    m.roughnessMetalTexture = r2.w >> 16u;
    m.normalTexture = r3.x & 0xffffu;
    m.emissionTexture = r3.x >> 16u;
    let eta01 = unpack2x16float(r3.y);
    let ek = unpack2x16float(r3.z);
    let k12 = unpack2x16float(r3.w);
    m.conductorEta = vec3f(eta01.x, eta01.y, ek.x);
    m.conductorK = vec3f(ek.y, k12.x, k12.y);
    m.baseWeight = rtLo(r4.x);
    m.metalness = rtHi(r4.x);
    m.specularRoughness = rtLo(r4.y);
    m.specularWeight = rtHi(r4.y);
    m.specularColor = vec3f(rtLo(r4.z), rtHi(r4.z), rtLo(r4.w));
    m.specularAnisotropy = rtHi(r4.w);
    m.anisotropyRotation = rtLo(r5.x);
    m.transmissionWeight = rtHi(r5.x);
    m.transmissionColor = vec3f(rtLo(r5.y), rtHi(r5.y), rtLo(r5.z));
    m.coatWeight = rtHi(r5.z);
    m.coatRoughness = rtLo(r5.w);
    m.coatColor = vec3f(rtHi(r5.w), rtLo(r6.x), rtHi(r6.x));
    m.sheenWeight = rtLo(r6.y);
    m.sheenRoughness = rtHi(r6.y);
    m.sheenColor = vec3f(rtLo(r6.z), rtHi(r6.z), rtLo(r6.w));
    m.opacity = rtHi(r6.w);
    m.alphaCutoff = rtLo(r7.x);
    return m;
}

fn rtMaterial(id: u32) -> RtMaterial {
    let h1 = rtSceneTable[1];
    let slot = select(h1.x, id, id < h1.x);
    return rtMaterialAt(h1.y + slot * ${MATERIAL_RECORD_VEC4S}u);
}

fn rtMaterialAlpha(m: RtMaterial) -> f32 {
    return m.specularRoughness * m.specularRoughness;
}
`;
}

export function rtLightsWgsl(): string {
  return /* wgsl */ `
const RT_LIGHT_TRIANGLE: u32 = ${LIGHT_KIND.triangle}u;
const RT_LIGHT_SUN: u32 = ${LIGHT_KIND.sun}u;
const RT_LIGHT_ENVIRONMENT: u32 = ${LIGHT_KIND.environment}u;
const RT_LIGHT_TWO_SIDED: u32 = ${LIGHT_FLAG.twoSided}u;
const RT_LIGHT_MAP_DENSE: u32 = ${LIGHT_MAP_MODE.dense}u;
const RT_LIGHT_MAP_TABLE: u32 = ${LIGHT_MAP_MODE.map}u;

struct RtLight {
    kind: u32,
    position: vec3f,
    edge1: vec3f,
    edge2: vec3f,
    normal: vec3f,
    area: f32,
    pmf: f32,
    radiance: vec3f,
    material: u32,
    instance: u32,
    triangle: u32,
    twoSided: bool,
}

struct RtSun {
    index: u32,
    direction: vec3f,
    angularRadius: f32,
    cosAngularRadius: f32,
    solidAngle: f32,
    irradiance: vec3f,
    pmf: f32,
}

struct RtEnvironment {
    index: u32,
    radiance: vec3f,
    scale: f32,
    rotationY: f32,
    mapSlot: u32,
    mapBase: u32,
    mapW: u32,
    mapH: u32,
    pdfNorm: f32,
    pmf: f32,
}

struct RtLightSample {
    index: u32,
    pmf: f32,
    u: f32,
}

fn rtLightCount() -> u32 {
    return rtSceneTable[1].z;
}

fn rtLightBase(index: u32) -> u32 {
    return rtSceneTable[1].w + index * ${LIGHT_RECORD_VEC4S}u;
}

fn rtLightPmf(index: u32) -> f32 {
    return bitcast<f32>(rtSceneTable[rtLightBase(index) + 2u].w);
}

fn rtLight(index: u32) -> RtLight {
    let b = rtLightBase(index);
    let r0 = rtSceneTable[b];
    let r1 = rtSceneTable[b + 1u];
    let r2 = rtSceneTable[b + 2u];
    let r3 = rtSceneTable[b + 3u];
    let r4 = rtSceneTable[b + 4u];
    let r5 = rtSceneTable[b + 5u];
    var l: RtLight;
    l.kind = r0.w;
    l.position = bitcast<vec3f>(r0.xyz);
    l.edge1 = bitcast<vec3f>(r1.xyz);
    l.area = bitcast<f32>(r1.w);
    l.edge2 = bitcast<vec3f>(r2.xyz);
    l.pmf = bitcast<f32>(r2.w);
    l.normal = bitcast<vec3f>(r3.xyz);
    l.material = r3.w;
    l.radiance = bitcast<vec3f>(r4.xyz);
    l.instance = r4.w;
    l.triangle = r5.x;
    l.twoSided = (r5.y & RT_LIGHT_TWO_SIDED) != 0u;
    return l;
}

fn rtHasSun() -> bool {
    return rtSceneTable[2].y != RT_NONE;
}

fn rtSun() -> RtSun {
    var s: RtSun;
    s.index = rtSceneTable[2].y;
    if (s.index == RT_NONE) {
        s.direction = vec3f(0.0, 1.0, 0.0);
        return s;
    }
    let b = rtLightBase(s.index);
    let r0 = rtSceneTable[b];
    let r1 = rtSceneTable[b + 1u];
    let r2 = rtSceneTable[b + 2u];
    let r4 = rtSceneTable[b + 4u];
    s.direction = bitcast<vec3f>(r0.xyz);
    s.angularRadius = bitcast<f32>(r1.x);
    s.cosAngularRadius = bitcast<f32>(r1.y);
    s.solidAngle = bitcast<f32>(r1.z);
    s.irradiance = bitcast<vec3f>(r4.xyz);
    s.pmf = bitcast<f32>(r2.w);
    return s;
}

fn rtHasEnvironment() -> bool {
    return rtSceneTable[2].z != RT_NONE;
}

fn rtEnvironment() -> RtEnvironment {
    var e: RtEnvironment;
    e.index = rtSceneTable[2].z;
    if (e.index == RT_NONE) {
        e.mapSlot = RT_NONE;
        return e;
    }
    let b = rtLightBase(e.index);
    let r1 = rtSceneTable[b + 1u];
    let r2 = rtSceneTable[b + 2u];
    let r3 = rtSceneTable[b + 3u];
    let r4 = rtSceneTable[b + 4u];
    e.radiance = bitcast<vec3f>(r4.xyz);
    e.scale = bitcast<f32>(r1.x);
    e.rotationY = bitcast<f32>(r1.y);
    e.mapSlot = r3.w;
    e.mapBase = r2.x;
    e.mapW = r2.y & 0xffffu;
    e.mapH = r2.y >> 16u;
    e.pdfNorm = bitcast<f32>(r2.z);
    e.pmf = bitcast<f32>(r2.w);
    return e;
}

fn rtSampleLight(u: f32) -> RtLightSample {
    let n = rtSceneTable[1].z;
    if (n == 0u) {
        return RtLightSample(RT_NONE, 0.0, u);
    }
    let x = u * f32(n);
    let i = min(u32(x), n - 1u);
    let frac = clamp(x - f32(i), 0.0, 0.99999994);
    let entry = rtSceneTable[rtLightBase(i) + 5u];
    let prob = bitcast<f32>(entry.z);
    let keep = frac < prob;
    let pick = select(entry.w, i, keep);
    let remap = select((frac - prob) / (1.0 - prob), frac / prob, keep);
    return RtLightSample(pick, rtLightPmf(pick), min(remap, 0.99999994));
}

fn rtLightIndexOfHit(instance: u32, triangle: u32) -> u32 {
    if (instance >= rtSceneTable[0].x) {
        return RT_NONE;
    }
    let record = ${SCENE_HEADER_VEC4S}u + instance * ${SURFACE_TABLE.INSTANCE_VEC4S}u;
    let mrow = rtSceneTable[record + ${SURFACE_TABLE.INSTANCE_META}u];
    if (mrow.z == RT_LIGHT_MAP_DENSE) {
        return mrow.w + triangle;
    }
    if (mrow.z == RT_LIGHT_MAP_TABLE) {
        return rtSceneData[rtSceneTable[record + ${INSTANCE_LIGHT.spareVec4}u].x + triangle];
    }
    return RT_NONE;
}

fn rtLightPmfOfHit(instance: u32, triangle: u32) -> f32 {
    let index = rtLightIndexOfHit(instance, triangle);
    if (index == RT_NONE) {
        return 0.0;
    }
    return rtLightPmf(index);
}
`;
}

/** The texture arrays, their samplers and the lookups. One `texture_2d_array` + sampler pair per bucket; `rtTexSample` switches on the slot's bucket. */
export function rtTexturesWgsl(o: { group: number; firstBinding: number; layout: TextureLayout; packing?: SceneCopyPacking }): string {
  const n = o.layout.buckets.length;
  const first = o.firstBinding + rtSceneBufferBindings(o.packing);
  const decls: string[] = [];
  for (let i = 0; i < n; i++) decls.push(`@group(${o.group}) @binding(${first + i}) var rtTex${i}: texture_2d_array<f32>;`);
  for (let i = 0; i < n; i++) decls.push(`@group(${o.group}) @binding(${first + n + i}) var rtSmp${i}: sampler;`);
  const cases = Array.from({ length: n }, (_, i) => `        case ${i}u: {\n            return textureSampleLevel(rtTex${i}, rtSmp${i}, uv, layer, lod);\n        }`).join('\n');
  return /* wgsl */ `
${decls.join('\n')}

fn rtTexRecord(slot: u32) -> vec4u {
    return rtSceneTable[rtSceneTable[3].z + slot];
}

fn rtTexSampleRecord(r: vec4u, uv: vec2f, lod: f32) -> vec4f {
    let layer = i32(r.y);
    switch (r.x) {
${cases}
        default: {
            return vec4f(1.0);
        }
    }
}

fn rtTexSample(slot: u32, uv: vec2f, lod: f32) -> vec4f {
    return rtTexSampleRecord(rtTexRecord(slot), uv, lod);
}

fn rtTexLodOf(r: vec4u, footprint: f32) -> f32 {
    let texels = footprint * sqrt(f32((r.z & 0xffffu) * (r.z >> 16u)));
    return max(log2(max(texels, 1.0e-8)), 0.0);
}

fn rtApplyMaterialTextures(m: RtMaterial, uv: vec2f, footprint: f32) -> RtMaterial {
    var o = m;
    if (m.baseColorTexture != RT_TEXTURE_NONE) {
        let r = rtTexRecord(m.baseColorTexture);
        let t = rtTexSampleRecord(r, uv, rtTexLodOf(r, footprint));
        o.baseColor = m.baseColor * t.rgb;
        o.opacity = m.opacity * t.a;
    }
    if (m.roughnessMetalTexture != RT_TEXTURE_NONE) {
        let r = rtTexRecord(m.roughnessMetalTexture);
        let t = rtTexSampleRecord(r, uv, rtTexLodOf(r, footprint));
        o.specularRoughness = m.specularRoughness * t.g;
        o.metalness = m.metalness * t.b;
    }
    if (m.emissionTexture != RT_TEXTURE_NONE) {
        let r = rtTexRecord(m.emissionTexture);
        let t = rtTexSampleRecord(r, uv, rtTexLodOf(r, footprint));
        o.emissionColor = m.emissionColor * t.rgb;
    }
    return o;
}

fn rtApplyNormalMap(n: vec3f, tangent: vec4f, texel: vec3f) -> vec3f {
    let m = texel * 2.0 - 1.0;
    let ml = dot(m, m);
    let d = tangent.xyz - n * dot(n, tangent.xyz);
    let dl = dot(d, d);
    if (ml < 1.0e-12 || dl < 1.0e-12) {
        return n;
    }
    let t = d * inverseSqrt(dl);
    let b = cross(n, t) * tangent.w;
    return normalize(t * m.x + b * m.y + n * m.z);
}
`;
}

/** The scene module's WGSL, without `rtSurface` (modules.ts adds it with `rtSurfaceWgsl`). */
export function rtSceneWgsl(o: SceneWgslOptions): string {
  let source = rtSceneBindingsWgsl(o);
  if (o.materials ?? true) source += rtMaterialWgsl();
  if (o.lights ?? true) source += rtLightsWgsl();
  if ((o.lights ?? true) && o.sampling) source += rtLightSampleWgsl();
  if (o.textures) {
    if (!(o.materials ?? true)) throw new Error('rtSceneWgsl: textures need materials');
    source += rtTexturesWgsl({ group: o.group, firstBinding: o.firstBinding, layout: o.textures, ...(o.packing ? { packing: o.packing } : {}) });
  }
  return (o.packing ?? DEFAULT_SCENE_PACKING) === 'packed' ? rtPackSceneAccess(source) : source;
}
