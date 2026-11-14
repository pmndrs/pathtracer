/**
 * WGSL for the scene copy (prefix `rt`): the two bindings, `rtMaterial`, the light functions and the data accessors
 * `rtSurface` uses. Generated text carries no comments (the module is near the ~50 KB kernel cap; see modules.ts
 * `stripComments`): the documentation is here and in constants.ts / material.ts / lights.ts.
 *
 * Bindings (group/binding from the caller, two consecutive): `rtSceneTable: array<vec4u>`, `rtSceneData: array<u32>`.
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
 */
import { INSTANCE_LIGHT, LIGHT_FLAG, LIGHT_KIND, LIGHT_MAP_MODE, LIGHT_RECORD_VEC4S, SCENE_HEADER_VEC4S, SCENE_NONE } from './constants.ts';
import { MATERIAL_FLAG, MATERIAL_RECORD_VEC4S } from './material.ts';
import { rtLightSampleWgsl } from './light-sample.ts';
import { SURFACE_TABLE } from '../trace/constants.ts';

export interface SceneWgslOptions {
  group: number;
  firstBinding: number;
  /** Emit `rtMaterial`. Default true. */
  materials?: boolean;
  /** Emit the light functions. Default true. */
  lights?: boolean;
  /** Also emit the light point sampling (`light-sample.ts`: `rtSampleLightLi` and friends). Needs `lights`. Default false (kernel size). */
  sampling?: boolean;
}

const hex = (value: number): string => `0x${value.toString(16)}u`;

/** The bindings and the generic accessors (always emitted). */
export function rtSceneBindingsWgsl(o: SceneWgslOptions): string {
  return /* wgsl */ `
const RT_NONE: u32 = ${hex(SCENE_NONE)};
@group(${o.group}) @binding(${o.firstBinding}) var<storage, read> rtSceneTable: array<vec4u>;
@group(${o.group}) @binding(${o.firstBinding + 1}) var<storage, read> rtSceneData: array<u32>;
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

/** The scene module's WGSL, without `rtSurface` (modules.ts adds it with `rtSurfaceWgsl`). */
export function rtSceneWgsl(o: SceneWgslOptions): string {
  let source = rtSceneBindingsWgsl(o);
  if (o.materials ?? true) source += rtMaterialWgsl();
  if (o.lights ?? true) source += rtLightsWgsl();
  if ((o.lights ?? true) && o.sampling) source += rtLightSampleWgsl();
  return source;
}
