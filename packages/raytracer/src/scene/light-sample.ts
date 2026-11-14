/**
 * Light point sampling (P1.4g): the second half of next-event estimation. `rtSampleLight` (lights.ts, alias table) picks a
 * light; this file turns a picked light and a shading point into a direction, a radiance and a solid-angle density.
 *
 *  - **Emissive triangle**: a uniformly distributed point on the triangle (pbrt's `SampleUniformTriangle` warp, area pdf
 *    1/area), converted to solid angle: `pdf = dist^2 / (|cos_light| * area)`. One-sided lights return nothing from behind
 *    (`cos_light <= 0`). pbrt-v4 switches to spherical-triangle sampling for large solid angles; we do not (DIVERGES V34).
 *  - **Sun**: a direction uniform inside the cone of its angular radius (pdf `1 / solidAngle`, radiance = irradiance / solidAngle).
 *    A sun of radius 0 is a delta direction: radiance = irradiance, `delta = true`, `pdf = 1` (the integrator takes no MIS
 *    weight for it). The pbrt/Mitsuba references treat every sun as delta (DIVERGES V35).
 *  - **Environment (constant radiance)**: a uniform direction on the sphere, `pdf = 1 / (4 pi)`. Radiance = `radiance * scale`.
 *    Cosine sampling would need the surface normal; it is not implemented (an open question in the integrator design).
 *
 * Every `pdf` here is a solid-angle density that does NOT include the probability of picking the light (`LightSample.pmf`):
 * the integrator multiplies the two where it needs `p_light = pmf * pdf` (MIS). `lightPdfLi` is the density of generating a
 * direction through the sampler, for MIS on a BSDF-sampled hit.
 *
 * This is the TypeScript twin (f64) of the WGSL in `rtLightSampleWgsl()`; the unit tests check pdf consistency and Monte
 * Carlo estimates against closed forms, the `pt-check` page compares the GPU against it.
 */
import type { RtEnvironmentLight, RtLight, RtSunLight, RtTriangleLight, Vec3Tuple } from './lights.ts';
import { ENV_TEXEL_WORDS } from './env-map.ts';
import { LUMA } from './constants.ts';

export type Rgb = [number, number, number];

/** What `sampleLightLi` returns. `valid = false`: no contribution (behind a one-sided light, zero area, zero distance). */
export interface LightLi {
  valid: boolean;
  /** Unit direction from the shading point toward the light sample. */
  wi: Vec3Tuple;
  /** Distance to the sample (Infinity for the sun and the environment). */
  distance: number;
  /** World position of the sample on a triangle light; `p + wi * distance` otherwise (NaN for infinite lights). */
  point: Vec3Tuple;
  /** Emitting normal at the sample (triangle lights), zero otherwise. */
  normal: Vec3Tuple;
  /** Emitted radiance toward the shading point, W/sr/m^2. */
  radiance: Rgb;
  /** Solid-angle density of the sample, not including the light-selection pmf. 1 for a delta light. */
  pdf: number;
  /** Delta light (sun of radius 0): `pdf` is not a density, MIS does not apply. */
  delta: boolean;
}

const INVALID: LightLi = { valid: false, wi: [0, 0, 1], distance: 0, point: [0, 0, 0], normal: [0, 0, 0], radiance: [0, 0, 0], pdf: 0, delta: false };

const sub = (a: Vec3Tuple, b: Vec3Tuple): Vec3Tuple => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a: Vec3Tuple, b: Vec3Tuple): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const scale = (a: Vec3Tuple, s: number): Vec3Tuple => [a[0] * s, a[1] * s, a[2] * s];
const add = (a: Vec3Tuple, b: Vec3Tuple): Vec3Tuple => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];

/** pbrt-v4 `SampleUniformTriangle`: (b1, b2) barycentric weights of vertices 1 and 2, uniform over the triangle. */
export function sampleUniformTriangle(u0: number, u1: number): [number, number] {
  let b0: number;
  let b1: number;
  if (u0 < u1) {
    b0 = u0 / 2;
    b1 = u1 - b0;
  } else {
    b1 = u1 / 2;
    b0 = u0 - b1;
  }
  return [b1, 1 - b0 - b1];
}

/** Orthonormal tangents of a unit vector (Duff et al. 2017). */
export function orthonormalBasis(n: Vec3Tuple): [Vec3Tuple, Vec3Tuple] {
  const s = n[2] >= 0 ? 1 : -1;
  const a = -1 / (s + n[2]);
  const b = n[0] * n[1] * a;
  return [
    [1 + s * n[0] * n[0] * a, s * b, -s * n[0]],
    [b, s + n[1] * n[1] * a, -n[1]],
  ];
}

/** Solid-angle density of sampling a point at `distance` on a surface of `area` whose normal makes `cosLight` with the direction away from the shading point. */
export function triangleSolidAnglePdf(area: number, distance: number, cosLight: number): number {
  const c = Math.abs(cosLight);
  if (!(c > 0) || !(area > 0)) return 0;
  return (distance * distance) / (c * area);
}

/** The point on a triangle light for uniform `u`. */
export function triangleLightPoint(light: RtTriangleLight, u0: number, u1: number): Vec3Tuple {
  const [b1, b2] = sampleUniformTriangle(u0, u1);
  return add(light.p0, add(scale(light.edge1, b1), scale(light.edge2, b2)));
}

export function sampleTriangleLi(light: RtTriangleLight, p: Vec3Tuple, u0: number, u1: number): LightLi {
  const point = triangleLightPoint(light, u0, u1);
  const d = sub(point, p);
  const dist2 = dot(d, d);
  if (!(dist2 > 0)) return INVALID;
  const distance = Math.sqrt(dist2);
  const wi = scale(d, 1 / distance);
  const cosLight = -dot(light.normal, wi);
  if (!light.twoSided && cosLight <= 0) return INVALID;
  const pdf = triangleSolidAnglePdf(light.area, distance, cosLight);
  if (!(pdf > 0) || !Number.isFinite(pdf)) return INVALID;
  return { valid: true, wi, distance, point, normal: light.normal, radiance: [...light.radiance], pdf, delta: false };
}

/** `1 - cos(angularRadius)` from the stored solid angle (exact for tiny radii, where `1 - cos` in f32 would lose most digits). */
export function sunOneMinusCos(sun: RtSunLight): number {
  return sun.solidAngle / (2 * Math.PI);
}

export function sampleSunLi(sun: RtSunLight, u0: number, u1: number): LightLi {
  if (!(sun.solidAngle > 0)) {
    return { valid: true, wi: [...sun.direction], distance: Infinity, point: [NaN, NaN, NaN], normal: [0, 0, 0], radiance: [...sun.irradiance], pdf: 1, delta: true };
  }
  const omc = sunOneMinusCos(sun);
  const t = u0 * omc; // 1 - cos(theta)
  const cosT = 1 - t;
  const sinT = Math.sqrt(Math.max(0, t * (2 - t)));
  const phi = 2 * Math.PI * u1;
  const [b0, b1] = orthonormalBasis(sun.direction);
  const wi = add(scale(sun.direction, cosT), add(scale(b0, sinT * Math.cos(phi)), scale(b1, sinT * Math.sin(phi))));
  const radiance: Rgb = [sun.irradiance[0] / sun.solidAngle, sun.irradiance[1] / sun.solidAngle, sun.irradiance[2] / sun.solidAngle];
  return { valid: true, wi, distance: Infinity, point: [NaN, NaN, NaN], normal: [0, 0, 0], radiance, pdf: 1 / sun.solidAngle, delta: false };
}

/** True when `dir` (unit) lies inside the sun's disc. Uses |dir - sunDir|^2 = 2 (1 - cos), exact where `dot` would not be. */
export function sunContains(sun: RtSunLight, dir: Vec3Tuple): boolean {
  if (!(sun.solidAngle > 0)) return false;
  const d = sub(dir, sun.direction);
  return dot(d, d) <= 2 * sunOneMinusCos(sun);
}

/** Radiance seen looking at the sun's disc. */
export function sunRadiance(sun: RtSunLight): Rgb {
  if (!(sun.solidAngle > 0)) return [0, 0, 0];
  return [sun.irradiance[0] / sun.solidAngle, sun.irradiance[1] / sun.solidAngle, sun.irradiance[2] / sun.solidAngle];
}

export function environmentRadiance(env: RtEnvironmentLight): Rgb {
  return [env.radiance[0] * env.scale, env.radiance[1] * env.scale, env.radiance[2] * env.scale];
}

export const ENV_UNIFORM_PDF = 1 / (4 * Math.PI);

export function sampleEnvironmentLi(env: RtEnvironmentLight, u0: number, u1: number): LightLi {
  const z = 1 - 2 * u0;
  const r = Math.sqrt(Math.max(0, 1 - z * z));
  const phi = 2 * Math.PI * u1;
  const wi: Vec3Tuple = [r * Math.cos(phi), r * Math.sin(phi), z];
  return { valid: true, wi, distance: Infinity, point: [NaN, NaN, NaN], normal: [0, 0, 0], radiance: environmentRadiance(env), pdf: ENV_UNIFORM_PDF, delta: false };
}

/** Samples a light seen from `p`. `u` = two uniform numbers in [0, 1). */
export function sampleLightLi(light: RtLight, p: Vec3Tuple, u0: number, u1: number): LightLi {
  switch (light.kind) {
    case 'triangle':
      return sampleTriangleLi(light, p, u0, u1);
    case 'sun':
      return sampleSunLi(light, u0, u1);
    case 'environment':
      return sampleEnvironmentLi(light, u0, u1);
  }
}

/**
 * The solid-angle density with which `sampleLightLi` produces the direction `wi` toward `light`, given where a ray along
 * `wi` meets the light: `hitDistance` and the cosine between the light's emitting normal and `-wi` (triangle lights only).
 * 0 for a delta sun (it cannot be hit) and for directions outside a sun's disc.
 */
export function lightPdfLi(light: RtLight, wi: Vec3Tuple, hitDistance = 0, cosLight = 0): number {
  switch (light.kind) {
    case 'triangle':
      return triangleSolidAnglePdf(light.area, hitDistance, cosLight);
    case 'sun':
      return sunContains(light, wi) ? 1 / light.solidAngle : 0;
    case 'environment':
      return ENV_UNIFORM_PDF;
  }
}

// ---------------------------------------------------------------------------------------------------------------
// WGSL twin
// ---------------------------------------------------------------------------------------------------------------

const f32 = (v: number): string => {
  const s = Math.fround(v).toString();
  return /[.e]/.test(s) ? s : `${s}.0`;
};

/**
 * WGSL for the functions above (prefix `rt`). Needs the scene module's light functions (`rtLight`, `rtSun`, `rtEnvironment`);
 * append it to `rtModuleWgsl(caps, { scene: { ... lights } })`. No comments in the generated text (kernel size cap).
 *
 *     struct RtLightLi { wi, distance, point, pdf, normal, radiance, delta, valid }
 *     fn rtSampleLightLi(index: u32, p: vec3f, u: vec2f) -> RtLightLi    dispatches on the record's kind
 *     fn rtTriangleSolidAnglePdf(area, distance, cosLight) -> f32
 *     fn rtSunContains(s: RtSun, dir: vec3f) -> bool, rtSunRadiance(s) -> vec3f, rtSunPdf(s, dir) -> f32
 *     fn rtEnvironmentRadiance(e) -> vec3f, RT_ENV_UNIFORM_PDF
 */

export interface LightSampleWgslOptions {
  /** Compile the environment-map lookup / pdf / sampling (a scene with an `hdri` environment). Default false: the constant radiance only. */
  envMap?: boolean;
  /** With `envMap`: sample the map by its luminance (`true`) or uniformly over the sphere (`false`, the A/B baseline). Default true. */
  envImportance?: boolean;
}

/** WGSL of the environment map (needs `envMap`): bilinear lookup, solid-angle pdf and importance sampling; twins in env-map.ts. */
export function rtEnvMapWgsl(importance: boolean): string {
  return /* wgsl */ `
const RT_INV_TWO_PI: f32 = ${f32(1 / (2 * Math.PI))};
const RT_INV_PI: f32 = ${f32(1 / Math.PI)};
const RT_PI: f32 = ${f32(Math.PI)};
const RT_ENV_IMPORTANCE: bool = ${importance};

fn rtEnvToMap(e: RtEnvironment, d: vec3f) -> vec3f {
    let c = cos(e.rotationY);
    let s = sin(e.rotationY);
    return vec3f(c * d.x - s * d.z, d.y, s * d.x + c * d.z);
}

fn rtEnvFromMap(e: RtEnvironment, m: vec3f) -> vec3f {
    let c = cos(e.rotationY);
    let s = sin(e.rotationY);
    return vec3f(c * m.x + s * m.z, m.y, -s * m.x + c * m.z);
}

fn rtEnvUv(m: vec3f) -> vec2f {
    let u = atan2(m.x, -m.z) * RT_INV_TWO_PI;
    return vec2f(u - floor(u), acos(clamp(m.y, -1.0, 1.0)) * RT_INV_PI);
}

fn rtEnvTexel(e: RtEnvironment, i: u32, j: u32) -> vec3f {
    let at = e.mapBase + e.mapH + (j * e.mapW + i) * ${ENV_TEXEL_WORDS}u;
    return vec3f(bitcast<f32>(rtSceneData[at]), bitcast<f32>(rtSceneData[at + 1u]), bitcast<f32>(rtSceneData[at + 2u]));
}

fn rtEnvMapLookup(e: RtEnvironment, d: vec3f) -> vec3f {
    let uv = rtEnvUv(rtEnvToMap(e, d));
    let fx = uv.x * f32(e.mapW) - 0.5;
    let fy = uv.y * f32(e.mapH) - 0.5;
    let x0 = floor(fx);
    let y0 = floor(fy);
    let tx = fx - x0;
    let ty = fy - y0;
    let w = i32(e.mapW);
    let xa = u32(((i32(x0) % w) + w) % w);
    let xb = u32(((i32(x0) + 1) % w + w) % w);
    let ya = u32(clamp(i32(y0), 0, i32(e.mapH) - 1));
    let yb = u32(clamp(i32(y0) + 1, 0, i32(e.mapH) - 1));
    return mix(mix(rtEnvTexel(e, xa, ya), rtEnvTexel(e, xb, ya), tx), mix(rtEnvTexel(e, xa, yb), rtEnvTexel(e, xb, yb), tx), ty);
}

fn rtEnvPixelPdf(e: RtEnvironment, i: u32, j: u32, sinTheta: f32) -> f32 {
    if (!(sinTheta > 1.0e-12)) {
        return 0.0;
    }
    let l = dot(rtEnvTexel(e, i, j), vec3f(${LUMA.map(f32).join(', ')}));
    return l * sin(RT_PI * (f32(j) + 0.5) / f32(e.mapH)) * e.pdfNorm / sinTheta;
}

fn rtEnvMapPdf(e: RtEnvironment, d: vec3f) -> f32 {
    let m = rtEnvToMap(e, d);
    let uv = rtEnvUv(m);
    let i = min(u32(uv.x * f32(e.mapW)), e.mapW - 1u);
    let j = min(u32(uv.y * f32(e.mapH)), e.mapH - 1u);
    return rtEnvPixelPdf(e, i, j, sqrt(max(0.0, 1.0 - m.y * m.y)));
}

fn rtEnvSearch(base: u32, stride: u32, n: u32, u: f32) -> u32 {
    var lo = 0u;
    var hi = n - 1u;
    while (lo < hi) {
        let mid = (lo + hi) >> 1u;
        if (bitcast<f32>(rtSceneData[base + mid * stride]) > u) {
            hi = mid;
        } else {
            lo = mid + 1u;
        }
    }
    return lo;
}

fn rtEnvCdfFrac(base: u32, stride: u32, k: u32, u: f32) -> f32 {
    let lo = select(0.0, bitcast<f32>(rtSceneData[base + (k - 1u) * stride]), k > 0u);
    let hi = bitcast<f32>(rtSceneData[base + k * stride]);
    return select(0.5, clamp((u - lo) / (hi - lo), 0.0, 0.99999994), hi > lo);
}

fn rtSampleEnvMap(e: RtEnvironment, u: vec2f) -> RtLightLi {
    let j = rtEnvSearch(e.mapBase, 1u, e.mapH, u.y);
    let dv = rtEnvCdfFrac(e.mapBase, 1u, j, u.y);
    let rowBase = e.mapBase + e.mapH + j * e.mapW * ${ENV_TEXEL_WORDS}u + 3u;
    let i = rtEnvSearch(rowBase, ${ENV_TEXEL_WORDS}u, e.mapW, u.x);
    let du = rtEnvCdfFrac(rowBase, ${ENV_TEXEL_WORDS}u, i, u.x);
    let phi = RT_TWO_PI * (f32(i) + du) / f32(e.mapW);
    let theta = RT_PI * (f32(j) + dv) / f32(e.mapH);
    let st = sin(theta);
    let wi = rtEnvFromMap(e, vec3f(st * sin(phi), cos(theta), -st * cos(phi)));
    return RtLightLi(wi, 3.0e38, vec3f(0.0), rtEnvPixelPdf(e, i, j, st), vec3f(0.0), false, rtEnvMapLookup(e, wi) * e.scale, true);
}

fn rtEnvironmentLe(e: RtEnvironment, dir: vec3f) -> vec3f {
    if (e.mapW == 0u) {
        return rtEnvironmentRadiance(e);
    }
    return rtEnvMapLookup(e, dir) * e.scale;
}

fn rtEnvironmentPdf(e: RtEnvironment, dir: vec3f) -> f32 {
    if (RT_ENV_IMPORTANCE && e.mapW > 0u) {
        return rtEnvMapPdf(e, dir);
    }
    return RT_ENV_UNIFORM_PDF;
}

fn rtSampleEnvironmentLi(e: RtEnvironment, u: vec2f) -> RtLightLi {
    if (RT_ENV_IMPORTANCE && e.mapW > 0u) {
        return rtSampleEnvMap(e, u);
    }
    let z = 1.0 - 2.0 * u.x;
    let r = sqrt(max(0.0, 1.0 - z * z));
    let phi = RT_TWO_PI * u.y;
    let wi = vec3f(r * cos(phi), r * sin(phi), z);
    return RtLightLi(wi, 3.0e38, vec3f(0.0), RT_ENV_UNIFORM_PDF, vec3f(0.0), false, rtEnvironmentLe(e, wi), true);
}
`;
}

const ENV_CONSTANT_WGSL = /* wgsl */ `
fn rtEnvironmentLe(e: RtEnvironment, dir: vec3f) -> vec3f {
    return rtEnvironmentRadiance(e);
}

fn rtEnvironmentPdf(e: RtEnvironment, dir: vec3f) -> f32 {
    return RT_ENV_UNIFORM_PDF;
}

fn rtSampleEnvironmentLi(e: RtEnvironment, u: vec2f) -> RtLightLi {
    let z = 1.0 - 2.0 * u.x;
    let r = sqrt(max(0.0, 1.0 - z * z));
    let phi = RT_TWO_PI * u.y;
    return RtLightLi(vec3f(r * cos(phi), r * sin(phi), z), 3.0e38, vec3f(0.0), RT_ENV_UNIFORM_PDF, vec3f(0.0), false, rtEnvironmentRadiance(e), true);
}
`;

export function rtLightSampleWgsl(options: LightSampleWgslOptions = {}): string {
  return /* wgsl */ `
const RT_TWO_PI: f32 = ${f32(2 * Math.PI)};
const RT_ENV_UNIFORM_PDF: f32 = ${f32(ENV_UNIFORM_PDF)};

struct RtLightLi {
    wi: vec3f,
    distance: f32,
    point: vec3f,
    pdf: f32,
    normal: vec3f,
    delta: bool,
    radiance: vec3f,
    valid: bool,
}

fn rtLightLiNone() -> RtLightLi {
    return RtLightLi(vec3f(0.0, 0.0, 1.0), 0.0, vec3f(0.0), 0.0, vec3f(0.0), false, vec3f(0.0), false);
}

fn rtOrthonormalA(n: vec3f) -> vec3f {
    let s = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    return vec3f(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
}

fn rtOrthonormalB(n: vec3f) -> vec3f {
    let s = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    return vec3f(b, s + n.y * n.y * a, -n.y);
}

fn rtSampleUniformTriangle(u: vec2f) -> vec2f {
    var b0: f32;
    var b1: f32;
    if (u.x < u.y) {
        b0 = u.x * 0.5;
        b1 = u.y - b0;
    } else {
        b1 = u.y * 0.5;
        b0 = u.x - b1;
    }
    return vec2f(b1, 1.0 - b0 - b1);
}

fn rtTriangleSolidAnglePdf(area: f32, distance: f32, cosLight: f32) -> f32 {
    let c = abs(cosLight);
    if (!(c > 0.0) || !(area > 0.0)) {
        return 0.0;
    }
    return distance * distance / (c * area);
}

fn rtSampleTriangleLi(l: RtLight, p: vec3f, u: vec2f) -> RtLightLi {
    let b = rtSampleUniformTriangle(u);
    let point = l.position + l.edge1 * b.x + l.edge2 * b.y;
    let d = point - p;
    let dist2 = dot(d, d);
    if (!(dist2 > 0.0)) {
        return rtLightLiNone();
    }
    let distance = sqrt(dist2);
    let wi = d / distance;
    let cosLight = -dot(l.normal, wi);
    if (!l.twoSided && cosLight <= 0.0) {
        return rtLightLiNone();
    }
    let pdf = rtTriangleSolidAnglePdf(l.area, distance, cosLight);
    if (!(pdf > 0.0) || pdf > 3.0e38) {
        return rtLightLiNone();
    }
    return RtLightLi(wi, distance, point, pdf, l.normal, false, l.radiance, true);
}

fn rtSunOneMinusCos(s: RtSun) -> f32 {
    return s.solidAngle / RT_TWO_PI;
}

fn rtSunRadiance(s: RtSun) -> vec3f {
    return select(vec3f(0.0), s.irradiance / s.solidAngle, s.solidAngle > 0.0);
}

fn rtSunContains(s: RtSun, dir: vec3f) -> bool {
    let d = dir - s.direction;
    return s.solidAngle > 0.0 && dot(d, d) <= 2.0 * rtSunOneMinusCos(s);
}

fn rtSunPdf(s: RtSun, dir: vec3f) -> f32 {
    return select(0.0, 1.0 / s.solidAngle, rtSunContains(s, dir));
}

fn rtSampleSunLi(s: RtSun, u: vec2f) -> RtLightLi {
    if (!(s.solidAngle > 0.0)) {
        return RtLightLi(s.direction, 3.0e38, vec3f(0.0), 1.0, vec3f(0.0), true, s.irradiance, true);
    }
    let t = u.x * rtSunOneMinusCos(s);
    let cosT = 1.0 - t;
    let sinT = sqrt(max(0.0, t * (2.0 - t)));
    let phi = RT_TWO_PI * u.y;
    let wi = s.direction * cosT + (rtOrthonormalA(s.direction) * cos(phi) + rtOrthonormalB(s.direction) * sin(phi)) * sinT;
    return RtLightLi(wi, 3.0e38, vec3f(0.0), 1.0 / s.solidAngle, vec3f(0.0), false, s.irradiance / s.solidAngle, true);
}

fn rtEnvironmentRadiance(e: RtEnvironment) -> vec3f {
    return e.radiance * e.scale;
}

fn rtSampleLightLi(index: u32, p: vec3f, u: vec2f) -> RtLightLi {
    let l = rtLight(index);
    if (l.kind == RT_LIGHT_TRIANGLE) {
        return rtSampleTriangleLi(l, p, u);
    }
    if (l.kind == RT_LIGHT_SUN) {
        return rtSampleSunLi(rtSun(), u);
    }
    return rtSampleEnvironmentLi(rtEnvironment(), u);
}
` + (options.envMap ? rtEnvMapWgsl(options.envImportance ?? true) : ENV_CONSTANT_WGSL);
}
