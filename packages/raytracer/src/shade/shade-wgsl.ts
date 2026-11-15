/**
 * WGSL for L2 `rtShade` (prefix `rt`). TypeScript twin and options: `shade.ts`. Needs, before it in the same module: the bvh walk, the
 * core (`rtVisible`, `rtSegment`), the scene copy's materials and lights with light sampling, and `rtSurface`; `rtModuleWgsl(caps,
 * { scene, shade })` assembles all of that. No comments in the output (kernel size cap).
 *
 *     fn rtShade(hit: RtHit, ray: RtRay, mode: u32, u: vec3f, mask: u32) -> RtShaded
 *         radiance at a hit (or, for a miss, the environment) toward -ray.direction. mode = RT_SHADE_FLAT | RT_SHADE_LAMBERT | RT_SHADE_FULL.
 *         u = (light selection, light point u, light point v). At most one shadow ray (none in FLAT). `mask` is the shadow ray's instance mask.
 *     fn rtShadePoint(hit, ray) -> RtShadePoint        the surface decode `rtShade` does (position, normals, material, albedo); hooks get it
 *     fn rtShadeAt(p: RtShadePoint, mode, u, mask) -> RtShaded   shade a point you already decoded
 *     fn rtShadeTotal(s: RtShaded) -> vec3f           emission + direct + indirect
 *     fn rtShadeMiss(direction) -> vec3f              environment radiance (+ the sun disc with `sunVisible`)
 *     fn rtShadeDiffuseFromIrradiance(p, E) -> vec3f  helper for caches that store irradiance
 *     fn rtOffsetOrigin(p, n) -> vec3f, rtSideNormal(n, w) -> vec3f   self-intersection-free ray origin (Waechter and Binder)
 *
 * Hooks (see docs/design/shade.md):
 *     fn rtShadeIndirect(p: RtShadePoint) -> vec3f          default: black. `cache: 'external'` = you declare it.
 *     fn rtShadeBsdf(p: RtShadePoint, wi: vec3f) -> vec3f   `bsdf: 'external'` = you declare it (f, no cosine). Used by FULL only.
 */
import { resolveShadeOptions, RT_SHADE_MODE, type RtShadeOptions } from './shade.ts';

const f32 = (v: number): string => {
  const s = Math.fround(v).toString();
  return /[.e]/.test(s) ? s : `${s}.0`;
};

export function rtShadeWgsl(options: RtShadeOptions = {}): string {
  const o = resolveShadeOptions(options);
  const bsdf =
    o.bsdf === 'external'
      ? ''
      : /* wgsl */ `
fn rtShadeFresnel(m: RtMaterial, voh: f32) -> vec3f {
    let d = (m.ior - 1.0) / (m.ior + 1.0);
    let w1 = 1.0 - voh;
    let w2 = w1 * w1;
    let w = w2 * w2 * w1;
    let dielectric = m.specularColor * (m.specularWeight * (d * d + (1.0 - d * d) * w));
    var f0m = m.baseColor * m.baseWeight;
    if (m.conductorEtaK) {
        let a = m.conductorEta - vec3f(1.0);
        let b = m.conductorEta + vec3f(1.0);
        f0m = (a * a + m.conductorK * m.conductorK) / (b * b + m.conductorK * m.conductorK);
    }
    let metal = f0m + (vec3f(1.0) - f0m) * w;
    return mix(dielectric, metal, m.metalness);
}

fn rtShadeLambda(c: f32, a2: f32) -> f32 {
    let c2 = c * c;
    return (sqrt(1.0 + a2 * (1.0 - c2) / c2) - 1.0) * 0.5;
}

fn rtShadeBsdf(p: RtShadePoint, wi: vec3f) -> vec3f {
    let nol = dot(p.normal, wi);
    if (!(nol > 0.0)) {
        return vec3f(0.0);
    }
    let nov = max(dot(p.normal, p.wo), 0.0001);
    let hRaw = p.wo + wi;
    let hl = length(hRaw);
    let diffuse = p.albedo * RT_SHADE_INV_PI;
    if (!(hl > 0.0)) {
        return diffuse;
    }
    let h = hRaw / hl;
    let noh = clamp(dot(p.normal, h), 0.0, 1.0);
    let voh = clamp(dot(p.wo, h), 0.0, 1.0);
    let alpha = max(rtMaterialAlpha(p.material), RT_SHADE_MIN_ALPHA);
    let a2 = alpha * alpha;
    let dd = noh * noh * (a2 - 1.0) + 1.0;
    let ggx = a2 / (RT_SHADE_PI * dd * dd);
    let g = 1.0 / (1.0 + rtShadeLambda(nov, a2) + rtShadeLambda(nol, a2));
    return diffuse + rtShadeFresnel(p.material, voh) * (ggx * g / (4.0 * nov * nol));
}
`;
  const indirect =
    o.cache === 'external'
      ? ''
      : /* wgsl */ `
fn rtShadeIndirect(p: RtShadePoint) -> vec3f {
    return vec3f(0.0);
}
`;
  return /* wgsl */ `
const RT_SHADE_FLAT: u32 = ${RT_SHADE_MODE.flat}u;
const RT_SHADE_LAMBERT: u32 = ${RT_SHADE_MODE.lambert}u;
const RT_SHADE_FULL: u32 = ${RT_SHADE_MODE.full}u;
const RT_SHADE_PI: f32 = ${f32(Math.PI)};
const RT_SHADE_INV_PI: f32 = ${f32(1 / Math.PI)};
const RT_SHADE_MIN_ALPHA: f32 = ${f32(o.minAlpha)};
const RT_SHADE_ENV_COSINE: bool = ${o.envSampling === 'cosine'};
const RT_SHADE_FLAT_ALBEDO: bool = ${o.flat === 'albedo'};
const RT_SHADE_SUN_VISIBLE: bool = ${o.sunVisible};
const RT_SHADE_FAR: f32 = 1.0e30;
const RT_ORIGIN_OFFSET: f32 = 0.03125;
const RT_FLOAT_SCALE: f32 = 0.0000152587890625;
const RT_INT_SCALE: f32 = 256.0;

struct RtShadePoint {
    position: vec3f,
    normal: vec3f,
    geometricNormal: vec3f,
    wo: vec3f,
    frontFace: bool,
    material: RtMaterial,
    albedo: vec3f,
}

struct RtShaded {
    emission: vec3f,
    direct: vec3f,
    indirect: vec3f,
    light: u32,
    occluded: bool,
}

fn rtOffsetOrigin(p: vec3f, n: vec3f) -> vec3f {
    let ofs = vec3i(RT_INT_SCALE * n);
    let pi = bitcast<vec3f>(bitcast<vec3i>(p) + select(ofs, -ofs, p < vec3f(0.0)));
    return select(pi, p + RT_FLOAT_SCALE * n, abs(p) < vec3f(RT_ORIGIN_OFFSET));
}

fn rtSideNormal(n: vec3f, w: vec3f) -> vec3f {
    return select(-n, n, dot(n, w) > 0.0);
}

fn rtShadeAlbedo(m: RtMaterial) -> vec3f {
    return m.baseColor * (m.baseWeight * (1.0 - m.metalness) * (1.0 - m.transmissionWeight));
}

fn rtShadePoint(hit: RtHit, ray: RtRay) -> RtShadePoint {
    let surf = rtSurface(hit, ray, RT_SURF_POSITION | RT_SURF_SHADING_NORMAL);
    var p: RtShadePoint;
    p.position = surf.position;
    p.frontFace = hit.frontFace;
    p.geometricNormal = select(-hit.normal, hit.normal, hit.frontFace);
    p.normal = select(-surf.shadingNormal, surf.shadingNormal, hit.frontFace);
    p.wo = -ray.direction;
    p.material = rtMaterial(hit.material);
    p.albedo = rtShadeAlbedo(p.material);
    return p;
}

fn rtShadeEmission(p: RtShadePoint) -> vec3f {
    let m = p.material;
    return select(vec3f(0.0), m.emissionColor, m.emissionLuminance > 0.0 && (p.frontFace || m.emissionTwoSided));
}

fn rtShadeDiffuseFromIrradiance(p: RtShadePoint, irradiance: vec3f) -> vec3f {
    return p.albedo * irradiance * RT_SHADE_INV_PI;
}
${indirect}${bsdf}
fn rtShadeConcentricDisk(u: vec2f) -> vec2f {
    let a = 2.0 * u.x - 1.0;
    let b = 2.0 * u.y - 1.0;
    if (a == 0.0 && b == 0.0) {
        return vec2f(0.0);
    }
    var r: f32;
    var theta: f32;
    if (abs(a) > abs(b)) {
        r = a;
        theta = 0.78539816339744830962 * (b / a);
    } else {
        r = b;
        theta = 1.57079632679489661923 - 0.78539816339744830962 * (a / b);
    }
    return vec2f(r * cos(theta), r * sin(theta));
}

fn rtShadeSampleEnvCosine(e: RtEnvironment, n: vec3f, u: vec2f) -> RtLightLi {
    let d = rtShadeConcentricDisk(u);
    let z = sqrt(max(0.0, 1.0 - dot(d, d)));
    let wi = rtOrthonormalA(n) * d.x + rtOrthonormalB(n) * d.y + n * z;
    return RtLightLi(wi, 3.0e38, vec3f(0.0), z * RT_SHADE_INV_PI, vec3f(0.0), false, rtEnvironmentRadiance(e), z > 0.0);
}

fn rtShadeMiss(direction: vec3f) -> vec3f {
    var le = vec3f(0.0);
    if (rtHasEnvironment()) {
        le += rtEnvironmentRadiance(rtEnvironment());
    }
    if (RT_SHADE_SUN_VISIBLE && rtHasSun()) {
        let s = rtSun();
        if (rtSunContains(s, direction)) {
            le += rtSunRadiance(s);
        }
    }
    return le;
}

fn rtShadeDirect(p: RtShadePoint, mode: u32, u: vec3f, mask: u32) -> RtShaded {
    var out: RtShaded;
    out.light = RT_NONE;
    let pick = rtSampleLight(u.x);
    if (pick.index == RT_NONE || !(pick.pmf > 0.0)) {
        return out;
    }
    var li: RtLightLi;
    if (RT_SHADE_ENV_COSINE && pick.index == rtSceneTable[2].z) {
        li = rtShadeSampleEnvCosine(rtEnvironment(), p.normal, u.yz);
    } else {
        li = rtSampleLightLi(pick.index, p.position, u.yz);
    }
    if (!li.valid || !(li.pdf > 0.0)) {
        return out;
    }
    var f: vec3f;
    if (mode == RT_SHADE_LAMBERT) {
        f = select(vec3f(0.0), p.albedo * RT_SHADE_INV_PI, dot(p.normal, li.wi) > 0.0);
    } else {
        f = rtShadeBsdf(p, li.wi);
    }
    let w = f * abs(dot(p.normal, li.wi));
    if (!(max(w.x, max(w.y, w.z)) > 0.0)) {
        return out;
    }
    out.light = pick.index;
    let origin = rtOffsetOrigin(p.position, rtSideNormal(p.geometricNormal, li.wi));
    var ray: RtRay;
    if (li.distance < 1.0e30) {
        ray = rtSegment(origin, rtOffsetOrigin(li.point, rtSideNormal(li.normal, -li.wi)), 0.0, 0.0);
    } else {
        ray = RtRay(origin, 0.0, li.wi, RT_SHADE_FAR);
    }
    if (!rtVisible(ray, mask)) {
        out.occluded = true;
        return out;
    }
    out.direct = li.radiance * w * (1.0 / (pick.pmf * li.pdf));
    return out;
}

fn rtShadeAt(p: RtShadePoint, mode: u32, u: vec3f, mask: u32) -> RtShaded {
    var out: RtShaded;
    if (mode == RT_SHADE_FLAT) {
        out.light = RT_NONE;
        if (RT_SHADE_FLAT_ALBEDO) {
            out.direct = p.material.baseColor * p.material.baseWeight;
        }
    } else {
        out = rtShadeDirect(p, mode, u, mask);
    }
    out.emission = rtShadeEmission(p);
    out.indirect = rtShadeIndirect(p);
    return out;
}

fn rtShade(hit: RtHit, ray: RtRay, mode: u32, u: vec3f, mask: u32) -> RtShaded {
    if (hit.t < 0.0) {
        var out: RtShaded;
        out.light = RT_NONE;
        out.emission = rtShadeMiss(ray.direction);
        return out;
    }
    return rtShadeAt(rtShadePoint(hit, ray), mode, u, mask);
}

fn rtShadeTotal(s: RtShaded) -> vec3f {
    return s.emission + s.direct + s.indirect;
}
`;
}
