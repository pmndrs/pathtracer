/**
 * WGSL twin of `bsdf.ts` (prefix `pt`). `ptBsdfWgsl(options)` returns a self-contained module fragment (no bindings,
 * no `enable` lines) to prepend to the integrator. Keep the two files line-for-line parallel: `bench/pages/bsdf-check.ts`
 * compares them on the GPU. Ported from pbrt-v4 (Apache-2.0), see `bsdf.ts` for the attribution and DIVERGES.md V28.
 *
 * Local shading frame: +Z = shading normal, wo/wi unit vectors pointing away from the surface.
 * Radiance is carried as vec3 in here; the integrator widens it to vec4 (spec section 6, spectral-ready).
 */

import type { PtBsdfOptions } from './bsdf.ts';

export function ptBsdfWgsl(options: PtBsdfOptions = {}): string {
  const cap = (options.vndf ?? 'standard') === 'cap';
  const radiance = (options.transport ?? 'radiance') === 'radiance';
  const thin = options.thin ?? false;
  const aniso = options.aniso ?? false;
  return /* wgsl */ `
const PT_PI: f32 = 3.14159265358979323846;
const PT_BSDF_DIFFUSE: u32 = 0u;
const PT_BSDF_CONDUCTOR: u32 = 1u;
const PT_BSDF_DIELECTRIC: u32 = 2u;
const PT_BSDF_BLACK: u32 = 3u;
${thin ? `const PT_BSDF_THIN: u32 = 4u;
` : ''}const PT_FLAG_DELTA: u32 = 1u;
const PT_FLAG_REFLECTION: u32 = 2u;
const PT_FLAG_TRANSMISSION: u32 = 4u;
const PT_SMOOTH_ALPHA: f32 = 1e-3;
const PT_MIN_ALPHA: f32 = 1e-4;
// Settings (generator options): spherical-cap VNDF sampler, radiance (1/etap^2) transport.
const PT_VNDF_CAP: bool = ${cap};
const PT_TRANSPORT_RADIANCE: bool = ${radiance};

struct PtBsdfParams {
  kind: u32,
  ior: f32,
  alpha: vec2f,
  reflectance: vec3f,
  eta: vec3f,
  k: vec3f,
}

struct PtBsdfEval {
  f: vec3f,
  pdf: f32,
}

struct PtBsdfSample {
  wi: vec3f,
  f: vec3f,
  pdf: f32,
  flags: u32,   // 0 = failed
  etap: f32,
}

struct PtFrame {
  t: vec3f,
  b: vec3f,
  n: vec3f,
}

fn ptFrameFromNormal(n: vec3f) -> PtFrame {
  let s = select(-1.0, 1.0, n.z >= 0.0);
  let a = -1.0 / (s + n.z);
  let b = n.x * n.y * a;
  return PtFrame(vec3f(1.0 + s * n.x * n.x * a, s * b, -s * n.x), vec3f(b, s + n.y * n.y * a, -n.y), n);
}
${aniso ? `// P2.26: local +X along the tangent made orthogonal to n, turned by rotation radians toward the bitangent cross(n, t) * tangent.w (glTF anisotropyRotation).
fn ptFrameFromTangent(n: vec3f, tangent: vec4f, rotation: f32) -> PtFrame {
  let x = tangent.xyz - n * dot(n, tangent.xyz);
  let l2 = dot(x, x);
  if (!(l2 > 1e-12)) { return ptFrameFromNormal(n); }
  let t0 = x * inverseSqrt(l2);
  var t = t0;
  if (rotation != 0.0) {
    let bt = cross(n, t0) * select(1.0, -1.0, tangent.w < 0.0);
    t = cos(rotation) * t0 + sin(rotation) * bt;
  }
  return PtFrame(t, cross(n, t), n);
}
` : ''}fn ptToLocal(f: PtFrame, v: vec3f) -> vec3f { return vec3f(dot(v, f.t), dot(v, f.b), dot(v, f.n)); }
fn ptFromLocal(f: PtFrame, v: vec3f) -> vec3f { return f.t * v.x + f.b * v.y + f.n * v.z; }

fn ptReflect(wo: vec3f, n: vec3f) -> vec3f { return -wo + 2.0 * dot(wo, n) * n; }

// pbrt Refract. ok = false on total internal reflection.
struct PtRefract { wt: vec3f, etap: f32, ok: bool }
fn ptRefract(wi: vec3f, n0: vec3f, eta0: f32) -> PtRefract {
  var n = n0;
  var eta = eta0;
  var cosI = dot(n, wi);
  if (cosI < 0.0) {
    eta = 1.0 / eta;
    cosI = -cosI;
    n = -n;
  }
  let sin2I = max(0.0, 1.0 - cosI * cosI);
  let sin2T = sin2I / (eta * eta);
  if (sin2T >= 1.0) { return PtRefract(vec3f(0.0), 1.0, false); }
  let cosT = sqrt(1.0 - sin2T);
  return PtRefract(-wi / eta + (cosI / eta - cosT) * n, eta, true);
}

fn ptFrDielectric(cosI0: f32, eta0: f32) -> f32 {
  var cosI = clamp(cosI0, -1.0, 1.0);
  var eta = eta0;
  if (cosI < 0.0) {
    eta = 1.0 / eta;
    cosI = -cosI;
  }
  let sin2T = (1.0 - cosI * cosI) / (eta * eta);
  if (sin2T >= 1.0) { return 1.0; }
  let cosT = sqrt(max(0.0, 1.0 - sin2T));
  let rParl = (eta * cosI - cosT) / (eta * cosI + cosT);
  let rPerp = (cosI - eta * cosT) / (cosI + eta * cosT);
  return (rParl * rParl + rPerp * rPerp) * 0.5;
}

fn ptCMul(a: vec2f, b: vec2f) -> vec2f { return vec2f(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
fn ptCDiv(a: vec2f, b: vec2f) -> vec2f {
  let d = dot(b, b);
  return vec2f(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y) / d;
}
fn ptCSqrt(z: vec2f) -> vec2f {
  let n = length(z);
  if (n == 0.0) { return vec2f(0.0); }
  let t1 = sqrt(0.5 * (n + abs(z.x)));
  let t2 = 0.5 * z.y / t1;
  if (z.x >= 0.0) { return vec2f(t1, t2); }
  return vec2f(abs(t2), select(-t1, t1, z.y >= 0.0));
}
fn ptFrComplex1(cosI0: f32, eta: f32, k: f32) -> f32 {
  let c = clamp(cosI0, 0.0, 1.0);
  let e = vec2f(eta, k);
  let sin2I = 1.0 - c * c;
  let sin2T = ptCDiv(vec2f(sin2I, 0.0), ptCMul(e, e));
  let cosT = ptCSqrt(vec2f(1.0 - sin2T.x, -sin2T.y));
  let ec = e * c;
  let rParl = ptCDiv(ec - cosT, ec + cosT);
  let ecT = ptCMul(e, cosT);
  let rPerp = ptCDiv(vec2f(c - ecT.x, -ecT.y), vec2f(c + ecT.x, ecT.y));
  return (dot(rParl, rParl) + dot(rPerp, rPerp)) * 0.5;
}
fn ptFrComplex(cosI: f32, eta: vec3f, k: vec3f) -> vec3f {
  return vec3f(ptFrComplex1(cosI, eta.x, k.x), ptFrComplex1(cosI, eta.y, k.y), ptFrComplex1(cosI, eta.z, k.z));
}

fn ptIsSmooth(alpha: vec2f) -> bool { return max(alpha.x, alpha.y) < PT_SMOOTH_ALPHA; }
fn ptRoughAlpha(alpha: vec2f) -> vec2f { return max(alpha, vec2f(PT_MIN_ALPHA)); }
fn ptSameHemisphere(a: vec3f, b: vec3f) -> bool { return a.z * b.z > 0.0; }

fn ptGgxD(alpha: vec2f, wm: vec3f) -> f32 {
  let z2 = wm.z * wm.z;
  let cos4 = z2 * z2;
  if (cos4 < 1e-16) { return 0.0; }
  let e = (wm.x * wm.x / (alpha.x * alpha.x) + wm.y * wm.y / (alpha.y * alpha.y)) / z2;
  return 1.0 / (PT_PI * alpha.x * alpha.y * cos4 * (1.0 + e) * (1.0 + e));
}
fn ptGgxLambda(alpha: vec2f, w: vec3f) -> f32 {
  let z2 = w.z * w.z;
  if (z2 == 0.0) { return 0.0; }
  let a2t2 = (w.x * w.x * alpha.x * alpha.x + w.y * w.y * alpha.y * alpha.y) / z2;
  return (sqrt(1.0 + a2t2) - 1.0) * 0.5;
}
fn ptGgxG1(alpha: vec2f, w: vec3f) -> f32 { return 1.0 / (1.0 + ptGgxLambda(alpha, w)); }
fn ptGgxG(alpha: vec2f, wo: vec3f, wi: vec3f) -> f32 { return 1.0 / (1.0 + ptGgxLambda(alpha, wo) + ptGgxLambda(alpha, wi)); }
fn ptGgxVisiblePdf(alpha: vec2f, w: vec3f, wm: vec3f) -> f32 {
  return ptGgxG1(alpha, w) / abs(w.z) * ptGgxD(alpha, wm) * abs(dot(w, wm));
}

fn ptGgxSampleWm(alpha: vec2f, w: vec3f, u0: f32, u1: f32) -> vec3f {
  var wh = normalize(vec3f(alpha.x * w.x, alpha.y * w.y, w.z));
  if (wh.z < 0.0) { wh = -wh; }
  if (PT_VNDF_CAP) {
    // Dupuy and Benyoub 2023: uniform point on the spherical cap visible from wh, then reproject to the hemisphere.
    let phi = 2.0 * PT_PI * u0;
    let z = (1.0 - u1) * (1.0 + wh.z) - wh.z;
    let sinT = sqrt(clamp(1.0 - z * z, 0.0, 1.0));
    let nh = vec3f(sinT * cos(phi) + wh.x, sinT * sin(phi) + wh.y, z + wh.z);
    return normalize(vec3f(alpha.x * nh.x, alpha.y * nh.y, max(1e-6, nh.z)));
  }
  // Heitz 2018 as in pbrt-v4.
  var t1 = vec3f(1.0, 0.0, 0.0);
  if (wh.z < 0.99999) { t1 = normalize(vec3f(-wh.y, wh.x, 0.0)); }
  let t2 = cross(wh, t1);
  let r = sqrt(u0);
  let theta = 2.0 * PT_PI * u1;
  let px = r * cos(theta);
  var py = r * sin(theta);
  let h = sqrt(1.0 - px * px);
  let s = (1.0 + wh.z) * 0.5;
  py = (1.0 - s) * h + s * py;
  let pz = sqrt(max(0.0, 1.0 - px * px - py * py));
  let nh = px * t1 + py * t2 + pz * wh;
  return normalize(vec3f(alpha.x * nh.x, alpha.y * nh.y, max(1e-6, nh.z)));
}

fn ptBsdfIsDelta(p: PtBsdfParams) -> bool {
  if (p.kind == PT_BSDF_CONDUCTOR) { return ptIsSmooth(p.alpha); }
  if (p.kind == PT_BSDF_DIELECTRIC) { return p.ior == 1.0 || ptIsSmooth(p.alpha); }
${thin ? `  if (p.kind == PT_BSDF_THIN) { return true; }
` : ''}  return false;
}${thin ? `

// pbrt ThinDielectricBxDF: R' = R + T^2 R / (1 - R^2), T' = 1 - R'.
fn ptThinReflectance(cos: f32, eta: f32) -> f32 {
  var R = ptFrDielectric(cos, eta);
  let T = 1.0 - R;
  if (R < 1.0) { R += T * T * R / (1.0 - R * R); }
  return R;
}` : ''}

// Generalised half vector (pbrt rough dielectric). ok = false when it is degenerate or back-facing.
struct PtHalf { wm: vec3f, isRefl: bool, etap: f32, ok: bool }
fn ptDielectricHalf(wo: vec3f, wi: vec3f, eta: f32) -> PtHalf {
  let cosO = wo.z;
  let cosI = wi.z;
  let refl = cosI * cosO > 0.0;
  var etap = 1.0;
  if (!refl) { etap = select(1.0 / eta, eta, cosO > 0.0); }
  var wm = wi * etap + wo;
  if (cosI == 0.0 || cosO == 0.0 || dot(wm, wm) == 0.0) { return PtHalf(vec3f(0.0), false, 1.0, false); }
  wm = normalize(wm);
  if (wm.z < 0.0) { wm = -wm; }
  if (dot(wm, wi) * cosI < 0.0 || dot(wm, wo) * cosO < 0.0) { return PtHalf(vec3f(0.0), false, 1.0, false); }
  return PtHalf(wm, refl, etap, true);
}

fn ptBsdfEval(p: PtBsdfParams, wo: vec3f, wi: vec3f) -> PtBsdfEval {
  var none = PtBsdfEval(vec3f(0.0), 0.0);
  if (p.kind == PT_BSDF_DIFFUSE) {
    if (!ptSameHemisphere(wo, wi)) { return none; }
    return PtBsdfEval(p.reflectance / PT_PI, abs(wi.z) / PT_PI);
  }
  if (p.kind == PT_BSDF_CONDUCTOR) {
    if (!ptSameHemisphere(wo, wi) || ptIsSmooth(p.alpha)) { return none; }
    let a = ptRoughAlpha(p.alpha);
    let cosO = abs(wo.z);
    let cosI = abs(wi.z);
    var wm = wi + wo;
    if (dot(wm, wm) == 0.0) { return none; }
    wm = normalize(wm);
    if (wm.z < 0.0) { wm = -wm; }
    let dwo = abs(dot(wo, wm));
    let F = ptFrComplex(dwo, p.eta, p.k);
    let s = ptGgxD(a, wm) * ptGgxG(a, wo, wi) / (4.0 * cosI * cosO);
    return PtBsdfEval(F * s, ptGgxVisiblePdf(a, wo, wm) / (4.0 * dwo));
  }
  if (p.kind == PT_BSDF_DIELECTRIC) {
    if (p.ior == 1.0 || ptIsSmooth(p.alpha)) { return none; }
    let a = ptRoughAlpha(p.alpha);
    let h = ptDielectricHalf(wo, wi, p.ior);
    if (!h.ok) { return none; }
    let cosO = wo.z;
    let cosI = wi.z;
    let dwo = dot(wo, h.wm);
    let dwi = dot(wi, h.wm);
    let F = ptFrDielectric(dwo, p.ior);
    let R = F;
    let T = 1.0 - F;
    let D = ptGgxD(a, h.wm);
    let G = ptGgxG(a, wo, wi);
    if (h.isRefl) {
      let fr = D * G * F / abs(4.0 * cosI * cosO);
      return PtBsdfEval(vec3f(fr), ptGgxVisiblePdf(a, wo, h.wm) / (4.0 * abs(dwo)) * (R / (R + T)));
    }
    let denomBase = dwi + dwo / h.etap;
    let denom = denomBase * denomBase * cosI * cosO;
    var ft = D * (1.0 - F) * G * abs(dwi * dwo / denom);
    if (PT_TRANSPORT_RADIANCE) { ft = ft / (h.etap * h.etap); }
    let dwmDwi = abs(dwi) / (denomBase * denomBase);
    return PtBsdfEval(vec3f(ft), ptGgxVisiblePdf(a, wo, h.wm) * dwmDwi * (T / (R + T)));
  }
  return none;
}

fn ptBsdfPdf(p: PtBsdfParams, wo: vec3f, wi: vec3f) -> f32 { return ptBsdfEval(p, wo, wi).pdf; }

// u = (uc, u0, u1). flags == 0u means failure.
fn ptBsdfSample(p: PtBsdfParams, wo: vec3f, u: vec3f) -> PtBsdfSample {
  let fail = PtBsdfSample(vec3f(0.0), vec3f(0.0), 0.0, 0u, 1.0);
  var s = fail;
  if (p.kind == PT_BSDF_DIFFUSE) {
    let ox = 2.0 * u.y - 1.0;
    let oy = 2.0 * u.z - 1.0;
    var dx = 0.0;
    var dy = 0.0;
    if (ox != 0.0 || oy != 0.0) {
      var r: f32;
      var th: f32;
      if (abs(ox) > abs(oy)) {
        r = ox;
        th = (PT_PI / 4.0) * (oy / ox);
      } else {
        r = oy;
        th = PT_PI / 2.0 - (PT_PI / 4.0) * (ox / oy);
      }
      dx = r * cos(th);
      dy = r * sin(th);
    }
    var z = sqrt(max(0.0, 1.0 - dx * dx - dy * dy));
    if (wo.z < 0.0) { z = -z; }
    s = PtBsdfSample(vec3f(dx, dy, z), p.reflectance / PT_PI, abs(z) / PT_PI, PT_FLAG_REFLECTION, 1.0);
  } else if (p.kind == PT_BSDF_CONDUCTOR) {
    if (ptIsSmooth(p.alpha)) {
      let wi = vec3f(-wo.x, -wo.y, wo.z);
      let c = abs(wi.z);
      s = PtBsdfSample(wi, ptFrComplex(c, p.eta, p.k) / c, 1.0, PT_FLAG_REFLECTION | PT_FLAG_DELTA, 1.0);
    } else {
      if (wo.z == 0.0) { return fail; }
      let a = ptRoughAlpha(p.alpha);
      let wm = ptGgxSampleWm(a, wo, u.y, u.z);
      let wi = ptReflect(wo, wm);
      if (!ptSameHemisphere(wo, wi)) { return fail; }
      let dwo = abs(dot(wo, wm));
      let pdf = ptGgxVisiblePdf(a, wo, wm) / (4.0 * dwo);
      let cosO = abs(wo.z);
      let cosI = abs(wi.z);
      if (cosI == 0.0 || cosO == 0.0) { return fail; }
      let f = ptFrComplex(dwo, p.eta, p.k) * (ptGgxD(a, wm) * ptGgxG(a, wo, wi) / (4.0 * cosI * cosO));
      s = PtBsdfSample(wi, f, pdf, PT_FLAG_REFLECTION, 1.0);
    }
  } else if (p.kind == PT_BSDF_DIELECTRIC) {
    let eta = p.ior;
    if (eta == 1.0 || ptIsSmooth(p.alpha)) {
      let R = ptFrDielectric(wo.z, eta);
      let T = 1.0 - R;
      if (u.x < R) {
        let wi = vec3f(-wo.x, -wo.y, wo.z);
        let c = abs(wi.z);
        s = PtBsdfSample(wi, vec3f(R / c), R, PT_FLAG_REFLECTION | PT_FLAG_DELTA, 1.0);
      } else {
        let r = ptRefract(wo, vec3f(0.0, 0.0, 1.0), eta);
        if (!r.ok) { return fail; }
        var ft = T / abs(r.wt.z);
        if (PT_TRANSPORT_RADIANCE) { ft = ft / (r.etap * r.etap); }
        s = PtBsdfSample(r.wt, vec3f(ft), T, PT_FLAG_TRANSMISSION | PT_FLAG_DELTA, r.etap);
      }
    } else {
      let a = ptRoughAlpha(p.alpha);
      let wm = ptGgxSampleWm(a, wo, u.y, u.z);
      let dwo = dot(wo, wm);
      let R = ptFrDielectric(dwo, eta);
      let T = 1.0 - R;
      if (u.x < R) {
        let wi = ptReflect(wo, wm);
        if (!ptSameHemisphere(wo, wi)) { return fail; }
        let pdf = ptGgxVisiblePdf(a, wo, wm) / (4.0 * abs(dwo)) * R;
        let fv = ptGgxD(a, wm) * ptGgxG(a, wo, wi) * R / (4.0 * wi.z * wo.z);
        s = PtBsdfSample(wi, vec3f(fv), pdf, PT_FLAG_REFLECTION, 1.0);
      } else {
        let r = ptRefract(wo, wm, eta);
        if (!r.ok) { return fail; }
        let wi = r.wt;
        if (ptSameHemisphere(wo, wi) || wi.z == 0.0) { return fail; }
        let dwi = dot(wi, wm);
        let denomBase = dwi + dwo / r.etap;
        let denom = denomBase * denomBase;
        let dwmDwi = abs(dwi) / denom;
        let pdf = ptGgxVisiblePdf(a, wo, wm) * dwmDwi * T;
        var ft = T * ptGgxD(a, wm) * ptGgxG(a, wo, wi) * abs(dwi * dwo / (wi.z * wo.z * denom));
        if (PT_TRANSPORT_RADIANCE) { ft = ft / (r.etap * r.etap); }
        s = PtBsdfSample(wi, vec3f(ft), pdf, PT_FLAG_TRANSMISSION, r.etap);
      }
    }
  }
${thin ? `  if (p.kind == PT_BSDF_THIN) {
    let c = abs(wo.z);
    let R = ptThinReflectance(c, p.ior);
    let T = 1.0 - R;
    if (u.x < R) {
      s = PtBsdfSample(vec3f(-wo.x, -wo.y, wo.z), vec3f(R / c), R, PT_FLAG_REFLECTION | PT_FLAG_DELTA, 1.0);
    } else {
      s = PtBsdfSample(-wo, p.reflectance * (T / c), T, PT_FLAG_TRANSMISSION | PT_FLAG_DELTA, 1.0);
    }
  }
` : ''}  if (s.pdf == 0.0 || s.wi.z == 0.0 || (s.f.x == 0.0 && s.f.y == 0.0 && s.f.z == 0.0)) { return fail; }
  return s;
}
`;
}
