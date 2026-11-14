/**
 * WGSL twin of `layered.ts` (P2.21). `ptLayeredWgsl(options)` returns a module fragment to append AFTER `ptBsdfWgsl` (it uses
 * its `PtBsdfParams`, `ptBsdfEval`/`ptBsdfSample` for the Lambert base, `ptFrDielectric`, `ptRefract`, `ptGgx*`,
 * `ptDielectricHalf`, `PT_VNDF_CAP`, `PT_TRANSPORT_RADIANCE`). Keep it line-for-line parallel with `layered.ts`:
 * `bench/pages/layered-check.ts` compares them on the GPU. Ported from pbrt-v4 (Apache-2.0), see `layered.ts` for the
 * attribution and DIVERGES V57/V58.
 *
 * Only the chosen layering is emitted (the megakernel pays for layered code only when a layered material exists):
 * `albedo` adds the interface helpers plus the table (about 3 KB of numbers), `stochastic` the interface helpers plus
 * the walk and no table. Settings are module constants, so each variant is its own pipeline.
 *
 * Entry points (a `seed` is mixed into the stochastic hash, ignored by the albedo layering):
 *   ptLayeredEval(p, wo, wi, seed) -> PtBsdfEval, ptLayeredPdf, ptLayeredSample(p, wo, u, seed) -> PtBsdfSample.
 */

import { PT_LAYERED_LUT_N, ptLayeredLut } from './layered-lut.ts';
import type { PtLayeredOptions } from './layered.ts';

function f32List(a: Float32Array): string {
  return Array.from(a)
    .map((v) => {
      const s = String(Number(v.toFixed(4)));
      if (/e/.test(s)) return s;
      return s.includes('.') ? s.replace(/^0\./, '.') : `${s}.0`;
    })
    .join(',');
}

export function ptLayeredWgsl(options: PtLayeredOptions = {}): string {
  const stochastic = (options.layering ?? 'albedo') === 'stochastic';
  const maxDepth = Math.max(1, Math.floor(options.maxDepth ?? 10));
  const nSamples = Math.max(1, Math.floor(options.nSamples ?? 1));
  const pbrtMis = (options.exitMis ?? 'pbrt') === 'pbrt';
  const internal = options.albedoBase === 'internal';
  const n = PT_LAYERED_LUT_N;
  const cond = options.conductorBase === true;
  return /* wgsl */ `
const PT_FLAG_PROPORTIONAL: u32 = 8u;
const PT_LAYER_REFLECTION: u32 = 1u;
const PT_LAYER_TRANSMISSION: u32 = 2u;
const PT_LAYER_ALL: u32 = 3u;
const PT_LAYER_MIN_THICKNESS: f32 = 1e-30;
const PT_LAYER_MAX_DEPTH: u32 = ${maxDepth}u;
const PT_LAYER_NSAMPLES: u32 = ${nSamples}u;
const PT_LAYER_MIS_PBRT: bool = ${pbrtMis};
const PT_LAYER_BASE_INTERNAL: bool = ${internal};

struct PtLayeredParams {
  baseColor: vec3f,
  ior: f32,
  alpha: vec2f,
  specularWeight: f32,
  specularColor: vec3f,
  thickness: f32,
  internalReflectance: f32,${cond ? `
  cEta: vec3f,
  cK: vec3f,
  cAlpha: f32,
  isCond: bool,` : ''}
}
struct PtTop { ior: f32, alpha: vec2f, s: vec3f }

fn ptAvg(v: vec3f) -> f32 { return (v.x + v.y + v.z) / 3.0; }
fn ptIsBlack(v: vec3f) -> bool { return v.x == 0.0 && v.y == 0.0 && v.z == 0.0; }
fn ptLayerTop(p: PtLayeredParams) -> PtTop { return PtTop(p.ior, p.alpha, p.specularColor * p.specularWeight); }
fn ptLayerBase(p: PtLayeredParams) -> PtBsdfParams {
  var b: PtBsdfParams;${cond ? `
  if (p.isCond) {
    b.kind = PT_BSDF_CONDUCTOR;
    b.ior = 1.0;
    b.alpha = vec2f(p.cAlpha);
    b.reflectance = vec3f(0.0);
    b.eta = p.cEta;
    b.k = p.cK;
    return b;
  }` : ''}
  b.kind = PT_BSDF_DIFFUSE;
  b.ior = 1.0;
  b.alpha = vec2f(0.0);
  b.reflectance = p.baseColor;
  b.eta = vec3f(0.0);
  b.k = vec3f(0.0);
  return b;
}
fn ptTopIsSpecular(t: PtTop) -> bool { return t.ior == 1.0 || ptIsSmooth(t.alpha); }
// R' = s F, but total internal reflection stays total.
fn ptLayerR(t: PtTop, F: f32) -> vec3f {
  if (F >= 1.0) { return vec3f(1.0); }
  return min(t.s * F, vec3f(1.0));
}
${cond ? `fn ptLayerSmoothBase(p: PtLayeredParams) -> bool { return p.isCond && ptIsSmooth(vec2f(p.cAlpha)); }
` : ''}fn ptLayerOk(s: PtBsdfSample) -> bool {
  return s.flags != 0u && !ptIsBlack(s.f) && s.pdf > 0.0 && s.wi.z != 0.0;
}
fn ptLayerFail() -> PtBsdfSample { return PtBsdfSample(vec3f(0.0), vec3f(0.0), 0.0, 0u, 1.0); }

${stochastic ? stochasticWgsl(cond) : albedoWgsl(n, cond)}
`;
}

function stochasticWgsl(cond: boolean): string {
  return /* wgsl */ `
// The dielectric interface with restricted sampling (pbrt DielectricBxDF + BxDFReflTransFlags), chromatic Fresnel scale.
fn ptTopSample(t: PtTop, wo: vec3f, uc: f32, u0: f32, u1: f32, radiance: bool, flags: u32) -> PtBsdfSample {
  let fail = ptLayerFail();
  let eta = t.ior;
  if (ptTopIsSpecular(t)) {
    let F = ptFrDielectric(wo.z, eta);
    let Rv = ptLayerR(t, F);
    let Tv = vec3f(1.0) - Rv;
    var pr = 0.0;
    if ((flags & PT_LAYER_REFLECTION) != 0u) { pr = ptAvg(Rv); }
    var pt = 0.0;
    if ((flags & PT_LAYER_TRANSMISSION) != 0u) { pt = ptAvg(Tv); }
    if (pr + pt == 0.0) { return fail; }
    if (uc < pr / (pr + pt)) {
      return PtBsdfSample(vec3f(-wo.x, -wo.y, wo.z), Rv / abs(wo.z), pr / (pr + pt), PT_FLAG_REFLECTION | PT_FLAG_DELTA, 1.0);
    }
    let r = ptRefract(wo, vec3f(0.0, 0.0, 1.0), eta);
    if (!r.ok) { return fail; }
    var k = 1.0 / abs(r.wt.z);
    if (radiance) { k = k / (r.etap * r.etap); }
    return PtBsdfSample(r.wt, Tv * k, pt / (pr + pt), PT_FLAG_TRANSMISSION | PT_FLAG_DELTA, r.etap);
  }
  let a = ptRoughAlpha(t.alpha);
  if (wo.z == 0.0) { return fail; }
  let wm = ptGgxSampleWm(a, wo, u0, u1);
  let dwo = dot(wo, wm);
  let F = ptFrDielectric(dwo, eta);
  let Rv = ptLayerR(t, F);
  let Tv = vec3f(1.0) - Rv;
  var pr = 0.0;
  if ((flags & PT_LAYER_REFLECTION) != 0u) { pr = ptAvg(Rv); }
  var pt = 0.0;
  if ((flags & PT_LAYER_TRANSMISSION) != 0u) { pt = ptAvg(Tv); }
  if (pr + pt == 0.0) { return fail; }
  let vis = ptGgxVisiblePdf(a, wo, wm);
  if (uc < pr / (pr + pt)) {
    let wi = ptReflect(wo, wm);
    if (wi.z * wo.z <= 0.0) { return fail; }
    let pdf = vis / (4.0 * abs(dwo)) * (pr / (pr + pt));
    let k = ptGgxD(a, wm) * ptGgxG(a, wo, wi) / (4.0 * wi.z * wo.z);
    return PtBsdfSample(wi, Rv * k, pdf, PT_FLAG_REFLECTION, 1.0);
  }
  let r = ptRefract(wo, wm, eta);
  if (!r.ok) { return fail; }
  let wi = r.wt;
  if (wi.z * wo.z > 0.0 || wi.z == 0.0) { return fail; }
  let dwi = dot(wi, wm);
  let denomBase = dwi + dwo / r.etap;
  let denom = denomBase * denomBase;
  let pdf = vis * (abs(dwi) / denom) * (pt / (pr + pt));
  var k = ptGgxD(a, wm) * ptGgxG(a, wo, wi) * abs(dwi * dwo / (wi.z * wo.z * denom));
  if (radiance) { k = k / (r.etap * r.etap); }
  return PtBsdfSample(wi, Tv * k, pdf, PT_FLAG_TRANSMISSION, r.etap);
}

// f over both lobes, pdf under the flags restriction. Delta interface: nothing.
fn ptTopEval(t: PtTop, wo: vec3f, wi: vec3f, radiance: bool, flags: u32) -> PtBsdfEval {
  let none = PtBsdfEval(vec3f(0.0), 0.0);
  if (ptTopIsSpecular(t)) { return none; }
  let a = ptRoughAlpha(t.alpha);
  let h = ptDielectricHalf(wo, wi, t.ior);
  if (!h.ok) { return none; }
  let cosO = wo.z;
  let cosI = wi.z;
  let dwo = dot(wo, h.wm);
  let dwi = dot(wi, h.wm);
  let F = ptFrDielectric(dwo, t.ior);
  let Rv = ptLayerR(t, F);
  let Tv = vec3f(1.0) - Rv;
  var pr = 0.0;
  if ((flags & PT_LAYER_REFLECTION) != 0u) { pr = ptAvg(Rv); }
  var pt = 0.0;
  if ((flags & PT_LAYER_TRANSMISSION) != 0u) { pt = ptAvg(Tv); }
  let D = ptGgxD(a, h.wm);
  let G = ptGgxG(a, wo, wi);
  let vis = ptGgxVisiblePdf(a, wo, h.wm);
  if (h.isRefl) {
    let k = D * G / abs(4.0 * cosI * cosO);
    var pdf = 0.0;
    if (pr + pt != 0.0) { pdf = vis / (4.0 * abs(dwo)) * (pr / (pr + pt)); }
    return PtBsdfEval(Rv * k, pdf);
  }
  let denomBase = dwi + dwo / h.etap;
  let denom = denomBase * denomBase * cosI * cosO;
  var k = D * G * abs(dwi * dwo / denom);
  if (radiance) { k = k / (h.etap * h.etap); }
  var pdf = 0.0;
  if (pr + pt != 0.0) { pdf = vis * (abs(dwi) / (denomBase * denomBase)) * (pt / (pr + pt)); }
  return PtBsdfEval(Tv * k, pdf);
}
// Counter-based PCG hash RNG (identical to ptPcg in layered.ts).
fn ptPcg(v: u32) -> u32 {
  let state = v * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}
fn ptRngNext(s: ptr<function, u32>) -> f32 {
  *s = ptPcg(*s);
  return f32(*s >> 8u) * 5.9604645e-8;
}
fn ptHashSeed(seed: u32, a: vec3f, b: vec3f) -> u32 {
  var h = ptPcg(seed);
  h = ptPcg(h ^ bitcast<u32>(a.x));
  h = ptPcg(h ^ bitcast<u32>(a.y));
  h = ptPcg(h ^ bitcast<u32>(a.z));
  h = ptPcg(h ^ bitcast<u32>(b.x));
  h = ptPcg(h ^ bitcast<u32>(b.y));
  h = ptPcg(h ^ bitcast<u32>(b.z));
  return h;
}
fn ptPowerHeuristic(f: f32, g: f32) -> f32 { return (f * f) / (f * f + g * g); }
fn ptLayerTr(dz: f32, w: vec3f) -> f32 {
  if (abs(dz) <= PT_LAYER_MIN_THICKNESS) { return 1.0; }
  return exp(-abs(dz / w.z));
}

fn ptLayeredF(p: PtLayeredParams, wo0: vec3f, wi0: vec3f, seed: u32) -> vec3f {
  var wo = wo0;
  var wi = wi0;
  if (wo.z < 0.0) { wo = -wo; wi = -wi; }
  if (wo.z == 0.0 || wi.z * wo.z <= 0.0) { return vec3f(0.0); }
  let top = ptLayerTop(p);
  let base = ptLayerBase(p);
  let thickness = max(p.thickness, PT_LAYER_MIN_THICKNESS);
  let topSpec = ptTopIsSpecular(top);${cond ? `
  let baseSpec = ptLayerSmoothBase(p);` : ''}
  let exitZ = thickness;
  var f = f32(PT_LAYER_NSAMPLES) * ptTopEval(top, wo, wi, PT_TRANSPORT_RADIANCE, PT_LAYER_ALL).f;
  var rs = ptHashSeed(seed, wo, wi);
  for (var s = 0u; s < PT_LAYER_NSAMPLES; s++) {
    let a0 = ptRngNext(&rs);
    let a1 = ptRngNext(&rs);
    let a2 = ptRngNext(&rs);
    let wos = ptTopSample(top, wo, a0, a1, a2, PT_TRANSPORT_RADIANCE, PT_LAYER_TRANSMISSION);
    if (!ptLayerOk(wos)) { continue; }
    let b0 = ptRngNext(&rs);
    let b1 = ptRngNext(&rs);
    let b2 = ptRngNext(&rs);
    let wis = ptTopSample(top, wi, b0, b1, b2, !PT_TRANSPORT_RADIANCE, PT_LAYER_TRANSMISSION);
    if (!ptLayerOk(wis)) { continue; }
    var beta = wos.f * (abs(wos.wi.z) / wos.pdf);
    var z = thickness;
    var w = wos.wi;
    for (var depth = 0u; depth < PT_LAYER_MAX_DEPTH; depth++) {
      let mb = max(beta.x, max(beta.y, beta.z));
      if (depth > 3u && mb < 0.25) {
        let q = max(0.0, 1.0 - mb);
        if (ptRngNext(&rs) < q) { break; }
        beta = beta / (1.0 - q);
      }
      z = select(thickness, 0.0, z == thickness);
      beta = beta * ptLayerTr(thickness, w);
      if (z == exitZ) {
        let c0 = ptRngNext(&rs);
        let c1 = ptRngNext(&rs);
        let c2 = ptRngNext(&rs);
        let bs = ptTopSample(top, -w, c0, c1, c2, PT_TRANSPORT_RADIANCE, PT_LAYER_REFLECTION);
        if (!ptLayerOk(bs)) { break; }
        beta = beta * (bs.f * (abs(bs.wi.z) / bs.pdf));
        w = bs.wi;
      } else {
${cond ? `        if (!baseSpec) {
          let nwi = -wis.wi;
          let fb = ptBsdfEval(base, -w, nwi);
          var wt = 1.0;
          if (!topSpec && PT_LAYER_MIS_PBRT) { wt = ptPowerHeuristic(wis.pdf, fb.pdf); }
          f = f + beta * fb.f * abs(wis.wi.z) * wt * ptLayerTr(thickness, wis.wi) * (wis.f / wis.pdf);
        }
` : `        let nwi = -wis.wi;
        let fb = ptBsdfEval(base, -w, nwi);
        var wt = 1.0;
        if (!topSpec && PT_LAYER_MIS_PBRT) { wt = ptPowerHeuristic(wis.pdf, fb.pdf); }
        f = f + beta * fb.f * abs(wis.wi.z) * wt * ptLayerTr(thickness, wis.wi) * (wis.f / wis.pdf);
`}        let d0 = ptRngNext(&rs);
        let d1 = ptRngNext(&rs);
        let d2 = ptRngNext(&rs);
        let bs = ptBsdfSample(base, -w, vec3f(d0, d1, d2));
        if (bs.flags == 0u || ptIsBlack(bs.f) || bs.pdf == 0.0 || bs.wi.z == 0.0) { break; }
        beta = beta * (bs.f * (abs(bs.wi.z) / bs.pdf));
        w = bs.wi;
        if (${cond ? '!topSpec && (PT_LAYER_MIS_PBRT || baseSpec)' : '!topSpec && PT_LAYER_MIS_PBRT'}) {
          let fe = ptTopEval(top, -w, wi, PT_TRANSPORT_RADIANCE, PT_LAYER_ALL).f;
          if (!ptIsBlack(fe)) {
            let exitPdf = ptTopEval(top, -w, wi, PT_TRANSPORT_RADIANCE, PT_LAYER_TRANSMISSION).pdf;
            let wt2 = ${cond ? 'select(ptPowerHeuristic(bs.pdf, exitPdf), 1.0, baseSpec)' : 'ptPowerHeuristic(bs.pdf, exitPdf)'};
            f = f + beta * (ptLayerTr(thickness, bs.wi) * wt2) * fe;
          }
        }
      }
    }
  }
  return f / f32(PT_LAYER_NSAMPLES);
}

fn ptLayeredSample(p: PtLayeredParams, wo0: vec3f, u: vec3f, seed: u32) -> PtBsdfSample {
  let fail = ptLayerFail();
  var wo = wo0;
  var flipWi = false;
  if (wo.z < 0.0) { wo = -wo; flipWi = true; }
  let top = ptLayerTop(p);
  let base = ptLayerBase(p);
  let thickness = max(p.thickness, PT_LAYER_MIN_THICKNESS);
  let bs0 = ptTopSample(top, wo, u.x, u.y, u.z, PT_TRANSPORT_RADIANCE, PT_LAYER_ALL);
  if (!ptLayerOk(bs0)) { return fail; }
  if ((bs0.flags & PT_FLAG_REFLECTION) != 0u) {
    var r = bs0;
    if (flipWi) { r.wi = -r.wi; }
    r.flags = r.flags | PT_FLAG_PROPORTIONAL;
    return r;
  }
  var w = bs0.wi;
  var specularPath = (bs0.flags & PT_FLAG_DELTA) != 0u;
  var rs = ptHashSeed(seed, wo, u);
  var f = bs0.f * abs(bs0.wi.z);
  var pdf = bs0.pdf;
  var z = thickness;
  for (var depth = 0u; depth < PT_LAYER_MAX_DEPTH; depth++) {
    let rrBeta = max(f.x, max(f.y, f.z)) / pdf;
    if (depth > 3u && rrBeta < 0.25) {
      let q = max(0.0, 1.0 - rrBeta);
      if (ptRngNext(&rs) < q) { return fail; }
      pdf = pdf * (1.0 - q);
    }
    if (w.z == 0.0) { return fail; }
    z = select(thickness, 0.0, z == thickness);
    f = f * ptLayerTr(thickness, w);
    let e0 = ptRngNext(&rs);
    let e1 = ptRngNext(&rs);
    let e2 = ptRngNext(&rs);
    var bs: PtBsdfSample;
    if (z == 0.0) { bs = ptBsdfSample(base, -w, vec3f(e0, e1, e2)); }
    else { bs = ptTopSample(top, -w, e0, e1, e2, PT_TRANSPORT_RADIANCE, PT_LAYER_ALL); }
    if (!ptLayerOk(bs)) { return fail; }
    f = f * bs.f;
    pdf = pdf * bs.pdf;
    if (pdf > 1e18 || pdf < 1e-18) {
      f = f / pdf;
      pdf = 1.0;
    }
    specularPath = specularPath && ((bs.flags & PT_FLAG_DELTA) != 0u);
    w = bs.wi;
    if ((bs.flags & PT_FLAG_TRANSMISSION) != 0u) {
      var kind = PT_FLAG_TRANSMISSION;
      if (w.z * wo.z > 0.0) { kind = PT_FLAG_REFLECTION; }
      if (specularPath) { kind = kind | PT_FLAG_DELTA; }
      var wi = w;
      if (flipWi) { wi = -wi; }
      return PtBsdfSample(wi, f, pdf, kind | PT_FLAG_PROPORTIONAL, 1.0);
    }
    f = f * abs(bs.wi.z);
  }
  return fail;
}

fn ptLayeredPdf(p: PtLayeredParams, wo0: vec3f, wi0: vec3f, seed: u32) -> f32 {
  var wo = wo0;
  var wi = wi0;
  if (wo.z < 0.0) { wo = -wo; wi = -wi; }
  let top = ptLayerTop(p);
  let base = ptLayerBase(p);
  let same = wi.z * wo.z > 0.0;
  var rs = ptHashSeed(seed, wi, wo);
  var pdfSum = 0.0;
  if (same) {
    pdfSum = pdfSum + f32(PT_LAYER_NSAMPLES) * ptTopEval(top, wo, wi, PT_TRANSPORT_RADIANCE, PT_LAYER_REFLECTION).pdf;
    let topSpec = ptTopIsSpecular(top);${cond ? `
    let baseSpec = ptLayerSmoothBase(p);` : ''}
    for (var s = 0u; s < PT_LAYER_NSAMPLES; s++) {
      let a0 = ptRngNext(&rs);
      let a1 = ptRngNext(&rs);
      let a2 = ptRngNext(&rs);
      let wos = ptTopSample(top, wo, a0, a1, a2, PT_TRANSPORT_RADIANCE, PT_LAYER_TRANSMISSION);
      let b0 = ptRngNext(&rs);
      let b1 = ptRngNext(&rs);
      let b2 = ptRngNext(&rs);
      let wis = ptTopSample(top, wi, b0, b1, b2, !PT_TRANSPORT_RADIANCE, PT_LAYER_TRANSMISSION);
      if (!(ptLayerOk(wos) && ptLayerOk(wis))) { continue; }
      if (topSpec) {
        pdfSum = pdfSum + ptBsdfEval(base, -wos.wi, -wis.wi).pdf;
      } else {
        let c0 = ptRngNext(&rs);
        let c1 = ptRngNext(&rs);
        let c2 = ptRngNext(&rs);
        let rsm = ptBsdfSample(base, -wos.wi, vec3f(c0, c1, c2));
        if (ptLayerOk(rsm)) {
${cond ? `          if (!baseSpec) {
            let rPdf = ptBsdfEval(base, -wos.wi, -wis.wi).pdf;
            pdfSum = pdfSum + ptPowerHeuristic(wis.pdf, rPdf) * rPdf;
          }
          let tPdf = ptTopEval(top, -rsm.wi, wi, PT_TRANSPORT_RADIANCE, PT_LAYER_ALL).pdf;
          pdfSum = pdfSum + select(ptPowerHeuristic(rsm.pdf, tPdf), 1.0, baseSpec) * tPdf;
` : `          let rPdf = ptBsdfEval(base, -wos.wi, -wis.wi).pdf;
          pdfSum = pdfSum + ptPowerHeuristic(wis.pdf, rPdf) * rPdf;
          let tPdf = ptTopEval(top, -rsm.wi, wi, PT_TRANSPORT_RADIANCE, PT_LAYER_ALL).pdf;
          pdfSum = pdfSum + ptPowerHeuristic(rsm.pdf, tPdf) * tPdf;
`}        }
      }
    }
  }
  return 0.1 / (4.0 * PT_PI) + 0.9 * (pdfSum / f32(PT_LAYER_NSAMPLES));
}

fn ptLayeredEval(p: PtLayeredParams, wo: vec3f, wi: vec3f, seed: u32) -> PtBsdfEval {
  return PtBsdfEval(ptLayeredF(p, wo, wi, seed), ptLayeredPdf(p, wo, wi, seed));
}
`;
}

function albedoWgsl(n: number, cond: boolean): string {
  const lut = ptLayeredLut();
  const text = /* wgsl */ `
// The interface's reflection lobe only (what the albedo layering needs; the walk's restricted-sampling pair is not emitted).
// Same maths as ptTopEval / ptTopSample with flags = REFLECTION (layered.ts).
fn ptTopReflEval(t: PtTop, wo: vec3f, wi: vec3f) -> PtBsdfEval {
  let none = PtBsdfEval(vec3f(0.0), 0.0);
  if (ptTopIsSpecular(t)) { return none; }
  let a = ptRoughAlpha(t.alpha);
  var wm = wi + wo;
  if (dot(wm, wm) == 0.0) { return none; }
  wm = normalize(wm);
  if (wm.z < 0.0) { wm = -wm; }
  let dwo = dot(wo, wm);
  let Rv = ptLayerR(t, ptFrDielectric(dwo, t.ior));
  var pdf = 0.0;
  if (ptAvg(Rv) > 0.0) { pdf = ptGgxVisiblePdf(a, wo, wm) / (4.0 * abs(dwo)); }
  return PtBsdfEval(Rv * (ptGgxD(a, wm) * ptGgxG(a, wo, wi) / abs(4.0 * wi.z * wo.z)), pdf);
}
fn ptTopReflSample(t: PtTop, wo: vec3f, u0: f32, u1: f32) -> PtBsdfSample {
  let fail = ptLayerFail();
  let a = ptRoughAlpha(t.alpha);
  let wm = ptGgxSampleWm(a, wo, u0, u1);
  let dwo = dot(wo, wm);
  let Rv = ptLayerR(t, ptFrDielectric(dwo, t.ior));
  if (ptAvg(Rv) == 0.0) { return fail; }
  let wi = ptReflect(wo, wm);
  if (wi.z * wo.z <= 0.0) { return fail; }
  let k = ptGgxD(a, wm) * ptGgxG(a, wo, wi) / (4.0 * wi.z * wo.z);
  return PtBsdfSample(wi, Rv * k, ptGgxVisiblePdf(a, wo, wm) / (4.0 * abs(dwo)), PT_FLAG_REFLECTION, 1.0);
}

// E(mu, r) split-sum table (layered-lut.ts): E = F0 * A + B; rows = roughness (sqrt axis), columns = mu (sqrt axis).
var<private> PT_LAYER_LUT: array<f32, ${lut.length}> = array<f32, ${lut.length}>(${f32List(lut)});
fn ptLayerAlbedoLut(f0: f32, mu: f32, rough: f32) -> f32 {
  let x = sqrt(clamp(mu, 0.0, 1.0)) * ${n - 1}.0;
  let y = sqrt(clamp(rough, 0.0, 1.0)) * ${n - 1}.0;
  let x0 = u32(min(floor(x), ${n - 2}.0));
  let y0 = u32(min(floor(y), ${n - 2}.0));
  let fx = x - f32(x0);
  let fy = y - f32(y0);
  let i00 = (y0 * ${n}u + x0) * 2u;
  let i01 = i00 + 2u;
  let i10 = i00 + ${n * 2}u;
  let i11 = i10 + 2u;
  let e00 = f0 * PT_LAYER_LUT[i00] + PT_LAYER_LUT[i00 + 1u];
  let e01 = f0 * PT_LAYER_LUT[i01] + PT_LAYER_LUT[i01 + 1u];
  let e10 = f0 * PT_LAYER_LUT[i10] + PT_LAYER_LUT[i10 + 1u];
  let e11 = f0 * PT_LAYER_LUT[i11] + PT_LAYER_LUT[i11 + 1u];
  return clamp((e00 * (1.0 - fx) + e01 * fx) * (1.0 - fy) + (e10 * (1.0 - fx) + e11 * fx) * fy, 0.0, 1.0);
}
fn ptLayeredTopAlbedo(p: PtLayeredParams, mu: f32) -> f32 {
  if (p.ior == 1.0) { return 0.0; }
  if (p.ior == 1.0 || ptIsSmooth(p.alpha)) { return ptFrDielectric(max(mu, 1e-6), p.ior); }
  let t = (p.ior - 1.0) / (p.ior + 1.0);
  return ptLayerAlbedoLut(t * t, mu, sqrt(0.5 * (p.alpha.x + p.alpha.y)));
}

fn ptLayerKeep(p: PtLayeredParams, mu: f32) -> vec3f {
  let s = p.specularColor * p.specularWeight;
  return vec3f(1.0) - min(s * ptLayeredTopAlbedo(p, mu), vec3f(1.0));
}
// 'internal' base: 1 / (eta^2 (1 - rho R_i)) per channel.
fn ptLayerInternalGain(p: PtLayeredParams) -> vec3f {
  return vec3f(1.0) / (p.ior * p.ior * (vec3f(1.0) - min(p.baseColor * p.internalReflectance, vec3f(0.9999))));
}
struct PtLayerMix { keep: vec3f, pS: f32 }
fn ptLayerMix(p: PtLayeredParams, mu: f32) -> PtLayerMix {
  let keep = ptLayerKeep(p, mu);
  let eS = ptAvg(vec3f(1.0) - keep);
  var eB = ptAvg(p.baseColor * keep);
  if (PT_LAYER_BASE_INTERNAL) { eB = ptAvg(p.baseColor * keep * ptLayerInternalGain(p)) * (1.0 - p.internalReflectance) * p.ior * p.ior; }
  var pS = 0.0;
  if (eS + eB > 0.0) { pS = eS / (eS + eB); }
  return PtLayerMix(keep, pS);
}

fn ptLayeredEval(p: PtLayeredParams, wo0: vec3f, wi0: vec3f, seed: u32) -> PtBsdfEval {
  var wo = wo0;
  var wi = wi0;
  if (wo.z < 0.0) { wo = -wo; wi = -wi; }
  if (wo.z == 0.0 || wi.z <= 0.0) { return PtBsdfEval(vec3f(0.0), 0.0); }
  let lm = ptLayerMix(p, wo.z);
  var fd = p.baseColor / PT_PI * lm.keep;
  if (PT_LAYER_BASE_INTERNAL) { fd = fd * ptLayerKeep(p, wi.z) * ptLayerInternalGain(p); }
  let spec = ptTopReflEval(ptLayerTop(p), wo, wi);
  return PtBsdfEval(fd + spec.f, lm.pS * spec.pdf + (1.0 - lm.pS) * (wi.z / PT_PI));
}
fn ptLayeredPdf(p: PtLayeredParams, wo: vec3f, wi: vec3f, seed: u32) -> f32 { return ptLayeredEval(p, wo, wi, seed).pdf; }

fn ptLayeredSample(p: PtLayeredParams, wo0: vec3f, u: vec3f, seed: u32) -> PtBsdfSample {
  let fail = ptLayerFail();
  var wo = wo0;
  var flip = false;
  if (wo.z < 0.0) { wo = -wo; flip = true; }
  if (wo.z == 0.0) { return fail; }
  let lm = ptLayerMix(p, wo.z);
  let top = ptLayerTop(p);
  var s = fail;
  if (u.x < lm.pS) {
    if (ptTopIsSpecular(top)) {
      let F = ptFrDielectric(wo.z, p.ior);
      let sc = p.specularColor * p.specularWeight;
      s = PtBsdfSample(vec3f(-wo.x, -wo.y, wo.z), sc * F / wo.z, lm.pS, PT_FLAG_REFLECTION | PT_FLAG_DELTA, 1.0);
    } else {
      let t = ptTopReflSample(top, wo, u.y, u.z);
      if (t.flags == 0u) { return fail; }
      let e = ptLayeredEval(p, wo, t.wi, seed);
      if (e.pdf == 0.0) { return fail; }
      s = PtBsdfSample(t.wi, e.f, e.pdf, PT_FLAG_REFLECTION, 1.0);
    }
  } else {
    let d = ptBsdfSample(ptLayerBase(p), wo, vec3f(0.0, u.y, u.z));
    if (d.flags == 0u) { return fail; }
    let e = ptLayeredEval(p, wo, d.wi, seed);
    if (e.pdf == 0.0 || ptIsBlack(e.f)) { return fail; }
    s = PtBsdfSample(d.wi, e.f, e.pdf, PT_FLAG_REFLECTION, 1.0);
  }
  if (flip) { s.wi = -s.wi; }
  if (s.flags == 0u || s.pdf == 0.0 || s.wi.z == 0.0 || ptIsBlack(s.f)) { return fail; }
  return s;
}
`;
  return cond ? condAlbedoWgsl(text) : text;
}

/**
 * Coat over a conductor, albedo layering (layered.ts condEval/condSample): the Lambert-base functions are renamed `...D` and wrapped by
 * dispatchers on `p.isCond`.
 */
function condAlbedoWgsl(text: string): string {
  const renamed = text.replace(/ptLayeredEval\(/g, 'ptLayeredEvalD(').replace(/ptLayeredPdf\(/g, 'ptLayeredPdfD(').replace(/ptLayeredSample\(/g, 'ptLayeredSampleD(');
  return /* wgsl */ `${renamed}
fn ptInsideCos(ior: f32, d: f32) -> f32 { return sqrt(max(0.0, 1.0 - (1.0 - d * d) / (ior * ior))); }
fn ptCondFresnelRatio(p: PtLayeredParams, d: f32) -> vec3f {
  let now = ptFrComplex(ptInsideCos(p.ior, d), p.cEta, p.cK);
  let was = ptFrComplex(d, p.cEta, p.cK);
  return select(vec3f(1.0), now / was, was > vec3f(0.0));
}
fn ptCondMix(p: PtLayeredParams, mu: f32) -> PtLayerMix {
  let keep = ptLayerKeep(p, mu);
  let eS = ptAvg(vec3f(1.0) - keep);
  let F = ptFrComplex(ptInsideCos(p.ior, mu), p.cEta, p.cK);
  let eB = ptAvg(F * keep);
  var pS = 0.0;
  if (eS + eB > 0.0) { pS = eS / (eS + eB); }
  return PtLayerMix(keep, pS);
}
fn ptCondEval(p: PtLayeredParams, wo: vec3f, wi: vec3f) -> PtBsdfEval {
  if (wo.z == 0.0 || wi.z <= 0.0) { return PtBsdfEval(vec3f(0.0), 0.0); }
  let lm = ptCondMix(p, wo.z);
  let m = ptBsdfEval(ptLayerBase(p), wo, wi);
  let spec = ptTopReflEval(ptLayerTop(p), wo, wi);
  var fm = m.f * lm.keep;
  if (m.pdf > 0.0) {
    let wm = normalize(wo + wi);
    fm = fm * ptCondFresnelRatio(p, abs(dot(wo, wm)));
    if (PT_LAYER_BASE_INTERNAL) { fm = fm * ptLayerKeep(p, wi.z); }
  }
  return PtBsdfEval(fm + spec.f, lm.pS * spec.pdf + (1.0 - lm.pS) * m.pdf);
}
fn ptCondSample(p: PtLayeredParams, wo: vec3f, u: vec3f) -> PtBsdfSample {
  let fail = ptLayerFail();
  let lm = ptCondMix(p, wo.z);
  let top = ptLayerTop(p);
  if (u.x < lm.pS) {
    if (ptTopIsSpecular(top)) {
      let F = ptFrDielectric(wo.z, p.ior);
      return PtBsdfSample(vec3f(-wo.x, -wo.y, wo.z), vec3f(F / wo.z), lm.pS, PT_FLAG_REFLECTION | PT_FLAG_DELTA, 1.0);
    }
    let t = ptTopReflSample(top, wo, u.y, u.z);
    if (t.flags == 0u) { return fail; }
    let e = ptCondEval(p, wo, t.wi);
    if (e.pdf == 0.0) { return fail; }
    return PtBsdfSample(t.wi, e.f, e.pdf, PT_FLAG_REFLECTION, 1.0);
  }
  let d = ptBsdfSample(ptLayerBase(p), wo, vec3f(0.0, u.y, u.z));
  if (d.flags == 0u) { return fail; }
  if ((d.flags & PT_FLAG_DELTA) != 0u) {
    var fd = d.f * lm.keep * ptCondFresnelRatio(p, wo.z);
    if (PT_LAYER_BASE_INTERNAL) { fd = fd * ptLayerKeep(p, d.wi.z); }
    return PtBsdfSample(d.wi, fd, 1.0 - lm.pS, d.flags, 1.0);
  }
  let e = ptCondEval(p, wo, d.wi);
  if (e.pdf == 0.0 || ptIsBlack(e.f)) { return fail; }
  return PtBsdfSample(d.wi, e.f, e.pdf, PT_FLAG_REFLECTION, 1.0);
}
fn ptLayeredEval(p: PtLayeredParams, wo0: vec3f, wi0: vec3f, seed: u32) -> PtBsdfEval {
  if (!p.isCond) { return ptLayeredEvalD(p, wo0, wi0, seed); }
  if (wo0.z < 0.0) { return ptCondEval(p, -wo0, -wi0); }
  return ptCondEval(p, wo0, wi0);
}
fn ptLayeredPdf(p: PtLayeredParams, wo: vec3f, wi: vec3f, seed: u32) -> f32 { return ptLayeredEval(p, wo, wi, seed).pdf; }
fn ptLayeredSample(p: PtLayeredParams, wo0: vec3f, u: vec3f, seed: u32) -> PtBsdfSample {
  if (!p.isCond) { return ptLayeredSampleD(p, wo0, u, seed); }
  var wo = wo0;
  var flip = false;
  if (wo.z < 0.0) { wo = -wo; flip = true; }
  if (wo.z == 0.0) { return ptLayerFail(); }
  var s = ptCondSample(p, wo, u);
  if (flip && s.flags != 0u) { s.wi = -s.wi; }
  if (s.flags == 0u || s.pdf == 0.0 || s.wi.z == 0.0 || ptIsBlack(s.f)) { return ptLayerFail(); }
  return s;
}
`;
}
