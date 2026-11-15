/**
 * WGSL twin of `fuzz.ts` (OpenPBR fuzz, P2.25). Compiled into the kernel only when `PtMaterialFeatures.fuzz` is set; the LTC table is a
 * read-only storage buffer at `@group(2) @binding(2)` (`PtMegakernel` creates and binds it: 1024 vec4f, `ptFuzzTableF32()`).
 *
 * The integrator sets the two private variables once per vertex from the material record (`sheenWeight`, `sheenColor`, `sheenRoughness`), then
 * calls `ptFuzzedEval` / `ptFuzzedSample` instead of `ptEval` / `ptBsdfEval` (resp. `ptSample` / `ptBsdfSample`).
 * Needs `ptBsdfWgsl` before it (PtBsdfParams, PtBsdfEval, PtBsdfSample, PT_FLAG_*, PT_BSDF_BLACK) and, with `layered`, `ptLayeredDispatchWgsl`.
 */
export interface PtFuzzWgslOptions {
  /** The base is dispatched through `ptEval` / `ptSample` (layered kinds exist); otherwise straight `ptBsdfEval` / `ptBsdfSample`. */
  layered: boolean;
}

export function ptFuzzWgsl(options: PtFuzzWgslOptions): string {
  const evalFn = options.layered ? 'ptEval' : 'ptBsdfEval';
  const sampleFn = options.layered ? 'ptSample' : 'ptBsdfSample';
  return /* wgsl */ `
@group(2) @binding(2) var<storage, read> ptFuzzTable: array<vec4f>;
var<private> ptFuzzW: f32;
var<private> ptFuzzC: vec4f;

fn ptFuzzCoeffs(cosO: f32, alpha: f32) -> vec3f {
  let row = clamp(alpha, 0.0, 0.99999994) * 31.0;
  let col = clamp(cosO, 0.0, 0.99999994) * 31.0;
  let r = floor(row);
  let c = floor(col);
  let i = u32(r) * 32u + u32(c);
  let top = mix(ptFuzzTable[i], ptFuzzTable[i + 1u], col - c);
  let bottom = mix(ptFuzzTable[i + 32u], ptFuzzTable[i + 33u], col - c);
  return mix(top, bottom, row - r).xyz;
}
fn ptFuzzActive(woZ: f32) -> bool {
  if (!(ptFuzzW > 0.0) || !(max(ptFuzzC.x, max(ptFuzzC.y, ptFuzzC.z)) > 0.0) || !(woZ > 0.0)) { return false; }
  let c = ptFuzzCoeffs(woZ, ptFuzzC.w);
  return c.x > 1e-4 && c.z > 0.0;
}
fn ptFuzzRot(v: vec3f, wo: vec3f, inverse: bool) -> vec3f {
  let rho = length(wo.xy);
  if (rho == 0.0) { return v; }
  let c = wo.x / rho;
  let s = wo.y / rho;
  if (!inverse) { return vec3f(v.x * c + v.y * s, -v.x * s + v.y * c, v.z); }
  return vec3f(v.x * c - v.y * s, v.x * s + v.y * c, v.z);
}
fn ptFuzzLobePdf(c: vec3f, wo: vec3f, wi: vec3f) -> f32 {
  if (!(wi.z > 0.0) || !(c.x > 1e-4)) { return 0.0; }
  let w = ptFuzzRot(wi, wo, false);
  let o = vec3f(c.x * w.x + c.y * w.z, c.x * w.y, w.z);
  let l2 = dot(o, o);
  return w.z * c.x * c.x / (PT_PI * l2 * l2);
}
fn ptFuzzLobeSample(c: vec3f, wo: vec3f, u0: f32, u1: f32) -> vec3f {
  let r = sqrt(u0);
  let phi = 2.0 * PT_PI * u1;
  let oz = sqrt(max(0.0, 1.0 - u0));
  let x = (r * cos(phi) - oz * c.y) / c.x;
  let y = r * sin(phi) / c.x;
  return ptFuzzRot(normalize(vec3f(x, y, oz)), wo, true);
}
fn ptFuzzChoice(woZ: f32, baseBlack: bool) -> f32 {
  if (baseBlack) { return 1.0; }
  let e = ptFuzzCoeffs(woZ, ptFuzzC.w).z;
  let wf = ptFuzzW * e * ((ptFuzzC.x + ptFuzzC.y + ptFuzzC.z) / 3.0);
  return clamp(wf / (wf + (1.0 - ptFuzzW * e)), 0.05, 0.95);
}
fn ptFuzzedEval(p: PtBsdfParams, wo: vec3f, wi: vec3f) -> PtBsdfEval {
  if (!ptFuzzActive(wo.z)) { return ${evalFn}(p, wo, wi); }
  let black = p.kind == PT_BSDF_BLACK;
  let pf = ptFuzzChoice(wo.z, black);
  let c = ptFuzzCoeffs(wo.z, ptFuzzC.w);
  var b = PtBsdfEval(vec3f(0.0), 0.0);
  if (!black) { b = ${evalFn}(p, wo, wi); }
  let pdf = ptFuzzLobePdf(c, wo, wi);
  var fz = vec3f(0.0);
  if (wi.z > 0.0) { fz = ptFuzzC.xyz * (c.z * pdf / wi.z); }
  return PtBsdfEval(ptFuzzW * fz + (1.0 - ptFuzzW * c.z) * b.f, pf * pdf + (1.0 - pf) * b.pdf);
}
fn ptFuzzedSample(p: PtBsdfParams, wo: vec3f, u: vec3f) -> PtBsdfSample {
  if (!ptFuzzActive(wo.z)) { return ${sampleFn}(p, wo, u); }
  let fail = PtBsdfSample(vec3f(0.0), vec3f(0.0), 0.0, 0u, 1.0);
  let black = p.kind == PT_BSDF_BLACK;
  let pf = ptFuzzChoice(wo.z, black);
  if (u.x < pf) {
    let wi = ptFuzzLobeSample(ptFuzzCoeffs(wo.z, ptFuzzC.w), wo, u.y, u.z);
    let e = ptFuzzedEval(p, wo, wi);
    if (!(e.pdf > 0.0)) { return fail; }
    return PtBsdfSample(wi, e.f, e.pdf, PT_FLAG_REFLECTION, 1.0);
  }
  let s = ${sampleFn}(p, wo, vec3f((u.x - pf) / (1.0 - pf), u.y, u.z));
  if (s.flags == 0u || !(s.pdf > 0.0)) { return fail; }
  if ((s.flags & PT_FLAG_DELTA) != 0u) {
    let keep = 1.0 - ptFuzzW * ptFuzzCoeffs(wo.z, ptFuzzC.w).z;
    return PtBsdfSample(s.wi, keep * s.f, (1.0 - pf) * s.pdf, s.flags, s.etap);
  }
  let e = ptFuzzedEval(p, wo, s.wi);
  if (!(e.pdf > 0.0)) { return fail; }
  return PtBsdfSample(s.wi, e.f, e.pdf, s.flags & (PT_FLAG_REFLECTION | PT_FLAG_TRANSMISSION), s.etap);
}
`;
}
