/**
 * WGSL twin of `thinfilm.ts` (OpenPBR thin film, P2.27). Compiled into the kernel only when `PtMaterialFeatures.thinFilm` is set (`ptBsdfWgsl`
 * with `thinFilm` for the conductor, `ptLayeredWgsl` with `film` for the dielectric interface, albedo layering). The integrator writes the
 * private `ptFilm` = (weight, thickness in micrometres, ior) once per vertex from the material record (`ptMaterialFilm`).
 * Needs `ptBsdfWgsl` before it (`ptFrComplex`, `ptFrDielectric`).
 */
export function ptThinFilmWgsl(): string {
  return /* wgsl */ `
var<private> ptFilm: vec3f;
const PT_FILM_WHITE: vec3f = vec3f(1.2047843, 0.9483008, 0.9088427);

fn ptMaterialFilm(id: u32) -> vec3f {
  let h1 = rtSceneTable[1];
  let slot = select(h1.x, id, id < h1.x);
  return bitcast<vec3f>(rtSceneTable[h1.y + slot * 8u + 7u].yzw);
}
fn ptFilmActive() -> bool { return ptFilm.x > 0.0 && ptFilm.y > 0.0; }
fn ptFilmFresnelDiel(c: f32, na: f32, nb: f32) -> f32 {
  let eta = nb / na;
  let s2 = (1.0 - c * c) / (eta * eta);
  if (s2 >= 1.0) { return 1.0; }
  let ct = sqrt(1.0 - s2);
  let rp = (eta * c - ct) / (eta * c + ct);
  let rs = (c - eta * ct) / (c + eta * ct);
  return 0.5 * (rp * rp + rs * rs);
}
fn ptFilmSensitivity(opd: f32, shift: vec3f) -> vec3f {
  let phase = 2.0 * PT_PI * opd * 1.0e-9;
  let val = vec3f(5.4856e-13, 4.4201e-13, 5.2481e-13);
  let pos = vec3f(1.6810e+06, 1.7953e+06, 2.2084e+06);
  let vr = vec3f(4.3278e+09, 9.3046e+09, 6.6121e+09);
  var xyz = val * sqrt(2.0 * PT_PI * vr) * cos(pos * phase + shift) * exp(-phase * phase * vr);
  xyz.x = xyz.x + 9.7470e-14 * sqrt(2.0 * PT_PI * 4.5282e+09) * cos(2.2399e+06 * phase + shift.x) * exp(-4.5282e+09 * phase * phase);
  xyz = xyz / 1.0685e-7;
  let rgb = vec3f(
    3.2404542 * xyz.x - 1.5371385 * xyz.y - 0.4985314 * xyz.z,
    -0.9692660 * xyz.x + 1.8760108 * xyz.y + 0.0415560 * xyz.z,
    0.0556434 * xyz.x - 0.2040259 * xyz.y + 1.0572252 * xyz.z);
  return rgb / PT_FILM_WHITE;
}
// Airy reflectance of the film on a base of normal-incidence reflectance f0 (per channel), outside IOR 1.
fn ptFilmAiry(cos1: f32, f0In: vec3f) -> vec3f {
  let th = ptFilm.y * 1000.0;
  let t = clamp(th / 30.0, 0.0, 1.0);
  let eta2 = 1.0 + (ptFilm.z - 1.0) * (t * t * (3.0 - 2.0 * t));
  let c1 = clamp(cos1, 0.0, 1.0);
  let cos2Sq = 1.0 - (1.0 - c1 * c1) / (eta2 * eta2);
  if (cos2Sq < 0.0) { return vec3f(1.0); }
  let cos2 = sqrt(cos2Sq);
  let R12 = ptFilmFresnelDiel(c1, 1.0, eta2);
  let T121 = 1.0 - R12;
  let phi21b = PT_PI - select(0.0, PT_PI, eta2 < 1.0);
  let opd = 2.0 * eta2 * th * cos2;
  let f0 = clamp(f0In, vec3f(0.0), vec3f(0.9999));
  let s = sqrt(f0);
  let baseIor = (vec3f(1.0) + s) / (vec3f(1.0) - s);
  let R23 = vec3f(ptFilmFresnelDiel(cos2, eta2, baseIor.x), ptFilmFresnelDiel(cos2, eta2, baseIor.y), ptFilmFresnelDiel(cos2, eta2, baseIor.z));
  let phi = vec3f(phi21b) + select(vec3f(0.0), vec3f(PT_PI), baseIor < vec3f(eta2));
  let R123 = clamp(R12 * R23, vec3f(1e-12), vec3f(0.9999));
  let r123 = sqrt(R123);
  let Rs = T121 * T121 * R23 / (vec3f(1.0) - R123);
  var I = vec3f(R12) + Rs;
  var Cm = Rs - vec3f(T121);
  for (var m = 1; m <= 2; m = m + 1) {
    Cm = Cm * r123;
    I = I + Cm * 2.0 * ptFilmSensitivity(f32(m) * opd, f32(m) * phi);
  }
  return clamp(I, vec3f(0.0), vec3f(1.0));
}
// The Fresnel with the film at the coverage weight; noFilm is the plain material's at the same angle.
fn ptFilmMix(noFilm: vec3f, c: f32, f0: vec3f) -> vec3f {
  if (!ptFilmActive()) { return noFilm; }
  return noFilm + ptFilm.x * (ptFilmAiry(c, f0) - noFilm);
}
fn ptFrConductor(c: f32, eta: vec3f, k: vec3f) -> vec3f {
  let F = ptFrComplex(c, eta, k);
  if (!ptFilmActive()) { return F; }
  let e1 = eta - vec3f(1.0);
  let e2 = eta + vec3f(1.0);
  return ptFilmMix(F, c, (e1 * e1 + k * k) / (e2 * e2 + k * k));
}
// The dielectric interface's Fresnel per channel (layered top): plain or with the film.
fn ptFilmF(c: f32, ior: f32) -> vec3f {
  let F = ptFrDielectric(c, ior);
  if (!ptFilmActive()) { return vec3f(F); }
  let t = (ior - 1.0) / (ior + 1.0);
  return ptFilmMix(vec3f(F), c, vec3f(t * t));
}
`;
}
