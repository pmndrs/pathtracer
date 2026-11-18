/**
 * The spectral mode's WGSL (P5, `docs/design/spectral.md`). Compiled only with `integrator.spectral`; every other kernel is byte-identical.
 *
 * A path carries `PT_SPEC_LANES` = 3 wavelengths (`ptLam`, the lanes of the existing vec3 throughput / radiance), drawn once per sample by `ptSpecBegin(u)`.
 * Every RGB parameter a vertex reads becomes the value of its sigmoid-polynomial spectrum at the three wavelengths (`ptSpecMaterial`, `ptSpecIllum`), so the BSDF, MIS,
 * layering and media code run unchanged on "channels" that are wavelengths. `ptSpecToRgb` turns the lane radiance into the sample's linear Rec.709 RGB
 * (`wavelengths.ts` has the estimator). Dispersion: a dielectric with `dispersion > 0` takes the hero lane's IOR and terminates the secondary lanes.
 *
 * Bindings: `@group(2) @binding(PT_SPEC_BINDING)` the table texture (`rgb2SpecTexture`, `rgba32float`, `unfilterable-float`, textureLoad).
 */
import { cauchyCoefficients, ABBE_LINES } from './wavelengths.ts';
import { MATERIAL_DISPERSION_MAX } from '@pmndrs/raytracer';
import { SPEC_XYZ_TO_RGB } from './rgb2spec.ts';

/** Group 2 binding of the table texture. Taken: 0-3, 9 (V-buffer), 10 (layer correction), 11 (ReSTIR shade), 12-14 (real time), 15 (ReSTIR sharedHit), 16-17 (volumes). */
export const PT_SPEC_BINDING = 18;

export interface PtSpectralOptions {
  /** Table resolution per axis (default 64, pbrt's); the texture is `res^2 x (3 res + 1)` texels of 16 bytes. Must match the table passed to `PtMegakernel` (`spectralTable`). */
  res?: number;
  /** Compile the dispersion code (`ptSpecIor`, `ptMaterialDispersion`): a dielectric's IOR follows a Cauchy law through `RtMaterial.dispersion`. Default true. */
  dispersion?: boolean;
}

export interface ResolvedSpectralOptions {
  res: number;
  dispersion: boolean;
}

export const PT_SPEC_DEFAULT_RES = 64;

export function resolveSpectralOptions(o: boolean | PtSpectralOptions | undefined | null): ResolvedSpectralOptions | null {
  if (!o) return null;
  const s = o === true ? {} : o;
  const res = s.res ?? PT_SPEC_DEFAULT_RES;
  if (!Number.isInteger(res) || res < 10 || res > 128) throw new RangeError(`integrator.spectral.res must be an integer in 10..128, got ${res}`);
  return { res, dispersion: s.dispersion ?? true };
}

const f32 = (v: number): string => {
  const s = String(Math.fround(v));
  return /[.e]/.test(s) ? s : `${s}.0`;
};

export function ptSpectralWgsl(o: ResolvedSpectralOptions): string {
  const m = SPEC_XYZ_TO_RGB.map(f32);
  const uc = 1 / (1 / (ABBE_LINES.F * 1e-3) ** 2 - 1 / (ABBE_LINES.C * 1e-3) ** 2);
  void cauchyCoefficients;
  return /* wgsl */ `
@group(2) @binding(${PT_SPEC_BINDING}) var ptSpecTex: texture_2d<f32>;
const PT_SPEC_RES: u32 = ${o.res}u;
var<private> ptLam: vec3f;
var<private> ptLamPdf: vec3f;
var<private> ptLamD65: vec3f;
var<private> ptLamX: vec3f;
var<private> ptLamY: vec3f;
var<private> ptLamZ: vec3f;
var<private> ptSpecHero: bool;
var<private> ptSpecDisp: bool;

fn ptSpecRow(lam: f32) -> vec4f {
    let f = (lam - 360.0) / 5.0;
    if (!(f >= 0.0) || f > 84.0) {
        return vec4f(0.0);
    }
    let i = min(u32(f), 83u);
    let t = f - f32(i);
    return mix(textureLoad(ptSpecTex, vec2u(i, 3u * PT_SPEC_RES), 0), textureLoad(ptSpecTex, vec2u(i + 1u, 3u * PT_SPEC_RES), 0), t);
}

fn ptSpecLambda(u: f32) -> f32 {
    return 538.0 - 138.888889 * atanh(0.85691062 - 1.82750197 * u);
}

fn ptSpecPdf(l: f32) -> f32 {
    let c = cosh(0.0072 * (l - 538.0));
    return 0.0039398042 / (c * c);
}

// One uniform number -> the path's three wavelengths (hero + two rotations of u) and the per-lane table values.
fn ptSpecBegin(u: f32) {
    ptLam = vec3f(ptSpecLambda(u), ptSpecLambda(fract(u + 0.33333334)), ptSpecLambda(fract(u + 0.6666667)));
    ptLamPdf = vec3f(ptSpecPdf(ptLam.x), ptSpecPdf(ptLam.y), ptSpecPdf(ptLam.z));
    let r0 = ptSpecRow(ptLam.x);
    let r1 = ptSpecRow(ptLam.y);
    let r2 = ptSpecRow(ptLam.z);
    ptLamX = vec3f(r0.x, r1.x, r2.x);
    ptLamY = vec3f(r0.y, r1.y, r2.y);
    ptLamZ = vec3f(r0.z, r1.z, r2.z);
    ptLamD65 = vec3f(r0.w, r1.w, r2.w);
    ptSpecHero = false;
    ptSpecDisp = false;
}

// Sample radiance of the three lanes -> linear Rec.709 RGB.
fn ptSpecToRgb(l: vec3f) -> vec3f {
    let w = l / (ptLamPdf * 3.0);
    let x = dot(w, ptLamX);
    let y = dot(w, ptLamY);
    let z = dot(w, ptLamZ);
    return vec3f(${m[0]} * x + ${m[1]} * y + ${m[2]} * z, ${m[3]} * x + ${m[4]} * y + ${m[5]} * z, ${m[6]} * x + ${m[7]} * y + ${m[8]} * z);
}

fn ptSpecInvSmooth(y: f32) -> f32 {
    return 0.5 - sin(asin(clamp(1.0 - 2.0 * y, -1.0, 1.0)) / 3.0);
}

fn ptSpecSmooth(x: f32) -> f32 {
    return x * x * (3.0 - 2.0 * x);
}

fn ptSpecFetch(m: u32, k: u32, j: u32, i: u32) -> vec3f {
    return textureLoad(ptSpecTex, vec2u(i + j * PT_SPEC_RES, m * PT_SPEC_RES + k), 0).xyz;
}

// Sigmoid coefficients of a reflectance RGB in [0, 1] (pbrt RGBToSpectrumTable::operator()).
fn ptSpecCoeffs(rgb: vec3f) -> vec3f {
    if (rgb.x == rgb.y && rgb.y == rgb.z) {
        return vec3f(0.0, 0.0, (rgb.x - 0.5) / sqrt(max(rgb.x * (1.0 - rgb.x), 2.5e-9)));
    }
    var m = 2u;
    if (rgb.x >= rgb.y && rgb.x >= rgb.z) {
        m = 0u;
    } else if (rgb.y >= rgb.z) {
        m = 1u;
    }
    var q = array<f32, 3>(rgb.x, rgb.y, rgb.z);
    let z = q[m];
    let res1 = f32(PT_SPEC_RES - 1u);
    let x = q[(m + 1u) % 3u] * res1 / z;
    let y = q[(m + 2u) % 3u] * res1 / z;
    let kf = res1 * ptSpecInvSmooth(ptSpecInvSmooth(z));
    let zi = min(u32(max(kf, 0.0)), PT_SPEC_RES - 2u);
    let z0 = ptSpecSmooth(ptSpecSmooth(f32(zi) / res1));
    let z1 = ptSpecSmooth(ptSpecSmooth(f32(zi + 1u) / res1));
    let dz = clamp((z - z0) / (z1 - z0), 0.0, 1.0);
    let xi = min(u32(x), PT_SPEC_RES - 2u);
    let yi = min(u32(y), PT_SPEC_RES - 2u);
    let dx = x - f32(xi);
    let dy = y - f32(yi);
    let a = mix(mix(ptSpecFetch(m, zi, yi, xi), ptSpecFetch(m, zi, yi, xi + 1u), dx), mix(ptSpecFetch(m, zi, yi + 1u, xi), ptSpecFetch(m, zi, yi + 1u, xi + 1u), dx), dy);
    let b = mix(mix(ptSpecFetch(m, zi + 1u, yi, xi), ptSpecFetch(m, zi + 1u, yi, xi + 1u), dx), mix(ptSpecFetch(m, zi + 1u, yi + 1u, xi), ptSpecFetch(m, zi + 1u, yi + 1u, xi + 1u), dx), dy);
    return mix(a, b, dz);
}

// The spectrum of reflectance \`rgb\` (in [0, 1]) at the path's three wavelengths.
fn ptSpecRefl(rgb: vec3f) -> vec3f {
    if (rgb.x == rgb.y && rgb.y == rgb.z) {
        return vec3f(rgb.x);
    }
    let c = ptSpecCoeffs(clamp(rgb, vec3f(0.0), vec3f(1.0)));
    let t = (ptLam - 360.0) / 470.0;
    let v = (c.x * t + c.y) * t + c.z;
    return 0.5 + v / (2.0 * sqrt(1.0 + v * v));
}

// pbrt RGBUnboundedSpectrum: scale 2 max(rgb) times the reflectance spectrum of rgb / scale.
fn ptSpecUnbounded(rgb: vec3f) -> vec3f {
    let s = 2.0 * max(rgb.x, max(rgb.y, rgb.z));
    if (!(s > 0.0)) {
        return vec3f(0.0);
    }
    if (rgb.x == rgb.y && rgb.y == rgb.z) {
        return vec3f(rgb.x);
    }
    return s * ptSpecRefl(rgb / s);
}

// pbrt RGBIlluminantSpectrum: the unbounded spectrum times D65 (normalised by Yd, so rgb (1, 1, 1) is the D65 white with Y = 1).
fn ptSpecIllum(rgb: vec3f) -> vec3f {
    return ptSpecUnbounded(rgb) * ptLamD65;
}

${o.dispersion ? `fn ptMaterialDispersion(id: u32) -> f32 {
    let h1 = rtSceneTable[1];
    let slot = select(h1.x, id, id < h1.x);
    return f32(rtSceneTable[h1.y + slot * 8u + 7u].x >> 16u) * RT_UNORM16_SCALE * ${f32(MATERIAL_DISPERSION_MAX)};
}

// Cauchy law n(l) = A + B / l^2 through \`ior\` at the d line with Abbe number 20 / dispersion, at the hero wavelength.
fn ptSpecIor(ior: f32, dispersion: f32) -> f32 {
    let b = (ior - 1.0) * dispersion * ${f32(uc / 20)};
    let a = ior - b / ${f32((ABBE_LINES.d * 1e-3) ** 2)};
    let um = ptLam.x * 0.001;
    return a + b / (um * um);
}

` : ''}// The material as the three wavelengths see it: colours uplifted (reflectances) or illuminant-weighted (emission), a dispersive dielectric at the hero IOR.
fn ptSpecMaterial(m0: RtMaterial, id: u32) -> RtMaterial {
    var m = m0;
    m.baseColor = ptSpecRefl(m.baseColor);
    m.specularColor = ptSpecRefl(m.specularColor);
    m.conductorEta = ptSpecUnbounded(m.conductorEta);
    m.conductorK = ptSpecUnbounded(m.conductorK);
    if (m.transmissionDepth > 0.0 && !m.thinWalled) {
        // an absorbing interior: pbrt uplifts the RGB sigma (an unbounded spectrum) and exponentiates, so do the same (the colour field then carries exp(-sigma(l) depth))
        let sg = -log(max(m.transmissionColor, vec3f(1.0e-6))) / m.transmissionDepth;
        m.transmissionColor = exp(-ptSpecUnbounded(sg) * m.transmissionDepth);
    } else {
        m.transmissionColor = ptSpecRefl(m.transmissionColor);
    }
    m.coatColor = ptSpecRefl(m.coatColor);
    m.sheenColor = ptSpecRefl(m.sheenColor);
    m.emissionColor = ptSpecIllum(m.emissionColor);
${o.dispersion ? `    let d = ptMaterialDispersion(id);
    if (d > 0.0) {
        m.ior = ptSpecIor(m.ior, d);
        ptSpecDisp = true;
    }
` : ''}    return m;
}
`;
}
