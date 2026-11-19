/** Bench-only phase 13 fixture. Camera → planar mirror → isotropic conductor → traced emitter/sky. */
import { nrcEncodingWgsl } from '@pathtracer/mlp';
import { ptBsdfWgsl } from '@pmndrs/pathtracer';

export const SPECULAR_ROUGHNESS = [0.18, 0.30, 0.45, 0.62] as const;
export const SPECULAR_VIEWS = ['oblique-left', 'oblique-center', 'oblique-right', 'grazing-left', 'grazing-center', 'grazing-right'] as const;
export const SPECULAR_BANDS = SPECULAR_ROUGHNESS.length + 2; // A delta mirror and glass below the conductor strips.
export const SPECULAR_LUT_SIZE = 48;

const FIXTURE = /* wgsl */ `
${ptBsdfWgsl()}
const SP_LUT_SIZE = ${SPECULAR_LUT_SIZE}u;
const SP_ETA = vec3f(0.31, 0.74, 1.25);
const SP_K = vec3f(2.1, 2.5, 3.3);
const SP_NORMAL = vec3f(0.0, 0.0, 1.0);
const SP_ROUGHNESS = array<f32, ${SPECULAR_ROUGHNESS.length}>(${SPECULAR_ROUGHNESS.join(', ')});

struct SpParams {
  width: u32, height: u32, seed: u32, trainSpp: u32,
  refSpp: u32, lutSpp: u32, _padding: vec2u,
}
struct SpHit { p: vec3f, wo: vec3f, roughness: f32, kind: u32 }
struct SpReference { raw: vec4f, split: vec4f }

fn spMix(x: u32) -> u32 {
  let state = x * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}
fn spRandom(x: u32) -> f32 { return f32(spMix(x) >> 8u) / 16777216.0; }
fn spSequence(i: u32, count: u32, rotation: vec2f) -> vec2f {
  return fract(vec2f((f32(i) + 0.5) / f32(count), f32(reverseBits(i) >> 8u) / 16777216.0) + rotation);
}

fn spHit(uv: vec2f, grazing: bool, shift: f32, band: u32) -> SpHit {
  let film = vec3f((2.0 * uv.x - 1.0) * 1.05, (1.0 - 2.0 * uv.y) * 0.58, 0.0);
  let camera = vec3f(select(-4.4, -21.0, grazing), shift, 3.0);
  let mirrorZ = 3.5;
  let unfoldedCamera = vec3f(camera.xy, 2.0 * mirrorZ - camera.z);
  let virtualRay = normalize(film - unfoldedCamera);
  let mirrorPoint = unfoldedCamera + virtualRay * ((mirrorZ - unfoldedCamera.z) / virtualRay.z);
  let reflected = ptReflect(normalize(camera - mirrorPoint), -SP_NORMAL);
  let p = mirrorPoint + reflected * (-mirrorPoint.z / reflected.z);
  var kind = PT_BSDF_CONDUCTOR;
  var roughness = 0.0;
  if (band < ${SPECULAR_ROUGHNESS.length}u) { roughness = SP_ROUGHNESS[band]; }
  if (band == ${SPECULAR_BANDS - 1}u) { kind = PT_BSDF_DIELECTRIC; }
  return SpHit(p, -reflected, roughness, kind);
}
fn spSupported(hit: SpHit) -> bool { return hit.kind == PT_BSDF_CONDUCTOR && !ptIsSmooth(vec2f(hit.roughness * hit.roughness)); }
fn spQuery(r: u32, params: SpParams) -> SpHit {
  let pixel = r % (params.width * params.height);
  let view = r / (params.width * params.height);
  let uv = (vec2f(f32(pixel % params.width), f32(pixel / params.width)) + 0.5) / vec2f(f32(params.width), f32(params.height));
  return spHit(uv, view >= 3u, (f32(view % 3u) - 1.0) * 0.25, min(u32(uv.y * ${SPECULAR_BANDS}.0), ${SPECULAR_BANDS - 1}u));
}
fn spBsdf(kind: u32, roughness: f32) -> PtBsdfParams {
  return PtBsdfParams(kind, 1.5, vec2f(roughness * roughness), vec3f(0.0), SP_ETA, SP_K);
}

// Analytic planar primitive intersection. First positive finite surface hit wins.
fn spRadiance(p: vec3f, wi: vec3f) -> vec3f {
  if (wi.z < 0.0) { return vec3f(0.10, 0.17, 0.24); }
  if (wi.z < 1e-5) { return vec3f(0.055); }
  let blocker = p.xy + wi.xy * (0.21 / wi.z);
  if (abs(blocker.x - 0.43) < 0.045 && abs(blocker.y - 0.18) < 0.055) { return vec3f(0.006); }
  let card = p.xy + wi.xy * (0.42 / wi.z);
  if (abs(card.x - 0.50) < 0.19 && abs(card.y) < 0.44) { return vec3f(1.2, 2.5, 11.0); }
  if (abs(card.x - 2.20) < 0.29 && abs(card.y) < 0.44) { return vec3f(1.2, 2.5, 11.0); }
  if (abs(card.x - 0.9) < 0.16 && abs(card.y + 0.40) < 0.16) { return vec3f(4.0, 1.1, 0.3); }
  return vec3f(0.055);
}

fn spRaw(hit: SpHit, count: u32, rotation: vec2f, unitLight: bool) -> vec3f {
  let p = spBsdf(hit.kind, hit.roughness);
  var sum = vec3f(0.0);
  for (var i = 0u; i < count; i++) {
    let u = spSequence(i, count, rotation);
    let s = ptBsdfSample(p, hit.wo, vec3f(fract(u.x * 19.0 + u.y * 7.0), u));
    if (s.pdf > 0.0) {
      var light = vec3f(1.0);
      if (!unitLight) { light = spRadiance(hit.p, s.wi); }
      sum += light * s.f * (abs(s.wi.z) / s.pdf);
    }
  }
  return sum / f32(count);
}

// Expected positive cosine under reflected GGX NDF sampling at N=V=R. Alpha = roughness squared.
// The normalizer is fixed for a roughness, so the same unbiased estimator works with one training ray.
fn spCosineNorm(a: f32) -> f32 {
  let b = a * a;
  let t = 1.0 - b;
  return 1.0 / t + 2.0 * b * log(2.0 * b / (1.0 + b)) / (t * t);
}
fn spPrefilter(hit: SpHit, count: u32, rotation: vec2f, unitLight: bool) -> vec3f {
  let alpha = hit.roughness * hit.roughness;
  let frame = ptFrameFromNormal(ptReflect(hit.wo, SP_NORMAL));
  var sum = vec3f(0.0);
  for (var i = 0u; i < count; i++) {
    let u = spSequence(i, count, rotation);
    // In the filter frame the view equals its normal, so the GGX visible distribution is the NDF.
    let h = ptGgxSampleWm(vec2f(alpha), SP_NORMAL, u.x, u.y);
    let wi = ptReflect(SP_NORMAL, h);
    if (wi.z > 0.0) {
      let world = ptFromLocal(frame, wi);
      var light = vec3f(1.0);
      if (!unitLight) {
        light = vec3f(0.0);
        if (world.z > 0.0) { light = spRadiance(hit.p, world); }
      }
      sum += light * wi.z;
    }
  }
  return sum / (f32(count) * spCosineNorm(alpha));
}

`;

const LOOKUP = /* wgsl */ `
fn spLookup(hit: SpHit) -> vec3f {
  let uv = clamp(vec2f((hit.wo.z - 0.05) / 0.95, (hit.roughness - 0.18) / 0.44), vec2f(0.0), vec2f(1.0));
  let p = uv * f32(SP_LUT_SIZE - 1u);
  let lo = vec2u(floor(p));
  let hi = min(lo + 1u, vec2u(SP_LUT_SIZE - 1u));
  let f = fract(p);
  let a = mix(spLut[lo.y * SP_LUT_SIZE + lo.x], spLut[lo.y * SP_LUT_SIZE + hi.x], f.x);
  let b = mix(spLut[hi.y * SP_LUT_SIZE + lo.x], spLut[hi.y * SP_LUT_SIZE + hi.x], f.x);
  return mix(a, b, f.y).xyz;
}
`;

export const SPECULAR_REFERENCE_WGSL = /* wgsl */ `
${FIXTURE}
@group(0) @binding(0) var<uniform> params: SpParams;
@group(0) @binding(1) var<storage, read_write> lutOut: array<vec4f>;
@group(0) @binding(2) var<storage, read> spLut: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> referenceOut: array<SpReference>;
@group(0) @binding(4) var<storage, read_write> furnaceOut: array<vec4f>;
${LOOKUP}

@compute @workgroup_size(64)
fn buildLut(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= SP_LUT_SIZE * SP_LUT_SIZE) { return; }
  let mu = 0.05 + 0.95 * f32(i % SP_LUT_SIZE) / f32(SP_LUT_SIZE - 1u);
  let rough = 0.18 + 0.44 * f32(i / SP_LUT_SIZE) / f32(SP_LUT_SIZE - 1u);
  let hit = SpHit(vec3f(0.0), vec3f(sqrt(1.0 - mu * mu), 0.0, mu), rough, PT_BSDF_CONDUCTOR);
  lutOut[i] = vec4f(spRaw(hit, params.lutSpp, vec2f(0.317, 0.181), true), 0.0);
}

@compute @workgroup_size(64)
fn reference(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= params.width * params.height * ${SPECULAR_VIEWS.length}u) { return; }
  let hit = spQuery(i, params);
  let a = vec2f(spRandom(i * 2u + 90003u), spRandom(i * 2u + 90004u));
  let raw = spRaw(hit, params.refSpp, a, false);
  var split = raw;
  let supported = spSupported(hit);
  if (supported) {
    let b = vec2f(spRandom(i * 2u + 193003u), spRandom(i * 2u + 193004u));
    split = spPrefilter(hit, params.refSpp, b, false) * spLookup(hit);
  }
  referenceOut[i] = SpReference(vec4f(raw, select(0.0, 1.0, supported)), vec4f(split, hit.wo.z));
}

@compute @workgroup_size(16)
fn furnace(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  let band = i % 4u;
  let mu = 0.12 + 0.2 * f32(i / 4u);
  let hit = SpHit(vec3f(0.0), vec3f(sqrt(1.0 - mu * mu), 0.0, mu), SP_ROUGHNESS[band], PT_BSDF_CONDUCTOR);
  let samples = max(4096u, params.lutSpp);
  let raw = spRaw(hit, samples, vec2f(0.37, 0.44), true);
  let filtered = spPrefilter(hit, samples, vec2f(0.91, 0.13), true);
  furnaceOut[i] = vec4f(abs(raw - filtered * spLookup(hit)), filtered.x);
}
`;

const ENCODING = /* wgsl */ `
${nrcEncodingWgsl(4)}
fn spEncode(hit: SpHit) -> array<f32, NN_IN> {
  let r = ptReflect(hit.wo, SP_NORMAL);
  let angles = vec2f(acos(clamp(r.y, -1.0, 1.0)) / PT_PI, atan2(r.z, r.x) / (2.0 * PT_PI) + 0.5);
  var x = nnEncodeNrc(clamp(hit.p * vec3f(1.0 / 1.05, 1.0 / 0.58, 1.0), vec3f(-1.0), vec3f(1.0)),
    angles, vec2f(0.5, 0.75), hit.roughness, vec3f(0.0), ptFrComplex(1.0, SP_ETA, SP_K));
  x[NN_IN - 2u] = 1.0 + 0.2 * f32(hit.kind);
  return x;
}
`;

export function specularTrainHooks(prefilter: boolean): string {
  return /* wgsl */ `
${FIXTURE}
${ENCODING}
@group(1) @binding(0) var<uniform> spParams: SpParams;
fn nnTrainLoad(r: u32, x: ptr<function, array<f32, NN_IN>>) -> vec3f {
  let key = spMix(r ^ spMix(nnState.step * 0x9e3779b9u + spParams.seed));
  let band = min(u32(spRandom(key + 2u) * 4.0), 3u);
  let uv = vec2f(spRandom(key), (f32(band) + spRandom(key + 1u)) / ${SPECULAR_BANDS}.0);
  let hit = spHit(uv, spRandom(key + 3u) < 0.5, (spRandom(key + 4u) - 0.5) * 0.65, band);
  *x = spEncode(hit);
  let rotation = vec2f(spRandom(key + 5u), spRandom(key + 6u));
  return ${prefilter ? 'spPrefilter' : 'spRaw'}(hit, spParams.trainSpp, rotation, false);
}
`;
}

export function specularInferHooks(prefilter: boolean): string {
  return /* wgsl */ `
${FIXTURE}
${ENCODING}
@group(1) @binding(0) var<uniform> spParams: SpParams;
@group(1) @binding(1) var<storage, read> spReference: array<SpReference>;
@group(1) @binding(2) var<storage, read_write> spOutput: array<vec4f>;
${prefilter ? `@group(1) @binding(3) var<storage, read> spLut: array<vec4f>;
${LOOKUP}` : ''}
fn nnInferLoad(r: u32) -> array<f32, NN_IN> { return spEncode(spQuery(r, spParams)); }
fn nnInferStore(r: u32, y: vec3f) {
  let traced = spReference[r].raw;
  var result = traced.xyz;
  if (traced.w == 1.0) {
    result = max(y, vec3f(0.0))${prefilter ? ' * spLookup(spQuery(r, spParams))' : ''};
  }
  spOutput[r] = vec4f(result, traced.w);
}
`;
}
