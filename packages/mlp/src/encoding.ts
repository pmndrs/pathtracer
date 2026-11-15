/**
 * Input encodings, CPU and WGSL, with the same math so the oracle and the GPU see identical inputs.
 *
 * - Frequency encoding: a triangle wave at 2^0 … 2^(n−1) times the input (a cheap stand-in for sin(πx) used in
 *   NRC implementations). tri(x) = 1 − 4·|fract(x/2 + 1/4) − 1/2|: tri(0) = 0, tri(1/2) = 1, tri(3/2) = −1.
 * - One-blob encoding (Müller et al. 2019): k bins on [0, 1]; each bin holds the mass of a kernel centred on x that
 *   falls inside it. The kernel is the quartic (biweight) K(u) = 15/16 (1 − u²)² with radius 1/k, whose CDF is
 *   1/2 + 15/16 (u − 2u³/3 + u⁵/5) on [−1, 1].
 *
 * The NRC layout (64 values) follows VkNRC's design: position (normalised to the scene box, [−1, 1]) 3 × 12
 * frequencies; scattered direction and normal as spherical (θ/π, φ/2π) pairs, 4 one-blob bins each; roughness as
 * 1 − e^(−r), 4 bins; diffuse and specular reflectance raw; two constant ones (they act as a bias input).
 */

export const FREQUENCIES = 12;
export const BLOB_BINS = 4;

export function triangleWave(x: number): number {
  const f = x / 2 + 0.25;
  return 1 - 4 * Math.abs(f - Math.floor(f) - 0.5);
}

function quarticCdf(u: number): number {
  const c = Math.min(Math.max(u, -1), 1);
  const c2 = c * c;
  return 0.5 + (15 / 16) * c * (1 - (2 / 3) * c2 + (1 / 5) * c2 * c2);
}

/** Writes BLOB_BINS values for x ∈ [0, 1] at out[offset]. */
export function oneBlob(x: number, out: Float32Array | Float64Array, offset: number): void {
  const inv = BLOB_BINS; // kernel radius = 1 / bins
  for (let b = 0; b < BLOB_BINS; b++) {
    const lo = b / BLOB_BINS;
    const hi = (b + 1) / BLOB_BINS;
    out[offset + b] = quarticCdf((hi - x) * inv) - quarticCdf((lo - x) * inv);
  }
}

/** Writes FREQUENCIES values at out[offset]. */
export function frequency(x: number, out: Float32Array | Float64Array, offset: number): void {
  let scale = 1;
  for (let f = 0; f < FREQUENCIES; f++, scale *= 2) out[offset + f] = triangleWave(x * scale);
}

/** One NRC query, already in the units the encoding wants. */
export interface NrcInput {
  /** Position normalised to the scene box, each component in [−1, 1]. */
  position: [number, number, number];
  /** Scattered direction as (θ/π, φ/2π), each in [0, 1]. */
  direction: [number, number];
  /** Shading normal as (θ/π, φ/2π). */
  normal: [number, number];
  roughness: number;
  diffuse: [number, number, number];
  specular: [number, number, number];
}

export const NRC_ENCODED = 64;

export function encodeNrc(input: NrcInput, out: Float32Array | Float64Array = new Float32Array(NRC_ENCODED)): typeof out {
  let o = 0;
  for (let a = 0; a < 3; a++, o += FREQUENCIES) frequency(input.position[a] as number, out, o);
  for (const v of [...input.direction, ...input.normal]) {
    oneBlob(v, out, o);
    o += BLOB_BINS;
  }
  oneBlob(1 - Math.exp(-input.roughness), out, o);
  o += BLOB_BINS;
  for (const v of [...input.diffuse, ...input.specular]) out[o++] = v;
  out[o++] = 1;
  out[o++] = 1;
  if (o !== NRC_ENCODED) throw new Error(`encodeNrc wrote ${o} values`);
  return out;
}

/**
 * Frequency layout of the image encoding (N1.3 A/B). `octaves` triangle waves per axis at 2^0 … 2^(octaves−1);
 * `phases: 2` adds a quarter-period-shifted twin of each wave (tri(x + 1/2), the cosine to the sine), so position is
 * recoverable inside every period. A wave at 2^k spans 2^(k−1) periods over [0, 1]: a size-pixel image resolves up to
 * size/2 periods, so octaves ≤ log2(size) (8 for 256²) stay under Nyquist.
 */
export interface ImageEncoding {
  octaves: number;
  phases: 1 | 2;
}

/** Today's layout (VkNRC's position encoding applied to (u, v)): 12 one-phase octaves. */
export const DEFAULT_IMAGE_ENCODING: ImageEncoding = { octaves: FREQUENCIES, phases: 1 };

/** Inputs an image encoding uses (before zero padding). */
export function imageEncodingWidth(e: ImageEncoding): number {
  return 2 * e.octaves * e.phases + 2 * BLOB_BINS + 1;
}

function validateImageEncoding(e: ImageEncoding, width: number): void {
  if (!Number.isInteger(e.octaves) || e.octaves < 1 || e.octaves > 16) throw new Error(`image encoding: octaves must be 1..16, got ${e.octaves}`);
  if (e.phases !== 1 && e.phases !== 2) throw new Error(`image encoding: phases must be 1 or 2, got ${e.phases}`);
  if (imageEncodingWidth(e) > width) throw new Error(`image encoding needs ${imageEncodingWidth(e)} inputs, the net has ${width}`);
}

/**
 * The "learn an image" encoding (N1): (u, v) ∈ [0, 1]². Frequencies of u, then of v (octaves × phases each; for each
 * octave the phase-0 wave, then the shifted one), one-blob of u and v (8), a constant one, zero padding to `width`.
 * The default layout is the original one (12 one-phase octaves, 24 + 8 + 1 values).
 */
export function encodeImage(u: number, v: number, width = 64, out: Float32Array | Float64Array = new Float32Array(width), e: ImageEncoding = DEFAULT_IMAGE_ENCODING): typeof out {
  validateImageEncoding(e, width);
  out.fill(0);
  let o = 0;
  for (const x of [u, v]) {
    let scale = 1;
    for (let f = 0; f < e.octaves; f++, scale *= 2) {
      out[o++] = triangleWave(x * scale);
      if (e.phases === 2) out[o++] = triangleWave(x * scale + 0.5);
    }
  }
  oneBlob(u, out, o);
  oneBlob(v, out, o + BLOB_BINS);
  out[o + 2 * BLOB_BINS] = 1;
  return out;
}

/**
 * WGSL twin of `encodeImage` for a given layout: `fn <name>(uv: vec2f) -> array<f32, NN_IN>`. Needs `encodingWgsl()`
 * (for `nnTriangle` and `nnOneBlob`), which every MLP kernel already includes.
 */
export function imageEncodingWgsl(e: ImageEncoding = DEFAULT_IMAGE_ENCODING, name = 'nnEncodeImageWith', width = 64): string {
  validateImageEncoding(e, width);
  const per = e.octaves * e.phases;
  return /* wgsl */ `
fn ${name}(uv: vec2f) -> array<f32, NN_IN> {
  var e: array<f32, NN_IN>;
  for (var a = 0u; a < 2u; a++) {
    let x = uv[a];
    var scale = 1.0;
    for (var f = 0u; f < ${e.octaves}u; f++) {
      e[a * ${per}u + f * ${e.phases}u] = nnTriangle(x * scale);${e.phases === 2 ? `
      e[a * ${per}u + f * 2u + 1u] = nnTriangle(x * scale + 0.5);` : ''}
      scale *= 2.0;
    }
  }
  nnOneBlob(uv.x, &e, ${2 * per}u);
  nnOneBlob(uv.y, &e, ${2 * per + BLOB_BINS}u);
  e[${2 * per + 2 * BLOB_BINS}u] = 1.0;
  return e;
}
`;
}

/** WGSL for the same encodings. `T` is the activation scalar type (f32 or f16). */
export function encodingWgsl(): string {
  return /* wgsl */ `
const NN_FREQUENCIES = ${FREQUENCIES}u;
const NN_BLOB_BINS = ${BLOB_BINS}u;

fn nnTriangle(x: f32) -> f32 {
  let f = x * 0.5 + 0.25;
  return 1.0 - 4.0 * abs(fract(f) - 0.5);
}

fn nnQuarticCdf(u: f32) -> f32 {
  let c = clamp(u, -1.0, 1.0);
  let c2 = c * c;
  return 0.5 + (15.0 / 16.0) * c * (1.0 - (2.0 / 3.0) * c2 + 0.2 * c2 * c2);
}

fn nnFrequency(x: f32, out: ptr<function, array<f32, NN_IN>>, offset: u32) {
  var scale = 1.0;
  for (var f = 0u; f < NN_FREQUENCIES; f++) {
    (*out)[offset + f] = nnTriangle(x * scale);
    scale *= 2.0;
  }
}

fn nnOneBlob(x: f32, out: ptr<function, array<f32, NN_IN>>, offset: u32) {
  let inv = f32(NN_BLOB_BINS);
  for (var b = 0u; b < NN_BLOB_BINS; b++) {
    let lo = f32(b) / inv;
    let hi = f32(b + 1u) / inv;
    (*out)[offset + b] = nnQuarticCdf((hi - x) * inv) - nnQuarticCdf((lo - x) * inv);
  }
}

fn nnEncodeImage(uv: vec2f) -> array<f32, NN_IN> {
  var e: array<f32, NN_IN>;
  nnFrequency(uv.x, &e, 0u);
  nnFrequency(uv.y, &e, NN_FREQUENCIES);
  nnOneBlob(uv.x, &e, 2u * NN_FREQUENCIES);
  nnOneBlob(uv.y, &e, 2u * NN_FREQUENCIES + NN_BLOB_BINS);
  e[2u * NN_FREQUENCIES + 2u * NN_BLOB_BINS] = 1.0;
  return e;
}
`;
}

/**
 * WGSL twin of `encodeNrc` (VkNRC's 64-input layout): `fn nnEncodeNrc(position, direction, normal, roughness, diffuse,
 * specular) -> array<f32, NN_IN>` with position already normalised to [−1, 1]³ and direction / normal as (θ/π, φ/2π).
 * Needs `encodingWgsl()` (every MLP kernel includes it) and NN_IN = 64.
 */
export function nrcEncodingWgsl(): string {
  return /* wgsl */ `
fn nnEncodeNrc(position: vec3f, direction: vec2f, normal: vec2f, roughness: f32, diffuse: vec3f, specular: vec3f) -> array<f32, NN_IN> {
  var e: array<f32, NN_IN>;
  nnFrequency(position.x, &e, 0u);
  nnFrequency(position.y, &e, NN_FREQUENCIES);
  nnFrequency(position.z, &e, 2u * NN_FREQUENCIES);
  let o = 3u * NN_FREQUENCIES;
  nnOneBlob(direction.x, &e, o);
  nnOneBlob(direction.y, &e, o + NN_BLOB_BINS);
  nnOneBlob(normal.x, &e, o + 2u * NN_BLOB_BINS);
  nnOneBlob(normal.y, &e, o + 3u * NN_BLOB_BINS);
  nnOneBlob(1.0 - exp(-roughness), &e, o + 4u * NN_BLOB_BINS);
  let c = o + 5u * NN_BLOB_BINS;
  e[c] = diffuse.x; e[c + 1u] = diffuse.y; e[c + 2u] = diffuse.z;
  e[c + 3u] = specular.x; e[c + 4u] = specular.y; e[c + 5u] = specular.z;
  e[c + 6u] = 1.0; e[c + 7u] = 1.0;
  return e;
}
`;
}
