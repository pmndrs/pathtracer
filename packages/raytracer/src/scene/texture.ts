/**
 * Material textures for the scene copy (P1.4c, first slice): the layout that works on every WebGPU device, its CPU build, and the
 * TypeScript twins of the shader lookup. Bindless (`chromium-experimental-sampling-resource-table`, textures only, not on macOS)
 * stays a later Tier 3 option behind the same slot API: a material references a texture by slot, the slot says where it lives.
 *
 * Layout ("size buckets"): every texture is resampled to a power-of-two size (nearest in log space, capped at `maxSize`),
 * gets a full mip chain built here on the CPU, and is stored as one layer of a `texture_2d_array` shared by all textures of the
 * same (width, height, colour space, wrap mode). One array = one texture binding + one sampler, so a scene costs
 * `buckets.length` of the 16 sampled textures a shader stage may bind (`maxSampledTexturesPerShaderStage`), whatever the number
 * of textures. Arrays avoid atlas bleeding: a tile never filters into its neighbour, and `repeat` wraps inside the texture.
 *
 *  - colour space: `srgb` textures (base colour, emission) use `rgba8unorm-srgb`, so the hardware decodes to linear BEFORE
 *    filtering (what pbrt-v4 and Mitsuba do); `linear` ones (metallic-roughness, normal) use `rgba8unorm`. Alpha is always linear.
 *  - wrap: part of the bucket (one sampler per bucket; WGSL has no sampler arrays), so `repeat`, `clamp` and `mirror` textures
 *    never share an array. glTF default is `repeat`.
 *  - mips: 2x2 box in linear light (colour) or on the raw values (data), written level by level; the sampler is trilinear.
 *  - slot record (1 vec4u in the scene table): (bucket, layer, width | height << 16, mip levels).
 *
 * If a scene needs more arrays than `maxBuckets` (default 8; layers cap at `maxLayers`, default 256), the buckets with the
 * cheapest merge are merged (their textures resampled up to the larger size) until it fits; a scene that still does not fit throws.
 *
 * Provenance: the size-bucket + array idea is common (three.js texture arrays, glTF viewers); nothing here is copied. sRGB transfer
 * function: IEC 61966-2-1.
 */

export type TextureWrap = 'repeat' | 'clamp' | 'mirror';
export type TextureColorSpace = 'srgb' | 'linear';

export interface TextureInput {
  width: number;
  height: number;
  /** RGBA8, row 0 = top (glTF: uv (0, 0) is the first texel of the first row), `width * height * 4` values 0-255. */
  data: ArrayLike<number>;
  /** `srgb` for base colour and emission, `linear` for metallic-roughness and normal maps. The material fields that use a texture check it. */
  colorSpace: TextureColorSpace;
  /** Both axes. Default `repeat`. */
  wrap?: TextureWrap;
  label?: string;
}

export interface TextureOptions {
  /** Largest stored width or height (power of two). Default 4096 (WebGPU guarantees 8192). Larger textures are downsampled. */
  maxSize?: number;
  /** Rounding of a non-power-of-two size: `nearest` in log space (default), `up` (sharper, more memory) or `down`. */
  rounding?: 'nearest' | 'up' | 'down';
  /** Most arrays (texture + sampler bindings) the scene may use. Default 8 (the shader stage limit is 16 of each). */
  maxBuckets?: number;
  /** Layers per array. Default 256 (the WebGPU default `maxTextureArrayLayers`). */
  maxLayers?: number;
  /**
   * Build the mip chain. Default true (on the CPU: 2x2 box in linear light, about 50 ms per megapixel, so 24 textures of 2048x2048
   * take ~5 s); false stores one level (every lookup is the base level); `'gpu'` plans the full chain but builds only level 0 here: the
   * upload (`uploadSceneCopy({ mipGenerator })`) fills levels 1+ on the GPU (a `@pmndrs/downsampler` `Downsampler`, X2.d). The CPU
   * twin then has level 0 only (`sampleSceneTexture` throws above it), and the codes can differ from the CPU chain by a rounding step.
   */
  mips?: boolean | 'gpu';
}

export interface TextureBucket {
  width: number;
  height: number;
  colorSpace: TextureColorSpace;
  wrap: TextureWrap;
  layers: number;
  mipLevels: number;
  /** One array per mip level: `layers * w * h * 4` bytes, layer-major, row 0 = top. Level 0 only when `SceneTextures.gpuMips`. */
  mips: Uint8Array[];
}

export interface TextureSlot {
  bucket: number;
  layer: number;
  /** Stored size (the bucket's). */
  width: number;
  height: number;
  mipLevels: number;
  /** The size the caller gave. */
  sourceWidth: number;
  sourceHeight: number;
  label?: string;
}

export interface SceneTextures {
  buckets: TextureBucket[];
  slots: TextureSlot[];
  /** Bytes of texel data including mips (the GPU's, also when `gpuMips` leaves them out of `buckets[].mips`). */
  byteLength: number;
  /** `TextureOptions.mips: 'gpu'`: `buckets[].mips` hold level 0 only and the upload needs a mip generator. */
  gpuMips?: boolean;
}

/** What the WGSL generator and the bind group layout need to know: no texel data. */
export interface TextureLayout {
  buckets: ReadonlyArray<Pick<TextureBucket, 'width' | 'height' | 'colorSpace' | 'wrap' | 'layers' | 'mipLevels'>>;
}

export const TEXTURE_RECORD_VEC4S = 1;
export const TEXTURE_RECORD_WORDS = 4;
export const DEFAULT_MAX_TEXTURE_BUCKETS = 8;

export function textureLayoutOf(textures: SceneTextures): TextureLayout {
  return { buckets: textures.buckets.map(({ width, height, colorSpace, wrap, layers, mipLevels }) => ({ width, height, colorSpace, wrap, layers, mipLevels })) };
}

// ---------------------------------------------------------------------------------------------------------------
// sRGB
// ---------------------------------------------------------------------------------------------------------------

/** 8-bit sRGB code -> linear, as f32 (what `rgba8unorm-srgb` hardware returns, to within its own rounding). */
export const SRGB_TO_LINEAR: Float32Array = (() => {
  const table = new Float32Array(256);
  for (let i = 0; i < 256; i++) {
    const c = i / 255;
    table[i] = c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
  }
  return table;
})();

export function srgbToLinear(code: number): number {
  return SRGB_TO_LINEAR[code & 255] as number;
}

/** Linear [0, 1] -> nearest 8-bit sRGB code. */
export function linearToSrgb8(linear: number): number {
  const x = Math.min(1, Math.max(0, linear));
  const c = x <= 0.0031308 ? x * 12.92 : 1.055 * Math.pow(x, 1 / 2.4) - 0.055;
  return Math.min(255, Math.max(0, Math.round(c * 255)));
}

// ---------------------------------------------------------------------------------------------------------------
// Sizes
// ---------------------------------------------------------------------------------------------------------------

export function isPowerOfTwo(n: number): boolean {
  return Number.isInteger(n) && n > 0 && (n & (n - 1)) === 0;
}

/** The stored size of one axis: a power of two near `n`, at most `maxSize`. */
export function bucketSize(n: number, rounding: 'nearest' | 'up' | 'down', maxSize: number): number {
  if (!(n >= 1)) throw new RangeError(`texture size must be >= 1, got ${n}`);
  let p: number;
  if (isPowerOfTwo(n)) p = n;
  else {
    const down = 2 ** Math.floor(Math.log2(n));
    const up = down * 2;
    // Nearest in log space: the geometric mean of the two candidates is the tie point.
    p = rounding === 'up' ? up : rounding === 'down' ? down : n * n >= down * up ? up : down;
  }
  return Math.min(p, maxSize);
}

/** Mip levels of a width x height texture: down to 1x1. */
export function mipLevelCount(width: number, height: number): number {
  return Math.floor(Math.log2(Math.max(width, height))) + 1;
}

// ---------------------------------------------------------------------------------------------------------------
// Resampling (only when the source size is not the stored size)
// ---------------------------------------------------------------------------------------------------------------

/** Separable tent resample of one axis. `src` has `channels` interleaved values per texel along `axis`. */
function resampleAxis(src: Float32Array, sw: number, sh: number, dw: number, dh: number, horizontal: boolean): Float32Array {
  const out = new Float32Array(dw * dh * 4);
  const n = horizontal ? sw : sh;
  const m = horizontal ? dw : dh;
  const scale = Math.max(1, n / m);
  for (let d = 0; d < m; d++) {
    const centre = ((d + 0.5) * n) / m;
    const lo = Math.max(0, Math.floor(centre - scale));
    const hi = Math.min(n - 1, Math.ceil(centre + scale));
    const weights: number[] = [];
    let sum = 0;
    for (let j = lo; j <= hi; j++) {
      const w = Math.max(0, 1 - Math.abs(j + 0.5 - centre) / scale);
      weights.push(w);
      sum += w;
    }
    for (let k = 0; k < weights.length; k++) weights[k] = (weights[k] as number) / sum;
    const other = horizontal ? sh : dw;
    for (let o = 0; o < other; o++) {
      for (let c = 0; c < 4; c++) {
        let acc = 0;
        for (let k = 0; k < weights.length; k++) {
          const j = lo + k;
          const index = horizontal ? (o * sw + j) * 4 + c : (j * sw + o) * 4 + c;
          acc += (weights[k] as number) * (src[index] as number);
        }
        out[horizontal ? (o * dw + d) * 4 + c : (d * dw + o) * 4 + c] = acc;
      }
    }
  }
  return out;
}

/** Decodes RGBA8 to linear floats (colour textures: sRGB -> linear on rgb; alpha and data textures are plain unorm). */
function toFloat(input: TextureInput): Float32Array {
  const out = new Float32Array(input.width * input.height * 4);
  const srgb = input.colorSpace === 'srgb';
  for (let i = 0; i < out.length; i++) {
    const code = (input.data[i] as number) & 255;
    out[i] = srgb && (i & 3) !== 3 ? (SRGB_TO_LINEAR[code] as number) : code / 255;
  }
  return out;
}

function toBytes(values: Float32Array, srgb: boolean): Uint8Array {
  const out = new Uint8Array(values.length);
  for (let i = 0; i < out.length; i++) {
    const v = values[i] as number;
    out[i] = srgb && (i & 3) !== 3 ? linearToSrgb8(v) : Math.min(255, Math.max(0, Math.round(v * 255)));
  }
  return out;
}

/** Resamples to `dw x dh` (tent filter: bilinear up, area-weighted down). Returns the RGBA8 bytes, colour space preserved. Exported for tests. */
export function resampleTexture(input: TextureInput, dw: number, dh: number): Uint8Array {
  // A byte array is copied with the typed-array constructor (memcpy); `Uint8Array.from(.., fn)` costs ~60 ns per byte (6 s for 24 textures of 1024x1024).
  if (input.width === dw && input.height === dh) return input.data instanceof Uint8Array || input.data instanceof Uint8ClampedArray ? new Uint8Array(input.data) : Uint8Array.from(input.data, (v) => v & 255);
  let values = toFloat(input);
  if (input.width !== dw) values = resampleAxis(values, input.width, input.height, dw, input.height, true);
  if (input.height !== dh) values = resampleAxis(values, dw, input.height, dw, dh, false);
  return toBytes(values, input.colorSpace === 'srgb');
}

/** One 2x2 box reduction in linear light (colour) or on the raw values (data). `w` and `h` are the source size; each is halved (minimum 1). */
export function downsampleBytes(src: Uint8Array, w: number, h: number, srgb: boolean): Uint8Array {
  const dw = Math.max(1, w >> 1);
  const dh = Math.max(1, h >> 1);
  const out = new Uint8Array(dw * dh * 4);
  for (let y = 0; y < dh; y++) {
    for (let x = 0; x < dw; x++) {
      const x0 = Math.min(w - 1, x * 2);
      const x1 = Math.min(w - 1, x * 2 + 1);
      const y0 = Math.min(h - 1, y * 2);
      const y1 = Math.min(h - 1, y * 2 + 1);
      for (let c = 0; c < 4; c++) {
        const a = src[(y0 * w + x0) * 4 + c] as number;
        const b = src[(y0 * w + x1) * 4 + c] as number;
        const d = src[(y1 * w + x0) * 4 + c] as number;
        const e = src[(y1 * w + x1) * 4 + c] as number;
        if (srgb && c < 3) out[(y * dw + x) * 4 + c] = linearToSrgb8(((SRGB_TO_LINEAR[a] as number) + (SRGB_TO_LINEAR[b] as number) + (SRGB_TO_LINEAR[d] as number) + (SRGB_TO_LINEAR[e] as number)) * 0.25);
        else out[(y * dw + x) * 4 + c] = (a + b + d + e + 2) >> 2;
      }
    }
  }
  return out;
}

// ---------------------------------------------------------------------------------------------------------------
// Planning and building
// ---------------------------------------------------------------------------------------------------------------

interface Group {
  width: number;
  height: number;
  colorSpace: TextureColorSpace;
  wrap: TextureWrap;
  members: number[];
}

const keyOf = (g: Pick<Group, 'width' | 'height' | 'colorSpace' | 'wrap'>) => `${g.width}x${g.height}/${g.colorSpace}/${g.wrap}`;
const arraysOf = (groups: readonly Group[], maxLayers: number) => groups.reduce((n, g) => n + Math.ceil(g.members.length / maxLayers), 0);

/** Merges the cheapest same-colour-space, same-wrap pair of groups (extra texels stored) until at most `maxBuckets` arrays remain. */
function mergeGroups(groups: Group[], maxBuckets: number, maxLayers: number): Group[] {
  while (arraysOf(groups, maxLayers) > maxBuckets) {
    let best: { a: number; b: number; cost: number } | null = null;
    for (let a = 0; a < groups.length; a++) {
      for (let b = a + 1; b < groups.length; b++) {
        const ga = groups[a] as Group;
        const gb = groups[b] as Group;
        if (ga.colorSpace !== gb.colorSpace || ga.wrap !== gb.wrap) continue;
        const w = Math.max(ga.width, gb.width);
        const h = Math.max(ga.height, gb.height);
        const cost = ga.members.length * (w * h - ga.width * ga.height) + gb.members.length * (w * h - gb.width * gb.height);
        if (!best || cost < best.cost) best = { a, b, cost };
      }
    }
    if (!best) throw new RangeError(`textures: ${arraysOf(groups, maxLayers)} arrays needed (colour space x wrap x size, ${maxLayers} layers each) but at most ${maxBuckets} are allowed; use fewer wrap modes or raise maxBuckets`);
    const ga = groups[best.a] as Group;
    const gb = groups[best.b] as Group;
    const merged: Group = { width: Math.max(ga.width, gb.width), height: Math.max(ga.height, gb.height), colorSpace: ga.colorSpace, wrap: ga.wrap, members: [...ga.members, ...gb.members].sort((x, y) => x - y) };
    // A merge may land on an existing group's key: fold that one in too.
    groups = groups.filter((_, i) => i !== best!.a && i !== best!.b);
    const same = groups.findIndex((g) => keyOf(g) === keyOf(merged));
    if (same >= 0) {
      merged.members = [...merged.members, ...(groups[same] as Group).members].sort((x, y) => x - y);
      groups.splice(same, 1);
    }
    groups.push(merged);
  }
  return groups;
}

/** Plans the buckets for `inputs` and builds every mip level. Slot `i` is `inputs[i]`. */
export function buildSceneTextures(inputs: readonly TextureInput[], options: TextureOptions = {}): SceneTextures {
  const maxSize = options.maxSize ?? 4096;
  if (!isPowerOfTwo(maxSize)) throw new RangeError(`textures: maxSize must be a power of two, got ${maxSize}`);
  const maxBuckets = options.maxBuckets ?? DEFAULT_MAX_TEXTURE_BUCKETS;
  const maxLayers = options.maxLayers ?? 256;
  const rounding = options.rounding ?? 'nearest';
  const wantMips = options.mips ?? true;
  const gpuMips = options.mips === 'gpu';

  inputs.forEach((t, i) => {
    if (!Number.isInteger(t.width) || !Number.isInteger(t.height) || t.width < 1 || t.height < 1) throw new RangeError(`texture ${i}${t.label ? ` (${t.label})` : ''}: bad size ${t.width}x${t.height}`);
    if (t.data.length !== t.width * t.height * 4) throw new RangeError(`texture ${i}${t.label ? ` (${t.label})` : ''}: ${t.data.length} values for ${t.width}x${t.height} RGBA8 (expected ${t.width * t.height * 4})`);
  });

  let groups: Group[] = [];
  inputs.forEach((t, i) => {
    const probe = { width: bucketSize(t.width, rounding, maxSize), height: bucketSize(t.height, rounding, maxSize), colorSpace: t.colorSpace, wrap: t.wrap ?? ('repeat' as TextureWrap) };
    const existing = groups.find((g) => keyOf(g) === keyOf(probe));
    if (existing) existing.members.push(i);
    else groups.push({ ...probe, members: [i] });
  });
  groups = mergeGroups(groups, maxBuckets, maxLayers).sort((a, b) => a.width * a.height - b.width * b.height || (a.colorSpace < b.colorSpace ? -1 : 1) || (a.wrap < b.wrap ? -1 : 1));

  const buckets: TextureBucket[] = [];
  const slots: TextureSlot[] = new Array(inputs.length);
  let byteLength = 0;
  for (const g of groups) {
    for (let first = 0; first < g.members.length; first += maxLayers) {
      const members = g.members.slice(first, first + maxLayers);
      const levels = wantMips ? mipLevelCount(g.width, g.height) : 1;
      const srgb = g.colorSpace === 'srgb';
      const mips: Uint8Array[] = [];
      for (let l = 0; l < (gpuMips ? 1 : levels); l++) mips.push(new Uint8Array(members.length * Math.max(1, g.width >> l) * Math.max(1, g.height >> l) * 4));
      members.forEach((inputIndex, layer) => {
        const input = inputs[inputIndex] as TextureInput;
        let level = resampleTexture(input, g.width, g.height);
        let w = g.width;
        let h = g.height;
        for (let l = 0; l < mips.length; l++) {
          (mips[l] as Uint8Array).set(level, layer * w * h * 4);
          if (l + 1 < mips.length) {
            level = downsampleBytes(level, w, h, srgb);
            w = Math.max(1, w >> 1);
            h = Math.max(1, h >> 1);
          }
        }
        slots[inputIndex] = { bucket: buckets.length, layer, width: g.width, height: g.height, mipLevels: levels, sourceWidth: input.width, sourceHeight: input.height, ...(input.label ? { label: input.label } : {}) };
      });
      buckets.push({ width: g.width, height: g.height, colorSpace: g.colorSpace, wrap: g.wrap, layers: members.length, mipLevels: levels, mips });
      byteLength += gpuMips ? members.length * 4 * Array.from({ length: levels }, (_, l) => Math.max(1, g.width >> l) * Math.max(1, g.height >> l)).reduce((n, texels) => n + texels, 0) : mips.reduce((n, m) => n + m.byteLength, 0);
    }
  }
  if (buckets.length > maxBuckets) throw new RangeError(`textures: ${buckets.length} arrays needed but at most ${maxBuckets} are allowed`);
  return { buckets, slots, byteLength, ...(gpuMips ? { gpuMips: true } : {}) };
}

/** Writes one slot record (4 words) at `words[wordOffset]`. */
export function packTextureSlot(slot: TextureSlot, words: Uint32Array, wordOffset: number): void {
  words[wordOffset] = slot.bucket;
  words[wordOffset + 1] = slot.layer;
  words[wordOffset + 2] = (slot.width | (slot.height << 16)) >>> 0;
  words[wordOffset + 3] = slot.mipLevels;
}

export interface UnpackedTextureSlot {
  bucket: number;
  layer: number;
  width: number;
  height: number;
  mipLevels: number;
}

export function unpackTextureSlot(words: ArrayLike<number>, wordOffset: number): UnpackedTextureSlot {
  const z = (words[wordOffset + 2] as number) >>> 0;
  return { bucket: (words[wordOffset] as number) >>> 0, layer: (words[wordOffset + 1] as number) >>> 0, width: z & 0xffff, height: z >>> 16, mipLevels: (words[wordOffset + 3] as number) >>> 0 };
}

// ---------------------------------------------------------------------------------------------------------------
// Twins of the shader lookup
// ---------------------------------------------------------------------------------------------------------------

/** Texel index of coordinate `i` along an axis of `n` texels under a wrap mode (the GPU's address modes). */
export function wrapIndex(i: number, n: number, wrap: TextureWrap): number {
  if (wrap === 'clamp') return Math.min(n - 1, Math.max(0, i));
  if (wrap === 'repeat') return ((i % n) + n) % n;
  const period = ((i % (2 * n)) + 2 * n) % (2 * n);
  return period < n ? period : 2 * n - 1 - period;
}

/** Bilinear lookup of one level and layer: linear-light rgb (sRGB textures decoded before filtering) and alpha. */
export function sampleBucketLevel(bucket: TextureBucket, layer: number, level: number, u: number, v: number): [number, number, number, number] {
  const w = Math.max(1, bucket.width >> level);
  const h = Math.max(1, bucket.height >> level);
  const data = bucket.mips[level];
  if (!data) throw new RangeError(`texture level ${level}: this scene's mips are built on the GPU (TextureOptions.mips 'gpu'); the CPU twin has level 0 only`);
  const x = u * w - 0.5;
  const y = v * h - 0.5;
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const fx = x - x0;
  const fy = y - y0;
  const out: [number, number, number, number] = [0, 0, 0, 0];
  const srgb = bucket.colorSpace === 'srgb';
  for (let k = 0; k < 4; k++) {
    const xi = [wrapIndex(x0, w, bucket.wrap), wrapIndex(x0 + 1, w, bucket.wrap)] as const;
    const yi = [wrapIndex(y0, h, bucket.wrap), wrapIndex(y0 + 1, h, bucket.wrap)] as const;
    const texel = (xx: number, yy: number) => {
      const code = data[((layer * h + yy) * w + xx) * 4 + k] as number;
      return srgb && k < 3 ? (SRGB_TO_LINEAR[code] as number) : code / 255;
    };
    out[k] = (1 - fy) * ((1 - fx) * texel(xi[0], yi[0]) + fx * texel(xi[1], yi[0])) + fy * ((1 - fx) * texel(xi[0], yi[1]) + fx * texel(xi[1], yi[1]));
  }
  return out;
}

/** CPU twin of WGSL `rtTexSample(slot, uv, lod)`: trilinear, `lod` clamped to the stored levels. */
export function sampleSceneTexture(textures: SceneTextures, slot: number, u: number, v: number, lod = 0): [number, number, number, number] {
  const s = textures.slots[slot];
  if (!s) throw new RangeError(`texture slot ${slot} of ${textures.slots.length}`);
  const bucket = textures.buckets[s.bucket] as TextureBucket;
  const l = Math.min(Math.max(lod, 0), s.mipLevels - 1);
  const l0 = Math.floor(l);
  const l1 = Math.min(l0 + 1, s.mipLevels - 1);
  const t = l - l0;
  const a = sampleBucketLevel(bucket, s.layer, l0, u, v);
  if (t === 0 || l1 === l0) return a;
  const b = sampleBucketLevel(bucket, s.layer, l1, u, v);
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t, a[3] + (b[3] - a[3]) * t];
}

/** CPU twin of WGSL `rtTexLod(slot, footprint)`: `footprint` is the filter width in uv units; the level is log2 of it in texels of the stored size, at least 0. */
export function textureLod(width: number, height: number, uvFootprint: number): number {
  const texels = uvFootprint * Math.sqrt(width * height);
  return Math.max(Math.log2(Math.max(texels, 1e-8)), 0);
}

/**
 * Ray-cone level of detail (Akenine-Moller et al., "Improved Shader and Texture Level of Detail Using Ray Cones", JCGT 2021),
 * in uv units: `coneWidth` metres across at the hit, seen at `cosTheta` (|n . d|) to the surface, `uvDensity` = sqrt(uv area /
 * world area) of the triangle (uv units per metre). Isotropic: the longer footprint axis, `width / |cos|`, so nothing aliases
 * at the cost of blur at grazing angles. The cosine is floored at 1e-3.
 */
export function coneUvFootprint(coneWidth: number, cosTheta: number, uvDensity: number): number {
  return (coneWidth * uvDensity) / Math.max(Math.abs(cosTheta), 1e-3);
}

/** Triangle uv density and the tangent frame the shader builds (CPU twin of the `rtSurface` texture frame), in the space of the given positions. */
export function triangleTextureFrame(p0: readonly number[], p1: readonly number[], p2: readonly number[], uv0: readonly number[], uv1: readonly number[], uv2: readonly number[]): { uvDensity: number; tangent: [number, number, number]; handedness: number } {
  const e1 = [(p1[0] as number) - (p0[0] as number), (p1[1] as number) - (p0[1] as number), (p1[2] as number) - (p0[2] as number)];
  const e2 = [(p2[0] as number) - (p0[0] as number), (p2[1] as number) - (p0[1] as number), (p2[2] as number) - (p0[2] as number)];
  const d1 = [(uv1[0] as number) - (uv0[0] as number), (uv1[1] as number) - (uv0[1] as number)];
  const d2 = [(uv2[0] as number) - (uv0[0] as number), (uv2[1] as number) - (uv0[1] as number)];
  const det = (d1[0] as number) * (d2[1] as number) - (d1[1] as number) * (d2[0] as number);
  const cross = [(e1[1] as number) * (e2[2] as number) - (e1[2] as number) * (e2[1] as number), (e1[2] as number) * (e2[0] as number) - (e1[0] as number) * (e2[2] as number), (e1[0] as number) * (e2[1] as number) - (e1[1] as number) * (e2[0] as number)];
  const worldArea = 0.5 * Math.hypot(cross[0] as number, cross[1] as number, cross[2] as number);
  const uvArea = 0.5 * Math.abs(det);
  if (!(Math.abs(det) > 1e-30) || !(worldArea > 0)) return { uvDensity: 0, tangent: [1, 0, 0], handedness: 1 };
  const inv = 1 / det;
  // dp/du, and the bitangent that points toward increasing glTF "up" (decreasing v): -dp/dv.
  const t = [0, 1, 2].map((k) => ((e1[k] as number) * (d2[1] as number) - (e2[k] as number) * (d1[1] as number)) * inv);
  const b = [0, 1, 2].map((k) => -((e2[k] as number) * (d1[0] as number) - (e1[k] as number) * (d2[0] as number)) * inv);
  const tl = Math.hypot(t[0] as number, t[1] as number, t[2] as number) || 1;
  const n = [(cross[0] as number) / (2 * worldArea), (cross[1] as number) / (2 * worldArea), (cross[2] as number) / (2 * worldArea)];
  const nxt = [(n[1] as number) * (t[2] as number) - (n[2] as number) * (t[1] as number), (n[2] as number) * (t[0] as number) - (n[0] as number) * (t[2] as number), (n[0] as number) * (t[1] as number) - (n[1] as number) * (t[0] as number)];
  const handedness = (nxt[0] as number) * (b[0] as number) + (nxt[1] as number) * (b[1] as number) + (nxt[2] as number) * (b[2] as number) < 0 ? -1 : 1;
  return { uvDensity: Math.sqrt(uvArea / worldArea), tangent: [(t[0] as number) / tl, (t[1] as number) / tl, (t[2] as number) / tl], handedness };
}

/** CPU twin of WGSL `rtApplyNormalMap`: perturbs shading normal `n` (unit) with a tangent-space texel (rgb 0-1, glTF convention: +Y up, +Z out). */
export function applyNormalMap(n: readonly number[], tangent: readonly number[], handedness: number, texel: readonly number[]): [number, number, number] {
  const x = 2 * (texel[0] as number) - 1;
  const y = 2 * (texel[1] as number) - 1;
  const z = 2 * (texel[2] as number) - 1;
  const il = 1 / (Math.hypot(x, y, z) || 1);
  const d = (n[0] as number) * (tangent[0] as number) + (n[1] as number) * (tangent[1] as number) + (n[2] as number) * (tangent[2] as number);
  let t = [(tangent[0] as number) - (n[0] as number) * d, (tangent[1] as number) - (n[1] as number) * d, (tangent[2] as number) - (n[2] as number) * d];
  const tl = Math.hypot(t[0] as number, t[1] as number, t[2] as number) || 1;
  t = t.map((c) => c / tl);
  const b = [((n[1] as number) * (t[2] as number) - (n[2] as number) * (t[1] as number)) * handedness, ((n[2] as number) * (t[0] as number) - (n[0] as number) * (t[2] as number)) * handedness, ((n[0] as number) * (t[1] as number) - (n[1] as number) * (t[0] as number)) * handedness];
  const out = [0, 1, 2].map((k) => (t[k] as number) * x * il + (b[k] as number) * y * il + (n[k] as number) * z * il);
  const ol = Math.hypot(out[0] as number, out[1] as number, out[2] as number) || 1;
  return [(out[0] as number) / ol, (out[1] as number) / ol, (out[2] as number) / ol];
}
