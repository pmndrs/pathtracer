/**
 * MaterialX node table (P5): every supported node once, with its inputs and defaults, a TypeScript implementation and the WGSL
 * expression that computes the same thing. The two live side by side so they stay twins: `eval.ts` runs `ts`, `wgsl.ts` emits `wgsl`.
 *
 * Values are four lanes (`MtlxValue`; WGSL `vec4f`). Lanes past the type's component count are don't-care: a scalar that feeds a vector
 * input is splatted into all four lanes, and the type-sensitive nodes (dotproduct, normalize, magnitude, crossproduct, normalmap) read
 * only the lanes their type has. Semantics follow the MaterialX 1.38 standard library GLSL / nodegraph definitions (Apache-2.0).
 */
import { MTLX_COMPONENTS, parseMtlxNumeric, type MtlxValue } from './document.ts';

export type { MtlxValue };

export class MtlxUnsupportedError extends Error {
  readonly category: string;
  constructor(category: string, message: string) {
    super(message);
    this.name = 'MtlxUnsupportedError';
    this.category = category;
  }
}

export const MTLX_SCALAR_TYPES: ReadonlySet<string> = new Set(['float', 'integer', 'boolean']);

/** Converts a value of declared type `from` to the type `to` (scalar to vector = splat; 3 to 4 components = w 1; the rest is lane copy). */
export function coerceValue(v: MtlxValue, from: string, to: string): MtlxValue {
  if (from === to) return v;
  const fromScalar = MTLX_SCALAR_TYPES.has(from);
  const toN = MTLX_COMPONENTS[to] ?? 4;
  if (fromScalar && toN > 1) return [v[0], v[0], v[0], v[0]];
  const fromN = MTLX_COMPONENTS[from] ?? 4;
  if (fromN === 3 && toN === 4) return [v[0], v[1], v[2], 1];
  return v;
}

/** WGSL twin of `coerceValue`: `expr` is a vec4f. */
export function coerceWgsl(expr: string, from: string, to: string): string {
  if (from === to) return expr;
  const fromScalar = MTLX_SCALAR_TYPES.has(from);
  const toN = MTLX_COMPONENTS[to] ?? 4;
  if (fromScalar && toN > 1) return `vec4f((${expr}).x)`;
  const fromN = MTLX_COMPONENTS[from] ?? 4;
  if (fromN === 3 && toN === 4) return `vec4f((${expr}).xyz, 1.0)`;
  return expr;
}

/** Colour spaces whose files are sRGB-encoded (decoded to linear before filtering). `g22_rec709` is gamma 2.2, treated as sRGB (V263). */
const SRGB_SPACES = new Set(['srgb_texture', 'srgb', 'srgb_displayp3', 'g22_rec709']);

/** The lookup request of an image node from its string inputs (`file`, `colorspace`, address modes) and its result type (only colour / 3+ component types are decoded). */
export function imageRequest(strings: Record<string, string>, type: string): MtlxImageRequest {
  return { file: strings.file ?? '', srgb: SRGB_SPACES.has(strings.colorspace ?? '') && dim(type) >= 3, uaddress: strings.uaddressmode ?? 'periodic', vaddress: strings.vaddressmode ?? 'periodic' };
}

/** What an image node needs from its host. */
export interface MtlxImageRequest {
  file: string;
  /** The file is sRGB-encoded colour (decoded to linear before filtering). */
  srgb: boolean;
  /** MaterialX address modes: `periodic` (default), `clamp`, `mirror`, `constant`. */
  uaddress: string;
  vaddress: string;
}

export interface MtlxTsContext {
  uv: readonly [number, number];
  position: readonly [number, number, number];
  normal: readonly [number, number, number];
  tangent: readonly [number, number, number];
  /** Bilinear lookup (linear light for sRGB files): RGBA, or undefined when the file is missing (the node's `default` applies). */
  sample(request: MtlxImageRequest, u: number, v: number): MtlxValue | undefined;
}

export interface MtlxWgslContext {
  /** The scene texture slot of an image (the WGSL reads it as `ptMtlxImg(slot, uv, fp)`); -1 when the scene has no such texture (the node's default applies). */
  image(request: MtlxImageRequest): number;
  /** Asks for a helper function of the prelude (`mod`, `smoothstep` ...). */
  need(helper: string): void;
}

/** The inputs of a node: for each, its coerced value (TS) or WGSL expression (a vec4f). Strings (`space`, `uaddressmode` ...) come in `strings`. */
export interface NodeArgs<T> {
  v: Record<string, T>;
  strings: Record<string, string>;
  /** The node's result type. */
  type: string;
  /** Name of the requested output (`outx` of `separate3`); undefined for single-output nodes. */
  output: string | undefined;
}

type Default = number[] | string;
interface InSpec {
  /** `same` = the node's own type; otherwise a MaterialX type name (a string type is `string`). */
  t: string;
  d: Default;
}

export interface NodeImpl {
  inputs: Record<string, InSpec>;
  /** True when the result depends on the shading point (texcoord, position, normal, ...), even with constant inputs. */
  geometry?: boolean;
  ts(a: NodeArgs<MtlxValue>, ctx: MtlxTsContext): MtlxValue;
  wgsl(a: NodeArgs<string>, ctx: MtlxWgslContext): string;
}

const same = (d: Default = [0]): InSpec => ({ t: 'same', d });
const f = (d: number): InSpec => ({ t: 'float', d: [d] });
const lane = (n: number[]): MtlxValue => [n[0] ?? 0, n[1] ?? 0, n[2] ?? 0, n[3] ?? 0];
const dim = (type: string): number => MTLX_COMPONENTS[type] ?? 4;

const map1 = (a: MtlxValue, fn: (x: number) => number): MtlxValue => [fn(a[0]), fn(a[1]), fn(a[2]), fn(a[3])];
const map2 = (a: MtlxValue, b: MtlxValue, fn: (x: number, y: number) => number): MtlxValue => [fn(a[0], b[0]), fn(a[1], b[1]), fn(a[2], b[2]), fn(a[3], b[3])];
const map3 = (a: MtlxValue, b: MtlxValue, c: MtlxValue, fn: (x: number, y: number, z: number) => number): MtlxValue => [fn(a[0], b[0], c[0]), fn(a[1], b[1], c[1]), fn(a[2], b[2], c[2]), fn(a[3], b[3], c[3])];

const smooth = (x: number, lo: number, hi: number): number => {
  if (x <= lo) return 0;
  if (x >= hi) return 1;
  const t = Math.min(1, Math.max(0, (x - lo) / (hi - lo)));
  return t * t * (3 - 2 * t);
};
const modf = (x: number, y: number): number => x - y * Math.floor(x / y);
const clamp01 = (x: number): number => Math.min(1, Math.max(0, x));
/** MaterialX `mx_aastep(threshold, value)` without derivatives: a hard step (the filter width is 0 at a point sample). */
const step = (edge: number, x: number): number => (x >= edge ? 1 : 0);

/** A componentwise node of one or two operands. */
function lanewise1(name: string, fn: (x: number) => number, wgsl: (x: string) => string): [string, NodeImpl] {
  return [name, { inputs: { in: same() }, ts: (a) => map1(a.v.in!, fn), wgsl: (a) => wgsl(a.v.in!) }];
}
function lanewise2(name: string, d1: number, d2: number, fn: (x: number, y: number) => number, wgsl: (x: string, y: string) => string, needs?: string): [string, NodeImpl] {
  return [name, { inputs: { in1: same([d1]), in2: same([d2]) }, ts: (a) => map2(a.v.in1!, a.v.in2!, fn), wgsl: (a, ctx) => {
    if (needs) ctx.need(needs);
    return wgsl(a.v.in1!, a.v.in2!);
  } }];
}

const lengthOf = (v: MtlxValue, n: number): number => Math.sqrt(v.slice(0, n).reduce((s, x) => s + x * x, 0));
const mask = (n: number): string => (n === 1 ? 'vec4f(1.0, 0.0, 0.0, 0.0)' : n === 2 ? 'vec4f(1.0, 1.0, 0.0, 0.0)' : n === 3 ? 'vec4f(1.0, 1.0, 1.0, 0.0)' : 'vec4f(1.0)');

/** `place2d` (`operationorder` 0 = SRT, 1 = TRS), degrees; the formulas of stdlib_ng.mtlx `NG_place2d_vector2`. */
function place2d(uv: readonly [number, number], pivot: MtlxValue, scale: MtlxValue, rotate: number, offset: MtlxValue, order: number): [number, number] {
  const rad = (rotate * Math.PI) / 180;
  const sa = Math.sin(rad), ca = Math.cos(rad);
  const rot = (x: number, y: number): [number, number] => [ca * x + sa * y, -sa * x + ca * y];
  const sx = uv[0] - pivot[0], sy = uv[1] - pivot[1];
  if (order === 0) {
    const [rx, ry] = rot(sx / scale[0], sy / scale[1]);
    return [rx - offset[0] + pivot[0], ry - offset[1] + pivot[1]];
  }
  const [rx, ry] = rot(sx - offset[0], sy - offset[1]);
  return [rx / scale[0] + pivot[0], ry / scale[1] + pivot[1]];
}

/** The `image` family: texcoord -> uv transform -> lookup; `tiledimage` adds `uvtiling`, `uvoffset`, `realworld*` (the 1.38 nodegraph). */
function imageNode(tiled: boolean): NodeImpl {
  const inputs: Record<string, InSpec> = {
    file: { t: 'filename', d: '' },
    default: same([0]),
    texcoord: { t: 'vector2', d: 'uv' },
    uaddressmode: { t: 'string', d: 'periodic' },
    vaddressmode: { t: 'string', d: 'periodic' },
    filtertype: { t: 'string', d: 'linear' },
    ...(tiled ? { uvtiling: { t: 'vector2', d: [1, 1] }, uvoffset: { t: 'vector2', d: [0, 0] }, realworldimagesize: { t: 'vector2', d: [1, 1] }, realworldtilesize: { t: 'vector2', d: [1, 1] } } : {}),
  };
  const request = (a: NodeArgs<unknown>): MtlxImageRequest => imageRequest(a.strings, a.type);
  const uvOf = (a: NodeArgs<MtlxValue>): [number, number] => {
    const t = a.v.texcoord!;
    if (!tiled) return [t[0], t[1]];
    const tiling = a.v.uvtiling!, off = a.v.uvoffset!, size = a.v.realworldimagesize!, tile = a.v.realworldtilesize!;
    return [(((t[0] * tiling[0] - off[0]) / size[0]) * tile[0]), (((t[1] * tiling[1] - off[1]) / size[1]) * tile[1])];
  };
  return {
    inputs,
    ts: (a, ctx) => {
      const [u, v] = uvOf(a);
      const texel = ctx.sample(request(a), u, v);
      if (!texel) return a.v.default!;
      const n = dim(a.type);
      return n === 1 ? [texel[0], 0, 0, 0] : n === 2 ? [texel[0], texel[1], 0, 0] : n === 3 ? [texel[0], texel[1], texel[2], 0] : texel;
    },
    wgsl: (a, ctx) => {
      const k = ctx.image(request(a));
      if (k < 0) return a.v.default!;
      const t = a.v.texcoord!;
      const uv = tiled
        ? `(((${t}).xy * (${a.v.uvtiling}).xy - (${a.v.uvoffset}).xy) / (${a.v.realworldimagesize}).xy) * (${a.v.realworldtilesize}).xy`
        : `(${t}).xy`;
      const n = dim(a.type);
      const m = mask(n);
      return `ptMtlxImg(${k}u, ${uv}, fp) * ${m}`;
    },
  };
}

const CROSS_TS = (a: MtlxValue, b: MtlxValue): MtlxValue => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0], 0];

export const MTLX_NODES: Readonly<Record<string, NodeImpl>> = Object.fromEntries([
  lanewise2('add', 0, 0, (x, y) => x + y, (x, y) => `(${x} + ${y})`),
  lanewise2('subtract', 0, 0, (x, y) => x - y, (x, y) => `(${x} - ${y})`),
  lanewise2('multiply', 0, 1, (x, y) => x * y, (x, y) => `(${x} * ${y})`),
  lanewise2('divide', 0, 1, (x, y) => x / y, (x, y) => `(${x} / ${y})`),
  lanewise2('modulo', 0, 1, modf, (x, y) => `ptMtlxMod(${x}, ${y})`, 'mod'),
  lanewise2('power', 0, 1, (x, y) => Math.pow(x, y), (x, y) => `pow(${x}, ${y})`),
  lanewise2('min', 0, 0, Math.min, (x, y) => `min(${x}, ${y})`),
  lanewise2('max', 0, 0, Math.max, (x, y) => `max(${x}, ${y})`),
  lanewise1('absval', Math.abs, (x) => `abs(${x})`),
  lanewise1('floor', Math.floor, (x) => `floor(${x})`),
  lanewise1('fract', (x) => x - Math.floor(x), (x) => `fract(${x})`),
  lanewise1('round', (x) => Math.floor(x + 0.5), (x) => `floor(${x} + 0.5)`),
  lanewise1('ceil', Math.ceil, (x) => `ceil(${x})`),
  lanewise1('sign', Math.sign, (x) => `sign(${x})`),
  lanewise1('sin', Math.sin, (x) => `sin(${x})`),
  lanewise1('cos', Math.cos, (x) => `cos(${x})`),
  lanewise1('tan', Math.tan, (x) => `tan(${x})`),
  lanewise1('asin', Math.asin, (x) => `asin(${x})`),
  lanewise1('acos', Math.acos, (x) => `acos(${x})`),
  lanewise1('sqrt', Math.sqrt, (x) => `sqrt(${x})`),
  lanewise1('exp', Math.exp, (x) => `exp(${x})`),
  lanewise1('ln', Math.log, (x) => `log(${x})`),
  ['atan2', { inputs: { iny: same([0]), inx: same([1]) }, ts: (a) => map2(a.v.iny!, a.v.inx!, Math.atan2), wgsl: (a) => `atan2(${a.v.iny}, ${a.v.inx})` }],
  ['clamp', { inputs: { in: same([0]), low: same([0]), high: same([1]) }, ts: (a) => map3(a.v.in!, a.v.low!, a.v.high!, (x, lo, hi) => Math.min(Math.max(x, lo), hi)), wgsl: (a) => `clamp(${a.v.in}, ${a.v.low}, ${a.v.high})` }],
  ['invert', { inputs: { in: same([0]), amount: same([1]) }, ts: (a) => map2(a.v.amount!, a.v.in!, (m, x) => m - x), wgsl: (a) => `(${a.v.amount} - ${a.v.in})` }],
  ['remap', { inputs: { in: same([0]), inlow: same([0]), inhigh: same([1]), outlow: same([0]), outhigh: same([1]) }, ts: (a) => {
    const i = a.v;
    return [0, 1, 2, 3].map((k) => i.outlow![k]! + ((i.in![k]! - i.inlow![k]!) * (i.outhigh![k]! - i.outlow![k]!)) / (i.inhigh![k]! - i.inlow![k]!)) as MtlxValue;
  }, wgsl: (a) => `(${a.v.outlow} + (${a.v.in} - ${a.v.inlow}) * (${a.v.outhigh} - ${a.v.outlow}) / (${a.v.inhigh} - ${a.v.inlow}))` }],
  ['smoothstep', { inputs: { in: same([0]), low: same([0]), high: same([1]) }, ts: (a) => map3(a.v.in!, a.v.low!, a.v.high!, smooth), wgsl: (a, ctx) => {
    ctx.need('smoothstep');
    return `ptMtlxSmooth(${a.v.in}, ${a.v.low}, ${a.v.high})`;
  } }],
  ['mix', { inputs: { fg: same([0]), bg: same([0]), mix: same([0]) }, ts: (a) => map3(a.v.bg!, a.v.fg!, a.v.mix!, (b, g, m) => b * (1 - m) + g * m), wgsl: (a) => `mix(${a.v.bg}, ${a.v.fg}, ${a.v.mix})` }],
  ['constant', { inputs: { value: same([0]) }, ts: (a) => a.v.value!, wgsl: (a) => a.v.value! }],
  ['dot', { inputs: { in: same([0]) }, ts: (a) => a.v.in!, wgsl: (a) => a.v.in! }],
  ['normalize', { inputs: { in: same([0]) }, ts: (a) => {
    const n = dim(a.type);
    const l = lengthOf(a.v.in!, n);
    return lane(Array.from({ length: n }, (_, k) => a.v.in![k]! / l));
  }, wgsl: (a, ctx) => {
    ctx.need('normalize');
    return `ptMtlxNormalize(${a.v.in}, ${mask(dim(a.type))})`;
  } }],
  ['magnitude', { inputs: { in: { t: 'vector3', d: [0] } }, ts: (a) => {
    const l = lengthOf(a.v.in!, 3);
    return [l, l, l, l];
  }, wgsl: (a) => `vec4f(length((${a.v.in}).xyz))` }],
  ['dotproduct', { inputs: { in1: { t: 'vector3', d: [0] }, in2: { t: 'vector3', d: [0] } }, ts: (a) => {
    const d = a.v.in1![0] * a.v.in2![0] + a.v.in1![1] * a.v.in2![1] + a.v.in1![2] * a.v.in2![2];
    return [d, d, d, d];
  }, wgsl: (a) => `vec4f(dot((${a.v.in1}).xyz, (${a.v.in2}).xyz))` }],
  ['crossproduct', { inputs: { in1: { t: 'vector3', d: [0] }, in2: { t: 'vector3', d: [0] } }, ts: (a) => CROSS_TS(a.v.in1!, a.v.in2!), wgsl: (a) => `vec4f(cross((${a.v.in1}).xyz, (${a.v.in2}).xyz), 0.0)` }],
  ['ifgreater', { inputs: { value1: f(0), value2: f(0), in1: same([0]), in2: same([0]) }, ts: (a) => (a.v.value1![0] > a.v.value2![0] ? a.v.in1! : a.v.in2!), wgsl: (a) => `select(${a.v.in2}, ${a.v.in1}, (${a.v.value1}).x > (${a.v.value2}).x)` }],
  ['ifgreatereq', { inputs: { value1: f(0), value2: f(0), in1: same([0]), in2: same([0]) }, ts: (a) => (a.v.value1![0] >= a.v.value2![0] ? a.v.in1! : a.v.in2!), wgsl: (a) => `select(${a.v.in2}, ${a.v.in1}, (${a.v.value1}).x >= (${a.v.value2}).x)` }],
  ['ifequal', { inputs: { value1: f(0), value2: f(0), in1: same([0]), in2: same([0]) }, ts: (a) => (a.v.value1![0] === a.v.value2![0] ? a.v.in1! : a.v.in2!), wgsl: (a) => `select(${a.v.in2}, ${a.v.in1}, (${a.v.value1}).x == (${a.v.value2}).x)` }],
  ['convert', { inputs: { in: { t: 'any', d: [0] } }, ts: (a) => a.v.in!, wgsl: (a) => a.v.in! }],
  ['combine2', { inputs: { in1: f(0), in2: f(0) }, ts: (a) => [a.v.in1![0], a.v.in2![0], 0, 0], wgsl: (a) => `vec4f((${a.v.in1}).x, (${a.v.in2}).x, 0.0, 0.0)` }],
  ['combine3', { inputs: { in1: f(0), in2: f(0), in3: f(0) }, ts: (a) => [a.v.in1![0], a.v.in2![0], a.v.in3![0], 0], wgsl: (a) => `vec4f((${a.v.in1}).x, (${a.v.in2}).x, (${a.v.in3}).x, 0.0)` }],
  ['combine4', { inputs: { in1: f(0), in2: f(0), in3: f(0), in4: f(0) }, ts: (a) => [a.v.in1![0], a.v.in2![0], a.v.in3![0], a.v.in4![0]], wgsl: (a) => `vec4f((${a.v.in1}).x, (${a.v.in2}).x, (${a.v.in3}).x, (${a.v.in4}).x)` }],
  ['extract', { inputs: { in: { t: 'any', d: [0] }, index: { t: 'integer', d: [0] } }, ts: (a) => {
    const x = a.v.in![Math.min(3, Math.max(0, Math.round(a.v.index![0])))]!;
    return [x, x, x, x];
  }, wgsl: (a) => `vec4f((${a.v.in})[clamp(i32(round((${a.v.index}).x)), 0, 3)])` }],
  ['swizzle', { inputs: { in: { t: 'any', d: [0] }, channels: { t: 'string', d: 'x' } }, ts: (a) => {
    const out: number[] = [];
    for (const ch of (a.strings.channels ?? 'x')) {
      const k = 'xyzw'.indexOf(ch) >= 0 ? 'xyzw'.indexOf(ch) : 'rgba'.indexOf(ch);
      out.push(ch === '0' ? 0 : ch === '1' ? 1 : (a.v.in![Math.max(0, k)] ?? 0));
    }
    return lane(out);
  }, wgsl: (a) => {
    const parts: string[] = [];
    for (const ch of (a.strings.channels ?? 'x')) {
      const k = 'xyzw'.indexOf(ch) >= 0 ? 'xyzw'.indexOf(ch) : 'rgba'.indexOf(ch);
      parts.push(ch === '0' ? '0.0' : ch === '1' ? '1.0' : `(${a.v.in}).${'xyzw'[Math.max(0, k)]}`);
    }
    while (parts.length < 4) parts.push('0.0');
    return `vec4f(${parts.slice(0, 4).join(', ')})`;
  } }],
  ['separate2', separate(2)],
  ['separate3', separate(3)],
  ['separate4', separate(4)],
  ['texcoord', { inputs: { index: { t: 'integer', d: [0] } }, geometry: true, ts: (_a, ctx) => [ctx.uv[0], ctx.uv[1], 0, 0], wgsl: () => 'vec4f(uv.x, uv.y, 0.0, 0.0)' }],
  ['position', { inputs: { space: { t: 'string', d: 'object' } }, geometry: true, ts: (_a, ctx) => [ctx.position[0], ctx.position[1], ctx.position[2], 0], wgsl: () => 'vec4f(pos, 0.0)' }],
  ['normal', { inputs: { space: { t: 'string', d: 'object' } }, geometry: true, ts: (_a, ctx) => [ctx.normal[0], ctx.normal[1], ctx.normal[2], 0], wgsl: () => 'vec4f(nrm, 0.0)' }],
  ['tangent', { inputs: { space: { t: 'string', d: 'object' } }, geometry: true, ts: (_a, ctx) => [ctx.tangent[0], ctx.tangent[1], ctx.tangent[2], 0], wgsl: () => 'vec4f(tg, 0.0)' }],
  ['place2d', { inputs: { texcoord: { t: 'vector2', d: 'uv' }, pivot: { t: 'vector2', d: [0, 0] }, scale: { t: 'vector2', d: [1, 1] }, rotate: f(0), offset: { t: 'vector2', d: [0, 0] }, operationorder: { t: 'integer', d: [0] } }, ts: (a) => {
    const [x, y] = place2d([a.v.texcoord![0], a.v.texcoord![1]], a.v.pivot!, a.v.scale!, a.v.rotate![0], a.v.offset!, a.v.operationorder![0]);
    return [x, y, 0, 0];
  }, wgsl: (a, ctx) => {
    ctx.need('place2d');
    return `ptMtlxPlace2d(${a.v.texcoord}, ${a.v.pivot}, ${a.v.scale}, ${a.v.rotate}, ${a.v.offset}, ${a.v.operationorder})`;
  } }],
  ['rotate2d', { inputs: { in: { t: 'vector2', d: [0, 0] }, amount: f(0) }, ts: (a) => {
    const rad = (a.v.amount![0] * Math.PI) / 180;
    const sa = Math.sin(rad), ca = Math.cos(rad);
    return [ca * a.v.in![0] + sa * a.v.in![1], -sa * a.v.in![0] + ca * a.v.in![1], 0, 0];
  }, wgsl: (a) => {
    const r = `(${a.v.amount}).x * 0.017453292519943295`;
    return `vec4f(cos(${r}) * (${a.v.in}).x + sin(${r}) * (${a.v.in}).y, -sin(${r}) * (${a.v.in}).x + cos(${r}) * (${a.v.in}).y, 0.0, 0.0)`;
  } }],
  ['image', imageNode(false)],
  ['tiledimage', imageNode(true)],
  ['ramplr', { inputs: { valuel: same([0]), valuer: same([1]), texcoord: { t: 'vector2', d: 'uv' } }, ts: (a) => map2(a.v.valuel!, a.v.valuer!, (l, r) => l * (1 - clamp01(a.v.texcoord![0])) + r * clamp01(a.v.texcoord![0])), wgsl: (a) => `mix(${a.v.valuel}, ${a.v.valuer}, clamp((${a.v.texcoord}).x, 0.0, 1.0))` }],
  ['ramptb', { inputs: { valuet: same([0]), valueb: same([1]), texcoord: { t: 'vector2', d: 'uv' } }, ts: (a) => map2(a.v.valuet!, a.v.valueb!, (t, b) => t * (1 - clamp01(a.v.texcoord![1])) + b * clamp01(a.v.texcoord![1])), wgsl: (a) => `mix(${a.v.valuet}, ${a.v.valueb}, clamp((${a.v.texcoord}).y, 0.0, 1.0))` }],
  ['splitlr', { inputs: { valuel: same([0]), valuer: same([1]), center: f(0.5), texcoord: { t: 'vector2', d: 'uv' } }, ts: (a) => map2(a.v.valuel!, a.v.valuer!, (l, r) => (step(a.v.center![0], a.v.texcoord![0]) ? r : l)), wgsl: (a) => `select(${a.v.valuel}, ${a.v.valuer}, (${a.v.texcoord}).x >= (${a.v.center}).x)` }],
  ['splittb', { inputs: { valuet: same([0]), valueb: same([1]), center: f(0.5), texcoord: { t: 'vector2', d: 'uv' } }, ts: (a) => map2(a.v.valuet!, a.v.valueb!, (t, b) => (step(a.v.center![0], a.v.texcoord![1]) ? b : t)), wgsl: (a) => `select(${a.v.valuet}, ${a.v.valueb}, (${a.v.texcoord}).y >= (${a.v.center}).x)` }],
  ['luminance', { inputs: { in: { t: 'color3', d: [0, 0, 0] }, lumacoeffs: { t: 'color3', d: [0.2722287168, 0.6740817658, 0.0536895174] } }, ts: (a) => {
    const l = a.v.in![0] * a.v.lumacoeffs![0] + a.v.in![1] * a.v.lumacoeffs![1] + a.v.in![2] * a.v.lumacoeffs![2];
    return [l, l, l, 0];
  }, wgsl: (a) => `vec4f(vec3f(dot((${a.v.in}).xyz, (${a.v.lumacoeffs}).xyz)), 0.0)` }],
  ['normalmap', { inputs: { in: { t: 'vector3', d: [0.5, 0.5, 1] }, scale: f(1), space: { t: 'string', d: 'tangent' }, normal: { t: 'vector3', d: 'normal' }, tangent: { t: 'vector3', d: 'tangent' } }, geometry: true, ts: (a) => {
    // mx_normalmap.glsl: decode, scale xy, rotate by the (T, B, N) frame (tangent space) or take it as given (object space).
    const w = a.v.in!;
    let m: [number, number, number] = w[0] === 0 && w[1] === 0 && w[2] === 0 ? [0, 0, 1] : [w[0] * 2 - 1, w[1] * 2 - 1, w[2] * 2 - 1];
    if ((a.strings.space ?? 'tangent') === 'tangent') {
      const n = a.v.normal!, t = a.v.tangent!;
      const b = CROSS_TS(n, t);
      const bl = Math.hypot(b[0], b[1], b[2]);
      const s = a.v.scale![0];
      m = [m[0] * s, m[1] * s, m[2]];
      m = [t[0] * m[0] + (b[0] / bl) * m[1] + n[0] * m[2], t[1] * m[0] + (b[1] / bl) * m[1] + n[1] * m[2], t[2] * m[0] + (b[2] / bl) * m[1] + n[2] * m[2]];
    }
    const l = Math.hypot(m[0], m[1], m[2]);
    return [m[0] / l, m[1] / l, m[2] / l, 0];
  }, wgsl: (a, ctx) => {
    ctx.need('normalmap');
    return `ptMtlxNormalmap(${a.v.in}, ${a.v.scale}, ${a.strings.space === 'object' ? '1.0' : '0.0'}, ${a.v.normal}, ${a.v.tangent})`;
  } }],
  ...(['transformpoint', 'transformvector', 'transformnormal'] as const).map((name): [string, NodeImpl] => [name, { inputs: { in: { t: 'vector3', d: [0] }, fromspace: { t: 'string', d: 'object' }, tospace: { t: 'string', d: 'world' } }, geometry: true, ts: (a) => {
    // Instances are placed with the identity object-to-world transform in the scenes this feeds (V260), so object, model and world coincide.
    const known = (s: string | undefined) => s === undefined || s === 'object' || s === 'model' || s === 'world';
    if (!known(a.strings.fromspace) || !known(a.strings.tospace)) throw new MtlxUnsupportedError(name, `${name}: space '${a.strings.fromspace}' -> '${a.strings.tospace}' is not supported (only object / model / world)`);
    return a.v.in!;
  }, wgsl: (a) => a.v.in! }]),
]) as Record<string, NodeImpl>;

function separate(n: number): NodeImpl {
  // 1.38 names the outputs outx / outr, 1.39 files use x / r: both read.
  const names = [['outx', 'x', 'outr', 'r'], ['outy', 'y', 'outg', 'g'], ['outz', 'z', 'outb', 'b'], ['outw', 'w', 'outa', 'a']];
  const index = (o: string | undefined): number => Math.max(0, names.findIndex((row) => row.includes(o ?? 'outx')));
  return {
    inputs: { in: { t: n === 2 ? 'vector2' : n === 3 ? 'vector3' : 'vector4', d: [0] } },
    ts: (a) => {
      const x = a.v.in![index(a.output)]!;
      return [x, x, x, x];
    },
    wgsl: (a) => `vec4f((${a.v.in}).${'xyzw'[index(a.output)]})`,
  };
}

/** Resolves an input default to a lane value for a node of type `type`. */
export function defaultOf(spec: InSpec, type: string): MtlxValue | string {
  if (typeof spec.d === 'string') return spec.d;
  void type;
  return lane(spec.d);
}

/** The input type a spec resolves to for a node of type `type`. */
export function specType(spec: InSpec, type: string): string {
  return spec.t === 'same' ? type : spec.t;
}

export { parseMtlxNumeric };

/** WGSL helper functions the node expressions call, emitted only when asked for (`MtlxWgslContext.need`). */
export const MTLX_WGSL_HELPERS: Readonly<Record<string, string>> = {
  image: `fn ptMtlxImg(slot: u32, uv: vec2f, fp: f32) -> vec4f {
    let r = rtTexRecord(slot);
    return rtTexSampleRecord(r, uv, rtTexLodOf(r, fp));
}
`,
  mod: `fn ptMtlxMod(x: vec4f, y: vec4f) -> vec4f {
    return x - y * floor(x / y);
}
`,
  smoothstep: `fn ptMtlxSmooth(x: vec4f, lo: vec4f, hi: vec4f) -> vec4f {
    let t = clamp((x - lo) / (hi - lo), vec4f(0.0), vec4f(1.0));
    let s = t * t * (3.0 - 2.0 * t);
    return select(select(s, vec4f(1.0), x >= hi), vec4f(0.0), x <= lo);
}
`,
  normalize: `fn ptMtlxNormalize(v: vec4f, m: vec4f) -> vec4f {
    let a = v * m;
    return a * inverseSqrt(dot(a, a));
}
`,
  place2d: `fn ptMtlxPlace2d(uvIn: vec4f, pivot: vec4f, scale: vec4f, rotate: vec4f, offset: vec4f, order: vec4f) -> vec4f {
    let r = rotate.x * 0.017453292519943295;
    let sa = sin(r);
    let ca = cos(r);
    let s = uvIn.xy - pivot.xy;
    if (order.x < 0.5) {
        let q = s / scale.xy;
        let rq = vec2f(ca * q.x + sa * q.y, -sa * q.x + ca * q.y);
        return vec4f(rq - offset.xy + pivot.xy, 0.0, 0.0);
    }
    let q = s - offset.xy;
    let rq = vec2f(ca * q.x + sa * q.y, -sa * q.x + ca * q.y);
    return vec4f(rq / scale.xy + pivot.xy, 0.0, 0.0);
}
`,
  normalmap: `fn ptMtlxNormalmap(v: vec4f, scale: vec4f, objectSpace: f32, n: vec4f, t: vec4f) -> vec4f {
    var m = v.xyz * 2.0 - 1.0;
    if (v.x == 0.0 && v.y == 0.0 && v.z == 0.0) {
        m = vec3f(0.0, 0.0, 1.0);
    }
    if (objectSpace < 0.5) {
        let b = normalize(cross(n.xyz, t.xyz));
        m = t.xyz * (m.x * scale.x) + b * (m.y * scale.x) + n.xyz * m.z;
    }
    return vec4f(normalize(m), 0.0);
}
`,
};
