/**
 * Runtime side of the blue-noise assets in `packages/raytracer/assets/bluenoise/` (made by
 * `tools/bluenoise/generate.ts`, `pnpm bluenoise:generate`; quality numbers in `tools/bluenoise/REPORT.md`).
 *
 * In-memory layout (both loaders): Float32Array, channel-interleaved, z-major then row-major (row 0 = top):
 * `index = ((z * height + y) * width + x) * channels + c`. Values are in [0, 1].
 *
 * Two file formats:
 *  - `<name>.png`, committed: 16-bit grey (scalar) or RGB with B = 0 (vec2); volumes are tiled into an atlas,
 *    `atlasColumns` slices per row, slice z at column z % atlasColumns, row floor(z / atlasColumns).
 *    Decoded here with `DecompressionStream` (browsers, Node >= 18), never through a canvas (8-bit).
 *  - `<name>.f32`, raw little-endian float32 in the in-memory layout; git-ignored, rebuilt by the generator.
 * No dependencies.
 */
export interface BlueNoiseMeta {
  name: string;
  /**
   * `vc2d`: classic void-and-cluster, spatial only (vec2 = two independent channels).
   * `st`: spatiotemporal (3D void-and-cluster, separate spatial / temporal sigma; vec2 = two independent channels).
   * `gf`: vec2 only, both channels optimised jointly (Georgiev & Fajardo 2016 pixel swaps).
   */
  kind: 'vc2d' | 'st' | 'gf';
  width: number;
  height: number;
  /** Temporal slices (1 for `vc2d` and `gf`). */
  depth: number;
  /** 1 (scalar) or 2 (vec2). */
  channels: 1 | 2;
  /** File names relative to the manifest. */
  f32: string;
  png: string;
  atlasColumns: number;
  sigmaSpatial: number;
  sigmaTemporal: number | null;
  seed: number;
}

export interface BlueNoiseTexture {
  meta: BlueNoiseMeta;
  data: Float32Array;
}

/** Wrap raw float32 bytes (any alignment, little-endian) as a Float32Array and check the length against the meta. */
export function parseBlueNoise(bytes: ArrayBuffer | ArrayBufferView, meta: BlueNoiseMeta): BlueNoiseTexture {
  const view = ArrayBuffer.isView(bytes)
    ? new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength)
    : new Uint8Array(bytes);
  const count = meta.width * meta.height * meta.depth * meta.channels;
  if (view.byteLength !== count * 4) {
    throw new Error(`blue noise ${meta.name}: expected ${count * 4} bytes, got ${view.byteLength}`);
  }
  const data = new Float32Array(count);
  new Uint8Array(data.buffer).set(view); // copies, so unaligned input is fine (hosts are little-endian)
  return { meta, data };
}

async function inflate(zlib: Uint8Array): Promise<Uint8Array> {
  const stream = new Blob([zlib as BlobPart]).stream().pipeThrough(new DecompressionStream('deflate'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/**
 * Decode one of our 16-bit PNGs (grey or RGB, 8 or 16 bit, any filter, not interlaced) into the in-memory layout.
 * Values are sample / 65535 (sample / 255 for 8-bit).
 */
export async function decodeBlueNoisePng(bytes: ArrayBuffer | ArrayBufferView, meta: BlueNoiseMeta): Promise<BlueNoiseTexture> {
  const buf = ArrayBuffer.isView(bytes) ? new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength) : new Uint8Array(bytes);
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const sig = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
  for (let i = 0; i < 8; i++) if (buf[i] !== sig[i]) throw new Error(`blue noise ${meta.name}: not a PNG`);
  let pos = 8;
  let width = 0;
  let height = 0;
  let bitDepth = 0;
  let colorType = -1;
  const idat: Uint8Array[] = [];
  while (pos + 8 <= buf.length) {
    const len = dv.getUint32(pos);
    const type = String.fromCharCode(buf[pos + 4]!, buf[pos + 5]!, buf[pos + 6]!, buf[pos + 7]!);
    const body = buf.subarray(pos + 8, pos + 8 + len);
    if (type === 'IHDR') {
      width = dv.getUint32(pos + 8);
      height = dv.getUint32(pos + 12);
      bitDepth = body[8]!;
      colorType = body[9]!;
      if (body[12] !== 0) throw new Error(`blue noise ${meta.name}: interlaced PNG`);
    } else if (type === 'IDAT') idat.push(body);
    else if (type === 'IEND') break;
    pos += 12 + len;
  }
  if ((bitDepth !== 8 && bitDepth !== 16) || (colorType !== 0 && colorType !== 2)) {
    throw new Error(`blue noise ${meta.name}: unsupported PNG (bit depth ${bitDepth}, colour type ${colorType})`);
  }
  const cols = meta.atlasColumns;
  const rows = Math.ceil(meta.depth / cols);
  if (width !== meta.width * cols || height !== meta.height * rows) {
    throw new Error(`blue noise ${meta.name}: PNG is ${width}x${height}, manifest implies ${meta.width * cols}x${meta.height * rows}`);
  }
  const pngChannels = colorType === 0 ? 1 : 3;
  if (pngChannels < meta.channels) throw new Error(`blue noise ${meta.name}: PNG has ${pngChannels} channels`);
  let total = 0;
  for (const c of idat) total += c.length;
  const zlib = new Uint8Array(total);
  let o = 0;
  for (const c of idat) {
    zlib.set(c, o);
    o += c.length;
  }
  const raw = await inflate(zlib);
  const bps = bitDepth / 8;
  const bpp = pngChannels * bps;
  const rowBytes = width * bpp;
  if (raw.length < (rowBytes + 1) * height) throw new Error(`blue noise ${meta.name}: truncated PNG data`);
  const scale = 1 / (bitDepth === 16 ? 65535 : 255);
  const prev = new Uint8Array(rowBytes);
  const cur = new Uint8Array(rowBytes);
  const out = new Float32Array(meta.width * meta.height * meta.depth * meta.channels);
  for (let y = 0; y < height; y++) {
    const base = y * (rowBytes + 1);
    const filter = raw[base]!;
    for (let i = 0; i < rowBytes; i++) {
      const x = raw[base + 1 + i]!;
      const a = i >= bpp ? cur[i - bpp]! : 0;
      const b = prev[i]!;
      const c = i >= bpp ? prev[i - bpp]! : 0;
      let v: number;
      if (filter === 0) v = x;
      else if (filter === 1) v = x + a;
      else if (filter === 2) v = x + b;
      else if (filter === 3) v = x + ((a + b) >> 1);
      else {
        const p = a + b - c;
        const pa = Math.abs(p - a);
        const pb = Math.abs(p - b);
        const pc = Math.abs(p - c);
        v = x + (pa <= pb && pa <= pc ? a : pb <= pc ? b : c);
      }
      cur[i] = v & 0xff;
    }
    const tileRow = Math.floor(y / meta.height);
    const ty = y - tileRow * meta.height;
    for (let px = 0; px < width; px++) {
      const tileCol = Math.floor(px / meta.width);
      const z = tileRow * cols + tileCol;
      if (z >= meta.depth) continue;
      const tx = px - tileCol * meta.width;
      const dst = ((z * meta.height + ty) * meta.width + tx) * meta.channels;
      for (let ch = 0; ch < meta.channels; ch++) {
        const s = px * bpp + ch * bps;
        out[dst + ch] = (bps === 2 ? (cur[s]! << 8) | cur[s + 1]! : cur[s]!) * scale;
      }
    }
    prev.set(cur);
  }
  return { meta, data: out };
}

/**
 * Fetch the manifest next to `baseUrl` and one texture by name. `format` picks the committed 16-bit PNG
 * (default) or the raw float32 file (present only after `pnpm bluenoise:generate`).
 */
export async function fetchBlueNoise(baseUrl: string, name: string, format: 'png' | 'f32' = 'png'): Promise<BlueNoiseTexture> {
  const base = baseUrl.endsWith('/') ? baseUrl : `${baseUrl}/`;
  const manifest = (await (await fetch(`${base}manifest.json`)).json()) as { textures: BlueNoiseMeta[] };
  const meta = manifest.textures.find((t) => t.name === name);
  if (!meta) throw new Error(`blue noise texture not in manifest: ${name}`);
  const file = format === 'png' ? meta.png : meta.f32;
  const res = await fetch(`${base}${file}`);
  if (!res.ok) throw new Error(`blue noise fetch failed: ${res.status} ${file}`);
  const bytes = await res.arrayBuffer();
  return format === 'png' ? decodeBlueNoisePng(bytes, meta) : parseBlueNoise(bytes, meta);
}

/** Texel read with toroidal wrap in all three axes: the textures tile in space and time. */
export function blueNoiseAt(tex: BlueNoiseTexture, x: number, y: number, z: number, channel = 0): number {
  const { width, height, depth, channels } = tex.meta;
  const xi = ((x % width) + width) % width;
  const yi = ((y % height) + height) % height;
  const zi = ((z % depth) + depth) % depth;
  return tex.data[((zi * height + yi) * width + xi) * channels + channel]!;
}
