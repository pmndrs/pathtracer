/**
 * Participating media for the native exporters (P5): pbrt-v4 `MakeNamedMedium` (homogeneous, uniformgrid) and Mitsuba 3 `<medium>` (homogeneous,
 * heterogeneous with a `gridvolume`). See docs/internal/design/volumes.md for the conventions and DIVERGES V245-V247 for what differs.
 *
 *  - pbrt `uniformgrid`: `density` samples at (i + 0.5) / n across [p0, p1] in the medium frame (the CTM at `MakeNamedMedium`), trilinear, zero outside the
 *    grid; `scale` multiplies sigma_a and sigma_s. (pbrt's `nanovdb` medium needs a NanoVDB file; the dense grid is the same density without the converter.)
 *  - Mitsuba `heterogeneous`: `sigma_t` = `gridvolume` (.vol file, unit cube mapped by `to_world`, cell-centred bilinear lookup, `clamp` outside) times
 *    `scale`, `albedo` = constant volume. A Mitsuba grid medium is a single-channel sigma_t (grey), so the exporter throws for a chromatic one. Scenes keep a
 *    zero border of at least one voxel so pbrt's zero outside and Mitsuba's clamp agree.
 */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import type { MediumDesc, Mat4, Rgb } from '../scenes/schema.ts';

/** The density values of a grid medium: inline, or read from `densityFile` under `assetsDir` (raw little-endian float32). */
export function mediumDensity(m: Extract<MediumDesc, { type: 'grid' }>, assetsDir: string): ArrayLike<number> {
  if (m.density) return m.density;
  if (!m.densityFile) throw new RangeError('grid medium has neither density nor densityFile');
  const buf = readFileSync(join(assetsDir, m.densityFile));
  const [nx, ny, nz] = m.resolution;
  if (buf.byteLength !== nx * ny * nz * 4) throw new RangeError(`${m.densityFile}: ${buf.byteLength} bytes for a ${nx} x ${ny} x ${nz} float32 grid (run the generator)`);
  return new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
}

const num = (v: number): string => (Number.isInteger(v) ? `${v}.0` : String(Number(v.toPrecision(9))));
const vec = (v: readonly number[]): string => v.map(num).join(' ');

/** pbrt `MakeNamedMedium` lines (an `AttributeBegin` block sets the medium frame for a grid). */
export function pbrtMedium(name: string, m: MediumDesc, transform: Mat4 | undefined, assetsDir: string): string[] {
  if (m.phase) throw new Error(`pbrt: medium '${name}' has a ${m.phase.type} phase function; pbrt-v4 only has one HG lobe (P5.v14 references: Mitsuba blendphase for hg2, Cycles for Draine / Mie)`);
  if (m.type === 'height') throw new Error(`pbrt: medium '${name}' is a height (exponential) medium; pbrt-v4 has no exponential medium (homogeneous, uniformgrid, rgbgrid, cloud, nanovdb): no reference (V4.5)`);
  if (m.Le && m.type === 'grid') throw new Error(`pbrt: grid medium '${name}' has Le; pbrt-v4's uniformgrid takes its emission from a separate "Lescale" grid, not the density: no export (V4.6)`);
  const le = m.Le ? ` "rgb Le" [ ${vec(m.Le)} ]` : '';
  const common = `"rgb sigma_a" [ ${vec(m.sigmaA)} ] "rgb sigma_s" [ ${vec(m.sigmaS)} ] "float scale" ${num(m.scale ?? 1)} "float g" ${num(m.g ?? 0)}${le}`;
  if (m.type === 'homogeneous') return [`MakeNamedMedium "${name}" "string type" "homogeneous" ${common}`];
  const [nx, ny, nz] = m.resolution;
  const lines = ['AttributeBegin'];
  if (transform) lines.push(`  ConcatTransform [ ${vec(transform)} ]`);
  lines.push(`  MakeNamedMedium "${name}" "string type" "uniformgrid" ${common} "integer nx" ${nx} "integer ny" ${ny} "integer nz" ${nz} "point3 p0" [ ${vec(m.bounds[0])} ] "point3 p1" [ ${vec(m.bounds[1])} ] "float density" [ ${Array.from(mediumDensity(m, assetsDir), (v) => num(v)).join(' ')} ]`);
  lines.push('AttributeEnd');
  return lines;
}

/** Mitsuba `.vol` file (version 3, float32, one channel, x fastest, then y, then z) with the bounding box `bounds`. */
export function mitsubaVol(resolution: readonly [number, number, number], density: ArrayLike<number>, bounds: readonly [Rgb, Rgb]): Buffer {
  const [nx, ny, nz] = resolution;
  const header = Buffer.alloc(48);
  header.write('VOL', 0, 'ascii');
  header.writeUInt8(3, 3);
  header.writeInt32LE(1, 4); // type: float32
  header.writeInt32LE(nx, 8);
  header.writeInt32LE(ny, 12);
  header.writeInt32LE(nz, 16);
  header.writeInt32LE(1, 20); // channels
  [...bounds[0], ...bounds[1]].forEach((v, i) => header.writeFloatLE(v, 24 + i * 4));
  const data = Buffer.alloc(nx * ny * nz * 4);
  for (let i = 0; i < nx * ny * nz; i++) data.writeFloatLE(density[i] as number, i * 4);
  return Buffer.concat([header, data]);
}

/** A Mitsuba `<medium id="name">` element (and the `.vol` file it needs, if any). */
export function mitsubaMedium(name: string, m: MediumDesc, transform: Mat4 | undefined, assetsDir: string): { xml: string; file?: { path: string; data: Buffer } } {
  if (m.type === 'height') throw new Error(`mitsuba: medium '${name}' is a height (exponential) medium; Mitsuba 3 has no exponential medium (V4.5)`);
  if (m.Le && (m.Le[0] > 0 || m.Le[1] > 0 || m.Le[2] > 0)) throw new Error(`mitsuba: medium '${name}' has Le; Mitsuba 3's volpath has no emissive medium (V4.6)`);
  const scale = m.scale ?? 1;
  const sigmaT = [0, 1, 2].map((i) => (m.sigmaA[i] as number) + (m.sigmaS[i] as number));
  const albedo = sigmaT.map((t, i) => (t > 0 ? (m.sigmaS[i] as number) / t : 0));
  if (m.phase && m.phase.type !== 'hg2') throw new Error(`mitsuba: medium '${name}' has a ${m.phase.type} phase function; Mitsuba 3 has no Draine lobe (use Cycles)`);
  // blendphase `weight` is the weight of the SECOND child
  const phase = m.phase
    ? `<phase type="blendphase"><float name="weight" value="${num(m.phase.w2)}"/><phase type="hg"><float name="g" value="${num(m.phase.g1)}"/></phase><phase type="hg"><float name="g" value="${num(m.phase.g2)}"/></phase></phase>`
    : `<phase type="hg"><float name="g" value="${num(m.g ?? 0)}"/></phase>`;
  const sMax = Math.max(...sigmaT);
  if (m.type === 'homogeneous') {
    // `rgb` on a spectrum-typed medium parameter is a reflectance in [0, 1]: sigma_t = scale * rgb with rgb = sigma / max(sigma)
    return { xml: `  <medium type="homogeneous" id="${name}"><rgb name="sigma_t" value="${vec(sigmaT.map((v) => Number((v / sMax).toPrecision(9))))}"/><rgb name="albedo" value="${vec(albedo.map((v) => Number(v.toPrecision(9))))}"/><float name="scale" value="${num(sMax * scale)}"/>${phase}</medium>` };
  }
  if (Math.abs(sigmaT[0]! - sigmaT[1]!) > 1e-9 * sMax || Math.abs(sigmaT[0]! - sigmaT[2]!) > 1e-9 * sMax) throw new Error(`mitsuba: grid medium '${name}' has a chromatic sigma_t; Mitsuba's heterogeneous medium takes a grey (single-channel) extinction volume here`);
  const [p0, p1] = m.bounds;
  // unit cube -> medium box -> world (the mesh transform)
  const sc = [0, 1, 2].map((a) => (p1[a] as number) - (p0[a] as number));
  const box = [sc[0], 0, 0, 0, 0, sc[1], 0, 0, 0, 0, sc[2], 0, p0[0], p0[1], p0[2], 1] as number[];
  const toWorld = transform ? mulCol(transform, box) : box;
  const matrix = [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15].map((i) => num(toWorld[i] as number)).join(' ');
  const file = `${name}.vol`;
  const xml = `  <medium type="heterogeneous" id="${name}"><volume name="sigma_t" type="gridvolume"><string name="filename" value="${file}"/><string name="filter_type" value="trilinear"/><string name="wrap_mode" value="clamp"/><transform name="to_world"><matrix value="${matrix}"/></transform></volume><volume name="albedo" type="constvolume"><rgb name="value" value="${vec(albedo.map((v) => Number(v.toPrecision(9))))}"/></volume><float name="scale" value="${num(sMax * scale)}"/>${phase}</medium>`;
  return { xml, file: { path: file, data: mitsubaVol(m.resolution, mediumDensity(m, assetsDir), [[0, 0, 0], [1, 1, 1]]) } };
}

/** a * b for column-major 4x4 matrices. */
function mulCol(a: readonly number[], b: readonly number[]): number[] {
  const out = new Array<number>(16).fill(0);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) for (let k = 0; k < 4; k++) out[c * 4 + r]! += (a[k * 4 + r] as number) * (b[c * 4 + k] as number);
  return out;
}
