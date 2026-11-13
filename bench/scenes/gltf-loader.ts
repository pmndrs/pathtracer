/**
 * Minimal glTF 2.0 / GLB triangle reader for the bench (Node only). Reads POSITION + indices + node transforms and
 * nothing else: no materials, textures, alpha modes, skins, morph targets or sparse accessors. Triangles only
 * (primitive mode 4); other modes throw rather than silently dropping geometry.
 *
 * Output is what every renderer receives for a `file` mesh: world-space (node transforms baked), **un-welded**
 * (3 vertices per triangle) with the **geometric** normal as vertex normal, so pbrt/Mitsuba/web renderers cannot
 * disagree through shading-normal handling or smoothing groups. Zero-area triangles are dropped (counted in `stats`).
 * Mirrored node transforms flip winding so geometric normals stay consistent with the file's front faces.
 */
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { createHash } from 'node:crypto';
import { compose, cross3, determinant3, multiply, transformPoint } from './geometry.ts';
import type { TriangleMesh } from './geometry.ts';
import type { Mat4, Vec3 } from './schema.ts';

interface Accessor { bufferView?: number; byteOffset?: number; componentType: number; count: number; type: string; sparse?: unknown; normalized?: boolean }
interface BufferView { buffer: number; byteOffset?: number; byteLength: number; byteStride?: number }
interface Primitive { attributes: Record<string, number>; indices?: number; mode?: number }
interface GltfNode { mesh?: number; children?: number[]; matrix?: number[]; translation?: number[]; rotation?: number[]; scale?: number[] }
interface GltfJson {
  accessors?: Accessor[]; bufferViews?: BufferView[]; buffers?: { uri?: string; byteLength: number }[];
  meshes?: { primitives: Primitive[] }[]; nodes?: GltfNode[]; scenes?: { nodes?: number[] }[]; scene?: number;
}

export interface GltfLoadStats {
  /** Triangles in the file (before dropping degenerates). */
  fileTriangles: number;
  degenerateDropped: number;
  /** Triangles in the returned mesh. */
  triangles: number;
  primitives: number;
  bounds: { min: [number, number, number]; max: [number, number, number] };
}

export interface GltfLoadResult { mesh: TriangleMesh; stats: GltfLoadStats }

const COMPONENT: Record<number, { bytes: number; read: (dv: DataView, o: number) => number }> = {
  5120: { bytes: 1, read: (dv, o) => dv.getInt8(o) },
  5121: { bytes: 1, read: (dv, o) => dv.getUint8(o) },
  5122: { bytes: 2, read: (dv, o) => dv.getInt16(o, true) },
  5123: { bytes: 2, read: (dv, o) => dv.getUint16(o, true) },
  5125: { bytes: 4, read: (dv, o) => dv.getUint32(o, true) },
  5126: { bytes: 4, read: (dv, o) => dv.getFloat32(o, true) },
};
const NCOMP: Record<string, number> = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 };

/** Split a GLB container into its JSON and (optional) BIN chunks. */
export function parseGlbContainer(buf: Buffer): { json: GltfJson; bin?: Buffer } {
  if (buf.length < 20 || buf.readUInt32LE(0) !== 0x46546c67) throw new Error('not a GLB file (bad magic)');
  if (buf.readUInt32LE(4) !== 2) throw new Error(`unsupported GLB version ${buf.readUInt32LE(4)}`);
  let off = 12;
  let json: GltfJson | undefined;
  let bin: Buffer | undefined;
  while (off + 8 <= Math.min(buf.length, buf.readUInt32LE(8))) {
    const len = buf.readUInt32LE(off), type = buf.readUInt32LE(off + 4);
    const data = buf.subarray(off + 8, off + 8 + len);
    if (type === 0x4e4f534a) json = JSON.parse(data.toString('utf8')) as GltfJson;
    else if (type === 0x004e4942 && !bin) bin = data;
    off += 8 + len;
  }
  if (!json) throw new Error('GLB has no JSON chunk');
  return { json, bin };
}

function localMatrix(n: GltfNode): Mat4 {
  if (n.matrix) return n.matrix;
  const t = n.translation ?? [0, 0, 0], q = n.rotation ?? [0, 0, 0, 1], s = n.scale ?? [1, 1, 1];
  const [x, y, z, w] = q as [number, number, number, number];
  const r = [
    1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w), 0,
    2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w), 0,
    2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y), 0,
    0, 0, 0, 1,
  ];
  const trs = r.slice();
  for (let c = 0; c < 3; c++) for (let k = 0; k < 3; k++) trs[c * 4 + k]! *= s[c]!;
  trs[12] = t[0]!; trs[13] = t[1]!; trs[14] = t[2]!;
  return trs;
}

/** Parse already-split glTF content. `buffers[i]` is the bytes of `json.buffers[i]`. */
export function triangulateGltf(json: GltfJson, buffers: Buffer[]): GltfLoadResult {
  const accessors = json.accessors ?? [], views = json.bufferViews ?? [], meshes = json.meshes ?? [], nodes = json.nodes ?? [];

  function readAccessor(index: number): number[] {
    const a = accessors[index];
    if (!a) throw new Error(`accessor ${index} missing`);
    if (a.sparse) throw new Error('sparse accessors are not supported');
    const comp = COMPONENT[a.componentType];
    if (!comp) throw new Error(`unsupported componentType ${a.componentType}`);
    const n = NCOMP[a.type];
    if (!n) throw new Error(`unsupported accessor type ${a.type}`);
    const out = new Array<number>(a.count * n).fill(0);
    if (a.bufferView === undefined) return out; // spec: all zeros
    const v = views[a.bufferView]!;
    const buf = buffers[v.buffer];
    if (!buf) throw new Error(`buffer ${v.buffer} missing`);
    const stride = v.byteStride ?? n * comp.bytes;
    const base = (v.byteOffset ?? 0) + (a.byteOffset ?? 0);
    const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
    for (let i = 0; i < a.count; i++) for (let k = 0; k < n; k++) out[i * n + k] = comp.read(dv, base + i * stride + k * comp.bytes);
    return out;
  }

  const roots = json.scenes?.[json.scene ?? 0]?.nodes ?? (nodes.length ? nodes.map((_, i) => i).filter((i) => !nodes.some((n) => n.children?.includes(i))) : []);
  const pos: number[] = [], nor: number[] = [];
  let fileTriangles = 0, degenerate = 0, primitives = 0;
  const min: [number, number, number] = [Infinity, Infinity, Infinity], max: [number, number, number] = [-Infinity, -Infinity, -Infinity];

  const visit = (ni: number, parent: Mat4, depth: number): void => {
    if (depth > 64) throw new Error('glTF node hierarchy too deep (cycle?)');
    const node = nodes[ni];
    if (!node) throw new Error(`node ${ni} missing`);
    const world = multiply(parent, localMatrix(node));
    if (node.mesh !== undefined) {
      const mirror = determinant3(world) < 0;
      for (const prim of meshes[node.mesh]!.primitives) {
        const mode = prim.mode ?? 4;
        if (mode !== 4) throw new Error(`primitive mode ${mode} is not triangles (only mode 4 is supported)`);
        primitives++;
        const p = readAccessor(prim.attributes['POSITION']!);
        const idx = prim.indices !== undefined ? readAccessor(prim.indices) : Array.from({ length: p.length / 3 }, (_, i) => i);
        for (let t = 0; t + 2 < idx.length; t += 3) {
          fileTriangles++;
          const order = mirror ? [0, 2, 1] : [0, 1, 2];
          const v = order.map((k) => transformPoint(world, [p[idx[t + k]! * 3]!, p[idx[t + k]! * 3 + 1]!, p[idx[t + k]! * 3 + 2]!]));
          // geometric normal from the float32-rounded positions, exactly what the renderers will see
          const f = v.map((q) => q.map(Math.fround) as unknown as Vec3);
          const e1: Vec3 = [f[1]![0] - f[0]![0], f[1]![1] - f[0]![1], f[1]![2] - f[0]![2]];
          const e2: Vec3 = [f[2]![0] - f[0]![0], f[2]![1] - f[0]![1], f[2]![2] - f[0]![2]];
          const c = cross3(e1, e2);
          const len = Math.hypot(c[0], c[1], c[2]);
          if (!(len > 0)) { degenerate++; continue; }
          for (const q of f) {
            pos.push(q[0], q[1], q[2]);
            nor.push(c[0] / len, c[1] / len, c[2] / len);
            for (let k = 0; k < 3; k++) { min[k] = Math.min(min[k]!, q[k]!); max[k] = Math.max(max[k]!, q[k]!); }
          }
        }
      }
    }
    for (const c of node.children ?? []) visit(c, world, depth + 1);
  };
  for (const r of roots) visit(r, compose(), 0);

  const nv = pos.length / 3;
  const indices = new Uint32Array(nv);
  for (let i = 0; i < nv; i++) indices[i] = i;
  return {
    mesh: { positions: Float32Array.from(pos), normals: Float32Array.from(nor), indices },
    stats: { fileTriangles, degenerateDropped: degenerate, triangles: nv / 3, primitives, bounds: { min, max } },
  };
}

/** Load a .glb or .gltf (with external .bin or data: URIs). `sha256` (hex of the main file) is verified when given. */
export function loadGltfTriangles(path: string, opts: { sha256?: string } = {}): GltfLoadResult {
  const file = readFileSync(path);
  if (opts.sha256) {
    const got = createHash('sha256').update(file).digest('hex');
    if (got !== opts.sha256.toLowerCase()) throw new Error(`${path}: sha256 mismatch (expected ${opts.sha256}, got ${got}); run \`node bench/assets/fetch.ts\``);
  }
  let json: GltfJson;
  let glbBin: Buffer | undefined;
  if (file.length >= 4 && file.readUInt32LE(0) === 0x46546c67) ({ json, bin: glbBin } = parseGlbContainer(file));
  else json = JSON.parse(file.toString('utf8')) as GltfJson;
  const buffers = (json.buffers ?? []).map((b, i) => {
    if (b.uri === undefined) {
      if (i !== 0 || !glbBin) throw new Error(`buffer ${i} has no uri and no GLB BIN chunk`);
      return glbBin;
    }
    if (b.uri.startsWith('data:')) return Buffer.from(b.uri.slice(b.uri.indexOf(',') + 1), 'base64');
    return readFileSync(resolve(dirname(path), decodeURIComponent(b.uri)));
  });
  return triangulateGltf(json, buffers);
}
