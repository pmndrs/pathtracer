/**
 * Minimal glTF 2.0 / GLB triangle reader for the bench (Node only). Reads POSITION + indices + node transforms and
 * nothing else: no materials, textures, alpha modes, skins or morph targets (the scene loader, `gltf-scene.ts`, reads those; accessor
 * strides, normalized and sparse accessors come from `gltf-read.ts`). Triangles only
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
import { localMatrix, readAccessor, rootNodes } from './gltf-read.ts';
import type { GltfJson } from './gltf-read.ts';
import type { Mat4, Vec3 } from './schema.ts';

export type { GltfJson } from './gltf-read.ts';

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

/** Parse already-split glTF content. `buffers[i]` is the bytes of `json.buffers[i]`. */
export function triangulateGltf(json: GltfJson, buffers: readonly Uint8Array[]): GltfLoadResult {
  const meshes = json.meshes ?? [], nodes = json.nodes ?? [];

  const readAcc = (index: number): Float64Array => readAccessor(json, buffers, index);

  const roots = rootNodes(json);
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
        const p = readAcc(prim.attributes['POSITION']!);
        const idx = prim.indices !== undefined ? readAcc(prim.indices) : Array.from({ length: p.length / 3 }, (_, i) => i);
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

/** Read a .glb or .gltf (with external .bin or data: URIs): its JSON and buffers. `sha256` (hex of the main file) is verified when given. */
export function readGltfFile(path: string, opts: { sha256?: string } = {}): { json: GltfJson; buffers: Buffer[]; glbBin?: Buffer } {
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
  return { json, buffers, ...(glbBin ? { glbBin } : {}) };
}

/** Load a .glb or .gltf (with external .bin or data: URIs). `sha256` (hex of the main file) is verified when given. */
export function loadGltfTriangles(path: string, opts: { sha256?: string } = {}): GltfLoadResult {
  const { json, buffers } = readGltfFile(path, opts);
  return triangulateGltf(json, buffers);
}
