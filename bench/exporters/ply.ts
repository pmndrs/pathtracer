import type { TriangleMesh } from '../scenes/geometry.ts';

export interface PlyOptions {
  /** Write `v' = 1 - v` instead of `v` (the bench's uvs are glTF's, origin top-left; renderers that put v = 0 at the bottom of the image need the flip). */
  flipV?: boolean;
}

/** Binary little-endian PLY with float32 x y z nx ny nz (and u v when the mesh has uvs) and uint8-count/uint32 triangle lists. */
export function encodePly(mesh: TriangleMesh, options: PlyOptions = {}): Buffer {
  const nv = mesh.positions.length / 3;
  const nt = mesh.indices.length / 3;
  const uvs = mesh.uvs;
  const header =
    `ply\nformat binary_little_endian 1.0\ncomment pathtracer bench export\n` +
    `element vertex ${nv}\nproperty float x\nproperty float y\nproperty float z\n` +
    `property float nx\nproperty float ny\nproperty float nz\n` +
    (uvs ? `property float u\nproperty float v\n` : '') +
    `element face ${nt}\nproperty list uchar uint vertex_indices\nend_header\n`;
  const h = Buffer.from(header, 'latin1');
  const stride = uvs ? 32 : 24;
  const body = Buffer.alloc(nv * stride + nt * 13);
  let o = 0;
  for (let i = 0; i < nv; i++) {
    for (let k = 0; k < 3; k++) { body.writeFloatLE(mesh.positions[i * 3 + k]!, o); o += 4; }
    for (let k = 0; k < 3; k++) { body.writeFloatLE(mesh.normals[i * 3 + k]!, o); o += 4; }
    if (uvs) {
      body.writeFloatLE(uvs[i * 2]!, o); o += 4;
      body.writeFloatLE(options.flipV ? 1 - uvs[i * 2 + 1]! : uvs[i * 2 + 1]!, o); o += 4;
    }
  }
  for (let t = 0; t < nt; t++) {
    body.writeUInt8(3, o); o += 1;
    for (let k = 0; k < 3; k++) { body.writeUInt32LE(mesh.indices[t * 3 + k]!, o); o += 4; }
  }
  return Buffer.concat([h, body]);
}

/** Same triangles with reversed winding and negated normals (used to emulate two-sided emission). */
export function flipMesh(mesh: TriangleMesh): TriangleMesh {
  const indices = new Uint32Array(mesh.indices);
  for (let t = 0; t < indices.length; t += 3) {
    const a = indices[t + 1]!;
    indices[t + 1] = indices[t + 2]!;
    indices[t + 2] = a;
  }
  return { positions: mesh.positions, normals: mesh.normals.map((v) => -v), indices, ...(mesh.uvs ? { uvs: mesh.uvs } : {}) };
}
