/** Geometry helper for one-sided quad emitters (world-space rectangles of two triangles). */
import { Matrix4, Quaternion, Vector3, type BufferGeometry } from 'three';

export interface QuadInfo {
  center: Vector3;
  width: number;
  height: number;
  /** Rotation that maps a three.js RectAreaLight (width along X, height along Y, emitting along -Z) onto the quad's front face. */
  rotation: Quaternion;
  /** Unit normal of the front (counter-clockwise) face. */
  normal: Vector3;
}

export function analyseQuad(geometry: BufferGeometry, world: Matrix4): QuadInfo | null {
  const position = geometry.getAttribute('position');
  const index = geometry.getIndex();
  if (!index || index.count !== 6) return null;
  const points: Vector3[] = [];
  const unique = new Map<string, Vector3>();
  for (let i = 0; i < position.count; i++) {
    const p = new Vector3().fromBufferAttribute(position, i).applyMatrix4(world);
    points.push(p);
    unique.set(`${p.x.toFixed(6)},${p.y.toFixed(6)},${p.z.toFixed(6)}`, p);
  }
  if (unique.size !== 4) return null;
  const v = [...unique.values()];
  // Front face (glTF: counter-clockwise) normal from the first triangle.
  const a = points[index.getX(0)]!, b = points[index.getX(1)]!, c = points[index.getX(2)]!;
  const normal = new Vector3().subVectors(b, a).cross(new Vector3().subVectors(c, a)).normalize();
  // Diagonal = the farthest pair; the other two vertices are the corners adjacent to v[0]'s opposite.
  let best = -1, di = 0, dj = 1;
  for (let i = 0; i < 4; i++) for (let j = i + 1; j < 4; j++) {
    const d = v[i]!.distanceTo(v[j]!);
    if (d > best) { best = d; di = i; dj = j; }
  }
  const others = [0, 1, 2, 3].filter((k) => k !== di && k !== dj);
  const origin = v[di]!;
  const u = new Vector3().subVectors(v[others[0]!]!, origin);
  const w = new Vector3().subVectors(v[others[1]!]!, origin);
  if (Math.abs(u.dot(w)) > 1e-4 * u.length() * w.length()) return null; // not a rectangle
  const center = new Vector3().addVectors(v[di]!, v[dj]!).multiplyScalar(0.5);
  // RectAreaLight emits along local -Z, width along local X, height along local Y.
  const x = u.clone().normalize();
  const z = normal.clone().negate();
  const y = new Vector3().crossVectors(z, x).normalize();
  const rotation = new Quaternion().setFromRotationMatrix(new Matrix4().makeBasis(x, y, z));
  return { center, width: u.length(), height: w.length(), rotation, normal };
}
