/**
 * Motion-compensated "boil" (calm) of an image sequence, after the Denoiser scorecard's definition (Denoiser docs/research/rt-denoiser/tracks/t0-scorecard.md,
 * "the five numbers", 1. Calm): the second temporal difference along the reprojected history,
 *
 *     boil_k = RMS over valid pixels of ( x_t(p) - 2 x_(t-k)(p_k) + x_(t-2k)(p_2k) ) / sqrt(6)
 *
 * on tonemapped luma, with the reprojection chained hop by hop through each frame's own motion vectors (NDC delta current - previous, +y up, the
 * frame planes' plane 3). A hop is valid when the reprojected position is inside the frame and the previous frame's view depth there agrees with
 * the expected one (plane 3 alpha = motionZ), so disocclusions do not count as boil. A static, noise-free image has boil 0 under exact reprojection; noise
 * of standard deviation s gives s (sqrt(1 + 4 + 1) / sqrt(6) = 1).
 *
 * Differences from the scorecard (documented in docs/design/realtime.md section 5): no per-region split (static / moving / disoccluded come from a per-frame
 * reference there; here only the depth-consistency validity), nearest-tap motion and bilinear luma, the reference's own boil floor is not available
 * (no per-frame references), strides 1 and 4 only. Compare variants on the same path, not across scenes. Pure TS: unit-tested without a GPU.
 */

export interface BoilFrame {
  /** Tonemapped luma at image resolution, row 0 = top. */
  luma: Float32Array;
  width: number;
  height: number;
  /** Motion plane at its own (render) resolution: x, y = NDC delta current - previous per pixel (2 floats). */
  motion: Float32Array;
  /** viewZ (negative in front, 0 = miss) and motionZ (viewZ(previous) - viewZ(current)) per motion pixel. */
  viewZ: Float32Array;
  motionZ: Float32Array;
  mw: number;
  mh: number;
}

export interface BoilResult {
  /** sqrt(mean(d^2) / 6) over valid pixels; NaN when no frame pair was complete. */
  boil: number;
  samples: number;
  /** Share of probed pixels that survived the validity chain. */
  validFrac: number;
}

export class BoilMeter {
  private readonly ring: BoilFrame[] = [];
  private sum = 0;
  private count = 0;
  private probed = 0;

  /** `stride` k (frames per hop group), `grid`: probe every `grid`-th pixel of the image in each direction. */
  constructor(readonly stride = 1, readonly grid = 2) {}

  /** Adds the next frame of the sequence (after a camera cut call `restart` first). */
  push(frame: BoilFrame): void {
    this.ring.push(frame);
    const need = 2 * this.stride + 1;
    while (this.ring.length > need) this.ring.shift();
    if (this.ring.length === need) this.measure();
  }

  /** Forget the history (a cut): frames before it are not reprojectable. */
  restart(): void {
    this.ring.length = 0;
  }

  result(): BoilResult {
    return { boil: this.count > 0 ? Math.sqrt(this.sum / this.count / 6) : NaN, samples: this.count, validFrac: this.probed > 0 ? this.count / this.probed : 0 };
  }

  private measure(): void {
    const k = this.stride;
    const ring = this.ring;
    const cur = ring[ring.length - 1] as BoilFrame;
    const { width: W, height: H } = cur;
    for (let y = 0; y < H; y += this.grid) {
      for (let x = 0; x < W; x += this.grid) {
        this.probed++;
        let nx = ((x + 0.5) / W) * 2 - 1;
        let ny = 1 - ((y + 0.5) / H) * 2;
        let ok = true;
        const lumas: number[] = [(cur.luma[y * W + x]) as number];
        for (let hop = 0; hop < 2 * k && ok; hop++) {
          const f = ring[ring.length - 1 - hop] as BoilFrame;
          const prev = ring[ring.length - 2 - hop] as BoilFrame;
          // Nearest motion pixel of this frame at the chained position.
          const mx = Math.min(f.mw - 1, Math.max(0, Math.floor(((nx + 1) / 2) * f.mw)));
          const my = Math.min(f.mh - 1, Math.max(0, Math.floor(((1 - ny) / 2) * f.mh)));
          const mi = my * f.mw + mx;
          const z = f.viewZ[mi] as number;
          const px = nx - (f.motion[mi * 2] as number);
          const py = ny - (f.motion[mi * 2 + 1] as number);
          if (!(px > -1 && px < 1 && py > -1 && py < 1)) {
            ok = false;
            break;
          }
          const qx = Math.min(prev.mw - 1, Math.max(0, Math.floor(((px + 1) / 2) * prev.mw)));
          const qy = Math.min(prev.mh - 1, Math.max(0, Math.floor(((1 - py) / 2) * prev.mh)));
          const zp = prev.viewZ[qy * prev.mw + qx] as number;
          const expected = z + (f.motionZ[mi] as number);
          if (z === 0 || zp === 0) {
            if (z !== zp) ok = false; // a miss must stay a miss
          } else if (Math.abs(zp - expected) > 0.03 * Math.abs(zp) + 1e-3) ok = false;
          nx = px;
          ny = py;
          if (hop === k - 1 || hop === 2 * k - 1) lumas.push(sampleLuma(prev, nx, ny));
        }
        if (!ok) continue;
        const d = (lumas[0] as number) - 2 * (lumas[1] as number) + (lumas[2] as number);
        this.sum += d * d;
        this.count++;
      }
    }
  }
}

/** Bilinear luma at an NDC position (+y up). */
function sampleLuma(f: BoilFrame, nx: number, ny: number): number {
  const { width: W, height: H, luma } = f;
  const fx = ((nx + 1) / 2) * W - 0.5;
  const fy = ((1 - ny) / 2) * H - 0.5;
  const x0 = Math.floor(fx);
  const y0 = Math.floor(fy);
  const tx = fx - x0;
  const ty = fy - y0;
  const at = (x: number, y: number) => luma[Math.min(H - 1, Math.max(0, y)) * W + Math.min(W - 1, Math.max(0, x))] as number;
  return (at(x0, y0) * (1 - tx) + at(x0 + 1, y0) * tx) * (1 - ty) + (at(x0, y0 + 1) * (1 - tx) + at(x0 + 1, y0 + 1) * tx) * ty;
}

/** Display luma of a linear rgba image after a Reinhard tonemap of the colour (the scorecard tonemaps before the difference). */
export function tonemappedLuma(data: Float32Array, width: number, height: number, channels = 4): Float32Array {
  const out = new Float32Array(width * height);
  for (let i = 0; i < out.length; i++) {
    const r = Math.max(0, data[i * channels] as number);
    const g = Math.max(0, data[i * channels + 1] as number);
    const b = Math.max(0, data[i * channels + 2] as number);
    const l = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    out[i] = l / (1 + l);
  }
  return out;
}
