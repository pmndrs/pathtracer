/**
 * V7.4: media in the real-time chain. A rig of two kernels fed the same cameras and seed:
 *
 *   A  (`frameMedia` off)  arms `raw` (the frame's own 1 spp beauty), `raw-acc` (the mean of the raw frames since the cut) and `svgf-all`
 *                          (SVGF on everything: the chain as it is; the medium's single scattering sits in the demodulated planes)
 *   B  (`frameMedia` on)   arm `svgf-split`: the camera ray's single scattering (orders 0-1 of the medium) is plane 6, accumulated over time with a
 *                          reprojected exponential history and NOT spatially filtered (`emissionTemporal`); SVGF filters only the rest (surfaces + multiple scattering)
 *
 * Both run the same path length and the same estimator (forced + equiangular on homogeneous media), so `raw` of A and B are the same image up to float rounding.
 * The upscaler is created by the chain but never run (`post(..., { upscale: false })`): the denoised stage is what is measured.
 *
 * Browser side (WebGPU); the metrics are in `media-metrics.ts` (pure).
 */
import type { PtLookAtCamera } from '@pmndrs/pathtracer';
import type { SceneDesc } from '../scenes/schema.ts';
import { readTextureRgba16f, type PipelineStage } from './chain.ts';
import { RealtimePipelineSession } from './session.ts';

export type MediaArm = 'raw' | 'raw-acc' | 'svgf-all' | 'svgf-split';
export const MEDIA_ARMS: readonly MediaArm[] = ['raw', 'raw-acc', 'svgf-all', 'svgf-split'];

export interface MediaImg {
  width: number;
  height: number;
  channels: 4;
  data: Float32Array;
}

export interface MediaRigOptions {
  scene: SceneDesc;
  width: number;
  height: number;
  maxDepth: number;
  /** Adapter settings of the media estimator (`volumeForcedNee`, `volumeEquiangular`, `volumeMarch`, ...). */
  media: Record<string, unknown>;
  seed: number;
  /** Which kernels to build (default both). */
  kernels?: ('all' | 'split')[];
  /** The plane's history floor and variance clip (`emissionTemporal`). */
  alpha?: number;
  clip?: number;
  /** Keep a CPU mean of the raw frames (`raw-acc`; one readback per frame, never in a timed region). */
  trackRaw?: boolean;
  /** SVGF options passed to the chain's denoiser (diagnostics). */
  denoiser?: Record<string, unknown>;
}

export class MediaRig {
  readonly a?: RealtimePipelineSession;
  readonly b?: RealtimePipelineSession;
  readonly o!: MediaRigOptions;
  frames = 0;
  private rawSum: Float64Array | null = null;
  private cutNext = true;

  private constructor(init: Record<string, unknown>) {
    Object.assign(this, init);
  }

  static async create(o: MediaRigOptions): Promise<MediaRig> {
    const kernels = o.kernels ?? ['all', 'split'];
    const mk = (split: boolean, share?: RealtimePipelineSession) =>
      RealtimePipelineSession.create({
        scene: o.scene,
        width: o.width,
        height: o.height,
        displayWidth: 2 * o.width,
        displayHeight: 2 * o.height,
        maxDepth: o.maxDepth,
        ...(o.denoiser ? { denoiser: o.denoiser } : {}),
        ...(share ? { shareDeviceWith: share } : {}),
        adapter: { ...o.media, seed: o.seed, frameMedia: split },
        emissionTemporal: split ? { alpha: o.alpha ?? 0.1, clip: o.clip ?? 0 } : false,
      });
    const a = kernels.includes('all') ? await mk(false) : undefined;
    const b = kernels.includes('split') ? await mk(true, a) : undefined;
    return new MediaRig({ a, b, o });
  }

  /** A camera cut on the next frame (histories restart; the frame counter and the raw mean too). */
  cut(): void {
    this.cutNext = true;
    this.a?.cutNext();
    this.b?.cutNext();
  }

  /** One frame of every kernel at `camera` (denoised stage only; no upscaler). */
  async frame(camera: PtLookAtCamera): Promise<void> {
    if (this.cutNext) {
      this.cutNext = false;
      this.frames = 0;
      this.rawSum = null;
    }
    for (const s of [this.a, this.b]) if (s) await s.frame(camera, { upscale: false });
    this.frames++;
    if (this.o.trackRaw && this.a) {
      const img = await readTextureRgba16f(this.a.device, this.a.arm.tex.raw);
      this.rawSum ??= new Float64Array(img.data.length);
      for (let i = 0; i < img.data.length; i++) this.rawSum[i] = (this.rawSum[i] as number) + (img.data[i] as number);
    }
  }

  async device(): Promise<GPUDevice[]> {
    return [this.a?.device, this.b?.device].filter((d): d is GPUDevice => !!d);
  }

  /** The image of an arm at the current frame (linear RGBA f32, render size). */
  async image(arm: MediaArm): Promise<MediaImg> {
    if (arm === 'raw-acc') {
      if (!this.rawSum) throw new Error('raw-acc: construct the rig with trackRaw');
      const data = new Float32Array(this.rawSum.length);
      for (let i = 0; i < data.length; i++) data[i] = (this.rawSum[i] as number) / this.frames;
      return { width: this.o.width, height: this.o.height, channels: 4, data };
    }
    const session = arm === 'svgf-split' ? this.b : this.a;
    if (!session) throw new Error(`arm ${arm}: its kernel was not built`);
    const stage: PipelineStage = arm === 'raw' ? 'raw' : 'denoised';
    return readTextureRgba16f(session.device, session.arm.stage(stage));
  }

  destroy(): void {
    this.a?.destroy();
    this.b?.destroy();
  }
}
