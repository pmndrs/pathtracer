/**
 * CPU oracle (f64) for the WGSL MLP: forward, backward, loss, Adam, EMA. Slow and obvious on purpose; every GPU arm
 * is checked against it (docs/internal/design/nrc.md §4).
 *
 * Network: a_0 = x; a_{l+1} = relu(W_l a_l) for l < hidden; y = W_hidden a_hidden (linear). No bias (the encoding
 * carries constant-one inputs instead).
 * Loss per record: Σ_c (y_c − t_c)² / (lum(max(y, 0))² + ε), with lum held constant (no gradient through the
 * denominator), averaged over the batch. Gradient dL/dy_c = 2 (y_c − t_c) / (lum² + ε) / N.
 */

import { DEFAULT_TRAIN, LUMA, layerCount, layerIn, layerOffset, layerOut, weightCount, type MlpShape, type TrainSettings } from './config.ts';
import { Rng } from './rng.ts';

/** He-normal init, σ = sqrt(2 / fan_in), seeded. */
export function initWeights(shape: MlpShape, seed: number): Float32Array {
  const rng = new Rng(seed);
  const w = new Float32Array(weightCount(shape));
  for (let l = 0; l < layerCount(shape); l++) {
    const sigma = Math.sqrt(2 / layerIn(shape, l));
    const off = layerOffset(shape, l);
    for (let i = 0; i < layerIn(shape, l) * layerOut(shape, l); i++) w[off + i] = sigma * rng.normal();
  }
  return w;
}

export interface ForwardTrace {
  /** a_0 … a_hidden, each `width` (a_0 is `inputs`). */
  activations: Float64Array[];
  output: Float64Array;
}

export function forward(shape: MlpShape, weights: ArrayLike<number>, input: ArrayLike<number>): ForwardTrace {
  const activations: Float64Array[] = [Float64Array.from(input)];
  let a = activations[0] as Float64Array;
  for (let l = 0; l < layerCount(shape); l++) {
    const nIn = layerIn(shape, l);
    const nOut = layerOut(shape, l);
    const off = layerOffset(shape, l);
    const z = new Float64Array(nOut);
    for (let o = 0; o < nOut; o++) {
      let s = 0;
      for (let i = 0; i < nIn; i++) s += (weights[off + o * nIn + i] as number) * (a[i] as number);
      z[o] = l === shape.hidden ? s : Math.max(s, 0);
    }
    if (l === shape.hidden) return { activations, output: z };
    activations.push(z);
    a = z;
  }
  throw new Error('unreachable');
}

export function luminance(y: ArrayLike<number>): number {
  return LUMA[0] * Math.max(y[0] as number, 0) + LUMA[1] * Math.max(y[1] as number, 0) + LUMA[2] * Math.max(y[2] as number, 0);
}

/** Per-record loss (not averaged). */
export function recordLoss(y: ArrayLike<number>, t: ArrayLike<number>, settings: TrainSettings = DEFAULT_TRAIN): number {
  const lum = luminance(y);
  const denom = lum * lum + settings.lossEpsilon;
  let s = 0;
  for (let c = 0; c < 3; c++) s += ((y[c] as number) - (t[c] as number)) ** 2;
  return s / denom;
}

/** dL/dy for one record, already divided by the batch size. */
export function lossGradient(y: ArrayLike<number>, t: ArrayLike<number>, batch: number, settings: TrainSettings = DEFAULT_TRAIN): Float64Array {
  const lum = luminance(y);
  const scale = 2 / ((lum * lum + settings.lossEpsilon) * batch);
  const g = new Float64Array(3);
  for (let c = 0; c < 3; c++) g[c] = scale * ((y[c] as number) - (t[c] as number));
  return g;
}

/**
 * Batch gradient dL/dW (mean loss). `inputs` is N × shape.inputs, `targets` N × 3.
 * Returns the gradient and the mean loss.
 */
export function batchGradient(
  shape: MlpShape,
  weights: ArrayLike<number>,
  inputs: ArrayLike<number>,
  targets: ArrayLike<number>,
  n: number,
  settings: TrainSettings = DEFAULT_TRAIN,
): { gradient: Float64Array; loss: number } {
  const gradient = new Float64Array(weightCount(shape));
  let loss = 0;
  for (let r = 0; r < n; r++) {
    const x = Array.prototype.slice.call(inputs, r * shape.inputs, (r + 1) * shape.inputs) as number[];
    const t = Array.prototype.slice.call(targets, r * 3, r * 3 + 3) as number[];
    const trace = forward(shape, weights, x);
    loss += recordLoss(trace.output, t, settings) / n;
    // Backward: g holds dL/dz of the current layer's output (z = pre-activation; ReLU mask applied below).
    let g = lossGradient(trace.output, t, n, settings);
    for (let l = shape.hidden; l >= 0; l--) {
      const nIn = layerIn(shape, l);
      const nOut = layerOut(shape, l);
      const off = layerOffset(shape, l);
      const a = trace.activations[l] as Float64Array;
      for (let o = 0; o < nOut; o++) {
        const go = g[o] as number;
        if (go === 0) continue;
        for (let i = 0; i < nIn; i++) gradient[off + o * nIn + i] = (gradient[off + o * nIn + i] as number) + go * (a[i] as number);
      }
      if (l === 0) break;
      const next = new Float64Array(nIn);
      for (let i = 0; i < nIn; i++) {
        if ((a[i] as number) <= 0) continue; // ReLU derivative: a = relu(z) > 0 ⇔ z > 0
        let s = 0;
        for (let o = 0; o < nOut; o++) s += (weights[off + o * nIn + i] as number) * (g[o] as number);
        next[i] = s;
      }
      g = next;
    }
  }
  return { gradient, loss };
}

/** Adam + EMA state, f32 master weights like the GPU (so the oracle sees the same rounding of the weights). */
export class AdamState {
  readonly weights: Float32Array;
  readonly ema: Float32Array;
  readonly m: Float64Array;
  readonly v: Float64Array;
  step = 0;

  private readonly _settings: TrainSettings;

  constructor(weights: Float32Array, settings: TrainSettings = DEFAULT_TRAIN) {
    this._settings = settings;
    this.weights = Float32Array.from(weights);
    this.ema = Float32Array.from(weights);
    this.m = new Float64Array(weights.length);
    this.v = new Float64Array(weights.length);
  }

  get settings(): TrainSettings {
    return this._settings;
  }

  /** One Adam step on a mean-loss gradient. Non-finite gradient entries count as 0 (as on the GPU). */
  apply(gradient: ArrayLike<number>): void {
    const { learningRate, beta1, beta2, epsilon, emaDecay } = this._settings;
    this.step++;
    const decay = this._settings.emaWarmup ? Math.min(emaDecay, (1 + this.step) / (10 + this.step)) : emaDecay;
    const c1 = 1 - beta1 ** this.step;
    const c2 = 1 - beta2 ** this.step;
    for (let i = 0; i < this.weights.length; i++) {
      let g = gradient[i] as number;
      if (!Number.isFinite(g)) g = 0;
      const m = beta1 * (this.m[i] as number) + (1 - beta1) * g;
      const v = beta2 * (this.v[i] as number) + (1 - beta2) * g * g;
      this.m[i] = m;
      this.v[i] = v;
      this.weights[i] = (this.weights[i] as number) - (learningRate * (m / c1)) / (Math.sqrt(v / c2) + epsilon);
      this.ema[i] = decay * (this.ema[i] as number) + (1 - decay) * (this.weights[i] as number);
    }
  }

  /** EMA of the weights. It starts at the initial weights rather than 0, so it needs no bias correction. */
  emaWeights(): Float32Array {
    return Float32Array.from(this.ema);
  }
}
