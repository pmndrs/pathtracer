/**
 * Network shape and training settings for research track N (docs/design/nrc.md).
 *
 * Defaults follow VkNRC's design (§1b, D28): 64 inputs → 5 hidden layers of 64 → 3 outputs, ReLU, no bias,
 * relative-L2 luminance loss, Adam lr 0.002 / β 0.9, 0.999, EMA 0.99, He-normal init. The code is ours.
 *
 * Weight layout: layer by layer, each layer row-major [out][in] (one row of `in` weights per output unit).
 * Layer l maps `layerIn(l)` → `layerOut(l)`; the last layer is linear, the others are followed by ReLU.
 */

export interface MlpShape {
  /** Encoded input width. Must be a multiple of 8 (matrix tiles). */
  inputs: number;
  /** Hidden width. Multiple of 8. */
  width: number;
  /** Hidden layers, i.e. ReLU layers. The network has `hidden + 1` weight matrices. */
  hidden: number;
  /** Output width (3 for RGB radiance). */
  outputs: number;
}

export interface TrainSettings {
  learningRate: number;
  beta1: number;
  beta2: number;
  epsilon: number;
  /** EMA decay for the inference weights (0 disables the EMA copy). */
  emaDecay: number;
  /** Relative-L2 denominator floor: loss = |p - t|² / (lum(p)² + lossEpsilon), lum(p) held constant. */
  lossEpsilon: number;
}

export const DEFAULT_SHAPE: MlpShape = { inputs: 64, width: 64, hidden: 5, outputs: 3 };

export const DEFAULT_TRAIN: TrainSettings = {
  learningRate: 0.002,
  beta1: 0.9,
  beta2: 0.999,
  epsilon: 1e-8,
  emaDecay: 0.99,
  lossEpsilon: 0.01,
};

/** Rec. 601 luma, as used by the relative-L2 luminance loss. */
export const LUMA = [0.299, 0.587, 0.114] as const;

export function layerCount(shape: MlpShape): number {
  return shape.hidden + 1;
}

export function layerIn(shape: MlpShape, l: number): number {
  return l === 0 ? shape.inputs : shape.width;
}

export function layerOut(shape: MlpShape, l: number): number {
  return l === shape.hidden ? shape.outputs : shape.width;
}

/** Offset of layer l's first weight. */
export function layerOffset(shape: MlpShape, l: number): number {
  let offset = 0;
  for (let i = 0; i < l; i++) offset += layerIn(shape, i) * layerOut(shape, i);
  return offset;
}

export function weightCount(shape: MlpShape): number {
  return layerOffset(shape, layerCount(shape));
}

export function validateShape(shape: MlpShape): void {
  for (const [k, v] of Object.entries(shape)) {
    if (!Number.isInteger(v) || v <= 0) throw new Error(`MlpShape.${k} must be a positive integer, got ${v}`);
  }
  if (shape.inputs % 8 || shape.width % 8) throw new Error('MlpShape inputs and width must be multiples of 8');
  if (shape.outputs > shape.width) throw new Error('MlpShape outputs must not exceed width');
}
