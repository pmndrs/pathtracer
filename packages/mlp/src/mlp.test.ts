import { describe, expect, it } from 'vitest';
import { DEFAULT_SHAPE, DEFAULT_TRAIN, layerOffset, weightCount, type MlpShape } from './config.ts';
import { BLOB_BINS, NRC_ENCODED, encodeImage, encodeNrc, oneBlob, triangleWave } from './encoding.ts';
import { AdamState, batchGradient, forward, initWeights, luminance } from './oracle.ts';
import { Rng } from './rng.ts';
import { BUFFER_HOOKS_INFER, BUFFER_HOOKS_TRAIN, adamWgsl, inferWgsl, prepareWgsl, trainWgsl, weightGradientWgsl } from './wgsl.ts';

const SMALL: MlpShape = { inputs: 8, width: 8, hidden: 2, outputs: 3 };

function randomBatch(shape: MlpShape, n: number, seed: number) {
  const rng = new Rng(seed);
  const inputs = Float64Array.from({ length: n * shape.inputs }, () => rng.next() * 2 - 1);
  const targets = Float64Array.from({ length: n * 3 }, () => rng.next() * 2);
  return { inputs, targets };
}

describe('config', () => {
  it('matches VkNRC-sized defaults: 5 × 64² + 64 × 3 weights', () => {
    expect(weightCount(DEFAULT_SHAPE)).toBe(5 * 64 * 64 + 64 * 3);
    expect(layerOffset(DEFAULT_SHAPE, 5)).toBe(5 * 64 * 64);
  });
});

describe('encoding', () => {
  it('triangle wave hits 0, 1, -1 where sin(πx) does', () => {
    expect(triangleWave(0)).toBeCloseTo(0, 12);
    expect(triangleWave(0.5)).toBeCloseTo(1, 12);
    expect(triangleWave(1)).toBeCloseTo(0, 12);
    expect(triangleWave(1.5)).toBeCloseTo(-1, 12);
    expect(triangleWave(-0.5)).toBeCloseTo(-1, 12);
  });

  it('one-blob bins sum to 1 when the kernel lies inside [0, 1] and peak at x', () => {
    const out = new Float64Array(BLOB_BINS);
    oneBlob(0.5, out, 0);
    expect(out.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    oneBlob(0.125, out, 0);
    expect(out.indexOf(Math.max(...out))).toBe(0);
    oneBlob(0.875, out, 0);
    expect(out.indexOf(Math.max(...out))).toBe(BLOB_BINS - 1);
  });

  it('NRC layout fills exactly 64 values ending in two ones', () => {
    const e = encodeNrc({ position: [0.1, -0.2, 0.3], direction: [0.4, 0.5], normal: [0.6, 0.7], roughness: 0.3, diffuse: [0.5, 0.5, 0.5], specular: [0.04, 0.04, 0.04] });
    expect(e.length).toBe(NRC_ENCODED);
    expect(e[62]).toBe(1);
    expect(e[63]).toBe(1);
  });

  it('image encoding has a constant-one input', () => {
    const e = encodeImage(0.3, 0.7);
    expect(e[32]).toBe(1);
    expect(e.slice(33).every((v) => v === 0)).toBe(true);
  });
});

describe('oracle', () => {
  it('backward matches central finite differences (denominator held constant)', () => {
    const w = initWeights(SMALL, 7);
    const n = 5;
    const { inputs, targets } = randomBatch(SMALL, n, 3);
    const { gradient } = batchGradient(SMALL, w, inputs, targets, n);

    // Loss with each record's luminance denominator frozen at the unperturbed weights, which is what the gradient uses.
    const denoms: number[] = [];
    for (let r = 0; r < n; r++) {
      const lum = luminance(forward(SMALL, w, inputs.subarray(r * SMALL.inputs, (r + 1) * SMALL.inputs)).output);
      denoms.push(lum * lum + DEFAULT_TRAIN.lossEpsilon);
    }
    const frozenLoss = (weights: Float64Array) => {
      let loss = 0;
      for (let r = 0; r < n; r++) {
        const y = forward(SMALL, weights, inputs.subarray(r * SMALL.inputs, (r + 1) * SMALL.inputs)).output;
        let s = 0;
        for (let c = 0; c < 3; c++) s += ((y[c] as number) - (targets[r * 3 + c] as number)) ** 2;
        loss += s / (denoms[r] as number) / n;
      }
      return loss;
    };
    const w64 = Float64Array.from(w);
    const h = 1e-6;
    let worst = 0;
    for (let i = 0; i < w64.length; i++) {
      const keep = w64[i] as number;
      w64[i] = keep + h;
      const up = frozenLoss(w64);
      w64[i] = keep - h;
      const down = frozenLoss(w64);
      w64[i] = keep;
      const fd = (up - down) / (2 * h);
      worst = Math.max(worst, Math.abs(fd - (gradient[i] as number)) / Math.max(1e-6, Math.abs(fd)));
    }
    expect(worst).toBeLessThan(1e-4);
  });

  it('one Adam step moves each weight by about lr against its gradient sign', () => {
    const w = new Float32Array([1, -1, 0.5]);
    const adam = new AdamState(w);
    adam.apply([0.3, -2, 0]);
    expect(adam.weights[0]).toBeCloseTo(1 - DEFAULT_TRAIN.learningRate, 6);
    expect(adam.weights[1]).toBeCloseTo(-1 + DEFAULT_TRAIN.learningRate, 6);
    expect(adam.weights[2]).toBeCloseTo(0.5, 6);
  });

  it('training reduces the loss on a small regression', () => {
    const shape: MlpShape = { inputs: 8, width: 16, hidden: 2, outputs: 3 };
    const adam = new AdamState(initWeights(shape, 11), { ...DEFAULT_TRAIN, learningRate: 0.01 });
    const n = 64;
    const { inputs } = randomBatch(shape, n, 5);
    const targets = new Float64Array(n * 3);
    for (let r = 0; r < n; r++) {
      const x0 = inputs[r * 8] as number;
      const x1 = inputs[r * 8 + 1] as number;
      targets.set([1 + x0, 1 + x1 * x1, 1 + 0.5 * x0 * x1], r * 3);
    }
    const first = batchGradient(shape, adam.weights, inputs, targets, n).loss;
    for (let s = 0; s < 200; s++) adam.apply(batchGradient(shape, adam.weights, inputs, targets, n).gradient);
    const last = batchGradient(shape, adam.weights, inputs, targets, n).loss;
    expect(last).toBeLessThan(first * 0.2);
  });
});

describe('wgsl generator', () => {
  const o32 = { shape: DEFAULT_SHAPE, precision: 'f32' as const };
  const o16 = { shape: DEFAULT_SHAPE, precision: 'f16' as const };

  it('emits every entry point', () => {
    expect(inferWgsl(o32, BUFFER_HOOKS_INFER)).toContain('fn nnInfer(');
    expect(trainWgsl(o32, BUFFER_HOOKS_TRAIN)).toContain('fn nnForwardBackward(');
    expect(weightGradientWgsl(o32)).toContain('fn nnWeightGradient(');
    expect(prepareWgsl(o32)).toContain('fn nnPrepare(');
    expect(adamWgsl(o32)).toContain('fn nnAdam(');
    expect(adamWgsl(o32)).toContain('fn nnSync(');
  });

  it('f16 enables f16 and f32 does not', () => {
    expect(trainWgsl(o16, BUFFER_HOOKS_TRAIN).startsWith('enable f16;')).toBe(true);
    expect(trainWgsl(o32, BUFFER_HOOKS_TRAIN)).not.toContain('enable f16');
  });

  it('never passes storage pointers into functions (needs a language feature)', () => {
    for (const code of [inferWgsl(o32, BUFFER_HOOKS_INFER), trainWgsl(o32, BUFFER_HOOKS_TRAIN)]) expect(code).not.toMatch(/ptr<storage/);
  });

  it('stays under the ~50 KB kernel size trap', () => {
    expect(trainWgsl(o16, BUFFER_HOOKS_TRAIN).length).toBeLessThan(50_000);
  });
});
