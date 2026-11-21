// A tiny dense network with one hidden layer. All distances in the lesson change
// one mesh marker along its model-space correction direction, by up to 20 mm.
export interface Pose { bend: number; twist: number }
export interface Reading extends Pose { target: number }
export type Inputs = 'bend' | 'both';
export const fullOffset = 20;
export const clampDrive = (drive: number) => Math.max(0, Math.min(1, drive));

function label(pose: Pose): Reading {
  const flex = Math.max(0, (pose.bend - 20) / 100);
  return { ...pose, target: flex * (0.65 + 0.35 * pose.twist / 60) };
}

export function recordSweep(twist: number): Reading[] {
  return Array.from({ length: 13 }, (_, i) => label({ bend: i * 10, twist }));
}

// A separate motion, sampled once and used only to evaluate predictions.
export const reach = Array.from({ length: 61 }, (_, i) => {
  const phase = 2 * Math.PI * i / 60;
  return label({ bend: 65 - 51 * Math.cos(phase), twist: 48 * Math.sin(phase) });
});

function features(p: Pose, inputs: Inputs) {
  return inputs === 'both' ? [p.bend / 120, p.twist / 60] : [p.bend / 120];
}

export function makeModel(inputs: Inputs) {
  // The two runs start with identical bend weights. The two-input run can
  // additionally learn twist weights. A fixed seed makes reset repeatable.
  let seed = 87;
  const random = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 2 ** 32; };
  const hidden = Array.from({ length: 8 }, () => {
    const bend = random() * 2 - 0.7, twist = random() * 1.2 - 0.6;
    return { w: inputs === 'both' ? [bend, twist] : [bend], bias: random() * 0.5 - 0.15, out: random() * 0.6 - 0.3 };
  });
  let outputBias = 0.15;

  function forward(x: number[]) {
    const h = hidden.map(n => Math.max(0, n.w.reduce((sum, w, j) => sum + w * x[j]!, n.bias)));
    const prediction = h.reduce((sum, value, k) => sum + value * hidden[k]!.out, outputBias);
    return { h, prediction };
  }

  function train(data: Reading[], steps: number) {
    if (!data.length) return;
    const rows = data.map(pose => ({ x: features(pose, inputs), target: pose.target }));
    for (let step = 0; step < steps; step++) {
      const grads = hidden.map(n => ({ w: n.w.map(() => 0), bias: 0, out: 0 }));
      let biasGrad = 0;
      for (const { x, target } of rows) {
        const { h, prediction } = forward(x);
        const dy = 2 * (prediction - target) / rows.length;
        biasGrad += dy;
        hidden.forEach((n, k) => {
          const g = grads[k]!;
          g.out += dy * h[k]!;
          if (h[k]! <= 0) return;
          const dz = dy * n.out;
          g.bias += dz;
          x.forEach((v, j) => { g.w[j]! += dz * v; });
        });
      }
      const rate = 0.12;
      outputBias -= rate * biasGrad;
      hidden.forEach((n, k) => {
        const g = grads[k]!;
        n.out -= rate * g.out;
        n.bias -= rate * g.bias;
        n.w = n.w.map((w, j) => w - rate * g.w[j]!);
      });
    }
  }
  return { predict: (p: Pose) => clampDrive(forward(features(p, inputs)).prediction), train };
}

// Average the observed labels at each bend, then interpolate between adjacent bends.
export function bendTable(data: Reading[]) {
  const bins = Array.from({ length: 13 }, (_, i) => {
    const matching = data.filter(row => row.bend === i * 10);
    return matching.reduce((sum, row) => sum + row.target, 0) / matching.length;
  });
  return (pose: Pose) => {
    const at = pose.bend / 10, lo = Math.floor(at), hi = Math.min(12, lo + 1);
    return bins[lo]! + (bins[hi]! - bins[lo]!) * (at - lo);
  };
}
