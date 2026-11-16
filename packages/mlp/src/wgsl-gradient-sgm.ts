/**
 * Subgroup-matrix weight gradient (`gradient: 'sgm'`, research track N, N2r-d): the twin of the tiled arm's
 * `nnGradientTile`, writing the same per-split partials, so `nnGradientReduce` and Adam run unchanged after it.
 *
 * dW_l = δ_lᵀ · a_l summed over a split of the records. Both operands come straight from storage: the left tile is a
 * column-major view of the delta rows (M output units × K records), the right tile a row-major view of the activation
 * rows (K records × N inputs). Accumulators are f32 (the config must have f32 results: a split sums hundreds of records).
 * Workgroup (split, layer); subgroup s owns output rows s·M … s·M + M − 1 and holds one accumulator per input column
 * block. The output layer is one padded row block: its tile goes through workgroup memory and only the real rows are
 * written (the padded rows would land in the next layer's partials).
 *
 * The record loop runs in whole K-record steps up to `roundup(count, K)`, so the deltas of records count …
 * roundup(count, K) − 1 must be zero and their activations finite: the blocked sgm forward-backward writes both for
 * every record below `capacity` (the engine requires `capacity` to be a multiple of K and that forward arm).
 * Deterministic: fixed split, step and reduction order, no atomics.
 */

import { layerCount, layerOffset } from './config.ts';
import { header, rowStride, type MlpWgslOptions } from './wgsl.ts';
import { type GradientOptions, validateGradientOptions } from './wgsl-gradient.ts';
import { laneLoop, type SgmConfig } from './wgsl-sgm.ts';

export interface SgmGradientOptions {
  /** Operand type = the MLP precision, results f32. */
  config: SgmConfig;
  enableName?: string;
}

export function validateSgmGradient(o: MlpWgslOptions, g: GradientOptions, s: SgmGradientOptions): void {
  validateGradientOptions(o, g);
  const { shape } = o;
  const c = s.config;
  if (c.resultComponentType !== 'f32') throw new Error('sgm gradient: the config must have f32 results');
  if (c.componentType !== o.precision) throw new Error(`sgm gradient: operand type ${c.componentType} must equal precision ${o.precision}`);
  if (shape.inputs !== shape.width) throw new Error('sgm gradient: inputs must equal width (one column-block count for every layer)');
  if (shape.width % c.M || shape.width % c.N) throw new Error(`sgm gradient: width ${shape.width} must be a multiple of M=${c.M} and N=${c.N}`);
  if (shape.outputs > c.M) throw new Error(`sgm gradient: outputs ${shape.outputs} must fit one row block (M=${c.M})`);
}

/** Subgroups per workgroup: one per output row block of a hidden layer. */
export function sgmGradientSubgroups(o: MlpWgslOptions, s: SgmGradientOptions): number {
  return o.shape.width / s.config.M;
}

export function gradientSgmWgsl(o: MlpWgslOptions, g: GradientOptions, s: SgmGradientOptions): string {
  validateSgmGradient(o, g, s);
  const { shape } = o;
  const c = s.config;
  const S0 = rowStride(shape);
  const W = shape.width;
  const lanes = c.maxSubgroupSize;
  const subgroups = sgmGradientSubgroups(o, s);
  const wg = lanes * subgroups;
  const nBlocks = W / c.N;
  const left = `subgroup_matrix_left<${c.componentType}, ${c.K}, ${c.M}>`;
  const right = `subgroup_matrix_right<${c.componentType}, ${c.N}, ${c.K}>`;
  const result = `subgroup_matrix_result<f32, ${c.N}, ${c.M}>`;
  const layers = layerCount(shape);
  const offsets = Array.from({ length: layers }, (_, l) => `${layerOffset(shape, l)}u`).join(', ');
  const outOffset = layerOffset(shape, shape.hidden);
  const directives = [`enable ${s.enableName ?? 'chromium_experimental_subgroup_matrix'};`, ...(subgroups > 1 ? ['enable subgroups;'] : [])];
  const acc = Array.from({ length: nBlocks }, (_, nb) => `  var z${nb} = ${result}();`).join('\n');
  const rights = Array.from({ length: nBlocks }, (_, nb) => `        let a${nb} = subgroupMatrixLoad<${right}>(&nnActs, row + ${nb * c.N}u, false, ${S0}u);
        z${nb} = subgroupMatrixMultiplyAccumulate(d, a${nb}, z${nb});`).join('\n');
  const storeHidden = Array.from({ length: nBlocks }, (_, nb) => `      subgroupMatrixStore(&nnPartials, dst + ${nb * c.N}u, z${nb}, false, ${W}u);`).join('\n');
  const storeOut = Array.from({ length: nBlocks }, (_, nb) => `      subgroupMatrixStore(&nnOutTile, ${nb * c.N}u, z${nb}, false, ${W}u);`).join('\n');
  return `${header(o, directives)}
const NN_SPLITS = ${g.splits}u;
const NN_LAYER_OFFSET = array<u32, ${layers}>(${offsets});

@group(0) @binding(0) var<storage, read> nnState: NnState;
@group(0) @binding(1) var<uniform> params: NnParams;
@group(0) @binding(2) var<storage, read> nnActs: array<T>;
@group(0) @binding(3) var<storage, read> nnDeltas: array<T>;
@group(0) @binding(4) var<storage, read_write> nnPartials: array<f32>;

var<workgroup> nnCountShared: u32;
var<workgroup> nnOutTile: array<f32, ${c.M * W}>; // output layer: the padded row block before its real rows are copied

// Subgroup matrix ${c.M}×${c.N}×${c.K} ${c.componentType} → f32; ${subgroups} subgroup(s) of ${lanes} per workgroup (split, layer).
@compute @workgroup_size(${wg})
fn nnGradientSgm(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32${subgroups > 1 ? ', @builtin(subgroup_id) sgid: u32' : ''}) {
  let split = wid.x;
  let l = wid.y;
  if (lid == 0u) { nnCountShared = nnState.count; }
  let count = workgroupUniformLoad(&nnCountShared);
  let padded = (count + ${c.K - 1}u) / ${c.K}u * ${c.K}u;
  let per = ((count + NN_SPLITS - 1u) / NN_SPLITS + ${c.K - 1}u) / ${c.K}u * ${c.K}u;
  let begin = min(split * per, padded);
  let end = min(begin + per, padded);
  let steps = (end - begin) / ${c.K}u;
  let sg = ${subgroups > 1 ? 'sgid' : '0u'};
  let isOut = l == NN_HIDDEN;
${acc}
  if (!isOut || sg == 0u) {
    let base = (l * params.capacity + begin) * ${S0}u;
    for (var s = 0u; s < 0xffffffffu; s++) {
      if (s >= steps) { break; }
      let row = base + s * ${c.K * S0}u;
      let d = subgroupMatrixLoad<${left}>(&nnDeltas, row + sg * ${c.M}u, true, ${S0}u);
${rights}
    }
    if (isOut) {
${storeOut}
    } else {
      let dst = split * NN_WEIGHTS + NN_LAYER_OFFSET[l] + sg * ${c.M * W}u;
${storeHidden}
    }
  }
  if (isOut) {
    workgroupBarrier();
${laneLoop(shape.outputs * W, wg, `nnPartials[split * NN_WEIGHTS + ${outOffset}u + i] = nnOutTile[i];`)}
  }
}
`;
}
