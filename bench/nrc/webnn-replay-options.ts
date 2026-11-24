/**
 * Hand-off variants of the replay (pmndrs/pathtracer WebNN sync study, bench/results/NRC-WEBNN-SYNC.md). Each default is the
 * sequence the November 19 numbers were measured with, so `--ab <key>=<default>,<other>` pairs one change against it.
 *  - `ctx`: `type` = `createContext({ deviceType })`; `device` = `createContext(gpuDevice)`, the explainer's best case for zero-copy export.
 *  - `tensors`: `shared` = one input / output tensor reused chunk by chunk; `chunk` = a pair per chunk, so every export, copy, return and
 *    dispatch of the frame is issued as one batch (about 5 waits per frame instead of 3 per chunk + 2).
 *  - `sync`: `drain` = wait for the WebGPU queue after returning the input and before `dispatch`; `none` = trust `destroy()` to order it
 *    (it did not on Chrome 157 / Mac, research/notes/13-radiance-cache.md: run it with `parity=report`).
 *  - `route`: `export` = WebNN tensors lent to WebGPU (`exportToGPU`); `cpu` = through script: `mapAsync` the features and weights,
 *    `writeTensor`, `readTensor`, `queue.writeBuffer` (the path the interop API exists to avoid; also the only one that runs where
 *    exportable tensors lose the context, such as SwiftShader on Linux).
 *  - `parity`: `strict` throws on stale or wrong predictions; `report` records them (`parity_failures`) and still times the arm.
 *  - `fold` (1, 2, 4, 8; default 1 = off): an exportable tensor is capped at 8,192 rows on macOS (its IOSurface height), so one dispatch
 *    cannot take a whole frame as [rows, 64]. `fold=F` keeps `rows` as the tensor height and makes it F times wider ([rows, F * 64]);
 *    the graph reshapes it to [rows * F, 64] and the result back to [rows, F * 4]. The bytes are the same row-major block, so
 *    `rows=4096 fold=8` is one dispatch of 32,768 queries. With `fold=1` no reshape is added and the graph is the earlier one.
 *  - `breakdown=1`: per-phase milliseconds (`webnn_phase_<name>_p50_ms`), timestamps around the existing awaits only (no extra waits).
 */
export const WEBNN_REPLAY_OPTIONS = {
  ctx: ['type', 'device'],
  tensors: ['shared', 'chunk'],
  sync: ['drain', 'none'],
  route: ['export', 'cpu'],
  parity: ['strict', 'report'],
} as const;
type ReplayOptions = { -readonly [K in keyof typeof WEBNN_REPLAY_OPTIONS]: (typeof WEBNN_REPLAY_OPTIONS)[K][number] };

/** Tensor and graph shapes of one dispatch for `rows` x `fold` (see `fold` above). `slab` is the number of queries per dispatch. */
export function webnnFoldShapes(rows: number, fold: number, inputs: number): { slab: number; x: number[]; y: number[]; flatX: number[]; flatY: number[] } {
  if (![1, 2, 4, 8].includes(fold)) throw new Error('fold must be 1, 2, 4 or 8');
  return { slab: rows * fold, x: [rows, fold * inputs], y: [rows, fold * 4], flatX: [rows * fold, inputs], flatY: [rows * fold, 4] };
}

/** Parses and cross-checks the hand-off variants (exported for the unit test). */
export function webnnReplayOptions(args: Record<string, string | undefined>): ReplayOptions & { breakdown: boolean } {
  const out = {} as Record<string, string>;
  for (const [key, allowed] of Object.entries(WEBNN_REPLAY_OPTIONS)) {
    const value = args[key] ?? allowed[0];
    if (!(allowed as readonly string[]).includes(value)) throw new Error(`${key} must be one of ${allowed.join(', ')}`);
    out[key] = value;
  }
  const options = { ...(out as ReplayOptions), breakdown: args.breakdown === '1' };
  if (options.route === 'cpu' && options.sync === 'none') throw new Error('sync=none only applies to route=export (script writes are ordered by WebNN itself)');
  if (options.tensors === 'chunk' && args.publish === 'slab') throw new Error('publish=slab needs tensors=shared (a batched frame publishes once)');
  return options;
}
