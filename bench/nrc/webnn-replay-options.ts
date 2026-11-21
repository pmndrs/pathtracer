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
