/**
 * V2.5a: variance-guided stop for reference renders (`integrator.volumes.outputs`). Pure logic (no GPU): the target math and the batch schedule.
 * `PtMegakernel.renderUntil` drives it with the kernel; tests drive it with a fake.
 */
import type { PtVolumeOutputs } from '../volume/volume.ts';

export interface PtStopTarget {
  /** Stop when the relative standard error of the IMAGE mean is below this: sqrt(sum_i se_i^2) / sum_i mean_i (pixels independent; see `measure`). */
  meanRelSe?: number;
  /** Stop when the `quantile` (default 0.99) of the per-pixel relative standard error se / max(mean, floor) is below `below`. */
  pixelRelSe?: { below: number; quantile?: number; floor?: number };
  /**
   * How the three channels become one number. `'luminance'` (default): Rec.709 luminance of the mean and of the per-channel errors, summed linearly (the errors of the
   * channels are correlated, so the sum is an upper bound of the luminance error: conservative). `'worst-channel'`: each channel on its own, the largest ratio counts.
   */
  measure?: 'luminance' | 'worst-channel';
}

export interface PtStopOptions extends PtStopTarget {
  /** Samples of the first batch (default 64). Each later batch doubles the total (64, 128, 256, ...), so the readbacks stay a fixed number of log2 steps. */
  firstBatch?: number;
  /** Hard cap on the total sample count (default 16384). The last batch is clipped to it. */
  maxSamples?: number;
  /** Cap on one submit (default 8 passes), as the bench adapter's `passesPerSubmit`. */
  passesPerSubmit?: number;
  /**
   * V2.5b: per-pixel adaptive sampling (needs `pixelRelSe`, a kernel built with `volumes.adaptive`). After every batch a pixel whose relative SE is below `pixelRelSe.below`
   * and that has at least `minSamples` samples is flagged done and stops being traced. Batches are then constant (`firstBatch`, default = `minSamples`) so the flags can follow the
   * error closely; the run ends when the share of unfinished pixels is at most `1 - quantile` (the `pixelRelSe` quantile semantics) or at `maxSamples` (per pixel).
   * BIAS: a pixel stopped on its own estimate is stopped preferentially at a noise realisation that read LOW (a stopping rule that correlates with the error), which biases its mean slightly
   * low and its SE estimate low; `minSamples` (default 64) keeps the SE estimate itself reliable before a pixel may stop. Judge the final image against a uniform render of the same target.
   */
  adaptive?: boolean;
  /** Fewest samples a pixel gets before it may be flagged done (adaptive only, default 64). */
  minSamples?: number;
  /**
   * V2.5c (adaptive only): the relative-SE test must hold for this many consecutive batches before a pixel is flagged (default 1). A heavy-tailed pixel that has not yet seen its rare
   * contribution reads a low SE for a batch or two, so waiting longer lets the rare event arrive (reduces, does not remove, the stopping bias).
   */
  consecutive?: number;
  /** V2.5c (adaptive only): a pixel is also held back while its largest single-batch luminance mean exceeds `maxBatchRatio` x its running mean (default off). A tail event makes one batch stand out. */
  maxBatchRatio?: number;
  /**
   * V2.5c (adaptive only): per-pixel sample budgets instead of the SE test: a pixel is flagged once it has `budgets[i]` samples. Unbiased when the budgets come from an INDEPENDENT pilot
   * (`ptAllocateBudgets`), because the stop no longer depends on the samples that form the estimate. `ptRenderAllocated` does pilot + fresh render.
   */
  budgets?: ArrayLike<number>;
}

export interface PtStopMeasure {
  /** Relative standard error of the image mean. */
  meanRelSe: number;
  /** The requested quantile of the per-pixel relative standard error (`pixelRelSe` target only; NaN otherwise). */
  pixelRelSe: number;
}

export interface PtStopStep extends PtStopMeasure {
  samples: number;
  /** Wall time of the batch's encode + submit + wait, ms (readback excluded). */
  batchMs: number;
}

export interface PtRenderUntilResult {
  /** Total samples in the accumulator. */
  samples: number;
  /** Whether every given target was met (false = the cap ended the run). */
  met: boolean;
  reason: 'target' | 'cap';
  /** The last measurement. */
  achieved: PtStopMeasure;
  /** One entry per batch: samples so far and the measurement after it. */
  history: PtStopStep[];
  /** The outputs of the last readback (the final image). */
  outputs: PtVolumeOutputs;
  /** Adaptive runs: samples per pixel (min / mean / max), the total traced (sum over pixels) and the share of pixels flagged done. */
  perPixel?: PtPerPixelStats;
}

export interface PtPerPixelStats { min: number; mean: number; max: number; total: number; doneFraction: number }

/** Statistics of the per-pixel sample counts. */
export function ptPerPixelStats(out: PtVolumeOutputs, done?: Uint8Array): PtPerPixelStats {
  const c = out.sampleCount;
  let min = Infinity, max = 0, total = 0, nd = 0;
  for (let i = 0; i < c.length; i++) {
    const x = c[i] as number;
    if (x < min) min = x;
    if (x > max) max = x;
    total += x;
    if (done && done[i]) nd++;
  }
  return { min: c.length ? min : 0, mean: c.length ? total / c.length : 0, max, total, doneFraction: done && c.length ? nd / c.length : 0 };
}

/**
 * The adaptive flag logic (V2.5b, pure): a pixel is done once it has `minSamples` samples and its relative SE (the `measure` of `ptStopMeasure`: luminance se / max(mean, floor), or the
 * worst channel) is below `below`. Flags are sticky (`prev`). Returns the flags and how many are set.
 */
export function ptAdaptiveFlags(out: PtVolumeOutputs, target: PtStopTarget, minSamples: number, prev?: Uint8Array, rule?: PtAdaptiveRule): { done: Uint8Array; count: number } {
  const q = target.pixelRelSe;
  if (!q) throw new Error('ptAdaptiveFlags: needs pixelRelSe');
  const need = Math.max(1, Math.floor(rule?.consecutive ?? 1));
  const st = rule?.state;
  const n = out.width * out.height;
  const worst = target.measure === 'worst-channel';
  const floor = q.floor ?? 1e-3;
  const done = prev ? prev.slice() : new Uint8Array(n);
  let count = 0;
  for (let i = 0; i < n; i++) {
    if (!done[i] && (out.sampleCount[i] as number) >= minSamples) {
      let mL = 0, sL = 0, worstPx = 0;
      const cnt = out.sampleCount[i] as number;
      for (let c = 0; c < 3; c++) {
        const m = out.radiance[i * 3 + c] as number, s = out.stdError[i * 3 + c] as number;
        mL += (W[c] as number) * m;
        sL += (W[c] as number) * s;
        worstPx = Math.max(worstPx, s / Math.max(m, floor));
      }
      const rel = worst ? worstPx : sL / Math.max(mL, floor);
      let ok = rel < q.below;
      if (st) {
        // luminance mean of the samples added since the last call (batch mean)
        const added = cnt - st.prevCount[i]!;
        if (added > 0) {
          const bm = (mL * cnt - st.prevSum[i]!) / added;
          if (bm > st.maxBatch[i]!) st.maxBatch[i] = bm;
        }
        st.prevSum[i] = mL * cnt;
        st.prevCount[i] = cnt;
        if (ok && rule?.maxBatchRatio !== undefined && st.maxBatch[i]! > rule.maxBatchRatio * Math.max(mL, floor)) ok = false;
        st.streak[i] = ok ? Math.min(255, st.streak[i]! + 1) : 0;
        ok = st.streak[i]! >= need;
      }
      if (ok) done[i] = 1;
    }
    if (done[i]) count++;
  }
  return { done, count };
}

/** V2.5c: per-pixel memory of the robust stopping rule (`consecutive`, `maxBatchRatio`). */
export interface PtAdaptiveState { streak: Uint8Array; maxBatch: Float32Array; prevSum: Float32Array; prevCount: Float32Array }
export interface PtAdaptiveRule { consecutive?: number; maxBatchRatio?: number; state?: PtAdaptiveState }
export function ptAdaptiveState(n: number): PtAdaptiveState {
  return { streak: new Uint8Array(n), maxBatch: new Float32Array(n), prevSum: new Float32Array(n), prevCount: new Float32Array(n) };
}

/** Flags for a per-pixel budget: done once the pixel has `budgets[i]` samples. */
export function ptBudgetFlags(out: PtVolumeOutputs, budgets: ArrayLike<number>): { done: Uint8Array; count: number } {
  const n = out.width * out.height;
  if (budgets.length !== n) throw new RangeError(`ptBudgetFlags: ${budgets.length} budgets for ${n} pixels`);
  const done = new Uint8Array(n);
  let count = 0;
  for (let i = 0; i < n; i++) if ((out.sampleCount[i] as number) >= (budgets[i] as number)) { done[i] = 1; count++; }
  return { done, count };
}

export interface PtAllocateOptions {
  /** The relative standard error each pixel should reach (the `pixelRelSe.below` of the target). */
  below: number;
  /** Mean floor of the relative error (default 1e-3). */
  floor?: number;
  /** Budget limits in samples per pixel (default 16 .. 4096). */
  min?: number;
  max?: number;
  /** Samples per pixel of the pilot (`out.samples` when omitted). */
  pilotSamples?: number;
  /** Grow each pixel's error estimate to the max over its 3 x 3 neighbourhood (default true): a pilot pixel that has not yet seen its rare bright sample reads a low variance, its neighbours usually have seen it. */
  dilate?: boolean;
  /** Round budgets up to a multiple of this (the batch size the fresh render uses; default 1). */
  grid?: number;
}

/**
 * V2.5c (pure): the sample budget of every pixel from an independent PILOT: relative SE scales as 1/sqrt(n), so reaching `below` takes n = P (rse_P / below)^2 samples.
 * The budget depends only on the pilot's samples, never on those of the render that uses it, so the render has no stopping bias (only the efficiency of the allocation is at stake).
 */
export function ptAllocateBudgets(pilot: PtVolumeOutputs, o: PtAllocateOptions): Float32Array {
  const w = pilot.width, h = pilot.height, n = w * h;
  const P = o.pilotSamples ?? pilot.samples;
  const floor = o.floor ?? 1e-3, lo = o.min ?? 16, hi = Math.max(lo, o.max ?? 4096), grid = Math.max(1, Math.floor(o.grid ?? 1));
  const rse = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let mL = 0, sL = 0;
    for (let c = 0; c < 3; c++) { mL += (W[c] as number) * (pilot.radiance[i * 3 + c] as number); sL += (W[c] as number) * (pilot.stdError[i * 3 + c] as number); }
    rse[i] = sL / Math.max(mL, floor);
  }
  let src = rse;
  if (o.dilate ?? true) {
    src = new Float32Array(n);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      let m = 0;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const xx = x + dx, yy = y + dy;
        if (xx >= 0 && yy >= 0 && xx < w && yy < h) m = Math.max(m, rse[yy * w + xx] as number);
      }
      src[y * w + x] = m;
    }
  }
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const r = (src[i] as number) / o.below;
    const b = Math.min(hi, Math.max(lo, Number.isFinite(r) ? P * r * r : hi));
    out[i] = Math.ceil(b / grid) * grid;
  }
  return out;
}

/** What `ptRenderAllocated` needs from a kernel: `renderUntil` and a way to start a fresh image with a new seed. */
export interface PtAllocKernel {
  renderUntil(options: PtStopOptions): Promise<PtRenderUntilResult>;
  restart(seed: number): Promise<void>;
}

export interface PtAllocatedOptions extends PtStopOptions {
  /** Pilot samples per pixel (default 64). */
  pilotSamples?: number;
  /** Budget limits (default minSamples .. maxSamples). */
  minSamples?: number;
  dilate?: boolean;
  /** Seed of the fresh render (the pilot runs on whatever seed the kernel has; this one must differ). */
  seed: number;
  /** Multiplier on the allocated budgets (default 1; > 1 buys margin against a pilot that under-reads the tails). */
  budgetScale?: number;
}

export interface PtAllocatedResult extends PtRenderUntilResult {
  pilotSamplesPerPixel: number;
  /** Fresh-render samples traced (sum over pixels) + the pilot's, per pixel count = film size x pilotSamples. */
  totalWithPilot: number;
  budgets: Float32Array;
}

/**
 * V2.5c: "allocate, then render fresh". A pilot pass (uniform, `pilotSamples` per pixel) measures each pixel's variance; budgets follow (`ptAllocateBudgets`); the kernel is restarted on a new
 * seed and rendered with per-pixel budgets. The returned image uses only the fresh samples, so it is independent of the budgets (no stopping bias). Needs a kernel built with `volumes.adaptive`.
 */
export async function ptRenderAllocated(kernel: PtAllocKernel, options: PtAllocatedOptions): Promise<PtAllocatedResult> {
  const q = options.pixelRelSe;
  if (!q) throw new Error('ptRenderAllocated: needs pixelRelSe');
  const P = Math.max(2, Math.floor(options.pilotSamples ?? 64));
  const lo = Math.max(2, Math.floor(options.minSamples ?? 16));
  const hi = options.maxSamples ?? 4096;
  const batch = Math.max(1, Math.floor(options.firstBatch ?? 16));
  const pilot = await kernel.renderUntil({ pixelRelSe: { below: 0, ...(q.floor !== undefined ? { floor: q.floor } : {}) }, firstBatch: P, maxSamples: P });
  const px = pilot.outputs.width * pilot.outputs.height;
  const raw = ptAllocateBudgets(pilot.outputs, { below: q.below, ...(q.floor !== undefined ? { floor: q.floor } : {}), min: lo, max: hi, pilotSamples: P, ...(options.dilate !== undefined ? { dilate: options.dilate } : {}), grid: batch });
  const scale = options.budgetScale ?? 1;
  const budgets = scale === 1 ? raw : raw.map((b) => Math.min(Math.ceil((hi) / batch) * batch, Math.ceil((b * scale) / batch) * batch));
  await kernel.restart(options.seed);
  let maxBudget = 0;
  for (const b of budgets) if (b > maxBudget) maxBudget = b;
  const r = await kernel.renderUntil({ pixelRelSe: q, adaptive: true, budgets, minSamples: 2, firstBatch: batch, maxSamples: maxBudget, ...(options.measure ? { measure: options.measure } : {}), ...(options.passesPerSubmit ? { passesPerSubmit: options.passesPerSubmit } : {}) });
  return { ...r, pilotSamplesPerPixel: P, totalWithPilot: (r.perPixel?.total ?? 0) + P * px, budgets };
}

const W = [0.2126, 0.7152, 0.0722] as const;

/** The measurements of `out` for a target (`pixelRelSe` is NaN unless the target asks for it). */
export function ptStopMeasure(out: PtVolumeOutputs, target: PtStopTarget): PtStopMeasure {
  const n = out.width * out.height;
  const worst = target.measure === 'worst-channel';
  const sum = [0, 0, 0], se2 = [0, 0, 0];
  let sumL = 0, se2L = 0;
  const q = target.pixelRelSe;
  const floor = q?.floor ?? 1e-3;
  const rel = q ? new Float32Array(n) : null;
  for (let i = 0; i < n; i++) {
    let mL = 0, sL = 0, worstPx = 0;
    for (let c = 0; c < 3; c++) {
      const m = out.radiance[i * 3 + c] as number, s = out.stdError[i * 3 + c] as number;
      sum[c]! += m;
      se2[c]! += s * s;
      mL += (W[c] as number) * m;
      sL += (W[c] as number) * s;
      worstPx = Math.max(worstPx, s / Math.max(m, floor));
    }
    sumL += mL;
    se2L += sL * sL;
    if (rel) rel[i] = worst ? worstPx : sL / Math.max(mL, floor);
  }
  let meanRelSe: number;
  if (worst) {
    meanRelSe = 0;
    for (let c = 0; c < 3; c++) meanRelSe = Math.max(meanRelSe, (sum[c] as number) > 0 ? Math.sqrt(se2[c] as number) / (sum[c] as number) : Infinity);
  } else meanRelSe = sumL > 0 ? Math.sqrt(se2L) / sumL : Infinity;
  let pixelRelSe = NaN;
  if (rel && q) {
    rel.sort();
    pixelRelSe = n > 0 ? (rel[Math.min(n - 1, Math.floor((q.quantile ?? 0.99) * n))] as number) : Infinity;
  }
  return { meanRelSe, pixelRelSe };
}

/** Whether `m` meets every target that is set (at least one must be). */
export function ptStopMet(m: PtStopMeasure, target: PtStopTarget): boolean {
  if (target.meanRelSe === undefined && !target.pixelRelSe) throw new Error('ptStopMet: set meanRelSe and/or pixelRelSe');
  return (target.meanRelSe === undefined || m.meanRelSe < target.meanRelSe) && (!target.pixelRelSe || m.pixelRelSe < target.pixelRelSe.below);
}

/**
 * Batch sizes of one call: `first`, then each batch is as big as everything rendered so far in the call, so the totals go first, 2 first, 4 first, ...
 * The last batch is clipped so that `done + sum` never passes `max`.
 */
export function ptStopBatches(first: number, max: number, done = 0): number[] {
  const out: number[] = [];
  let rendered = 0;
  while (done + rendered < max) {
    const want = rendered === 0 ? Math.max(1, Math.floor(first)) : rendered;
    const n = Math.min(want, max - done - rendered);
    out.push(n);
    rendered += n;
  }
  return out;
}

/** What the loop needs from a kernel. */
export interface PtStopDriver {
  /** Encodes, submits and waits for `n` more sample passes. */
  run(n: number): Promise<void>;
  /** Reads the outputs of everything accumulated so far (outside any timed region). */
  read(): Promise<PtVolumeOutputs>;
  now?(): number;
  /** Adaptive runs: writes the done flags (1 = stop tracing the pixel) into the accumulator. */
  markDone?(done: Uint8Array): Promise<void>;
}

export async function ptRenderUntil(driver: PtStopDriver, options: PtStopOptions, startSamples = 0): Promise<PtRenderUntilResult> {
  if (options.meanRelSe === undefined && !options.pixelRelSe) throw new Error('renderUntil: set meanRelSe and/or pixelRelSe');
  const now = driver.now ?? (() => performance.now());
  const history: PtStopStep[] = [];
  const max = Math.max(options.maxSamples ?? 16384, startSamples + 1);
  let samples = startSamples;
  let out: PtVolumeOutputs | null = null;
  let m: PtStopMeasure = { meanRelSe: Infinity, pixelRelSe: NaN };
  let met = false;
  if (options.adaptive) return ptRenderUntilAdaptive(driver, options, startSamples, now);
  for (const batch of ptStopBatches(options.firstBatch ?? 64, max, startSamples)) {
    const t0 = now();
    await driver.run(batch);
    const batchMs = now() - t0;
    samples += batch;
    out = await driver.read();
    m = ptStopMeasure(out, options);
    history.push({ samples, ...m, batchMs });
    if (samples >= 2 && ptStopMet(m, options)) { met = true; break; }
  }
  if (!out) throw new Error('renderUntil: nothing rendered');
  return { samples, met, reason: met ? 'target' : 'cap', achieved: m, history, outputs: out };
}

async function ptRenderUntilAdaptive(driver: PtStopDriver, options: PtStopOptions, startSamples: number, now: () => number): Promise<PtRenderUntilResult> {
  if (!options.pixelRelSe) throw new Error('renderUntil: adaptive needs pixelRelSe');
  if (!driver.markDone) throw new Error('renderUntil: adaptive needs a kernel built with integrator.volumes.adaptive');
  const minSamples = Math.max(2, Math.floor(options.minSamples ?? 64));
  const batch = Math.max(1, Math.floor(options.firstBatch ?? minSamples));
  const max = Math.max(options.maxSamples ?? 16384, startSamples + 1);
  const quantile = options.pixelRelSe.quantile ?? 0.99;
  const history: PtStopStep[] = [];
  let samples = startSamples;
  let done: Uint8Array | undefined;
  let out: PtVolumeOutputs | null = null;
  let m: PtStopMeasure = { meanRelSe: Infinity, pixelRelSe: NaN };
  let met = false;
  let rule: PtAdaptiveRule | undefined;
  const robust = (options.consecutive ?? 1) > 1 || options.maxBatchRatio !== undefined;
  while (samples < max) {
    const n = Math.min(batch, max - samples);
    const t0 = now();
    await driver.run(n);
    const batchMs = now() - t0;
    samples += n;
    out = await driver.read();
    m = ptStopMeasure(out, options);
    history.push({ samples, ...m, batchMs });
    if (robust && !rule) rule = { ...(options.consecutive !== undefined ? { consecutive: options.consecutive } : {}), ...(options.maxBatchRatio !== undefined ? { maxBatchRatio: options.maxBatchRatio } : {}), state: ptAdaptiveState(out.width * out.height) };
    const f = options.budgets ? ptBudgetFlags(out, options.budgets) : ptAdaptiveFlags(out, options, minSamples, done, rule);
    done = f.done;
    if (f.count / Math.max(1, done.length) >= (options.budgets ? 1 : quantile)) { met = true; break; }
    await driver.markDone(done);
  }
  if (!out) throw new Error('renderUntil: nothing rendered');
  return { samples, met, reason: met ? 'target' : 'cap', achieved: m, history, outputs: out, perPixel: ptPerPixelStats(out, done) };
}
