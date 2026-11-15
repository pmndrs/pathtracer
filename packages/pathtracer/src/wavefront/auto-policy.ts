/**
 * P3.9: the auto policy of the scheduling settings, as code. One pure function from facts that are known before the first frame (film size, depth limit,
 * device features) to the scheduler and its settings: `scheduler`, `refill`, `layers`, `appends`, `sort`. The adapter / engine passes the result to
 * `PtMegakernel.create` or `PtWavefront.create`; every field stays an explicit setting (D11: nothing here removes a switch), `auto` is only a way to pick them.
 *
 * The rules come from paired ratios (wavefront / megakernel, same page, ABBA, 5 rounds, M5 Pro, load 4-20 from other agents: PROVISIONAL, TASKS P3.3 / P3.10, DIVERGES V110-V112;
 * `node bench/runner/run.ts --page wf-sched-ab`), the quiet-pass queue re-measures them:
 *  - paths of at most 2 vertices (primary + one NEE, the real-time regime): the megakernel (quad-light, maxDepth 1: every wavefront flavour 1.6-2.4x slower);
 *  - a big film (>= 2^19 pixels, Sponza 1280x720): one path per pixel fills the GPU; the plain scheduler is as good as `batch` (0.73-0.80 of the megakernel) and has no `finish()` drain;
 *    inline refill loses coherence there (0.93-1.07);
 *  - a mid film (>= 2^18, Cornell 512x512, deep paths): inline refill, one layer (0.53-0.55 against 0.59-0.61 plain); more layers did not help;
 *  - a small film leaves the GPU empty (env-sphere 384x256: wavefront 1.4-1.6x slower at one layer): inline refill with `layers` until ~2 M slots (0.68-0.73 at L4, ~0.3-0.5 at L16);
 *  - subgroup-aggregated appends whenever the device has the feature (never measurably slower; 4-8% faster under inline refill on Sponza / foliage);
 *  - block sorts and `ordered` restart are NOT picked: the sorts cost 3-15% on every scene measured (single-material or low divergence), `ordered` beat inline only on Sponza
 *    (0.85-0.98 vs 0.93-1.07) and lost to plain / batch there and to inline elsewhere.
 * Anything the policy cannot know (a scene with many divergent materials, say) is the caller's to override: spread the result, then your own fields.
 */
export interface PtwfAutoInput {
  width: number;
  height: number;
  /** The path vertex limit (`maxDepth`). */
  maxDepth: number;
  /** Device has the `subgroups` feature (`caps.subgroups`). */
  subgroups: boolean;
}

export interface PtwfAutoChoice {
  scheduler: 'megakernel' | 'wavefront';
  refill: false | 'inline' | 'batch' | 'ordered';
  layers: number;
  appends: 'atomic' | 'subgroup';
  sort: 'none' | 'material' | 'octant' | 'both';
  /** One line: which rule fired (shown by the bench adapter in its config). */
  reason: string;
}

/** Slots (paths in flight) the policy aims for on a small film. */
export const PTWF_AUTO_TARGET_SLOTS = 1 << 21;
/** Films of at least this many pixels fill the GPU with one path per pixel (no refill). */
export const PTWF_AUTO_BIG_FILM = 1 << 19;
/** Films of at least this many pixels keep one layer (inline refill, no extra slots). */
export const PTWF_AUTO_MID_FILM = 1 << 18;
/** Paths this short (`maxDepth <=`) are not worth the wavefront's dispatches. */
export const PTWF_AUTO_SHORT_PATH = 2;

/** `layers` for a small film: the largest power of two with `pixels * layers <= PTWF_AUTO_TARGET_SLOTS / 2` (so the slots land in (1 M, 2 M]), in 1..32 (above 32 never helped: TASKS P3.2 sweep). */
export function ptwfAutoLayers(pixels: number): number {
  let layers = 1;
  while (layers < 32 && pixels * layers * 2 <= PTWF_AUTO_TARGET_SLOTS) layers *= 2;
  return layers;
}

export function ptwfAutoPolicy(input: PtwfAutoInput): PtwfAutoChoice {
  const pixels = input.width * input.height;
  if (!(pixels > 0) || !Number.isFinite(pixels)) throw new RangeError(`ptwfAutoPolicy: bad film ${input.width}x${input.height}`);
  const appends = input.subgroups ? ('subgroup' as const) : ('atomic' as const);
  if (input.maxDepth <= PTWF_AUTO_SHORT_PATH) return { scheduler: 'megakernel', refill: false, layers: 1, appends: 'atomic', sort: 'none', reason: `maxDepth ${input.maxDepth} <= ${PTWF_AUTO_SHORT_PATH}: too little work per dispatch` };
  if (pixels >= PTWF_AUTO_BIG_FILM) return { scheduler: 'wavefront', refill: false, layers: 1, appends, sort: 'none', reason: `${pixels} pixels fill the GPU: plain wavefront` };
  if (pixels >= PTWF_AUTO_MID_FILM) return { scheduler: 'wavefront', refill: 'inline', layers: 1, appends, sort: 'none', reason: `${pixels} pixels: inline refill, one slot per pixel` };
  const layers = ptwfAutoLayers(pixels);
  return { scheduler: 'wavefront', refill: 'inline', layers, appends, sort: 'none', reason: `${pixels} pixels leave the GPU empty: inline refill with ${layers} layers (${pixels * layers} slots)` };
}
