/**
 * Generated WGSL modules: the inline module (bvh walk + L0 + L1 + optional rtSurface) and the batch kernels (queue ABI
 * v0). All of it goes through the caps preprocessor (`buildWgsl`): `#if` blocks in wgsl-source.ts, caps and layer defines.
 *
 * Kernel size is capped (~50 KB, Windows DXC compile cliff, spec section 5); the tests check every combination.
 */
import { DEFAULT_NODE_FORMAT, TRACER_GPU_LAYOUT_VERSION, traceModuleEntries, traceModuleWgsl } from '@pmndrs/bvh';
import { BVH_LAYOUT_VERSION } from '../contract/ray.ts';
import type { AnyHitOrder, BindingSlot, ModuleBindingEntry, NodeFormat } from '@pmndrs/bvh';
import { buildWgsl } from '../caps/wgsl.ts';
import type { Caps } from '../caps/detect.ts';
import type { Defines } from '../caps/wgsl.ts';
import { DEFAULT_BATCH_WORKGROUP_SIZE, STORAGE_BUFFER_BUDGET, type BatchWorkgroupSize } from './constants.ts';
import { surfaceVertexOffsets, type SurfaceLayout } from './surface-table.ts';
import { RT_CORE_WGSL, rtBatchArgsWgsl, rtBatchWgsl, rtSurfaceWgsl } from './wgsl-source.ts';
import { RT_INTERVAL_WGSL } from './interval.ts';
import { DEFAULT_SCENE_PACKING, SCENE_HEADER_VEC4S, sceneCopyStorageBuffers, type SceneCopyPacking } from '../scene/constants.ts';
import { rtPackSceneAccess, rtSceneWgsl, rtTextureTableEnable } from '../scene/scene-wgsl.ts';
import type { TextureLayout } from '../scene/texture.ts';
import { rtShadeWgsl } from '../shade/shade-wgsl.ts';
import type { RtShadeOptions } from '../shade/shade.ts';
import { rtComposeSourceMaps, rtWgslIdentityMap, type RtWgslBuild } from './wgsl-source-map.ts';

export type { AnyHitOrder, BindingSlot, ModuleBindingEntry, NodeFormat };

type BvhWalkOptions = NonNullable<Parameters<typeof traceModuleWgsl>[0]>;
/** bvh4 walk shape (`'nested'` default | `'single'`, lever L7) and wide plane decode (`'exact'` default | `'folded'`, lever L1); bvh does not export the names. */
export type WideWalk = NonNullable<BvhWalkOptions['wideWalk']>;
export type WideDecode = NonNullable<BvhWalkOptions['wideDecode']>;

/** The experimental wide node formats of bvh (X1.i, D25): `bvh4` and `cwbvh8`. */
export function rtIsWideNodeFormat(format: NodeFormat | undefined): boolean {
  return format === 'bvh4' || format === 'cwbvh8';
}

/**
 * The scene's formats, compiled into the walk. Read them off the buffers with `sceneFormatOf`. `nodeFormat` is bvh's:
 * `'quantized'` (default), `'f32'`, or the experimental wide `'bvh4'` / `'cwbvh8'` (X1.i; the scene must have been assembled with the same
 * `nodeFormat`, see `buildTraceScene({ nodeFormat })`). `wideWalk` (bvh4: `'nested'` | `'single'`) and `wideDecode` (`'exact'` | `'folded'`)
 * are compile options of the wide walks, not properties of the buffers: wide formats default to `wideDecode: 'folded'` (lever L1, D25).
 */
export interface SceneFormat {
  nodeFormat?: NodeFormat;
  triangleLeaves?: 'records' | 'indexed';
  positionLayout?: 'vec4' | 'packed';
  wideWalk?: WideWalk;
  wideDecode?: WideDecode;
}

/** `sceneFormatOf`'s result: the three buffer formats are always set, the wide walk options only when given or defaulted. */
export type ResolvedSceneFormat = Required<Pick<SceneFormat, 'nodeFormat' | 'triangleLeaves' | 'positionLayout'>> & Pick<SceneFormat, 'wideWalk' | 'wideDecode'>;

/**
 * The formats of bvh's `TraceSceneBuffers` (`SceneTrees.traceBuffers`, `uploadTraceScene`). Absent fields mean bvh's
 * defaults: quantized nodes (`DEFAULT_NODE_FORMAT`), record leaves, vec4 positions. A wide node format gets `wideDecode: 'folded'`
 * unless the caller names one.
 */
export function sceneFormatOf(buffers: SceneFormat): ResolvedSceneFormat {
  const nodeFormat = buffers.nodeFormat ?? DEFAULT_NODE_FORMAT;
  const wide = rtIsWideNodeFormat(nodeFormat);
  const wideDecode = buffers.wideDecode ?? (wide ? 'folded' : undefined);
  return {
    nodeFormat,
    triangleLeaves: buffers.triangleLeaves ?? 'records',
    positionLayout: buffers.positionLayout ?? 'vec4',
    ...(wide && buffers.wideWalk ? { wideWalk: buffers.wideWalk } : {}),
    ...(wideDecode ? { wideDecode } : {}),
  };
}

export interface RtModuleOptions extends SceneFormat {
  /** Where bvh's walk bindings land. Default group 0, binding 0. */
  traceSlot?: BindingSlot;
  /**
   * Include the scene copy (D4): `rtSurface` (unless `surface: false`), `rtMaterial` and the light functions, over the scene
   * copy's two bindings at `slot` (`rtSceneEntries`). `layout` is `SceneCopy.layout`. Default: no scene, no `rtSurface`.
   */
  scene?: { slot: BindingSlot; layout: SurfaceLayout; surface?: boolean; materials?: boolean; lights?: boolean; sampling?: boolean; /** Material textures (P1.4c): `SceneCopy.textureLayout`. Adds `rtTexSample`, `rtApplyMaterialTextures`, `rtApplyNormalMap` and the `RT_SURF_TEXFRAME` surface data; needs materials, uvs and positions in the layout. */ textures?: TextureLayout | null; /** `split` (two bindings) or `packed` (one: `rtScene`, P3.11; default `DEFAULT_SCENE_PACKING`, D32); upload with the same `uploadSceneCopy({ packing })`. */ packing?: SceneCopyPacking } | false;
  /**
   * Add L2 `rtShade` (P1.3, `shade/`): one-bounce shading with a hook for the indirect term. Needs `scene` with surface, materials and lights;
   * it adds the light point sampling (`rtSampleLightLi`, as `scene.sampling`), so do not append `rtLightSampleWgsl()` yourself. `true` = defaults.
   */
  shade?: RtShadeOptions | boolean;
  /**
   * Remove `//` and block comments from the generated text (line breaks stay, so compiler line numbers still match). The bvh
   * walk is 40 KB of mostly comments; with the scene copy the module would pass the ~50 KB kernel cap without this.
   * Default: true.
   */
  stripComments?: boolean;
  /**
   * Also drop the indentation and the spaces around `{ } ( ) , ;` (line breaks stay). About 17% of the text, applied last, to the
   * finished module only (append your own code through `rtMinifyWgsl` if you want it small too). Default false.
   * With `shade` and indexed leaves the module is 50.2 KB, over the ~50 KB kernel cap: turn this on (about 42.2 KB).
   */
  minify?: boolean;
  /** Instance masks against ray masks (bvh's `rayMasks`). Default true. */
  rayMasks?: boolean;
  /**
   * Add the interval queries (P5.rc1, `interval.ts`): `rtClosestInterval(origin, direction, t0, t1, mask)` over the half-open `(t0, t1]`,
   * `rtClosestIntervalLo` over `[t0, t1)`, the `rtVisibleInterval` any-hit forms, the cascade arithmetic and `rtOctEqualAreaDecode`.
   * Default false: the text is unchanged (kernel bytes identical).
   */
  intervals?: boolean;
  /** bvh's `bvhTrianglePasses`; default every triangle blocks, `null` to declare it yourself. */
  triangleFilter?: string | null;
  /** Any-hit child order (bvh's `anyHitOrder`). Default 'far'. */
  anyHitOrder?: AnyHitOrder;
}

/** Spec section 3: check the bvh layout version before generating any walk. Throws on a bvh build we were not written for. */
export function assertBvhLayout(version: number = TRACER_GPU_LAYOUT_VERSION): void {
  if (version !== BVH_LAYOUT_VERSION) {
    throw new Error(`@pmndrs/raytracer expects bvh GPU layout v${BVH_LAYOUT_VERSION}, the installed @pmndrs/bvh walk is v${version}`);
  }
}

/** The comment bvh's walk functions end with (each leaves only through a `return` inside its endless `loop`). */
const WALK_TAIL = "// No return here: the loop above only exits through 'return'.";
/** What each walk function returns once its pop loop finds the stack empty, in the order the functions appear. */
const WALK_RETURNS = ['return bvhFinishClosest(result, hitRecord, hitSlot);', 'return false;'];

/**
 * P-W1 (Windows/NVIDIA): rewrites the descent and pop `loop {}` of bvh's two walk functions as `for` loops with a constant bound,
 * the only loop shape Tint's PreventInfiniteLoops leaves alone (index declared in the for-init, `< constant`, `++`). Every other
 * loop gets an injected 64-bit iteration counter; inside the megakernel (both walks inlined into the bounce loop) the counters
 * of these two loops made DXC's SM 6.6+ code on an RTX 3060 3.7x slower (Sponza, Chrome D3D12: 66 -> 17.7 ms per sample at depth
 * 1; Vulkan, Metal and the wavefront kernels never paid it). Same paths: the bound, 2^32 - 1 iterations per loop, is never reached
 * by a walk that terminates. Returns the text unchanged when bvh's walk is not the expected shape (`rtBoundWalkLoopsApplies`;
 * a unit test checks it applies to every walk variant, so a bvh change that breaks the pattern fails the tests, not a user).
 */
export function rtBoundWalkLoops(walk: string): string {
  if (!rtBoundWalkLoopsApplies(walk)) return walk;
  let index = 0;
  const bounded = walk.replace(/\bloop \{/g, () => {
    const step = `rtWalkStep${index++}`; // unique per walk loop (rtShortenWgsl wants unique names)
    return `for (var ${step} = 0u; ${step} < 0xffffffffu; ${step}++) {`;
  });
  if (bounded.includes(WALK_TAIL)) {
    const [head = '', ...rest] = bounded.split(WALK_TAIL);
    return rest.reduce((out, part, i) => out + WALK_RETURNS[i] + part, head);
  }
  // Wide walks (bvh4 / cwbvh8, X1.i): the helper loops end in `break` and already `return` after the loop; the two walk functions end with the loop, so each gets its final return.
  return WIDE_WALK_FUNCTIONS.reduce((out, [name, ret]) => {
    const start = out.indexOf(`fn ${name}(`);
    const end = out.indexOf('\n    }\n}\n', start);
    return out.slice(0, end + 7) + `    ${ret}\n` + out.slice(end + 7);
  }, bounded);
}

/** The wide walks' two entry functions (each ends with its endless descent loop) and the return the rewrite appends. */
const WIDE_WALK_FUNCTIONS: ReadonlyArray<readonly [string, string]> = [
  // Unreachable (the loop only leaves through its own returns, and 2^32 - 1 iterations are never run), so the miss record serves whatever the walk calls its locals (bvh4 `single` has its own).
  ['bvhWalkClosest', 'return BvhHit(-1.0, 0u, 0u, vec2f(0.0), true, BVH_NO_MATERIAL, vec3f(0.0));'],
  ['bvhWalkAny', 'return false;'],
];

/**
 * True when `rtBoundWalkLoops` recognises bvh's walk: the binary walk (four `loop`s, two functions ending in WALK_TAIL) or a wide walk
 * (bvh4 / cwbvh8: every `loop` rewritable, both walk functions ending with the loop). Anything else is left alone.
 */
export function rtBoundWalkLoopsApplies(walk: string): boolean {
  const loops = (walk.match(/\bloop \{/g) ?? []).length;
  if (loops === 0 || !/\bfn bvhWalkClosest\b[\s\S]*\bfn bvhWalkAny\b/.test(walk)) return false;
  if (walk.includes(WALK_TAIL)) return loops === 4 && walk.split(WALK_TAIL).length === 3;
  return WIDE_WALK_FUNCTIONS.every(([name]) => {
    const start = walk.indexOf(`fn ${name}(`);
    const end = start < 0 ? -1 : walk.indexOf('\n    }\n}\n', start);
    return end > 0 && !walk.slice(start, end).includes('\n}\n') && /\bloop \{/.test(walk.slice(start, end));
  });
}

function bvhWalk(options: RtModuleOptions & { traceSlot: BindingSlot }): string {
  assertBvhLayout();
  const format = sceneFormatOf(options);
  const walk = traceModuleWgsl({
    traceSlot: options.traceSlot,
    rayMasks: options.rayMasks ?? true,
    triangleFilter: options.triangleFilter,
    anyHitOrder: options.anyHitOrder,
    triangleLeaves: format.triangleLeaves,
    positionLayout: format.positionLayout,
    nodeFormat: format.nodeFormat,
    ...(format.wideWalk ? { wideWalk: format.wideWalk } : {}),
    ...(format.wideDecode ? { wideDecode: format.wideDecode } : {}),
  });
  return rtBoundWalkLoops(walk);
}

/** Removes `//` line comments and block comments, keeping every line break (WGSL has no string literals, so no `//` is ever text). */
export function stripWgslComments(source: string): string {
  return source.replace(/\/\*[\s\S]*?\*\//g, (block) => block.replace(/[^\n]/g, '')).replace(/[ \t]*\/\/[^\n]*/g, '');
}

/** Indentation and the spaces around `{ } ( ) , ;` removed; line breaks kept. Only safe on comment-free, preprocessed source (no `#` lines). */
export function rtMinifyWgsl(code: string): string {
  return code.replace(/^[ \t]+/gm, '').replace(/[ \t]*([{}(),;])[ \t]*/g, '$1');
}

/**
 * Drops top-level `struct`/`fn`/`const`/`alias`/`override`/`var` declarations no entry point or binding reaches (a call-graph walk by
 * identifier). Line breaks of what stays are untouched, but a dropped declaration removes its lines, so error line numbers after
 * it shift: run it on the source you compile, not on a copy used to read compiler messages (or keep `prune: false`).
 * Entry points are declarations preceded by `@compute`/`@vertex`/`@fragment`; every `@group` binding is kept (the pipeline layout
 * names it). Needs comment-free, preprocessed, NOT yet minified source (declarations start in column 0). Returns the kept
 * source, the dropped names and `lineMap` (output line -> input line, 0-based; see `RtWgslSourceMap`).
 */
export function rtPruneWgsl(code: string): { code: string; dropped: string[]; lineMap: Int32Array } {
  const declRe = /^(?:@[^\n]*?\s)?(?:struct|fn|const|alias|override|var(?:<[^>]*>)?)\s+([A-Za-z_]\w*)/;
  const lines = code.split('\n');
  type Chunk = { name: string; lines: string[]; src: number[]; root: boolean };
  const chunks: Chunk[] = [{ name: '', lines: [], src: [], root: true }];
  lines.forEach((line, at) => {
    const m = declRe.exec(line);
    if (m) {
      // A lone attribute line (`@compute @workgroup_size(..)`) belongs to the declaration after it.
      const prev = chunks[chunks.length - 1] as Chunk;
      let carry: string[] = [];
      let carrySrc: number[] = [];
      const last = prev.lines[prev.lines.length - 1];
      if (prev.lines.length > 0 && last !== undefined && /^@(compute|vertex|fragment)\b/.test(last) && !declRe.test(last)) {
        carry = [prev.lines.pop() as string];
        carrySrc = [prev.src.pop() as number];
      }
      chunks.push({ name: m[1] as string, lines: [...carry, line], src: [...carrySrc, at], root: false });
    } else {
      const c = chunks[chunks.length - 1] as Chunk;
      c.lines.push(line);
      c.src.push(at);
    }
  });
  for (const c of chunks) if (c.name) c.root = /^@(group|compute|vertex|fragment)\b/.test(c.lines[0] as string);
  const byName = new Map(chunks.filter((c) => c.name).map((c) => [c.name, c]));
  const seen = new Set<string>();
  const stack = chunks.filter((c) => c.root).map((c) => c);
  const visit = (c: { name: string; lines: string[] }): void => {
    for (const id of c.lines.join('\n').match(/[A-Za-z_]\w*/g) ?? []) {
      const d = byName.get(id);
      if (d && !seen.has(id)) { seen.add(id); stack.push(d); }
    }
  };
  while (stack.length) visit(stack.pop() as { name: string; lines: string[] });
  const dropped: string[] = [];
  const out: string[] = [];
  const srcOf: number[] = [];
  for (const c of chunks) {
    if (!c.root && !seen.has(c.name)) dropped.push(c.name);
    else { out.push(...c.lines); srcOf.push(...c.src); }
  }
  return { code: out.join('\n'), dropped, lineMap: Int32Array.from(srcOf) };
}

/**
 * Renames top-level `struct`/`fn`/`const`/`alias`/`var` names to short ones (`A`, `B`, ..., `Aa`, ...), most-used first, in every
 * place the name appears. Line breaks and everything else stay, so error line numbers still match, but compiler messages show the
 * short names: use `names` (old -> new) to read them, or do not shorten while debugging. Saves ~8 KB of the 50 KB megakernel (the
 * names are long: `rtSampleEnvironmentLi`).
 *
 * Never renamed: names in `keep` (entry points the host looks up), `override`s (set by name through pipeline constants), `@group`
 * bindings (readable in tests and tools), and any name also used as a struct member (`.name`), a field or parameter (`name:`), or a
 * `let`/`var`/`const` local. Fresh names avoid every identifier already in the source. Needs comment-free, preprocessed,
 * not yet minified source (declarations start in column 0).
 */
export function rtShortenWgsl(code: string, keep: readonly string[] = []): { code: string; names: Map<string, string> } {
  const declRe = /^(@[^\n]*?\s)?(?:(struct|fn|const|alias|override|var(?:<[^>]*>)?)\s+)([A-Za-z_]\w*)/gm;
  const cand = new Map<string, number>(); // name -> position of its declaring identifier
  const blocked = new Set<string>(keep);
  for (const m of code.matchAll(declRe)) {
    const name = m[3] as string;
    const pos = (m.index as number) + m[0].length - name.length;
    if (m[2] === 'override' || /^@group\b/.test(m[1] ?? '')) blocked.add(name);
    if (cand.has(name)) blocked.add(name);
    cand.set(name, pos);
  }
  const declPositions = new Set(cand.values());
  const count = new Map<string, number>();
  const used = new Set<string>();
  for (const m of code.matchAll(/[A-Za-z_]\w*/g)) {
    const id = m[0];
    used.add(id);
    if (!cand.has(id)) continue;
    count.set(id, (count.get(id) ?? 0) + 1);
    const at = m.index as number;
    if (declPositions.has(at)) continue;
    const before = code.slice(Math.max(0, at - 6), at);
    const after = code.slice(at + id.length, at + id.length + 3);
    if (before.endsWith('.') || /^\s*:/.test(after) || /(?:let|var|const)\s+$/.test(before) || /(?:let|var|const)\s*<[^>]*>\s+$/.test(before)) blocked.add(id);
  }
  const order = [...count.keys()].filter((n) => !blocked.has(n)).sort((a, b) => (count.get(b) as number) * b.length - (count.get(a) as number) * a.length || (a < b ? -1 : 1));
  const letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  const fresh = function* (): Generator<string> {
    for (const c of letters.slice(0, 26)) yield c;
    for (const a of letters.slice(0, 26)) for (const b of letters) yield a + b;
    for (const a of letters.slice(0, 26)) for (const b of letters) for (const c of letters) yield a + b + c;
  }();
  const names = new Map<string, string>();
  for (const old of order) {
    let next = fresh.next().value as string;
    while (used.has(next)) next = fresh.next().value as string;
    if (next.length >= old.length) continue;
    names.set(old, next);
  }
  return { code: code.replace(/[A-Za-z_]\w*/g, (id) => names.get(id) ?? id), names };
}

export interface RtWgslTransformOptions {
  /** `rtPruneWgsl`. Default true. */
  prune?: boolean;
  /** `rtShortenWgsl`. Default true. */
  shorten?: boolean;
  /** `rtMinifyWgsl`. Default true. */
  minify?: boolean;
  /** Names `rtShortenWgsl` must keep (entry points the host looks up). */
  keep?: readonly string[];
}

/**
 * The size pipeline in one call: prune, shorten, minify (each optional), with the source map that undoes it. `code` is the
 * readable, comment-free, preprocessed text (any `enable` lines first); the map's source is that text. Each stage composes its line
 * map onto the previous one. Read compiler messages with `rtMapCompilationInfo(info, map)`.
 */
export function rtTransformWgsl(code: string, options: RtWgslTransformOptions = {}): RtWgslBuild & { dropped: string[] } {
  let map = rtWgslIdentityMap(code);
  let text = code;
  let dropped: string[] = [];
  if (options.prune ?? true) {
    const pruned = rtPruneWgsl(text);
    text = pruned.code;
    dropped = pruned.dropped;
    map = rtComposeSourceMaps(map, { lines: pruned.lineMap });
  }
  if (options.shorten ?? true) {
    const shortened = rtShortenWgsl(text, options.keep ?? []);
    text = shortened.code;
    const names: Record<string, string> = {};
    for (const [old, short] of shortened.names) names[short] = old;
    map = rtComposeSourceMaps(map, { names });
  }
  if (options.minify ?? true) text = rtMinifyWgsl(text); // keeps every line break: the line map is unchanged
  return { code: text, map, dropped };
}

function surfaceLayoutOf(options: RtModuleOptions): SurfaceLayout | null {
  if (options.scene && (options.scene.surface ?? true)) return options.scene.layout;
  return null;
}

function coreDefines(options: RtModuleOptions): Defines {
  const layout = surfaceLayoutOf(options);
  return {
    RT_RAY_MASKS: options.rayMasks ?? true,
    RT_SURFACE: !!layout,
    RT_SURFACE_NORMALS: !!layout?.normals,
    RT_SURFACE_UVS: !!layout?.uvs,
    RT_SURFACE_POSITIONS: !!layout?.positions,
    RT_SURFACE_TEXFRAME: !!layout && !!options.scene && !!options.scene.textures,
  };
}

function inlineSource(options: RtModuleOptions & { traceSlot: BindingSlot }): string {
  let source = `${bvhWalk(options)}\n${RT_CORE_WGSL}`;
  if (options.intervals) source += RT_INTERVAL_WGSL;
  const layout = surfaceLayoutOf(options);
  if (options.scene) {
    const { slot } = options.scene;
    const shade = !!options.shade;
    if (shade && (options.scene.surface === false || options.scene.materials === false || options.scene.lights === false)) {
      throw new Error('rtModuleWgsl: shade needs scene surface, materials and lights');
    }
    source += rtSceneWgsl({ group: slot.group, firstBinding: slot.firstBinding, materials: options.scene.materials, lights: options.scene.lights, sampling: options.scene.sampling || shade, textures: options.scene.textures, packing: options.scene.packing });
  } else if (options.shade) {
    throw new Error('rtModuleWgsl: shade needs the scene copy (scene option)');
  }
  if (options.scene && options.scene.textures && !(layout && layout.uvs && layout.positions)) throw new Error('rtModuleWgsl: scene textures need a surface layout with uvs and positions');
  if (layout) {
    const offsets = surfaceVertexOffsets(layout);
    const base = { vertexStride: Math.max(offsets.stride, 1), normalOffset: offsets.normal, uvOffset: offsets.uv, positionOffset: offsets.position, tangentOffset: offsets.tangent, tangents: !!layout.tangents, rayPositionError: !layout.positions };
    source += rtSurfaceWgsl({ ...base, headerVec4s: SCENE_HEADER_VEC4S });
  }
  if (options.shade) source += rtShadeWgsl(options.shade === true ? {} : options.shade);
  // The surface and shade text is written against the split bindings: point it at the packed buffer (the scene part already is).
  if (options.scene && (options.scene.packing ?? DEFAULT_SCENE_PACKING) === 'packed') source = rtPackSceneAccess(source);
  return source;
}

/**
 * The inline module: paste it into your own compute shader and call `rtVisible`, `rtClosest`, `rtClosestCone` and (with
 * `scene`) `rtSurface`, the materials and the lights, and (with `shade`) `rtShade`. Bindings: bvh's walk at `traceSlot` (3, or 4 for indexed leaves), then the scene copy's 2.
 */
export function rtModuleWgsl(caps: Caps, options: RtModuleOptions = {}): string {
  const traceSlot = options.traceSlot ?? { group: 0, firstBinding: 0 };
  const source = inlineSource({ ...options, traceSlot });
  let code = buildWgsl((options.stripComments ?? true) ? stripWgslComments(source) : source, caps, coreDefines(options));
  if (options.scene && options.scene.textures?.binding === 'table') {
    if (caps.resourceTable === 'none') throw new Error("rtModuleWgsl: textures with binding 'table' need a device with a resource table (caps.resourceTable is 'none'); build the textures with binding 'arrays'");
    // Enable lines go first; this one before the caps' own (order among enables is free). The extra line shifts error lines by one (`rtMapCompilationInfo` maps from the final text).
    code = `${rtTextureTableEnable(caps.featureNames.samplingResourceTable ?? caps.featureNames.heterogeneousResourceTable)}\n${code}`;
  }
  return options.minify ? rtMinifyWgsl(code) : code;
}

/** Bind group layout entries (without binding numbers) for the walk, as bvh's `traceModuleEntries`. */
export function rtWalkEntries(options: SceneFormat = {}): readonly ModuleBindingEntry[] {
  return traceModuleEntries({ triangleLeaves: sceneFormatOf(options).triangleLeaves });
}

/** The scene copy's bindings, in order: table, data (one binding, the packed buffer, with `packing: 'packed'`). */
export function rtSceneEntries(packing: SceneCopyPacking = DEFAULT_SCENE_PACKING): readonly ModuleBindingEntry[] {
  const entry: ModuleBindingEntry = { visibility: 0x4 /* GPUShaderStage.COMPUTE */, buffer: { type: 'read-only-storage' } };
  return packing === 'packed' ? [entry] : [entry, entry];
}

/** Storage buffers the module adds to a kernel (spec section 3 binding budget). */
export function rtModuleStorageBuffers(options: RtModuleOptions = {}): number {
  const walk = sceneFormatOf(options).triangleLeaves === 'indexed' ? STORAGE_BUFFER_BUDGET.walkIndexed : STORAGE_BUFFER_BUDGET.walk;
  return walk + (options.scene ? sceneCopyStorageBuffers(options.scene.packing) : 0);
}

// ---------------------------------------------------------------------------------------------------------------
// Batch kernels (queue ABI v0)
// ---------------------------------------------------------------------------------------------------------------

export type BatchMode = 'closest' | 'visible';
export type RayLayout = 'aos' | 'soa';

export interface BatchKernelOptions extends SceneFormat {
  mode: BatchMode;
  /** Workgroup size setting (D11). Default 64. */
  workgroupSize?: BatchWorkgroupSize;
  /** Ray buffer layout setting (D11): 'aos' one 32 B `Ray` array (contract; default) or 'soa' two vec4 arrays. */
  rayLayout?: RayLayout;
  /** GPU-side ray count (`rtBatchCounts`). Default false: the count is a uniform. */
  indirect?: boolean;
  /** Per-ray cones in, cone width at the hit out (closest mode only). Default false. */
  cones?: boolean;
  /** 40-byte records (default) or the 24-byte compact record without material and normal. */
  hitRecord?: 'full' | 'compact';
  rayMasks?: boolean;
  triangleFilter?: string | null;
  anyHitOrder?: AnyHitOrder;
  /** Device limit `maxComputeWorkgroupsPerDimension`, used by the indirect-args kernel. Default 65535. */
  maxWorkgroupsPerDimension?: number;
}

export interface ResolvedBatchOptions {
  mode: BatchMode;
  workgroupSize: BatchWorkgroupSize;
  rayLayout: RayLayout;
  indirect: boolean;
  cones: boolean;
  hitRecord: 'full' | 'compact';
  rayMasks: boolean;
  maxWorkgroupsPerDimension: number;
}

export function resolveBatchOptions(options: BatchKernelOptions): ResolvedBatchOptions {
  if (options.cones && options.mode === 'visible') throw new RangeError('rt batch: cones apply to closest mode only');
  const workgroupSize = options.workgroupSize ?? DEFAULT_BATCH_WORKGROUP_SIZE;
  return {
    mode: options.mode,
    workgroupSize,
    rayLayout: options.rayLayout ?? 'aos',
    indirect: options.indirect ?? false,
    cones: options.cones ?? false,
    hitRecord: options.hitRecord ?? 'full',
    rayMasks: options.rayMasks ?? true,
    maxWorkgroupsPerDimension: options.maxWorkgroupsPerDimension ?? 65535,
  };
}

/** The batch kernel (entry point `rtBatch`): scene in group 0 at binding 0, batch IO in group 1. */
export function rtBatchKernelWgsl(caps: Caps, options: BatchKernelOptions): string {
  const resolved = resolveBatchOptions(options);
  const moduleOptions = { ...options, traceSlot: { group: 0, firstBinding: 0 }, rayMasks: resolved.rayMasks };
  const source = `${inlineSource(moduleOptions)}\n${rtBatchWgsl({ workgroupSize: resolved.workgroupSize, maxWorkgroupsPerDimension: resolved.maxWorkgroupsPerDimension })}`;
  return buildWgsl(source, caps, {
    ...coreDefines(moduleOptions),
    RT_MODE_CLOSEST: resolved.mode === 'closest',
    RT_MODE_VISIBLE: resolved.mode === 'visible',
    RT_INDIRECT: resolved.indirect,
    RT_CONES: resolved.cones,
    RT_RAYS_SOA: resolved.rayLayout === 'soa',
    RT_HIT_FULL: resolved.hitRecord === 'full',
  });
}

/** The one-thread kernel that writes DispatchIndirect args (entry point `rtBatchArgs`) from the GPU-side count. */
export function rtBatchArgsKernelWgsl(caps: Caps, options: Pick<BatchKernelOptions, 'workgroupSize' | 'maxWorkgroupsPerDimension'> = {}): string {
  return buildWgsl(
    rtBatchArgsWgsl({ workgroupSize: options.workgroupSize ?? DEFAULT_BATCH_WORKGROUP_SIZE, maxWorkgroupsPerDimension: options.maxWorkgroupsPerDimension ?? 65535 }),
    caps,
  );
}

/** Group 1 layout entries of a batch kernel, by binding number. Explicit so pipelines and bind groups never disagree. */
export function rtBatchEntries(options: ResolvedBatchOptions): GPUBindGroupLayoutEntry[] {
  const COMPUTE = 0x4;
  const storage = (binding: number, type: 'read-only-storage' | 'storage'): GPUBindGroupLayoutEntry => ({ binding, visibility: COMPUTE, buffer: { type } });
  const entries: GPUBindGroupLayoutEntry[] = [storage(0, 'read-only-storage')];
  if (options.rayLayout === 'soa') entries.push(storage(1, 'read-only-storage'));
  entries.push(storage(2, 'storage'));
  entries.push({ binding: 3, visibility: COMPUTE, buffer: { type: 'uniform' } });
  if (options.indirect) entries.push(storage(4, 'read-only-storage'));
  if (options.cones) {
    entries.push(storage(5, 'read-only-storage'));
    entries.push(storage(6, 'storage'));
  }
  return entries;
}
