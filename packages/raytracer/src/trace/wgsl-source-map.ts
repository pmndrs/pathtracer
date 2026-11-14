/**
 * Source maps for generated WGSL (P2.10b). The transforms that shrink a kernel (comment strip, `rtPruneWgsl`, `rtShortenWgsl`,
 * `rtMinifyWgsl`, prepended `enable` lines) move lines and rename identifiers, so a compiler message points at the minified text
 * and names like `C5`. A `RtWgslSourceMap` travels with the code and undoes both: output line -> readable source line, short
 * name -> original name. The "source" is the readable text (preprocessed, comments stripped, nothing pruned, long names, the
 * `enable` lines included): exactly what `prune: false, shortenNames: false, minify: false` would have compiled.
 *
 * Lines and names only (no columns): minify and shorten change every column, and a line plus the original names is what you read.
 */

export interface RtWgslSourceMap {
  /** Output line (0-based) -> source line (0-based); -1 for a line with no source. */
  lines: Int32Array;
  /** Short name -> original name (empty when nothing was shortened). */
  names: Record<string, string>;
  /** The readable source, one entry per source line. Lets a message show the original text around the error. */
  sources?: string[];
  /** Leading `enable ...;` lines the preprocessor prepends: template line = source line - enableLines. */
  enableLines?: number;
}

/** Code and the map that reads its compiler messages. */
export interface RtWgslBuild {
  code: string;
  map: RtWgslSourceMap;
}

/** The subset of `GPUCompilationMessage` the mapping reads (so it also runs without a GPU). */
export interface RtCompilationMessageLike {
  message: string;
  type: 'error' | 'warning' | 'info';
  /** 1-based; 0 when the message has no line. */
  lineNum: number;
  linePos: number;
  offset?: number;
  length?: number;
}

export interface RtMappedCompilationMessage {
  type: 'error' | 'warning' | 'info';
  /** The message with short identifiers replaced by the original names. */
  message: string;
  /** Source line, 1-based; 0 when the message had no line or the line has no source. */
  lineNum: number;
  /** The source line's text, when the map carries `sources`. */
  sourceLine?: string;
  /** The compiled (minified) position, as the compiler reported it. */
  outputLineNum: number;
  outputLinePos: number;
}

/** A map from a text to itself: the start of a pipeline, and the map of a text that is not transformed. */
export function rtWgslIdentityMap(code: string): RtWgslSourceMap {
  const sources = code.split('\n');
  const lines = new Int32Array(sources.length);
  for (let i = 0; i < lines.length; i++) lines[i] = i;
  return { lines, names: {}, sources, enableLines: countEnableLines(sources) };
}

function countEnableLines(lines: readonly string[]): number {
  let n = 0;
  while (n < lines.length && /^enable\s/.test(lines[n] as string)) n++;
  return n;
}

/** `outToMid` then `midToSrc`: output line -> source line, -1 stays -1. */
export function rtComposeLineMaps(outToMid: Int32Array, midToSrc: Int32Array): Int32Array {
  const out = new Int32Array(outToMid.length);
  for (let i = 0; i < out.length; i++) {
    const mid = outToMid[i] as number;
    out[i] = mid >= 0 && mid < midToSrc.length ? (midToSrc[mid] as number) : -1;
  }
  return out;
}

/** Composes a stage map onto the map of the text the stage ran on (`previous`): `stage.lines` index `previous`' output lines. */
export function rtComposeSourceMaps(previous: RtWgslSourceMap, stage: { lines?: Int32Array; names?: Record<string, string> }): RtWgslSourceMap {
  const names = { ...previous.names };
  // A name introduced by a later stage maps back through the earlier renames.
  for (const [short, was] of Object.entries(stage.names ?? {})) names[short] = previous.names[was] ?? was;
  return {
    lines: stage.lines ? rtComposeLineMaps(stage.lines, previous.lines) : previous.lines,
    names,
    sources: previous.sources,
    enableLines: previous.enableLines,
  };
}

/** Source line (0-based) of an output line (0-based), or -1. */
export function rtWgslSourceLine(map: RtWgslSourceMap, outputLine: number): number {
  return outputLine >= 0 && outputLine < map.lines.length ? (map.lines[outputLine] as number) : -1;
}

const renamers = new WeakMap<RtWgslSourceMap, { count: number; re: RegExp | null }>();

/** Replaces every whole-word short identifier in `text` with its original name (not after a `.`, so members are left alone). */
export function rtRenameWgslNames(text: string, map: Pick<RtWgslSourceMap, 'names'>): string {
  const keys = Object.keys(map.names);
  if (keys.length === 0) return text;
  let entry = renamers.get(map as RtWgslSourceMap);
  if (!entry || entry.count !== keys.length) {
    const alternation = keys.sort((a, b) => b.length - a.length || (a < b ? -1 : 1)).map((k) => k.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|');
    entry = { count: keys.length, re: new RegExp(`(?<![\\w.])(?:${alternation})(?!\\w)`, 'g') };
    renamers.set(map as RtWgslSourceMap, entry);
  }
  return text.replace(entry.re as RegExp, (id) => map.names[id] ?? id);
}

/** One message with the line and the names of the readable source. */
export function rtMapCompilationMessage(msg: RtCompilationMessageLike, map: RtWgslSourceMap): RtMappedCompilationMessage {
  const src = msg.lineNum > 0 ? rtWgslSourceLine(map, msg.lineNum - 1) : -1;
  const out: RtMappedCompilationMessage = {
    type: msg.type,
    message: rtRenameWgslNames(msg.message, map),
    lineNum: src >= 0 ? src + 1 : 0,
    outputLineNum: msg.lineNum,
    outputLinePos: msg.linePos,
  };
  if (src >= 0 && map.sources?.[src] !== undefined) out.sourceLine = map.sources[src];
  return out;
}

/** All messages of a `GPUCompilationInfo` (anything with a `messages` array). */
export function rtMapCompilationInfo(info: { messages: readonly RtCompilationMessageLike[] }, map: RtWgslSourceMap): RtMappedCompilationMessage[] {
  return info.messages.map((m) => rtMapCompilationMessage(m, map));
}

/** The source lines around `lineNum` (1-based), numbered, the line itself marked with `>`. Empty when the map has no sources. */
export function rtFormatSourceContext(map: RtWgslSourceMap, lineNum: number, context = 2): string {
  const sources = map.sources;
  if (!sources || lineNum < 1 || lineNum > sources.length) return '';
  const from = Math.max(1, lineNum - context);
  const to = Math.min(sources.length, lineNum + context);
  const width = String(to).length;
  const rows: string[] = [];
  for (let n = from; n <= to; n++) {
    const text = sources[n - 1] as string;
    rows.push(`${n === lineNum ? '>' : ' '} ${String(n).padStart(width)} | ${text.length > 160 ? `${text.slice(0, 160)}...` : text}`);
  }
  return rows.join('\n');
}

/** Human text for the messages of one compile: `label`, `type line N: message`, then the source context of each error/warning. */
export function rtFormatCompilationMessages(messages: readonly RtMappedCompilationMessage[], map: RtWgslSourceMap, label?: string): string {
  const parts = messages.map((m) => {
    const head = `${m.type}${m.lineNum > 0 ? ` at source line ${m.lineNum}` : ''}${m.outputLineNum > 0 ? ` (compiled line ${m.outputLineNum})` : ''}: ${m.message}`;
    const ctx = m.lineNum > 0 ? rtFormatSourceContext(map, m.lineNum) : '';
    return ctx ? `${head}\n${ctx}` : head;
  });
  return `${label ? `WGSL (${label}) ` : 'WGSL '}failed to compile:\n${parts.join('\n')}`;
}

/** Reads `module.getCompilationInfo()` and throws an Error with translated text when there is an error. Returns the messages otherwise. */
export async function rtCheckShaderModule(module: GPUShaderModule, map: RtWgslSourceMap, label?: string): Promise<RtMappedCompilationMessage[]> {
  const mapped = rtMapCompilationInfo(await module.getCompilationInfo(), map);
  if (mapped.some((m) => m.type === 'error')) throw new Error(rtFormatCompilationMessages(mapped, map, label));
  return mapped;
}

export interface RtComputePipelineSource {
  label?: string;
  layout: GPUPipelineLayout | 'auto';
  code: string;
  /** The map of `code`; without it messages show the compiled text as is. */
  map?: RtWgslSourceMap;
  entryPoint: string;
  constants?: Record<string, number>;
  /** Also pass a Source Map v3 as `GPUShaderModuleDescriptor.sourceMap` (browsers may ignore it). Default false. */
  sourceMap?: boolean;
}

/**
 * `createShaderModule` + `createComputePipelineAsync`. Nothing extra runs on success. When the pipeline fails, the module's
 * compilation messages are read, translated through `map` and thrown (original line numbers, names and source text), with the
 * GPU error as `cause`.
 */
export async function rtCreateComputePipelineAsync(device: GPUDevice, source: RtComputePipelineSource): Promise<GPUComputePipeline> {
  const descriptor: GPUShaderModuleDescriptor & { sourceMap?: object } = { label: source.label, code: source.code };
  if (source.sourceMap && source.map) descriptor.sourceMap = rtWgslSourceMapV3(source.map, { file: source.label });
  const module = device.createShaderModule(descriptor);
  try {
    return await device.createComputePipelineAsync({
      label: source.label,
      layout: source.layout,
      compute: { module, entryPoint: source.entryPoint, ...(source.constants ? { constants: source.constants } : {}) },
    });
  } catch (cause) {
    const map = source.map ?? rtWgslIdentityMap(source.code);
    const mapped = rtMapCompilationInfo(await module.getCompilationInfo(), map);
    if (!mapped.some((m) => m.type === 'error')) throw cause;
    throw new Error(rtFormatCompilationMessages(mapped, map, source.label), { cause });
  }
}

const B64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';

function vlq(value: number): string {
  let v = value < 0 ? (-value << 1) | 1 : value << 1;
  let out = '';
  do {
    let digit = v & 31;
    v >>>= 5;
    if (v > 0) digit |= 32;
    out += B64[digit];
  } while (v > 0);
  return out;
}

export interface RtSourceMapV3 {
  version: 3;
  file?: string;
  sources: string[];
  sourcesContent?: string[];
  names: string[];
  mappings: string;
}

/**
 * Standard Source Map v3, line level: every compiled line maps to column 0 of its source line. `names` lists the original names
 * (no name segments: columns are not tracked). `sourcesContent` only with `includeContent`.
 */
export function rtWgslSourceMapV3(map: RtWgslSourceMap, options: { file?: string; sourceName?: string; includeContent?: boolean } = {}): RtSourceMapV3 {
  let prev = 0;
  const lines: string[] = [];
  for (let i = 0; i < map.lines.length; i++) {
    const src = map.lines[i] as number;
    if (src < 0) {
      lines.push('');
      continue;
    }
    lines.push(`AA${vlq(src - prev)}A`); // generated column 0, source 0, source line delta, source column 0
    prev = src;
  }
  const out: RtSourceMapV3 = { version: 3, sources: [options.sourceName ?? 'kernel.wgsl'], names: [...new Set(Object.values(map.names))], mappings: lines.join(';') };
  if (options.file) out.file = options.file;
  if (options.includeContent && map.sources) out.sourcesContent = [map.sources.join('\n')];
  return out;
}
