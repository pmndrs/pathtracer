/**
 * MaterialX graph lowering and TypeScript evaluation (P5).
 *
 * `lowerInput` turns an input of a node (a constant, an interface input, a node reference, a nodegraph output) into a small
 * expression DAG (`MtlxIr`): scoping, defaults, `output` selection and type coercion are settled once, so the TypeScript evaluator
 * (`evalIr`, the oracle), the constant folder, the texture baker (`bake.ts`) and the WGSL generator (`wgsl.ts`) all walk the same DAG.
 */
import { MTLX_COMPONENTS, parseMtlxNumeric, type MtlxDocument, type MtlxGraph, type MtlxInput, type MtlxNode } from './document.ts';
import { coerceValue, imageRequest, MTLX_NODES, MtlxUnsupportedError, specType, type MtlxImageRequest, type MtlxTsContext, type MtlxValue, type NodeArgs, type NodeImpl } from './nodes.ts';

export interface IrConst {
  kind: 'const';
  type: string;
  value: MtlxValue;
}

/** One argument of an op: the source expression, its own type (`from`) and the type the op wants (`to`). */
export interface IrArg {
  ir: MtlxIr;
  from: string;
  to: string;
}

export interface IrOp {
  kind: 'op';
  category: string;
  /** The node's result type. */
  type: string;
  /** The requested output of a multi-output node. */
  output: string | undefined;
  args: Record<string, IrArg>;
  /** String inputs (`file`, `space`, `uaddressmode`, `channels`, ...) and `colorspace` of the file input. */
  strings: Record<string, string>;
  impl: NodeImpl;
  /** `<graph or document>/<node name>`, for messages. */
  label: string;
}

export type MtlxIr = IrConst | IrOp;

interface Scope {
  graph: MtlxGraph | null;
  /** For messages and the memo key. */
  id: string;
}

export interface LowerOptions {
  /** Joins a `fileprefix` and a file name. Default: plain concatenation (MaterialX semantics). */
  joinPath?: (prefix: string, file: string) => string;
}

const STRING_TYPES = new Set(['string', 'filename']);

/** Lowering state for one document: memoizes nodes so shared subgraphs stay shared. */
export class MtlxLowering {
  readonly doc: MtlxDocument;
  private readonly memo = new Map<string, MtlxIr>();
  private readonly joinPath: (prefix: string, file: string) => string;

  constructor(doc: MtlxDocument, options: LowerOptions = {}) {
    this.doc = doc;
    this.joinPath = options.joinPath ?? ((p, f) => p + f);
  }

  /** The expression of input `name` of `node`; undefined when the input is absent (the caller applies its own default). */
  inputIr(node: MtlxNode, name: string, scope: Scope | null = null): { ir: MtlxIr; type: string } | undefined {
    const input = node.inputs[name];
    if (!input) return undefined;
    return this.lowerInput(input, scope ?? { graph: null, id: 'document' }, node);
  }

  /** The top-level node `name`. */
  topNode(name: string): MtlxNode {
    const n = this.doc.nodes.get(name);
    if (!n) throw new Error(`MaterialX: no node '${name}'`);
    return n;
  }

  private lowerInput(input: MtlxInput, scope: Scope, owner: MtlxNode): { ir: MtlxIr; type: string } | undefined {
    if (input.interfacename !== undefined) {
      const iface = scope.graph?.interface.get(input.interfacename);
      if (!iface) throw new Error(`MaterialX: ${scope.id}/${owner.name}.${input.name}: interface input '${input.interfacename}' is not defined`);
      if (iface.value !== undefined) return { ir: this.constIr(iface.type || input.type, iface.value), type: iface.type || input.type };
      // An interface input without a value takes the nodedef default of the consumer: let the caller apply its own.
      return undefined;
    }
    if (input.nodename !== undefined) {
      const target = (scope.graph ? scope.graph.nodes.get(input.nodename) : undefined) ?? (scope.graph ? undefined : this.doc.nodes.get(input.nodename));
      if (!target) throw new Error(`MaterialX: ${scope.id}/${owner.name}.${input.name}: node '${input.nodename}' not found`);
      const ir = this.lowerNode(target, scope, input.output);
      const out = target.outputs.find((o) => o.name === input.output);
      return { ir, type: out?.type ?? target.type };
    }
    if (input.nodegraph !== undefined) {
      const g = this.doc.graphs.get(input.nodegraph);
      if (!g) throw new Error(`MaterialX: ${scope.id}/${owner.name}.${input.name}: nodegraph '${input.nodegraph}' not found`);
      const outName = input.output ?? [...g.outputs.keys()][0];
      const out = outName !== undefined ? g.outputs.get(outName) : undefined;
      if (!out) throw new Error(`MaterialX: nodegraph '${g.name}' has no output '${outName ?? ''}'`);
      return { ir: this.lowerGraphOutput(g, out.name), type: out.type };
    }
    if (input.value !== undefined && !(input.value === '' && !STRING_TYPES.has(input.type))) {
      return { ir: this.constIr(input.type, input.value), type: input.type };
    }
    return undefined;
  }

  /** The expression of an output of a top-level nodegraph (its interface inputs take their authored defaults). */
  lowerGraphOutput(g: MtlxGraph, outputName: string): MtlxIr {
    const out = g.outputs.get(outputName);
    if (!out) throw new Error(`MaterialX: nodegraph '${g.name}' has no output '${outputName}'`);
    const scope: Scope = { graph: g, id: g.name };
    if (out.nodename === undefined) throw new Error(`MaterialX: output '${outputName}' of '${g.name}' is not connected to a node`);
    const target = g.nodes.get(out.nodename);
    if (!target) throw new Error(`MaterialX: output '${outputName}' of '${g.name}': node '${out.nodename}' not found`);
    return this.lowerNode(target, scope, out.output);
  }

  private constIr(type: string, text: string): IrConst {
    if (STRING_TYPES.has(type)) throw new Error(`MaterialX: a string value '${text}' cannot feed a numeric input`);
    return { kind: 'const', type, value: parseMtlxNumeric(type, text) };
  }

  lowerNode(node: MtlxNode, scope: Scope, output?: string): MtlxIr {
    const key = `${scope.id}/${node.name}/${output ?? ''}`;
    const hit = this.memo.get(key);
    if (hit) return hit;
    const impl = MTLX_NODES[node.category];
    if (!impl) throw new MtlxUnsupportedError(node.category, `MaterialX node <${node.category}> '${node.name}' is not supported`);
    const type = output !== undefined ? (node.outputs.find((o) => o.name === output)?.type ?? node.type) : node.type;
    const args: Record<string, IrArg> = {};
    const strings: Record<string, string> = {};
    const nodeType = type;
    // Multi-output nodes (separate*) keep their `in` type, not the output's.
    for (const [name, spec] of Object.entries(impl.inputs)) {
      const to = specType(spec, nodeType);
      if (STRING_TYPES.has(to)) {
        const inp = node.inputs[name];
        const iface = inp?.interfacename !== undefined ? scope.graph?.interface.get(inp.interfacename) : undefined;
        let text = iface?.value ?? inp?.value ?? (typeof spec.d === 'string' ? spec.d : '');
        if (name === 'file' && inp) text = this.joinPath(iface?.fileprefix ?? inp.fileprefix ?? '', text);
        strings[name] = text;
        if (name === 'file') strings.colorspace = iface?.colorspace ?? inp?.colorspace ?? this.doc.colorspace;
        continue;
      }
      const src = this.inputIrScoped(node, name, scope);
      if (src) {
        args[name] = { ir: src.ir, from: src.type, to: to === 'any' ? src.type : to };
        continue;
      }
      // Absent input: the nodedef default (a number, or `uv` / `normal` / `tangent` = the geometry streams).
      const d = spec.d;
      if (typeof d === 'string') {
        const geomCategory = d === 'uv' ? 'texcoord' : d;
        const gi = MTLX_NODES[geomCategory];
        if (!gi) throw new Error(`MaterialX: internal: unknown default '${d}'`);
        const ir: IrOp = { kind: 'op', category: geomCategory, type: d === 'uv' ? 'vector2' : 'vector3', output: undefined, args: {}, strings: {}, impl: gi, label: `${scope.id}/${node.name}.${name}` };
        args[name] = { ir, from: ir.type, to: ir.type };
      } else {
        // A one-number default is a scalar default: it fills every lane, as a scalar operand does.
        const value: MtlxValue = d.length === 1 ? [d[0]!, d[0]!, d[0]!, d[0]!] : [d[0] ?? 0, d[1] ?? 0, d[2] ?? 0, d[3] ?? 0];
        const t = to === 'any' ? nodeType : to;
        args[name] = { ir: { kind: 'const', type: t, value }, from: t, to: t };
      }
    }
    // String-valued attributes that are not inputs (`space` on position is an input in 1.38; `index` too): keep the node attributes as fallbacks.
    for (const [k, v] of Object.entries(node.attributes)) if (!(k in strings) && !(k in impl.inputs)) strings[k] = v;
    const op: IrOp = { kind: 'op', category: node.category, type, output, args, strings, impl, label: `${scope.id}/${node.name}` };
    this.memo.set(key, op);
    return op;
  }

  private inputIrScoped(node: MtlxNode, name: string, scope: Scope): { ir: MtlxIr; type: string } | undefined {
    const input = node.inputs[name];
    if (!input) return undefined;
    return this.lowerInput(input, scope, node);
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------------------------------------------

/** True when the value depends on the shading point (uv, position, normal, tangent). */
export function dependsOnGeometry(ir: MtlxIr, cache = new Map<MtlxIr, boolean>()): boolean {
  if (ir.kind === 'const') return false;
  const hit = cache.get(ir);
  if (hit !== undefined) return hit;
  let dep = !!ir.impl.geometry;
  for (const a of Object.values(ir.args)) dep = dep || dependsOnGeometry(a.ir, cache);
  cache.set(ir, dep);
  return dep;
}

function argsOf(ir: IrOp, valueOf: (a: IrArg) => MtlxValue): NodeArgs<MtlxValue> {
  const v: Record<string, MtlxValue> = {};
  for (const [name, a] of Object.entries(ir.args)) v[name] = coerceValue(valueOf(a), a.from, a.to);
  return { v, strings: ir.strings, type: ir.type, output: ir.output };
}

/** Evaluates an expression at one shading point. `memo` shares work across the expressions of one point. */
export function evalIr(ir: MtlxIr, ctx: MtlxTsContext, memo: Map<MtlxIr, MtlxValue> = new Map()): MtlxValue {
  if (ir.kind === 'const') return ir.value;
  const hit = memo.get(ir);
  if (hit) return hit;
  const value = ir.impl.ts(argsOf(ir, (a) => evalIr(a.ir, ctx, memo)), ctx);
  memo.set(ir, value);
  return value;
}

/** The shading-point-independent context: lookups of images still work, geometry reads zero. */
export function constantContext(sample: MtlxTsContext['sample'] = () => undefined): MtlxTsContext {
  return { uv: [0, 0], position: [0, 0, 0], normal: [0, 0, 1], tangent: [1, 0, 0], sample };
}

/** Replaces every subexpression that does not depend on the shading point by its value. Shared nodes stay shared. */
export function foldConstants(ir: MtlxIr, ctx: MtlxTsContext = constantContext(), memo = new Map<MtlxIr, MtlxIr>(), depCache = new Map<MtlxIr, boolean>()): MtlxIr {
  if (ir.kind === 'const') return ir;
  const hit = memo.get(ir);
  if (hit) return hit;
  let out: MtlxIr;
  if (!dependsOnGeometry(ir, depCache)) {
    out = { kind: 'const', type: ir.type, value: evalIr(ir, ctx) };
  } else {
    const args: Record<string, IrArg> = {};
    for (const [k, a] of Object.entries(ir.args)) args[k] = { ...a, ir: foldConstants(a.ir, ctx, memo, depCache) };
    out = { ...ir, args };
  }
  memo.set(ir, out);
  return out;
}

/** Number of components of a numeric type (`float` 1, `color3` 3, ...). */
export const componentsOf = (type: string): number => MTLX_COMPONENTS[type] ?? 4;

/** Every image an expression reads (for loading / baking), in discovery order, de-duplicated by request. */
export function collectImages(ir: MtlxIr, seen = new Set<MtlxIr>(), out: MtlxImageRequest[] = []): MtlxImageRequest[] {
  if (ir.kind === 'const' || seen.has(ir)) return out;
  seen.add(ir);
  if (ir.category === 'image' || ir.category === 'tiledimage') {
    const r = imageRequest(ir.strings, ir.type);
    if (!out.some((o) => o.file === r.file && o.srgb === r.srgb && o.uaddress === r.uaddress && o.vaddress === r.vaddress)) out.push(r);
  }
  for (const a of Object.values(ir.args)) collectImages(a.ir, seen, out);
  return out;
}

/** Categories of every op in an expression (for the support report). */
export function collectCategories(ir: MtlxIr, seen = new Set<MtlxIr>(), out = new Set<string>()): Set<string> {
  if (ir.kind === 'const' || seen.has(ir)) return out;
  seen.add(ir);
  out.add(ir.category);
  for (const a of Object.values(ir.args)) collectCategories(a.ir, seen, out);
  return out;
}
