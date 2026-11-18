/**
 * MaterialX document model (P5): a `.mtlx` file read into nodes, node graphs and inputs. Only structure and attribute
 * inheritance (`fileprefix`, `colorspace`) are resolved here; node semantics live in `nodes.ts`, evaluation in `eval.ts`.
 *
 * The format follows the MaterialX 1.38 / 1.39 specification (Apache-2.0, https://materialx.org). Node definitions
 * (`<nodedef>`) and the functional graphs that implement them are skipped: the supported nodes are built in.
 */
import { parseXml, type XmlElement } from './xml.ts';

/** A node input (or an interface input of a node graph). Exactly one source is set: `value`, `nodename`, `nodegraph` or `interfacename`; none = the nodedef default. */
export interface MtlxInput {
  name: string;
  type: string;
  value?: string;
  nodename?: string;
  /** A top-level node graph reference (`nodegraph="NG_x" output="out"`). */
  nodegraph?: string;
  output?: string;
  interfacename?: string;
  /** Resolved `colorspace` attribute of the input or an ancestor (undefined = the document default is meant). */
  colorspace?: string;
  /** Resolved `fileprefix` of the input or an ancestor. */
  fileprefix?: string;
}

export interface MtlxOutput {
  name: string;
  type: string;
  nodename?: string;
  nodegraph?: string;
  output?: string;
}

export interface MtlxNode {
  name: string;
  category: string;
  type: string;
  inputs: Record<string, MtlxInput>;
  /** Extra attributes (`space`, `index` ...). */
  attributes: Record<string, string>;
  /** Named outputs of a multi-output node (`<output>` children). */
  outputs: MtlxOutput[];
}

export interface MtlxGraph {
  name: string;
  nodes: Map<string, MtlxNode>;
  outputs: Map<string, MtlxOutput>;
  /** Interface inputs (`<input name type value>` directly in the graph): the defaults `interfacename` refers to. */
  interface: Map<string, MtlxInput>;
}

export interface MtlxDocument {
  /** `1.38`, `1.39`, ... as written. */
  version: string;
  /** The document's colour space (default `lin_rec709`). */
  colorspace: string;
  nodes: Map<string, MtlxNode>;
  graphs: Map<string, MtlxGraph>;
  /** `surfacematerial` node names, in document order. */
  materials: string[];
  /** Elements the reader skipped on purpose (`look`, `nodedef`, `xi:include`...), by tag: informational. */
  skipped: string[];
}

const SKIP = new Set(['look', 'materialassign', 'collection', 'geominfo', 'geompropdef', 'typedef', 'unitdef', 'unittypedef', 'variant', 'variantset', 'variantassign', 'visibility', 'propertyset', 'propertysetassign', 'xi:include', 'implementation', 'attributedef', 'nodedef']);

function readInput(el: XmlElement, inherited: { fileprefix?: string; colorspace?: string }): MtlxInput {
  const a = el.attributes;
  const input: MtlxInput = { name: a.name ?? '', type: a.type ?? '' };
  if (a.value !== undefined) input.value = a.value;
  if (a.nodename !== undefined) input.nodename = a.nodename;
  if (a.nodegraph !== undefined) input.nodegraph = a.nodegraph;
  if (a.output !== undefined) input.output = a.output;
  if (a.interfacename !== undefined) input.interfacename = a.interfacename;
  const colorspace = a.colorspace ?? inherited.colorspace;
  if (colorspace !== undefined) input.colorspace = colorspace;
  const fileprefix = a.fileprefix ?? inherited.fileprefix;
  if (fileprefix !== undefined) input.fileprefix = fileprefix;
  return input;
}

function readOutput(el: XmlElement): MtlxOutput {
  const a = el.attributes;
  const o: MtlxOutput = { name: a.name ?? '', type: a.type ?? '' };
  if (a.nodename !== undefined) o.nodename = a.nodename;
  if (a.nodegraph !== undefined) o.nodegraph = a.nodegraph;
  if (a.output !== undefined) o.output = a.output;
  return o;
}

function readNode(el: XmlElement, inherited: { fileprefix?: string; colorspace?: string }): MtlxNode {
  const own = { ...inherited };
  if (el.attributes.fileprefix !== undefined) own.fileprefix = el.attributes.fileprefix;
  if (el.attributes.colorspace !== undefined) own.colorspace = el.attributes.colorspace;
  const node: MtlxNode = { name: el.attributes.name ?? '', category: el.name, type: el.attributes.type ?? '', inputs: {}, attributes: {}, outputs: [] };
  for (const [k, v] of Object.entries(el.attributes)) if (k !== 'name' && k !== 'type') node.attributes[k] = v;
  for (const child of el.children) {
    if (child.name === 'input' || child.name === 'token') {
      const input = readInput(child, own);
      node.inputs[input.name] = input;
    } else if (child.name === 'output') node.outputs.push(readOutput(child));
  }
  return node;
}

/** Parses a `.mtlx` text. Throws on malformed XML or a root that is not `<materialx>`. */
export function parseMaterialX(source: string): MtlxDocument {
  const root = parseXml(source);
  if (root.name !== 'materialx') throw new Error(`MaterialX: root element is <${root.name}>, expected <materialx>`);
  const doc: MtlxDocument = { version: root.attributes.version ?? '1.38', colorspace: root.attributes.colorspace ?? 'lin_rec709', nodes: new Map(), graphs: new Map(), materials: [], skipped: [] };
  const top: { fileprefix?: string; colorspace?: string } = {};
  if (root.attributes.fileprefix !== undefined) top.fileprefix = root.attributes.fileprefix;
  if (root.attributes.colorspace !== undefined) top.colorspace = root.attributes.colorspace;
  for (const el of root.children) {
    if (SKIP.has(el.name)) {
      doc.skipped.push(el.name);
      continue;
    }
    if (el.name === 'nodegraph') {
      if (el.attributes.nodedef !== undefined) {
        doc.skipped.push('nodegraph(nodedef)');
        continue;
      }
      const own = { ...top };
      if (el.attributes.fileprefix !== undefined) own.fileprefix = el.attributes.fileprefix;
      if (el.attributes.colorspace !== undefined) own.colorspace = el.attributes.colorspace;
      const g: MtlxGraph = { name: el.attributes.name ?? '', nodes: new Map(), outputs: new Map(), interface: new Map() };
      for (const c of el.children) {
        if (c.name === 'input') {
          const i = readInput(c, own);
          g.interface.set(i.name, i);
        } else if (c.name === 'output') {
          const o = readOutput(c);
          g.outputs.set(o.name, o);
        } else if (!SKIP.has(c.name)) {
          const n = readNode(c, own);
          g.nodes.set(n.name, n);
        }
      }
      doc.graphs.set(g.name, g);
      continue;
    }
    const node = readNode(el, top);
    doc.nodes.set(node.name, node);
    if (node.category === 'surfacematerial') doc.materials.push(node.name);
  }
  return doc;
}

// ---------------------------------------------------------------------------------------------------------------
// Values
// ---------------------------------------------------------------------------------------------------------------

/** Component count of the numeric MaterialX types (0 = not numeric). */
export const MTLX_COMPONENTS: Record<string, number> = { float: 1, integer: 1, boolean: 1, color3: 3, color4: 4, vector2: 2, vector3: 3, vector4: 4 };

/** A numeric value is four lanes (unused lanes 0), a string value is a string. The WGSL twin keeps the same shape (`vec4f`). */
export type MtlxValue = [number, number, number, number];

/** Reads a literal of a numeric type into four lanes. `boolean` is 1 / 0. Throws on a malformed literal. */
export function parseMtlxNumeric(type: string, text: string): MtlxValue {
  const n = MTLX_COMPONENTS[type];
  if (!n) throw new Error(`MaterialX: '${type}' is not a numeric type`);
  const out: MtlxValue = [0, 0, 0, 0];
  if (type === 'boolean') {
    out[0] = text.trim() === 'true' ? 1 : 0;
    return out;
  }
  const parts = text.split(/[,\s]+/).filter((s) => s.length > 0);
  if (parts.length !== n) throw new Error(`MaterialX: '${text}' does not have ${n} components for type ${type}`);
  parts.forEach((p, k) => {
    const v = Number(p);
    if (!Number.isFinite(v)) throw new Error(`MaterialX: bad number '${p}' in '${text}'`);
    out[k] = v;
  });
  return out;
}
