/**
 * A small XML reader for MaterialX documents (P5): elements, attributes, comments, `<?...?>`, `<!...>` and CDATA are
 * handled; namespaces are kept as part of the name (`xi:include`). No entity decoding beyond the five predefined ones.
 * Pure TypeScript, no dependencies, so it runs in Node, the browser and the bench pages alike.
 */
export interface XmlElement {
  name: string;
  attributes: Record<string, string>;
  children: XmlElement[];
  /** Character data directly inside the element (trimmed, concatenated). */
  text: string;
}

const ENTITIES: Record<string, string> = { '&lt;': '<', '&gt;': '>', '&amp;': '&', '&quot;': '"', '&apos;': "'" };
const decode = (s: string): string => (s.includes('&') ? s.replace(/&(lt|gt|amp|quot|apos);|&#(\d+);|&#x([0-9a-fA-F]+);/g, (m, _n, dec, hex) => (dec ? String.fromCodePoint(Number(dec)) : hex ? String.fromCodePoint(parseInt(hex, 16)) : (ENTITIES[m] ?? m))) : s);

/** Parses `source` and returns the root element. Throws an Error with the line number on malformed input. */
export function parseXml(source: string): XmlElement {
  let i = 0;
  const n = source.length;
  const line = (): number => source.slice(0, i).split('\n').length;
  const fail = (msg: string): never => {
    throw new Error(`XML: ${msg} (line ${line()})`);
  };
  const stack: XmlElement[] = [];
  let root: XmlElement | null = null;
  const isSpace = (c: string): boolean => c === ' ' || c === '\n' || c === '\t' || c === '\r';

  while (i < n) {
    const lt = source.indexOf('<', i);
    if (lt < 0) break;
    if (lt > i && stack.length > 0) {
      const t = decode(source.slice(i, lt)).trim();
      if (t) stack[stack.length - 1]!.text += t;
    }
    i = lt;
    if (source.startsWith('<!--', i)) {
      const end = source.indexOf('-->', i + 4);
      if (end < 0) fail('unterminated comment');
      i = end + 3;
    } else if (source.startsWith('<![CDATA[', i)) {
      const end = source.indexOf(']]>', i);
      if (end < 0) fail('unterminated CDATA');
      if (stack.length > 0) stack[stack.length - 1]!.text += source.slice(i + 9, end);
      i = end + 3;
    } else if (source.startsWith('<?', i)) {
      const end = source.indexOf('?>', i);
      if (end < 0) fail('unterminated processing instruction');
      i = end + 2;
    } else if (source.startsWith('<!', i)) {
      const end = source.indexOf('>', i);
      if (end < 0) fail('unterminated declaration');
      i = end + 1;
    } else if (source.startsWith('</', i)) {
      const end = source.indexOf('>', i);
      if (end < 0) fail('unterminated end tag');
      const name = source.slice(i + 2, end).trim();
      const open = stack.pop();
      if (!open || open.name !== name) fail(`end tag </${name}> does not match <${open?.name ?? 'nothing'}>`);
      i = end + 1;
    } else {
      // Start tag: name, attributes, `>` or `/>`.
      i++;
      let j = i;
      while (j < n && !isSpace(source[j]!) && source[j] !== '>' && source[j] !== '/') j++;
      const el: XmlElement = { name: source.slice(i, j), attributes: {}, children: [], text: '' };
      if (!el.name) fail('empty element name');
      i = j;
      let selfClosing = false;
      for (;;) {
        while (i < n && isSpace(source[i]!)) i++;
        if (i >= n) fail('unterminated start tag');
        const c = source[i]!;
        if (c === '>') {
          i++;
          break;
        }
        if (c === '/') {
          if (source[i + 1] !== '>') fail("stray '/' in a start tag");
          selfClosing = true;
          i += 2;
          break;
        }
        let k = i;
        while (k < n && !isSpace(source[k]!) && source[k] !== '=' && source[k] !== '>' && source[k] !== '/') k++;
        const attr = source.slice(i, k);
        i = k;
        while (i < n && isSpace(source[i]!)) i++;
        if (source[i] !== '=') fail(`attribute '${attr}' has no value`);
        i++;
        while (i < n && isSpace(source[i]!)) i++;
        const q = source[i];
        if (q !== '"' && q !== "'") fail(`attribute '${attr}' value is not quoted`);
        const close = source.indexOf(q!, i + 1);
        if (close < 0) fail('unterminated attribute value');
        el.attributes[attr] = decode(source.slice(i + 1, close));
        i = close + 1;
      }
      const parent = stack[stack.length - 1];
      if (parent) parent.children.push(el);
      else if (root) fail('more than one root element');
      else root = el;
      if (!selfClosing) stack.push(el);
    }
  }
  if (stack.length > 0) fail(`element <${stack[stack.length - 1]!.name}> is not closed`);
  if (!root) throw new Error('XML: no root element');
  return root;
}
