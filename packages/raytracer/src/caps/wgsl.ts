import type { Caps } from './detect.ts';

export type Defines = Record<string, boolean | number>;

/** Preprocessor defines derived from caps. Pipelines should be cached by (kernel, capsHash). */
export function capsDefines(caps: Caps): Defines {
  return {
    HAS_SUBGROUPS: caps.subgroups,
    HAS_SUBGROUP_SIZE_CONTROL: caps.subgroupSizeControl,
    HAS_F16: caps.shaderF16,
    HAS_TIMESTAMP: caps.timestampQuery,
    HAS_IMMEDIATES: caps.immediates,
    BINDLESS: caps.resourceTable !== 'none',
    BINDLESS_HETEROGENEOUS: caps.resourceTable === 'heterogeneous',
    HAS_SUBGROUP_MATRIX: caps.subgroupMatrix,
    TIER: caps.tier,
  };
}

/**
 * WGSL `enable` lines for the features the device has. Immediates are a language feature and need no `enable`.
 * Bindless needs no enable line in current Chrome; update here if that changes.
 */
export function wgslEnableLines(caps: Caps): string {
  const lines: string[] = [];
  if (caps.shaderF16) lines.push('enable f16;');
  if (caps.subgroups) lines.push('enable subgroups;');
  if (caps.subgroupMatrix) {
    lines.push(
      caps.featureNames.subgroupMatrix?.startsWith('chromium-experimental')
        ? 'enable chromium_experimental_subgroup_matrix;'
        : 'enable subgroup_matrix;',
    );
  }
  return lines.join('\n');
}

/**
 * Tiny line-based preprocessor: `#if NAME`, `#if !NAME`, `#elif NAME`, `#else`, `#endif`, nestable.
 * A name is true when its define is truthy; unknown names are false. Dropped lines become empty so that
 * WGSL compiler line numbers still match the source.
 */
export function preprocess(source: string, defines: Defines): string {
  type Frame = { parentActive: boolean; taken: boolean; active: boolean; sawElse: boolean };
  const stack: Frame[] = [];
  const out: string[] = [];
  const active = () => (stack.length ? stack[stack.length - 1]!.active : true);
  const test = (expr: string, line: number): boolean => {
    const m = /^(!?)\s*([A-Za-z_]\w*)$/.exec(expr.trim());
    if (!m) throw new Error(`wgsl preprocess: bad condition "${expr}" at line ${line}`);
    return !!defines[m[2]!] !== (m[1] === '!');
  };
  source.split('\n').forEach((text, i) => {
    const n = i + 1;
    const d = /^\s*#(if|elif|else|endif)\b(.*)$/.exec(text);
    if (!d) {
      out.push(active() ? text : '');
      return;
    }
    const [, kind, rest] = d;
    if (kind === 'if') {
      const parentActive = active();
      const v = parentActive && test(rest!, n);
      stack.push({ parentActive, taken: v, active: v, sawElse: false });
    } else {
      const f = stack[stack.length - 1];
      if (!f) throw new Error(`wgsl preprocess: #${kind} without #if at line ${n}`);
      if (kind === 'elif') {
        if (f.sawElse) throw new Error(`wgsl preprocess: #elif after #else at line ${n}`);
        const v = f.parentActive && !f.taken && test(rest!, n);
        f.active = v;
        f.taken ||= v;
      } else if (kind === 'else') {
        if (f.sawElse) throw new Error(`wgsl preprocess: duplicate #else at line ${n}`);
        f.sawElse = true;
        f.active = f.parentActive && !f.taken;
        f.taken = true;
      } else {
        stack.pop();
      }
    }
    out.push('');
  });
  if (stack.length) throw new Error('wgsl preprocess: missing #endif');
  return out.join('\n');
}

/** Convenience: enable lines + preprocessed source with caps defines (plus any extras). */
export function buildWgsl(source: string, caps: Caps, extraDefines: Defines = {}): string {
  const enables = wgslEnableLines(caps);
  return (enables ? enables + '\n' : '') + preprocess(source, { ...capsDefines(caps), ...extraDefines });
}
