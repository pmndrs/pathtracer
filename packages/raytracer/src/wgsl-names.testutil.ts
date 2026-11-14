/** Helpers for tests that inspect generated WGSL text: declared names, reserved words, bracket balance. */

export function declared(wgsl: string): string[] {
  return [...wgsl.matchAll(/^(?:@[^\n]*?\s)?(?:struct|fn|const|alias|override|var(?:<[^>]*>)?)\s+([A-Za-z_]\w*)/gm)].map((m) => m[1] as string);
}

export function balanced(wgsl: string): boolean {
  const depth = { '{': 0, '(': 0, '[': 0 };
  for (const ch of wgsl) {
    if (ch in depth) depth[ch as keyof typeof depth]++;
    if (ch === '}') depth['{']--;
    if (ch === ')') depth['(']--;
    if (ch === ']') depth['[']--;
    if (depth['{'] < 0 || depth['('] < 0 || depth['['] < 0) return false;
  }
  return depth['{'] === 0 && depth['('] === 0 && depth['['] === 0;
}

/** Words WGSL reserves (spec section 'Reserved Words') plus its keywords: a shader that uses one as an identifier fails to compile, and only a GPU says so. */
export const RESERVED = new Set(
  `NULL Self abstract active alignas alignof as asm asm_fragment async attribute auto await become binding_array cast catch class co_await co_return co_yield coherent column_major common compile compile_fragment concept const_cast consteval constexpr constinit crate debugger decltype delete demote demote_to_helper do dynamic_cast enum explicit export extends extern external fallthrough filter final finally friend from fxgroup get goto groupshared highp impl implements import inline instanceof interface layout lowp macro macro_rules match mediump meta mod module move mut mutable namespace new nil noexcept noinline nointerpolation noperspective null nullptr of operator package packoffset partition pass patch pixelfragment precise precision premerge priv protected pub public readonly ref regardless register reinterpret_cast require resource restrict self set shared sizeof smooth snorm static static_assert static_cast std subroutine super target template this thread_local throw trait try type typedef typeid typename typeof union unless unorm unsafe unsized use using varying virtual volatile wgsl where with writeonly yield alias break case const const_assert continue continuing default diagnostic discard else enable false fn for if let loop override requires return struct switch true var while`.split(/\s+/),
);

/** Identifiers the code declares (let, var, const, fn, struct, parameters, fields). */
export function declaredIdentifiers(wgsl: string): Set<string> {
  const names = new Set<string>();
  for (const m of wgsl.matchAll(/\b(?:let|var(?:<[^>]*>)?|const|fn|struct|alias)\s+([A-Za-z_]\w*)/g)) names.add(m[1] as string);
  for (const m of wgsl.matchAll(/\bfn\s+\w+\(([^)]*)\)/g)) for (const part of (m[1] as string).split(',')) names.add(part.split(':')[0]!.trim());
  for (const m of wgsl.matchAll(/^\s{4}([A-Za-z_]\w*):\s*[\w<>]+,?\s*(?:\/\/.*)?$/gm)) names.add(m[1] as string);
  names.delete('');
  return names;
}
