/**
 * Approach (b) of P5: a WGSL function per MaterialX material, evaluated at the hit. The generated text defines
 *
 *     fn ptMtlxApply(m: RtMaterial, id: u32, uv: vec2f, pos: vec3f, nrm: vec3f, tg: vec3f, fp: f32) -> RtMaterial
 *     fn ptMtlxNormal(id: u32, uv, pos, nrm, tg, fp) -> vec4f      (tangent-space texel in xyz, w = 1 when the material has a normal map)
 *
 * `id` is the scene-copy material index (`hit.material`); a material without a program passes through. The kernel calls `ptMtlxApply` on the
 * result of `rtApplyMaterialTextures`, so the program overrides the record's fields with the graph's value at the hit: base colour, roughness,
 * metalness, emission, opacity, the coat / sheen / transmission parameters and the rest of `RtMaterial` (thin film, diffuse roughness and
 * subsurface are not in `RtMaterial`: a varying value of those is not generated and `notes` says so). Procedural patterns are exact (position
 * and normal inputs work), image nodes read scene textures through `rtTexSample`.
 *
 * Every node's expression is the one its TypeScript twin (`MTLX_NODES[...].ts`) computes, on `vec4f` lanes; `bench/pages/materialx-check.ts`
 * runs the generated function on the GPU against `evalMtlxParams` at sample points.
 */
import { type IrOp, type MtlxIr } from './eval.ts';
import { evalMtlxParams, mtlxConstantParams, type MtlxMaterial } from './material.ts';
import { coerceWgsl, MTLX_WGSL_HELPERS, type MtlxImageRequest, type MtlxWgslContext, type NodeArgs } from './nodes.ts';
import { constantContext } from './eval.ts';
import type { MtlxSlot, OpenPbrKey } from './surface.ts';

export interface MtlxProgramInput {
  /** The scene-copy material index the program applies to. */
  material: number;
  mtl: MtlxMaterial;
}

export interface MtlxProgramsOptions {
  /** The scene texture slot of an image the graph reads (`-1` = not in the scene: the node's default). */
  textureSlot(request: MtlxImageRequest): number;
}

export interface MtlxPrograms {
  /** WGSL for `PtIntegratorOptions.materialx.wgsl`. */
  wgsl: string;
  /** Some program has a normal map: pass `materialx.normal`. */
  normal: boolean;
  /** Varying inputs that could not be generated, and other limits, one line each. */
  notes: string[];
}

type FieldKind = 'f' | 'rgb';
/** `OpenPbrKey` -> the `RtMaterial` field the program overrides. Emission is handled apart; keys not listed have no field. */
const FIELD: Partial<Record<OpenPbrKey, { field: string; kind: FieldKind; unit?: boolean }>> = {
  baseWeight: { field: 'baseWeight', kind: 'f' }, baseColor: { field: 'baseColor', kind: 'rgb' }, baseMetalness: { field: 'metalness', kind: 'f' },
  specularWeight: { field: 'specularWeight', kind: 'f' }, specularColor: { field: 'specularColor', kind: 'rgb' }, specularRoughness: { field: 'specularRoughness', kind: 'f' },
  specularIor: { field: 'ior', kind: 'f' }, specularRoughnessAnisotropy: { field: 'specularAnisotropy', kind: 'f' },
  transmissionWeight: { field: 'transmissionWeight', kind: 'f' }, transmissionColor: { field: 'transmissionColor', kind: 'rgb' }, transmissionDepth: { field: 'transmissionDepth', kind: 'f', unit: true },
  coatWeight: { field: 'coatWeight', kind: 'f' }, coatColor: { field: 'coatColor', kind: 'rgb' }, coatRoughness: { field: 'coatRoughness', kind: 'f' }, coatIor: { field: 'coatIor', kind: 'f' },
  fuzzWeight: { field: 'sheenWeight', kind: 'f' }, fuzzColor: { field: 'sheenColor', kind: 'rgb' }, fuzzRoughness: { field: 'sheenRoughness', kind: 'f' },
  geometryOpacity: { field: 'opacity', kind: 'f' },
};

/** A WGSL f32 literal that round-trips the f32 value (nine significant digits). */
export function wgslFloat(x: number): string {
  if (!Number.isFinite(x)) throw new RangeError(`WGSL literal: ${x} is not finite`);
  const s = Math.fround(x).toPrecision(9);
  const t = s.includes('e') ? s.replace(/\.?0+e/, 'e') : s.includes('.') ? s.replace(/\.?0+$/, '') : s;
  return /[.e]/.test(t) ? t : `${t}.0`;
}

class Emitter {
  readonly lines: string[] = [];
  readonly needs = new Set<string>();
  private readonly names = new Map<MtlxIr, string>();
  private count = 0;
  readonly notes: string[];
  readonly textureSlot: MtlxProgramsOptions['textureSlot'];
  constructor(options: MtlxProgramsOptions, notes: string[]) {
    this.textureSlot = options.textureSlot;
    this.notes = notes;
  }

  /** The WGSL expression (a vec4f variable or literal) of `ir`; emits its `let`s on the way. */
  expr(ir: MtlxIr): string {
    if (ir.kind === 'const') return `vec4f(${ir.value.map(wgslFloat).join(', ')})`;
    const known = this.names.get(ir);
    if (known) return known;
    const v: Record<string, string> = {};
    for (const [name, a] of Object.entries(ir.args)) v[name] = coerceWgsl(this.expr(a.ir), a.from, a.to);
    const args: NodeArgs<string> = { v, strings: ir.strings, type: ir.type, output: ir.output };
    const ctx: MtlxWgslContext = { image: (r) => this.textureSlot(r), need: (h) => this.needs.add(h) };
    const body = (ir as IrOp).impl.wgsl(args, ctx);
    const name = `n${this.count++}`;
    this.lines.push(`    let ${name} = ${body};`);
    this.names.set(ir, name);
    return name;
  }
}

const f1 = (x: number): string => wgslFloat(x);

/** The value of a slot as the WGSL expression of its field (`f32` or `vec3f`). */
function slotWgsl(e: Emitter, m: MtlxMaterial, slot: MtlxSlot, kind: FieldKind, unit: boolean): string {
  const x = e.expr(slot.ir);
  const s = slot.scale * (unit ? m.lengthScale : 1);
  const scale = s === 1 ? '' : ` * ${f1(s)}`;
  if (kind === 'rgb') return `(${x}).xyz${scale}`;
  if (slot.reduce === 'avg') return `(((${x}).x + (${x}).y + (${x}).z) / 3.0)${scale}`;
  return `(${x}).x${scale}`;
}

/** Generates the programs' WGSL. Throws when a material's normal input is not a tangent-space `normalmap`. */
export function mtlxProgramsWgsl(programs: readonly MtlxProgramInput[], options: MtlxProgramsOptions): MtlxPrograms {
  const notes: string[] = [];
  const fns: string[] = [];
  const cases: string[] = [];
  const normalCases: string[] = [];
  const needs = new Set<string>();
  let anyNormal = false;
  const seen = new Set<number>();
  for (const { material, mtl } of programs) {
    if (seen.has(material)) throw new Error(`mtlxProgramsWgsl: two programs for material ${material}`);
    seen.add(material);
    const e = new Emitter(options, notes);
    const sets: string[] = [];
    const constants = mtlxConstantParams(mtl);
    // Emission: the record keeps colour * luminance; with either varying the program multiplies the two itself.
    const emColor = mtl.varying.emissionColor, emLum = mtl.varying.emissionLuminance;
    for (const slot of Object.values(mtl.varying)) {
      const map = FIELD[slot.key];
      if (slot.key === 'emissionColor' || slot.key === 'emissionLuminance') continue;
      if (!map) {
        notes.push(`material ${material}: ${slot.key} varies but RtMaterial has no field for it; the record holds its value at uv (0.5, 0.5)`);
        continue;
      }
      sets.push(`    o.${map.field} = ${slotWgsl(e, mtl, slot, map.kind, !!map.unit)};`);
    }
    if (emColor || emLum) {
      const color = emColor ? slotWgsl(e, mtl, emColor, 'rgb', false) : `vec3f(${(Array.isArray(constants.emissionColor) ? constants.emissionColor : [1, 1, 1]).map(f1).join(', ')})`;
      const lum = emLum ? slotWgsl(e, mtl, emLum, 'f', false) : f1(typeof constants.emissionLuminance === 'number' ? constants.emissionLuminance : 0);
      sets.push(`    let emL = ${lum};`, `    o.emissionColor = ${color} * emL;`, `    o.emissionLuminance = emL;`);
    }
    if (sets.length > 0) {
      for (const h of e.needs) needs.add(h);
      fns.push(`fn ptMtlxEval${material}(m: RtMaterial, uv: vec2f, pos: vec3f, nrm: vec3f, tg: vec3f, fp: f32) -> RtMaterial {\n    var o = m;\n${e.lines.join('\n')}\n${sets.join('\n')}\n    return o;\n}\n`);
      cases.push(`        case ${material}u: {\n            return ptMtlxEval${material}(m, uv, pos, nrm, tg, fp);\n        }`);
    }
    if (mtl.normal && mtl.normal.ir.kind === 'op') {
      const nm = mtl.normal.ir;
      if (nm.category !== 'normalmap' || (nm.strings.space ?? 'tangent') !== 'tangent') throw new Error(`mtlxProgramsWgsl: material ${material}: the normal input must be a tangent-space <normalmap> (got <${nm.category}>)`);
      const en = new Emitter(options, notes);
      const inArg = nm.args.in!;
      const scaleArg = nm.args.scale;
      const x = coerceWgsl(en.expr(inArg.ir), inArg.from, inArg.to);
      const s = scaleArg ? `(${coerceWgsl(en.expr(scaleArg.ir), scaleArg.from, scaleArg.to)}).x` : '1.0';
      for (const h of en.needs) needs.add(h);
      // mx_normalmap.glsl: zero decodes to +Z, xy scaled, normalised, re-encoded for rtApplyNormalMap (which decodes texel * 2 - 1).
      fns.push(`fn ptMtlxNormal${material}(uv: vec2f, pos: vec3f, nrm: vec3f, tg: vec3f, fp: f32) -> vec4f {\n${en.lines.join('\n')}\n    let w = (${x}).xyz;\n    var d = w * 2.0 - 1.0;\n    if (w.x == 0.0 && w.y == 0.0 && w.z == 0.0) {\n        d = vec3f(0.0, 0.0, 1.0);\n    }\n    d = vec3f(d.xy * ${s}, d.z);\n    return vec4f(normalize(d) * 0.5 + 0.5, 1.0);\n}\n`);
      normalCases.push(`        case ${material}u: {\n            return ptMtlxNormal${material}(uv, pos, nrm, tg, fp);\n        }`);
      anyNormal = true;
    }
  }
  const helperText = [...needs].map((h) => MTLX_WGSL_HELPERS[h]).filter((t) => !!t).join('');
  const apply = `fn ptMtlxApply(m: RtMaterial, id: u32, uv: vec2f, pos: vec3f, nrm: vec3f, tg: vec3f, fp: f32) -> RtMaterial {\n    switch (id) {\n${cases.join('\n')}\n        default: {\n            return m;\n        }\n    }\n}\n`;
  const normalFn = anyNormal ? `fn ptMtlxNormal(id: u32, uv: vec2f, pos: vec3f, nrm: vec3f, tg: vec3f, fp: f32) -> vec4f {\n    switch (id) {\n${normalCases.join('\n')}\n        default: {\n            return vec4f(0.0);\n        }\n    }\n}\n` : '';
  return { wgsl: `${helperText}${fns.join('')}${apply}${normalFn}`, normal: anyNormal, notes };
}

/** The record values a program leaves in the scene copy for a material: the parameters at uv (0.5, 0.5), the fallback of kernels without codegen. */
export function mtlxFallbackParams(m: MtlxMaterial): ReturnType<typeof evalMtlxParams> {
  return evalMtlxParams(m, { ...constantContext(), uv: [0.5, 0.5] });
}
