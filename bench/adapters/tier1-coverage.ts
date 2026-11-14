/**
 * Tier 1 feature-coverage matrix, the pure part (P2.31, docs/design/tier1.md section 3): row definitions, the cell rules and the Markdown page.
 * No I/O and no page imports, so `tier1-coverage.test.ts` runs without a GPU; `tier1-matrix.ts` feeds it the measured numbers.
 *
 * Cell rules (columns ours, three-gpu-pathtracer, mutracer, measured by ablation; pbrt-v4 and Mitsuba 3, read from the exporters' notes):
 *  - A renderer is run on the full scene and on `<scene>--no-<feature>` (the feature stripped from every material). `noise` is the relMSE between
 *    its full render at N spp and its own N/2 spp snapshot (for independent halves that equals the relMSE of two independent renders of N spp:
 *    E|S_N - S_N/2|^2 = var(S_N), so it estimates the renderer's own noise). `ablation` is relMSE(full, stripped). Distinguishable = ablation >= 10 x noise.
 *  - ○ missing: not distinguishable, or the renderer failed.
 *  - ● supported: distinguishable and, for reference classes A and B, the excess relMSE vs the reference (relMSE(full, ref) - noise) is within
 *    `k x floor` (k 1.5 for class A as in the tier1.md section 6 gate, 5 for class B, the "reference-bracketing" bound of section 7), floor = relMSE(Mitsuba, pbrt).
 *    Classes C and none have no reference: distinguishable is enough.
 *  - ◐ approximate: distinguishable but off the reference.
 */

export const SENSITIVITY = 10;
export const AGREE_A = 1.5;
export const AGREE_B = 5;

export type CellStatus = 'full' | 'approx' | 'missing' | 'no-scene';
export const SYMBOL: Record<CellStatus, string> = { full: '●', approx: '◐', missing: '○', 'no-scene': '–' };
export const RENDERER_IDS = ['ours', 'tgpt', 'mutracer', 'pbrt', 'mitsuba'] as const;
export type RendererColumn = (typeof RENDERER_IDS)[number];
export const RENDERER_TITLES: Record<RendererColumn, string> = { ours: 'ours (pt-megakernel)', tgpt: 'three-gpu-pathtracer', mutracer: 'mutracer', pbrt: 'pbrt-v4', mitsuba: 'Mitsuba 3' };
/** Adapter id of each measured column. */
export const ADAPTER_OF: Record<'ours' | 'tgpt' | 'mutracer', string> = { ours: 'pt-megakernel', tgpt: 'tgpt-wavefront', mutracer: 'mutracer' };

export type RefClass = 'A' | 'B' | "B'" | 'C' | 'none';

export interface CoverageRowDef {
  id: string;
  label: string;
  /** glTF extension(s) or `core`. */
  extension: string;
  refClass: RefClass;
  /** Wave-1 scene the competitors and ours run (full); absent = no scene yet. */
  scene?: string;
  /** Ablation tag (`bench/scenes/tier1/ablations.ts`); the stripped scene is `<scene>--no-<tag>`. */
  ablation?: string;
  /** References that judge the cell, in order (the first is the primary). Default both, pbrt first. */
  refs?: ('pbrt' | 'mitsuba')[];
  /** Why a reference does not apply (shown in the notes). */
  refNote?: string;
  /** OpenPBR parameters of the probe material the native exporters are asked about (pbrt, Mitsuba). */
  probe: Record<string, unknown>;
  /** What is stripped, one line, for the page. */
  strip?: string;
}

export const ROWS: readonly CoverageRowDef[] = [
  { id: 'core-metal', label: 'core metallic-roughness: metal', extension: 'core', refClass: 'A', scene: 'MetalRoughSpheresNoTextures-grey-metal', ablation: 'metal', refs: ['pbrt', 'mitsuba'], refNote: 'the seven grey pure-metal spheres alone (the whole MetalRoughSpheresNoTextures image is 84 class B dielectric rows plus saturated gold)', probe: { baseMetalness: 1, baseColor: [0.9, 0.9, 0.9], specularRoughness: 0.3 }, strip: 'baseMetalness forced to 0' },
  { id: 'core-dielectric', label: 'core dielectric over diffuse (class B)', extension: 'core', refClass: 'B', scene: 'MetalRoughSpheresNoTextures', ablation: 'spec', probe: { baseMetalness: 0, specularRoughness: 0.4 }, strip: 'dielectric specular lobe removed (specularWeight 0; written as KHR_materials_specular 0, so a consumer that ignores that extension reads as missing)' },
  { id: 'transmission-thin', label: 'KHR_materials_transmission, thin-walled, smooth', extension: 'KHR_materials_transmission', refClass: 'A', scene: 'TransmissionThinwallTestGrid-notex', ablation: 'transmission', probe: { transmissionWeight: 1, geometryThinWalled: true, specularRoughness: 0 }, strip: 'transmissionWeight forced to 0 (an opaque white dielectric remains)' },
  { id: 'volume-absorption', label: 'KHR_materials_volume: absorption (attenuationColor / Distance)', extension: 'KHR_materials_volume', refClass: 'A', scene: 'AttenuationTest-notex', ablation: 'attenuation', refs: ['mitsuba'], refNote: 'Mitsuba only: pbrt\'s RGB sigma_a uplift is wrong on saturated thick tints (V62); the floor here is relMSE(Mitsuba, pbrt) and so an upper bound', probe: { transmissionWeight: 1, transmissionColor: [0.1, 0.5, 0.9], transmissionDepth: 1, specularRoughness: 0 }, strip: 'attenuation removed (clear medium, transmission kept)' },
  { id: 'ior', label: 'KHR_materials_ior', extension: 'KHR_materials_ior', refClass: 'A', scene: 'IORTestGrid-notex-A', ablation: 'ior', probe: { baseColor: [0, 0, 0], specularIor: 1.76, specularRoughness: 0 }, strip: 'every ior forced to 1.5' },
  { id: 'specular-1', label: 'KHR_materials_specular, factor 1 (pure Fresnel reflection)', extension: 'KHR_materials_specular', refClass: 'A', scene: 'IORTestGrid-notex-A', ablation: 'spec', probe: { baseColor: [0, 0, 0], specularWeight: 1, specularRoughness: 0 }, strip: 'dielectric specular lobe removed (specularWeight 0)' },
  { id: 'specular-lt1', label: 'KHR_materials_specular, factor < 1 (class C)', extension: 'KHR_materials_specular', refClass: 'C', scene: 'IORTestGrid-notex-C', ablation: 'specfactor', refNote: 'no reference: both exporters scale F0 through the IOR, so agreement between them says nothing', probe: { baseColor: [0, 0, 0], specularWeight: 0.25, specularRoughness: 0 }, strip: 'specularFactor 0.25 forced to 1' },
  { id: 'clearcoat', label: 'KHR_materials_clearcoat (class B)', extension: 'KHR_materials_clearcoat', refClass: 'B', scene: 'ClearCoatTest-notex', ablation: 'coat', probe: { baseColor: [0.5, 0.02, 0.01], coatWeight: 1, coatRoughness: 0.03, specularRoughness: 0.44 }, strip: 'coatWeight forced to 0' },
  { id: 'sheen', label: 'KHR_materials_sheen (class C)', extension: 'KHR_materials_sheen', refClass: 'C', scene: 'SheenTestGrid-notex', ablation: 'fuzz', refNote: 'no reference (neither pbrt nor Mitsuba has a sheen lobe)', probe: { baseColor: [0, 0, 0.5], fuzzWeight: 1, fuzzColor: [1, 1, 1], fuzzRoughness: 0.5, specularRoughness: 0.75 }, strip: 'fuzzWeight forced to 0' },
  { id: 'emissive-strength', label: 'KHR_materials_emissive_strength', extension: 'KHR_materials_emissive_strength', refClass: 'A', scene: 'EmissiveStrengthTest-notex', ablation: 'strength', probe: { baseColor: [0, 0, 0], emissionLuminance: 4, emissionColor: [0.1, 0.5, 0.9] }, strip: 'emissive strength forced to 1' },
  // No wave-1 scene yet: the competitor and our cells read "no scene yet"; pbrt and Mitsuba come from the exporters.
  { id: 'iridescence', label: 'KHR_materials_iridescence (class C)', extension: 'KHR_materials_iridescence', refClass: 'C', probe: { thinFilmWeight: 1, thinFilmThickness: 0.4, thinFilmIor: 1.3, specularRoughness: 0.1 } },
  { id: 'anisotropy', label: 'KHR_materials_anisotropy', extension: 'KHR_materials_anisotropy', refClass: 'A', probe: { baseMetalness: 1, specularRoughness: 0.5, specularRoughnessAnisotropy: 0.9 } },
  { id: 'dispersion', label: "KHR_materials_dispersion (class B')", extension: 'KHR_materials_dispersion', refClass: "B'", probe: { transmissionWeight: 1, transmissionDispersionScale: 1, transmissionDispersionAbbeNumber: 20, specularRoughness: 0 } },
  { id: 'diffuse-transmission', label: "KHR_materials_diffuse_transmission (class B')", extension: 'KHR_materials_diffuse_transmission', refClass: "B'", probe: { subsurfaceWeight: 1, geometryThinWalled: true, baseColor: [0, 0.25, 0.25], specularRoughness: 0.5 } },
  { id: 'transmission-tint', label: 'transmission with a base colour tint (class B)', extension: 'KHR_materials_transmission', refClass: 'B', probe: { transmissionWeight: 1, transmissionColor: [0.5, 0.7, 0.9], specularRoughness: 0 } },
  { id: 'transmission-thin-rough', label: 'transmission, thin-walled, rough (class C)', extension: 'KHR_materials_transmission', refClass: 'C', probe: { transmissionWeight: 1, geometryThinWalled: true, specularRoughness: 0.5 } },
  { id: 'alpha-mask', label: 'alphaMode MASK / BLEND', extension: 'core', refClass: 'B', probe: { alphaMode: 'mask', specularRoughness: 0.5 } },
  { id: 'textures', label: 'texture slots (baseColor, metalRough, emissive, normal)', extension: 'core', refClass: 'B', probe: { baseColorTexture: 'tex', specularRoughness: 0.5 } },
];

export interface Cell {
  status: CellStatus | 'error';
  symbol: string;
  /** relMSE of the full render against each reference (measured columns, rows with a scene). */
  relMSE_pbrt?: number;
  relMSE_mitsuba?: number;
  /** relMSE(Mitsuba, pbrt) of the scene (the reference floor). */
  floor?: number;
  /** relMSE(full, stripped). */
  ablationRelMSE?: number;
  /** The renderer's own noise at the run's spp (relMSE(full, half-spp snapshot)). */
  noise?: number;
  /** ablationRelMSE / noise. */
  ratio?: number;
  /** Pixels of the full render above the firefly cap (4 x the reference's maximum); they are left out of every relMSE here (and counted). */
  fireflyPixels?: number;
  /** Mean RGB of the full render over the mean of the reference, minus 1, per reference. */
  relBias_pbrt?: number;
  relBias_mitsuba?: number;
  notes: string[];
  /** Viewer query string (`a=...&b=...&r=...`) of the comparison behind the cell, or null. */
  reportRef: string | null;
}

export interface CellInput {
  refClass: RefClass;
  ablation: number;
  noise: number;
  /** One per judging reference: excess is checked against `k x floor`. */
  refs: { name: string; relMSE: number; floor: number }[];
}

export interface CellVerdict { status: 'full' | 'approx' | 'missing'; ratio: number; notes: string[]; refPass: boolean | null }

const sci = (v: number): string => v.toExponential(1);

/** The cell rules (see the module comment). Pure. */
export function judgeCell(input: CellInput): CellVerdict {
  const ratio = input.noise > 0 ? input.ablation / input.noise : input.ablation > 0 ? Infinity : 0;
  const notes: string[] = [];
  if (!(ratio >= SENSITIVITY)) {
    notes.push(`full and stripped renders are not distinguishable: relMSE ${sci(input.ablation)} is ${Number.isFinite(ratio) ? ratio.toFixed(1) : String(ratio)}x the noise ${sci(input.noise)} (needs >= ${SENSITIVITY}x)`);
    return { status: 'missing', ratio, notes, refPass: null };
  }
  notes.push(`distinguishable from stripped: ${ratio.toFixed(0)}x the noise (relMSE ${sci(input.ablation)} vs ${sci(input.noise)})`);
  if ((input.refClass !== 'A' && input.refClass !== 'B') || input.refs.length === 0) {
    notes.push('no reference to judge against: distinguishable counts as supported');
    return { status: 'full', ratio, notes, refPass: null };
  }
  const k = input.refClass === 'A' ? AGREE_A : AGREE_B;
  let pass = true;
  for (const r of input.refs) {
    const excess = r.relMSE - input.noise;
    const ok = excess <= k * r.floor;
    if (!ok) pass = false;
    notes.push(`vs ${r.name}: relMSE ${sci(r.relMSE)}, excess over own noise ${sci(excess)} ${ok ? '<=' : '>'} ${k} x floor ${sci(r.floor)}`);
  }
  return { status: pass ? 'full' : 'approx', ratio, notes, refPass: pass };
}

// ---------------------------------------------------------------------------------------------------------------
// Native cells (pbrt-v4, Mitsuba 3): from what the exporters did with a probe material
// ---------------------------------------------------------------------------------------------------------------

export type NativeOutcome = 'exact' | 'approximate' | 'coverage' | 'none';

export interface NativeProbeResult {
  /** The exporter ran without the `coverage` option. */
  ranStrict: boolean;
  /** It ran with `coverage` (only tried when the strict run threw). */
  ranCoverage: boolean;
  /** Reference class of the probe material (`classifyOpenPbr`). */
  cls: RefClass;
  /** `approximations` of the run that succeeded (strict if it ran, else coverage). */
  notes: string[];
  error?: string;
}

/** What the exporter's coverage notes say: approximated (`coverage`) or dropped altogether (`none`). */
const COVERAGE_APPROX = [/through an IOR/, /smooth thindielectric written/];
const COVERAGE_DROPPED = [/no dispersion/, /no transmitted light/, /rendered opaque/, /dropped or approximated/];
const STRICT_APPROX = [/not expressible/, /unverified/, /exact only with/];

export function nativeOutcome(p: NativeProbeResult): { outcome: NativeOutcome; status: CellStatus; notes: string[] } {
  const keep = p.notes.filter((n) => /coverage:|not expressible|unverified|exact only|clear coat|dispersion|tint|has no|ignored/.test(n));
  if (p.ranStrict) {
    const approx = p.cls !== 'A' || p.notes.some((n) => STRICT_APPROX.some((re) => re.test(n)));
    return { outcome: approx ? 'approximate' : 'exact', status: approx ? 'approx' : 'full', notes: keep };
  }
  if (!p.ranCoverage) return { outcome: 'none', status: 'missing', notes: [`exporter cannot express it even with coverage: ${p.error ?? ''}`.trim()] };
  const cov = p.notes.filter((n) => n.startsWith('coverage:'));
  const approx = cov.some((n) => COVERAGE_APPROX.some((re) => re.test(n)));
  if (approx) return { outcome: 'coverage', status: 'approx', notes: keep };
  const dropped = cov.length > 0 && cov.every((n) => COVERAGE_DROPPED.some((re) => re.test(n)));
  return dropped ? { outcome: 'none', status: 'missing', notes: [...keep, 'dropped: only a coverage fallback is written, the feature is absent'] } : { outcome: 'coverage', status: 'approx', notes: keep };
}

// ---------------------------------------------------------------------------------------------------------------
// The page
// ---------------------------------------------------------------------------------------------------------------

export interface CoverageRow {
  id: string;
  label: string;
  extension: string;
  refClass: RefClass;
  scene: string | null;
  ablation: string | null;
  stripped: string | null;
  refs: string[];
  noScene: boolean;
  note?: string;
  cells: Record<RendererColumn, Cell>;
}

export interface CoverageJson {
  schema: 1;
  generated: string;
  report: string;
  spp: number;
  halfSpp: number;
  thresholds: { sensitivity: number; agreeA: number; agreeB: number };
  conditions: { load1?: { min: number; max: number }; gpuProcs?: { min: number; max: number }; loadedRuns: number; runs: number; versions: Record<string, string> };
  legend: Record<string, string>;
  rows: CoverageRow[];
}

const cellText = (c: Cell): string => {
  if (c.status === 'no-scene') return '–';
  const bits = [c.symbol];
  if (c.ratio !== undefined && Number.isFinite(c.ratio)) bits.push(`${c.ratio.toFixed(0)}x`);
  else if (c.ratio === Infinity) bits.push('inf');
  if (c.fireflyPixels) bits.push(`ff${c.fireflyPixels}`);
  return bits.join(' ');
};

export function renderMarkdown(j: CoverageJson): string {
  const L: string[] = [];
  L.push('# Tier 1 feature-coverage matrix');
  L.push('');
  L.push(`Generated ${j.generated} by \`node bench/adapters/tier1-matrix.ts --report ${j.report}\` (P2.31, docs/design/tier1.md section 3). Data: \`bench/results/reports/tier1/coverage.json\`; the viewer's Coverage tab links each cell to its side-by-side comparison.`);
  L.push('');
  L.push(`Columns ours, three-gpu-pathtracer and mutracer are **measured by ablation** at ${j.spp} spp (noise from the ${j.halfSpp} spp snapshot): each renders the full wave-1 scene and the same scene with the feature stripped from every material. The number after the symbol is ablation relMSE / own noise; "ffN" = N firefly pixels (above 4x the reference maximum) left out of the relMSEs. pbrt-v4 and Mitsuba 3 are **read from the exporters**: what the exporter did with a probe material of that row.`);
  L.push('');
  L.push('| symbol | measured columns | pbrt / Mitsuba columns |');
  L.push('| --- | --- | --- |');
  L.push(`| ● | full render is at least ${j.thresholds.sensitivity}x the noise away from the stripped render, and (reference class A / B) agrees with the reference: excess relMSE over own noise within ${j.thresholds.agreeA} x (A) or ${j.thresholds.agreeB} x (B) of the floor relMSE(Mitsuba, pbrt) | exact: the exporter writes it without any approximation note |`);
  L.push('| ◐ | distinguishable but off the reference | approximate (class B layering, tint) or only a coverage fallback is written (class C approximation) |');
  L.push('| ○ | indistinguishable from stripped (or the renderer failed) | not expressible: the exporter drops the feature |');
  L.push('| – | no scene yet | |');
  L.push('');
  if (j.conditions.load1) L.push(`Run conditions: load average (1 min) ${j.conditions.load1.min.toFixed(1)}..${j.conditions.load1.max.toFixed(1)}${j.conditions.gpuProcs ? `, browser GPU processes ${j.conditions.gpuProcs.min}..${j.conditions.gpuProcs.max}` : ''}, ${j.conditions.loadedRuns} of ${j.conditions.runs} runs flagged loaded. Timing is not the point here; no timing in this page.`);
  L.push(`Versions: ${Object.entries(j.conditions.versions).map(([k, v]) => `${k} ${v}`).join('; ')}.`);
  L.push('');
  L.push(`| row | class | ${RENDERER_IDS.map((r) => RENDERER_TITLES[r]).join(' | ')} |`);
  L.push(`| --- | --- | ${RENDERER_IDS.map(() => '---').join(' | ')} |`);
  for (const r of j.rows) L.push(`| ${r.label}${r.noScene ? ' (no scene yet)' : ''} | ${r.refClass} | ${RENDERER_IDS.map((id) => cellText(r.cells[id])).join(' | ')} |`);
  L.push('');
  L.push('## Per row');
  for (const r of j.rows) {
    L.push('');
    L.push(`### ${r.label}`);
    L.push('');
    L.push(`- extension: ${r.extension}; reference class ${r.refClass}${r.refs.length ? `; judged against ${r.refs.join(', ')}` : '; no reference'}`);
    L.push(`- scene: ${r.scene ? `\`${r.scene}\`, stripped \`${r.stripped}\`` : 'no scene yet'}${r.note ? `\n- note: ${r.note}` : ''}`);
    for (const id of RENDERER_IDS) {
      const c = r.cells[id];
      const nums: string[] = [];
      if (c.relMSE_pbrt !== undefined) nums.push(`relMSE vs pbrt ${sci(c.relMSE_pbrt)}`);
      if (c.relMSE_mitsuba !== undefined) nums.push(`vs Mitsuba ${sci(c.relMSE_mitsuba)}`);
      if (c.floor !== undefined) nums.push(`floor ${sci(c.floor)}`);
      if (c.relBias_pbrt !== undefined) nums.push(`mean bias ${(c.relBias_pbrt * 100).toFixed(1)}% / ${(c.relBias_mitsuba! * 100).toFixed(1)}%`);
      if (c.fireflyPixels) nums.push(`${c.fireflyPixels} firefly px excluded`);
      if (c.ablationRelMSE !== undefined) nums.push(`ablation ${sci(c.ablationRelMSE)}`);
      if (c.noise !== undefined) nums.push(`noise ${sci(c.noise)}`);
      L.push(`- **${RENDERER_TITLES[id]}** ${c.symbol} ${c.status}${nums.length ? ` (${nums.join(', ')})` : ''}${c.notes.length ? `: ${c.notes.join('; ')}` : ''}`);
    }
  }
  L.push('');
  return `${L.join('\n')}\n`;
}
