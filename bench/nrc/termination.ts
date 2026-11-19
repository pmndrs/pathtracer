// Phase 2 compares display termination while keeping the training suffix and its estimator fixed.
export interface NrcTerminationOptions {
  termination: 'fixed' | 'spread' | 'diffuse-first';
  spreadThreshold: number;
  /** Largest acceptable coarse-level offset for a display hit. Omit to use every populated level. */
  maxDisplayCoarse?: number;
  /** Per-frame diagnostic atomics. Timed runs leave them off. */
  terminationStats: boolean;
}

export function terminationArgs(args: Record<string, string | undefined>): Partial<NrcTerminationOptions> {
  if (args.termination !== undefined && args.termination !== 'fixed' && args.termination !== 'spread' && args.termination !== 'diffuse-first') {
    throw new Error('termination must be fixed, spread or diffuse-first');
  }
  return {
    ...(args.termination ? { termination: args.termination } : {}),
    ...(args.spreadThreshold !== undefined ? { spreadThreshold: Number(args.spreadThreshold) } : {}),
    ...(args.maxDisplayCoarse !== undefined ? { maxDisplayCoarse: Number(args.maxDisplayCoarse) } : {}),
    terminationStats: args.terminationStats === '1',
  };
}

// The NRC megakernel traces one camera path per invocation. This private state lives across its scattering vertices.
export function candidateWgsl(policy?: NrcTerminationOptions): string {
  const spread = policy?.termination === 'spread';
  const diffuseFirst = policy?.termination === 'diffuse-first';
  return /* wgsl */ `
${diffuseFirst ? 'var<private> nrcDiffusePrimary: bool;' : ''}
${spread ? `var<private> nrcPrimaryArea: f32;
var<private> nrcSpreadRoot: f32;
var<private> nrcAttempted: bool;` : ''}
${policy?.terminationStats ? `fn nrcSurfaceBin(v: PtPathVertex) -> u32 {
    if (v.delta) { return 0u; }
    return select(2u, 1u, nrcRoughness(v) < 0.5);
}` : ''}
fn nrcDisplayCandidate(v: PtPathVertex) -> bool {
    ${policy?.terminationStats ? 'atomicAdd(&nrcCount[4u + nrcSurfaceBin(v)], 1u);' : ''}
    ${spread ? `let cosArrival = max(abs(dot(v.normal, v.wo)), 1e-6);
    if (v.depth == 0u) {
        nrcPrimaryArea = v.segmentLength * v.segmentLength / (12.5663706144 * cosArrival);
        nrcSpreadRoot = 0.0;
        nrcAttempted = false;
        return false;
    }
    if (!v.previousDelta && v.previousPdf > 0.0) {
        nrcSpreadRoot += v.segmentLength * inverseSqrt(max(v.previousPdf * cosArrival, 1e-12));
    }
    if (nrcAttempted || v.depth < nrcHook.stopDepth || v.delta ||
        nrcSpreadRoot * nrcSpreadRoot <= ${Number(policy.spreadThreshold).toExponential(9)} * nrcPrimaryArea) { return false; }
    nrcAttempted = true;
    return true;` : diffuseFirst ? `if (nrcHook.stopDepth != 1u) { return v.depth == nrcHook.stopDepth; }
    if (v.depth == 0u) { nrcDiffusePrimary = v.materialClass == PT_BSDF_DIFFUSE; }
    return v.depth == select(1u, 0u, nrcDiffusePrimary);` : 'return v.depth == nrcHook.stopDepth;'}
}`;
}
