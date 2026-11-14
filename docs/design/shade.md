# L2 `rtShade` (P1.3)

One-bounce shading of a hit for consumers that want radiance at a hit without a path tracer: reflections' hit shading,
DDGI probe rays, radiance-cache updates. Code: `packages/raytracer/src/shade/` (`shade.ts` options + TypeScript twin,
`shade-wgsl.ts` the WGSL). Check page: `bench/pages/shade-check.ts`
(`node bench/runner/run.ts --page shade-check`). Spec: §2 table, row L2 SHADE1; D3 and D4 (this repo owns shading and
the cache *argument*, `@pmndrs/gi` owns the caches).

## API

Wire it through the inline module; everything is plain functions you call from your own kernel:

```ts
const code = rtModuleWgsl(caps, {
  ...format, scene: { slot, layout: copy.layout },
  shade: { cache: 'external' },      // true = defaults; see RtShadeOptions
  minify: true,                      // the module with shade is 50.2 KB unminified (indexed leaves), 42.2 KB minified
});
```

```wgsl
let hit = rtClosest(ray, RT_MASK_ALL);
let s   = rtShade(hit, ray, RT_SHADE_LAMBERT, vec3f(u0, u1, u2), RT_MASK_ALL);
let L   = rtShadeTotal(s);            // emission + direct + indirect
```

| Function | |
|---|---|
| `rtShade(hit, ray, mode, u, mask) -> RtShaded` | radiance toward `-ray.direction` at the hit. A miss returns the environment in `emission` (`rtShadeMiss`). |
| `rtShadePoint(hit, ray) -> RtShadePoint` | the surface decode: world position, shading and geometric normal **on the viewer's side**, `wo`, decoded material, diffuse albedo. What the hooks receive. |
| `rtShadeAt(point, mode, u, mask)` | shade a point you already decoded (several lights/modes per hit, or a cache that wants the point). |
| `rtShadeTotal(s)`, `rtShadeMiss(dir)`, `rtShadeDiffuseFromIrradiance(p, E)`, `rtOffsetOrigin(p, n)`, `rtSideNormal(n, w)` | helpers |

`RtShaded { emission, direct, indirect: vec3f, light: u32, occluded: bool }`: the three terms stay separate (a denoiser
or a cache writer wants them apart); `light` is the sampled light (`RT_NONE` if none contributed); `occluded` is "a
shadow ray was traced and blocked", which the check page uses to replay visibility into the twin.

**`u`** is three independent numbers in [0, 1) from the caller's sampler: `u.x` picks the light (alias table), `u.yz`
the point on it. The leftover of the alias pick is not used, so the dimensions are really independent. The caller owns
the sampler (Owen-Sobol per pixel for a progressive consumer, blue noise for a real-time one); the module has no state.

**Modes** (`depth` in the spec), a runtime argument, so one pipeline serves all three:

| Mode | Rays | Content |
|---|---|---|
| `RT_SHADE_FLAT` | 0 | `emission + indirect` (the cache lookup). With `flat: 'albedo'`: `emission + baseColor * baseWeight` unlit, in `direct`. |
| `RT_SHADE_LAMBERT` | ≤ 1 shadow | `emission + direct(Lambert: albedo/π) + indirect`. `albedo = baseColor·baseWeight·(1−metalness)·(1−transmission)`: a pure metal is black here. |
| `RT_SHADE_FULL` | ≤ 1 shadow | as Lambert but the BSDF is diffuse + GGX specular (builtin) or yours (`bsdf: 'external'`). |

**Estimator.** One light from the alias table (power-weighted: triangles, sun, environment), one point/direction on it
(`rtSampleLightLi`: uniform triangle area, sun cone or delta, uniform sphere or cosine hemisphere for the sky), one
shadow ray (`rtVisible`, any-hit, `mask` is the ray mask): `direct = Le · f · |cos| / (pmf · pdf)`. **No MIS**: there is
no BSDF sample inside `rtShade`; the consumer's continuation ray is the BSDF sample and ends in its own `rtShade` (the
path tracer's MIS between the two needs the path state and stays in `@pmndrs/pathtracer`). Unbiased; noise is that of
single-sample NEE. Shadow ray origins use Wächter and Binder's integer-ulp offset along the geometric normal
(`rtOffsetOrigin`, the same code the path tracer has as `ptOffsetOrigin`), ends are offset the same way on the light.

**Emission** is the hit material's `emissionColor` on the authored front face (`emissionTwoSided` for both). Back-face
hits are shaded as two-sided surfaces: normals flip toward the viewer.

**Not done.** Textures: the material's texture slots are not sampled (P1.4c), so base colour and roughness are the
constants. Environment maps: constant radiance only (P1.4d). The 24 B compact batch hit has no material or normal, so
`rtShade` needs the 40 B record (or the inline `rtClosest`).

## The two hooks (D3: the cache seam)

WGSL has no virtual functions, so a hook is a function the module either defines (default) or expects you to define:

```ts
shade: { cache: 'external', bsdf: 'external' }     // the module then leaves rtShadeIndirect and rtShadeBsdf out
```
```wgsl
// anywhere in the final shader (declaration order is free in WGSL):
fn rtShadeIndirect(p: RtShadePoint) -> vec3f { ... }          // outgoing radiance toward p.wo, already BSDF-weighted
fn rtShadeBsdf(p: RtShadePoint, wi: vec3f) -> vec3f { ... }   // f(wo, wi) without the cosine; FULL only
```

- **`rtShadeIndirect`**, default `vec3f(0.0)`. Called in every mode, once, with the decoded point. It returns outgoing
  radiance, so a radiance cache (GI-1.x world hash cache) returns its value times whatever it needs, and an irradiance
  cache (DDGI probes) calls `rtShadeDiffuseFromIrradiance(p, E)` = `albedo · E / π`. gi's types never appear here:
  `RtShadePoint` is the whole interface. Consumers bring their own bindings by declaring them in their own text.
- **`rtShadeBsdf`**, default builtin. For `bsdf: 'external'` the check page plugs in the path tracer's BSDF
  (`ptBsdfEval(ptBsdfFromMaterial(p.material), wo, wi).f` in the shading frame, 7 lines) and agrees with the twin using
  the same hook to 6e-6.

## Full mode: builtin BSDF or a hook (decision per D11: both are built, `bsdf` setting)

Builtin (`rtShadeBsdf` in `shade-wgsl.ts`, twin `builtinBsdf`): Lambert diffuse `albedo/π` + GGX specular
`D·G·F/(4 cosθo cosθi)`, GGX `D`, height-correlated Smith `G` (as pbrt/pt), Schlick `F`, `alpha = max(roughness², minAlpha)`.
`F` = dielectric `specularWeight · specularColor · (F0d + (1−F0d)(1−v·h)⁵)` with `F0d = ((ior−1)/(ior+1))²`, mixed with
a metal term (`baseColor·baseWeight`, or the normal-incidence reflectance of `conductorEta/K`) by `metalness`. It is
continuous in metalness, and `specularWeight 0` is exactly Lambert (the bench `diffuse` material).

Verification of the builtin model (unit tests, no GPU): white metal at normal incidence integrates to 0.9957 / 0.9158 /
0.3068 at roughness 0.25 / 0.5 / 1, equal to the path tracer BSDF's documented furnace values (`docs/design/bsdf.md`) to
1e-3; reciprocal to 1e-9; GGX `D` normalised to 1e-3; a rough dielectric stays under albedo 1.

## Verification (M5 Pro, Chrome 157, 2026-11-14; load average 4.4, one GPU process)

Unit tests (`pnpm test`, 22 new): the twin against closed forms (Lambert plane under a quad light: form factor, 1.2% at
60k samples; delta sun + sky; sun cone vs delta), modes (FLAT never calls visibility, ≤ 1 ray otherwise), emission and
back-face rules, both hooks, the BSDF facts above, WGSL (no reserved identifier in any of the 16 option variants, no
duplicate or non-`rt` names, balanced, minify keeps lines and tokens, size).

`shade-check` page, **0 mismatches**:

| Check | Result |
|---|---|
| GPU `rtShade` vs the f64 twin: Cornell, quad-light, plastic, copper, half-metal (sun + sky), modes flat/lambert/full, `envSampling` uniform and cosine, 4096 jittered primary rays each (≈ 3,808 hits), random `u` | max relative error 5.9e-5 (Cornell), 2.6e-5 (plastic), 1.6e-5 (copper); 0 light-pick or occlusion flips |
| Same with both hooks external (irradiance cache stand-in + `ptBsdfEval`), 4 scenes × 3 modes | max rel 6.3e-6 |
| Lambert point under the 1 m quad light (Le 10, albedo 0.5), 2^18 samples | 1.1974 vs closed form 1.1973 (+0.011%) |
| Lambert plane, delta sun + uniform sky, uniform / cosine sky sampler | −0.047% / −0.044% (expected 0.46686) |
| Same, sun disc radius 0.05 | −0.099% / −0.095% |
| Cornell, 64² pixels × 1024 spp, `rtShade` vs `PtMegakernel` maxDepth 1 (an independent estimator with MIS) | mean −0.046% (lambert), +0.13% (full); per-pixel z rms 0.86 / 0.85, max 3.9 / 3.3 over 4096 pixels (noise only) |
| Kernel (shade + walk + scene + the check kernel), minified | 42,742 B (unminified module + shade: 48.9 KB records, 50.2 KB indexed; shade 7.1 KB, the light sampling it pulls in 3.7 KB) |

A/B, builtin BSDF vs the path tracer's (same rays, same light samples, FULL, mean of direct):
copper (eta/k) 1.0015× (|diff| 0.15% of the mean; Schlick vs exact complex Fresnel); plastic (red, roughness 0.3) 1.027×
(|diff| 2.7%, by design: the path tracer's adapter treats a non-metal as pure Lambert, the builtin adds the 4% specular lobe
OpenPBR has). Half metal is not comparable: the adapter thresholds metalness at 0.5.

Throughput, **best effort, not a result** (load 4.4, wall clock around 16 passes × 512² Cornell primary samples, each a
closest hit plus ≤ 1 shadow ray, min of 5 interleaved rounds; no readback inside the timed region): flat 2.4 ms
(2.4–3.2), lambert 4.2 ms (4.2–5.6), full 4.3 ms (4.3–4.6) ≈ 1750 / 1000 / 975 M shaded samples/s. The shadow ray costs
about 1.75× the primary hit on this tiny scene; Lambert vs Full is within the noise. Sponza and the 3060 are for the
quiet pass.

## Open choices (each one is a setting or a settled seam; defaults hold until measured)

### 1. Full BSDF: builtin or a consumer hook (`bsdf: 'builtin' | 'external'`)

**What it is.** FULL needs a BSDF. Either raytracer carries a small one, or it asks the consumer for one.
**Suggestion: builtin default; hook for fidelity.** The builtin keeps `@pmndrs/raytracer` standalone for reflections and
DDGI (they must not depend on the path tracer); the hook lets a consumer that links `@pmndrs/pathtracer` get the exact same
BSDF as the path tracer, or its own.

| | Builtin | Hook |
|---|---|---|
| Pros | no dependency, ~40 lines, continuous OpenPBR-like (metalness, specular weight), matches pt on conductors to 0.15% | any model (pt's exact Fresnel, dielectrics, later layered/sheen), one source of truth with pt |
| Cons | a second BSDF to maintain; approximations (Schlick, no transmission, no coat/sheen, no multiple-scattering); differs from pt on non-metals by design | consumer must wire it; the adapter cost is on the consumer; raytracer's own tests can't cover FULL |

- **Speed:** builtin ≈ hook in cost here (Lambert vs Full within noise); a hook with `ptBsdfEval`'s complex Fresnel costs more per sample (unmeasured).
- **Quality:** the one measured gap: plastic +2.7%, copper +0.15% vs pt. Reflections would show it on glossy plastics.
- **Setting:** yes, already (`bsdf`).
- **Settle it:** P1.5 renders a Tier 1 scene (a plastic + metal one) with both through the bench viewer and compares to the pt reference;
  suggested follow-up: export a `ptShadeBsdfHook()` from `@pmndrs/pathtracer` (the 7 lines in `shade-check.ts`) so a consumer needs no glue.

### 2. Sky sampling: uniform sphere or cosine hemisphere (`envSampling`)

**What it is.** When the light pick lands on the environment, the direction is either uniform over the sphere (what the path
tracer's NEE does; half of the directions are wasted below the surface) or cosine-weighted over the shading hemisphere.
**Suggestion: cosine for LAMBERT, uniform until FULL is A/B'd** (a cosine lobe is a poor match for a glossy lobe). Today the default is 'uniform' for both.

| | Uniform sphere | Cosine hemisphere |
|---|---|---|
| Pros | same distribution as the path tracer's NEE (so MIS weights there stay valid); no frame needed | zero variance for a Lambertian surface under a uniform sky; never wastes a sample below the surface |
| Cons | high variance indoors/outdoors alike | needs the shading normal (a basis); not the pt's pdf; poor for glossy lobes |

- **Speed:** cosine adds a basis + a concentric map (a few ALU ops). **Quality:** measured on the sun+sky plane (sun 3 W/m², sky 0.2), per-sample standard deviation 0.2966 (uniform) vs 0.2265 (cosine): 24% lower, i.e. 1.7× fewer samples for the same noise. Most of what remains is the sun-vs-sky pick (selection noise).
- **Setting:** yes (`envSampling`). **Settle it:** relMSE at equal sample count on `env-sky` and a glossy plane, FULL and LAMBERT, both settings; flip the default for the winner. A further variant (not built): one cosine sample shared with sun selection (pick the sun deterministically when the plane faces it).

### 3. The indirect hook's contract: outgoing radiance or irradiance

**Suggestion: outgoing radiance, with `rtShadeDiffuseFromIrradiance` for irradiance caches** (as built).

| | Radiance (built) | Irradiance `rtShadeIrradiance(p) -> vec3f`, we multiply by `albedo/π` |
|---|---|---|
| Pros | works for any cache (radiance cache, probes, SH, a constant ambient); the consumer can weight by its own BSDF; one signature | trivial for DDGI; the albedo (and the future `(1−F)` factors) stay in one place |
| Cons | an irradiance consumer writes one helper call | a radiance cache would divide the albedo out only to have it multiplied back; specular lookups don't fit |

- **Speed/quality:** none either way (one multiply). **Setting:** no, it is a seam; a second hook could be added next to this one without breaking it.
- **Settle it:** write the first two real consumers: webgpu-ddgi's probe shading (irradiance) and the gi world cache (radiance). If both are happy with the helper, keep.

### 4. What FLAT means (`flat: 'cache' | 'albedo'`)

**Suggestion: `'cache'`** (emission + indirect, zero rays), because flat then is "the cheap tier of the same function" for a consumer with a cache.
`'albedo'` returns the unlit base colour in `direct` (a debug/preview reading of "flat").

| | cache | albedo |
|---|---|---|
| Pros | with a cache it is a complete (diffuse GI) shade at zero rays | works with no cache at all; useful for debugging |
| Cons | with the default (black) cache a non-emissive hit is black | not physically meaningful; unlit |

- **Speed:** same. **Quality:** n/a. **Setting:** yes. **Settle it:** reflections' first integration (P1.5): if it wants a no-ray fallback without a cache, make `albedo` the default when `cache` is `'none'`.

### 5. Kernel size: `minify` default

The module with shade is 48.9 KB (records) to 50.2 KB (indexed leaves) unminified, over the ~50 KB DXC cap once a kernel is
added; minified it is 41.1 to 42.2 KB. **Suggestion: keep `minify` off by default** (readable generated WGSL for error line
numbers; they stay valid either way) and have consumers who use shade turn it on; the pt megakernel already does this itself.
- **Speed/quality:** compile time only. **Setting:** yes. **Settle it:** the Windows DXC compile time of the unminified vs minified kernel (W-task).

### 6. Number of light samples per call

The spec says 1 NEE ray. A consumer that wants more calls `rtShadeAt` several times with different `u`
(`rtShadeAt` takes the decoded point, so decoding is paid once). **Suggestion: keep 1.** More samples per hit are a
consumer policy (reflections at half resolution may want 2). **Setting:** no (loop in the consumer). **Settle it:** P1.5 equal-ray-budget relMSE, 1 sample × N hits vs N samples × 1 hit.

### 7. `minAlpha` floor on the builtin lobe (default 0.01)

One light sample on a near-mirror lobe is a hit-or-miss lottery (fireflies). The floor makes roughness below ~0.1 look
blurrier than it is. **Suggestion: 0.01.** **Speed:** none. **Quality:** only visible on near-mirrors (which reflections
traces as real mirror rays anyway). **Setting:** yes (`minAlpha`). **Settle it:** firefly count and max FLIP on a polished metal sphere, 0.001 vs 0.01 vs 0.05.

### 8. Shadow-ray ends and the sun disc in a miss

`sunVisible` (default false) mirrors the path tracer: the sun disc is in `rtShadeMiss` only if asked. Suggestion: false (the
references' sun is a delta light and invisible; reflections of a visible disc need an energy-aware blur anyway). Setting: yes.

## Follow-ups

- P1.5: reflections' `TraceProvider` shades hits with `rtShade` (hit record → `RtHit`; needs the full 40 B record).
- Export `ptShadeBsdfHook()` from `@pmndrs/pathtracer`; migrate pt's `ptOffsetOrigin` to `rtOffsetOrigin` (one copy).
- Migrate pt's `rtLightSampleWgsl()` append to `scene.sampling` / `shade` (it is pulled in by `shade: true`; appending it as well declares it twice).
- Textures (P1.4c) and environment maps (P1.4d) change `RtShadePoint` inputs and the sky sampler; the hooks are unaffected.
