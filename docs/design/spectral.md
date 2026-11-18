# Spectral transport and dispersion (P5, TASKS P5.s1-s5, P2.28)

Opt-in: `integrator.spectral: true` (or `{ res, dispersion }`, page arg `pt_spectral=1|nodisp`, `pt_spectral_res=N`). Megakernel only. Off (the default) every
kernel is byte-identical to before (`node bench/size-snapshot.ts`, 224 variants compared). D7 said "spectral is v2, radiance is carried as vec4 from day
one"; this is that v2, kept cheap: it changes how a path is *parameterised*, not how it scatters.

## 1. The idea in one paragraph

A path carries three wavelengths (`ptLam`, nm) instead of three colour channels. Everything that was an RGB parameter at a vertex (base colour, specular /
coat / sheen / transmission colour, conductor eta and k, emission, the environment and the sun) is replaced by the value of that colour's spectrum at the
three wavelengths, so the existing BSDF, layering, MIS and medium code runs unchanged on "channels" that are wavelengths: `beta`, `radiance`, `f`, `pdf` are
the same `vec3f`. At the end of the path the three lane radiances are weighted by the CIE 1931 matching functions over the wavelength density and become
the sample's linear Rec.709 RGB. The accumulator, the film, the denoiser inputs and every consumer of the image see ordinary RGB.

```
sample u (Owen-Sobol pair 1, x)  ->  l0 = F^-1(u), l1 = F^-1(u + 1/3), l2 = F^-1(u + 2/3)         (pbrt SampleVisibleWavelengths)
hit:   mat' = ptSpecMaterial(mat)        colours -> sigmoid-polynomial spectra at l0..l2; emission / sky / sun x D65
       ...the unchanged BSDF / NEE / MIS code...
end:   XYZ = 1/3 sum_k  L_k cmf(l_k) / p(l_k),    RGB = M_norm XYZ
```

## 2. Pieces and where they live (`packages/pathtracer/src/spectral/`)

| file | what |
|---|---|
| `cie1931.ts` | CIE 1931 2 degree CMFs and D65, 360-780 nm at 5 nm (moved from `bench/metrics`; the thin-film oracle still imports it) |
| `rgb2spec.ts` | the RGB -> sigmoid-polynomial table (our own Gauss-Newton fit of the method of Jakob and Hanika 2019), pbrt's lookup, the kernel texture layout |
| `wavelengths.ts` | hero wavelengths, the pdf, the lane -> RGB estimator, the Cauchy dispersion law (TS twins of the WGSL) |
| `spectral-wgsl.ts` | the kernel text: `ptSpecBegin`, `ptSpecCoeffs`, `ptSpecRefl`, `ptSpecUnbounded`, `ptSpecIllum`, `ptSpecMaterial`, `ptSpecToRgb`, `ptSpecIor` |

`ptIntegratorWgsl` calls `ptSpecBegin` once per sample (after the pixel seed), wraps `rtMaterial(...)` in `ptSpecMaterial`, `ptEscaped`'s environment / sun
terms and `ptSampleLd`'s `li.radiance` in `ptSpecIllum`, and returns `ptSpecToRgb(radiance)`. Nothing else in the integrator changes. Not combined with hooks,
frame, AOVs or ReSTIR (`resolveIntegratorOptions` throws): their planes and callbacks are RGB. The wavefront throws (`ptwfUnsupported`): its path state has no
room for the wavelengths yet (they are a pure function of (pixel, sample), so they can be recomputed per queue entry; open).

## 3. Uplift

*Reflectances* (`ptSpecRefl`): a table of `(c0, c1, c2)` indexed like pbrt's (`[max component][z][y][x]`, z on `smoothstep(smoothstep(k/(res-1)))` nodes),
`s(t) = sigmoid(c0 t^2 + c1 t + c2)`, `t = (l - 360)/470`. A grey is a constant spectrum (analytic, no table read). Values are clamped to [0, 1].
*Unbounded* colours (conductor eta / k, an absorber's sigma): `2 max` times the uplift of `rgb / (2 max)` (pbrt `RGBUnboundedSpectrum`). *Illuminants*
(`ptSpecIllum`): the unbounded spectrum times `D65 / Yd` (pbrt `RGBIlluminantSpectrum`), `Yd` = integral of D65 ybar, so `(1, 1, 1)` is the D65 white with Y = 1.

The table is fitted by `buildRgb2SpecTable(res)` (Gauss-Newton on a CIELAB residual under D65, 3 coefficients, warm-started along z and across the grid, a
damped step): res 32 takes 0.5 s, res 64 (the default, pbrt's resolution) 4 s in Node. `node bench/spectral-table.ts` writes it to `bench/assets/generated/`
(git-ignored) and the adapter loads it; otherwise the kernel fits it on first use. It is uploaded as ONE `rgba32float` texture (`res^2 x (3 res + 1)`, read
with `textureLoad`, eight loads per non-grey colour): the coefficient block, and below it a row with the CMF and `D65 / Yd` at 5 nm. Binding: group 2,
binding `PT_SPEC_BINDING` = 15 (no new storage buffer; the Apple 10-buffer limit is untouched).

Conventions (`rgb2spec.ts`, one place): `XYZ = integral S D65 cmf / Yd`; `RGB = M_norm XYZ`, the Rec.709 matrix with each row scaled so the table's D65 white is
exactly (1, 1, 1): a constant spectrum `c` is the grey `(c, c, c)` and a furnace stays a furnace. The fit uses the same matrix, so uplift -> light with white
-> film is the identity up to the table error (measured below).

## 4. Estimator and noise

Wavelengths: pbrt's `SampleVisibleWavelengths` (density `0.0039398042 / cosh^2(0.0072 (l - 538))` on [360, 830] nm, closed-form inverse CDF), one random number
`u` per sample rotated by thirds, so each lane is marginally distributed by the density and the average of the three single-wavelength estimates is unbiased
(`spectral.test.ts`: white and coloured emitters average to their RGB to 1e-2 over 20000 stratified `u`). pbrt uses four lanes; the existing vec3 state gives
three (DIVERGES V250). Hero sampling is not free: even a grey path now has colour noise (the three wavelengths see the spectrum of a coloured surface
differently). Measured at equal spp in section 7.

## 5. Dispersion (P2.28)

`RtMaterial.dispersion` (KHR_materials_dispersion = 20 / Abbe number; OpenPBR `transmission_dispersion_scale x 20 / abbe_number`) lives in the high half of
material-record word 28 (unorm16 of `dispersion / 8`; the half was spare, a record without dispersion packs the same words as before). `ptMaterialDispersion(id)`
reads it (it is not a member of the WGSL `RtMaterial`: that struct is in every kernel). A dispersive dielectric uses a Cauchy law `n(l) = A + B / l^2` through
`ior` at the d line (587.56 nm) with `V_d = 20 / dispersion` (`B = (n_d - 1) / V_d / (1/l_F^2 - 1/l_C^2)`): the same law and Abbe lines as the pbrt exporter's
`cauchyFit`, which writes `spectrum eta`. The BSDF is evaluated at the hero lane's IOR, and at the first dispersive hit the secondary lanes are terminated
(`beta *= (3, 0, 0)`; pbrt's `TerminateSecondary`), so the 3-lane average is the hero's single-wavelength estimate. Radiance already gathered before that
hit keeps its three lanes. A kernel built with `dispersion: false` has none of this code (`spectral: { dispersion: false }`).

## 6. Media and layers in the spectral mode

An absorbing interior (`transmissionDepth > 0`) follows pbrt: the RGB sigma (`-ln(color) / depth`) is uplifted as an *unbounded* spectrum and exponentiated
(`exp(-sigma(l) depth)`), so a saturated tint is spectral exactly where pbrt's `volpath` is (V62), and Mitsuba's per-channel closed form is the other
reference (DIVERGES V254). Layered dielectrics, coated metals, sheen: the code is the same, on wavelength lanes (a base colour at the three wavelengths,
a coat colour, the conductor's eta and k). Thin film is still the Belcour-Barla RGB model of P2.27 (it does not read the lanes' wavelengths; an exact spectral
film is a follow-up, V257).

## 7. Results (2026-11-18, M5 Pro, Chrome; 1024 spp seed 0, references pbrt-v4 4096 spp, DispersionTest 1024 spp)

Timings in this file are **provisional**: the Mac was shared (load average 25-50, other agents' pages on the one GPU lock). Image statistics are not affected.

**Colour round trip** (`spectral-swatches-*`, `bench/adapters/spectral-swatches.ts`): 24 colours of a 6 x 4 grid, reflectance under a white sky and emitters (4x), plus a coloured sky.
Every cell within 0.6% of its RGB (the errors at the film edge, +0.004 absolute, are the same in the RGB render: scene box edges, not the uplift); the saturated
swatches (gold (1, 0.78, 0.34), (0.02, 0.02, 0.9), (0.9, 0.02, 0.02), (0.85, 0.7, 0)) within 0.1%. Sky (2.4, 0.8, 0.05) reads (2.3988, 0.7995, 0.0489). Grey scenes: furnace-diffuse
0.5000, furnace-dielectric 0.9999, furnace-conductor 0.6929 (RGB 0.6928); relMSE of a furnace is noise only (1.8e-5 spectral vs 2e-7 RGB: see noise).

**Spectral vs RGB vs pbrt-v4** (whole image, mean ratio - 1 per channel, R / G / B):

| scene | RGB vs pbrt | spectral vs pbrt | spectral vs RGB |
|---|---|---|---|
| cornell | +0.39 / +0.56 / +2.04% | **+0.02 / -0.03 / +0.07%** | -0.37 / -0.59 / -1.93% |
| env-sphere | -0.90 / +0.06 / -0.32% | **+0.00 / +0.02 / +0.00%** | +0.91 / -0.04 / +0.32% |
| ClearCoatTest-notex (coated dielectric over a saturated red) | -3.02 / +0.77 / +0.04% | **+0.05 / +0.02 / -0.01%** | +3.17 / -0.75 / -0.05% |
| ClearCoatTest-notex-metal (coated gold) | +1.06 / -0.09 / +0.03% | **+0.01 / +0.00 / -0.00%** | -1.03 / +0.09 / -0.03% |
| MetalRoughSpheresNoTextures (gold rows) | +0.02 / -0.05 / -0.23% | **+0.01 / +0.01 / +0.00%** | -0.01 / +0.05 / +0.23% |
| IORTestGrid-notex-A | +0.04 / +0.04 / +0.01% | +0.01 / +0.01 / -0.01% | -0.02 / -0.03 / -0.02% |
| SheenTestGrid-notex (class B base, not sheen) | -0.82 / +4.49 / -0.58% | -0.67 / +4.17 / -0.71% | +0.15 / -0.30 / -0.13% |
| furnaces (3) | exact | exact | 0.00% |

The spectral render sits within 0.1% of pbrt wherever the RGB render was off by pbrt's spectral uplift (the Cornell walls, the sky map's saturated sun, the saturated red base under a coat, the
coated gold). Against Mitsuba (RGB) it moves the other way by the same amount, as expected (Cornell -0.4 / -0.6 / -1.9%). Residual gaps that are NOT uplift and stay: SheenTestGrid (+4.2% green: the class B
base, coateddiffuse vs roughplastic, P2.32a), ClearCoatTest-notex-metal vs Mitsuba (principled clear coat). relMSE against pbrt: cornell 4.8e-4 -> 1.2e-4 (floor 4.4e-4), ClearCoatTest-notex 2.4e-3 -> 7.4e-4, env-sphere 6.9e-4 -> 4.7e-4.
The gold rows of MetalRoughSpheres whose blue pbrt-vs-Mitsuba gap was 12-25% (P2.32a) are now within 0.01% of pbrt on the whole image. AttenuationTest (saturated absorbers) was not re-rendered: the sigma uplift
is implemented as pbrt does it (V254) but has no measurement yet (P5.s3 follow-up).

**Dispersion** (all within the spectral mode):

| scene | spectral vs pbrt relBias R / G / B | relMSE spectral | relMSE without dispersion (`pt_spectral=nodisp`) |
|---|---|---|---|
| dispersion-prism | +0.01 / +0.01 / +0.01% | 1.5e-3 | 1.6e-1 (the fringes are the difference) |
| dispersion-prism-off | -0.00 / -0.00 / -0.00% (RGB mode: -0.00%) | 1.5e-4 (RGB 3.8e-5) | |
| DispersionTest-notex (25 prisms, pbrt 1024 spp) | -0.02 / +0.02 / +0.00% | 1.2e-2 | 7.1e-2 (relBias +0.14 / -0.19 / -0.71%) |

The picket-through-prism render shows the same red / cyan fringes as pbrt (`scratchpad` comparison image, side by side). Without dispersion the spectral mode equals the RGB render in the mean (prism-off, 0.6848 in all channels).

**Noise at equal spp** (two seeds each, relMSE of the two independent renders; chroma noise = variance of r / (r + g + b) x 1e4):

| scene | RGB relMSE | spectral relMSE | ratio | chroma noise ratio |
|---|---|---|---|---|
| cornell | 1.14e-4 | 1.98e-4 | 1.73x | 3.9x |
| MetalRoughSpheresNoTextures | 3.52e-4 | 5.34e-4 | 1.52x | 3.9x |
| ClearCoatTest-notex | 3.76e-4 | 1.08e-3 | 2.89x | 3.3x |
| dispersion-prism-off (grey, vs pbrt) | 3.8e-5 | 1.5e-4 | 4x | |

Hero sampling makes colour noise 3-4x at equal spp (a grey scene has colour noise that RGB mode does not have), luminance-weighted relMSE 1.5-3x. That is the price of spectral on a 3-lane path; a fourth lane (P5.s7) or
stratifying the wavelengths across the pixel's samples against each other (they already are across spp by the Owen-Sobol pair) are the knobs. A denoiser works on the colour noise well.

**Cost.** Kernel text +4.8 KB on the plain kernel (40.8 -> 45.5 KB; env 43.5 -> 48.3; textured 44.3 -> 49.1; the OpenPBR albedo kernel 49.5 -> 54.3 KB, stochastic 50.7 -> 55.5 KB: over the 50 KB DXC class, as those kernels already
were). Provisional ms per spp on a loaded machine (RGB -> spectral): Cornell 13.3 -> 21.3, furnace-diffuse 0.37 -> 0.42, env-sphere 1.74 -> 1.94, MetalRoughSpheres 6.9 -> 6.8, IORTestGrid 7.1 -> 7.3, SheenTestGrid 16.5 -> 16.8: roughly
1.0-1.2x where materials are few; the table fetches (8 per non-grey colour per vertex) show on small saturated scenes. Quiet-pass ratio is P5.s8.

## 8. Open

- Wavefront (P5.s7), a fourth lane, exact spectral thin film and dispersive coats (P5.s6), AttenuationTest with spectral sigma vs pbrt, lazy per-lobe uplift (cost), the quiet-pass timing (P5.s8).
- The uplift treats every colour as an independent sigmoid spectrum: two colours multiplied along a path (red wall x red wall) are the product of spectra, which is what pbrt does and RGB mode cannot (that is the colour shift of the Cornell inter-reflections).
