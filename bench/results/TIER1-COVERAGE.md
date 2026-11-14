# Tier 1 feature-coverage matrix

Generated 2026-11-14 by `node bench/adapters/tier1-matrix.ts --report t1cov-1024` (P2.31, docs/design/tier1.md section 3). Data: `bench/results/reports/tier1/coverage.json`; the viewer's Coverage tab links each cell to its side-by-side comparison.

Columns ours, three-gpu-pathtracer and mutracer are **measured by ablation** at 1024 spp (noise from the 512 spp snapshot): each renders the full wave-1 scene and the same scene with the feature stripped from every material. The number after the symbol is ablation relMSE / own noise; "ffN" = N firefly pixels (above 4x the reference maximum) left out of the relMSEs. pbrt-v4 and Mitsuba 3 are **read from the exporters**: what the exporter did with a probe material of that row.

| symbol | measured columns | pbrt / Mitsuba columns |
| --- | --- | --- |
| ● | full render is at least 10x the noise away from the stripped render, and (reference class A / B) agrees with the reference: excess relMSE over own noise within 1.5 x (A) or 5 x (B) of the floor relMSE(Mitsuba, pbrt) | exact: the exporter writes it without any approximation note |
| ◐ | distinguishable but off the reference | approximate (class B layering, tint, unverified tangent) or only a coverage fallback is written (class C approximation) |
| ○ | indistinguishable from stripped (or the renderer failed) | not expressible: the exporter drops the feature |
| – | no scene yet | |

Run conditions: load average (1 min) 2.5..29.1, browser GPU processes 0..0, 14 of 66 runs flagged loaded. Timing is not the point here; no timing in this page.
Versions: repo commit 49773b4-di; browser Chrome/157.0.8103.52 headless; three-gpu-pathtracer 0.0.28; @meimu/mutracer 0.3.1.

| row | class | ours (pt-megakernel) | three-gpu-pathtracer | mutracer | pbrt-v4 | Mitsuba 3 |
| --- | --- | --- | --- | --- | --- | --- |
| core metallic-roughness: metal | A | ● 8055x | ◐ 18275x | ◐ 2455x | ● | ● |
| core dielectric over diffuse (class B) | B | ● 168x | ◐ 51x | ◐ 9443x | ◐ | ◐ |
| KHR_materials_transmission, thin-walled, smooth | A | ● 221x | ◐ 15889x | ◐ 52x | ● | ● |
| KHR_materials_volume: absorption (attenuationColor / Distance) | A | ● 158x | ◐ 564x | ◐ 29x | ● | ● |
| KHR_materials_ior | A | ● 48x | ◐ 14177x ff1 | ◐ 11x | ● | ● |
| KHR_materials_specular, factor 1 (pure Fresnel reflection) | A | ● 220x | ◐ 52534x ff1 | ◐ 64x | ● | ● |
| KHR_materials_specular, factor < 1 (class C) | C | ● 180x | ● 308x ff3 | ● 56x | ◐ | ◐ |
| KHR_materials_clearcoat (class B) | B | ● 68x | ◐ 524x | ◐ 263x | ◐ | ◐ |
| KHR_materials_sheen (class C) | C | ○ 0x | ● 1275x | ● 734x | ○ | ○ |
| KHR_materials_emissive_strength | A | ● 6235x | ● 8883x | ● 3406x | ● | ● |
| KHR_materials_iridescence (class C) (no scene yet) | C | – | – | – | ○ | ○ |
| KHR_materials_anisotropy (no scene yet) | A | – | – | – | ◐ | ◐ |
| KHR_materials_dispersion (class B') (no scene yet) | B' | – | – | – | ◐ | ○ |
| KHR_materials_diffuse_transmission (class B') (no scene yet) | B' | – | – | – | ◐ | ○ |
| transmission with a base colour tint (class B) (no scene yet) | B | – | – | – | ◐ | ◐ |
| transmission, thin-walled, rough (class C) (no scene yet) | C | – | – | – | ◐ | ◐ |
| alphaMode MASK / BLEND (no scene yet) | B | – | – | – | ○ | ○ |
| texture slots (baseColor, metalRough, emissive, normal) (no scene yet) | B | – | – | – | ◐ | ◐ |

## Per row

### core metallic-roughness: metal

- extension: core; reference class A; judged against pbrt, mitsuba
- scene: `MetalRoughSpheresNoTextures-grey-metal`, stripped `MetalRoughSpheresNoTextures-grey-metal--no-metal`
- note: stripped: baseMetalness forced to 0. the seven grey pure-metal spheres alone (the whole MetalRoughSpheresNoTextures image is 84 class B dielectric rows plus saturated gold)
- **ours (pt-megakernel)** ● full (relMSE vs pbrt 4.0e-4, vs Mitsuba 1.8e-4, floor 5.5e-4, mean bias 0.0% / 0.1%, ablation 5.0e-2, noise 6.3e-6): distinguishable from stripped: 8055x the noise (relMSE 5.0e-2 vs 6.3e-6); vs pbrt: relMSE 4.0e-4, excess over own noise 3.9e-4 <= 1.5 x floor 5.5e-4; vs mitsuba: relMSE 1.8e-4, excess over own noise 1.8e-4 <= 1.5 x floor 5.5e-4
- **three-gpu-pathtracer** ◐ approx (relMSE vs pbrt 1.7e-2, vs Mitsuba 1.9e-2, floor 5.5e-4, mean bias 1.2% / 1.2%, ablation 3.8e-2, noise 2.1e-6): distinguishable from stripped: 18275x the noise (relMSE 3.8e-2 vs 2.1e-6); vs pbrt: relMSE 1.7e-2, excess over own noise 1.7e-2 > 1.5 x floor 5.5e-4; vs mitsuba: relMSE 1.9e-2, excess over own noise 1.9e-2 > 1.5 x floor 5.5e-4
- **mutracer** ◐ approx (relMSE vs pbrt 2.3e-2, vs Mitsuba 2.6e-2, floor 5.5e-4, mean bias 1.3% / 1.4%, ablation 3.3e-2, noise 1.4e-5): distinguishable from stripped: 2455x the noise (relMSE 3.3e-2 vs 1.4e-5); vs pbrt: relMSE 2.3e-2, excess over own noise 2.3e-2 > 1.5 x floor 5.5e-4; vs mitsuba: relMSE 2.6e-2, excess over own noise 2.6e-2 > 1.5 x floor 5.5e-4
- **pbrt-v4** ● full (floor 5.5e-4): exporter: exact
- **Mitsuba 3** ● full (floor 5.5e-4): exporter: exact

### core dielectric over diffuse (class B)

- extension: core; reference class B; judged against pbrt, mitsuba
- scene: `MetalRoughSpheresNoTextures`, stripped `MetalRoughSpheresNoTextures--no-spec`
- note: stripped: dielectric specular lobe removed (specularWeight 0; written as KHR_materials_specular 0, so a consumer that ignores that extension reads as missing)
- **ours (pt-megakernel)** ● full (relMSE vs pbrt 8.6e-4, vs Mitsuba 1.1e-3, floor 1.8e-3, mean bias -0.1% / 0.8%, ablation 3.0e-2, noise 1.8e-4): distinguishable from stripped: 168x the noise (relMSE 3.0e-2 vs 1.8e-4); vs pbrt: relMSE 8.6e-4, excess over own noise 6.9e-4 <= 5 x floor 1.8e-3; vs mitsuba: relMSE 1.1e-3, excess over own noise 9.2e-4 <= 5 x floor 1.8e-3
- **three-gpu-pathtracer** ◐ approx (relMSE vs pbrt 1.3e-1, vs Mitsuba 1.6e-1, floor 1.8e-3, mean bias 13.0% / 14.0%, ablation 2.0e-3, noise 3.8e-5): distinguishable from stripped: 51x the noise (relMSE 2.0e-3 vs 3.8e-5); vs pbrt: relMSE 1.3e-1, excess over own noise 1.3e-1 > 5 x floor 1.8e-3; vs mitsuba: relMSE 1.6e-1, excess over own noise 1.6e-1 > 5 x floor 1.8e-3
- **mutracer** ◐ approx (relMSE vs pbrt 1.4e-1, vs Mitsuba 1.7e-1, floor 1.8e-3, mean bias 13.5% / 14.5%, ablation 9.1e-1, noise 9.7e-5): distinguishable from stripped: 9443x the noise (relMSE 9.1e-1 vs 9.7e-5); vs pbrt: relMSE 1.4e-1, excess over own noise 1.4e-1 > 5 x floor 1.8e-3; vs mitsuba: relMSE 1.7e-1, excess over own noise 1.7e-1 > 5 x floor 1.8e-3
- **pbrt-v4** ◐ approx (floor 1.8e-3): exporter: approximate
- **Mitsuba 3** ◐ approx (floor 1.8e-3): exporter: approximate

### KHR_materials_transmission, thin-walled, smooth

- extension: KHR_materials_transmission; reference class A; judged against pbrt, mitsuba
- scene: `TransmissionThinwallTestGrid-notex`, stripped `TransmissionThinwallTestGrid-notex--no-transmission`
- note: stripped: transmissionWeight forced to 0 (an opaque white dielectric remains)
- **ours (pt-megakernel)** ● full (relMSE vs pbrt 1.7e-3, vs Mitsuba 1.8e-3, floor 8.6e-4, mean bias 0.0% / -0.0%, ablation 3.3e-1, noise 1.5e-3): distinguishable from stripped: 221x the noise (relMSE 3.3e-1 vs 1.5e-3); vs pbrt: relMSE 1.7e-3, excess over own noise 2.3e-4 <= 1.5 x floor 8.6e-4; vs mitsuba: relMSE 1.8e-3, excess over own noise 2.7e-4 <= 1.5 x floor 8.6e-4
- **three-gpu-pathtracer** ◐ approx (relMSE vs pbrt 5.3e-3, vs Mitsuba 5.3e-3, floor 8.6e-4, mean bias -0.3% / -0.3%, ablation 3.3e-1, noise 2.1e-5): non-finite pixels excluded: 0 full, 2 stripped; distinguishable from stripped: 15889x the noise (relMSE 3.3e-1 vs 2.1e-5); vs pbrt: relMSE 5.3e-3, excess over own noise 5.2e-3 > 1.5 x floor 8.6e-4; vs mitsuba: relMSE 5.3e-3, excess over own noise 5.3e-3 > 1.5 x floor 8.6e-4
- **mutracer** ◐ approx (relMSE vs pbrt 4.8e-2, vs Mitsuba 4.8e-2, floor 8.6e-4, mean bias -0.5% / -0.6%, ablation 3.9e-1, noise 7.6e-3): distinguishable from stripped: 52x the noise (relMSE 3.9e-1 vs 7.6e-3); vs pbrt: relMSE 4.8e-2, excess over own noise 4.0e-2 > 1.5 x floor 8.6e-4; vs mitsuba: relMSE 4.8e-2, excess over own noise 4.0e-2 > 1.5 x floor 8.6e-4
- **pbrt-v4** ● full (floor 8.6e-4): exporter: exact
- **Mitsuba 3** ● full (floor 8.6e-4): exporter: exact

### KHR_materials_volume: absorption (attenuationColor / Distance)

- extension: KHR_materials_volume; reference class A; judged against mitsuba
- scene: `AttenuationTest-notex`, stripped `AttenuationTest-notex--no-attenuation`
- note: stripped: attenuation removed (clear medium, transmission kept). Mitsuba only: pbrt's RGB sigma_a uplift is wrong on saturated thick tints (V62); the floor here is relMSE(Mitsuba, pbrt) and so an upper bound
- **ours (pt-megakernel)** ● full (relMSE vs pbrt 4.4e-3, vs Mitsuba 2.3e-3, floor 5.0e-3, mean bias 1.5% / 0.0%, ablation 6.3e-2, noise 4.0e-4): distinguishable from stripped: 158x the noise (relMSE 6.3e-2 vs 4.0e-4); vs mitsuba: relMSE 2.3e-3, excess over own noise 1.9e-3 <= 1.5 x floor 5.0e-3
- **three-gpu-pathtracer** ◐ approx (relMSE vs pbrt 2.1e-2, vs Mitsuba 1.9e-2, floor 5.0e-3, mean bias -2.3% / -3.7%, ablation 6.3e-2, noise 1.1e-4): distinguishable from stripped: 564x the noise (relMSE 6.3e-2 vs 1.1e-4); vs mitsuba: relMSE 1.9e-2, excess over own noise 1.9e-2 > 1.5 x floor 5.0e-3
- **mutracer** ◐ approx (relMSE vs pbrt 1.9e-2, vs Mitsuba 1.7e-2, floor 5.0e-3, mean bias 0.1% / -1.3%, ablation 6.5e-2, noise 2.3e-3): distinguishable from stripped: 29x the noise (relMSE 6.5e-2 vs 2.3e-3); vs mitsuba: relMSE 1.7e-2, excess over own noise 1.5e-2 > 1.5 x floor 5.0e-3
- **pbrt-v4** ● full (floor 5.0e-3): exporter: exact
- **Mitsuba 3** ● full (floor 5.0e-3): exporter: exact

### KHR_materials_ior

- extension: KHR_materials_ior; reference class A; judged against pbrt, mitsuba
- scene: `IORTestGrid-notex-A`, stripped `IORTestGrid-notex-A--no-ior`
- note: stripped: every ior forced to 1.5
- **ours (pt-megakernel)** ● full (relMSE vs pbrt 5.8e-4, vs Mitsuba 5.2e-4, floor 3.1e-4, mean bias 0.0% / -0.0%, ablation 2.2e-2, noise 4.6e-4): distinguishable from stripped: 48x the noise (relMSE 2.2e-2 vs 4.6e-4); vs pbrt: relMSE 5.8e-4, excess over own noise 1.2e-4 <= 1.5 x floor 3.1e-4; vs mitsuba: relMSE 5.2e-4, excess over own noise 5.9e-5 <= 1.5 x floor 3.1e-4
- **three-gpu-pathtracer** ◐ approx (relMSE vs pbrt 2.7e-2, vs Mitsuba 2.7e-2, floor 3.1e-4, mean bias 7.7% / 7.7%, 1 firefly px excluded, ablation 4.5e-2, noise 3.2e-6): 1 firefly pixel(s) above 20.7 (4x the reference maximum, brightest 25528) left out of every relMSE; the mean bias below includes them; non-finite pixels excluded: 1 full, 0 stripped; distinguishable from stripped: 14177x the noise (relMSE 4.5e-2 vs 3.2e-6); vs pbrt: relMSE 2.7e-2, excess over own noise 2.7e-2 > 1.5 x floor 3.1e-4; vs mitsuba: relMSE 2.7e-2, excess over own noise 2.7e-2 > 1.5 x floor 3.1e-4
- **mutracer** ◐ approx (relMSE vs pbrt 2.6e-3, vs Mitsuba 2.5e-3, floor 3.1e-4, mean bias 0.2% / 0.2%, ablation 2.3e-2, noise 2.0e-3): distinguishable from stripped: 11x the noise (relMSE 2.3e-2 vs 2.0e-3); vs pbrt: relMSE 2.6e-3, excess over own noise 6.3e-4 > 1.5 x floor 3.1e-4; vs mitsuba: relMSE 2.5e-3, excess over own noise 4.9e-4 > 1.5 x floor 3.1e-4
- **pbrt-v4** ● full (floor 3.1e-4): exporter: exact
- **Mitsuba 3** ● full (floor 3.1e-4): exporter: exact

### KHR_materials_specular, factor 1 (pure Fresnel reflection)

- extension: KHR_materials_specular; reference class A; judged against pbrt, mitsuba
- scene: `IORTestGrid-notex-A`, stripped `IORTestGrid-notex-A--no-spec`
- note: stripped: dielectric specular lobe removed (specularWeight 0)
- **ours (pt-megakernel)** ● full (relMSE vs pbrt 5.8e-4, vs Mitsuba 5.2e-4, floor 3.1e-4, mean bias 0.0% / -0.0%, ablation 1.0e-1, noise 4.6e-4): distinguishable from stripped: 220x the noise (relMSE 1.0e-1 vs 4.6e-4); vs pbrt: relMSE 5.8e-4, excess over own noise 1.2e-4 <= 1.5 x floor 3.1e-4; vs mitsuba: relMSE 5.2e-4, excess over own noise 5.9e-5 <= 1.5 x floor 3.1e-4
- **three-gpu-pathtracer** ◐ approx (relMSE vs pbrt 2.7e-2, vs Mitsuba 2.7e-2, floor 3.1e-4, mean bias 7.7% / 7.7%, 1 firefly px excluded, ablation 1.7e-1, noise 3.2e-6): 1 firefly pixel(s) above 20.7 (4x the reference maximum, brightest 25528) left out of every relMSE; the mean bias below includes them; non-finite pixels excluded: 1 full, 0 stripped; distinguishable from stripped: 52534x the noise (relMSE 1.7e-1 vs 3.2e-6); vs pbrt: relMSE 2.7e-2, excess over own noise 2.7e-2 > 1.5 x floor 3.1e-4; vs mitsuba: relMSE 2.7e-2, excess over own noise 2.7e-2 > 1.5 x floor 3.1e-4
- **mutracer** ◐ approx (relMSE vs pbrt 2.6e-3, vs Mitsuba 2.5e-3, floor 3.1e-4, mean bias 0.2% / 0.2%, ablation 1.3e-1, noise 2.0e-3): distinguishable from stripped: 64x the noise (relMSE 1.3e-1 vs 2.0e-3); vs pbrt: relMSE 2.6e-3, excess over own noise 6.3e-4 > 1.5 x floor 3.1e-4; vs mitsuba: relMSE 2.5e-3, excess over own noise 4.9e-4 > 1.5 x floor 3.1e-4
- **pbrt-v4** ● full (floor 3.1e-4): exporter: exact
- **Mitsuba 3** ● full (floor 3.1e-4): exporter: exact

### KHR_materials_specular, factor < 1 (class C)

- extension: KHR_materials_specular; reference class C; no reference
- scene: `IORTestGrid-notex-C`, stripped `IORTestGrid-notex-C--no-specfactor`
- note: stripped: specularFactor 0.25 forced to 1. no reference: both exporters scale F0 through the IOR, so agreement between them says nothing
- **ours (pt-megakernel)** ● full (ablation 6.7e-3, noise 3.8e-5): non-finite pixels excluded: 0 full, 1 stripped; distinguishable from stripped: 180x the noise (relMSE 6.7e-3 vs 3.8e-5); no reference to judge against: distinguishable counts as supported
- **three-gpu-pathtracer** ● full (3 firefly px excluded, ablation 6.5e-3, noise 2.1e-5): 3 firefly pixel(s) above 4.7 (4x the reference maximum, brightest 7064) left out of every relMSE; the mean bias below includes them; non-finite pixels excluded: 3 full, 3 stripped; distinguishable from stripped: 308x the noise (relMSE 6.5e-3 vs 2.1e-5); no reference to judge against: distinguishable counts as supported
- **mutracer** ● full (ablation 4.3e-3, noise 7.7e-5): distinguishable from stripped: 56x the noise (relMSE 4.3e-3 vs 7.7e-5); no reference to judge against: distinguishable counts as supported
- **pbrt-v4** ◐ approx: exporter: coverage; coverage: pbrt: material 'm': class C: specular weight < 1 or specular colour != 1: glTF scales F0 only, OpenPBR scales F at both ends, no reference expresses either; dropped or approximated; coverage: pbrt: material 'm': specular_weight/specular_color scale F0 only through an IOR of 1.2222 (OpenPBR scales F at both ends)
- **Mitsuba 3** ◐ approx: exporter: coverage; coverage: mitsuba: material 'm': class C: specular weight < 1 or specular colour != 1: glTF scales F0 only, OpenPBR scales F at both ends, no reference expresses either; dropped or approximated; coverage: mitsuba: material 'm': specular_weight/specular_color scale F0 only through an IOR of 1.2222 (OpenPBR scales F at both ends)

### KHR_materials_clearcoat (class B)

- extension: KHR_materials_clearcoat; reference class B; judged against pbrt, mitsuba
- scene: `ClearCoatTest-notex`, stripped `ClearCoatTest-notex--no-coat`
- note: stripped: coatWeight forced to 0
- **ours (pt-megakernel)** ● full (relMSE vs pbrt 2.4e-3, vs Mitsuba 2.4e-4, floor 3.1e-3, mean bias -0.8% / 0.1%, ablation 1.3e-2, noise 1.9e-4): distinguishable from stripped: 68x the noise (relMSE 1.3e-2 vs 1.9e-4); vs pbrt: relMSE 2.4e-3, excess over own noise 2.2e-3 <= 5 x floor 3.1e-3; vs mitsuba: relMSE 2.4e-4, excess over own noise 4.2e-5 <= 5 x floor 3.1e-3
- **three-gpu-pathtracer** ◐ approx (relMSE vs pbrt 3.3e-2, vs Mitsuba 5.5e-2, floor 3.1e-3, mean bias 5.4% / 6.4%, ablation 2.2e-2, noise 4.3e-5): distinguishable from stripped: 524x the noise (relMSE 2.2e-2 vs 4.3e-5); vs pbrt: relMSE 3.3e-2, excess over own noise 3.3e-2 > 5 x floor 3.1e-3; vs mitsuba: relMSE 5.5e-2, excess over own noise 5.5e-2 > 5 x floor 3.1e-3
- **mutracer** ◐ approx (relMSE vs pbrt 2.3e-2, vs Mitsuba 4.5e-2, floor 3.1e-3, mean bias 4.5% / 5.5%, ablation 1.1e-2, noise 4.2e-5): distinguishable from stripped: 263x the noise (relMSE 1.1e-2 vs 4.2e-5); vs pbrt: relMSE 2.3e-2, excess over own noise 2.3e-2 > 5 x floor 3.1e-3; vs mitsuba: relMSE 4.5e-2, excess over own noise 4.5e-2 > 5 x floor 3.1e-3
- **pbrt-v4** ◐ approx (floor 3.1e-3): exporter: approximate; pbrt: material 'm': clear coat on a dielectric: coateddiffuse with the coat IOR/roughness as the interface; the base dielectric specular lobe is not added
- **Mitsuba 3** ◐ approx (floor 3.1e-3): exporter: approximate; mitsuba: material 'm': clear coat on a dielectric: roughplastic (nonlinear) with the coat IOR/roughness; the base dielectric specular lobe is not added

### KHR_materials_sheen (class C)

- extension: KHR_materials_sheen; reference class C; no reference
- scene: `SheenTestGrid-notex`, stripped `SheenTestGrid-notex--no-fuzz`
- note: stripped: fuzzWeight forced to 0. no reference (neither pbrt nor Mitsuba has a sheen lobe)
- **ours (pt-megakernel)** ○ missing (relMSE vs pbrt 4.4e-4, vs Mitsuba 5.5e-4, floor 6.4e-4, mean bias 0.1% / 0.5%, ablation 0.0e+0, noise 1.3e-4): full and stripped renders are not distinguishable: relMSE 0.0e+0 is 0.0x the noise 1.3e-4 (needs >= 10x)
- **three-gpu-pathtracer** ● full (relMSE vs pbrt 2.2e-1, vs Mitsuba 2.5e-1, floor 6.4e-4, mean bias 10.9% / 11.3%, ablation 6.2e-2, noise 4.9e-5): distinguishable from stripped: 1275x the noise (relMSE 6.2e-2 vs 4.9e-5); no reference to judge against: distinguishable counts as supported
- **mutracer** ● full (relMSE vs pbrt 2.5e-1, vs Mitsuba 2.7e-1, floor 6.4e-4, mean bias 11.2% / 11.6%, ablation 9.3e-2, noise 1.3e-4): distinguishable from stripped: 734x the noise (relMSE 9.3e-2 vs 1.3e-4); no reference to judge against: distinguishable counts as supported
- **pbrt-v4** ○ missing (floor 6.4e-4): exporter: none; coverage: pbrt: material 'm': class C: fuzz (sheen) has no reference; dropped or approximated; dropped: only a coverage fallback is written, the feature is absent
- **Mitsuba 3** ○ missing (floor 6.4e-4): exporter: none; coverage: mitsuba: material 'm': class C: fuzz (sheen) has no reference; dropped or approximated; dropped: only a coverage fallback is written, the feature is absent

### KHR_materials_emissive_strength

- extension: KHR_materials_emissive_strength; reference class A; judged against pbrt, mitsuba
- scene: `EmissiveStrengthTest-notex`, stripped `EmissiveStrengthTest-notex--no-strength`
- note: stripped: emissive strength forced to 1
- **ours (pt-megakernel)** ● full (relMSE vs pbrt 4.4e-4, vs Mitsuba 5.2e-4, floor 1.8e-4, mean bias 0.0% / 0.0%, ablation 2.5e+0, noise 4.0e-4): distinguishable from stripped: 6235x the noise (relMSE 2.5e+0 vs 4.0e-4); vs pbrt: relMSE 4.4e-4, excess over own noise 3.3e-5 <= 1.5 x floor 1.8e-4; vs mitsuba: relMSE 5.2e-4, excess over own noise 1.1e-4 <= 1.5 x floor 1.8e-4
- **three-gpu-pathtracer** ● full (relMSE vs pbrt 4.2e-4, vs Mitsuba 5.1e-4, floor 1.8e-4, mean bias -0.4% / -0.4%, ablation 2.5e+0, noise 2.8e-4): distinguishable from stripped: 8883x the noise (relMSE 2.5e+0 vs 2.8e-4); vs pbrt: relMSE 4.2e-4, excess over own noise 1.4e-4 <= 1.5 x floor 1.8e-4; vs mitsuba: relMSE 5.1e-4, excess over own noise 2.2e-4 <= 1.5 x floor 1.8e-4
- **mutracer** ● full (relMSE vs pbrt 7.8e-4, vs Mitsuba 8.6e-4, floor 1.8e-4, mean bias -0.1% / -0.1%, ablation 2.5e+0, noise 7.4e-4): distinguishable from stripped: 3406x the noise (relMSE 2.5e+0 vs 7.4e-4); vs pbrt: relMSE 7.8e-4, excess over own noise 4.1e-5 <= 1.5 x floor 1.8e-4; vs mitsuba: relMSE 8.6e-4, excess over own noise 1.2e-4 <= 1.5 x floor 1.8e-4
- **pbrt-v4** ● full (floor 1.8e-4): exporter: exact
- **Mitsuba 3** ● full (floor 1.8e-4): exporter: exact

### KHR_materials_iridescence (class C)

- extension: KHR_materials_iridescence; reference class C; no reference
- scene: no scene yet
- **ours (pt-megakernel)** – no-scene: no wave-1 scene yet
- **three-gpu-pathtracer** – no-scene: no wave-1 scene yet
- **mutracer** – no-scene: no wave-1 scene yet
- **pbrt-v4** ○ missing: exporter: none; coverage: pbrt: material 'm': class C: thin film (iridescence) has no reference; dropped or approximated; dropped: only a coverage fallback is written, the feature is absent
- **Mitsuba 3** ○ missing: exporter: none; coverage: mitsuba: material 'm': class C: thin film (iridescence) has no reference; dropped or approximated; dropped: only a coverage fallback is written, the feature is absent

### KHR_materials_anisotropy

- extension: KHR_materials_anisotropy; reference class A; no reference
- scene: no scene yet
- **ours (pt-megakernel)** – no-scene: no wave-1 scene yet
- **three-gpu-pathtracer** – no-scene: no wave-1 scene yet
- **mutracer** – no-scene: no wave-1 scene yet
- **pbrt-v4** ◐ approx: exporter: approximate; pbrt: material 'm': anisotropic: the GGX tangent axis is the renderer's own dp/du of the mesh uvs, not the glTF TANGENT attribute (unverified, P2.26)
- **Mitsuba 3** ◐ approx: exporter: approximate; mitsuba: material 'm': anisotropic: the GGX tangent axis is the renderer's own dp/du of the mesh uvs, not the glTF TANGENT attribute (unverified, P2.26)

### KHR_materials_dispersion (class B')

- extension: KHR_materials_dispersion; reference class B'; no reference
- scene: no scene yet
- **ours (pt-megakernel)** – no-scene: no wave-1 scene yet
- **three-gpu-pathtracer** – no-scene: no wave-1 scene yet
- **mutracer** – no-scene: no wave-1 scene yet
- **pbrt-v4** ◐ approx: exporter: approximate; pbrt: material 'm': dispersion: spectrum eta from a Cauchy fit n = A + B/lambda^2 (A = 1.46208, B = 0.013092 um^2) of n_d = 1.5 and Abbe 20.000; pbrt terminates the secondary wavelengths at each refraction
- **Mitsuba 3** ○ missing: exporter: none; coverage: mitsuba: material 'm': dispersion (transmission_dispersion_scale > 0) has no Mitsuba equivalent; wrote a constant IOR of 1.5 (no dispersion); dropped: only a coverage fallback is written, the feature is absent

### KHR_materials_diffuse_transmission (class B')

- extension: KHR_materials_diffuse_transmission; reference class B'; no reference
- scene: no scene yet
- **ours (pt-megakernel)** – no-scene: no wave-1 scene yet
- **three-gpu-pathtracer** – no-scene: no wave-1 scene yet
- **mutracer** – no-scene: no wave-1 scene yet
- **pbrt-v4** ◐ approx: exporter: approximate; pbrt: material 'm': diffusetransmission has no specular lobe: exact only with specular_weight 0; the dielectric specular is dropped here (class B')
- **Mitsuba 3** ○ missing: exporter: none; coverage: mitsuba: material 'm': thin-walled diffuse transmission (subsurface_weight > 0) has no Mitsuba equivalent; wrote the diffuse reflection only (no transmitted light); dropped: only a coverage fallback is written, the feature is absent

### transmission with a base colour tint (class B)

- extension: KHR_materials_transmission; reference class B; no reference
- scene: no scene yet
- **ours (pt-megakernel)** – no-scene: no wave-1 scene yet
- **three-gpu-pathtracer** – no-scene: no wave-1 scene yet
- **mutracer** – no-scene: no wave-1 scene yet
- **pbrt-v4** ◐ approx: exporter: approximate; pbrt: material 'm': transmission_color tint at depth 0 is not expressible (neither renderer tints a dielectric): exported untinted, compare white-base only
- **Mitsuba 3** ◐ approx: exporter: approximate; mitsuba: material 'm': transmission_color tint at depth 0 is not expressible (neither renderer tints a dielectric): exported untinted, compare white-base only

### transmission, thin-walled, rough (class C)

- extension: KHR_materials_transmission; reference class C; no reference
- scene: no scene yet
- **ours (pt-megakernel)** – no-scene: no wave-1 scene yet
- **three-gpu-pathtracer** – no-scene: no wave-1 scene yet
- **mutracer** – no-scene: no wave-1 scene yet
- **pbrt-v4** ◐ approx: exporter: coverage; coverage: pbrt: material 'm': class C: thin-walled rough transmission has no reference; dropped or approximated; coverage: pbrt: material 'm': rough thin-walled transmission has no reference: smooth thindielectric written
- **Mitsuba 3** ◐ approx: exporter: coverage; coverage: mitsuba: material 'm': class C: thin-walled rough transmission has no reference; dropped or approximated; coverage: mitsuba: material 'm': rough thin-walled transmission has no reference: smooth thindielectric written

### alphaMode MASK / BLEND

- extension: core; reference class B; no reference
- scene: no scene yet
- **ours (pt-megakernel)** – no-scene: no wave-1 scene yet
- **three-gpu-pathtracer** – no-scene: no wave-1 scene yet
- **mutracer** – no-scene: no wave-1 scene yet
- **pbrt-v4** ○ missing: exporter: none; coverage: pbrt: material 'm': alpha mask ignored (rendered opaque); dropped: only a coverage fallback is written, the feature is absent
- **Mitsuba 3** ○ missing: exporter: none; coverage: mitsuba: material 'm': alpha mask ignored (rendered opaque); dropped: only a coverage fallback is written, the feature is absent

### texture slots (baseColor, metalRough, emissive, normal)

- extension: core; reference class B; no reference
- scene: no scene yet
- **ours (pt-megakernel)** – no-scene: no wave-1 scene yet
- **three-gpu-pathtracer** – no-scene: no wave-1 scene yet
- **mutracer** – no-scene: no wave-1 scene yet
- **pbrt-v4** ◐ approx: exporter: coverage; coverage: pbrt: material 'm': texture 'baseColorTexture' dropped (the factor alone is exported)
- **Mitsuba 3** ◐ approx: exporter: coverage; coverage: mitsuba: material 'm': texture 'baseColorTexture' dropped (the factor alone is exported)

