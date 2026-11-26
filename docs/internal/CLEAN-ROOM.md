# Clean room: how we keep this repo free of code we may not use

The single place for this repo's clean-room rules, fences and procedures (D35). It applies to people and to every agent,
in this repo and when we research for its siblings. CLAUDE.md ("Licences and clean room") carries the short form;
`THIRD_PARTY.md` records what we did port. The Denoiser's rules
(`~/Developer/Denoiser/docs/research/rt-denoiser/RULES.md`) apply here in full; this page adds what this repo does on top.

## 1. Why

Both packages ship under MIT and are meant for commercial use. NVIDIA's real-time SDKs (NRD, RTXDI, RTXGI, RTXPT, SHaRC)
ship under the NVIDIA RTX SDKs License: use and object-code distribution inside an application, all rights reserved in every
source file, and §4(e) bans anything that would put the SDK under an open-source licence (Denoiser
`sources/nrd-provenance.md` §1.1). NRD covers exactly what our real-time mode and the Denoiser sibling build, so one
transcribed line, or a design written from its code, would taint MIT code that many people depend on. The rule is stricter
than copyright on purpose: we stay away from the code, not only from copying it.

## 2. Source register

| Tier | What we may do | Sources |
|---|---|---|
| Port with notice | Read, port with attribution in `THIRD_PARTY.md` | MIT/BSD/Apache/zlib code from its own upstream: pbrt-v4, Mitsuba 3/Dr.Jit, Cycles, tinybvh, Embree, OIDN, OpenPGL, HIPRT/Orochi, Bevy Solari, kajiya, Tungsten, appleseed, LuxCore, lightmetrica, Capsaicin, three-gpu-pathtracer, three-mesh-bvh, Rayzee (minus its NRD port), strahl, Web-RTRT, ShadyTracer, ThreeRT from `5cb95f6` on (§4) |
| Cite and compare only, by choice | Read, cite, benchmark; never copy | mutracer |
| Ideas only | Papers, talks, books, blog posts; no code transcription | NVIDIA RTX SDK repos other than NRD (RTXPT, RTXDI, RTXGI, SHaRC); Ray Tracing Gems I/II prose and equations (listings are CC BY-NC-ND: never port); GPL/LGPL; rindr (PolyForm Noncommercial); repos with no licence file (relight, wtrace, webgpu-raytracer) |
| Fenced | Nobody opens the listed paths; the rest of the repo keeps its own tier | see §4 |
| Off limits | Never clone, open, summarise or paraphrase | NRD and NRD-Sample code, README code blocks and implementation prose; any fork, mirror or port of NRD; `~/Developer/nrd-quarantine`; NVIDIA STBN code and textures (non-commercial) |

Allowed about NRD: public papers and talks (RTG2 ch. 49 prose, GTC 2020, Metro Exodus GTC/GDC 2019), NVIDIA's public web
pages, the interface facts in the Denoiser's `sources/nrd-provenance.md` §3, and native NRD output consumed as images
through the Denoiser's Windows harness (D10).

## 3. Procedure for a new third-party source

1. **Clone** shallow into `research/sources/<web|native>/<name>`, read only, never built or run (research/BRIEF.md). Record
   the licence in `research/sources/MANIFEST.md`.
2. **Scan before anyone reads it**:

   ```bash
   node research/tools/provenance-scan.ts research/sources/web/<name> --git
   ```

   It prints only paths, token counts, matched identifiers and the commits that introduced them, never the text around a
   match, so running it exposes nobody to flagged code. Use `git fetch --unshallow` in the clone first for the history.
3. **Classify every hit**, without opening the flagged files:

   | Evidence | Meaning | Action |
   |---|---|---|
   | NVIDIA copyright/licence header, `.hlsl` from an SDK, an SDK file vendored | copied code | whole repo, or that subtree, **off limits**; delete the files from our clone and say so in MANIFEST |
   | Source pointer: an SDK identifier (`REBLUR_*`, `RELAX_*`, ...), SDK file name or SDK repo URL in the author's notes | the author consulted SDK code for that area | **fence** the area it touches (§4); provenance unknown until the authors say otherwise |
   | Concept name only ("ReBLUR's fast history", "anti-lag", "NRD's default is 30") | public ideas from papers, talks, settings | no fence; note it in the research note |

   Say which evidence class you found, never more. "Written from NRD source" needs a pointer *and* code we have not seen;
   say "cites NRD source for X" instead (lesson of 2026-11-14, §6).
4. **Brief every auditing agent** with the fence before it starts. Template:

   > CLEAN-ROOM RULE (mandatory): `<paths>` of `<repo>` were written with NVIDIA NRD source consulted. Do not open them. If
   > a file in your scope turns out to contain temporal accumulation, history, denoise-filter or other fenced logic, stop
   > reading that part and note the path. You may report pass names and measured results (ms, error numbers); never how it
   > works: no formulas, thresholds, constants, frame counts or steps. When in doubt, leave it out.

5. **Record** the fence in four places: §4 below, the MANIFEST row, a CLAUDE.md trap, and §0 of the source's research note.

## 4. Fences in force

| Repo (clone) | Fenced paths | Evidence | Since | Lift when |
|---|---|---|---|---|
| ThreeRT (`research/sources/web/ThreeRT`; our clone `8964dc4`) | **All code before `5cb95f6`**, our clone included: nobody opens it. `5cb95f6` and later: ordinary MIT tier (row "Port with notice", §2), after a fresh clone and a provenance scan; read their `docs/PROVENANCE.md` first and skip anything it marks as kept from before. | Our scan 2026-11-14 found one NRD source pointer (ROADMAP cites `REBLUR_TemporalAccumulation`, #39). The authors' answer ([meshy-dev/ThreeRT#78](https://github.com/meshy-dev/ThreeRT/issues/78)): ReBLUR-named passes of unverified provenance and code written from Unreal Engine source, both replaced in `5cb95f6` (#268) | 2026-11-14, revised 2026-11-26 | Lifted for `5cb95f6` and later (D35); permanent for earlier history |
| Rayzee (`research/sources/web/rayzee-renderer`, `afdfd0d`) | Deleted from our clone (off limits): `rayzee/src/viewer/denoise/NRD.js`, `docs/design/NRD_DENOISER.md`, `tests/gpu/nrdDemodulation.test.js`. Fenced (scan 2026-11-14 still finds NRD identifiers: settings, presets, hit-distance parameters): all of `rayzee/src/viewer/denoise/`, the NRD settings in `app/src/Constants.js` and `app/src/store/pathTracer.js`, `tests/unit/viewer/denoise/`. Design docs mention it by name only (`CLAUDE.md`, `docs/design/*`, `docs/TODO.md`): skip their NRD sections | self-described NRD port | 2026-11-13, widened 2026-11-14 | never for the port; the rest of Rayzee stays usable |
| NRD (`native/nrd`) | whole repo, clone deleted; do not re-clone | NVIDIA RTX SDKs License | 2026-11-13 | never |

## 5. What we ship

- No NVIDIA SDK code in either package (`THIRD_PARTY.md` §1.4). Ported permissive code is listed there with its licence.
- Our own blue noise (D16), our own FLIP implementation (BSD-3 reference, `THIRD_PARTY.md` §1.2).
- Comparisons against competitors stay internal until a methodology page exists and the other authors have seen it (D8).

## 6. Log

| Date | Event |
|---|---|
| 2026-11-13 | The native research pass had cloned NRD, and Rayzee's clone held an NRD port. Both removed; MANIFEST note; BRIEF carries the rule. |
| 2026-11-14 | ThreeRT audit: three agents briefed with a fence after a manual grep found NRD/ReBLUR mentions. The first write-up said its denoiser was "written from NRD's ReBLUR source"; the maintainer questioned it, and a citation-only scan showed one source pointer (#39) plus public concept names. Wording corrected everywhere; fence kept as a precaution (D22 on its branch, D35 on main). `research/tools/provenance-scan.ts` added so the next scan comes before the audit, not after. Its first run on Rayzee found NRD identifiers in 14 files beyond the 3 deleted on 2026-11-13; Rayzee fence widened (nobody opened them). |
| 2026-11-26 | ThreeRT provenance resolved: the authors answered on [meshy-dev/ThreeRT#78](https://github.com/meshy-dev/ThreeRT/issues/78) (closed 2026-11-24): their session records show no NRD read, but the three ReBLUR-named changes (#33, #39, #55) were written where no records were kept, so they treated them as unverified and replaced them; the same review found code written from Unreal Engine source (the post chain, part of the specular temporal pass, the specular energy terms, part of the atmosphere, a fog density), also replaced. The replacement (#268, `5cb95f6`) was written by fresh sessions in a history-free tree from a behaviour spec, black-box measurements and cited public sources (their `docs/PROVENANCE.md`). They ask clean rooms to exclude everything before `5cb95f6`. Fence moved from "the denoiser files" to "all history before `5cb95f6`" (D35, was D22 on its branch). research/notes/16-threert.md describes our pre-`5cb95f6` clone: measured results and public concepts only, no code; take any design from the public sources, not from it. |
