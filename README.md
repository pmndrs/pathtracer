# pathtracer

A WebGPU ray and path tracing library: a fast, GPU-driven trace core shared by our rendering tools (reflections,
GI, radiance cascades, baking) and a full PBR path tracer built on it.

Status: **research phase.** No code yet.

- [research/BRIEF.md](research/BRIEF.md) — goals, rules and the brief every research pass works from
- [research/notes/](research/notes/) — one notes file per research area
- [research/INDEX.md](research/INDEX.md) — cross-area synthesis (written once the passes land)
- [spec/SPEC-draft.md](spec/SPEC-draft.md) — loose spec drafted from the research
- `research/sources/` — shallow clones of third-party code, read-only and git-ignored; see `research/sources/MANIFEST.md`

Built alongside our sibling packages: `@pmndrs/bvh` (geometry and ray walk), `downsampler` (SPD), `Denoiser`,
`@pmndrs/upscaler`, `@pmndrs/reflections`, `@pmndrs/gi`, and gpuTex.
