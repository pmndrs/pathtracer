# pathtracer

A WebGPU ray and path tracing library: a fast, GPU-driven trace core shared by our rendering tools (reflections,
GI, radiance cascades, baking) and a full PBR path tracer built on it.

Packages: [`@pmndrs/raytracer`](packages/raytracer) and [`@pmndrs/pathtracer`](packages/pathtracer) (0.1, on npm).
Docs: <https://pmndrs.github.io/pathtracer/> ([source](docs/)) · Examples: <https://pmndrs.github.io/pathtracer/examples/>
([source](examples/), `pnpm examples` runs them locally). Where the project stands: [STATUS.md](STATUS.md).

Research and design records:

- [research/BRIEF.md](research/BRIEF.md) — goals, rules and the brief every research pass works from
- [research/notes/](research/notes/) — one notes file per research area
- [research/INDEX.md](research/INDEX.md) — cross-area synthesis (written once the passes land)
- [spec/SPEC-draft.md](spec/SPEC-draft.md) — loose spec drafted from the research
- [THIRD_PARTY.md](THIRD_PARTY.md) — ported or adapted code, third-party assets, and the packages we only run
- `research/sources/` — shallow clones of third-party code, read-only and git-ignored; see `research/sources/MANIFEST.md`

Built alongside our sibling packages: `@pmndrs/bvh` (geometry and ray walk), `downsampler` (SPD), `Denoiser`,
`@pmndrs/upscaler`, `@pmndrs/reflections`, `@pmndrs/gi`, and gpuTex.
