# Status

_Last updated 2026-11-13._

**Phase:** P0 (foundations) has started.

**Done so far**
- Research pass across 10 areas, synthesized in research/INDEX.md, with a draft spec in spec/SPEC-draft.md.
- The plan is agreed: phases P0–P5 (PHASES.md) and the first 13 decisions (DECISIONS.md).
- Repo created at github.com/pmndrs/pathtracer.

**Happening now**
- Setting up the monorepo, the shared conventions, and the first bench pieces: GPU runner, metrics and A/B viewer,
  reference scenes with pbrt-v4/Mitsuba exporters, and device capability detection.
- A short literature check (NVIDIA licence scope, Chrome timer flags, unverified numbers in the notes).

**Next**
- Run mutracer and three-gpu-pathtracer inside the bench and get the first "time to equal error" table.
- Push the `windows-testing` branch with a handoff doc so the Windows machine can run the same bench.

**Waiting on the maintainer:** nothing right now.
