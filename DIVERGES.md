# Divergences

Where we deliberately differ from a source we built on (paper, sibling package, competitor, native renderer), and
why, so each one can be revisited later. One row per divergence. Kinds: **platform** (browser/driver/WGSL limit),
**perf** (measured speed choice), **licence**, **simplification** (do it properly later), **bug-workaround**.

| ID | Date | Source | What we do differently | Kind | Revisit when |
|---|---|---|---|---|---|
| V1 | 2026-11-13 | Spec / research note 06 (bindless) | Texture arrays (gpuTex) are the default texture path; the bindless resource table is opt-in | platform | Chrome ships the resource table on macOS and drops the `chromium-experimental-` prefix |
| V2 | 2026-11-13 | NRD (real-time denoiser reference) | No NRD code or ports anywhere in this repo; native NRD output is used only as images | licence | never (clean room) |
