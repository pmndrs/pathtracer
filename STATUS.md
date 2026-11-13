# Status

_Last updated 2026-11-13._

**Phase:** P0 (foundations), most of the way there. What's left is the first baseline table and the Windows branch.

**Working now**
- **Bench runner.** Any bench page runs headless in Chrome on the Mac's GPU. Results go to CSV files, images to PFM.
  A/B runs alternate A B B A automatically.
- **A/B viewer.** Opens any two float images with swipe, flip, difference, error heat map and FLIP modes, a pixel
  inspector and an exposure control. There is also a grid view of a whole run.
- **Reference renders.**
  - Six small Tier 0 scenes plus "sponza-white": Sponza with plain grey walls, a sun and a sky.
  - Each is rendered by pbrt-v4 and by Mitsuba 3 on the Mac at 4096 samples per pixel. Sponza takes 24 min in
    Mitsuba and 57 min in pbrt.
  - The furnace tests hit their exact expected values in both renderers.
  - The two renderers agree to within their noise: relMSE 2e-8 to 6.4e-4 on Tier 0, 4.1e-3 on Sponza.
- **Competitors in the bench.** three-gpu-pathtracer (both its schedulers) and mutracer run with their denoisers off.
  - All three pass the diffuse, glass and single-light checks.
  - mutracer is the fastest: 6.3 ms per sample on the Cornell box, vs 13 ms for three-gpu-pathtracer.
  - Both libraries add an energy-compensation trick to rough metals that can't be switched off, so that one test is
    reported separately.
- **Capability detection.** This Mac is "tier 2": subgroups, f16 and timestamps, but no bindless.
- **Literature check.**
  - OIDN 3 on Nov 30 is an alpha; temporal denoising for all GPUs comes in March 2027.
  - NVIDIA's blue-noise textures can't be used (non-commercial licence), so we generate our own.
  - Native wide-BVH traversal is 6.3× ahead of our current BVH, not 3.5×.

**Next**
- Get the web competitors loading Sponza, then produce the first "time to equal error" table (the P0 exit gate).
- Push the `windows-testing` branch with a handoff doc.
- **Wide-BVH experiment** in the bvh repo: approved to start now. The session is ready and waits for the bvh folder
  to be trusted in the app.

**Waiting on the maintainer:** Sponza reference samples (keep 4096, or 2048 for pbrt). See TASKS.md.
