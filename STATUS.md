# Status

_Last updated 2026-11-13._

**Phase:** P0 (foundations), about two thirds done.

**Working now**
- **Bench runner.** Any bench page runs headless in Chrome on the Mac's GPU. Results go to CSV files, images to PFM.
  A/B runs alternate A B B A automatically.
- **A/B viewer.** Opens any two float images with swipe, flip, difference, error heat map and FLIP modes, a pixel
  inspector and an exposure control. There is also a grid view of a whole run.
- **Reference renders.** Six small test scenes: three "furnace" energy tests, a Cornell box, a single area light and
  a sky-lit sphere. Each is rendered by pbrt-v4 and by Mitsuba 3 on the Mac.
  - The furnace tests hit their exact expected values in both renderers.
  - The two renderers agree closely: relMSE 2e-8 to 6.4e-4. That gap is our noise floor.
- **Capability detection.** This Mac is "tier 2": subgroups, f16 and timestamps, but no bindless.
- **Literature check.**
  - OIDN 3 on Nov 30 is an alpha; temporal denoising for all GPUs comes in March 2027.
  - NVIDIA's blue-noise textures can't be used (non-commercial licence), so we generate our own.
  - Native wide-BVH traversal is 6.3× ahead of our current BVH, not 3.5×.

**Next**
- Run mutracer and three-gpu-pathtracer inside the bench (in progress).
- Add a Sponza scene with plain grey materials, then produce the first "time to equal error" table.
- Push the `windows-testing` branch with a handoff doc.

**Waiting on the maintainer:** whether to start the wide-BVH experiment in the bvh repo now or at P1.
