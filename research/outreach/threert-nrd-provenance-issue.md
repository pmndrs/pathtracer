# Draft: provenance question to meshy-dev/ThreeRT

Status: posted by the maintainer as [meshy-dev/ThreeRT#78](https://github.com/meshy-dev/ThreeRT/issues/78) and resolved 2026-11-24: the authors answered on [meshy-dev/ThreeRT#78](https://github.com/meshy-dev/ThreeRT/issues/78) (closed 2026-11-24): their session records show no NRD read, but the three ReBLUR-named changes (#33, #39, #55) were written where no records were kept, so they treated them as unverified and replaced them; the same review found code written from Unreal Engine source (the post chain, part of the specular temporal pass, the specular energy terms, part of the atmosphere, a fog density), also replaced. The replacement (#268, `5cb95f6`) was written by fresh sessions in a history-free tree from a behaviour spec, black-box measurements and cited public sources (their `docs/PROVENANCE.md`). They ask clean rooms to exclude everything before `5cb95f6`. Recorded in docs/internal/CLEAN-ROOM.md §4 / §6 and DECISIONS D35.

The text as drafted 2026-11-14:

---

**Title:** Provenance question: NRD source cited for the specular history pass (#39)

**Body:**

Hi, and congratulations on the release. The browser build and the write-ups in docs/ are great reading.

I maintain [pmndrs/pathtracer](https://github.com/pmndrs/pathtracer), an MIT WebGPU path tracer, and we have been reading
ThreeRT as a reference. We keep a strict clean room around NVIDIA's NRD. Its licence (the NVIDIA RTX SDKs License) reserves
all rights in the source files, allows distribution only in object-code form inside an application, and §4(e) prohibits
anything that would make the SDK subject to an open-source licence. So before we read more of ThreeRT, I wanted to ask
about one passage.

`docs/ROADMAP.md` §1a, added in #39 ("The specular history per pixel: bounded by each reflection's own parallax"), describes
how ReBLUR bounds the specular history and points to "its source, `REBLUR_TemporalAccumulation`". The same PR changed
`renderer/shaders/live.wgsl`.

Because of that reference we have not opened `live.wgsl` or the denoiser code, so we are **not** assuming anything was
copied. The other ReBLUR mentions (fast history, anti-lag) match what is public in *Ray Tracing Gems II* ch. 49 and the GTC
2020 ReBLUR talk.

Could you clarify:

1. Was any NRD shader code translated or closely followed in `live.wgsl` (in #39 or in other denoiser passes), or was the
   NRD source only consulted for the idea?
2. If it was only consulted, would you consider a short provenance note in the docs and citing the public sources instead?
   That would let downstream users of the MIT code, us included, rely on it with confidence.
3. If code was followed, it may be worth checking that part against NVIDIA's licence terms before it ships under MIT.
   Rewriting it from the public descriptions would settle it.

Not legal advice, just the line we hold for our own MIT packages. Happy to share our notes on the public ReBLUR material if
that helps. Thanks for open-sourcing all of this.
