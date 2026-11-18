"""Render a Mitsuba 3 scene to a PFM (little-endian float32, bottom-to-top scanlines per the PFM spec).

usage: mitsuba_render.py <scene.xml> <out.pfm> <variant> <spp>
Prints one JSON line: {"mitsuba": version, "variant": ..., "seconds": ..., "spp": ...}
"""
import json
import sys
import time

import numpy as np
import mitsuba as mi


def write_pfm(path, img):
    img = np.asarray(img, dtype="<f4")
    h, w, c = img.shape
    assert c == 3
    with open(path, "wb") as f:
        f.write(f"PF\n{w} {h}\n-1.0\n".encode("latin1"))
        f.write(np.ascontiguousarray(img[::-1]).tobytes())


def main():
    scene_path, out, variant, spp = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    mi.set_variant(variant)
    scene = mi.load_file(scene_path)
    t0 = time.time()
    sensor = scene.sensors()[0]
    w, h = sensor.film().size()
    # JIT (llvm_ad) variants allocate one wavefront entry per sample: render in chunks of <= 2^25 samples and average
    # (distinct seeds), instead of one 1e9-sample wavefront. Scalar variants stream blocks and need no chunking.
    chunk = spp if variant.startswith("scalar") else max(1, min(spp, (1 << 25) // (w * h)))
    acc = None
    wsum = None
    nan_hits = 0
    done = 0
    seed = 0
    while done < spp:
        n = min(chunk, spp - done)
        part = np.array(mi.render(scene, spp=n, seed=seed)).astype(np.float64)  # forces evaluation
        # P5: Mitsuba's heterogeneous medium returns a NaN sample in about 1 pixel per 3000 per 512 spp. A pixel that is non-finite in a chunk drops THAT chunk
        # (its other chunks keep their samples; `nan_pixel_chunks` is reported). Without a NaN this is exactly the plain average.
        bad = ~np.isfinite(part).all(axis=-1)
        nan_hits += int(bad.sum())
        part = np.where(bad[..., None], 0.0, part)
        wt = np.where(bad, 0.0, float(n))
        acc = part * n if acc is None else acc + part * n
        wsum = wt if wsum is None else wsum + wt
        done += n
        seed += 1
    arr = np.where(wsum[..., None] > 0, acc / np.maximum(wsum[..., None], 1.0), np.nan).astype(np.float32)
    seconds = time.time() - t0
    write_pfm(out, arr[..., :3])
    print(json.dumps({"mitsuba": mi.__version__, "variant": variant, "seconds": seconds, "spp": spp, "nan_pixel_chunks": nan_hits,
                      "mean": [float(x) for x in np.nanmean(arr[..., :3].reshape(-1, 3), axis=0)]}))

main()
