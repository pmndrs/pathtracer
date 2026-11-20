"""Prepare the NRC showcase assets on a development machine with Blender and Node."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import urllib.request
import zipfile

here = Path(__file__).resolve().parent
root = here.parents[1]
assets = root / "bench" / "assets"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("scene", choices=["all", "classroom", "sponza"], default="all", nargs="?")
parser.add_argument("--blender", default=os.environ.get("BLENDER_PATH") or shutil.which("blender") or
                    "/Applications/Blender.app/Contents/MacOS/Blender")
args = parser.parse_args()
if not Path(args.blender).is_file() and not shutil.which(args.blender):
    parser.error("Pass --blender with the path to your Blender executable.")
manifest = json.loads((here / "color-assets.json").read_text())


def verify(file, entry):
    return (file.is_file() and file.stat().st_size == entry["bytes"]
            and hashlib.sha256(file.read_bytes()).hexdigest() == entry["sha256"])


def fetch(entry):
    dest = assets / entry["path"]
    if verify(dest, entry):
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading", entry["url"], flush=True)
    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as output:
        tmp = Path(output.name)
    try:
        # A browser user agent and a few retries: download.blender.org has answered CI runners with 403 (site.yml caches the result).
        request = urllib.request.Request(entry["url"], headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as output:
                    shutil.copyfileobj(response, output)
                break
            except OSError as error:
                if attempt == 3:
                    raise
                print(f"  retry {attempt + 1} after {error}", flush=True)
                time.sleep(5 * (attempt + 1))
        if not verify(tmp, entry):
            raise ValueError(f"Size or SHA-256 mismatch for {dest}")
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)


for kind in ["classroom", "sponza"] if args.scene == "all" else [args.scene]:
    extra = []
    if kind == "sponza":
        subprocess.run(["node", "bench/assets/fetch.ts", "--only", "sponza"], cwd=root, check=True)
    for entry in manifest[kind]:
        fetch(entry)
    if kind == "classroom":
        with zipfile.ZipFile(assets / manifest[kind][0]["path"]) as archive:
            archive.extractall(assets / "downloads")
        extra.append(str(assets / "downloads/classroom/classroom.blend"))
    subprocess.run([args.blender, "-b", "--disable-autoexec", *extra, "--python-exit-code", "1",
                    "--python", str(here / "prepare-blender.py"), "--", kind], cwd=root, check=True)
    subprocess.run(["node", "bench/hero/prepare.ts", kind], cwd=root, check=True)
