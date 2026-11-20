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


BLENDER_ORG = "https://download.blender.org/"
BLENDER_MIRRORS = [
    "https://mirrors.ocf.berkeley.edu/blender/",
    "https://ftp.nluug.nl/pub/graphics/blender/",
    "https://ftp.halifax.rwth-aachen.de/blender/",
    "https://mirror.clarkson.edu/blender/",
]


def fetch(entry):
    dest = assets / entry["path"]
    if verify(dest, entry):
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading", entry["url"], flush=True)
    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as output:
        tmp = Path(output.name)
    try:
        # download.blender.org has answered CI runners with 403: try it, then Blender's official mirrors (same tree), each with a browser
        # user agent and a retry. The SHA-256 below decides whatever served the file (site.yml caches the result).
        urls = [entry["url"]]
        if entry["url"].startswith(BLENDER_ORG):
            urls += [mirror + entry["url"][len(BLENDER_ORG):] for mirror in BLENDER_MIRRORS]
        for i, url in enumerate(urls):
            try:
                if i:
                    print("  trying", url, flush=True)
                request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"})
                for attempt in range(2):
                    try:
                        with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as output:
                            shutil.copyfileobj(response, output)
                        break
                    except OSError as error:
                        if attempt == 1:
                            raise
                        print(f"  retry after {error}", flush=True)
                        time.sleep(5)
                if verify(tmp, entry):
                    break
                print(f"  {url}: size or SHA-256 mismatch", flush=True)
            except OSError as error:
                print(f"  {url}: {error}", flush=True)
        if not verify(tmp, entry):
            raise ValueError(f"No source served {dest.name} with the expected size and SHA-256 (tried {len(urls)})")
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
