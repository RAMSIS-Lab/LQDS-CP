#!/usr/bin/env python3
"""Download and verify the public datasets used in the paper."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets"
MANIFEST = json.loads((ROOT / "data_manifest.json").read_text(encoding="utf-8"))

CQR_RAW = "https://raw.githubusercontent.com/yromano/cqr/master/datasets"
DIRECT = {
    "CASP.csv": f"{CQR_RAW}/CASP.csv",
    "STAR.csv": f"{CQR_RAW}/STAR.csv",
    "bike_train.csv": f"{CQR_RAW}/bike_train.csv",
    "communities.data": f"{CQR_RAW}/communities.data",
    "communities_attributes.csv": f"{CQR_RAW}/communities_attributes.csv",
}
ARCHIVES = {
    "blog": "https://archive.ics.uci.edu/static/public/304/blogfeedback.zip",
    "facebook": (
        "https://archive.ics.uci.edu/static/public/363/"
        "facebook+comment+volume+dataset.zip"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "CQR-reproducibility"})
    with urllib.request.urlopen(request) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def install_member(archive: Path, suffix: str, destination: Path) -> None:
    with zipfile.ZipFile(archive) as zipped:
        matches = [name for name in zipped.namelist() if name.endswith(suffix)]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one archive member ending in {suffix!r}: {matches}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zipped.open(matches[0]) as source, destination.open("wb") as target:
            shutil.copyfileobj(source, target)


def verify(names: list[str] | None = None) -> bool:
    ok = True
    selected = names or list(MANIFEST)
    for name in selected:
        path = DATA / name
        if not path.exists():
            print(f"MISSING  {name}")
            ok = False
            continue
        actual = sha256(path)
        if actual == MANIFEST[name]:
            print(f"OK       {name}")
        else:
            print(f"MISMATCH {name}\n  expected {MANIFEST[name]}\n  found    {actual}")
            ok = False
    return ok


def download(force: bool) -> None:
    DATA.mkdir(exist_ok=True)
    for name, url in DIRECT.items():
        destination = DATA / name
        if force or not destination.exists():
            print(f"Downloading {name}")
            fetch(url, destination)

    with tempfile.TemporaryDirectory(prefix="cqr-data-") as temporary:
        temporary = Path(temporary)
        blog_archive = temporary / "blog.zip"
        facebook_archive = temporary / "facebook.zip"
        if force or not (DATA / "blogData_train.csv").exists():
            print("Downloading BlogFeedback from UCI")
            fetch(ARCHIVES["blog"], blog_archive)
            install_member(blog_archive, "blogData_train.csv", DATA / "blogData_train.csv")
        facebook_targets = [
            DATA / "facebook" / "Features_Variant_1.csv",
            DATA / "facebook" / "Features_Variant_2.csv",
        ]
        if force or any(not path.exists() for path in facebook_targets):
            print("Downloading Facebook Comment Volume from UCI")
            fetch(ARCHIVES["facebook"], facebook_archive)
            for variant, destination in enumerate(facebook_targets, start=1):
                install_member(
                    facebook_archive,
                    f"Training/Features_Variant_{variant}.csv",
                    destination,
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.verify_only:
        download(args.force)
    names = None if args.verify_only else [name for name in MANIFEST if not name.startswith("meps_")]
    if not verify(names):
        raise SystemExit(1)
    if not args.verify_only:
        print("\nNon-MEPS downloads are complete. Follow README.md to prepare MEPS 19--21.")
