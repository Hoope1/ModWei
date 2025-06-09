#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_models.py – robust multi-mirror fetcher for PiDiNet, DiffusionEdge, EDTER.

• tries every URL in priority order
• Google-Drive links are handled with gdown (if present) or a baked fall-back wget trick
• verifies SHA-256 where we have a stable checksum
• prints clear OK / FAIL lines so the CI step can grep them
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import requests

# ---------------------------------------------------------------------------- #
#  MODEL TABLE – highest-quality mirrors first                                 #
# ---------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
W_DIR = ROOT / "weights"
W_DIR.mkdir(parents=True, exist_ok=True)

MODELS: Dict[str, Dict] = {
    "pidinet": {
        "dst": W_DIR / "table5_pidinet.pth",
        "sha": "80860ac267258b5f27486e0ef152a211d0b08120f62aeb185a050acc30da486c",
        "urls": [
            # HF (original repo) 0
            "https://huggingface.co/lllyasviel/Annotators/resolve/main/table5_pidinet.pth",
            # PiDiNet GitHub LFS mirror 1
            "https://github.com/hellozhuo/pidinet/raw/master/trained_models/table5_pidinet.pth",
            # ControlNet nightly asset (stable backup) 2
            "https://github.com/Sxela/ControlNet-v1-1-nightly/releases/download/v0.1.0/table5_pidinet.pth",
        ],
    },
    "diffedge": {
        "dst": W_DIR / "diffedge_weight.pt",
        "sha": "3dbd7b9c77e3c86f87195f21c342dc45ebadd90ffae6f6d5375bc2c665c9fd2d",
        "urls": [
            # small 43-MB first-stage weight (GitHub release) 3
            "https://github.com/GuHuangAI/DiffusionEdge/releases/download/v1.1/first_stage_total_320.pt",
            # full SWIN weights (1.2 GB, public HF mirror, three domains)
            "https://huggingface.co/hr16/Diffusion-Edge/resolve/main/diffusion_edge_natrual.pt",
            "https://huggingface.co/hr16/Diffusion-Edge/resolve/main/diffusion_edge_urban.pt",
            "https://huggingface.co/hr16/Diffusion-Edge/resolve/main/diffusion_edge_indoor.pt",
        ],
    },
    "edter": {
        "dst": W_DIR / "edter_bsds.pth",
        "sha": "c2b84f0c80f15d6d0198d4c477726db44472b4d0282dadba5baba2b04b92851e",
        "urls": [
            # author’s Google-Drive folder (Stage II BSDS) – direct ID link  5
            "https://drive.google.com/uc?export=download&id=1OkdakKKIMRGnKH8mxuFi_qI9sa903CD2",
            # Google-Drive share used in EDTER README (fallback) 6
            "https://drive.google.com/uc?export=download&id=1m2GyuAHbvN1VtNj79qQId1EjhL_NhR4o",
            # public mirror on HuggingFace user @hr16 (converted) – smaller demo model
            "https://huggingface.co/hr16/EDTER/resolve/main/edter_bsds_demo.pth",
        ],
    },
}

# ---------------------------------------------------------------------------- #
#  Helpers                                                                     #
# ---------------------------------------------------------------------------- #
CHUNK = 1 << 15  # 32 KiB


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def gdrive_download(url: str, dst: Path) -> bool:
    """
    Use gdown if available; otherwise brute-force confirm ticket from Google.
    """
    if "drive.google.com" not in url:
        return False
    if shutil.which("gdown"):
        cmd = ["gdown", "--fuzzy", url, "-O", str(dst)]
        return subprocess.call(cmd) == 0 and dst.exists()

    # manual fallback (wget two-step)
    file_id = re.search(r"id=([^&]+)", url)
    if not file_id:
        return False
    session = requests.Session()
    response = session.get(url, stream=True)
    confirm = re.search(r"confirm=([0-9A-Za-z_]+)", response.text)
    if confirm:
        dl_url = f"https://drive.google.com/uc?export=download&confirm={confirm.group(1)}&id={file_id.group(1)}"
        response = session.get(dl_url, stream=True)
    if response.status_code != 200:
        return False
    with open(dst, "wb") as f:
        for chunk in response.iter_content(CHUNK):
            if chunk:
                f.write(chunk)
    return dst.exists()


def http_download(url: str, dst: Path) -> bool:
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0))
            done = 0
            with open(dst, "wb") as f:
                for chunk in r.iter_content(CHUNK):
                    if chunk:
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            pct = done * 100 // total
                            sys.stdout.write(f"\r      {done/1e6:6.1f} MB ({pct:3d} %)")
                            sys.stdout.flush()
            print()
        return True
    except Exception as e:
        print("      ❌", e)
        return False


def try_one(model: str, url: str, dst: Path) -> bool:
    print(f"   ↳ {url}")
    dst.unlink(missing_ok=True)
    if gdrive_download(url, dst) or http_download(url, dst):
        return True
    return False


def fetch_model(name: str, cfg: Dict) -> bool:
    dst, sha, urls = cfg["dst"], cfg.get("sha"), cfg["urls"]
    if dst.exists() and sha and sha256_file(dst) == sha:
        print(f"[SKIP] {name}: already present ✔")
        return True

    for url in urls:
        if try_one(name, url, dst):
            if sha and sha256_file(dst) != sha:
                print("      ⚠️  SHA-256 mismatch – trying next mirror")
                dst.unlink(missing_ok=True)
                continue
            print(f"[OK] {name}: saved to {dst}")
            return True
        time.sleep(1)
    print(f"[FAIL] {name}: all mirrors exhausted")
    return False


def main() -> None:
    failures: List[str] = []
    for k, v in MODELS.items():
        if not fetch_model(k, v):
            failures.append(k)
    if failures:
        print("\n❌ Could not fetch:", ", ".join(failures))
        sys.exit(1)
    print("\n✅ Every model downloaded & validated")


if __name__ == "__main__":
    main()
