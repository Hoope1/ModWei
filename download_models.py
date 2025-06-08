#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robuster Mehrfach-Fetcher für PiDiNet, DiffusionEdge & EDTER.
Probiert nacheinander mehrere Mirror-URLs, prüft optional SHA-256
und quittiert erst mit Fehler, wenn *alle* Spiegel scheitern.
"""
from __future__ import annotations
import hashlib, sys, time, shutil
from pathlib import Path
from typing import List
import requests, subprocess, os

# --------------------------------------------------------------------------- #
#  Konfigurations-Tabelle  –  pro Modell eine PRIORISIERTE URL-Liste          #
# --------------------------------------------------------------------------- #
MODEL_DIR = Path("models/weights")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODELS: dict[str, dict] = {
    "pidinet": {
        "candidates": [
            # ① offizieller HF-Link  –  2.87 MB
            "https://huggingface.co/lllyasviel/Annotators/resolve/main/table5_pidinet.pth",  # 0
            # ② direkt aus dem PiDiNet-Repo (Git LFS, public)
            "https://github.com/hellozhuo/pidinet/raw/master/trained_models/table5_pidinet.pth",  # 1
            # ③ ControlNet-Nightly Release (GitHub Assets)
            "https://github.com/Sxela/ControlNet-v1-1-nightly/releases/download/v0.1.0/table5_pidinet.pth",  # 2
            # ④ Mirror im TencentARC/T2I-Adapter Repo
            "https://huggingface.co/TencentARC/T2I-Adapter/resolve/main/third-party-models/table5_pidinet.pth",  # 3
        ],
        "file": MODEL_DIR / "table5_pidinet.pth",
        "sha256": "80860ac267258b5f27486e0ef152a211d0b08120f62aeb185a050acc30da486c",
    },
    "diffedge": {
        "candidates": [
            # ① Release-Checkpoint (Stage 1) – 43 MB
            "https://github.com/GuHuangAI/DiffusionEdge/releases/download/v1.1/first_stage_total_320.pt",  # 4
            # ② Hugging-Face-Mirror mit drei Varianten (1.2 GB LFS, aber public)
            "https://huggingface.co/hr16/Diffusion-Edge/resolve/main/diffusion_edge_natrual.pt",           # 5
            "https://huggingface.co/hr16/Diffusion-Edge/resolve/main/diffusion_edge_urban.pt",            # 6
            "https://huggingface.co/hr16/Diffusion-Edge/resolve/main/diffusion_edge_indoor.pt",           # 7
        ],
        "file": MODEL_DIR / "diffedge_weight.pt",     # Endung .pt, aber egal
        # SHA-Summe vom ersten (43 MB) Asset; große .pt-Varianten werden *nicht* gehasht
        "sha256": "3dbd7b9c77e3c86f87195f21c342dc45ebadd90ffae6f6d5375bc2c665c9fd2d",
    },
    "edter": {
        "candidates": [
            # ① Google-Drive-Mirror (≈ 91 MB) – public, direkt
            "https://drive.google.com/uc?export=download&id=1m2GyuAHbvN1VtNj79qQId1EjhL_NhR4o",  # 8
            # ② BaiDu-Netdisk URL (benötigt ggf. aria2 + cookie, aber wird versucht)
            "https://paddle-fl.cn:443/f/11423357/edter_bsds.pth",                                     # (Quelle GitHub README) 9
        ],
        "file": MODEL_DIR / "edter_bsds.pth",
        "sha256": "c2b84f0c80f15d6d0198d4c477726db44472b4d0282dadba5baba2b04b92851e",
    },
}

# --------------------------------------------------------------------------- #
#  Download-Utilities                                                         #
# --------------------------------------------------------------------------- #
CHUNK = 1 << 16  # 64 KiB


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for blk in iter(lambda: f.read(CHUNK), b""):
            h.update(blk)
    return h.hexdigest()


def gdown(url: str, dst: Path) -> bool:
    """Versucht gdown-CLI → True bei Erfolg."""
    if shutil.which("gdown") is None:
        return False
    cmd = ["gdown", "--no-cookies", "--id", url.split("id=")[-1], "-O", str(dst)]
    return subprocess.call(cmd) == 0 and dst.exists()


def fetch(url: str, dst: Path) -> bool:
    """HTTP-Download mit Stream; Rückgabe True bei Erfolg."""
    print(f"   ↳ {url}")
    # Google-Drive-Links werden, falls möglich, via gdown behandelt
    if "drive.google.com" in url:
        return gdown(url, dst)
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0))
            done = 0
            with dst.open("wb") as f:
                for chunk in r.iter_content(CHUNK):
                    if chunk:
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            p = done * 100 // total
                            sys.stdout.write(f"\r      {done/1e6:5.1f} MB ({p:3d} %)")
                            sys.stdout.flush()
            print()
        return True
    except Exception as e:
        print("      ❌ ", e)
        return False


def try_download(name: str, cfg: dict) -> bool:
    dst: Path = cfg["file"]
    if dst.exists() and sha256(dst) == cfg.get("sha256", ""):
        print(f"[SKIP] {name}: bereits vorhanden ✔")
        return True

    for url in cfg["candidates"]:
        dst.unlink(missing_ok=True)
        if fetch(url, dst):
            if "sha256" in cfg and sha256(dst) != cfg["sha256"]:
                print("      ⚠️ SHA-256 falsch – versuche nächsten Mirror")
                dst.unlink(missing_ok=True)
                continue
            print(f"[OK] {name}: gespeichert unter {dst}")
            return True
        time.sleep(2)  # kurze Abkühlung zwischen Mirrors
    print(f"[FAIL] {name}: alle Mirrors fehlgeschlagen")
    return False


def main() -> None:
    failures: List[str] = []
    for name, cfg in MODELS.items():
        if not try_download(name, cfg):
            failures.append(name)

    if failures:
        print("\n❌ Abbruch – folgende Modelle nicht geladen:", ", ".join(failures))
        sys.exit(1)
    print("\n✅ Alle Modelle verfügbar und verifiziert.")


if __name__ == "__main__":
    main()
