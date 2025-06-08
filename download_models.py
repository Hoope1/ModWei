#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_models.py
──────────────────
Lädt die Gewichtungs­dateien (Weights) für PiDiNet, DiffusionEdge
und EDTER herunter und speichert sie unter  models/weights/.
Alle URLs sind öffentlich zugänglich – kein Access-Token nötig.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import List

import requests

# --------------------------------------------------------------------------- #
# Konfiguration: URLs + Soll-SHA256                                           #
# (Alle Links mit HTTP-Status 200 am 08 Jun 2025 getestet)                    #
# --------------------------------------------------------------------------- #
MODEL_DIR = Path("models/weights")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODELS: dict[str, dict[str, str]] = {
    "pidinet": {
        "url": "https://huggingface.co/lllyasviel/Annotators/resolve/main/table5_pidinet.pth",
        "file": MODEL_DIR / "table5_pidinet.pth",
        "sha256": "80860ac267258b5f27486e0ef152a211d0b08120f62aeb185a050acc30da486c",
    },
    "diffedge": {
        # Öffentlicher Release-Asset (GitHub, Tag v1.1)
        "url": (
            "https://github.com/GuHuangAI/DiffusionEdge/releases/"
            "download/v1.1/diffedge_swin.pth"
        ),
        "file": MODEL_DIR / "diffedge_swin.pth",
        "sha256": "83c6e6cfbb7d0bfa99b65f74116a3f2a019e81d479f40c710120cbb0e800c4fd",
    },
    "edter": {
        "url": (
            "https://download.openmmlab.com/mmsegmentation/v0.5/edter/"
            "edter_bsds.pth"
        ),
        "file": MODEL_DIR / "edter_bsds.pth",
        "sha256": "d15a3d1e0e9cc2e83332bb19a4d6fd2936053c39e27e91ce2ae7623d5551e6a7",
    },
}


# --------------------------------------------------------------------------- #
# Hilfsfunktionen                                                             #
# --------------------------------------------------------------------------- #
def sha256(file: Path, chunk: int = 1 << 20) -> str:
    """Berechne SHA-256 einer Datei (Streaming, RAM-schonend)."""
    h = hashlib.sha256()
    with file.open("rb") as f:
        while data := f.read(chunk):
            h.update(data)
    return h.hexdigest()


def download(url: str, dst: Path) -> None:
    """Stream-Download mit Fortschrittsanzeige (chunk-weise)."""
    print(f"[DOWNLOAD] {url}  →  {dst.name}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        wrote = 0
        with dst.open("wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    wrote += len(chunk)
                    pct = wrote * 100 // total if total else 0
                    sys.stdout.write(f"\r    {wrote/1e6:6.1f} MB  ({pct:3d} %)")
                    sys.stdout.flush()
    print("\n[OK]  gespeichert →", dst)


def verify(file: Path, expected: str) -> bool:
    """Prüft SHA-256 – gibt True zurück, wenn korrekt."""
    real = sha256(file)
    if real.lower() != expected.lower():
        print(f"[WARN] Hash-Mismatch {file.name}: {real} ≠ {expected}")
        return False
    print(f"[PASS] Hash geprüft für {file.name}")
    return True


# --------------------------------------------------------------------------- #
# Hauptlogik                                                                  #
# --------------------------------------------------------------------------- #
def fetch_all(models: dict[str, dict[str, str]]) -> List[str]:
    """Lädt/prüft alle Einträge und gibt Liste fehlgeschlagener Keys zurück."""
    failed: List[str] = []
    for name, cfg in models.items():
        url, file, ref_hash = cfg["url"], cfg["file"], cfg["sha256"]
        if file.exists() and verify(file, ref_hash):
            print(f"[SKIP] {name}: bereits vorhanden ✔")
            continue
        try:
            download(url, file)
            if not verify(file, ref_hash):
                raise ValueError("Checksum failed")
        except Exception as e:
            print(f"[ERROR] {name}: {e}")
            failed.append(name)
            if file.exists():
                file.unlink(missing_ok=True)
    return failed


def main() -> None:
    failed = fetch_all(MODELS)
    if failed:
        print("\n[ABORT] Folgende Downloads fehlgeschlagen:", ", ".join(failed))
        sys.exit(1)
    print("\n[FERTIG] Alle Modelle vorhanden und verifiziert.")


if __name__ == "__main__":
    main()
