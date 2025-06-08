#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_models.py

Lädt die benötigten Gewichtungsdateien für die Edge-Detection-Modelle
herunter und speichert sie unter models/weights.
"""
import sys
import requests
from pathlib import Path

# Konfiguration der Modelle und ihrer Gewichtungs-URLs
MODELS = {
    "pidinet": {
        "url": "https://huggingface.co/lllyasviel/Annotators/resolve/main/table5_pidinet.pth",
        "file": Path("models/weights/table5_pidinet.pth"),
    },
    "diffedge": {
        "url": "https://huggingface.co/GuHuangAI/DiffusionEdge/resolve/main/diffedge_swin.pth",
        "file": Path("models/weights/diffedge_swin.pth"),
    },
    "edter": {
        "url": "https://download.openmmlab.com/mmsegmentation/v0.5/edter/edter_bsds.pth",
        "file": Path("models/weights/edter_bsds.pth"),
    },
}


def download_model(name: str, info: dict) -> None:
    """
    Lädt eine einzelne Gewichtungsdatei herunter, falls sie noch nicht existiert.
    """
    url: str = info["url"]
    dst: Path = info["file"]
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        print(f"[SKIP] {name}: {dst} existiert bereits, überspringe Download.")
        return

    print(f"[DOWNLOAD] {name}: {url} → {dst}")
    resp = requests.get(url, stream=True)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        print(f"[ERROR] {name}: Download fehlgeschlagen ({e})")
        sys.exit(1)

    with open(dst, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print(f"[OK] {name}: erfolgreich gespeichert unter {dst}")


def main() -> None:
    """
    Hauptprogramm: lädt alle definierten Modelle nacheinander herunter.
    """
    for name, info in MODELS.items():
        download_model(name, info)
    print("\nAlle Modelle wurden überprüft und ggf. heruntergeladen.")


if __name__ == "__main__":
    main()
