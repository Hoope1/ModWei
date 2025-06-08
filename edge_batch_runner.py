#!/usr/bin/env python3.10
from __future__ import annotations

"""Run multiple edge detectors on a folder of images."""

import argparse
import sys

__version__ = "0.1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", metavar="PATH", help="image folder; GUI if omitted")
    parser.add_argument("--version", action="version", version=__version__)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import os
    import pathlib
    import subprocess
    import tkinter as tk
    import urllib.request
    from tkinter import filedialog

    import cv2
    import torch
    import tqdm  # torch.version.cuda statt Treiberversion

    ROOT = pathlib.Path(__file__).resolve().parent
    MODEL_DIR, WEIGHT_DIR = ROOT / "Models", ROOT / "weights"
    MODEL_DIR.mkdir(exist_ok=True)
    WEIGHT_DIR.mkdir(exist_ok=True)

    CUDA_OK = torch.cuda.is_available()
    if CUDA_OK:
        print(f"[CUDA] {torch.cuda.get_device_name(0)} • Runtime {torch.version.cuda}")
    else:
        print("[WARN]  Keine CUDA-GPU gefunden – CPU-Modus.")

    def clear_vram() -> None:
        if CUDA_OK:
            torch.cuda.empty_cache()

    def run(cmd, cwd=None):
        subprocess.run(
            cmd if isinstance(cmd, list) else cmd.split(), cwd=cwd, check=True
        )

    def wget(url: str, dst: pathlib.Path) -> None:
        if not dst.exists():
            print(f"[Download] {dst.name}")
            urllib.request.urlretrieve(url, dst)

    def clone(url: str) -> pathlib.Path:
        tgt = MODEL_DIR / pathlib.Path(url).stem
        if not tgt.exists():
            run(["git", "clone", "--depth", "1", url, str(tgt)])
        return tgt

    MODELS = {
        "pidinet": {
            "repo": "https://github.com/hellozhuo/pidinet",
            "url": "https://huggingface.co/lllyasviel/Annotators/resolve/main/table5_pidinet.pth",
            "file": WEIGHT_DIR / "table5_pidinet.pth",
        },
        "diffedge": {
            "repo": "https://github.com/GuHuangAI/DiffusionEdge",
            "url": "https://huggingface.co/BRIAAI/DiffEdge/resolve/main/diffedge_swin.pth",
            "file": WEIGHT_DIR / "diffedge_swin.pth",
        },
        "edter": {
            "repo": "https://github.com/MengyangPu/EDTER",
            "url": "https://download.openmmlab.com/mmsegmentation/v0.5/edter/edter_bsds.pth",
            "file": WEIGHT_DIR / "edter_bsds.pth",
        },
    }

    for m in MODELS.values():
        clone(m["repo"])
        wget(m["url"], m["file"])

    root = tk.Tk()
    root.withdraw()
    folder = args.dir or filedialog.askdirectory(title="Bilderordner wählen")
    if not folder:
        sys.exit()
    in_dir = pathlib.Path(folder)

    EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    out_dirs = {k: (in_dir / k).mkdir(exist_ok=True) or in_dir / k for k in MODELS}

    def to_line(gray):
        if gray.mean() > 127:
            gray = 255 - gray
        return cv2.threshold(gray, 32, 255, cv2.THRESH_BINARY)[1]

    def call_pidinet(img, out):
        r = MODEL_DIR / "pidinet"
        run(
            [
                sys.executable,
                r / "demo.py",
                "--config",
                r / "configs/pidinet/table5_pidinet.yaml",
                "--model",
                MODELS["pidinet"]["file"],
                "--input",
                img,
                "--save",
            ]
        )
        tmp = (
            pathlib.Path(img)
            .with_suffix(".png")
            .with_name(pathlib.Path(img).stem + "_edge.png")
        )
        cv2.imwrite(out, to_line(cv2.imread(str(tmp), 0)))
        tmp.unlink(missing_ok=True)
        clear_vram()

    def call_diffedge(img, out):
        r = MODEL_DIR / "DiffusionEdge"
        run(
            [
                sys.executable,
                r / "demo.py",
                "--checkpoint",
                MODELS["diffedge"]["file"],
                "--input",
                img,
                "--save",
                "edge_tmp.png",
                "--fp16",
            ]
        )
        cv2.imwrite(out, to_line(cv2.imread("edge_tmp.png", 0)))
        os.remove("edge_tmp.png")
        clear_vram()

    def call_edter(img, out):
        r = MODEL_DIR / "EDTER"
        run(
            [
                sys.executable,
                r / "demo/test_single.py",
                "--config",
                r / "configs/edter_bsds.py",
                "--checkpoint",
                MODELS["edter"]["file"],
                "--img",
                img,
                "--out",
                "edge_tmp.png",
            ]
        )
        cv2.imwrite(out, to_line(cv2.imread("edge_tmp.png", 0)))
        os.remove("edge_tmp.png")
        clear_vram()

    DISPATCH = {"pidinet": call_pidinet, "diffedge": call_diffedge, "edter": call_edter}

    imgs = [p for p in in_dir.rglob("*") if p.suffix.lower() in EXT]
    for p in tqdm.tqdm(imgs):
        for k in DISPATCH:
            dst = out_dirs[k] / f"{p.stem}_{k}.png"
            if not dst.exists():
                try:
                    DISPATCH[k](str(p), str(dst))
                except Exception as e:
                    print(f"[{k}] {p.name}: {e}")

    print("✓ fertig")


if __name__ == "__main__":
    main()
