#!/usr/bin/env python3.10
"""Batch edge detection runner."""
from __future__ import annotations

import argparse
import subprocess
import sys
import tkinter as tk
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog
from typing import Callable, Dict

import cv2
import numpy as np

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover - no torch available
    torch = None  # type: ignore
from loguru import logger
from tqdm import tqdm

MODEL_DIR = Path("models")
WEIGHT_DIR = MODEL_DIR / "weights"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class ModelInfo:
    repo: str
    url: str
    file: Path


MODELS: Dict[str, ModelInfo] = {
    "pidinet": ModelInfo(
        repo="https://github.com/hellozhuo/pidinet",
        url="https://huggingface.co/lllyasviel/Annotators/resolve/main/table5_pidinet.pth",
        file=WEIGHT_DIR / "table5_pidinet.pth",
    ),
    "diffedge": ModelInfo(
        repo="https://github.com/GuHuangAI/DiffusionEdge",
        url="https://huggingface.co/BRIAAI/DiffEdge/resolve/main/diffedge_swin.pth",
        file=WEIGHT_DIR / "diffedge_swin.pth",
    ),
    "edter": ModelInfo(
        repo="https://github.com/MengyangPu/EDTER",
        url="https://download.openmmlab.com/mmsegmentation/v0.5/edter/edter_bsds.pth",
        file=WEIGHT_DIR / "edter_bsds.pth",
    ),
}


def run_cmd(cmd: list[str], cwd: Path | None = None) -> None:  # pragma: no cover
    """Execute a subprocess."""
    subprocess.run(cmd, cwd=cwd, check=True)


def clone_repo(url: str) -> Path:  # pragma: no cover
    """Clone the model repository if necessary."""
    path = MODEL_DIR / Path(url).stem
    if not path.exists():
        logger.info("Cloning {}", url)
        run_cmd(["git", "clone", "--depth", "1", url, str(path)])
    return path


def download_weight(url: str, dst: Path) -> None:  # pragma: no cover
    """Download the weight file if missing."""
    if dst.exists():
        return
    logger.info("Downloading {}", dst.name)
    urllib.request.urlretrieve(url, dst)


def clear_vram() -> None:
    """Release cached CUDA memory."""
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def to_line(gray: np.ndarray) -> np.ndarray:
    """Binarise a greyscale edge map."""
    if gray.mean() > 127:
        gray = 255 - gray
    _, thr = cv2.threshold(gray, 32, 255, cv2.THRESH_BINARY)
    return thr


def call_pidinet(img: Path, out: Path) -> None:  # pragma: no cover
    """Run PiDiNet on one image."""
    repo = MODEL_DIR / "pidinet"
    run_cmd(
        [
            sys.executable,
            str(repo / "demo.py"),
            "--config",
            str(repo / "configs/pidinet/table5_pidinet.yaml"),
            "--model",
            str(MODELS["pidinet"].file),
            "--input",
            str(img),
            "--save",
        ]
    )
    tmp = img.with_suffix(".png").with_name(img.stem + "_edge.png")
    cv2.imwrite(str(out), to_line(cv2.imread(str(tmp), 0)))
    tmp.unlink(missing_ok=True)
    clear_vram()


def call_diffedge(img: Path, out: Path) -> None:  # pragma: no cover
    """Run DiffusionEdge on one image."""
    repo = MODEL_DIR / "DiffusionEdge"
    run_cmd(
        [
            sys.executable,
            str(repo / "demo.py"),
            "--checkpoint",
            str(MODELS["diffedge"].file),
            "--input",
            str(img),
            "--save",
            "edge_tmp.png",
            "--fp16",
        ]
    )
    cv2.imwrite(str(out), to_line(cv2.imread("edge_tmp.png", 0)))
    Path("edge_tmp.png").unlink()
    clear_vram()


def call_edter(img: Path, out: Path) -> None:  # pragma: no cover
    """Run EDTER on one image."""
    repo = MODEL_DIR / "EDTER"
    run_cmd(
        [
            sys.executable,
            str(repo / "demo/test_single.py"),
            "--config",
            str(repo / "configs/edter_bsds.py"),
            "--checkpoint",
            str(MODELS["edter"].file),
            "--img",
            str(img),
            "--out",
            "edge_tmp.png",
        ]
    )
    cv2.imwrite(str(out), to_line(cv2.imread("edge_tmp.png", 0)))
    Path("edge_tmp.png").unlink()
    clear_vram()


DISPATCH: Dict[str, Callable[[Path, Path], None]] = {
    "pidinet": call_pidinet,
    "diffedge": call_diffedge,
    "edter": call_edter,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Batch edge detection")
    parser.add_argument("--input", type=Path, help="Folder with images")
    parser.add_argument(
        "--streamlit", action="store_true", help="Run with Streamlit GUI"
    )
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        logger.error("Unknown arguments: %s", unknown)
        sys.exit(40)
    return args


def choose_input_dir(given: Path | None) -> Path:
    """Return the input directory, using Tkinter if not provided."""
    if given:
        return given
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Select the image folder")
    if not folder:
        raise ValueError("No folder selected")
    return Path(folder)


def setup_models() -> None:
    """Ensure all models and weights are present."""
    MODEL_DIR.mkdir(exist_ok=True)
    WEIGHT_DIR.mkdir(exist_ok=True)
    for cfg in MODELS.values():
        clone_repo(cfg.repo)
        download_weight(cfg.url, cfg.file)


def process_images(in_dir: Path) -> None:  # pragma: no cover
    """Run all models on images inside the directory."""
    out_dirs = {k: in_dir / k for k in MODELS}
    for d in out_dirs.values():
        d.mkdir(exist_ok=True)
    imgs = [p for p in in_dir.rglob("*") if p.suffix.lower() in EXTENSIONS]
    for img in tqdm(imgs, desc="processing"):
        for name, func in DISPATCH.items():
            dst = out_dirs[name] / f"{img.stem}_{name}.png"
            if not dst.exists():
                func(img, dst)


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    try:
        input_dir = choose_input_dir(args.input)
    except ValueError as exc:
        logger.error(str(exc))
        return 20
    setup_models()
    process_images(input_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
