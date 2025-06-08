import os
import sys
import cv2
import pathlib
import urllib.request
import tkinter as tk
from tkinter import filedialog
import tqdm
from subprocess import run

# --- Pfaddefinitionen ------------------------------------
MODEL_DIR = pathlib.Path("models")
WEIGHT_DIR = MODEL_DIR / "weights"
MODEL_DIR.mkdir(exist_ok=True)
WEIGHT_DIR.mkdir(exist_ok=True)

# --- Modellbeschreibungen und Download --------------------
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

# Klone Repos und lade Modelle herunter
for m in MODELS.values():
    clone(m["repo"])
    if not m["file"].exists():
        print(f"[Download] {m['file'].name}")
        urllib.request.urlretrieve(m["url"], m["file"])

# --- GUI-Auswahl für Eingabeordner ------------------------
root = tk.Tk()
root.withdraw()
folder = filedialog.askdirectory(title="Bilderordner wählen")
if not folder:
    sys.exit()
in_dir = pathlib.Path(folder)

# --- Bildformate & Ausgabeordner vorbereiten --------------
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
out_dirs = {k: (in_dir / k) for k in MODELS}
for d in out_dirs.values():
    d.mkdir(exist_ok=True)

# --- Bildverarbeitung -------------------------------------
def to_line(gray):
    if gray.mean() > 127:
        gray = 255 - gray
    return cv2.threshold(gray, 32, 255, cv2.THRESH_BINARY)[1]

def clear_vram():
    pass  # Platzhalter, falls CUDA oder Torch verwendet wird.

# --- Modell-Aufrufe ---------------------------------------
def call_pidinet(img, out):
    r = MODEL_DIR / 'pidinet'
    run([
        sys.executable, r / 'demo.py',
        '--config', r / 'configs/pidinet/table5_pidinet.yaml',
        '--model', MODELS["pidinet"]["file"],
        '--input', img,
        '--save'
    ])
    tmp = pathlib.Path(img).with_suffix(".png").with_name(pathlib.Path(img).stem + "_edge.png")
    cv2.imwrite(out, to_line(cv2.imread(str(tmp), 0)))
    tmp.unlink(missing_ok=True)
    clear_vram()

def call_diffedge(img, out):
    r = MODEL_DIR / 'DiffusionEdge'
    run([
        sys.executable, r / 'demo.py',
        '--checkpoint', MODELS["diffedge"]["file"],
        '--input', img,
        '--save', 'edge_tmp.png',
        '--fp16'
    ])
    cv2.imwrite(out, to_line(cv2.imread('edge_tmp.png', 0)))
    os.remove('edge_tmp.png')
    clear_vram()

def call_edter(img, out):
    r = MODEL_DIR / 'EDTER'
    run([
        sys.executable, r / 'demo/test_single.py',
        '--config', r / 'configs/edter_bsds.py',
        '--checkpoint', MODELS["edter"]["file"],
        '--img', img,
        '--out', 'edge_tmp.png'
    ])
    cv2.imwrite(out, to_line(cv2.imread('edge_tmp.png', 0)))
    os.remove('edge_tmp.png')
    clear_vram()

DISPATCH = {
    "pidinet": call_pidinet,
    "diffedge": call_diffedge,
    "edter": call_edter
}

# --- Batch-Verarbeitung -----------------------------------
imgs = [p for p in in_dir.rglob("*") if p.suffix.lower() in EXT]
for p in tqdm.tqdm(imgs):
    for k, func in DISPATCH.items():
        dst = out_dirs[k] / f"{p.stem}_{k}.png"
        if not dst.exists():
            func(str(p), str(dst))
