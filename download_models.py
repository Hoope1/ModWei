# download_models.py

from dataclasses import dataclass
from pathlib import Path
import requests

# Verzeichnisstruktur
MODEL_DIR = Path("models")
WEIGHT_DIR = MODEL_DIR / "weights"
WEIGHT_DIR.mkdir(parents=True, exist_ok=True)

# erlaubte Bild-Extensions (falls du später Filtern nach Bilddateien nutzt)
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

@dataclass
class ModelInfo:
    repo: str
    url: str
    file: Path

MODELS = {
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

def download_model(model: ModelInfo):
    print(f"Downloading {model.url} → {model.file}")
    resp = requests.get(model.url, stream=True)
    resp.raise_for_status()
    with open(model.file, "wb") as f:
        for chunk in resp.iter_content(16 * 1024):
            f.write(chunk)

if __name__ == "__main__":
    for name, info in MODELS.items():
        download_model(info)
    print("Alle Modelle heruntergeladen.")
