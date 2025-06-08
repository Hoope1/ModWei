# AGENTS.md
#
# Diese Datei steuert alle Schritte, die Codex beim ersten Task‐Run
# ausführen muss, um die Edge-Batch‐Pipeline lauffähig zu machen.
# Sie folgt dem OpenAI-Agenten-Schema:   Abschnitte ➜ Reihenfolge ➜ Befehle.
# Jeder Shell-Block wird synchron ausgeführt; Fehler beenden den Run.

## 0. Ziel
- Vollautomatisches Setup & Ausführung von `edge_batch_runner.py`
- Unterstützung für NVIDIA-GPUs (CUDA 11.8) mit nur 4 GB VRAM
- Download aller benötigten Repos, Gewichte und Python-Pakete
- Keine Re-Downloads oder Neu-Installationen, falls bereits vorhanden
- Maximale Nachvollziehbarkeit durch konsolenlogische Einzelschritte

---

## 1. Virtuelle Umgebung anlegen/aktivieren
**Bedingung:** Wenn der Ordner `venv/` fehlt.  
```cmd
cmd: call "%~dp0run_edge_cuda.bat" --only-venv

2. Pip aktualisieren & NumPy pinnen

Begründung: PyTorch < 2.2 kollidiert mit NumPy ≥ 2; wir pinnen daher
numpy<2 nach dem obligatorischen pip install --upgrade pip.

cmd: "%VENV_PYTHON%" -m pip install --upgrade pip
cmd: "%VENV_PYTHON%" -m pip install "numpy<2"

3. CUDA-fähiges PyTorch 2.2 installieren

Index: https://download.pytorch.org/whl/cu118

cmd: "%VENV_PYTHON%" -m pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 torchaudio==2.2.0+cu118 --index-url https://download.pytorch.org/whl/cu118

4. OpenMMLab-Stack (mmcv / mmengine / mmsegmentation)

cmd: "%VENV_PYTHON%" -m pip install -U openmim
cmd: mim install "mmcv==2.2.0" --timeout 1200
cmd: "%VENV_PYTHON%" -m pip install "mmengine>=0.10.4" "mmsegmentation>=1.3.2"

5. Sonstige Abhängigkeiten

cmd: "%VENV_PYTHON%" -m pip install -r requirements_edge.txt

6. Repositories & Gewichte prüfen / herunterladen

# PiDiNet
cmd: git -C Models\pidinet pull || git clone --depth 1 https://github.com/hellozhuo/pidinet Models\pidinet
cmd: if not exist weights\table5_pidinet.pth curl -L -o weights\table5_pidinet.pth https://huggingface.co/lllyasviel/Annotators/resolve/main/table5_pidinet.pth

# DiffusionEdge
cmd: git -C Models\DiffusionEdge pull || git clone --depth 1 https://github.com/GuHuangAI/DiffusionEdge Models\DiffusionEdge
cmd: if not exist weights\diffedge_swin.pth curl -L -o weights\diffedge_swin.pth https://huggingface.co/BRIAAI/DiffEdge/resolve/main/diffedge_swin.pth

# EDTER
cmd: git -C Models\EDTER pull || git clone --depth 1 https://github.com/MengyangPu/EDTER Models\EDTER
cmd: if not exist weights\edter_bsds.pth curl -L -o weights\edter_bsds.pth https://download.openmmlab.com/mmsegmentation/v0.5/edter/edter_bsds.pth

7. Hauptskript starten

cmd: "%VENV_PYTHON%" edge_batch_runner.py

8. Abschluss & Aufräumen

cmd: "%VENV_PYTHON%" - <<PY
import torch, shutil, os
torch.cuda.empty_cache()
print("[DONE] VRAM-Cache geleert.")
# Optional: temporäre Edge-Maps entfernen, falls noch vorhanden
for tmp in ("edge_tmp.png",):
    if os.path.exists(tmp):
        os.remove(tmp)
PY


---

Code-Stil & Linting

Formatierung mit Black (120 Zeichen Zeilenlänge)

Import-Sortierung via isort

Vor jedem Commit:

cmd: "%VENV_PYTHON%" -m black .
cmd: "%VENV_PYTHON%" -m isort .


Tests

Smoke-Test: python edge_batch_runner.py --help MUSS ohne Fehler laufen

Bei Pull Requests: Upload von 1-2 Beispielbildern (< 1 MB) zur CI-Strecke


Lizenz-Hinweise

Dieses Projekt selbst: MIT

Externe Modelle: jeweilige Upstream-Lizenzen (BSD-3, Apache-2.0)

Keine großen Binär-Artefakte in Git, stattdessen Download bei Bedarf


---

### Warum dieser Aufbau?

1. **Codex verarbeitet Commands sequenziell** – jeder Abschnitt mit `cmd:` wird im selben Sandbox-Container ausgeführt und gestoppt, wenn ein Exit-Code ≠ 0 zurückkehrt 5.  
2. **Environment-Bootstrap** über eine eigene Batch-Datei ist empfohlen, weil Windows-Nutzer sonst PowerShell- vs. cmd-Konflikte riskieren 6.  
3. **NumPy-Pinning** vermeidet ABI-Probleme, bis alle Abhängigkeiten (bes. MMCV) offiziell NumPy ≥ 2 unterstützen 7.  
4. **`openmim`** beschleunigt die Installation der korrekten MMCV-Wheels enorm und verhindert typische CUDA-Mismatch-Fehler 8.  
5. Das **Download-Fallback** prüft zuerst, ob Repo/Datei existiert, um unnötige Bandbreite zu sparen – ein empfohlenes Pattern in der offiziellen Codex-CLI-Demo 9.

Mit dieser `AGENTS.md` kann ein Codex-Agent das gesamte Projekt **ohne weitere menschliche Eingriffe** betriebsbereit machen, alle Modelle laden und direkt mit der Batch-Verarbeitung starten. Viel Erfolg beim Ausprobieren!10

