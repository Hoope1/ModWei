# AGENTS.md  –  Projekt-Root  (“Edge-Batch-Runner”)

> **Zweck:** Diese Datei ist das oberste Drehbuch für Codex-Agenten.  
> Sie schreibt verbindlich vor, **_wann_**, **_wie_** und **_mit welchen Exit-Codes_** der
> komplette Setup-->Batch-->Cleanup-Workflow abläuft.  
> Unterordner-AGENTS.md dürfen Details präzisieren, nicht aber die hier definierten
> Pflichtprüfungen auslassen.

## 0. Rahmenbedingungen  (Vorbedingung = Blocker)
- **OS** Windows 11 (Build ≥ 22000)  
- **Python** 3.10.x, erreichbar via `py -3.10 --version`  
- **GPU** CUDA-fähige NVIDIA Karte (CC ≥ 7.5) + Treiber-Serie ≥ R 530  
- **Platz** ≥ 10 GB frei im Projekt-Volume  
- **Netz** Port 443 offen für GitHub & Hugging Face – andernfalls Abbruch mit Exit 30  

## 1. Verzeichnis-Konvention

<root>
│  AGENTS.md          ← diese Datei
│  run_edge_cuda.bat
│  requirements_edge.txt
│  edge_batch_runner.py
├─ Models\             (wird geklo nt / gepflegt)
├─ weights\            (nur .pth / .ckpt Dateien)
└─ venv\               (autogeneriert)
```
> **Regel:** Namen **niemals** ändern. Scripts rely on them.2. Code-Stil & Tools

Black (v. ≥24.4) mit --line-length 120

isort (v. ≥ 5.13) – Profil „black“

Keine Tab-Indents; nur 4 Spaces

Max. 1 Import pro Zeile, keine „Star-Imprts“


3. Install- & Build-Pipeline (9 Phasen – muss in genau dieser Reihenfolge durchlaufen)

Phase 0 Virtuelle Umgebung

1. Existenz venv/Scripts/python.exe prüfen.


2. Falls fehlt → run_edge_cuda.bat --only-venv ausführen.


3. Erfolg prüfen: "%~dp0venv\Scripts\python.exe" -m pip --version.



Phase 1 Pip-Upgrade & NumPy < 2

Befehlskette:

"%VENV_PYTHON%" -m pip install --upgrade pip
"%VENV_PYTHON%" -m pip install "numpy<2"

Validierung:

import numpy, sys; assert int(numpy.__version__.split('.')[0])<2, sys.exit(13)


Phase 2 PyTorch 2.2 + cu118

Install-Befehl (eine Zeile!):

"%VENV_PYTHON%" -m pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 torchaudio==2.2.0+cu118 --index-url https://download.pytorch.org/whl/cu118

Test:

import torch, sys; assert torch.cuda.is_available() and torch.version.cuda=="11.8", sys.exit(15)


Phase 3 OpenMMLab-Stack

"%VENV_PYTHON%" -m pip install -U openmim
mim install "mmcv==2.2.0" --timeout 1200
"%VENV_PYTHON%" -m pip install "mmengine>=0.10.4" "mmsegmentation>=1.3.2"

Phase 4 Sonstige Requirements

"%VENV_PYTHON%" -m pip install -r requirements_edge.txt

dann "%VENV_PYTHON%" -m pip check → muss “No broken requirements” melden.


Phase 5 Repos & Checkpoints

Siehe Unterordner Models/AGENTS.md & weights/AGENTS.md

Hauptregel: Nichts klonen oder herunterladen, wenn Hash bereits stimmt.


Phase 6 Hauptskript starten

Befehl: "%VENV_PYTHON%" edge_batch_runner.py

Muss GPU-Zeile drucken und Tk-Dialog öffnen.

Abbruch bei „Cancel“ → Exit 0 (kein Stack-Trace).


Phase 7 Batch-Verarbeitung

Ablaufdefinition: siehe Unterabschnitt „Pipeline-Details“ weiter unten

tqdm muss exakt so viele Schritte zeigen wie Bilder×Modelle behandelt wurden.


Phase 8 Cleanup

torch.cuda.empty_cache()

Rest-TMP-PNG <= 0

Exit-Codes:

0 = mind. 1 erfolgreiches PNG erzeugt

20 = kein Bild gefunden

21 = alle Modellläufe fehlgeschlagen

≥30 = Setup-Errors



4. Pipeline-Details (für Phase 7)

(Hier keine Shell-Befehle, sondern Soll-Beschreibung, damit Codex das Verhalten validiert)

1. Unterordner:  pidinet/, diffedge/, edter/ anlegen.


2. Pro Bild × Modell:

Demo-Script mit korrekten Pfaden + Low-VRAM-Flags aufrufen.

Edge-Map → to_line() → PNG im Unterordner speichern.

Temp-Dateien sofort löschen.



3. VRAM-Flush: nach jedem Bild.


4. Error-Handling: Einzelner Modell-Fail bricht Batch nicht ab.



5. CI-Checks (vor Merge)

1. python edge_batch_runner.py --help ↦ Exit 0


2. black --check . && isort --check .


3. SHA-256 der drei .pth mit SOLL-Hash vergleichen.


4. Max. 15 % Lines changed ohne passenden Unit-/Smoke-Test ➔ Block-Merge.



6. PR-Richtlinien

Titel: [Feat] …, [Fix] …, [Docs] …, [Refactor] …

Beschreibung:

1. Was wurde geändert?


2. Warum?


3. “Testing Done” mit Konsolen-Paste der erfolgreichen Batch-Verarbeitung.




7. Dokumentations-Regeln

Funktions-Docstrings im Google-Style.

Minimale Inline-Kommentare (Deutsch, ganze Sätze).

Ein README-Update pro öffentlich sichtbarer Verhaltensänderung.


8. Sicherheits-/Lizenz-Pflichten

Keine Checkpoints > 1 GB in Git (nur Download on-demand).

MIT-Lizenz-Header in neuen Python-Dateien.

Verweis auf Upstream-Lizenzen (PiDiNet BSD-3, DiffusionEdge Apache-2.0, EDTER Apache-2.0).


> Abbruchbedingung (Fail-Fast): Jeder Verstoß gegen Abschnitt 0 bis 3 endet mit sofortigem Prozess-Exit (Code ≥ 30) und Konsolenlog „SETUP-HALTED“.
