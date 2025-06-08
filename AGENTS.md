# AGENTS.md – Projekt-Root („Edge-Batch-Runner“)

> **Zweck:** Diese Datei ist das oberste Drehbuch für den Codex-Agenten. Sie beschreibt verbindlich, **wann**, **wie** und mit welchen Exit-Codes der vollständige Setup→Batch→Cleanup-Workflow abläuft. Unterordner-AGENTS.md dürfen Detailanweisungen hinzufügen, aber dürfen die hier definierten Pflichtprüfungen **nicht** auslassen.

## 0. Rahmenbedingungen  (Vorbedingung = Blocker)

* **OS:** Windows 11 (Build ≥ 22000)
* **Python:** 3.10.x (z.B. `py -3.10 --version`)
* **GPU:** CUDA-fähige NVIDIA-Karte (Compute Capability ≥ 7.5, getestet z.B. Quadro T1000 4GB)
* **Treiber:** GeForce/Quadro-Treiber ≥ R530 mit aktiviertem CUDA-Runtime-Support
* **Platz:** ≥ 10 GB freier Speicher auf Projekt-Partition
* **Netzwerk:** Port 443 (HTTPS) offen für GitHub/HuggingFace – ist dies gesperrt, sofort Exit-Code 30 (“SETUP-HALTED”).

## 1. Verzeichnis-Konvention  (sollten **jetzt** existieren, sonst anlegen)

| Pfad                    | Muss vorhanden sein? | Zweck                                                   |
| ----------------------- | -------------------- | ------------------------------------------------------- |
| `<project-root>\`       | **Ja**               | Wurzelverzeichnis des Projekts.                         |
| `run_edge_cuda.bat`     | **Ja**               | Batch-Wrapper für Setup & Start.                        |
| `requirements_edge.txt` | **Ja**               | Liste aller Projekt-Abhängigkeiten (außer CUDA-Pakete). |
| `edge_batch_runner.py`  | **Ja**               | Hauptskript (Batch-Verarbeitung).                       |
| `tests\`                | **Ja**               | Enthält Unit-Tests (`pytest`).                          |
| `Models\`               | **Nein**             | Modelle-Repos (wird geklont/aktualisiert).              |
| `weights\`              | **Nein**             | Modell-Gewichte (nur Downloads, keine Commits).         |
| `venv\`                 | **Nein**             | Virtuelle Python-Umgebung.                              |

* **Regel:** Namen **nicht** ändern (Skripte und Pfade erwarten diese Bezeichnungen exakt).

## 2. Code-Stil & Werkzeuge

* **Python-Version:** 3.10.x (keine älteren Versionen).

* **Formatierung:** Black (Version ≥ 24.4) mit `--line-length 120`.

* **Import-Orden:** isort (Version ≥ 5.13) mit Profil „black“.

* **Einrückung:** Keine Tabs, nur 4 Leerzeichen.

* **Imports:** Maximal ein Import pro Zeile, **keine** Stern-Imports.

* **Linting:** Flake8 (Version ≥ 4.x) mit max-line-length 120, keine Lint-Fehler.

* **Typprüfung:** mypy (im strengen Modus, `--strict`), **keine** Typfehler. Alle Funktionen und Klassen mit Type Hints versehen.

* **Unittests:** Verwendung von pytest für Unit-Tests (Dateien in `tests/` anlegen).

* **Code Coverage:** Coverage-Report anfertigen; mindestens 80 % Abdeckung (gemessen mit `coverage run`).

* **Logging:** Loguru (aktuellste Version) für die Protokollierung einsetzen (statt `print`), um strukturierte Logs zu liefern.

* **GUI (optional):** Streamlit für eine Web-GUI (z.B. zum Auswählen des Eingabeordners) verwenden. Wenn `--streamlit` übergeben wird, soll die Anwendung über Streamlit laufen; sonst klassisch über Tkinter-Dialog.

## 3. Install- & Build-Pipeline (Phasen 0–8 – **genau in dieser Reihenfolge**; bei jedem Fehler sofort abbrechen)

**Phase 0 – Virtuelle Umgebung:**

1. Prüfe, ob `venv\Scripts\python.exe` existiert.
2. Falls **nicht**, führe `run_edge_cuda.bat --only-venv` aus (legt venv an).
3. Nach Erstellung: `"%~dp0venv\Scripts\python.exe" -m pip --version` ausführen.

   * Kein Fehler erlaubt. Bei Fehlschlag: Batch abbrechen, Exit-Code ≥30, Log „VENV-FAILED“.

**Phase 1 – Pip-Upgrade & NumPy:**

1. In venv: `python -m pip install --upgrade pip`.
2. Sicherstellen, dass NumPy in Version <2 installiert ist:

   ```bash
   python -m pip install "numpy<2"
   ```
3. Validierung:

   ```bash
   python - <<PYCODE
   ```

import numpy, sys
assert int(numpy.**version**.split('.')\[0]) < 2, sys.exit(13)
PYCODE

````
– Exit-Code 0 heißt Erfolg. Sonst Abbruch mit Code 13.

**Phase 2 – PyTorch 2.2 + CUDA 11.8:**  
1. Installiere exakt: `torch==2.2.0+cu118`, `torchvision==0.17.0+cu118`, `torchaudio==2.2.0+cu118` vom offiziellen PyTorch-Index:  
```bash
python -m pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 torchaudio==2.2.0+cu118 --index-url https://download.pytorch.org/whl/cu118
````

2. Teste:

   ```bash
   python - <<PYCODE
   ```

import torch, sys
assert torch.cuda.is\_available() and torch.version.cuda == "11.8", sys.exit(15)
PYCODE

````
– Exit-Code 0 heißt Erfolg. Bei Fehlern Abbruch (Code 14/15).

**Phase 3 – OpenMMLab-Stack:**  
1. Mim-Tool installieren: `python -m pip install -U openmim`.  
2. MMCV genau passend zu cu118:  
```bash
mim install "mmcv==2.2.0" --timeout 1200
````

3. Installiere aktuelle Versionen von MMEngine (≥0.10.4) und MMSegmentation (≥1.3.2):

   ```bash
   python -m pip install "mmengine>=0.10.4" "mmsegmentation>=1.3.2"
   ```
4. Smoke-Test:

   ```bash
   python - <<PYCODE
   ```

import mmcv, mmengine, mmseg
print("MMCV", mmcv.**version**)
PYCODE

````
– Ausgabe sichtbar und ohne Fehler.

**Phase 4 – Sonstige Anforderungen:**  
1. Installiere alle Pakete aus `requirements_edge.txt` (unverändert).  
2. Zusätzliche Tools:  
```bash
python -m pip install loguru pytest coverage flake8 mypy black isort streamlit
````

3. Validierung:

   ```bash
   python -m pip check
   ```

   – Ausgabe muss “No broken requirements” sein.

**Phase 5 – Repos & Checkpoints:**
Regeln: Nur klonen/nicht neu herunterladen, falls vorhanden & richtig.

| Modell       | Git-Repo-URL              | Zielordner         | Checkpoint-Datei      | Zieldatei                    |
| ------------ | ------------------------- | ------------------ | --------------------- | ---------------------------- |
| **pidinet**  | (z.B. PiDiNet Repo)       | `Models\pidinet\`  | …/table5\_pidinet.pth | `weights\table5_pidinet.pth` |
| **diffedge** | (z.B. DiffusionEdge Repo) | `Models\diffedge\` | …/diffedge\_swin.pth  | `weights\diffedge_swin.pth`  |
| **edter**    | (EDTER Repo)              | `Models\edter\`    | …/edter\_bsds.pth     | `weights\edter_bsds.pth`     |

Ablauf für jedes Modell (nacheinander):

* Wenn Zielordner fehlt oder leer: `git clone --depth 1 <repo-url> <zielordner>`.
* Ist der Ordner da, dann: `git -C <zielordner> pull --ff-only`.
* Checkpoint-Datei: existiert oder (falls Größe < 95% erwartet) fehlerhaft: lade per HTTP-GET herunter ins `weights\`-Verzeichnis.
* Berechne SHA-256 aller drei gewichte: sie müssen mit den im Kommentar hinterlegten Soll-Hashes übereinstimmen (siehe `/weights/AGENTS.md`).
* Bei Hash-Fehler oder Downloadproblem: Lösche ggf. defekte Datei, zweiter Versuch. Scheitert erneut, Abbruch mit Exit-Code 51 (“WEIGHT-MISSING-<TAG>”).

**Phase 6 – Hauptskript starten:**
Führe das Hauptskript aus:

```bash
"%VENV_PYTHON%" edge_batch_runner.py
```

* Der Start des Skripts muss zuerst GPU-Information ausgeben, z.B.:

  ```
  [CUDA] <GPU-Name> • Driver <Version> • VRAM <x.x> GB
  ```
* **Benutzer-Dialog:** Öffnet eine grafische Ordnerauswahl.

  * **Standard:** Tkinter-Dialog zur Auswahl des Bildordners.
  * **Optional:** Bei Übergabe von `--streamlit` soll stattdessen eine Streamlit-Web-GUI gestartet werden, die den Benutzer das Eingabeverzeichnis wählen lässt.
* Warte, bis der Benutzer einen Ordner auswählt oder abbrechen drückt.
* Abbruch: Wählt der Benutzer “Cancel”, dann sauber beenden mit Exit-Code 0 (kein Stack-Trace!).

**Phase 7 – Batch-Verarbeitung (Pipeline):**
*(Kein Shell-Code hier – nur Soll-Beschreibung, damit Codex das Verhalten prüfen kann)*

1. **Unterordner anlegen:** Im gewählten Bildordner sicherstellen, dass es die drei Unterordner `pidinet/`, `diffedge/` und `edter/` gibt (ggf. erstellen).
2. **Für jedes Eingabebild** (Formate: jpg, jpeg, png, bmp, tif, tiff, webp):
   2.1. Für jedes Modell (`pidinet`, `diffedge`, `edter`):

   * Rufe das jeweilige Inference-Script auf mit Parametern: Eingabebild, Pfad zum Checkpoint, temporäres Ausgabepfad.
   * Falls VRAM < 4GB erkannt wird: benutze Low-VRAM-Flag (`--fp16` oder Tiling) für das Modell.
   * Bei Erfolg: Lade die temporäre Graustufen-Edge-Map, wende `to_line()` an (Konvertierung zu Linienzeichnung).
   * Speichere das Resultat als PNG in den entsprechenden Ziel-Unterordner mit Namen `<Originalname>_<Modelltag>.png`.
   * Lösche die temporäre Datei sofort, um Speicher unter 500 MB zu halten.
     2.2. Nach jedem Bild: `torch.cuda.empty_cache()` aufrufen, um GPU-Speicher zu leeren.
3. **Fortschrittsanzeige:** Verwende eine tqdm-Fortschrittsleiste. Sie muss bei jedem Bildfortschritt um genau 1 steigen (also Gesamtschritte = Anzahl Bilder × Anzahl Modelle verarbeitet).
4. **Fehlertoleranz:** Scheitert ein Modell für ein Bild, bricht das Batch-Skript **nicht** komplett ab. Stattdessen:

   * Ausgabe einer Log-Meldung: `[FAIL] <Modelltag> on <Datei>: <Fehlertext>` (z.B. via Loguru).
   * Weiterverarbeitung mit nächstem Bild/Modell.
5. **Abschluss:** Nach Ende des Durchlaufs eine Zusammenfassung in der Konsole:

   ```
   [Done] Results:
     pidinet  -> <Pfad zur Unterordner pidinet>
     diffedge -> <Pfad zur Unterordner diffedge>
     edter    -> <Pfad zur Unterordner edter>
   ```

**Phase 8 – Nachbereitung & Aufräumen:**

1. `torch.cuda.empty_cache()` nochmals aufrufen.
2. Alle in Phase 7 erzeugten temporären Dateien (Zwischen-PNGs, Log-Files o.Ä.) löschen – es dürfen max. 500 MB temporärer Speicher übrig bleiben.
3. **Exit-Code:**

   * **0:** Mindestens ein Modell hat mindestens ein Bild erfolgreich verarbeitet.
   * **20:** Im Eingabeordner wurde kein Bild gefunden.
   * **21:** Alle Modell-Läufe sind fehlgeschlagen (kein Bild von keinem Modell verarbeitet).
   * **≥30:** Alle anderen Setup- oder Installationsfehler (siehe Abschnitt 4 unten).

---

## 4. Erwartete Endresultate (akzeptanzkritisch)

| Kriterium          | Muss erfüllt sein                                                                                                   |
| ------------------ | ------------------------------------------------------------------------------------------------------------------- |
| **Ordnerstruktur** | Im gewählten Bildordner existieren genau drei neue Unterordner: `pidinet`, `diffedge`, `edter`.                     |
| **Dateinamen**     | Für jedes Eingangsbild liegt bis zu drei PNG-Dateien vor – genau ein PNG pro Modell (`<Orig>_<Modell>.png`).        |
| **Bildinhalt**     | Jedes Ergebnisbild zeigt schwarze Kantenlinien auf rein weißem Hintergrund (keine Graustufen). Auflösung = Eingabe. |
| **Prozess-Log**    | In der Konsole dürfen **keine** unbehandelten Tracebacks erscheinen (nur die gezielt gefangenen `[FAIL]`-Zeilen).   |
| **Ressourcen**     | Am Programmende: ≤2 GB GPU-Speicher belegt und ≤500 MB temporäre Dateien im Projekt.                                |

---

## 5. Fehlerbehandlung (Fail-Fast-Regeln)

* **Installations-Fehler:** Jeder Fehler beim Install/Setup → sofortiger Abbruch. Exit-Code ≥30. Log-Datei `install_error.log` mit Fehlermeldung erstellen.
* **Hash-Mismatch:** Bei falschem SHA-256 → betroffene Datei löschen, erneuter Download-Versuch. Scheitert es wieder → Abbruch (Exit 51).
* **Unbekannter Parameter:** Übergabe eines CLI-Parameters, der nicht definiert ist → Skript sofort beenden mit Exit-Code 40.
* **VRAM-Overflow (OOM):** Fängt ein Modell eine CUDA-OOM (OutOfMemory) ab, erfolgt ein Wiederholungsversuch mit halbierter Eingangsauflösung (OpenCV-Resize). Scheitert dann endgültig → `[FAIL]`-Log und weiter mit nächsten.
* **Mypy-/Flake8-Fehler:** Treten bei Tests oder Checks Fehler auf, gilt dies als Setup-Fehler (siehe CI-Checks).

> **Hinweis:** Jeder Verstoß gegen Abschnitt 0–2 (Rahmenbedingungen bis Tools) führt zu sofortigem Prozess-Abbruch (Exit-Code ≥30) und Log „SETUP-HALTED“.

---

## 6. CI-Checks (vor Merge/Commit)

Vor jedem Merge müssen folgende Prüfungen **grün** sein:

* `python edge_batch_runner.py --help` muss einen Usage-Text ausgeben und mit Code 0 beenden.
* **Formatierung:** `black --check .` und `isort --check .` dürfen keine Änderungen vorschlagen.
* **Linting:** `flake8 --max-line-length 120 .` darf keine Fehler melden.
* **Typprüfung:** `mypy --strict .` darf keine Fehler ausgeben.
* **Unittests:** `pytest --maxfail=1 --disable-warnings -q` muss alle Tests bestehen.
* **Coverage:** `coverage run -m pytest` → Coverage mindestens 80 %.
* **SHA-256:** Die SHA-256-Hashes der drei Checkpoints in `weights/` müssen den Soll-Werten (siehe Kommentar) entsprechen.
* **Änderungsumfang:** Max. 15% der Zeilen verändert **ohne** neue Unit/Smoke-Tests – sonst Merge-Block.

---

## 7. Pull-Request-Richtlinien

* **Titel-Format:** Beginnt mit Typ-Tag: `[Feat]`, `[Fix]`, `[Docs]`, `[Refactor]` gefolgt von kurzer Beschreibung.
* **Beschreibung:**

  1. **Was** wurde geändert? (Kurzfassung).
  2. **Warum** diese Änderung? (Motivation).
  3. **Testing Done:** Konsolen-Auszug der erfolgreichen Ausführung (z.B. `edge_batch_runner.py`) anhängen.

---

## 8. Dokumentations-Regeln

* **Docstrings:** Google-Style für alle Funktionen/Klassen.
* **Kommentare:** Nur dort, wo nötig (Erläuterungen in Deutsch, vollständige Sätze). Keine unnötigen Anmerkungen.
* **Readme:** Jede öffentlich sichtbare Änderung (neues Feature, Workflow-Änderung) muss im `README.md` dokumentiert werden.

---

## 9. Sicherheits-/Lizenz-Pflichten

* **Keine Gewichte im Repo:** Modell-Checkpoints (Dateien >100 MB) dürfen **nicht** ins Git-Repo gelangen. Nur Download on-demand.
* **Abhängigkeiten:** Eigene Skripte unter MIT-Lizenz versehen. Upstream-Lizenzen respektieren (PiDiNet BSD-3, DiffusionEdge Apache-2.0, EDTER Apache-2.0).

---

<div style="border-top:1px solid #888;margin-top:1em"></div>

**Zusammenfassung:** Dieses AGENTS.md definiert den **exakten Ablauf** für Setup, Test und Ausführung unseres Edge-Batch-Runner-Projekts. Alle Schritte (0–9) sind lückenlos vorgeschrieben. Unterordner-AGENTS.md (s.u.) fügen nur Spezifika hinzu. Bei jeder Abweichung oder Fehler bedingungslos abbruch mit passendem Exit-Code. So ist reproduzierbare Vollautomatisierung gesichert.
