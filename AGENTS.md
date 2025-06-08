# AGENTS.md  
**Version 1.1 — „Kein-Fragen-offen“-Edition**

> **Wofür ist diese Datei?**  
> Sie liefert dem OpenAI-Codex-Agenten ein **haargenaues Drehbuch**, damit er das Projekt _Edge-Batch-Runner_ vollkommen selbstständig einrichtet, ausführt und am Ende — ohne einen einzigen unbeantworteten Zwischenschritt — gültige, bereinigte PNG-Linienzeichnungen in den drei Unterordnern `pidinet/`, `diffedge/` und `edter/` ablegt.  
>  
> **Wichtig:** Jede Teilaufgabe ist so granular beschrieben, dass sie auch dann noch eindeutig bleibt, wenn der ausführende Automat keinerlei Kontext‐Rückfragen stellen dürfte.  

---

## 0 Rahmenbedingungen (muss **vor** jeder Ausführung zutreffen)

1. **Betriebssystem:** Windows 11 (Build ≥ 22000)  
2. **GPU:** NVIDIA-Karte mit Compute Capability ≥ 7.5 (getestet auf Quadro T1000 4 GB)  
3. **Treiber:** GeForce/Quadro-Treiber-Serie ≥ R 530 mit aktivem CUDA-Runtime-Support  
4. **Python:** Vorinstalliert in Version 3.10.x, global zugänglich über `py --version`  
5. **Festplattenplatz:** ≥ 10 GB frei auf derselben Partition, auf der sich das Projektverzeichnis befindet  
6. **Netzwerkzugang:** HTTPS-Ports (443) offen für GitHub und Hugging Face Downloads  

---

## 1 Verzeichnis-Konventionen (sollten **jetzt** schon existieren, sonst anlegen)

| Ordner | Muss vorhanden sein? | Zweck |
|-------|----------------------|-------|
| `<project-root>\` | **Ja** | Wurzel aller Pfade in dieser Anleitung |
| `run_edge_cuda.bat` | **Ja** | Batch-Wrapper zum Installieren + Starten |
| `requirements_edge.txt` | **Ja** | Liste aller Nicht-CUDA-Python-Pakete |
| `Models\` | **Nein** | Wird vom Agent angelegt & gefüllt |
| `weights\` | **Nein** |   »   |
| `venv\` | **Nein** | Eigenständige Python-Umgebung |

_Namens- und Groß-/Kleinschreibung **nicht** ändern – spätere Skripte erwarten exakt diese Pfade._

---

## 2 Schritt-für-Schritt-Checkliste  
*(jede Hauptnummer muss abgeschlossen und validiert sein, bevor die nächste beginnt)*

### 2.1 Virtuelle Umgebung vorbereiten

1. **Existenz prüfen:**  
   *Wenn* `<project-root>\venv\Scripts\python.exe` **nicht** existiert,  
   → `run_edge_cuda.bat --only-venv` aufrufen.  
2. **Erfolg verifizieren:** In derselben Konsole `"%~dp0venv\Scripts\python.exe" -m pip --version` ausführen.  
   Ausgabe muss ohne Fehlermeldung erscheinen.  
3. **Fehlerfall:**  
   - Falls `pip` fehlt → Batch bricht ab → Stop & Log „VENV-FAILED“.  
   - Erst wenn neu angelegte venv einwandfrei antwortet, weiter zu 2.2.

### 2.2 Pip aktualisieren & NumPy pinnen

1. **Pip-Upgrade:**  
   `python -m pip install --upgrade pip` ausführen (innerhalb der venv).  
2. **Kompatibles NumPy installieren:**  
   `python -m pip install "numpy<2"` — egal, ob bereits vorhanden oder nicht, Version < 2 ist Pflicht.  
3. **Validierung:**  
   `python - <<py\nimport numpy, sys; assert int(numpy.__version__.split('.')[0]) < 2, sys.exit(13)\npy`  
   Exit-Code **0** heißt OK.

### 2.3 CUDA-fähiges PyTorch installieren

1. **Installationsquelle:** Offizieller PyTorch-Index für **cu118**.  
2. **Zu installierende Pakete:**  
   `torch==2.2.0+cu118`, `torchvision==0.17.0+cu118`, `torchaudio==2.2.0+cu118`.  
3. **Nachkontrolle:**  
   ```python
   import torch, sys; 
   assert torch.version.cuda == "11.8", sys.exit(14)
   assert torch.cuda.is_available(), sys.exit(15)

→ Exit-Code 0 oder logische Fehlermeldung und Abbruch.

2.4 OpenMMLab-Stack einrichten

1. openmim installieren/aktualisieren.


2. mmcv==2.2.0 exakt passend zu cu118 via mim install.


3. mmengine & mmsegmentation jeweils neueste stabile Version ≥ 0.10 & ≥ 1.3.


4. Smoke-Test:

import mmcv, mmengine, mmseg; print("MMCV", mmcv.__version__)



2.5 Sonstige Python-Abhängigkeiten

1. Inhalt von requirements_edge.txt ohne Änderung installieren.


2. Nachinstallation: python -m pip check muss “No broken requirements” melden.



2.6 Model-Repos & Checkpoints

> Regel: Nicht neu herunterladen, falls Repos bereits existieren und Gewichte bereits vollständig sind (Dateigröße ≥ 95 % der erwarteten Bytes).



Modell-Tag	Git-Repo (Zielordner)	Checkpoint-URL	Zieldatei

pidinet	Models\pidinet\	…/table5_pidinet.pth	weights\table5_pidinet.pth
diffedge	Models\DiffusionEdge\	…/diffedge_swin.pth	weights\diffedge_swin.pth
edter	Models\EDTER\	…/edter_bsds.pth	weights\edter_bsds.pth


Ablauf pro Eintrag (nacheinander, nicht parallel!):

1. Wenn Zielordner leer oder fehlt → git clone --depth 1 <repo-url> <ziel>.


2. Prüfen, ob Zieldatei existiert u/o Hash stimmt; sonst per HTTP GET herunterladen.


3. Hash-Vergleich bei allen 3 Gewichten (SHA-256 → Sollwerte in Kommentarblock am Dateiende notieren).


4. Wenn Download‐ oder Hash-Fehler → Abbruch & Log „WEIGHT-MISSING-<TAG>“.



2.7 Start des Hauptskripts

1. GPU-Info ausgeben: edge_batch_runner.py muss beim Start eine Zeile mit
[CUDA] <GPU-Name> • Driver <Version> • VRAM <x.x> GB drucken.


2. Benutzer­dialog: Tk-Dialog zur Bildordner­wahl muss erscheinen.


3. Benutzerinteraktion erzwingen: Warte solange, bis Ordner gewählt wurde oder Dialog abgebrochen wird.


4. Abbruchfall: Wenn Benutzer „Cancel“ wählt → sauberen Exit mit Code 0, ohne Stack-Trace.



2.8 Verarbeitungs-Pipeline

> Dieser Abschnitt definiert das Soll-Verhalten – der Agent soll keinen Code auflisten, sondern sicherstellen, dass der existierende Code exakt diese Ergebnisse erzielt.



1. Für jedes Bild (Erweiterung in {jpg,jpeg,png,bmp,tif,tiff,webp}):
1.1. Unterordner-Erstellung
Vor dem ersten Durchlauf eines Modells muss der Agent sicherstellen, dass
pidinet\, diffedge\ und edter\ im Bildordner existieren.
1.2. Pro Modell (Reihenfolge beliebig, aber konstant)
* Aufruf des zugehörigen Demo-/Inference-Scripts mit:
- Eingabepfad
- Checkpoint
- Ausgabepfad (temporär)
- Optionalen Low-VRAM-Flags (fp16 oder Tiling), wenn VRAM < 4 GB erkannt wurde
* Nach erfolgreicher Inferenz:
- Laden des Graustufen-Edge-Maps → Funktion to_line() anwenden
- Speichern als PNG im Zielunterordner mit Namensschema
<Originalname>_<modelltag>.png
- Temporäre Edge-Maps löschen (Speicherplatz < 500 MB halten).
1.3. VRAM freigeben: Nach jedem Bild → torch.cuda.empty_cache() aufrufen.


2. Fortschrittsanzeige: Eine tqdm-Leiste soll sichtbar sein, die bei jedem Bildfortschritt um exakt 1 Schritt hochzählt.


3. Fehlertoleranz:

Sollte ein Modell auf ein Bild scheitern, darf das Batch-Skript weiterlaufen; es loggt nur [FAIL] <tag> on <file>: <Fehlertext> in die Konsole.

Nach Abschluss Druck einer Zusammenfassung:

[Done]  Results:
  pidinet  →  <Pfad>
  diffedge →  <Pfad>
  edter    →  <Pfad>




2.9 Nachbereitung & Aufräumen

1. CUDA-Cache: Unabhängig vom Erfolg → nochmals torch.cuda.empty_cache().


2. Temporäre Dateien: Alle in Schritt 2.8 erzeugten Zwischen-PNGs oder Log-Files entfernen.


3. Exit-Code:

0: Alle Modelle mindestens 1 Bild erfolgreich verarbeitet.

20: Kein Bild im Ordner gefunden.

21: Alle Modellläufe fehlgeschlagen.
Fehlercodes > 21 reserviert für spezifische interne Checks.





---

3 Erwartete Endresultate (akzeptanz­kritisch)

Kriterium	Muss erfüllt sein

Ordnerstruktur	Im ursprünglich gewählten Bildordner müssen drei neue Unterordner existieren (pidinet, diffedge, edter).
Dateinamen	Für jedes Eingabebild liegen bis zu drei PNGs vor – exakt ein PNG pro Modell.
Bildinhalt	Schwarze Linien auf rein weißem Hintergrund, keine Graustufen, Auflösung identisch zur Eingabe.
Prozess-Log	Konsole zeigt keine Tracebacks (außer gezielt gefangene Fehlermeldungen).
Speicher-Fußabdruck	Nach Programmende ≤ 2 GB VRAM belegt und ≤ 500 MB temporärer Speicher im gesamten Projektverzeichnis.



---

4 Fehler­behandlung (konsequent anwenden)

Install-Fehler → sofortiger Abbruch, Exit-Code > 30, Log‐Datei install_error.log anlegen.

Hash-Mismatch bei Checkpoints → Löschen der betroffenen Datei + erneuter Download-Versuch.

Unbekannter CLI-Parameter → Skript verweigert Ausführung mit Exit-Code 40.

VRAM-OOM während Inferenz → erneuter Versuch mit halber Eingabeauflösung (OpenCV Resize), danach endgültiger [FAIL].



---

5 Konformitäts-Checks vor Commit (CI-Friendly)

1. python -m pip check ohne Fehler.


2. edge_batch_runner.py --help gibt Usage-Text und endet mit Code 0.


3. Black + isort laufen ohne Änderungen (otherwise: „Format before commit!“).


4. SHA-256-Summen der drei Checkpoints entsprechen den im Kommentar hinterlegten Sollwerten.




---

6 Lizenz, Autor, Versionierung

Dieses Setup-Skript: MIT-Lizenz

Modell-Repos: eigene Lizenzen gem. Upstream

Änderungen an dieser Datei nur per Pull-Request und Versionseintrag im Kopfbereich



---

> Merke: Jede einzelne Nummer oder Unternummer aus Abschnitt 2 gilt als „roter Faden“.
Kein Schritt darf übersprungen, zusammengezogen oder in anderer Reihenfolge ausgeführt werden,
solange nicht explizit ein „Fehlerfall-Abzweig“ hierfür vorgesehen ist.
Entdeckt der Agent eine Unklarheit, abbrechen und Fehlermeldung ausgeben — niemals raten!





