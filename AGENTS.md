## Übersicht und Ziele

* **Setup-Anweisungen:** Schrittweises Einrichten der Entwicklungsumgebung und Installation aller Abhängigkeiten.
* **Qualitätssicherung:** Verwendung von *Loguru* für Logging/Protokollierung, sowie automatisierte Prüfungen mit `pytest` (Tests), `coverage` (Testabdeckung), `mypy` (statische Typprüfung), `flake8` (Linting) und `black` (Formatierung).
* **Streamlit GUI:** Integration einer Streamlit-Oberfläche zur visuellen Kontrolle, inklusive Headless-Start für automatisierte Tests.
* **Batch-Verarbeitung von Bildern:** Regeln zur Verarbeitung ganzer Bilddatensätze mithilfe verschiedener Modelle (PiDiNet, DiffusionEdge, EDTER).
* **Recovery-Logik:** Fehlerbehandlungsstrategie nach dem Paradigma **„Recover → Retry → Continue“** für robuste Batch-Prozesse.
* **Codex-Steuerhinweise:** Projektspezifische Vorgaben für den Codex-Agent (keine Rückfragen, fehlerresistentes Verhalten, sofortiges Abbrechen bei kritischen Fehlern mit aussagekräftigen Exit-Codes).
* **Tests & Logging:** Richtlinien zur Erstellung und Ausführung von Tests sowie konsistente Nutzung von Logging für Debugging und Nachverfolgbarkeit.
* **Formatierung & PR-Richtlinien:** Einheitliche Code-Formatierung, Linting-Vorgaben und Richtlinien für Pull Requests, damit Beiträge konsistent und überprüfbar sind.

## Verzeichnisstruktur & Modelle

Das Projekt erwartet bestimmte **Unterordner** für Modelle und deren Gewichte. Große Modelldateien (Gewichte) werden nicht im Code-Repository selbst versioniert, sondern in separaten Verzeichnissen bereitgestellt. Der Agent muss die Präsenz und Integrität dieser Dateien überprüfen, bevor die Verarbeitung startet:

* Quellcode der Modelle liegt z.B. unter `./models/` (Modulcode für PiDiNet, DiffEdge, EDTER etc.).
* Gewichtungs-Dateien liegen z.B. unter `./weights/` (vortrainierte Modelle im binären Format, z.B. `.pth` oder `.pt` Dateien).

**Integritätsprüfung der Gewichte:** Für jede erwartete Gewichts-Datei ist ein bekannter Hash-Wert hinterlegt (SHA-256 oder ähnlich). Der Agent berechnet den Hash der vorliegenden Datei und vergleicht ihn mit dem Soll-Wert, um die Korrektheit zu verifizieren. Eine tabellarische Übersicht der Modelle und ihrer Gewichtungs-Dateien mit erwarteten Hashes:

| Modell       | Gewichts-Datei         | Erwarteter SHA-256 Hash (Beispiel) |
| ------------ | ---------------------- | ---------------------------------- |
| **PiDiNet**  | `weights/pidinet.pth`  | `abc123...def` (Soll-Hash)         |
| **DiffEdge** | `weights/diffedge.pth` | `ghi456...jkl` (Soll-Hash)         |
| **EDTER**    | `weights/edter.pth`    | `mno789...pqr` (Soll-Hash)         |

*Hinweis:* Die tatsächlichen Hash-Werte sind in der Projektdokumentation oder Konfiguration hinterlegt und müssen hier entsprechend gepflegt werden. Stimmen die Hashes nicht überein oder fehlt eine Datei, wird dies als Fehler behandelt (siehe Schritte unten zur Fehlerbehandlung).

## Setup der Entwicklungsumgebung

Stelle zunächst sicher, dass die richtige Python-Version und alle notwendigen Bibliotheken verfügbar sind. Es wird empfohlen, ein **virtuelles Environment** zu verwenden, um Abhängigkeiten sauber zu isolieren.

1. **Python-Version und Virtualenv:** Prüfe, ob **Python 3.x** (z.B. 3.10+) installiert ist. Erstelle ein virtuelles Environment (z.B. mit `python -m venv .venv`) und aktiviere es (`source .venv/bin/activate` auf Linux/Mac, `.\.venv\Scripts\activate` auf Windows).
2. **Projektquellcode bereitstellen:** Stelle sicher, dass der Projektordner **Edge-Batch-Runner** mit allen Dateien vorliegt. Falls ein Git-Repository genutzt wird, führe `git clone <repo-url>` aus oder aktualisiere den Code mit `git pull`. Wechsle in das Projektverzeichnis (`cd edge-batch-runner/`).
3. **Abhängigkeiten installieren:** Installiere alle Python-Abhängigkeiten aus der bereitgestellten **Anforderungsdatei**. Typischerweise:

   ```bash
   $ pip install -r requirements.txt
   ```

   Dadurch werden Bibliotheken wie *loguru*, *pytest*, *streamlit*, *torch* (für die Modelle) etc. installiert. Sollte es ein `pyproject.toml`/`poetry.lock` geben, kann alternativ `poetry install` genutzt werden. **Wichtig:** Erfolgt bei der Installation ein Fehler, bricht der Agent mit Exit-Code 1 ab (Fail-fast), da ohne vollständige Abhängigkeiten nicht fortgefahren werden kann.
4. **Grundlegende Ordner prüfen:** Verifiziere, dass wichtige Verzeichnisse existieren: `src/` oder ähnlicher Quellcode-Ordner, `tests/` für Tests, `models/` und `weights/` wie oben beschrieben. Sollten notwendige Ordner oder Dateien fehlen, protokolliert der Agent den Fehler und **bricht ab** (Exit-Code 1), da ein inkonsistentes Repository vorliegt.

## Initiale Modellsuche und Hash-Prüfung

Bevor fortgefahren wird, prüft der Agent die **Verfügbarkeit der Modell-Gewichtsdateien** und validiert deren Integrität:

5. **Gewichtsdateien verifizieren:** Für jede in der Tabelle oben aufgelistete Gewichtsdatei führt der Agent:

   * Eine Existenzprüfung durch: Ist die Datei z.B. `weights/pidinet.pth` vorhanden?
   * Eine Hash-Prüfung durch: Berechne den SHA-256 Hash der Datei (mit einem Python-Skript oder dem Tool `sha256sum`) und vergleiche mit dem erwarteten Wert.
   * **Fall A (Erfolg):** Wenn alle Dateien vorhanden sind **und** die Hashes stimmen, fahre fort mit Schritt 6.
   * **Fall B (Fehler):** Wenn eine Datei fehlt oder fehlerhaft ist:

     * Protokolliere eine **Fehlermeldung** mit Angabe, welche Datei betroffen ist.
     * **Recover:** Falls ein offline verfügbares **Download-Skript** existiert (z.B. `scripts/download_models.sh`), versucht der Agent dieses auszuführen, um die Datei zu beschaffen. Warte kurz auf dessen Abschluss.
     * **Retry:** Prüfe erneut das Vorhandensein und den Hash der Datei.
     * Falls die Datei jetzt korrekt vorliegt, fahre fort. Andernfalls **bricht der Agent ab** mit einem spezifischen Exit-Code (z.B. Exit-Code 2 für "Modellgewichte fehlen/ungültig"), da ohne gültige Modelle die Verarbeitung nicht möglich ist.

**Anmerkung:** Da der Codex-Agent im Ausführungsmodus **keinen Internetzugriff** hat, müssen alle notwendigen Gewichte im Vorfeld bereitgestellt oder im Setup-Skript heruntergeladen werden. Ein Abbruch an dieser Stelle weist den Benutzer darauf hin, die Gewichte manuell bereitzustellen und den Lauf erneut zu starten.

## Logging-Richtlinien (Loguru)

Eine konsistente **Protokollierung (Logging)** ist essentiell für Nachvollziehbarkeit und Fehleranalyse. In diesem Projekt wird die Bibliothek **Loguru** verwendet, um das Python-Logging zu vereinfachen und zu verbessern:

6. **Log-Konfiguration:** Stelle sicher, dass zu Beginn der Anwendung die Loguru-Logger konfiguriert sind. Beispielsweise:

   ```python
   from loguru import logger
   logger.add("run.log", rotation="10 MB", level="INFO")
   ```

   Dadurch werden Logs in die Datei `run.log` (mit Rotation bei 10 MB) sowie standardmäßig auf die Konsole ausgegeben. Das Log-Level kann je nach Bedarf angepasst werden (`DEBUG` für ausführlichere Diagnose, `INFO` für normalen Betrieb).
7. **Nutzung von Loguru:** Alle Module des Projekts sollen anstelle des Standard-`logging`-Moduls den Loguru-`logger` verwenden. Dies garantiert konsistente Formatierung und Thread-sicheres Logging. In jedem Python-Modul:

   ```python
   from loguru import logger
   ```

   verwenden und z.B. `logger.info("Nachricht")` oder `logger.error("Fehler: {}", err)` aufrufen.
8. **Fehler- und Ausnahme-Logging:** Kritische Abschnitte des Codes sind mit Loguru's *Decorator* oder *Context Manager* abgesichert, um Ausnahmen automatisch zu protokollieren. Beispiel:

   ```python
   @logger.catch
   def process_image(path: str):
       # ... Bildverarbeitungscode
   ```

   Der `@logger.catch` Dekorator stellt sicher, dass unerwartete Exceptions mit Stacktrace im Log landen, anstatt das Programm unprotokolliert abstürzen zu lassen.
9. **Logging in Tests:** Auch während des Testlaufs (pytest) sollen Logs erfasst werden, um im Fehlerfall Details zu haben. Der Agent konfiguriert ggf. `pytest` so, dass Log-Ausgaben im Fail-Fall angezeigt werden (z.B. mit `-s` oder Log-Capture Optionen). Allerdings sollten Tests selbst überwiegend verifizieren, nicht loggen – Logging dient hier hauptsächlich der Diagnose.
10. **Log-Level in verschiedenen Umgebungen:** Im Batch-Betrieb kann das Standard-Log-Level `INFO` oder `WARNING` sein, während für **Debugging** oder während Tests auch `DEBUG` aktiviert werden kann. Der Agent kann die Umgebungsvariable oder Konfiguration entsprechend setzen (z.B. `LOGGER_LEVEL=DEBUG`) wenn detaillierte Ausgabe benötigt wird.

## Testen mit Pytest und Coverage

Automatisierte Tests stellen sicher, dass der Code korrekt funktioniert und Änderungen nichts kaputt machen. Dieses Projekt verwendet **pytest** für Unit- und Integrationstests, ergänzt durch **Coverage** zur Messung der Testabdeckung:

11. **Tests ausführen:** Führe die Test-Suite mit Pytest aus:

    ```bash
    $ pytest
    ```

    Verwende ggf. Optionen wie `-q` (quiet) und `--disable-warnings`, außer wenn Debug-Informationen benötigt werden. Der Agent nutzt **Fail-Fast**: sollte ein Test fehlschlagen, bricht der Lauf an dieser Stelle mit Exit-Code 4 ab. (Die Nummer 4 steht hier beispielhaft für "Test fehlgeschlagen" – die Exit-Codes sind im Projekt einheitlich definiert.) Vor Abbruch werden die fehlgeschlagenen Tests und Fehlermeldungen protokolliert.
12. **Testabdeckung messen:** Nach erfolgreichem Testdurchlauf erstellt der Agent einen Coverage-Bericht. Z.B.:

    ```bash
    $ coverage run -m pytest
    $ coverage report -m
    ```

    Der Terminalbericht zeigt die Abdeckung (% der ausgeführten Zeilen pro Modul). Der Agent prüft, ob die **Coverage** über dem definierten Schwellenwert liegt (z.B. **90%**). Ist die Abdeckung zu niedrig, wird dies **protokolliert** als Warnung. (Ein Unterschreiten der Coverage führt **nicht** zum sofortigen Abbruch, aber es sollte idealerweise durch zusätzliche Tests behoben werden.)
13. **Testorganisation:** Stelle sicher, dass alle Tests im Verzeichnis `tests/` liegen und sinnvolle Namen haben. Der Agent achtet darauf, dass neue Funktionen von entsprechenden Tests begleitet werden (Test-Driven Development wird begrüßt). Sollten während des Agentenlaufs neue Tests generiert worden sein (etwa bei Fehlerbehebungen), führt der Agent diese erneut aus, bevor fortzufahren.
14. **Externe Aufrufe mocken:** Da der Codex-Agent offline läuft, dürfen Tests keine echten Internet- oder API-Aufrufe enthalten. Entsprechende Funktionen sind zu **mocken**. (Dies wurde in der Projekt-AGENTS.md hinterlegt, damit der Agent z.B. keine realen Downloads versucht.) Der Agent überprüft Tests auf potenzielle externe Aufrufe und schlägt vor, *Mock-Objekte* zu verwenden, falls solche Stellen gefunden werden.

## Statische Typprüfung (mypy)

Im gesamten Projekt wird auf **statische Typisierung** mittels Typannotationen geachtet. **mypy** wird verwendet, um die Typkorrektheit sicherzustellen:

15. **Type-Check ausführen:** Der Agent führt nach den Tests eine statische Typprüfung über den Code aus:

    ```bash
    $ mypy .
    ```

    Etwaige mypy-Fehlermeldungen (Inkompatible Typen, nicht behandelte None-Typen, etc.) führen zum **Abbruch mit Exit-Code 3** (steht für "Lint/Typecheck Fehler"), da solche Probleme die Codequalität beeinträchtigen oder zur Laufzeit Fehler verursachen könnten. Alle neuen Codebeiträge müssen mypy **fehlerfrei** passieren.
16. **Mypy-Konfiguration:** In der Regel ist mypy sehr strikt konfiguriert (ggf. in `mypy.ini` oder `pyproject.toml`). Beispielsweise: `strict = True`, `disallow_untyped_defs = True`, etc. Der Agent berücksichtigt diese Einstellungen. Falls externe Bibliotheken genutzt werden, für die keine Typ-Stubfiles vorhanden sind, kann in der Konfiguration `ignore_missing_imports = True` gesetzt sein, um unnötige Fehler zu vermeiden. Der Agent achtet aber darauf, dass für eigene Module alle Typen deklariert sind.
17. **Typisierungsrichtlinien:** Entwickler (und damit auch der Codex-Agent beim Generieren von Code) sollten alle Funktionen und Methoden **mit Typannotationen** versehen (PEP 484). Konventionen wie `Optional[...]` für Parameter, generische Typen für Collections und `-> None` für Funktionen ohne Rückgabewert sind einzuhalten. mypy dient als Wächter hierfür.

## Linting mit Flake8

Um codestyle und potenzielle Fehlerquellen zu prüfen, wird **Flake8** als Linter eingesetzt:

18. **PEP8-Stilprüfung:** Führe Flake8 über den Code aus:

    ```bash
    $ flake8
    ```

    Dies stellt sicher, dass der Code den gängigen PEP8 Stilrichtlinien entspricht (z.B. Zeilenlänge, Einrückungen, Import-Sortierung falls Plugins aktiv sind, etc.). **Alle Flake8-Warnungen/Fehler müssen behoben werden.** Der Agent bricht bei jeglichen Lint-Verstößen sofort ab (Exit-Code 3, zusammen mit Typfehlern kategorisiert).
19. **Konfiguration:** Flake8 ist ggf. über eine Konfigdatei (`setup.cfg` oder `.flake8`) eingestellt. Typischerweise ist die maximale Zeilenlänge auf 88 oder 100 gesetzt (wenn Black genutzt wird, wird dieser Wert synchron gehalten). Bestimmte Warnungen können ausgeblendet sein, z.B. `E501 line too long` falls Black übernimmt, oder spezielle Regeln für Naming. Der Agent liest die Konfigurationsdatei ein und befolgt sie.
20. **Lint-Fehlerbehandlung:** Treten Flake8-Fehler auf, protokolliert der Agent die betreffenden Stellen (Datei und Zeilennummer, sowie die Fehlermeldung). Automatische Korrekturen nimmt der Agent **nur** vor, wenn es sich um reine Formatierungsfehler handelt (siehe Black unten). Andere Lint-Verstöße (wie ungenutzte Variablen, falsche Importe) erfordern Code-Änderungen – hier würde der Agent einen Fehler melden und abbrechen, da konzeptionelle Anpassungen nötig sind, die nicht ohne Weiteres autonom erfolgen sollen.

## Automatische Formatierung mit Black

Die Code-Formatierung wird strikt von **Black** übernommen, um einen einheitlichen Stil zu gewährleisten und manuelle Formatierungsarbeit zu minimieren:

21. **Black-Check ausführen:** Zunächst prüft der Agent, ob der Code bereits Black-konform formatiert ist:

    ```bash
    $ black . --check
    ```

    Sollte Black Abweichungen feststellen, werden diese im Terminal ausgegeben.
22. **Automatische Formatierung:** Anstatt einen Fehler auszulösen, formatiert der Agent den Code automatisch:

    ```bash
    $ black .
    ```

    Alle Änderungen, die Black vornimmt, werden protokolliert (der Agent zeigt ein `diff` oder die Black-Ausgabe an). Nach diesem Schritt ist der Code sauber formatiert. Das Motto ist **"formatieren statt diskutieren"** – es werden keine manuellen Stilfragen erörtert, Black ist das Maß der Dinge.
23. **Black-Konfiguration:** Standardmäßig verwendet Black eine Zeilenlänge von 88 Zeichen. Falls im Projekt eine andere Vorgabe gilt (z.B. 100 oder 120), ist dies in `pyproject.toml` konfiguriert. Der Agent respektiert diese Einstellung. Weitere Black-Optionen wie Ausschluss bestimmter Pfade (z.B. Migrations, falls irrelevant) werden ebenso beachtet.
24. **Nachkontrolle:** Nach Ausführen von Black (und eventueller Behebung von Linting-Fehlern) führt der Agent **Schritt 18 (flake8)** und **Schritt 15 (mypy)** erneut durch, um sicherzustellen, dass Formatierungsänderungen keine neuen Lint- oder Typfehler erzeugt haben. Erst wenn alle diese Checks grün sind, geht es weiter. (Dies entspricht einer *Fail-fast* Strategie: sobald ein Schritt endgültig fehlschlägt, wird abgebrochen, ansonsten iteriert der Agent ggf. kleine Korrekturschritte bis alle Checks bestanden sind.)

## Streamlit-GUI testen

Das Projekt beinhaltet eine **Streamlit**-basierte GUI (grafische Oberfläche) zur interaktiven Nutzung. Um sicherzustellen, dass auch diese Oberfläche funktioniert und keine versteckten Fehler enthält, wird ein kurzer Starttest durchgeführt:

25. **GUI-Start im Headless-Modus:** Der Agent startet die Streamlit-App in einem headless Modus (ohne Browser-GUI) für einen kurzen Integrationstest:

    ```bash
    $ streamlit run app.py --server.headless true --browser.forceClose true
    ```

    (Angepasst auf den tatsächlichen Pfad/Dateinamen der Streamlit-App, z.B. `app.py` oder `gui.py` im Projekt.) Durch `--server.headless true` wird verhindert, dass Streamlit auf eine Benutzerinteraktion wartet oder ein Browser-Fenster öffnet.
26. **Laufzeitprüfung:** Der Agent lässt die GUI-Anwendung wenige Sekunden laufen und überwacht die Konsolenausgabe. Wichtig ist, dass **keine Exceptions** beim Start auftreten. Wenn die App erfolgreich startet (typischerweise gibt Streamlit im Terminal eine URL und Statusmeldungen aus), wird der Test als bestanden betrachtet.
27. **GUI stoppen:** Nach der Überprüfung beendet der Agent den Streamlit-Prozess wieder. (Gegebenenfalls durch Senden eines Keyboard-Interrupt oder, falls im Script vorgesehen, durch einen speziellen Parameter/test-Flag, um das Programm nach dem Init zu beenden.)
28. **Fehlerbehandlung:** Sollte Streamlit beim Start einen Fehler werfen (Syntax Error, Import Error, etc.), bricht der Agent ab **mit Exit-Code 5** und protokolliert die Fehlermeldung. Da die GUI ein integraler Bestandteil ist, gelten Startfehler als kritisch (fail-fast). Kleinere Warnungen (z.B. Deprecation Warnings) werden geloggt, aber führen nicht zum Abbruch, solange die App läuft.
29. **Sync mit Batch-Logik:** Der Agent achtet darauf, dass die in der GUI genutzte Logik mit der Batch-Verarbeitung konsistent ist (kein duplizierter oder auseinanderlaufender Code). Änderungen im Batch-Code sollten nach Möglichkeit **DRY** (Don't Repeat Yourself) auch von der GUI referenziert werden. Falls die GUI separaten Code enthält, sollten für beide Pfade Tests existieren.

## Regeln zur Batch-Verarbeitung (PiDiNet, DiffEdge, EDTER)

Kernstück des Projekts ist die **stapelweise Verarbeitung von Bilddateien** mithilfe verschiedener Edge-Detection-Modelle (PiDiNet, DiffusionEdge, EDTER). Der Agent befolgt folgende Regeln, um eine effiziente und robuste Verarbeitung sicherzustellen:

* **Modell-Lazy-Loading:** Die Modelle werden **nicht für jedes Bild neu geladen**, sondern jeweils einmal initialisiert. Zu Beginn der Verarbeitung lädt der Agent jedes benötigte Modell (z.B. durch `torch.load` eines Gewichts und Aufbau des Netzwerks) und behält die Modelle im Speicher. Dadurch wird die Verarbeitung vieler Bilder deutlich beschleunigt.
* **Batch-Ordner und Ausgabe:** Es wird erwartet, dass ein Eingabeordner (z.B. `input_images/`) mit den zu verarbeitenden Bildern vorhanden ist. Alle unterstützten Bildformate (typischerweise `.png`, `.jpg`) werden berücksichtigt. Für die Ausgabe sollte ein Ordner (z.B. `output/`) vorhanden sein; dieser kann vom Agenten auch angelegt werden, falls nicht existent.
* **Verarbeitung pro Bild:** Für **jedes Bild** im Eingabeordner durchläuft der Agent die folgenden Schritte:

  1. Lade das Bild in den Arbeitsspeicher (z.B. mit OpenCV, PIL oder einer geeigneten Bibliothek).
  2. **PiDiNet-Inferenz:** Wende das PiDiNet-Modell auf das Bild an, um eine Konturkarte zu erzeugen. Speichere das Ausgabe-Bild (z.B. `{bildname}_pidinet.png`) im Ausgabeordner.
  3. **DiffusionEdge-Inferenz:** Wende das DiffEdge-Modell auf das Bild an und speichere entsprechend (z.B. `{bildname}_diffedge.png`).
  4. **EDTER-Inferenz:** Wende das EDTER-Modell auf das Bild an und speichere z.B. `{bildname}_edter.png`.
* **Modellauswahl:** Falls die Verarbeitung nicht für alle Modelle erfolgen soll (z.B. via Konfiguration nur ein bestimmtes Modell wählen), berücksichtigt der Agent die Projektvorgaben oder Benutzerparameter. Standardmäßig werden aber alle drei Modelle nacheinander auf jedes Bild angewendet (für maximale Ausgabe).
* **Performance-Aspekt:** Die Verarbeitung kann ggf. auf **GPU** erfolgen, sofern verfügbar, um Geschwindigkeit zu erhöhen (z.B. Erkennung durch PyTorch `cuda`). Der Agent prüft, ob CUDA verfügbar ist, und nutzt es, andernfalls erfolgt CPU-Verarbeitung. Große Bildbatches verarbeitet der Agent sequentiell, es sei denn, die Hardware ermöglicht Parallelisierung und das Projekt ist darauf ausgelegt. Standard ist sequentiell, um Ressourcen zu schonen.
* **Ausgabebenennung und -format:** Die generierten Edge-Detections pro Modell sollen klar benannt sein (siehe oben). Das Format kann PNG sein, um verlustfreie Speicherung der Binär-Konturbilder zu gewährleisten. Falls Alphakanal nicht benötigt, 8-Bit Graustufen reicht, aber das hängt von der Modellimplementierung ab (der Agent verwendet die im Code vorgesehenen Methoden, meist liefern die Modelle Wahrscheinlichkeits- oder Maskenbilder, die eventuell normalisiert und als PNG gespeichert werden).
* **Protokollierung der Ergebnisse:** Nach erfolgreicher Verarbeitung eines Bildes für alle Modelle loggt der Agent eine Zusammenfassung, z.B.: `INFO - Bild XYZ.png verarbeitet: PiDiNet OK, DiffEdge OK, EDTER OK`. Sollte ein Modell für das Bild Fehler erzeugt haben (siehe Recovery unten), wird dies ebenfalls im Log vermerkt (z.B. `WARNING - Bild XYZ.png: EDTER fehlgeschlagen, übersprungen`).

## Recovery-Logik: Recover → Retry → Continue

Treten während der Batch-Verarbeitung **Fehler** auf, greift die dreistufige Recovery-Strategie, um den Prozess so robust wie möglich zu gestalten:

* **Recover (Wiederherstellen):** Sobald ein Fehler bei der Verarbeitung eines Bildes mit einem Modell auftritt (sei es ein Laufzeitfehler, Speicherüberlauf, etc.), unternimmt der Agent einen Wiederherstellungsschritt. Beispielmaßnahmen:

  * **Speicher leeren:** Falls GPU-Speicher knapp (Out-Of-Memory), führt der Agent `torch.cuda.empty_cache()` aus, um unbenutzten GPU-Speicher freizugeben.
  * **Modell neu laden:** Wenn der Fehler auf einen korrumpierten Modellzustand hindeutet, wird das betreffende Modell neu aus dem Gewicht geladen, um einen frischen Zustand zu erhalten.
  * **Alternative Pfade:** Bei Dateifehlern (z.B. Bild konnte nicht gelesen werden) überspringt dieser Schritt direkt zum Continue (denn Retry würde dasselbe Problem ergeben).
* **Retry (Wiederholen):** Nach dem Recover-Schritt versucht der Agent **erneut**, das aktuelle Bild mit dem gleichen Modell zu verarbeiten. Zuvor kann eine kurze Wartezeit eingebaut werden (z.B. 1-2 Sekunden *Backoff*), um dem System Zeit zu geben, sich zu stabilisieren. Der zweite Versuch wird genau protokolliert.

  * **Erneuter Fehler:** Sollte auch der Wiederholungsversuch fehlschlagen, bricht der Agent **nicht sofort** den gesamten Batch ab, sondern wechselt zum Continue-Schritt für dieses Bild/Modell.
  * **Erfolg im zweiten Anlauf:** Falls der Fehler durch Recover behoben wurde und der zweite Versuch erfolgreich ist, setzt der Agent den normalen Batch-Prozess fort, als wäre nichts gewesen. Im Log wird vermerkt, dass ein Fehler aufgetreten war, aber durch Wiederholung behoben wurde (z.B. `INFO - Zweiter Versuch erfolgreich für Bild XYZ mit Modell EDTER`).
* **Continue (Fortfahren):** Kann das Problem nicht gelöst werden (auch der Retry schlägt fehl), **überspringt** der Agent die Verarbeitung mit dem fehlerhaften Modell für das gegebene Bild. Das heißt:

  * Der Fehler wird im Log als **Fehlermeldung** festgehalten, inklusive Exception-Text/Stacktrace, damit Entwickler ihn später untersuchen können.
  * Der Agent fährt mit dem **nächsten Modell** (für dasselbe Bild) fort, oder falls alle Modelle für dieses Bild durch sind, mit dem **nächsten Bild**.
  * Wichtig: Der Gesamtprozess wird *nicht* komplett abgebrochen – so viele Bilder wie möglich sollen verarbeitet werden, selbst wenn einzelne Modelle auf einzelnen Bildern scheitern.
* **Globaler Fehlerzähler:** Der Agent führt Buch über alle aufgetretenen Fehler (z.B. in einer Variablen `error_count` und vielleicht detailliert in einer Liste). Diese Informationen fließen in den Abschlussbericht ein. Wenn die Fehlerzahl einen gewissen Schwellwert überschreitet (z.B. mehr als 50% der Bilder schlagen fehl), könnte der Agent entscheiden, dass etwas Grundlegendes nicht stimmt und den Prozess doch vorzeitig abbrechen. (Diese Schwelle ist konfigurierbar; standardmäßig wird jedoch versucht, **alle** Bilder zu verarbeiten.)

Die **Recover→Retry→Continue**-Logik stellt sicher, dass vorübergehende oder einzelne Fehler die gesamte Batch-Verarbeitung nicht ungewollt stoppen, während dennoch ernsthafte Probleme nicht unbemerkt bleiben.

## Codex-spezifische Steuerhinweise

Diese Sektion richtet sich speziell an den OpenAI Codex Agent, der diesen Leitfaden befolgt. Sie stellt sicher, dass der Agent im autonomen Modus das gewünschte Verhalten zeigt:

* **Keine Rückfragen:** Der Agent soll **keinerlei Rückfragen** oder Unklarheiten äußern, sondern gemäß diesem Skript handeln. Alle erforderlichen Informationen liegen vor; Interaktivität ist nicht vorgesehen. Insbesondere sollen keine Eingabe-Prompts vom Agent initiiert werden (auch nicht in der Streamlit-App, da dort `--headless` genutzt wird).
* **Autonomes Handeln:** Der Agent trifft Entscheidungen auf Basis der hier definierten Regeln (z.B. Recovery-Strategie, Abbruchkriterien) selbstständig. Es wird erwartet, dass er kleine Anpassungen (etwa Formatierung mit Black) ohne Bestätigung durchführt.
* **Fehlerresistenz:** Nicht jeder Warnhinweis oder nicht-kritische Fehler soll zum sofortigen Stopp führen. Der Agent unterscheidet zwischen **kritischen Fehlern** (z.B. Abhängigkeiten fehlen, Syntaxfehler, Tests schlagen fehl – dann Abbruch) und **handhabbaren Fehlern** (z.B. einzelne Bildverarbeitung schlägt fehl – dann überspringen). Durch Logging und die Recovery-Mechanismen wird Robustheit erreicht.
* **Fail-Fast mit Exit-Codes:** Sobald ein **nicht behebbarer Fehler** auftritt, beendet der Agent den Lauf unverzüglich mit einem aussagekräftigen Exit-Code. Hier die vereinbarten Codes (Beispiele):

  * `0` – Erfolgreiche Ausführung ohne Fehler.
  * `1` – Allgemeiner Setup-Fehler (Umgebung nicht korrekt, Abhängigkeit fehlt, o.ä.).
  * `2` – Kritischer Ressourcenfehler (z.B. Modellgewichte fehlen/korrupt).
  * `3` – Linting/Typing-Fehler (Codequalität nicht erfüllt).
  * `4` – Test fehlgeschlagen.
  * `5` – Fehler beim Starten der GUI.
  * (Weitere Codes nach Bedarf für unterschiedliche Kategorien von Abbruchursachen.)

  Diese Exit-Codes erlauben einer übergeordneten Pipeline oder dem Entwickler sofort zu erkennen, woran es gescheitert ist.
* **Schrittweises Vorgehen:** Der Agent arbeitet Schritt für Schritt wie nummeriert. Jeder Schritt muss erfolgreich abgeschlossen sein, bevor zum nächsten übergegangen wird. Bei Misserfolg eines Schritts greift entweder eine definierte Recovery oder es erfolgt der Abbruch. Dieses **kleinschrittige Vorgehen** verhindert, dass Fehler cascaden und erleichtert das Debugging.
* **Backoff-Strategie:** Wo angegeben (z.B. beim Recovery-Retry), setzt der Agent kurze Wartezeiten ein, die sich exponentiell erhöhen können, um z.B. bei temporären Systemlast-Spitzen dem System Zeit zu geben. Der Agent verwaltet diese Wartezeiten intern und fährt danach automatisch fort.
* **Keine externen Netzwerkausgänge:** Der Agent weiß, dass er in einer sandboxed Umgebung ohne Internet läuft. Jeder Versuch, externe URLs zu laden (sei es in Tests oder zur Laufzeit), würde fehlschlagen. Daher werden solche Aktionen entweder vorher abgefangen oder gar nicht erst unternommen. Download-Aufgaben müssen im Setup (vor Start) erfolgen.
* **Determinismus:** Der Agent strebt deterministisches Verhalten an. Zufallsfaktoren (z.B. Shuffle von Daten) werden, falls relevant, mit festem Seed versehen, damit Durchläufe reproduzierbar sind. Das betrifft v.a. Tests und eventuell Modelldeterminismus (bei neuronalen Netzen ggf. `torch.manual_seed(...)` setzen).
* **Verlassen des Prozesses:** Nach Abschluss aller Schritte (oder bei fatalem Fehler) beendet der Agent den Prozess kontrolliert. Offene Dateien werden geschlossen, Resourcen (z.B. GPU-Speicher) freigegeben, und es werden **keine Hintergrundprozesse** zurückgelassen. Sollte der Agent Teil eines übergeordneten Workflows sein (z.B. CI), signalisiert der Exit-Code den Status klar.

## Formatierungs- und PR-Richtlinien

Neben dem reinen Ausführen des Codes sind auch **Beitragsrichtlinien** für menschliche und AI-Entwickler definiert, um eine konsistente Codebasis und saubere Versionsverwaltung sicherzustellen:

* **Code-Formatierung:** Vor jedem Commit müssen Entwickler (und entsprechend der Codex-Agent) sicherstellen, dass `black` und `flake8` **ohne Befund** durchlaufen. D.h., der Code ist formatiert und lint-frei. Diese Schritte hat der Agent oben automatisiert (Schritte 18-24). Als Erinnerung: **keine manuellen Formatierungsänderungen** im Code-Stil, es gilt immer das von Black vorgegebene Format.
* **Commit-Nachrichten:** Verwende das **Conventional Commits** Schema für Git-Commit-Nachrichten. Zum Beispiel:

  * `feat: ...` für neue Features,
  * `fix: ...` für Bugfixes,
  * `docs: ...` für Dokumentationsänderungen (wie an dieser AGENTS.md),
  * etc.
    Der Agent beim Erstellen von Commits wird diesem Schema folgen, um konsistente Versionshistorie zu gewährleisten.
* **Pull Request Größe:** Änderungen sollen in **überschaubaren Pull Requests** eingereicht werden. Ein PR sollte idealerweise eine zusammenhängende Änderung beinhalten (z.B. ein Bugfix oder ein Feature) und nicht unzählige unabhängige Änderungen auf einmal. Der Agent versucht, größere Aufgaben in aufeinanderfolgende kleinere PRs zu unterteilen, wenn er viele Änderungen vornimmt.
* **PR-Prüfliste:** Bevor ein PR gestellt wird, stellt der Agent/Entwickler sicher:

  * Alle **Tests laufen grün** (`pytest` ohne Fehler).
  * **Linting/Typecheck bestanden** (flake8, mypy ohne Befund).
  * **Coverage** mindestens auf dem definierten Niveau (oder es wird begründet, warum nicht).
  * Keine sensiblen Daten im Diff (der Agent achtet z.B. darauf, keine Secrets oder großen Binärdateien einzuchecken).
  * Die **Dokumentation** ist aktualisiert, falls nötig (README, AGENTS.md, Docstrings).
* **Review-Prozess:** Pull Requests werden von mindestens einer Person (Reviewer) geprüft. Der Codex-Agent kann Code vorschlagen, aber ein menschlicher Entwickler sollte das letzte Wort haben. Der Agent markiert in seiner PR-Beschreibung gegebenenfalls Bereiche, bei denen Unsicherheiten bestehen oder die besonders begutachtet werden sollten.
* **PR-Beschreibung:** Jede Pull Request soll eine klare Beschreibung enthalten, *was* geändert wurde und *warum*. Bei Bugfixes sollte referenziert werden, welches Issue oder welcher Bug damit behoben ist. Der Codex-Agent formuliert PR-Beschreibungen sachlich und präzise, vermeidet aber Fragen. Wenn der PR ein Issue adressiert, fügt der Agent am Ende z.B. "Closes #123" ein, um die Verknüpfung herzustellen.
* **Squash/Merge-Strategie:** Das Projekt bevorzugt, dass Commits in der Historie sauber bleiben. Falls ein PR viele kleine Fix-Commits enthält (z.B. durch mehrere Agent-Iterationen), kann beim Merge **squash** gewählt werden, um einen einzelnen Commit zu erzeugen. Der Agent selbst könnte seine Commits vorsortieren, aber letztlich obliegt dies dem Maintainer.

## Abschluss und Ergebnis-Reporting

Nachdem alle oben genannten Schritte erfolgreich durchlaufen wurden, erstellt der Agent einen **Abschlussbericht** und beendet den Lauf:

30. **Erstellung des finish\_report.md:** Der Agent fasst die Ergebnisse aller Aktionen in einer Markdown-Datei namens `finish_report.md` zusammen. Dieser Report enthält:

    * Eine Zusammenfassung der durchgeführten Schritte (Setup, Checks, Tests, Verarbeitung).
    * Die **Anzahl der verarbeiteten Bilder** und ggf. aufgetretene Fehler pro Modell (z.B. "10 Bilder verarbeitet, 2 mit Warnungen/Fehlern").
    * Wichtige Metriken: Testanzahl und bestandene Tests, Coverage-Wert, Linting-Ergebnis (z.B. "0 Flake8-Warnungen, 0 mypy-Fehler"), Formatierungsstatus.
    * Falls Fehler übersprungen wurden (Recovery-Continue Fälle), eine Liste der betroffenen Dateien und Modelle.
    * Zeitpunkt des Beginns und Endes der Ausführung (für Laufzeitanalyse).
    * Ggf. Hinweise für den Entwickler, was als Nächstes zu tun ist (z.B. bestimmte Tests ergänzen, bestimmte Fehler manuell analysieren).

    Der Report ist in Markdown formatiert, evtl. mit Tabellen oder Listen für Übersichtlichkeit. Beispiel einer Tabelle im Bericht:

    | Prüfschritt          | Ergebnis                                          |
    | -------------------- | ------------------------------------------------- |
    | Abhängigkeiten       | OK (alle installiert)                             |
    | Modellgewichte       | OK (3/3 verifiziert)                              |
    | Formatierung (Black) | OK (keine Änderungen notwendig)                   |
    | Linting (Flake8)     | OK (0 Fehler)                                     |
    | Typprüfung (mypy)    | OK (0 Fehler)                                     |
    | Tests (pytest)       | OK (alle 42 Tests bestanden)                      |
    | Coverage             | 92% (über Mindestwert)                            |
    | Streamlit-Start      | OK (keine Fehler)                                 |
    | Bildverarbeitung     | **WARNUNG** – 1 Bild fehlgeschlagen (siehe unten) |

    Darunter könnten dann Details zu Warnungen/Fehlern folgen, z.B. welches Bild mit welchem Modell Probleme hatte.
31. **Sauberer Abschluss:** Nachdem `finish_report.md` geschrieben ist, schließt der Agent alle Resourcen. Log-Dateien (z.B. `run.log`) werden flush-geschrieben, offene Files geschlossen. Der Agent gibt eine letzte Konsolenmeldung aus, dass der Prozess erfolgreich beendet wurde (oder mit Fehler, falls zutreffend).
32. **Exit-Code setzen:** Zum Schluss beendet sich der Prozess mit dem passenden Exit-Code:

    * `0` bei komplett erfolgreich durchlaufener Pipeline (auch wenn wenige nicht-kritische Fehler bei der Bildverarbeitung aufgetreten sein mögen – diese wurden ja geloggt und nicht als kritisch eingestuft).
    * Einen der oben definierten Fehlercodes, falls irgendwo abgebrochen wurde. Im Falle eines regulären Abschlusses trotz z.B. übersprungener Bilder bleibt es bei `0`, da der Batch insgesamt nicht fehlgeschlagen ist.

    Damit endet die Ausführung. Ein nachgelagerter CI-Prozess oder der Entwickler kann anhand des Exit-Codes und des `finish_report.md` nachvollziehen, ob alles geklappt hat.

---

**Hinweis:** Diese AGENTS.md dient als verbindlicher Ablaufplan und Regelwerk für das Edge-Batch-Runner Projekt. Sie sollte im Projekt-Root versioniert werden, damit sowohl menschliche Entwickler als auch der Codex-Agent sie vorfinden und befolgen können. Durch die klare Struktur und detaillierten Mikro-Schritte wird sichergestellt, dass neue Änderungen konsistent erfolgen und der Projektstandard jederzeit gewahrt bleibt. Viel Erfolg bei der Umsetzung!
