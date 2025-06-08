# AGENTS.md – Unterordner `/Models`

## Zweck

Verhindern, dass Codex-Agenten unabsichtlich den Modell-Code ändern. Modelle sollen nur über Git-Operationen aktualisiert werden. Unbekannte Änderungen oder Commits sind verboten.

## Verhaltensregeln

* **Schreibschutz:** Alles unter `Models/` gilt als schreibgeschützt. Jede Veränderung ohne legitimen Grund führt zu sofortigem Exit-Code 41.
* **Update-Pfad:**

  * Fehlt ein Modell-Repo-Ordner oder ist leer: `git clone --depth 1 <Repo-URL> Models/<Name>`.
  * Ist der Ordner vorhanden: `git -C Models/<Name> pull --ff-only`.
* **Keine Hand-Edits:** Änderungen am Code der Modell-Repos sind **nur** erlaubt, wenn ein bekannter Upstream-Bug existiert und bereits ein Patch in `edge_suite_local_patch.diff` bereitliegt. Sonst sofort Fehler.
* **CI-Linting:** Model-Repos vom allgemeinen Linting ausnehmen (z.B. `flake8`, `black`) – nur unser eigener Code soll format-konform sein.
