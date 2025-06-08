---

## 2 Feinleitfaden für geklonte Repos (`/Models/AGENTS.md`)

```markdown
# AGENTS.md  –  Unterordner  /Models

## Zweck
1. Verhindern, dass Codex versehentlich Upstream-Code ändert oder committet.
2. Garantieren, dass Repos **nur** per `git pull` aktualisiert werden, niemals
   mit “Hand-Edits”.

## Verhaltensregeln
- **Schreibschutz:** Jede Datei unter /Models gilt als _read-only_.  
  Änderungen → sofortiger Exit 41.
- **Update-Pfad:**  
  - Fehlt ein Repo → `git clone --depth 1 <url>`  
  - Existiert → `git -C <repo> pull --ff-only`
- **Patchen?** Nur erlaubt, wenn ein Hash-Mismatch einen _bekannten_
  Upstream-Bug betrifft. Patch muss vorher in
  `<repo>/edge_suite_local_patch.diff` existieren.
- **CI-Linting:** Repos werden von Black/isort ausgenommen.
