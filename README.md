# Bone Voyage 🦴✈️

> *From CT to finite-element models — no passport required, no Qt 5.5 either.*

**Bone Voyage** is a local-first web app: drop a CT, auto-split bright anatomy into **Bone 1, Bone 2, …**, toggle them on a 3D view, and build FE-ready meshes with material mapping.

It is a modern, macOS-friendly successor spirit to [MITK-GEM](https://github.com/araex/mitk-gem) (2017) — without the SuperBuild archaeology.

**Live UI (GitHub Pages):** [julian-baumeler.github.io/bone-voyage](https://julian-baumeler.github.io/bone-voyage/)

The hosted page is the real Three.js UI. Processing stays on your machine — start the engine below, keep this tab open, and the page talks to `http://127.0.0.1:8742`.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pip install PyMaxflow   # optional, for GraphCut

opengem serve
# → http://127.0.0.1:8742  (or open the GitHub Pages UI + this backend)
```

Data stays on your machine (`~/.opengem/projects/`). Research use only — not for diagnosis or treatment.

### Hosted UI + local engine

1. Open the [GitHub Pages app](https://julian-baumeler.github.io/bone-voyage/).
2. Run `opengem serve` in a clone of this repo.
3. Refresh the Pages tab — status should show version + “drop a CT…”.

Override the API URL with `?api=http://host:port` if needed.

## What it does

1. **Import CT** (NRRD / NIfTI / MHA)
2. **Auto bone split** — dual-threshold 3D components → Bone 1…N
3. **2D + 3D** — same colors on slices and meshes; checkboxes to hide/show
4. **Generate** — surface → volume mesh → material map → export (VTU, ASCII, Abaqus, Ansys, LS-DYNA)
5. Optional brush for fine ROIs

## Project layout

```
src/opengem/
  core/     # algorithms (bones, surface, volume, material, …)
  api/      # FastAPI + background jobs
  web/      # static UI (HTML/CSS/JS + Three.js) — source of truth
docs/       # GitHub Pages copy of the UI (+ PARITY.md)
```

After UI edits, sync Pages assets:

```bash
cp src/opengem/web/index.html docs/index.html
cp src/opengem/web/assets/* docs/assets/
```

## License

GPL-3.0-or-later (see `pyproject.toml`). Inspired by MITK-GEM (ZHAW / Pauchard et al.); this is a clean reimplementation, not a fork.

## Name

Because every CT is a small **voyage** into dense tissue — and every FE mesh is a return ticket.
