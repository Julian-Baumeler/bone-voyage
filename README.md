# Bone Voyage 🦴✈️

> *From CT to finite-element models — no passport required, no Qt 5.5 either.*

**Bone Voyage** is a local-first web app: drop a CT, auto-split bright anatomy into **Bone 1, Bone 2, …**, toggle them on a 3D view, and build FE-ready meshes with material mapping.

It is a modern, macOS-friendly successor spirit to [MITK-GEM](https://github.com/araex/mitk-gem) (2017) — without the SuperBuild archaeology.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pip install PyMaxflow   # optional, for GraphCut

opengem serve
# → http://127.0.0.1:8742
```

Data stays on your machine (`~/.opengem/projects/`). Research use only — not for diagnosis or treatment.

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
  web/      # static UI (HTML/CSS/JS + Three.js)
```

## License

GPL-3.0-or-later (see `pyproject.toml`). Inspired by MITK-GEM (ZHAW / Pauchard et al.); this is a clean reimplementation, not a fork.

## Name

Because every CT is a small **voyage** into dense tissue — and every FE mesh is a return ticket.
