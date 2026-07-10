"""OpenGEM FastAPI — simple session UI, no project management."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from opengem import __version__
from opengem.api.jobs import get_job, start_auto_split_job, start_generate_job
from opengem.api.session import Session
from opengem.api.workspace import Workspace
from opengem.core.image_io import read_image, to_numpy, write_image

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
ws = Workspace()
session = Session(ws)

app = FastAPI(
    title="OpenGEM",
    description="CT → FE model (local web app)",
    version=__version__,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "version": __version__, "product": "OpenGEM"}


@app.get("/api/session")
def get_session() -> dict:
    sid = session.ensure()
    m = ws.get(sid)
    return {
        "id": sid,
        "files": m.files,
        "status": m.status,
        "last_error": m.last_error,
    }


@app.post("/api/session/reset")
def reset_session() -> dict:
    sid = session.reset()
    return {"id": sid, "files": {}, "status": {}}


@app.post("/api/session/upload-ct")
async def upload_ct(file: UploadFile = File(...)) -> dict:
    """Upload CT into the current session (replaces previous CT)."""
    sid = session.reset()  # fresh session each new CT
    suffix = Path(file.filename or "ct.nrrd").suffix or ".nrrd"
    # handle .nii.gz
    name = file.filename or "ct.nrrd"
    if name.lower().endswith(".nii.gz"):
        dest_name = "ct.nii.gz"
    else:
        dest_name = f"ct{suffix}"
    dest = ws.uploads(sid) / dest_name
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    rel = f"uploads/{dest_name}"
    ws.set_file(sid, "ct", rel)

    img = read_image(dest)
    arr, spacing, origin = to_numpy(img)
    preview = ws.outputs(sid) / "ct_preview.nii"
    write_image(preview, img)
    ws.set_file(sid, "ct_preview", "outputs/ct_preview.nii")

    # Auto-detect separate bones in background
    split_job = start_auto_split_job(session, hu_min=200.0, min_voxels=1000, open_radius=1)

    return {
        "ok": True,
        "session": sid,
        "filename": file.filename,
        "shape": list(arr.shape),
        "spacing": list(spacing),
        "origin": list(origin),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "split_job_id": split_job,
    }


@app.get("/api/session/ct-volume")
def ct_volume_raw():
    sid = session.ensure()
    try:
        ct_path = ws.resolve(sid, "ct")
    except FileNotFoundError as e:
        raise HTTPException(404, "No CT — drop a file first") from e
    img = read_image(ct_path)
    arr, spacing, origin = to_numpy(img)
    arr = np.ascontiguousarray(arr.astype(np.float32))
    z, y, x = arr.shape
    return Response(
        content=arr.tobytes(order="C"),
        media_type="application/octet-stream",
        headers={
            "X-Shape": f"{z},{y},{x}",
            "X-Spacing": ",".join(str(s) for s in spacing),
            "X-Origin": ",".join(str(o) for o in origin),
            "X-Dtype": "float32",
            "Access-Control-Expose-Headers": "X-Shape,X-Spacing,X-Origin,X-Dtype",
        },
    )


@app.post("/api/session/generate")
async def generate(
    paint: UploadFile | None = File(None),
    shape_z: int = Form(0),
    shape_y: int = Form(0),
    shape_x: int = Form(0),
    bone_hu_min: float = Form(200.0),
    cell_size: float = Form(4.0),
    quadratic: bool = Form(True),
    grow_mode: str = Form("local"),
    grow_radius_mm: float = Form(8.0),
    selected_bones: str = Form(""),  # comma-separated ids e.g. "1,3"
):
    """Start generate job (returns immediately). Poll /api/session/jobs/{id} for progress."""
    sid = session.ensure()
    if "ct" not in ws.get(sid).files:
        raise HTTPException(400, "Drop a CT first")

    paint_arr = None
    if paint is not None and shape_z > 0:
        raw = await paint.read()
        n = shape_z * shape_y * shape_x
        if len(raw) == n:
            paint_arr = np.frombuffer(raw, dtype=np.uint8).reshape((shape_z, shape_y, shape_x)).copy()
        elif len(raw) > 0:
            return {
                "ok": False,
                "error": f"Paint buffer size {len(raw)} ≠ expected {n}. Reload CT.",
                "log": [],
            }

    bone_ids: list[int] | None = None
    if selected_bones.strip():
        bone_ids = [int(x) for x in selected_bones.split(",") if x.strip().isdigit()]

    mode = grow_mode if grow_mode in ("local", "component", "paint_only") else "local"
    job_id = start_generate_job(
        session,
        paint=paint_arr,
        hu_min=bone_hu_min,
        cell_size=cell_size,
        quadratic=quadratic,
        grow_mode=mode,
        grow_radius_mm=float(grow_radius_mm),
        selected_bone_ids=bone_ids,
    )
    return {"ok": True, "job_id": job_id, "session": sid}


@app.get("/api/session/bones")
def list_bones():
    sid = session.ensure()
    p = ws.path(sid) / "outputs" / "bones.json"
    if not p.exists():
        return {"n_bones": 0, "bones": [], "ready": False}
    data = json.loads(p.read_text())
    data["ready"] = True
    # normalize preview URLs
    for b in data.get("bones", []):
        bid = b.get("id")
        if bid and (ws.path(sid) / "outputs" / "bones" / f"bone_{bid}_preview.json").exists():
            b["preview_url"] = f"/api/session/bones/{bid}/preview"
        else:
            b["preview_url"] = None
    return data


@app.get("/api/session/bones/{bone_id}/preview")
def bone_preview(bone_id: int):
    sid = session.ensure()
    p = ws.path(sid) / "outputs" / "bones" / f"bone_{bone_id}_preview.json"
    if not p.exists():
        raise HTTPException(404, f"No preview for Bone {bone_id}")
    return json.loads(p.read_text())


@app.get("/api/session/bones-labels")
def bones_labels_volume():
    """Labeled bone volume (uint16 ZYX) for 2D color overlay — same IDs as Bone 1..N."""
    sid = session.ensure()
    p = ws.path(sid) / "outputs" / "bones_labeled.nrrd"
    if not p.exists():
        raise HTTPException(404, "No bone labels yet — wait for auto-detect")
    img = read_image(p)
    arr, spacing, origin = to_numpy(img)
    arr = np.ascontiguousarray(arr.astype(np.uint16))
    z, y, x = arr.shape
    return Response(
        content=arr.tobytes(order="C"),
        media_type="application/octet-stream",
        headers={
            "X-Shape": f"{z},{y},{x}",
            "X-Spacing": ",".join(str(s) for s in spacing),
            "X-Dtype": "uint16",
            "Access-Control-Expose-Headers": "X-Shape,X-Spacing,X-Dtype",
        },
    )


@app.post("/api/session/bones/redetect")
async def redetect_bones(
    bone_hu_min: float = Form(200.0),
    min_voxels: int = Form(1000),
    open_radius: int = Form(1),
):
    sid = session.ensure()
    if "ct" not in ws.get(sid).files:
        raise HTTPException(400, "Drop a CT first")
    job_id = start_auto_split_job(
        session,
        hu_min=bone_hu_min,
        min_voxels=min_voxels,
        open_radius=open_radius,
    )
    return {"ok": True, "job_id": job_id}


@app.get("/api/session/jobs/{job_id}")
def job_status(job_id: str):
    j = get_job(job_id)
    if not j:
        raise HTTPException(404, "Unknown job")
    # Do NOT embed multi‑MB mesh JSON here — client fetches /api/session/preview
    if j.get("result"):
        res = dict(j["result"])
        res.pop("surface_preview", None)
        res["preview_url"] = "/api/session/preview"
        j = dict(j)
        j["result"] = res
    return j


@app.get("/api/session/preview")
def session_preview():
    """Lightweight surface mesh for Three.js (separate from job status).

    Rebuilds from STL/VTP if the JSON preview is missing (common after crashes).
    """
    sid = session.ensure()
    out = ws.path(sid) / "outputs"
    prev = out / "surface_preview.json"
    if prev.exists() and prev.stat().st_size > 100:
        try:
            return json.loads(prev.read_text())
        except Exception:
            pass

    # Rebuild from surface mesh file
    from opengem.core.surface import read_surface, surface_to_json, write_vtp

    stl = out / "surface.stl"
    vtp = out / "surface.vtp"
    path = vtp if vtp.exists() else stl if stl.exists() else None
    if path is None:
        raise HTTPException(
            404,
            "No surface yet. Run Generate first. (debug: no surface.stl / preview)",
        )
    try:
        poly = read_surface(path)
        preview = surface_to_json(poly, max_points=40_000)
        prev.write_text(json.dumps(preview))
        ws.set_file(sid, "surface_preview", "outputs/surface_preview.json")
        return preview
    except Exception as e:
        raise HTTPException(500, f"Failed to build preview from {path.name}: {e}") from e


@app.get("/api/session/debug")
def session_debug():
    """Human-readable last-run diagnostics for the UI debug panel."""
    sid = session.ensure()
    meta = ws.get(sid)
    out: dict[str, Any] = {
        "session": sid,
        "files": meta.files,
        "status": meta.status,
        "last_error": meta.last_error,
        "outputs": {},
    }
    out_dir = ws.path(sid) / "outputs"
    if out_dir.exists():
        for f in sorted(out_dir.iterdir()):
            if f.is_file():
                out["outputs"][f.name] = {
                    "bytes": f.stat().st_size,
                    "mb": round(f.stat().st_size / 1e6, 3),
                }
    # mask stats
    mask_path = out_dir / "mask_auto.nrrd"
    if mask_path.exists():
        try:
            import SimpleITK as sitk

            arr = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path)))
            out["mask_fg_voxels"] = int((arr > 0).sum())
            out["mask_shape"] = list(arr.shape)
            out["mask_fg_pct"] = round(100.0 * float((arr > 0).mean()), 4)
        except Exception as e:
            out["mask_error"] = str(e)
    prev = out_dir / "surface_preview.json"
    if prev.exists():
        try:
            d = json.loads(prev.read_text())
            out["preview"] = {
                "n_points": d.get("n_points"),
                "n_faces": d.get("n_faces"),
                "mb": round(prev.stat().st_size / 1e6, 3),
            }
        except Exception as e:
            out["preview_error"] = str(e)
    out["has_surface_file"] = (out_dir / "surface.stl").exists()
    out["has_volume_file"] = (out_dir / "volume.vtu").exists()
    out["has_preview_file"] = prev.exists()
    return out


@app.get("/api/session/files/{file_key:path}")
def session_file(file_key: str):
    sid = session.ensure()
    meta = ws.get(sid)
    if file_key in meta.files:
        path = ws.path(sid) / meta.files[file_key]
    else:
        path = ws.path(sid) / file_key
    try:
        path.resolve().relative_to(ws.path(sid).resolve())
    except ValueError:
        raise HTTPException(400, "Invalid path") from None
    if not path.is_file():
        raise HTTPException(404, f"File not found: {file_key}")
    return FileResponse(path, filename=path.name)


# Static frontend
if WEB_DIR.exists():
    # Disable caching so hard-refresh isn't required after UI fixes
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request as StarletteRequest
    from starlette.responses import Response as StarletteResponse

    class NoCacheAssets(BaseHTTPMiddleware):
        async def dispatch(self, request: StarletteRequest, call_next):
            response: StarletteResponse = await call_next(request)
            if request.url.path.startswith("/assets/") or request.url.path == "/":
                response.headers["Cache-Control"] = "no-store, max-age=0"
            return response

    app.add_middleware(NoCacheAssets)
    app.mount("/assets", StaticFiles(directory=WEB_DIR / "assets"), name="assets")

    @app.get("/")
    def index() -> HTMLResponse:
        return HTMLResponse(
            (WEB_DIR / "index.html").read_text(encoding="utf-8"),
            headers={"Cache-Control": "no-store, max-age=0"},
        )
else:

    @app.get("/")
    def index_missing() -> dict:
        return {"error": "Web UI missing", "expected": str(WEB_DIR)}
