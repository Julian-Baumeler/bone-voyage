"""Background jobs with live progress for the UI."""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from opengem.api.session import Session, run_auto_pipeline
from opengem.core.bones import auto_split_bones
from opengem.core.image_io import read_image

_lock = threading.Lock()
_jobs: dict[str, dict[str, Any]] = {}


def _new_job() -> str:
    job_id = uuid.uuid4().hex[:12]
    with _lock:
        _jobs[job_id] = {
            "id": job_id,
            "state": "queued",
            "step": "queued",
            "percent": 0,
            "message": "Queued…",
            "log": [],
            "result": None,
            "error": None,
            "kind": "job",
            "started": time.time(),
            "updated": time.time(),
        }
    return job_id


def _set_progress(job_id: str, step: str, pct: int, msg: str) -> None:
    with _lock:
        j = _jobs[job_id]
        j["state"] = "running"
        j["step"] = step
        j["percent"] = int(max(0, min(100, pct)))
        j["message"] = msg
        j["log"].append(msg)
        j["updated"] = time.time()


def _finish(job_id: str, result: dict | None, error: str | None = None) -> None:
    with _lock:
        j = _jobs[job_id]
        j["result"] = result
        j["state"] = "error" if error else ("done" if (result or {}).get("ok", True) else "error")
        if error:
            j["state"] = "error"
            j["error"] = error
            j["message"] = error
            j["step"] = "error"
        else:
            j["error"] = None if (result or {}).get("ok", True) else (result or {}).get("error")
            j["message"] = (result or {}).get("error") or "Done"
            j["step"] = "done" if j["state"] == "done" else "error"
        j["percent"] = 100
        j["updated"] = time.time()


def start_auto_split_job(
    session: Session,
    *,
    hu_min: float = 200.0,
    min_voxels: int = 1000,
    open_radius: int = 1,
) -> str:
    job_id = _new_job()
    with _lock:
        _jobs[job_id]["kind"] = "auto_split"

    def worker() -> None:
        try:
            sid = session.ensure()
            ws = session.ws
            _set_progress(job_id, "load", 3, "Loading CT for bone analysis…")
            ct = read_image(ws.resolve(sid, "ct"))
            out_dir = ws.outputs(sid)

            def on_progress(step: str, pct: int, msg: str) -> None:
                _set_progress(job_id, step, pct, msg)

            result = auto_split_bones(
                ct,
                hu_min=hu_min,
                min_voxels=min_voxels,
                open_radius=open_radius,
                out_dir=out_dir,
                on_progress=on_progress,
            )
            # register files
            if result.labeled_path:
                # written as bones_labeled.nrrd inside out_dir
                ws.set_file(sid, "bones_labeled", "outputs/bones_labeled.nrrd")
            ws.set_file(sid, "bones", "outputs/bones.json")
            for b in result.bones:
                if b.preview_file:
                    ws.set_file(sid, f"bone_{b.id}_preview", b.preview_file)

            payload = {
                "ok": result.n_bones > 0,
                "kind": "auto_split",
                "n_bones": result.n_bones,
                "bones": [
                    {
                        "id": b.id,
                        "name": b.name,
                        "voxels": b.voxels,
                        "volume_mm3": b.volume_mm3,
                        "color": b.color,
                        "color_hex": b.color_hex,
                        "preview_url": f"/api/session/bones/{b.id}/preview" if b.preview_file else None,
                    }
                    for b in result.bones
                ],
                "debug": result.debug,
                "error": result.debug.get("error") if result.n_bones == 0 else None,
            }
            if result.n_bones == 0:
                _finish(job_id, payload, error=payload.get("error") or "No bones found")
            else:
                _finish(job_id, payload)
        except Exception as e:
            _finish(job_id, None, error=str(e))

    threading.Thread(target=worker, daemon=True, name=f"opengem-split-{job_id}").start()
    return job_id


def start_generate_job(
    session: Session,
    *,
    paint: np.ndarray | None,
    hu_min: float,
    cell_size: float,
    quadratic: bool,
    grow_mode: str = "local",
    grow_radius_mm: float = 8.0,
    selected_bone_ids: list[int] | None = None,
) -> str:
    job_id = _new_job()
    with _lock:
        _jobs[job_id]["kind"] = "generate"

    def worker() -> None:
        def on_progress(step: str, pct: int, msg: str) -> None:
            _set_progress(job_id, step, pct, msg)

        try:
            _set_progress(job_id, "load", 2, "Starting FE pipeline…")
            result = run_auto_pipeline(
                session,
                paint=paint,
                hu_min=hu_min,
                cell_size=cell_size,
                quadratic=quadratic,
                do_material=True,
                grow_mode=grow_mode,
                grow_radius_mm=grow_radius_mm,
                selected_bone_ids=selected_bone_ids,
                on_progress=on_progress,
            )
            result["session"] = session.ensure()
            result["kind"] = "generate"
            _finish(job_id, result, error=None if result.get("ok") else result.get("error"))
        except Exception as e:
            _finish(job_id, None, error=str(e))

    threading.Thread(target=worker, daemon=True, name=f"opengem-gen-{job_id}").start()
    return job_id


def get_job(job_id: str) -> dict[str, Any] | None:
    with _lock:
        j = _jobs.get(job_id)
        if not j:
            return None
        return {
            "id": j["id"],
            "kind": j.get("kind", "job"),
            "state": j["state"],
            "step": j["step"],
            "percent": j["percent"],
            "message": j["message"],
            "log": list(j["log"][-50:]),
            "error": j["error"],
            "result": j["result"],
            "elapsed_s": round(time.time() - j["started"], 1),
        }
