"""On-disk project workspace for uploads and pipeline artifacts."""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path.home() / ".opengem" / "projects"


@dataclass
class ProjectMeta:
    id: str
    name: str
    created_at: str
    files: dict[str, str] = field(default_factory=dict)
    status: dict[str, Any] = field(default_factory=dict)
    last_error: str | None = None


class Workspace:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root or DEFAULT_ROOT)
        self.root.mkdir(parents=True, exist_ok=True)

    def create(self, name: str = "Untitled") -> ProjectMeta:
        pid = uuid.uuid4().hex[:12]
        path = self.root / pid
        path.mkdir(parents=True, exist_ok=True)
        (path / "uploads").mkdir()
        (path / "outputs").mkdir()
        meta = ProjectMeta(
            id=pid,
            name=name,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._save_meta(meta)
        return meta

    def list(self) -> list[ProjectMeta]:
        out = []
        for p in sorted(self.root.iterdir()):
            if (p / "meta.json").exists():
                out.append(self.get(p.name))
        return out

    def get(self, project_id: str) -> ProjectMeta:
        data = json.loads((self.path(project_id) / "meta.json").read_text())
        return ProjectMeta(**data)

    def path(self, project_id: str) -> Path:
        p = self.root / project_id
        if not p.exists():
            raise FileNotFoundError(project_id)
        return p

    def uploads(self, project_id: str) -> Path:
        return self.path(project_id) / "uploads"

    def outputs(self, project_id: str) -> Path:
        return self.path(project_id) / "outputs"

    def set_file(self, project_id: str, key: str, relative_path: str) -> ProjectMeta:
        meta = self.get(project_id)
        meta.files[key] = relative_path
        self._save_meta(meta)
        return meta

    def set_status(self, project_id: str, **kwargs: Any) -> ProjectMeta:
        meta = self.get(project_id)
        meta.status.update(kwargs)
        self._save_meta(meta)
        return meta

    def set_error(self, project_id: str, error: str | None) -> ProjectMeta:
        meta = self.get(project_id)
        meta.last_error = error
        self._save_meta(meta)
        return meta

    def resolve(self, project_id: str, key: str) -> Path:
        meta = self.get(project_id)
        if key not in meta.files:
            raise FileNotFoundError(f"No file registered as {key}")
        return self.path(project_id) / meta.files[key]

    def delete(self, project_id: str) -> None:
        shutil.rmtree(self.path(project_id), ignore_errors=True)

    def _save_meta(self, meta: ProjectMeta) -> None:
        path = self.root / meta.id / "meta.json"
        path.write_text(json.dumps(asdict(meta), indent=2))
