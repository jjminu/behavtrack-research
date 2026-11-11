"""Session folder layout.

<session>/session.json      config snapshot + metadata (source, fps, animal id, group, overrides)
<session>/tracks_<id>.csv   per-frame track of each animal (id = arena name, or arena-individual)
<session>/reference.png     empty-arena image used for plots
<session>/raw.mp4           (live sessions) the unprocessed recording, reusable for pose estimation
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from pathlib import Path

from . import __version__

SESSION_FILE = "session.json"


def safe_name(text: str) -> str:
    return re.sub(r"[^\w\-.가-힣]+", "_", str(text)).strip("_") or "session"


def create_session_dir(root, name: str) -> Path:
    root = Path(root)
    base = safe_name(name)
    path = root / base
    n = 2
    while path.exists():
        path = root / f"{base}_{n}"
        n += 1
    path.mkdir(parents=True)
    return path


def tracks_path(session_dir, track_id: str) -> Path:
    return Path(session_dir) / f"tracks_{track_id}.csv"


def save_session(session_dir, cfg: dict, source: dict, tracks: dict, meta: dict | None = None) -> dict:
    meta = meta or {}
    data = {
        "behavtrack_version": __version__,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "animal_id": meta.get("animal_id", ""),
        "cage_id": meta.get("cage_id", ""),
        "treatment": meta.get("treatment", ""),
        "administration": meta.get("administration", "unspecified"),
        "animals": meta.get("animals", len(tracks)),
        "group": meta.get("group", ""),
        "notes": meta.get("notes", ""),
        "overrides": list(meta.get("overrides", [])),
        "source": source,
        "tracks": tracks,
        "config": cfg,
    }
    with open(Path(session_dir) / SESSION_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return data


def load_session(session_dir) -> dict:
    path = Path(session_dir) / SESSION_FILE
    if not path.exists():
        raise SystemExit(f"세션 폴더가 아닙니다 (session.json 없음): {session_dir}")
    from .integrity import read_json
    return read_json(path)


def find_sessions(paths) -> list[Path]:
    """Expand arguments that are session folders or folders containing sessions."""
    found = []
    for p in map(Path, paths):
        if (p / SESSION_FILE).exists():
            found.append(p)
        elif p.is_dir():
            found += sorted(q.parent for q in p.glob(f"*/{SESSION_FILE}"))
    return found
