"""Small durable, local-file provenance primitives (no database)."""
from __future__ import annotations
import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


def utcnow():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def read_json(path):
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result:
                raise ValueError(f"{path}: 중복 ID/필드 '{key}'가 있습니다.")
            result[key]=value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8-sig"),object_pairs_hook=unique)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    # Windows readers/antivirus can briefly hold a file without FILE_SHARE_DELETE.
    # Keep the durable old file intact while retrying the atomic replacement.
    for attempt in range(6):
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            if attempt==5:
                raise
            time.sleep(.01*(attempt+1))


def append_event(path, value):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_events(path):
    if not Path(path).exists():
        return []
    try:
        return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    except (ValueError, OSError) as exc:
        raise ValueError(f"이력 파일이 손상됐습니다. 원본을 보관하고 확인하세요: {path}") from exc


def contained_file(folder, name):
    folder = Path(folder).resolve()
    path = (folder / name).resolve()
    if path.parent != folder:
        raise ValueError("원본 파일 경로가 세션 폴더 밖을 가리킵니다.")
    return path


def csv_safe(value):
    return "'" + value if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")) else value
