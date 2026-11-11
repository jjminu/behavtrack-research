import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(os.name != "nt", reason="Windows batch installer")
def test_offline_setup_preserves_old_environment_when_python_is_invalid(tmp_path):
    root = Path(__file__).resolve().parents[1]
    shutil.copyfile(root / "offline_setup.cmd", tmp_path / "offline_setup.cmd")
    offline = tmp_path / "offline_files"
    (offline / "wheelhouse").mkdir(parents=True)
    (offline / "python-3.11.9-amd64.exe").touch()
    (offline / "wheelhouse" / "dummy.whl").touch()
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "keep.txt").write_text("keep", encoding="utf-8")
    local = tmp_path / "local" / "Programs" / "Python" / "Python311"
    local.mkdir(parents=True)
    # A valid executable that is not Python fails validation without opening a Windows error dialog.
    shutil.copyfile(Path(os.environ["SystemRoot"]) / "System32" / "where.exe", local / "python.exe")
    env = dict(os.environ, LOCALAPPDATA=str(tmp_path / "local"), PROCESSOR_ARCHITECTURE="AMD64")
    result = subprocess.run(["cmd.exe", "/d", "/c", str(tmp_path / "offline_setup.cmd")],
                            cwd=tmp_path, env=env, capture_output=True, timeout=20)
    backups = list(tmp_path.glob(".venv_backup_*"))
    assert result.returncode == 1
    assert b"Python installation or tkinter validation failed" in result.stdout
    assert len(backups) == 1 and (backups[0] / "keep.txt").read_text() == "keep"


@pytest.mark.skipif(os.name != "nt", reason="Windows batch installer")
def test_ai_setup_reports_missing_model_without_installing(tmp_path):
    root = Path(__file__).resolve().parents[1]
    shutil.copyfile(root / "ai_setup.cmd", tmp_path / "ai_setup.cmd")
    result = subprocess.run(["cmd.exe", "/d", "/c", str(tmp_path / "ai_setup.cmd")],
                            cwd=tmp_path, capture_output=True, timeout=20)
    assert result.returncode == 1
    assert b"Bundled AI files are missing" in result.stdout
    assert not (tmp_path / ".ai_venv").exists()
