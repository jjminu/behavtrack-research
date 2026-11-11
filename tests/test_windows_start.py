import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(os.name != "nt", reason="Windows batch launcher")
def test_windows_scripts_use_crlf():
    root = Path(__file__).resolve().parents[1]
    for name in ("실행하기.cmd", "offline_setup.cmd", "ai_setup.cmd", "설치하기.cmd"):
        data = (root / name).read_bytes()
        assert data.count(b"\n") == data.count(b"\r\n"), name


@pytest.mark.skipif(os.name != "nt", reason="Windows batch launcher")
def test_launcher_runs_from_unicode_directory(tmp_path):
    root = Path(__file__).resolve().parents[1]
    folder = tmp_path / "새 폴더" / "BehavTrack_Windows"
    folder.mkdir(parents=True)
    shutil.copyfile(root / "실행하기.cmd", folder / "start.cmd")
    venv = folder / ".venv"
    (venv / "Scripts").mkdir(parents=True)
    site = venv / "Lib" / "site-packages"
    site.mkdir(parents=True)
    shutil.copyfile(sys.executable, venv / "Scripts" / "python.exe")
    shutil.copyfile(Path(sys.prefix) / "pyvenv.cfg", venv / "pyvenv.cfg")
    # Python 3.11 reads .pth using the Windows locale, unlike Python 3.14.
    # ASCII Python escapes exercise Korean paths without a locale-dependent fixture.
    (site / "project-test.pth").write_text("import sys; sys.path.extend(" + ascii([str(Path(sys.prefix) / "Lib" / "site-packages"),str(root)]) + ")\n", encoding="ascii")
    (folder / "windows_launcher.py").write_text('print("LAUNCHED_OK")\n', encoding="utf-8")
    result = subprocess.run(["cmd.exe", "/d", "/c", str(folder / "start.cmd")], cwd=tmp_path,
                            input=b"\n", capture_output=True, timeout=60)
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert b"Starting BehavTrack" in result.stdout and b"LAUNCHED_OK" in result.stdout
