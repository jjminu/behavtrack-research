import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(os.name != "nt", reason="Windows batch installer")
def test_installer_rejects_copied_or_incomplete_virtualenv(tmp_path):
    source = Path(__file__).resolve().parents[1] / "설치하기.cmd"
    installer = tmp_path / "install.cmd"
    shutil.copyfile(source, installer)
    (tmp_path / ".venv").mkdir()
    result = subprocess.run(["cmd.exe", "/d", "/c", str(installer)], cwd=tmp_path,
                            input=b"\n", capture_output=True, timeout=20)
    assert result.returncode == 1
    assert b"existing .venv cannot run" in result.stdout
    assert (tmp_path / ".venv").is_dir()
