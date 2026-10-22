"""Guard against publishing the private folder. Skipped when git or a repository is not available."""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_private_folder_is_not_tracked():
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("not a git repository")
    files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    assert not [f for f in files if f.startswith("private/")]
    assert "private/" in (ROOT / ".gitignore").read_text().split()
