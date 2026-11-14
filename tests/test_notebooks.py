"""Every notebook's setup cell must find the project root from the repo root and notebooks/."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src import config

NOTEBOOKS = sorted((config.ROOT / "notebooks").glob("*.ipynb"))


def setup_cell(path: Path) -> str:
    cells = json.loads(path.read_text(encoding="utf8"))["cells"]
    first_code = next(c for c in cells if c["cell_type"] == "code")
    return "".join(first_code["source"])


def test_there_are_four_notebooks():
    assert len(NOTEBOOKS) == 4


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: p.stem)
@pytest.mark.parametrize("start", [".", "notebooks"])
def test_setup_cell_finds_project_root(notebook, start):
    source = setup_cell(notebook)
    assert "ROOT = next(" in source, "the setup cell must be the first code cell"
    # Run the cell in a fresh interpreter started in `start` (not Colab, so no clone/install),
    # then check it found the root and that `src` imports from there.
    check = "\nimport src.config\nprint(ROOT.resolve())\nprint(src.config.ROOT)"
    result = subprocess.run(
        [sys.executable, "-c", source + check],
        cwd=config.ROOT / start,
        capture_output=True,
        text=True,
        check=True,
    )
    found, imported = result.stdout.splitlines()[:2]
    assert Path(found) == config.ROOT
    assert Path(imported) == config.ROOT


def test_no_notebook_uses_the_old_cwd_parent_bootstrap():
    for notebook in NOTEBOOKS:
        assert "Path.cwd().parent))" not in notebook.read_text(encoding="utf8"), notebook.name
