"""One-command entry point: the Python equivalent of `npm run dev`.

    python run.py            dev (default): set up whatever is missing, then launch the app
    python run.py build      force a full rebuild: pipeline, India (if CSV present), notebooks
    python run.py check      ruff + pytest
    python run.py app        launch the app only

Add --no-browser to dev/app to start the server without opening a browser tab.

Standard library only, so it works before any package is installed. Every step runs
with the project's own interpreter (.venv), so nothing ever needs activating.
"Already done" is decided by the step's output files existing.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from src import config  # plain pathlib constants: importable before any install

ROOT = config.ROOT
VENV_DIR = ROOT / ".venv"
VENV_PYTHON = VENV_DIR / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
VENV_PYTHON_SHOWN = r".venv\Scripts\python.exe"  # how the re-run hints are printed
KAGGLE_DIR = Path.home() / ".kaggle"
KAGGLE_TOKEN_FILES = [KAGGLE_DIR / "access_token", KAGGLE_DIR / "kaggle.json"]  # existence only
COMPETITION = "crop-yield-prediction-challenge"
MENDELEY_URL = "https://data.mendeley.com/datasets/ncw2vbcgnk"
APP_PORT = 8501

# Files the app cannot start without; all are written by `python -m src.pipeline`.
PIPELINE_OUTPUTS = [
    config.MODELS_DIR / "best_overall.joblib",
    config.MODELS_DIR / "model_names.json",
    config.APP_METADATA_JSON,
    config.KAGGLE_TRAIN_PARQUET,
]
INDIA_OUTPUTS = [
    config.TABLES_DIR / "india_test.csv",
    config.MODELS_DIR / "india_best_overall.joblib",
]
NOTEBOOKS = ["01_eda", "02_diagnostics", "03_explain", "04_india"]

EXIT_FAILED = 1  # a step failed
EXIT_SETUP_NEEDED = 2  # the user must do something first (instructions printed)
EXIT_INTERRUPTED = 130  # Ctrl+C


class StepFailed(Exception):
    """A step's command returned a non-zero exit code."""

    def __init__(self, label: str, rerun: str, code: int) -> None:
        super().__init__(label)
        self.label, self.rerun, self.code = label, rerun, code


class SetupNeeded(Exception):
    """Something only the user can provide is missing; the message says what to do."""


def elapsed(seconds: float) -> str:
    """Format a duration like '4m 07s' or '12.3s'."""
    return f"{int(seconds // 60)}m {int(seconds % 60):02d}s" if seconds >= 60 else f"{seconds:.1f}s"


def child_env() -> dict[str, str]:
    """Environment for every subprocess: UTF-8 mode, as in all verified runs."""
    return {**os.environ, "PYTHONUTF8": "1"}


def header(number: int, total: int, title: str, note: str = "") -> None:
    """Print a step header such as '[3/5] Training models - first run only, about 10 min'."""
    print(f"\n[{number}/{total}] {title}" + (f" - {note}" if note else ""), flush=True)


def run_command(label: str, command: list, rerun: str) -> None:
    """Run one command from the project root; raise StepFailed on a non-zero exit."""
    start = time.perf_counter()
    code = subprocess.run(command, cwd=ROOT, env=child_env(), check=False).returncode
    if code != 0:
        raise StepFailed(label, rerun, code)
    print(f"      done in {elapsed(time.perf_counter() - start)}", flush=True)


def skipped(reason: str) -> None:
    """Report that a step's outputs already exist."""
    print(f"      already done ({reason}) - skipped", flush=True)


def missing(paths: list[Path]) -> list[Path]:
    """The paths from the list that do not exist."""
    return [p for p in paths if not p.exists()]


def rel(path: Path) -> str:
    """Path relative to the project root, for messages."""
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


# --- steps ----------------------------------------------------------------------


def step_environment(n: int, total: int) -> None:
    """Create .venv and install the exact locked versions, if .venv is missing."""
    header(n, total, "Python environment", "first run only, a few minutes")
    if VENV_PYTHON.exists():
        skipped(rel(VENV_PYTHON) + " exists")
        return
    run_command(
        "Create the virtual environment",
        [sys.executable, "-m", "venv", str(VENV_DIR)],
        rerun="python -m venv .venv",
    )
    run_command(
        "Install packages (requirements-lock.txt)",
        [str(VENV_PYTHON), "-m", "pip", "install", "-r", "requirements-lock.txt"],
        rerun=f"{VENV_PYTHON_SHOWN} -m pip install -r requirements-lock.txt",
    )


def step_kaggle_data(n: int, total: int) -> None:
    """Download and unzip the competition data if the training CSV is missing."""
    header(n, total, "Kaggle competition data", "first run only, under a minute")
    if config.KAGGLE_TRAIN_CSV.exists():
        skipped(rel(config.KAGGLE_TRAIN_CSV) + " exists")
        return
    if not any(p.exists() for p in KAGGLE_TOKEN_FILES):  # existence check only; never read
        raise SetupNeeded(
            "The Kaggle data is missing and no Kaggle API token was found.\n"
            f"  1. Sign in at https://www.kaggle.com/competitions/{COMPETITION} and click\n"
            "     'Join Competition' / accept the rules (downloads fail with 403 otherwise).\n"
            "  2. Kaggle -> Settings -> API -> create a new token.\n"
            f"  3. Save it as {KAGGLE_DIR / 'access_token'} (a file with no extension).\n"
            "  4. Run start.bat (or python run.py) again."
        )
    config.RAW_KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
    download = (
        f"{VENV_PYTHON_SHOWN} -m kaggle competitions download -c {COMPETITION} -p data/raw/kaggle"
    )
    try:
        run_command(
            "Download the Kaggle data",
            [
                str(VENV_PYTHON),
                "-m",
                "kaggle",
                "competitions",
                "download",
                "-c",
                COMPETITION,
                "-p",
                str(config.RAW_KAGGLE_DIR),
            ],
            rerun=download,
        )
    except StepFailed as failure:
        failure.rerun += (
            "\n  (401 = the token is missing or invalid; 403 = accept the competition rules"
            f" at https://www.kaggle.com/competitions/{COMPETITION})"
        )
        raise
    for archive in config.RAW_KAGGLE_DIR.glob("*.zip"):
        shutil.unpack_archive(archive, config.RAW_KAGGLE_DIR)
        print(f"      unzipped {archive.name}", flush=True)
    if not config.KAGGLE_TRAIN_CSV.exists():
        raise StepFailed("Unzip the Kaggle data", download, EXIT_FAILED)


def step_pipeline(n: int, total: int, force: bool = False) -> None:
    """Train and evaluate all Kaggle models if the app's model files are missing."""
    header(
        n, total, "Training models", "about 5-15 min" if force else "first run only, about 5-15 min"
    )
    absent = missing(PIPELINE_OUTPUTS)
    if not force and not absent:
        skipped("models/best_overall.joblib and the app's other inputs exist")
        return
    run_command(
        "Training models (python -m src.pipeline)",
        [str(VENV_PYTHON), "-m", "src.pipeline"],
        rerun=f"{VENV_PYTHON_SHOWN} -m src.pipeline",
    )


def step_india(n: int, total: int, force: bool = False) -> None:
    """Run the India validation if its CSV is present and its results are missing."""
    header(
        n,
        total,
        "India real-data validation",
        "about 15-25 min" if force else "first run only, about 15-25 min",
    )
    if not config.INDIA_CSV.exists():
        print(
            f"      India CSV not found - download it from {MENDELEY_URL} and save it as "
            f"{rel(config.INDIA_CSV)} (skipped; the app still works).",
            flush=True,
        )
        return
    if not force and not missing(INDIA_OUTPUTS):
        skipped("results/tables/india_test.csv exists")
        return
    run_command(
        "India validation (python -m src.india)",
        [str(VENV_PYTHON), "-m", "src.india"],
        rerun=f"{VENV_PYTHON_SHOWN} -m src.india",
    )


def step_notebooks(n: int, total: int) -> None:
    """Execute all four notebooks top to bottom, in place."""
    header(n, total, "Executing the 4 notebooks", "about 2 min")
    paths = [f"notebooks/{name}.ipynb" for name in NOTEBOOKS]
    command = [
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--inplace",
        "--ExecutePreprocessor.timeout=1800",
        *paths,
    ]
    run_command(
        "Executing the notebooks",
        [str(VENV_PYTHON), *command],
        rerun=f"{VENV_PYTHON_SHOWN} " + " ".join(command),
    )


def step_app(n: int, total: int, open_browser: bool) -> None:
    """Launch the Streamlit app (blocks until the user stops it with Ctrl+C)."""
    header(n, total, "Launching the app", f"http://localhost:{APP_PORT} - press Ctrl+C to stop")
    if missing(PIPELINE_OUTPUTS):
        raise SetupNeeded(
            "The trained models are missing. Run `python run.py` (or start.bat) first."
        )
    command = [
        str(VENV_PYTHON),
        "-m",
        "streamlit",
        "run",
        "app.py",
        "--server.port",
        str(APP_PORT),
        "--server.showEmailPrompt",
        "false",  # the first-run prompt would hang a double-click
        "--browser.gatherUsageStats",
        "false",
    ]
    if not open_browser:
        command += ["--server.headless", "true"]
    shown = f"{VENV_PYTHON_SHOWN} -m streamlit run app.py"
    code = subprocess.run(command, cwd=ROOT, env=child_env(), check=False).returncode
    if code != 0:
        raise StepFailed("Launching the app", shown, code)


def require_venv() -> None:
    """Commands other than dev/build need the environment to exist already."""
    if not VENV_PYTHON.exists():
        raise SetupNeeded("The Python environment (.venv) is missing. Run `python run.py` first.")


# --- commands -------------------------------------------------------------------


def cmd_dev(args: argparse.Namespace) -> None:
    """Set up whatever is missing, then launch the app."""
    total = 5
    step_environment(1, total)
    step_kaggle_data(2, total)
    step_pipeline(3, total)
    step_india(4, total)
    step_app(5, total, open_browser=not args.no_browser)


def cmd_build(args: argparse.Namespace) -> None:
    """Force a full rebuild of every result, then execute all notebooks."""
    total = 5
    step_environment(1, total)
    step_kaggle_data(2, total)
    step_pipeline(3, total, force=True)
    step_india(4, total, force=True)
    step_notebooks(5, total)
    print("\nFull rebuild complete. Start the app with: python run.py app")


def cmd_check(args: argparse.Namespace) -> None:
    """Lint and test."""
    require_venv()
    header(1, 2, "Lint (ruff)")
    run_command(
        "Lint",
        [str(VENV_PYTHON), "-m", "ruff", "check", "src", "tests", "ui", "app.py", "run.py"],
        rerun=f"{VENV_PYTHON_SHOWN} -m ruff check src tests ui app.py run.py",
    )
    header(2, 2, "Tests (pytest)")
    run_command(
        "Tests", [str(VENV_PYTHON), "-m", "pytest", "-q"], rerun=f"{VENV_PYTHON_SHOWN} -m pytest -q"
    )
    print("\nAll checks passed.")


def cmd_app(args: argparse.Namespace) -> None:
    """Launch the app only."""
    require_venv()
    step_app(1, 1, open_browser=not args.no_browser)


def main() -> int:
    """Parse the command, run it, and turn failures into clear messages and exit codes."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "command", nargs="?", default="dev", choices=["dev", "build", "check", "app"]
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="do not open a browser tab (dev/app)"
    )
    args = parser.parse_args()
    commands = {"dev": cmd_dev, "build": cmd_build, "check": cmd_check, "app": cmd_app}
    start = time.perf_counter()
    try:
        commands[args.command](args)
    except StepFailed as failure:
        print(f"\nFAILED: '{failure.label}' (exit code {failure.code}). Nothing after it was run.")
        print(f"Re-run just this step with:\n  {failure.rerun}")
        return EXIT_FAILED
    except SetupNeeded as need:
        print(f"\nSETUP NEEDED: {need}")
        return EXIT_SETUP_NEEDED
    except KeyboardInterrupt:
        print("\nStopped (Ctrl+C).")
        return EXIT_INTERRUPTED
    print(f"\nFinished in {elapsed(time.perf_counter() - start)}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
