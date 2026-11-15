"""run.py build --fast: argument parsing and which steps run."""

import sys

import pytest

import run


@pytest.fixture
def steps(monkeypatch):
    """Replace every build step with a recorder; returns the list of (step, force) calls."""
    calls = []
    for name in ("step_environment", "step_kaggle_data", "step_notebooks"):
        monkeypatch.setattr(run, name, lambda n, total, _name=name: calls.append((_name, n, total)))
    for name in ("step_pipeline", "step_india"):
        monkeypatch.setattr(
            run, name, lambda n, total, force=False, _name=name: calls.append((_name, n, total))
        )
    return calls


@pytest.mark.parametrize(
    "argv, expected",
    [
        (["build"], ["step_environment", "step_kaggle_data", "step_pipeline", "step_india",
                     "step_notebooks"]),
        (["build", "--fast"], ["step_environment", "step_kaggle_data", "step_pipeline",
                               "step_notebooks"]),
    ],
)
def test_build_step_selection(steps, monkeypatch, argv, expected):
    monkeypatch.setattr(sys, "argv", ["run.py", *argv])
    assert run.main() == 0
    assert [c[0] for c in steps] == expected
    total = len(expected)
    assert [c[1] for c in steps] == list(range(1, total + 1))  # numbered 1..total, no gaps
    assert {c[2] for c in steps} == {total}


def test_fast_flag_parses_and_defaults_off(monkeypatch):
    seen = {}
    monkeypatch.setattr(run, "cmd_build", lambda args: seen.update(fast=args.fast))
    monkeypatch.setattr(sys, "argv", ["run.py", "build"])
    run.main()
    assert seen == {"fast": False}
    monkeypatch.setattr(sys, "argv", ["run.py", "build", "--fast"])
    run.main()
    assert seen == {"fast": True}
