"""Streamlit Cloud readiness: runtime import set, cache bounds, missing-table health card."""

import sys

# --- runtime dependency set ------------------------------------------------------------

IMPORT_TO_DIST = {"sklearn": "scikit-learn", "yaml": "pyyaml", "PIL": "pillow"}


def _imports(path):
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module


def app_third_party_imports() -> set[str]:
    """Top-level third-party imports reachable from app.py and app_pages/ (lazy ones too)."""
    from src import config

    todo = [config.ROOT / "app.py", *sorted((config.ROOT / "app_pages").glob("*.py"))]
    seen, found = set(), set()
    while todo:
        path = todo.pop()
        if path in seen:
            continue
        seen.add(path)
        for name in _imports(path):
            top = name.split(".")[0]
            if top in ("src", "ui", "app_pages"):
                target = config.ROOT / (name.replace(".", "/") + ".py")
                todo += [target] if target.exists() else []
            elif top not in sys.stdlib_module_names:
                found.add(top)
    return found


def requirement_names(file: str) -> set[str]:
    from src import config

    lines = (config.ROOT / file).read_text(encoding="utf-8").splitlines()
    return {
        line.split("==")[0].strip().lower().replace("_", "-")
        for line in lines
        if line and not line.startswith(("#", "-r"))
    }


def test_runtime_requirements_cover_every_app_import():
    runtime = requirement_names("requirements.txt")
    for top in app_third_party_imports():
        dist = IMPORT_TO_DIST.get(top, top).lower().replace("_", "-")
        if dist == "xgboost":
            assert {"xgboost", "xgboost-cpu"} <= runtime
        else:
            assert dist in runtime, f"{top} is imported by the app but not in requirements.txt"


def test_pipeline_only_packages_stay_out_of_the_app():
    imports = app_third_party_imports()
    assert not imports & {"statsmodels", "seaborn", "jupyter", "kaggle", "pytest"}
    assert not requirement_names("requirements.txt") & {"statsmodels", "seaborn"}
    assert {"statsmodels", "seaborn"} <= requirement_names("requirements-pipeline.txt")


def test_linux_uses_the_cpu_only_xgboost():
    from src import config

    text = (config.ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert 'xgboost-cpu==3.4.1; sys_platform == "linux"' in text
    assert 'xgboost==3.4.1; sys_platform != "linux"' in text


# --- cache bounds and the health check ---------------------------------------------------


def test_caches_keyed_on_inputs_are_bounded():
    """Every st.cache_data keyed on the field's inputs (unbounded key space) has max_entries."""
    import ast

    from src import config

    tree = ast.parse((config.ROOT / "ui" / "data.py").read_text(encoding="utf-8"))
    keyed = 0
    for fn in (n for n in tree.body if isinstance(n, ast.FunctionDef)):
        args = {a.arg for a in fn.args.args}
        for dec in fn.decorator_list:
            target = dec.func if isinstance(dec, ast.Call) else dec
            if ast.unparse(target) != "st.cache_data" or not args & {"inputs"}:
                continue
            keyed += 1
            kwargs = {k.arg for k in dec.keywords} if isinstance(dec, ast.Call) else set()
            assert "max_entries" in kwargs, f"{fn.name} is cached per input set without a bound"
    assert keyed >= 4  # prediction, explanation, heatmap, whatif_curve


def test_app_tables_list_covers_every_table_the_pages_read():
    import re

    from src import config
    from ui import data

    source = "\n".join(
        p.read_text(encoding="utf-8")
        for p in [*(config.ROOT / "app_pages").glob("*.py"), *(config.ROOT / "ui").glob("*.py")]
    )
    literal = set(re.findall(r'load_table\("([a-z0-9_]+)"', source))
    literal |= {f"{n}.json" for n in re.findall(r'load_json\("([a-z0-9_]+)"', source)}
    literal = {n if n.endswith(".json") else f"{n}.csv" for n in literal}
    india = (config.ROOT / "app_pages" / "india.py").read_text(encoding="utf-8")
    block = india[india.index("TABLES = {"):india.index("\n}\n", india.index("TABLES = {"))]
    picker = set(re.findall(r'^\s+"(india_[a-z_]+)",$', block, flags=re.MULTILINE))
    assert len(picker) >= 9
    assert literal | {f"{n}.csv" for n in picker} <= set(data.APP_TABLES)
    assert data.rebuild_commands(["india_cv.csv", "tie_check.csv", "conformal.json"]) == [
        "python -m src.india", "python -m src.tie_check", "python run.py build --fast"]


def test_missing_tables_show_a_health_card_not_a_traceback(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from src import config
    from ui import data

    monkeypatch.setattr(config, "TABLES_DIR", tmp_path)  # every results table missing
    st_cache_clear()
    at = AppTest.from_file(str(config.ROOT / "app.py"), default_timeout=180)
    at.run()
    assert not at.exception
    html = " ".join(str(h.proto.body) for h in at.get("html"))
    assert f"Health check: {len(data.APP_TABLES)} results tables are missing" in html
    for mode in ("Simple", "Show details"):
        at.segmented_control(key="mode").set_value(mode).run()
        for page in ("overview", "predict", "whatif", "models", "explain", "explorer", "india"):
            at.switch_page(f"app_pages/{page}.py").run()
            assert not at.exception, (mode, page, [e.value for e in at.exception])
    st_cache_clear()


def st_cache_clear():
    import streamlit as st

    st.cache_data.clear()
