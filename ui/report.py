"""The downloadable prediction report: one tidy table, as CSV and as a one-page PDF.

Pure functions of their inputs (no Streamlit), so tests/test_report.py can check them.
"""

import io

import matplotlib

matplotlib.use("Agg")  # no display needed; must precede pyplot
import matplotlib.pyplot as plt
import pandas as pd

from src import config
from ui import labels
from ui.theme import mpl_style

REPORT_COLUMNS = ["section", "item", "value", "unit"]
TOP_CONTRIBUTORS = 5


def report_frame(
    inputs: dict,
    prediction: float,
    lo: float | None,
    hi: float | None,
    model_key: str,
    contributions: dict[str, float],
    timestamp: str,
    version: str,
    commit: str,
) -> pd.DataFrame:
    """Every row of the report: inputs (display names + units), prediction, 90% range,
    model, the 5 largest SHAP contributions, and when/what produced it."""
    rows = []
    for col in config.FEATURES:
        value = inputs[col]
        shown = f"{value:.2f}" if isinstance(value, float) else str(value)
        rows.append(("Input", labels.name(col), shown, labels.unit(col)))
    rows.append(("Prediction", "Predicted yield", f"{prediction:.2f}", "t/ha"))
    if lo is not None and hi is not None:
        rows.append(("Prediction", "90% range, lower", f"{lo:.2f}", "t/ha"))
        rows.append(("Prediction", "90% range, upper", f"{hi:.2f}", "t/ha"))
    rows.append(("Prediction", "Model", labels.model(model_key), ""))
    top = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)[:TOP_CONTRIBUTORS]
    for rank, (col, value) in enumerate(top, start=1):
        rows.append(("Top factors (SHAP)", f"{rank}. {labels.name(col)}", f"{value:+.3f}", "t/ha"))
    rows.append(("About", "Created", timestamp, ""))
    rows.append(("About", "App version", version, ""))
    rows.append(("About", "Code commit", commit, ""))
    return pd.DataFrame(rows, columns=REPORT_COLUMNS)


def report_csv(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def report_pdf(frame: pd.DataFrame) -> bytes:
    """A one-page A4 PDF with the same table (light theme: it is meant for paper)."""
    with plt.rc_context(mpl_style("light")):
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.07, 0.95, "Crop yield prediction report", fontsize=18, fontweight="bold")
        prediction = frame.loc[frame["item"] == "Predicted yield", "value"].iloc[0]
        fig.text(0.07, 0.92, f"Predicted yield: {prediction} t/ha", fontsize=12)
        ax = fig.add_axes([0.07, 0.05, 0.86, 0.84])
        ax.axis("off")
        cells = frame.assign(unit=frame["unit"].replace("", "–")).to_numpy().tolist()
        table = ax.table(
            cellText=cells,
            colLabels=["Section", "Item", "Value", "Unit"],
            colWidths=[0.24, 0.40, 0.22, 0.14],
            loc="upper left",
            cellLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.25)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="pdf")
        plt.close(fig)
    return buffer.getvalue()
