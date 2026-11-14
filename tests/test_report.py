"""The prediction report: columns, rows and both file formats."""

import io

import pandas as pd

from src import config
from ui.report import REPORT_COLUMNS, report_csv, report_frame, report_pdf

INPUTS = {c: (1.5 if c in config.NUMERIC_FEATURES else "Rice") for c in config.FEATURES}
CONTRIB = {c: (i - 7) / 100 for i, c in enumerate(config.FEATURES)}


def make(lo=5.0, hi=7.0):
    return report_frame(
        INPUTS, 6.0, lo, hi, "polynomial", CONTRIB, "2026-01-01 00:00:00 UTC", "2.0.0", "abc1234"
    )


def test_columns_and_sections():
    frame = make()
    assert list(frame.columns) == REPORT_COLUMNS
    assert list(frame["section"].unique()) == ["Input", "Prediction", "Top factors (SHAP)", "About"]
    assert (frame["section"] == "Input").sum() == len(config.FEATURES)


def test_values_and_units():
    frame = make().set_index("item")
    assert frame.loc["Predicted yield", "value"] == "6.00"
    assert frame.loc["Predicted yield", "unit"] == "t/ha"
    assert frame.loc["90% range, lower", "value"] == "5.00"
    assert frame.loc["90% range, upper", "value"] == "7.00"
    assert frame.loc["Model", "value"] == "Polynomial + Ridge"
    assert frame.loc["Rainfall", "unit"] == "mm"
    assert frame.loc["Code commit", "value"] == "abc1234"
    assert not any("_" in item for item in frame.index)  # display names, not column names


def test_top_five_factors_by_absolute_size():
    top = make().query("section == 'Top factors (SHAP)'")
    assert len(top) == 5
    sizes = [abs(float(v)) for v in top["value"]]
    assert sizes == sorted(sizes, reverse=True)
    assert sizes[0] == max(abs(v) for v in CONTRIB.values())


def test_no_range_rows_without_an_interval():
    assert "90% range, lower" not in set(make(None, None)["item"])


def test_csv_round_trip_and_pdf_bytes():
    frame = make()
    back = pd.read_csv(io.BytesIO(report_csv(frame)), dtype=str, keep_default_na=False)
    pd.testing.assert_frame_equal(back, frame.astype(str))
    pdf = report_pdf(frame)
    assert pdf.startswith(b"%PDF") and len(pdf) > 1000
