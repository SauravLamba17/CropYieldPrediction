"""Small HTML building blocks styled by ui/theme.py (cards, KPI grid, setup card).

Each returns or renders HTML only; the numbers always come from the caller.
"""

from dataclasses import dataclass
from functools import partial
from html import escape

import pandas as pd
import streamlit as st

from ui import labels
from ui.theme import tip


@dataclass
class Kpi:
    label: str
    value: str
    unit: str = ""
    note: str = ""
    accent: bool = False
    help_text: str | None = None
    help_left: bool = False


def kpi_row(kpis: list[Kpi]) -> None:
    """KPI cards on a CSS grid: 4 per row on wide screens, 2 then 1 as space shrinks."""
    cards = []
    for k in kpis:
        unit = f" <span>{k.unit}</span>" if k.unit else ""
        note = f'<p class="k-note">{k.note}</p>' if k.note else ""
        cards.append(
            f'<div class="kpi{" accent" if k.accent else ""}">'
            f'<p class="k-label">{k.label}{tip(k.help_text, k.help_left)}</p>'
            f'<p class="k-value">{k.value}{unit}</p>{note}</div>'
        )
    st.html(f'<div class="kpis">{"".join(cards)}</div>')


def headline_card(text: str, eyebrow: str = "In simple words") -> None:
    """The one- or two-sentence summary that opens every page (key numbers in <b>)."""
    st.html(
        f'<div class="headline"><p class="eyebrow">{eyebrow}</p><p class="h-text">{text}</p></div>'
    )


def setup_card(title: str, body: str, command: str | None = None) -> None:
    """A friendly card for something the user still has to run, with the exact command."""
    cmd = f'<span class="cmd">{escape(command)}</span>' if command else ""
    st.html(
        f'<div class="setup"><p class="s-title">{title}</p>'
        f'<p class="s-body">{body}</p>{cmd}</div>'
    )


def card_title(
    title: str, caption: str | None = None, inline: bool = False, help_text: str | None = None
) -> None:
    """A card heading (optional ⓘ) with an optional caption, below it or on the same line."""
    title_html = f"{title}{tip(help_text)}"
    if inline and caption:
        st.html(
            f'<p class="card-title">{title_html}<span class="inline-caption">{caption}</span></p>'
        )
        return
    st.html(
        f'<p class="card-title">{title_html}</p>'
        + (f'<p class="card-caption">{caption}</p>' if caption else "")
    )


def caption(text: str) -> None:
    st.html(f'<p class="card-caption">{text}</p>')


def page_title(text: str, caption_text: str | None = None) -> None:
    st.html(
        f'<p class="page-title">{text}</p>'
        + (f'<p class="card-caption">{caption_text}</p>' if caption_text else "")
    )


MINUS = "−"


def number(value: float, decimals: int) -> str:
    """Fixed decimals with a typographic minus; a value that rounds to zero shows as 0.00."""
    text = f"{value:.{decimals}f}"
    if float(text) == 0:
        text = f"{0:.{decimals}f}"
    return text.replace("-", MINUS)


def decimals_for(col: str) -> int:
    """Display precision: R² to 2 decimals, row counts as integers, everything else 3."""
    return 2 if "r2" in col else 0 if col == "n_test_rows" else 3


def table_view(frame: pd.DataFrame, simple: bool = False) -> "pd.io.formats.style.Styler":
    """Readable headers (ui.labels) and numbers; the best value per metric column in green.

    "Best" is judged at the displayed precision, so values that look tied are all
    highlighted. Spreads, deltas and training-set errors are context, never "best".
    """
    df = frame.copy()
    if "model" in df:
        df["model"] = df["model"].map(lambda m: labels.model(m, simple) if m in labels.MODELS else m)
    if "predictor" in df:
        df["predictor"] = df["predictor"].str.replace(" x ", " × ").str.capitalize()
    if "feature" in df:
        df["feature"] = df["feature"].map(lambda f: labels.label(f) if f in labels.FEATURES else f)
    for col in ("crop", "Crop"):
        if col in df:
            df[col] = df[col].map(labels.level)
    numeric = list(df.select_dtypes("number"))
    metric_cols = [
        c for c in numeric if not c.endswith("_std") and not c.startswith(("delta", "train_"))
    ]
    lower = [c for c in metric_cols if any(t in c for t in ("rmse", "mae", "median_abs"))]
    higher = [c for c in metric_cols if "r2" in c]
    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    for col in lower + higher:
        shown = df[col].round(decimals_for(col))
        target = shown.min() if col in lower else shown.max()
        styles.loc[shown == target, col] = "color: var(--cy-green); font-weight: 700"
    headers = {c: labels.column(c) for c in df.columns}
    formats = {headers[c]: partial(number, decimals=decimals_for(c)) for c in numeric}
    styler = df.rename(columns=headers).style.format(formats, na_rep="—")
    styler = styler.apply(lambda _: styles.rename(columns=headers), axis=None)
    return styler.hide(axis="index")


def show_table(frame: pd.DataFrame, simple: bool = False) -> None:
    st.table(table_view(frame, simple))


def tie_card(tie: pd.DataFrame | None, simple: bool = False) -> None:
    """The paired tie check (results/tables/tie_check.csv) for one dataset, as a small table.

    tie: rows of tie_check.csv already filtered to one dataset, or None if not built yet.
    """
    card_title(
        "Statistically distinguishable?",
        "Paired test on the 5 CV folds: RMSE of A − RMSE of B, fold by fold",
        help_text=(
            "For each fold, the difference in RMSE between the two models on the same rows. A paired "
            "t-test asks whether the mean difference is zero; 'yes' means p < 0.05. The exact Wilcoxon "
            "test cannot go below p = 0.0625 with 5 folds. Folds share training rows, so both tests are "
            "optimistic: a 'no' is the robust answer."
        ),
    )
    if tie is None:
        setup_card("tie_check.csv is missing", "It is written by the tie check (about 25 minutes):",
                   command="python -m src.tie_check")
        return
    rows = pd.DataFrame(
        {
            "A": tie["model_a"].map(lambda m: labels.model(m, simple)),
            "B": tie["model_b"].map(lambda m: labels.model(m, simple)),
            "Mean diff (A − B)": tie["mean_diff"].map(lambda v: f"{v:+.4f}"),
            "sd of diff": tie["sd_diff"].map(lambda v: f"{v:.4f}"),
            "t-test p": tie["t_test_p"].map(lambda v: f"{v:.3f}"),
            "Wilcoxon p": tie["wilcoxon_p"].map(lambda v: f"{v:.4f}"),
            "Distinguishable?": tie["distinguishable"].map({True: "yes", False: "no"}),
        }
    )
    st.table(rows.set_index("A"))
