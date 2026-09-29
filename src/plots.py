"""Shared figure-saving helper so every figure uses the same format."""

from matplotlib.figure import Figure

from src import config

FIGURE_DPI = 150


def save_figure(fig: Figure, name: str) -> None:
    """Save a figure to results/figures/<name>.png with tight layout at FIGURE_DPI."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / f"{name}.png", dpi=FIGURE_DPI)


def axis_label(column: str) -> str:
    """Readable axis label with units, e.g. 'fertilizer amount (kg/ha)'."""
    if column == config.TARGET:
        return f"Yield ({config.UNITS[column]})"
    unit = config.UNITS.get(column)
    name = column.replace("_", " ")
    return f"{name} ({unit})" if unit else name
