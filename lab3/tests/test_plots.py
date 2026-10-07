"""Рисунки строятся и сохраняются."""

from __future__ import annotations

from lipo_ols import backward_elimination, contribution_table, matched_pairs
from lipo_ols.plots import (
    plot_coefficients,
    plot_correlations,
    plot_data_overview,
    plot_elimination,
    plot_fit,
    plot_matched_pairs,
    plot_residuals,
    save,
)


def test_all_plots(tmp_path, data, fit):
    steps = backward_elimination(data)
    figs = {
        "overview": plot_data_overview(data),
        "corr": plot_correlations(data),
        "fit": plot_fit(fit, data),
        "res": plot_residuals(fit, data),
        "coef": plot_coefficients(contribution_table(fit, data)),
        "pairs": plot_matched_pairs(matched_pairs(fit, data)),
        "elim": plot_elimination(steps, [1.0] * len(steps), [0.9] * len(steps)),
    }
    for name, fig in figs.items():
        path = tmp_path / f"{name}.png"
        save(fig, path)
        assert path.stat().st_size > 20_000
