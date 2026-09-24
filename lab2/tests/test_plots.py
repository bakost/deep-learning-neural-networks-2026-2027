"""Рисунки строятся без ошибок и сохраняются (проверка «на дым»)."""

from __future__ import annotations

import numpy as np
import pytest

from level_spacing import exact_pm1_spacing, run_study, sample_matrices, simulate_eigenvalues
from level_spacing.experiments import ks_by_ensemble_size
from level_spacing.plots import (
    plot_2x2_geometry,
    plot_discrete_ensemble,
    plot_histogram_grid,
    plot_ks_vs_size,
    plot_normal_mirror,
    plot_pair_choice,
    plot_residuals,
    plot_small_s,
)


@pytest.fixture(scope="module")
def goe():
    return {n: run_study("goe", n, 5000, seed=1) for n in (2, 4, 16)}


def _check(fig, tmp_path, name):
    path = tmp_path / f"{name}.png"
    fig.savefig(path, dpi=60)
    assert path.stat().st_size > 5000
    assert len(fig.axes) >= 1


def test_geometry(tmp_path):
    _check(plot_2x2_geometry(sample_matrices("goe", 2, 3000, rng=0)), tmp_path, "geometry")


def test_histogram_grid(goe, tmp_path):
    fig = plot_histogram_grid([[goe[2], goe[4]], [goe[4], goe[16]]], widths=(0.2, 0.1))
    assert len(fig.axes) == 4
    _check(fig, tmp_path, "grid")


def test_small_s(goe, tmp_path):
    _check(plot_small_s(list(goe.values())), tmp_path, "small")


def test_residuals(goe, tmp_path):
    _check(plot_residuals(list(goe.values())), tmp_path, "residuals")


def test_ks_vs_size(goe, tmp_path):
    data = {n: ks_by_ensemble_size(goe[n].raw, [300, 1000]) for n in goe}
    _check(plot_ks_vs_size(data, plateau=0.0066, noise=0.66), tmp_path, "ks")


@pytest.mark.parametrize("name, construction", [("pm1", "mirror"), ("pm1-sum", "sum")])
def test_discrete(name, construction, tmp_path):
    studies = [run_study(name, n, 3000, seed=2) for n in (2, 4, 16)]
    exact = {n: exact_pm1_spacing(n, construction) for n in (2, 4)}
    _check(plot_discrete_ensemble(studies, exact), tmp_path, name)


def test_pair_choice(tmp_path):
    ev = simulate_eigenvalues("goe", 16, 2000, rng=3)
    _check(plot_pair_choice(ev, radius=2 * np.sqrt(32)), tmp_path, "pairs")


def test_normal_mirror(goe, tmp_path):
    studies = [run_study("normal-mirror", n, 3000, seed=4) for n in (2, 16)]
    _check(plot_normal_mirror(studies, goe[2]), tmp_path, "mirror")
