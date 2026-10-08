"""Командная строка и генерация отчёта."""

from __future__ import annotations

import json

from lipo_mlp.__main__ import EXIT_ERROR, EXIT_OK, main
from lipo_mlp.report import ReportConfig, generate_report


def test_train_json(capsys):
    assert main(["train", "-H", "4", "--max-iter", "500", "--json"]) == EXIT_OK
    out = json.loads(capsys.readouterr().out)
    assert out["hidden"] == 4 and out["params"] == 105 and out["iterations"] == 500


def test_train_text_with_schedule_and_plot(tmp_path, capsys):
    path = tmp_path / "c.png"
    assert main(["train", "-H", "2", "--max-iter", "300", "--schedule", "inverse", "--tau", "100",
                 "--plot", str(path)]) == EXIT_OK
    assert "ε₀/(1 + j/τ)" in capsys.readouterr().out and path.exists()


def test_sweep(capsys):
    assert main(["sweep", "-H", "2", "4", "--seeds", "1", "--max-iter", "200", "-j", "1"]) == EXIT_OK
    assert "105" in capsys.readouterr().out


def test_sklearn_command(capsys):
    assert main(["sklearn", "-H", "2", "--iterations", "100"]) == EXIT_OK
    assert "max |w_наш − w_sklearn|" in capsys.readouterr().out


def test_errors(capsys):
    assert main(["train", "--lr", "-1"]) == EXIT_ERROR
    assert main(["train", "-H", "0"]) == EXIT_ERROR
    assert "Ошибка" in capsys.readouterr().err


def test_report_tiny(tmp_path):
    cfg = ReportConfig(seeds=1, max_iter=300, cv_folds=5, cv_repeats=1, cv_max_iter=500, ea_evals=300,
                       bench_units=100, workers=2)
    generate_report(tmp_path, cfg, verbose=False)
    tables = (tmp_path / "results" / "tables.md").read_text(encoding="utf-8")
    assert "Таблица 1" in tables and "Таблица 6" in tables and "Таблица 8" in tables
    assert len(list((tmp_path / "figures").glob("*.png"))) == 7


def test_evolve(capsys):
    assert main(["evolve", "-H", "2", "--evals", "600"]) == EXIT_OK
    assert "вычислений J" in capsys.readouterr().out
    assert main(["evolve", "-H", "2", "--evals", "0"]) == EXIT_ERROR
