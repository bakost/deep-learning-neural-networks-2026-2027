"""Командная строка."""

from __future__ import annotations

import json

from lipo_ols.__main__ import EXIT_ERROR, EXIT_OK, main


def test_fit_table(capsys):
    assert main(["fit"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "R² = 0.8380" in out and "HBA2" in out and "2.0025" in out


def test_fit_json_subset(capsys):
    assert main(["fit", "--columns", "C", "PSA", "--json", "--level", "0.9"]) == EXIT_OK
    data = json.loads(capsys.readouterr().out)
    assert [r["label"] for r in data["coefficients"]] == ["const", "C", "PSA"]


def test_residuals(capsys):
    assert main(["residuals", "--sort"]) == EXIT_OK
    lines = capsys.readouterr().out.splitlines()
    assert lines[1].startswith("Brinzolamide") and len(lines) == 84


def test_diagnostics(capsys):
    assert main(["diagnostics"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "Бройш–Паган" in out and "Benzaldehyde" in out


def test_errors(capsys, tmp_path):
    assert main(["fit", "--columns", "NOPE"]) == EXIT_ERROR
    assert main(["--data", str(tmp_path / "no.csv"), "fit"]) == EXIT_ERROR
    assert main(["fit", "--level", "1.5"]) == EXIT_ERROR
    assert "Ошибка" in capsys.readouterr().err


def test_report(tmp_path, capsys):
    assert main(["report", "--output", str(tmp_path), "--repeats", "3"]) == EXIT_OK
    assert (tmp_path / "results" / "summary.json").exists()
    assert (tmp_path / "results" / "residuals.csv").read_text(encoding="utf-8").count("\n") == 83
    assert len(list((tmp_path / "figures").glob("*.png"))) == 7
    tables = (tmp_path / "results" / "tables.md").read_text(encoding="utf-8")
    assert "Таблица 5" in tables and "MLR-3" in tables
