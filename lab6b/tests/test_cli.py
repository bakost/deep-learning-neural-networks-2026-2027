"""Командная строка и генерация отчёта."""

from __future__ import annotations

import json

from sgd_lab.__main__ import EXIT_ERROR, EXIT_OK, main
from sgd_lab.report import ReportConfig, generate_report


def test_minimize(capsys, tmp_path):
    assert main(["minimize", "-m", "golden", "--plot", str(tmp_path / "f.png")]) == EXIT_OK
    out = capsys.readouterr().out
    assert "0.3803215" in out and (tmp_path / "f.png").exists()


def test_train_json(capsys):
    assert main(["train", "linear", "-e", "0.037", "-n", "5000", "-d", "1e-6", "--json"]) == EXIT_OK
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "converged" and abs(out["sse"] - 0.19987) < 1e-4


def test_train_text_minibatch_adam_plot(capsys, tmp_path):
    assert main(["train", "relu", "-m", "adam", "-b", "3", "-n", "50", "--l2", "0.1",
                 "--plot", str(tmp_path / "p.png")]) == EXIT_OK
    assert "Adam" in capsys.readouterr().out and (tmp_path / "p.png").exists()


def test_errors(capsys):
    assert main(["train", "linear", "-e", "-1"]) == EXIT_ERROR
    assert main(["train", "linear", "-b", "20"]) == EXIT_ERROR
    assert main(["minimize", "--tol", "0"]) == EXIT_ERROR
    assert "Ошибка" in capsys.readouterr().err


def test_report_tiny(tmp_path):
    cfg = ReportConfig(ref_starts=2, relu_seeds=4, seeds=2, grid_seeds=2, relu_epochs=30, relu_gd_iter=200,
                       epochs3=30, epochs4={"linear": 20, "quadratic": 30, "relu": 30}, l5_starts=2, loo_inits=2,
                       workers=2)
    R = generate_report(cfg, tmp_path, log=lambda *_: None)
    assert (tmp_path / "results" / "tables.md").read_text(encoding="utf-8").count("## Таблица") == 15
    assert len(list((tmp_path / "figures").glob("*.png"))) == 10
    assert json.loads((tmp_path / "results" / "summary.json").read_text(encoding="utf-8"))["task1"]
    assert R["task2"]["linear"]["status"] == "converged"
