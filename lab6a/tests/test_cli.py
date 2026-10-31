import json

from transformer_lab.__main__ import main
from transformer_lab.report import generate_report


def test_cli_counts_and_softmax(capsys):
    assert main(["counts", "-D", "3", "-N", "4", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["qkv_total"] == 36
    assert main(["softmax", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["naive_J33"] == 0
    assert result["stable_J33"] > 0
    assert main(["equivariance", "--scaled"]) == 0
    assert "Sa(XP)" in capsys.readouterr().out


def test_cli_invalid_dimensions(capsys):
    assert main(["counts", "-N", "0"]) == 1
    assert main(["equivariance", "-D", "-1"]) == 1
    assert "Ошибка" in capsys.readouterr().err


def test_report_artifacts_and_determinism(tmp_path):
    result = generate_report(tmp_path)
    assert json.loads((tmp_path / "results/summary.json").read_text()) == result
    assert (tmp_path / "results/tables.md").read_text().count("## Таблица") == 5
    assert len((tmp_path / "results/jacobian.csv").read_text().splitlines()) == 6
    for filename in ("parameters.png", "equivariance.png", "softmax_jacobian.png"):
        assert (tmp_path / "figures" / filename).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    from transformer_lab.experiments import run_experiments

    assert result == run_experiments()
