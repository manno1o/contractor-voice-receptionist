from evals.run import main


def test_eval_suite_passes_with_mock(tmp_path):
    assert main(["--out", str(tmp_path)]) == 0
    assert "Pass rate" in (tmp_path / "report.md").read_text()


def test_eval_harness_catches_regressions(tmp_path, monkeypatch):
    # Break safety routing: the harness must fail the run.
    import app.llm as llm
    monkeypatch.setattr(llm, "SAFETY", __import__("re").compile(r"$^"))
    assert main(["--out", str(tmp_path), "--only", "gas|co_alarm|smoke", "--min-pass-rate", "1.0"]) == 1
