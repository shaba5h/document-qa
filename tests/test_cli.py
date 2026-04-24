import os

from typer.testing import CliRunner

from document_qa.entrypoints.cli import app


runner = CliRunner()


def clear_dqa_environment(monkeypatch) -> None:
    for name in list(os.environ):
        if name.startswith("DQA_"):
            monkeypatch.delenv(name, raising=False)


def test_root_help_lists_local_cli_commands(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    clear_dqa_environment(monkeypatch)

    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "ingest" in result.output
    assert "ask" in result.output
    assert "retrieve" in result.output


def test_ingest_missing_path_fails_before_building_pipeline(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    clear_dqa_environment(monkeypatch)

    result = runner.invoke(app, ["ingest", "missing.pdf"])

    assert result.exit_code == 1
    assert "Path does not exist" in result.output


def test_ask_without_chat_model_config_fails_before_building_pipeline(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    clear_dqa_environment(monkeypatch)

    result = runner.invoke(app, ["ask", "What is indexed?"])

    assert result.exit_code == 1
    assert "DQA_CHATMODEL__MODEL_NAME is required" in result.output


def test_retrieve_rejects_non_positive_k(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    clear_dqa_environment(monkeypatch)

    result = runner.invoke(app, ["retrieve", "refund", "--k", "0"])

    assert result.exit_code == 1
    assert "--k must be greater than 0" in result.output
