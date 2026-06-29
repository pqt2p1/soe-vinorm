from soe_vinorm.training import analyze_nsw_data


def test_analyze_file_renders_markdown_report(tmp_path):
    data_file = tmp_path / "train.jsonl"
    data_file.write_text(
        "\n".join(
            [
                '{"tokens":["cao","3","m"],"labels":["O","B-NNUM","B-MEA"]}',
                '{"tokens":["tốc","độ","5foobar"],"labels":["O","O","B-MEA"]}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    analysis = analyze_nsw_data.analyze_file(str(data_file), top_k=5)
    report = analyze_nsw_data.render_markdown(analysis)

    assert "# NSW Training Data Report" in report
    assert "- Valid examples: 2" in report
    assert "| `B-MEA` | 2 |" in report
    assert "`foobar` (1)" in report
    assert "[5foobar/B-MEA]" in report


def test_analyze_file_reports_validation_errors_without_crashing(tmp_path):
    data_file = tmp_path / "train.jsonl"
    data_file.write_text(
        "\n".join(
            [
                '{"tokens":["Năm","2021"],"labels":["O","B-NNUM"]}',
                '{"tokens":["Năm","2021"],"labels":["O"]}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    analysis = analyze_nsw_data.analyze_file(str(data_file))
    report = analyze_nsw_data.render_markdown(analysis)

    assert len(analysis["errors"]) == 1
    assert "- Valid examples: 1" in report
    assert "tokens and labels must have the same length" in report


def test_main_writes_output_file(tmp_path, monkeypatch):
    data_file = tmp_path / "train.jsonl"
    output_file = tmp_path / "report.md"
    data_file.write_text(
        '{"tokens":["Năm","2021"],"labels":["O","B-NNUM"]}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "analyze-nsw-data",
            "--input",
            str(data_file),
            "--output",
            str(output_file),
        ],
    )

    exit_code = analyze_nsw_data.main()

    assert exit_code == 0
    assert output_file.exists()
    assert "NSW Training Data Report" in output_file.read_text(encoding="utf-8")


def test_main_strict_returns_error_for_invalid_data(tmp_path, monkeypatch):
    data_file = tmp_path / "train.jsonl"
    data_file.write_text(
        '{"tokens":["Năm","2021"],"labels":["O"]}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        ["analyze-nsw-data", "--input", str(data_file), "--strict"],
    )

    assert analyze_nsw_data.main() == 1


def test_actionable_checklist_reports_unknown_measurement_units(tmp_path):
    data_file = tmp_path / "train.jsonl"
    data_file.write_text(
        '{"tokens":["cao","5foobar"],"labels":["O","B-MEA"]}\n',
        encoding="utf-8",
    )

    analysis = analyze_nsw_data.analyze_file(str(data_file))
    report = analyze_nsw_data.render_markdown(analysis)

    assert "## Actionable Coverage Checklist" in report
    assert "HIGH - MEA units missing from MEASUREMENT_UNITS_MAPPING" in report
    assert "`foobar` (1)" in report
    assert "line 1:" in report
    assert "[5foobar/B-MEA]" in report


def test_actionable_checklist_reports_token_label_conflicts(tmp_path):
    data_file = tmp_path / "train.jsonl"
    data_file.write_text(
        "\n".join(
            [
                '{"tokens":["mã","AI"],"labels":["O","B-LSEQ"]}',
                '{"tokens":["ứng","dụng","AI"],"labels":["O","O","B-LWRD"]}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    analysis = analyze_nsw_data.analyze_file(str(data_file))
    report = analyze_nsw_data.render_markdown(analysis)

    assert "HIGH - Same token appears with multiple entity labels" in report
    assert "[AI/B-LSEQ]" in report
    assert "[AI/B-LWRD]" in report


def test_actionable_checklist_reports_pattern_mismatches(tmp_path):
    data_file = tmp_path / "train.jsonl"
    data_file.write_text(
        '{"tokens":["ngày","12/08/2026"],"labels":["O","O"]}\n',
        encoding="utf-8",
    )

    analysis = analyze_nsw_data.analyze_file(str(data_file))
    report = analyze_nsw_data.render_markdown(analysis)

    assert "MEDIUM - Tokens matching NDAT patterns use other labels" in report
    assert "line 1:" in report
    assert "12/08/2026" in report
