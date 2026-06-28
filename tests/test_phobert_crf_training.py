import pytest

from soe_vinorm.nsw_detector import get_nsw_bio_labels
from soe_vinorm.training.phobert_crf_dataset import (
    load_jsonl_examples,
    split_examples,
)


def test_get_nsw_bio_labels_contains_expander_labels():
    labels = get_nsw_bio_labels()
    assert "O" in labels
    assert "B-NNUM" in labels
    assert "I-NNUM" in labels
    assert "B-LABB" in labels


def test_load_jsonl_examples(tmp_path):
    data_file = tmp_path / "train.jsonl"
    data_file.write_text(
        '{"tokens":["Năm","2021"],"labels":["O","B-NNUM"]}\n',
        encoding="utf-8",
    )

    examples = load_jsonl_examples(data_file)

    assert examples == [{"tokens": ["Năm", "2021"], "labels": ["O", "B-NNUM"]}]


def test_load_jsonl_examples_rejects_length_mismatch(tmp_path):
    data_file = tmp_path / "train.jsonl"
    data_file.write_text(
        '{"tokens":["Năm","2021"],"labels":["O"]}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="same length"):
        load_jsonl_examples(data_file)


def test_load_jsonl_examples_rejects_unknown_label(tmp_path):
    data_file = tmp_path / "train.jsonl"
    data_file.write_text(
        '{"tokens":["Năm"],"labels":["B-UNKNOWN"]}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported labels"):
        load_jsonl_examples(data_file)


def test_load_jsonl_examples_rejects_i_label_at_sentence_start(tmp_path):
    data_file = tmp_path / "train.jsonl"
    data_file.write_text(
        '{"tokens":["USD"],"labels":["I-MONEY"]}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid BIO transition"):
        load_jsonl_examples(data_file)


def test_load_jsonl_examples_rejects_i_label_after_o(tmp_path):
    data_file = tmp_path / "train.jsonl"
    data_file.write_text(
        '{"tokens":["giá","USD"],"labels":["O","I-MONEY"]}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid BIO transition"):
        load_jsonl_examples(data_file)


def test_load_jsonl_examples_rejects_i_label_after_different_entity(tmp_path):
    data_file = tmp_path / "train.jsonl"
    data_file.write_text(
        '{"tokens":["999","USD"],"labels":["B-NNUM","I-MONEY"]}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid BIO transition"):
        load_jsonl_examples(data_file)


def test_load_jsonl_examples_accepts_valid_i_label(tmp_path):
    data_file = tmp_path / "train.jsonl"
    data_file.write_text(
        '{"tokens":["999","USD"],"labels":["B-MONEY","I-MONEY"]}\n',
        encoding="utf-8",
    )

    examples = load_jsonl_examples(data_file)

    assert examples == [{"tokens": ["999", "USD"], "labels": ["B-MONEY", "I-MONEY"]}]


def test_load_jsonl_examples_reports_all_validation_errors(tmp_path):
    data_file = tmp_path / "train.jsonl"
    data_file.write_text(
        "\n".join(
            [
                '{"tokens":["Năm","2021"],"labels":["O"]}',
                '{"tokens":["Năm"],"labels":["B-UNKNOWN"]}',
                '{"tokens":["USD"],"labels":["I-MONEY"]}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc_info:
        load_jsonl_examples(data_file)

    message = str(exc_info.value)
    assert "found 3 validation error" in message
    assert "train.jsonl:1" in message
    assert "train.jsonl:2" in message
    assert "train.jsonl:3" in message
    assert "tokens and labels must have the same length" in message
    assert "unsupported labels" in message
    assert "invalid BIO transition" in message


def test_split_examples():
    examples = [
        {"tokens": [str(index)], "labels": ["O"]}
        for index in range(10)
    ]

    train_examples, valid_examples = split_examples(examples, valid_ratio=0.2, seed=42)

    assert len(train_examples) == 8
    assert len(valid_examples) == 2
