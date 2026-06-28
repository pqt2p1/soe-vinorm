import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

from soe_vinorm.nsw_detector import get_nsw_bio_labels
from soe_vinorm.phobert_crf import PhoBertInputEncoder


class TokenLabelDataset:
    """Lightweight torch-compatible dataset for token-level NSW labels."""

    def __init__(self, examples, input_encoder: PhoBertInputEncoder, label2id):
        self.examples = examples
        self.input_encoder = input_encoder
        self.label2id = label2id

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        return self.input_encoder.encode(
            example["tokens"],
            labels=example["labels"],
            label2id=self.label2id,
        )


def load_jsonl_examples(path: Union[str, Path], valid_labels: Sequence[str] = None):
    """Load token-level JSONL examples and validate the fixed NSW label set."""
    valid_label_set = set(valid_labels or get_nsw_bio_labels())
    path = Path(path)
    examples = []
    errors = []

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                example = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{path}:{line_number}: invalid JSON: {exc.msg}")
                continue

            tokens = example.get("tokens")
            labels = example.get("labels")
            if not isinstance(tokens, list) or not all(
                isinstance(token, str) for token in tokens
            ):
                errors.append(
                    f"{path}:{line_number}: tokens must be a list of strings"
                )
                continue
            if not isinstance(labels, list) or not all(
                isinstance(label, str) for label in labels
            ):
                errors.append(
                    f"{path}:{line_number}: labels must be a list of strings"
                )
                continue
            if len(tokens) != len(labels):
                errors.append(
                    f"{path}:{line_number}: tokens and labels must have the same "
                    f"length (tokens={len(tokens)}, labels={len(labels)})"
                )
                continue
            invalid_labels = sorted(set(labels) - valid_label_set)
            if invalid_labels:
                errors.append(
                    f"{path}:{line_number}: unsupported labels: {invalid_labels}"
                )
                continue

            errors.extend(_validate_bio_labels(labels, path, line_number))
            if tokens:
                examples.append({"tokens": tokens, "labels": labels})

    if errors:
        details = "\n".join(f"- {error}" for error in errors)
        raise ValueError(f"{path}: found {len(errors)} validation error(s):\n{details}")

    if not examples:
        raise ValueError(f"{path}: no training examples found")

    return examples


def _validate_bio_labels(labels: List[str], path: Path, line_number: int):
    errors = []
    previous_entity = None
    previous_prefix = "O"

    for index, label in enumerate(labels):
        if label == "O":
            previous_entity = None
            previous_prefix = "O"
            continue

        try:
            prefix, entity = label.split("-", 1)
        except ValueError:
            errors.append(
                f"{path}:{line_number}: invalid BIO label at token {index}: {label}"
            )
            continue

        if prefix == "B":
            previous_entity = entity
            previous_prefix = prefix
            continue

        if prefix != "I":
            errors.append(
                f"{path}:{line_number}: invalid BIO label at token {index}: {label}"
            )
            continue

        if previous_prefix == "O" or previous_entity != entity:
            errors.append(
                f"{path}:{line_number}: invalid BIO transition at token {index}: "
                f"{label} cannot follow {labels[index - 1] if index else 'START'}"
            )
            continue

        previous_entity = entity
        previous_prefix = prefix

    return errors


def split_examples(
    examples: List[Dict[str, List[str]]],
    valid_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict[str, List[str]]], List[Dict[str, List[str]]]]:
    if not 0 < valid_ratio < 1:
        raise ValueError("valid_ratio must be between 0 and 1")

    shuffled = examples.copy()
    random.Random(seed).shuffle(shuffled)
    valid_size = max(1, int(len(shuffled) * valid_ratio))
    if valid_size >= len(shuffled):
        valid_size = len(shuffled) - 1
    if valid_size <= 0:
        raise ValueError("at least two examples are required when auto-splitting")

    return shuffled[valid_size:], shuffled[:valid_size]
