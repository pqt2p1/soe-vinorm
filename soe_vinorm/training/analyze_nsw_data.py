import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Tuple

from soe_vinorm.constants import MEASUREMENT_UNITS_MAPPING, MONEY_UNITS_MAPPING
from soe_vinorm.nsw_detector import get_nsw_bio_labels
from soe_vinorm.training.phobert_crf_dataset import _validate_bio_labels


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="analyze-nsw-data",
        description="Analyze token-level NSW JSONL training data.",
    )
    parser.add_argument("--input", required=True, help="Path to training JSONL file.")
    parser.add_argument("--output", help="Optional Markdown report output path.")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--examples-per-label", type=int, default=5)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with status 1 when validation errors are found.",
    )
    return parser


def analyze_file(path: str, top_k: int = 20, examples_per_label: int = 5):
    input_path = Path(path)
    valid_labels = set(get_nsw_bio_labels())
    examples = []
    errors = []
    label_counts = Counter()
    entity_token_counts = defaultdict(Counter)
    entity_span_counts = Counter()
    entity_span_lengths = defaultdict(list)
    entity_span_text_counts = defaultdict(Counter)
    examples_by_entity = defaultdict(list)
    measurement_units = Counter()
    money_units = Counter()
    token_label_counts = defaultdict(Counter)
    span_entity_counts = defaultdict(Counter)
    sentence_lengths = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            example, line_errors = _parse_and_validate_line(
                input_path, line_number, line, valid_labels
            )
            if line_errors:
                errors.extend(line_errors)
                continue

            tokens = example["tokens"]
            labels = example["labels"]
            examples.append({"line": line_number, "tokens": tokens, "labels": labels})
            sentence_lengths.append(len(tokens))

            for token, label in zip(tokens, labels):
                label_counts[label] += 1
                token_label_counts[token][label] += 1
                entity = label_to_entity(label)
                if entity != "O":
                    entity_token_counts[entity][token] += 1

            for span in iter_spans(tokens, labels):
                entity = span["entity"]
                entity_span_counts[entity] += 1
                entity_span_lengths[entity].append(len(span["tokens"]))
                span_text = " ".join(span["tokens"])
                entity_span_text_counts[entity][span_text] += 1
                span_entity_counts[span_text][entity] += 1
                if len(examples_by_entity[entity]) < examples_per_label:
                    examples_by_entity[entity].append(
                        format_highlighted_example(tokens, labels)
                    )
                if entity == "MEA":
                    for unit in extract_measurement_units(span["tokens"]):
                        measurement_units[unit] += 1
                if entity == "MONEY":
                    for unit in extract_money_units(span["tokens"]):
                        money_units[unit] += 1

    return {
        "path": str(input_path),
        "top_k": top_k,
        "examples_per_label": examples_per_label,
        "examples": examples,
        "errors": errors,
        "label_counts": label_counts,
        "entity_token_counts": entity_token_counts,
        "entity_span_counts": entity_span_counts,
        "entity_span_lengths": entity_span_lengths,
        "entity_span_text_counts": entity_span_text_counts,
        "examples_by_entity": examples_by_entity,
        "measurement_units": measurement_units,
        "money_units": money_units,
        "token_label_counts": token_label_counts,
        "span_entity_counts": span_entity_counts,
        "sentence_lengths": sentence_lengths,
    }


def _parse_and_validate_line(path: Path, line_number: int, line: str, valid_labels):
    errors = []
    try:
        example = json.loads(line)
    except json.JSONDecodeError as exc:
        return None, [f"{path}:{line_number}: invalid JSON: {exc.msg}"]

    tokens = example.get("tokens")
    labels = example.get("labels")
    if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
        return None, [f"{path}:{line_number}: tokens must be a list of strings"]
    if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
        return None, [f"{path}:{line_number}: labels must be a list of strings"]
    if len(tokens) != len(labels):
        return None, [
            f"{path}:{line_number}: tokens and labels must have the same "
            f"length (tokens={len(tokens)}, labels={len(labels)})"
        ]

    invalid_labels = sorted(set(labels) - valid_labels)
    if invalid_labels:
        errors.append(f"{path}:{line_number}: unsupported labels: {invalid_labels}")

    errors.extend(_validate_bio_labels(labels, path, line_number))
    if errors:
        return None, errors

    return {"tokens": tokens, "labels": labels}, []


def label_to_entity(label: str) -> str:
    return "O" if label == "O" else label.split("-", 1)[1]


def iter_spans(tokens: List[str], labels: List[str]):
    index = 0
    while index < len(tokens):
        label = labels[index]
        if label == "O":
            index += 1
            continue

        entity = label_to_entity(label)
        start = index
        span_tokens = [tokens[index]]
        index += 1
        while (
            index < len(tokens)
            and labels[index].startswith("I-")
            and label_to_entity(labels[index]) == entity
        ):
            span_tokens.append(tokens[index])
            index += 1

        yield {
            "entity": entity,
            "start": start,
            "end": index,
            "tokens": span_tokens,
        }


def extract_measurement_units(tokens: List[str]) -> List[str]:
    units = []
    for token in tokens:
        unit = strip_number_prefix(token)
        if unit and not looks_numeric(unit):
            units.append(unit)
    return units


def extract_money_units(tokens: List[str]) -> List[str]:
    units = []
    for token in tokens:
        without_leading_number = re.sub(r"^-?[0-9.,]+", "", token).strip()
        without_trailing_number = re.sub(r"[0-9.,]+$", "", token).strip()
        if without_leading_number != token and without_leading_number:
            units.append(without_leading_number)
        elif without_trailing_number != token and without_trailing_number:
            units.append(without_trailing_number)
        elif not looks_numeric(token):
            units.append(token)
    return units


def strip_number_prefix(token: str) -> str:
    return re.sub(r"^-?[0-9.,]+", "", token).strip()


def looks_numeric(token: str) -> bool:
    return bool(re.match(r"^-?[0-9.,]+$", token))


def format_highlighted_example(tokens: List[str], labels: List[str]) -> str:
    parts = []
    for token, label in zip(tokens, labels):
        if label == "O":
            parts.append(token)
        else:
            parts.append(f"[{token}/{label}]")
    return " ".join(parts)


def format_example_with_line(example: Dict[str, object]) -> str:
    return (
        f"line {example['line']}: "
        f"{format_highlighted_example(example['tokens'], example['labels'])}"
    )


def render_markdown(analysis: Dict[str, object]) -> str:
    top_k = analysis["top_k"]
    examples = analysis["examples"]
    errors = analysis["errors"]
    lengths = analysis["sentence_lengths"]
    label_counts = analysis["label_counts"]
    entity_span_counts = analysis["entity_span_counts"]
    entity_span_lengths = analysis["entity_span_lengths"]
    entity_token_counts = analysis["entity_token_counts"]
    entity_span_text_counts = analysis["entity_span_text_counts"]
    examples_by_entity = analysis["examples_by_entity"]

    total_tokens = sum(label_counts.values())
    non_o_tokens = total_tokens - label_counts.get("O", 0)
    lines = [
        "# NSW Training Data Report",
        "",
        "## Overview",
        "",
        f"- File: `{analysis['path']}`",
        f"- Valid examples: {len(examples)}",
        f"- Validation errors: {len(errors)}",
        f"- Tokens: {total_tokens}",
        f"- Sentence length: min={min(lengths) if lengths else 0}, "
        f"max={max(lengths) if lengths else 0}, "
        f"avg={mean(lengths):.2f}" if lengths else "- Sentence length: n/a",
        f"- Non-O tokens: {non_o_tokens} ({percentage(non_o_tokens, total_tokens)})",
        "",
    ]

    if errors:
        lines.extend(["## Validation Errors", ""])
        for error in errors[:top_k]:
            lines.append(f"- {error}")
        if len(errors) > top_k:
            lines.append(f"- ... {len(errors) - top_k} more")
        lines.append("")

    lines.extend(render_actionable_checklist(analysis))

    lines.extend(["## Label Distribution", ""])
    lines.extend(render_counter_table(label_counts, "Label", "Tokens"))
    lines.append("")

    lines.extend(["## Entity Span Distribution", ""])
    lines.extend(["| Entity | Spans | Avg span len | Max span len |", "|---|---:|---:|---:|"])
    for entity, count in sorted(entity_span_counts.items(), key=lambda item: (-item[1], item[0])):
        span_lengths = entity_span_lengths[entity]
        lines.append(
            f"| {entity} | {count} | {mean(span_lengths):.2f} | {max(span_lengths)} |"
        )
    lines.append("")

    lines.extend(render_coverage_notes(analysis))

    lines.extend(["## Top Tokens By Entity", ""])
    for entity in sorted(entity_token_counts):
        lines.append(f"### {entity}")
        lines.append("")
        lines.extend(render_counter_table(entity_token_counts[entity], "Token", "Count", top_k))
        lines.append("")

    lines.extend(["## Top Spans By Entity", ""])
    for entity in sorted(entity_span_text_counts):
        lines.append(f"### {entity}")
        lines.append("")
        lines.extend(render_counter_table(entity_span_text_counts[entity], "Span", "Count", top_k))
        lines.append("")

    lines.extend(["## Example Sentences", ""])
    for entity in sorted(examples_by_entity):
        lines.append(f"### {entity}")
        lines.append("")
        for example in examples_by_entity[entity]:
            lines.append(f"- {example}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_counter_table(
    counter: Counter, key_name: str, value_name: str, limit: int = None
) -> List[str]:
    rows = [f"| {key_name} | {value_name} |", "|---|---:|"]
    items = counter.most_common(limit)
    if not items:
        rows.append("| _none_ | 0 |")
        return rows

    for key, value in items:
        rows.append(f"| `{key}` | {value} |")
    return rows


def render_actionable_checklist(analysis: Dict[str, object]) -> List[str]:
    issues = build_actionable_issues(analysis)
    lines = ["## Actionable Coverage Checklist", ""]
    if not issues:
        return lines + ["- No actionable coverage issues found.", ""]

    for issue in issues:
        lines.append(
            f"### {issue['severity'].upper()} - {issue['title']}"
        )
        lines.append("")
        lines.append(f"- Category: {issue['category']}")
        lines.append(f"- Count: {issue['count']}")
        lines.append(f"- Suggested action: {issue['suggestion']}")
        if issue["examples"]:
            lines.append("- Examples:")
            for example in issue["examples"]:
                lines.append(f"  - {example}")
        lines.append("")
    return lines


def build_actionable_issues(analysis: Dict[str, object]) -> List[Dict[str, object]]:
    issues = []
    top_k = min(int(analysis["top_k"]), 10)

    unknown_measurements = [
        unit
        for unit in analysis["measurement_units"]
        if unit not in MEASUREMENT_UNITS_MAPPING
    ]
    add_counter_issue(
        issues,
        analysis,
        issue_id="unknown-measurement-units",
        severity="high",
        category="Mapping",
        title="MEA units missing from MEASUREMENT_UNITS_MAPPING",
        counter=analysis["measurement_units"],
        keys=unknown_measurements,
        suggestion="Add intended readings to MEASUREMENT_UNITS_MAPPING, or keep unmapped units only if pass-through is intentional.",
        entity="MEA",
        top_k=top_k,
    )

    measurement_case_variants = [
        unit
        for unit in unknown_measurements
        if unit.lower() in MEASUREMENT_UNITS_MAPPING
        or unit.upper() in MEASUREMENT_UNITS_MAPPING
    ]
    add_counter_issue(
        issues,
        analysis,
        issue_id="measurement-case-variants",
        severity="medium",
        category="Mapping",
        title="MEA casing variants have no direct mapping",
        counter=analysis["measurement_units"],
        keys=measurement_case_variants,
        suggestion="Add explicit uppercase/lowercase aliases when they should normalize the same way.",
        entity="MEA",
        top_k=top_k,
    )

    unknown_money = [
        unit for unit in analysis["money_units"] if unit not in MONEY_UNITS_MAPPING
    ]
    add_counter_issue(
        issues,
        analysis,
        issue_id="unknown-money-units",
        severity="high",
        category="Mapping",
        title="MONEY units missing from MONEY_UNITS_MAPPING",
        counter=analysis["money_units"],
        keys=unknown_money,
        suggestion="Add currency codes/symbol variants to MONEY_UNITS_MAPPING.",
        entity="MONEY",
        top_k=top_k,
    )

    token_conflicts = find_token_label_conflicts(analysis["token_label_counts"])
    if token_conflicts:
        examples = find_examples_for_tokens(
            analysis["examples"], [token for token, _ in token_conflicts[:top_k]], top_k
        )
        issues.append(
            {
                "id": "token-label-conflicts",
                "severity": "high",
                "category": "Label consistency",
                "title": "Same token appears with multiple entity labels",
                "count": len(token_conflicts),
                "suggestion": "Review whether these are intentional context differences; otherwise standardize labels or add more disambiguating examples.",
                "examples": examples,
            }
        )

    span_conflicts = find_span_entity_conflicts(analysis["span_entity_counts"])
    if span_conflicts:
        examples = find_examples_for_spans(
            analysis["examples"], [span for span, _ in span_conflicts[:top_k]], top_k
        )
        issues.append(
            {
                "id": "span-label-conflicts",
                "severity": "high",
                "category": "Label consistency",
                "title": "Same span appears with multiple entity labels",
                "count": len(span_conflicts),
                "suggestion": "Review repeated spans with conflicting labels before training; they create direct supervision noise.",
                "examples": examples,
            }
        )

    pattern_mismatches = find_pattern_mismatches(analysis["examples"])
    for expected_entity, mismatches in sorted(pattern_mismatches.items()):
        if not mismatches:
            continue
        issues.append(
            {
                "id": f"pattern-{expected_entity.lower()}-mismatches",
                "severity": "medium",
                "category": "Pattern coverage",
                "title": f"Tokens matching {expected_entity} patterns use other labels",
                "count": len(mismatches),
                "suggestion": "Review these examples; add data or correction rules if the pattern label should win.",
                "examples": [format_example_with_line(item) for item in mismatches[:top_k]],
            }
        )

    return sorted(
        issues,
        key=lambda issue: (
            {"high": 0, "medium": 1, "low": 2}.get(issue["severity"], 3),
            issue["category"],
            issue["title"],
        ),
    )


def add_counter_issue(
    issues: List[Dict[str, object]],
    analysis: Dict[str, object],
    issue_id: str,
    severity: str,
    category: str,
    title: str,
    counter: Counter,
    keys: List[str],
    suggestion: str,
    entity: str,
    top_k: int,
):
    if not keys:
        return

    keys = sorted(keys, key=lambda key: (-counter[key], key))
    examples = find_examples_for_units(analysis["examples"], entity, keys[:top_k], top_k)
    top_items = ", ".join(f"`{key}` ({counter[key]})" for key in keys[:top_k])
    issues.append(
        {
            "id": issue_id,
            "severity": severity,
            "category": category,
            "title": title,
            "count": sum(counter[key] for key in keys),
            "suggestion": f"{suggestion} Top items: {top_items}.",
            "examples": examples,
        }
    )


def find_token_label_conflicts(token_label_counts) -> List[Tuple[str, Counter]]:
    conflicts = []
    for token, labels in token_label_counts.items():
        entities = {label_to_entity(label) for label in labels if label != "O"}
        if "O" in labels and entities:
            entities.add("O")
        if len(entities) > 1:
            conflicts.append((token, labels))
    return sorted(conflicts, key=lambda item: (-sum(item[1].values()), item[0]))


def find_span_entity_conflicts(span_entity_counts) -> List[Tuple[str, Counter]]:
    conflicts = [
        (span, entities)
        for span, entities in span_entity_counts.items()
        if len(entities) > 1
    ]
    return sorted(conflicts, key=lambda item: (-sum(item[1].values()), item[0]))


def find_pattern_mismatches(examples: List[Dict[str, object]]):
    mismatches = defaultdict(list)
    for example in examples:
        tokens = example["tokens"]
        labels = example["labels"]
        for index, token in enumerate(tokens):
            expected = expected_entity_for_token(token)
            if not expected:
                continue
            actual = label_to_entity(labels[index])
            if should_report_pattern_mismatch(expected, {actual}):
                mismatches[expected].append(example)
                break
        for index in range(len(tokens) - 1):
            expected = expected_entity_for_bigram(tokens[index], tokens[index + 1])
            if not expected:
                continue
            actual_left = label_to_entity(labels[index])
            actual_right = label_to_entity(labels[index + 1])
            if should_report_pattern_mismatch(expected, {actual_left, actual_right}):
                mismatches[expected].append(example)
                break
    return mismatches


def expected_entity_for_token(token: str):
    if looks_money_token(token):
        return "MONEY"
    if looks_time_token(token):
        return "NTIM"
    if looks_percent_token(token):
        return "NPER"
    if looks_fraction_token(token):
        return "NFRC"
    if looks_score_token(token):
        return "NSCR"
    if looks_range_token(token):
        return "NRNG"
    if looks_date_token(token):
        return "NDAT"
    if looks_month_token(token):
        return "NMON"
    if looks_day_token(token):
        return "NDAY"
    if looks_measure_token(token):
        return "MEA"
    return None


def expected_entity_for_bigram(left: str, right: str):
    if looks_numeric(left) and right == "%":
        return "NPER"
    if looks_numeric(left) and is_known_or_case_variant(right, MEASUREMENT_UNITS_MAPPING):
        return "MEA"
    if looks_numeric(left) and is_known_or_case_variant(right, MONEY_UNITS_MAPPING):
        return "MONEY"
    return None


def should_report_pattern_mismatch(expected: str, actual_entities: set) -> bool:
    if expected in actual_entities:
        return False
    if actual_entities == {"O"}:
        return True
    if expected in {"MEA", "MONEY", "NPER"} and actual_entities.issubset({"NNUM", "O"}):
        return True
    return False


def looks_money_token(token: str) -> bool:
    units = "|".join(re.escape(unit) for unit in sorted(MONEY_UNITS_MAPPING, key=len, reverse=True))
    return bool(re.match(rf"^-?[0-9.,]+({units})$", token)) or bool(
        re.match(rf"^({units})-?[0-9.,]+$", token)
    )


def looks_measure_token(token: str) -> bool:
    unit = strip_number_prefix(token)
    return unit != token and is_measurement_unit_like(unit)


def looks_time_token(token: str) -> bool:
    return bool(re.match(r"^\d{1,2}(:|h)\d{2}$", token)) or bool(
        re.match(r"^\d{1,2}h$", token)
    )


def looks_date_token(token: str) -> bool:
    return bool(re.match(r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$", token))


def looks_month_token(token: str) -> bool:
    return bool(re.match(r"^\d{1,2}[/-]\d{4}$", token))


def looks_day_token(token: str) -> bool:
    return bool(re.match(r"^\d{1,2}[/-]\d{1,2}$", token))


def looks_percent_token(token: str) -> bool:
    return bool(re.match(r"^-?[0-9.,]+%$", token)) or token == "%"


def looks_fraction_token(token: str) -> bool:
    return bool(re.match(r"^\d+/\d+$", token))


def looks_score_token(token: str) -> bool:
    return bool(re.match(r"^\d{1,2}-\d{1,2}$", token))


def looks_range_token(token: str) -> bool:
    return bool(re.match(r"^-?[0-9.,]+[-–]-?[0-9.,]+$", token))


def is_measurement_unit_like(unit: str) -> bool:
    if unit in MEASUREMENT_UNITS_MAPPING:
        return True
    return bool(re.match(r"^[A-Za-zµμ°⁰Ω]+[A-Za-z0-9µμ°⁰Ω/²³]*$", unit))


def is_known_or_case_variant(unit: str, mapping: Dict[str, str]) -> bool:
    return (
        unit in mapping
        or unit.lower() in mapping
        or unit.upper() in mapping
    )


def find_examples_for_units(
    examples: List[Dict[str, object]],
    entity: str,
    units: List[str],
    limit: int,
) -> List[str]:
    unit_set = set(units)
    results = []
    for example in examples:
        for span in iter_spans(example["tokens"], example["labels"]):
            if span["entity"] != entity:
                continue
            if entity == "MEA":
                found_units = extract_measurement_units(span["tokens"])
            elif entity == "MONEY":
                found_units = extract_money_units(span["tokens"])
            else:
                found_units = []
            if unit_set.intersection(found_units):
                results.append(format_example_with_line(example))
                break
        if len(results) >= limit:
            break
    return results


def find_examples_for_tokens(
    examples: List[Dict[str, object]], tokens: List[str], limit: int
) -> List[str]:
    token_set = set(tokens)
    results = []
    for example in examples:
        if token_set.intersection(example["tokens"]):
            results.append(format_example_with_line(example))
        if len(results) >= limit:
            break
    return results


def find_examples_for_spans(
    examples: List[Dict[str, object]], spans: List[str], limit: int
) -> List[str]:
    span_set = set(spans)
    results = []
    for example in examples:
        for span in iter_spans(example["tokens"], example["labels"]):
            if " ".join(span["tokens"]) in span_set:
                results.append(format_example_with_line(example))
                break
        if len(results) >= limit:
            break
    return results


def render_coverage_notes(analysis: Dict[str, object]) -> List[str]:
    notes = ["## Coverage Brainstorm", ""]
    span_counts = analysis["entity_span_counts"]
    all_entities = [label for label in get_base_labels() if label != "O"]
    missing = [entity for entity in all_entities if span_counts.get(entity, 0) == 0]
    rare = [
        entity
        for entity in all_entities
        if 0 < span_counts.get(entity, 0) < 20
    ]

    notes.append(f"- Missing entity types: {', '.join(missing) if missing else 'none'}")
    notes.append(f"- Rare entity types (<20 spans): {', '.join(rare) if rare else 'none'}")

    no_i_labels = []
    label_counts = analysis["label_counts"]
    for entity in all_entities:
        if label_counts.get(f"B-{entity}", 0) and not label_counts.get(f"I-{entity}", 0):
            no_i_labels.append(entity)
    notes.append(
        "- Entity types with B labels but no I labels: "
        + (", ".join(no_i_labels) if no_i_labels else "none")
    )

    unknown_measurements = [
        unit
        for unit in analysis["measurement_units"]
        if unit not in MEASUREMENT_UNITS_MAPPING
    ]
    unknown_money = [
        unit for unit in analysis["money_units"] if unit not in MONEY_UNITS_MAPPING
    ]
    notes.append(
        "- MEA units not in MEASUREMENT_UNITS_MAPPING: "
        + format_counter_subset(analysis["measurement_units"], unknown_measurements)
    )
    notes.append(
        "- MONEY units not in MONEY_UNITS_MAPPING: "
        + format_counter_subset(analysis["money_units"], unknown_money)
    )
    notes.append("")
    return notes


def format_counter_subset(counter: Counter, keys: Iterable[str]) -> str:
    items = [(key, counter[key]) for key in keys]
    if not items:
        return "none"
    items.sort(key=lambda item: (-item[1], item[0]))
    return ", ".join(f"`{key}` ({count})" for key, count in items[:20])


def get_base_labels() -> List[str]:
    labels = []
    for label in get_nsw_bio_labels():
        entity = label_to_entity(label)
        if entity not in labels:
            labels.append(entity)
    return labels


def percentage(part: int, total: int) -> str:
    if not total:
        return "0.00%"
    return f"{part / total * 100:.2f}%"


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()
    analysis = analyze_file(
        args.input,
        top_k=args.top_k,
        examples_per_label=args.examples_per_label,
    )
    report = render_markdown(analysis)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
    else:
        sys.stdout.write(report)

    return 1 if args.strict and analysis["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
