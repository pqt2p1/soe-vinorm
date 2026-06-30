import argparse
import json
import sys
from pathlib import Path
from textwrap import dedent
from typing import Dict, List, Union

from soe_vinorm import SoeNormalizer


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        prog="soe-vinorm",
        description="Vietnamese text normalization toolkit - Convert text to spoken form",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent("""
            Examples:
              # Normalize text from stdin
              echo "Năm 2021" | soe-vinorm

              # Normalize texts from a file line by line
              soe-vinorm -i input.txt -o output.txt

              # Process with custom options
              soe-vinorm -i input.txt --no-expand-sequence --no-expand-urle

              # Batch process with parallel workers
              soe-vinorm -i input.txt -o output.txt --n-jobs 4 --show-progress
            """),
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Input file path. If not specified, read from stdin.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output file path. If not specified, write to stdout.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to the model repository directory for loading pre-downloaded weights.",
    )
    parser.add_argument(
        "--detector",
        choices=["crf", "phobert_crf", "phobert_crf_onnx"],
        default="crf",
        help="NSW detector backend to use (default: crf).",
    )
    parser.add_argument(
        "--detector-model-path",
        type=str,
        help="Path to a trained detector model. Defaults to --model-path when omitted.",
    )
    parser.add_argument(
        "--no-expand-sequence",
        action="store_false",
        dest="expand_sequence",
        default=True,
        help="Disable expansion of unknown sequences (default: expand enabled).",
    )
    parser.add_argument(
        "--no-expand-urle",
        action="store_false",
        dest="expand_urle",
        default=True,
        help="Disable expansion of URLs and emails (default: expand enabled).",
    )
    parser.add_argument(
        "--no-expand-unknown",
        action="store_false",
        dest="expand_unknown",
        default=True,
        help="Disable spelling out unknown non-NSW tokens (default: expand enabled).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs for batch processing (default: 1).",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Show progress bar during batch processing.",
    )
    parser.add_argument(
        "--detect-only",
        action="store_true",
        help="Output preprocessed tokens and NSW labels as JSONL without normalization.",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help=(
            "Output tokens, labels, expanded tokens, and normalized text as JSONL "
            "for debugging."
        ),
    )
    parser.add_argument(
        "--explain-format",
        choices=["json", "table", "changed"],
        help=(
            "Format for --explain output. Defaults to json in batch mode and "
            "table in interactive mode."
        ),
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Keep the process alive and process one stdin line at a time.",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="store_true",
        help="Show version information.",
    )

    return parser


def read_input(input_path: Union[str, None] = None) -> List[str]:
    """Read input text from file or stdin."""
    if input_path:
        path = Path(input_path)
        if not path.exists():
            print(f"Error: Input file '{input_path}' not found.", file=sys.stderr)
            sys.exit(1)
        with open(path, "r", encoding="utf-8") as f:
            return [line.rstrip("\n") for line in f]
    else:
        return [line.rstrip("\n") for line in sys.stdin]


def write_output(lines: List[str], output_path: Union[str, None] = None):
    """Write output text to file or stdout."""
    if output_path:
        path = Path(output_path)
        with open(path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
    else:
        for line in lines:
            print(line)


def format_explanation(
    explanation: Dict[str, Union[List[str], str]],
    output_format: str,
    input_text: Union[str, None] = None,
):
    """Format explanation output for machines or terminal debugging."""
    if output_format == "json":
        return json.dumps(explanation, ensure_ascii=False)

    tokens = explanation["tokens"]
    labels = explanation["labels"]
    expanded_tokens = explanation["expanded_tokens"]
    normalized = explanation["normalized"]

    if output_format == "changed":
        rows = [
            (str(index), token, label, expanded)
            for index, (token, label, expanded) in enumerate(
                zip(tokens, labels, expanded_tokens)
            )
            if label != "O" or token != expanded
        ]
        if not rows:
            body = "(no token-level changes)"
        else:
            body = "\n".join(
                f"{index}  {token}  {label}  {token} -> {expanded}"
                for index, token, label, expanded in rows
            )
        prefix = f"INPUT: {input_text}\n\n" if input_text is not None else ""
        return f"{prefix}{body}\nNORMALIZED: {normalized}"

    headers = ("idx", "token", "label", "expanded")
    rows = [
        (str(index), token, label, expanded)
        for index, (token, label, expanded) in enumerate(
            zip(tokens, labels, expanded_tokens)
        )
    ]
    widths = [
        max(len(headers[column]), *(len(row[column]) for row in rows))
        if rows
        else len(headers[column])
        for column in range(len(headers))
    ]

    def format_row(row):
        return "  ".join(value.ljust(width) for value, width in zip(row, widths))

    table_lines = [
        *(["INPUT:", str(input_text), ""] if input_text is not None else []),
        format_row(headers),
        format_row(tuple("-" * width for width in widths)),
        *(format_row(row) for row in rows),
        "",
        "NORMALIZED:",
        str(normalized),
    ]
    return "\n".join(table_lines)


def process_line(
    normalizer: SoeNormalizer,
    line: str,
    detect_only: bool,
    explain: bool = False,
    explain_format: str = "json",
) -> str:
    """Process a single input line."""
    if explain:
        return format_explanation(normalizer.explain(line), explain_format, line)
    if detect_only:
        return json.dumps(normalizer.detect(line), ensure_ascii=False)
    return normalizer.normalize(line)


def run_interactive(
    normalizer: SoeNormalizer,
    detect_only: bool,
    explain: bool,
    explain_format: str,
):
    """Process stdin line by line while keeping the normalizer in memory."""
    if sys.stdin.isatty():
        print(
            "soe-vinorm interactive mode ready. Press Ctrl-D to exit.",
            file=sys.stderr,
        )

    for line in sys.stdin:
        line = line.rstrip("\n")
        try:
            print(
                process_line(
                    normalizer,
                    line,
                    detect_only,
                    explain,
                    explain_format,
                ),
                flush=True,
            )
            if explain and explain_format != "json":
                print("-" * 72, flush=True)
        except Exception as e:
            print(f"Error during normalization: {e}", file=sys.stderr, flush=True)


def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.version:
        from soe_vinorm import __version__

        print(f"Soe Vinorm {__version__}")
        sys.exit(0)

    if args.interactive and (args.input or args.output):
        print(
            "Error: --interactive cannot be used with --input or --output.",
            file=sys.stderr,
        )
        sys.exit(1)

    explain_format = args.explain_format or ("table" if args.interactive else "json")

    try:
        kwargs = {
            "detector": args.detector,
            "expand_sequence": args.expand_sequence,
            "expand_urle": args.expand_urle,
            "expand_unknown": args.expand_unknown,
        }
        if args.model_path:
            kwargs["model_path"] = args.model_path
        if args.detector_model_path:
            kwargs["detector_model_path"] = args.detector_model_path

        normalizer = SoeNormalizer(**kwargs)
    except Exception as e:
        print(f"Error initializing normalizer: {e}", file=sys.stderr)
        sys.exit(1)

    if args.interactive:
        run_interactive(normalizer, args.detect_only, args.explain, explain_format)
        return

    try:
        lines = read_input(args.input)
    except Exception as e:
        print(f"Error reading input: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        if args.explain:
            if args.n_jobs != 1:
                print(
                    "Warning: --explain ignores --n-jobs and runs in one process.",
                    file=sys.stderr,
                )
            normalized_texts = [
                format_explanation(normalizer.explain(line), explain_format)
                for line in lines
            ]
        elif args.detect_only:
            if args.n_jobs != 1:
                print(
                    "Warning: --detect-only ignores --n-jobs and runs in one process.",
                    file=sys.stderr,
                )
            normalized_texts = [
                json.dumps(result, ensure_ascii=False)
                for result in normalizer.batch_detect(lines)
            ]
        else:
            normalized_texts = normalizer.batch_normalize(
                lines, n_jobs=args.n_jobs, show_progress=args.show_progress
            )
    except Exception as e:
        print(f"Error during normalization: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        write_output(normalized_texts, args.output)
    except Exception as e:
        print(f"Error writing output: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
