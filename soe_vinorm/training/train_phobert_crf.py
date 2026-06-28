import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from tqdm import tqdm

from soe_vinorm.nsw_detector import get_nsw_bio_labels
from soe_vinorm.phobert_crf import (
    PhoBertCRFTokenClassifier,
    PhoBertInputEncoder,
    _require_ml_dependencies,
    load_label_maps,
    save_phobert_crf_artifacts,
)
from soe_vinorm.training.phobert_crf_dataset import (
    TokenLabelDataset,
    load_jsonl_examples,
    split_examples,
)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train-phobert-crf",
        description="Train a PhoBERT+CRF NSW detector from token-level JSONL data.",
    )
    parser.add_argument("--train", required=True, help="Path to training JSONL file.")
    parser.add_argument("--valid", help="Path to validation JSONL file.")
    parser.add_argument("--output-dir", required=True, help="Directory for model artifacts.")
    parser.add_argument("--model-name", default="vinai/phobert-base")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", help="Device override, e.g. cuda, cuda:0, or cpu.")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate JSONL data and exit before loading ML dependencies or models.",
    )
    return parser


def move_batch_to_device(batch: Dict[str, object], device):
    return {key: value.to(device) for key, value in batch.items()}


def flatten_predictions(
    paths: List[List[int]],
    labels,
    token_mask,
    id2label: Dict[int, str],
) -> Tuple[List[str], List[str]]:
    gold_labels = []
    pred_labels = []
    labels = labels.detach().cpu().tolist()
    mask = token_mask.detach().cpu().tolist()

    for batch_index, path in enumerate(paths):
        for token_index, pred_id in enumerate(path):
            if not mask[batch_index][token_index]:
                continue
            gold_labels.append(id2label[labels[batch_index][token_index]])
            pred_labels.append(id2label[pred_id])

    return gold_labels, pred_labels


def compute_metrics(gold_labels: Iterable[str], pred_labels: Iterable[str]):
    gold = list(gold_labels)
    pred = list(pred_labels)
    if not gold:
        return {"accuracy": 0.0, "non_o_micro_f1": 0.0}

    correct = sum(1 for gold_label, pred_label in zip(gold, pred) if gold_label == pred_label)
    accuracy = correct / len(gold)

    tp = sum(
        1
        for gold_label, pred_label in zip(gold, pred)
        if gold_label != "O" and gold_label == pred_label
    )
    fp = sum(
        1
        for gold_label, pred_label in zip(gold, pred)
        if pred_label != "O" and gold_label != pred_label
    )
    fn = sum(
        1
        for gold_label, pred_label in zip(gold, pred)
        if gold_label != "O" and gold_label != pred_label
    )
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {"accuracy": accuracy, "non_o_micro_f1": f1}


def evaluate(model, data_loader, device, id2label):
    model.eval()
    torch, _, _, _ = _require_ml_dependencies()
    total_loss = 0.0
    steps = 0
    gold = []
    pred = []

    with torch.no_grad():
        for batch in data_loader:
            batch = move_batch_to_device(batch, device)
            outputs = model(**batch)
            paths = model.decode(outputs["emissions"], batch["token_mask"])
            batch_gold, batch_pred = flatten_predictions(
                paths,
                batch["labels"],
                batch["token_mask"],
                id2label,
            )
            gold.extend(batch_gold)
            pred.extend(batch_pred)
            total_loss += outputs["loss"].item()
            steps += 1

    metrics = compute_metrics(gold, pred)
    metrics["loss"] = total_loss / max(steps, 1)
    return metrics


def train(args):
    labels = get_nsw_bio_labels()
    label2id, id2label = load_label_maps(labels)

    train_examples = load_jsonl_examples(args.train, labels)
    if args.valid:
        valid_examples = load_jsonl_examples(args.valid, labels)
    else:
        train_examples, valid_examples = split_examples(
            train_examples,
            valid_ratio=args.valid_ratio,
            seed=args.seed,
        )

    print(
        json.dumps(
            {
                "train_examples": len(train_examples),
                "valid_examples": len(valid_examples),
                "train_tokens": sum(len(example["tokens"]) for example in train_examples),
                "valid_tokens": sum(len(example["tokens"]) for example in valid_examples),
            },
            ensure_ascii=False,
        )
    )

    if args.validate_only:
        return

    torch, _, _, AutoTokenizer = _require_ml_dependencies()
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    input_encoder = PhoBertInputEncoder(tokenizer, max_length=args.max_length)
    train_dataset = TokenLabelDataset(train_examples, input_encoder, label2id)
    valid_dataset = TokenLabelDataset(valid_examples, input_encoder, label2id)

    data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=input_encoder.collate,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=input_encoder.collate,
    )

    model = PhoBertCRFTokenClassifier.create(
        model_name=args.model_name,
        num_labels=len(labels),
        dropout=args.dropout,
    )
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_f1 = -1.0
    output_dir = Path(args.output_dir)
    training_log = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        progress = tqdm(data_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for batch in progress:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            steps += 1
            progress.set_postfix(loss=total_loss / steps)

        valid_metrics = evaluate(model, valid_loader, device, id2label)
        train_loss = total_loss / max(steps, 1)
        epoch_metrics = {"epoch": epoch, "train_loss": train_loss, **valid_metrics}
        training_log.append(epoch_metrics)
        print(json.dumps(epoch_metrics, ensure_ascii=False))

        if valid_metrics["non_o_micro_f1"] > best_f1:
            best_f1 = valid_metrics["non_o_micro_f1"]
            save_phobert_crf_artifacts(
                output_dir=output_dir,
                model=model,
                tokenizer=tokenizer,
                config={
                    "model_type": "phobert_crf",
                    "model_name": args.model_name,
                    "max_length": args.max_length,
                    "dropout": args.dropout,
                },
                label2id=label2id,
            )

    with open(output_dir / "training_metrics.json", "w", encoding="utf-8") as f:
        json.dump(training_log, f, ensure_ascii=False, indent=2)


def main():
    parser = create_parser()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
