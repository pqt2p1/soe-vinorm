import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from soe_vinorm.nsw_detector import NSWDetector

ML_IMPORT_ERROR = (
    "PhoBERT+CRF support requires the optional ML dependencies. "
    "Install them with `uv sync --group ml` or install torch and transformers."
)


def _require_ml_dependencies():
    try:
        import torch
        from torch import nn
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError(ML_IMPORT_ERROR) from exc

    return torch, nn, AutoModel, AutoTokenizer


class LinearChainCRF:
    """Factory wrapper so importing this module does not require torch."""

    @staticmethod
    def create(num_tags: int):
        torch, nn, _, _ = _require_ml_dependencies()

        class _LinearChainCRF(nn.Module):
            def __init__(self, num_tags: int):
                super().__init__()
                self.num_tags = num_tags
                self.start_transitions = nn.Parameter(torch.empty(num_tags))
                self.end_transitions = nn.Parameter(torch.empty(num_tags))
                self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
                self.reset_parameters()

            def reset_parameters(self):
                nn.init.uniform_(self.start_transitions, -0.1, 0.1)
                nn.init.uniform_(self.end_transitions, -0.1, 0.1)
                nn.init.uniform_(self.transitions, -0.1, 0.1)

            def forward(self, emissions, tags, mask):
                numerator = self._compute_score(emissions, tags, mask)
                denominator = self._compute_normalizer(emissions, mask)
                return numerator - denominator

            def decode(self, emissions, mask):
                return self._viterbi_decode(emissions, mask)

            def _compute_score(self, emissions, tags, mask):
                batch_size, seq_len = tags.shape
                batch_indices = torch.arange(batch_size, device=emissions.device)
                score = self.start_transitions[tags[:, 0]]
                score += emissions[batch_indices, 0, tags[:, 0]]

                for i in range(1, seq_len):
                    transition_score = self.transitions[tags[:, i - 1], tags[:, i]]
                    emission_score = emissions[batch_indices, i, tags[:, i]]
                    score += (transition_score + emission_score) * mask[:, i]

                lengths = mask.long().sum(dim=1) - 1
                last_tags = tags[batch_indices, lengths]
                score += self.end_transitions[last_tags]
                return score

            def _compute_normalizer(self, emissions, mask):
                score = self.start_transitions + emissions[:, 0]

                for i in range(1, emissions.size(1)):
                    next_score = (
                        score.unsqueeze(2)
                        + self.transitions.unsqueeze(0)
                        + emissions[:, i].unsqueeze(1)
                    )
                    next_score = torch.logsumexp(next_score, dim=1)
                    score = torch.where(mask[:, i].unsqueeze(1), next_score, score)

                score += self.end_transitions
                return torch.logsumexp(score, dim=1)

            def _viterbi_decode(self, emissions, mask):
                score = self.start_transitions + emissions[:, 0]
                history = []

                for i in range(1, emissions.size(1)):
                    next_score = (
                        score.unsqueeze(2)
                        + self.transitions.unsqueeze(0)
                        + emissions[:, i].unsqueeze(1)
                    )
                    next_score, indices = next_score.max(dim=1)
                    score = torch.where(mask[:, i].unsqueeze(1), next_score, score)
                    history.append(indices)

                score += self.end_transitions
                best_last_tags = score.argmax(dim=1)

                best_paths = []
                lengths = mask.long().sum(dim=1)
                for batch_idx, length in enumerate(lengths.tolist()):
                    tag = best_last_tags[batch_idx]
                    path = [tag.item()]
                    for hist in reversed(history[: max(length - 1, 0)]):
                        tag = hist[batch_idx][tag]
                        path.append(tag.item())
                    best_paths.append(list(reversed(path)))

                return best_paths

        return _LinearChainCRF(num_tags)


class PhoBertCRFTokenClassifier:
    """Factory wrapper for a PhoBERT encoder with a CRF token-classification head."""

    @staticmethod
    def create(model_name: str, num_labels: int, dropout: float = 0.1):
        torch, nn, AutoModel, _ = _require_ml_dependencies()

        class _PhoBertCRFTokenClassifier(nn.Module):
            def __init__(self, model_name: str, num_labels: int, dropout: float):
                super().__init__()
                self.encoder = AutoModel.from_pretrained(model_name)
                self.dropout = nn.Dropout(dropout)
                hidden_size = self.encoder.config.hidden_size
                self.classifier = nn.Linear(hidden_size, num_labels)
                self.crf = LinearChainCRF.create(num_labels)

            def forward(
                self,
                input_ids,
                attention_mask,
                token_positions,
                token_mask,
                labels=None,
            ):
                outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                hidden = outputs.last_hidden_state
                batch_indices = torch.arange(hidden.size(0), device=hidden.device)
                token_hidden = hidden[batch_indices.unsqueeze(1), token_positions]
                emissions = self.classifier(self.dropout(token_hidden))

                loss = None
                if labels is not None:
                    log_likelihood = self.crf(emissions, labels, token_mask)
                    loss = -log_likelihood.mean()

                return {"loss": loss, "emissions": emissions}

            def decode(self, emissions, token_mask):
                return self.crf.decode(emissions, token_mask)

        return _PhoBertCRFTokenClassifier(model_name, num_labels, dropout)


class PhoBertInputEncoder:
    """Convert token-level examples to PhoBERT inputs and token positions."""

    def __init__(self, tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id
        self.pad_token_id = tokenizer.pad_token_id
        if self.cls_token_id is None or self.sep_token_id is None:
            raise ValueError("Tokenizer must define cls_token_id and sep_token_id")
        if self.pad_token_id is None:
            self.pad_token_id = 1

    def encode(
        self,
        tokens: List[str],
        labels: Optional[List[str]] = None,
        label2id: Optional[Dict[str, int]] = None,
    ) -> Dict[str, List[int]]:
        if labels is not None and len(tokens) != len(labels):
            raise ValueError("tokens and labels must have the same length")

        input_ids = [self.cls_token_id]
        token_positions = []
        encoded_labels = []

        for index, token in enumerate(tokens):
            sub_tokens = self.tokenizer.tokenize(token)
            sub_token_ids = self.tokenizer.convert_tokens_to_ids(sub_tokens)
            if not sub_token_ids:
                sub_token_ids = [self.tokenizer.unk_token_id]

            if len(input_ids) + len(sub_token_ids) + 1 > self.max_length:
                break

            token_positions.append(len(input_ids))
            input_ids.extend(sub_token_ids)
            if labels is not None and label2id is not None:
                encoded_labels.append(label2id[labels[index]])

        input_ids.append(self.sep_token_id)
        attention_mask = [1] * len(input_ids)

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_positions": token_positions,
            "token_mask": [1] * len(token_positions),
        }
        if labels is not None:
            item["labels"] = encoded_labels

        return item

    def collate(self, batch: List[Dict[str, List[int]]]) -> Dict[str, object]:
        torch, _, _, _ = _require_ml_dependencies()
        max_input_len = max(len(item["input_ids"]) for item in batch)
        max_tokens = max(len(item["token_positions"]) for item in batch)

        input_ids = []
        attention_mask = []
        token_positions = []
        token_mask = []
        labels = []
        has_labels = "labels" in batch[0]

        for item in batch:
            input_pad = max_input_len - len(item["input_ids"])
            token_pad = max_tokens - len(item["token_positions"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * input_pad)
            attention_mask.append(item["attention_mask"] + [0] * input_pad)
            token_positions.append(item["token_positions"] + [0] * token_pad)
            token_mask.append(item["token_mask"] + [0] * token_pad)
            if has_labels:
                labels.append(item["labels"] + [0] * token_pad)

        result = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_positions": torch.tensor(token_positions, dtype=torch.long),
            "token_mask": torch.tensor(token_mask, dtype=torch.bool),
        }
        if has_labels:
            result["labels"] = torch.tensor(labels, dtype=torch.long)

        return result


class PhoBertCRFNSWDetector(NSWDetector):
    """NSW detector backed by a fine-tuned PhoBERT+CRF token classifier."""

    def __init__(
        self,
        model_path: Union[str, Path],
        device: Optional[str] = None,
        **kwargs,
    ):
        torch, _, _, AutoTokenizer = _require_ml_dependencies()
        super().__init__()
        self.model_path = Path(model_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        config_path = self.model_path / "config.json"
        label_path = self.model_path / "id2label.json"
        weights_path = self.model_path / "pytorch_model.bin"
        if not config_path.exists() or not label_path.exists() or not weights_path.exists():
            raise FileNotFoundError(
                "PhoBERT+CRF model_path must contain config.json, "
                "id2label.json, and pytorch_model.bin"
            )

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        with open(label_path, "r", encoding="utf-8") as f:
            raw_id2label = json.load(f)

        self.id2label = {int(key): value for key, value in raw_id2label.items()}
        self.label2id = {value: key for key, value in self.id2label.items()}

        tokenizer_path = self.model_path
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), use_fast=False)
        self.input_encoder = PhoBertInputEncoder(
            self.tokenizer,
            max_length=int(self.config.get("max_length", 256)),
        )

        encoder_path = self.model_path / "encoder"
        model_name = str(encoder_path) if encoder_path.exists() else self.config.get(
            "model_name", str(self.model_path)
        )
        self.model = PhoBertCRFTokenClassifier.create(
            model_name=model_name,
            num_labels=len(self.id2label),
            dropout=float(self.config.get("dropout", 0.1)),
        )
        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def detect(self, tokenized_text: List[str]) -> List[str]:
        if not isinstance(tokenized_text, list) or not all(
            isinstance(token, str) for token in tokenized_text
        ):
            raise TypeError("tokenized_text must be a list of strings")

        return self.batch_detect([tokenized_text])[0]

    def batch_detect(self, tokenized_texts: List[List[str]]) -> List[List[str]]:
        if not isinstance(tokenized_texts, list) or not all(
            isinstance(text, list) and all(isinstance(token, str) for token in text)
            for text in tokenized_texts
        ):
            raise TypeError("tokenized_texts must be a list of lists of strings")

        torch, _, _, _ = _require_ml_dependencies()
        if not tokenized_texts:
            return []

        results = [[] for _ in tokenized_texts]
        non_empty_items = [
            (index, tokens) for index, tokens in enumerate(tokenized_texts) if tokens
        ]
        if not non_empty_items:
            return results

        encoded = [
            self.input_encoder.encode(tokens) for _, tokens in non_empty_items
        ]
        batch = self.input_encoder.collate(encoded)
        batch = {key: value.to(self.device) for key, value in batch.items()}

        with torch.no_grad():
            outputs = self.model(**batch)
            paths = self.model.decode(outputs["emissions"], batch["token_mask"])

        for (original_index, _), path in zip(non_empty_items, paths):
            results[original_index] = [self.id2label[tag_id] for tag_id in path]

        return results

    def get_labels(self) -> List[str]:
        return [self.id2label[index] for index in sorted(self.id2label)]


class PhoBertCRFONNXNSWDetector(NSWDetector):
    """NSW detector backed by ONNX PhoBERT emissions and NumPy CRF decoding."""

    def __init__(self, model_path: Union[str, Path], device: Optional[str] = None, **kwargs):
        try:
            from onnxruntime import InferenceSession, SessionOptions
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "PhoBERT+CRF ONNX support requires onnxruntime and transformers."
            ) from exc

        super().__init__()
        self.model_path = Path(model_path)

        config_path = self.model_path / "config.json"
        label_path = self.model_path / "id2label.json"
        onnx_path = self.model_path / "model.onnx"
        transitions_path = self.model_path / "crf_transitions.npz"
        if (
            not config_path.exists()
            or not label_path.exists()
            or not onnx_path.exists()
            or not transitions_path.exists()
        ):
            raise FileNotFoundError(
                "PhoBERT+CRF ONNX model_path must contain config.json, "
                "id2label.json, model.onnx, and crf_transitions.npz"
            )

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        with open(label_path, "r", encoding="utf-8") as f:
            raw_id2label = json.load(f)

        self.id2label = {int(key): value for key, value in raw_id2label.items()}
        self.label2id = {value: key for key, value in self.id2label.items()}

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            use_fast=False,
        )
        self.input_encoder = PhoBertInputEncoder(
            self.tokenizer,
            max_length=int(self.config.get("max_length", 256)),
        )

        providers = ["CPUExecutionProvider"]
        if device and str(device).startswith("cuda"):
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        session_options = SessionOptions()
        self.session = InferenceSession(
            str(onnx_path),
            sess_options=session_options,
            providers=providers,
        )
        self.input_names = {item.name for item in self.session.get_inputs()}

        transitions = np.load(transitions_path)
        self.start_transitions = transitions["start_transitions"]
        self.end_transitions = transitions["end_transitions"]
        self.transitions = transitions["transitions"]

    def detect(self, tokenized_text: List[str]) -> List[str]:
        if not isinstance(tokenized_text, list) or not all(
            isinstance(token, str) for token in tokenized_text
        ):
            raise TypeError("tokenized_text must be a list of strings")

        return self.batch_detect([tokenized_text])[0]

    def batch_detect(self, tokenized_texts: List[List[str]]) -> List[List[str]]:
        if not isinstance(tokenized_texts, list) or not all(
            isinstance(text, list) and all(isinstance(token, str) for token in text)
            for text in tokenized_texts
        ):
            raise TypeError("tokenized_texts must be a list of lists of strings")

        if not tokenized_texts:
            return []

        results = [[] for _ in tokenized_texts]
        non_empty_items = [
            (index, tokens) for index, tokens in enumerate(tokenized_texts) if tokens
        ]
        if not non_empty_items:
            return results

        encoded = [self.input_encoder.encode(tokens) for _, tokens in non_empty_items]
        batch = self.input_encoder.collate(encoded)
        feeds = {
            name: batch[name].detach().cpu().numpy()
            for name in self.input_names
            if name in batch
        }
        emissions = self.session.run(None, feeds)[0]
        token_mask = batch["token_mask"].detach().cpu().numpy().astype(bool)
        paths = _viterbi_decode_numpy(
            emissions,
            token_mask,
            self.start_transitions,
            self.end_transitions,
            self.transitions,
        )

        for (original_index, _), path in zip(non_empty_items, paths):
            results[original_index] = [self.id2label[tag_id] for tag_id in path]

        return results

    def get_labels(self) -> List[str]:
        return [self.id2label[index] for index in sorted(self.id2label)]


def _viterbi_decode_numpy(
    emissions: np.ndarray,
    mask: np.ndarray,
    start_transitions: np.ndarray,
    end_transitions: np.ndarray,
    transitions: np.ndarray,
) -> List[List[int]]:
    score = start_transitions + emissions[:, 0]
    history = []

    for index in range(1, emissions.shape[1]):
        next_score = (
            score[:, :, None]
            + transitions[None, :, :]
            + emissions[:, index][:, None, :]
        )
        indices = next_score.argmax(axis=1)
        next_score = next_score.max(axis=1)
        score = np.where(mask[:, index][:, None], next_score, score)
        history.append(indices)

    score = score + end_transitions
    best_last_tags = score.argmax(axis=1)

    best_paths = []
    lengths = mask.sum(axis=1).astype(int)
    for batch_index, length in enumerate(lengths.tolist()):
        tag = int(best_last_tags[batch_index])
        path = [tag]
        for hist in reversed(history[: max(length - 1, 0)]):
            tag = int(hist[batch_index][tag])
            path.append(tag)
        best_paths.append(list(reversed(path)))

    return best_paths


def export_phobert_crf_to_onnx(
    model_dir: Union[str, Path],
    output_path: Union[str, Path, None] = None,
    opset: int = 18,
    device: Optional[str] = None,
) -> Path:
    """Export trained PhoBERT+CRF emissions to ONNX and save CRF transitions."""
    torch, nn, _, AutoTokenizer = _require_ml_dependencies()

    model_dir = Path(model_dir)
    output_path = Path(output_path) if output_path else model_dir / "model.onnx"
    config_path = model_dir / "config.json"
    label_path = model_dir / "id2label.json"
    weights_path = model_dir / "pytorch_model.bin"
    if not config_path.exists() or not label_path.exists() or not weights_path.exists():
        raise FileNotFoundError(
            "model_dir must contain config.json, id2label.json, and pytorch_model.bin"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    with open(label_path, "r", encoding="utf-8") as f:
        id2label = {int(key): value for key, value in json.load(f).items()}

    run_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    encoder_path = model_dir / "encoder"
    model_name = str(encoder_path) if encoder_path.exists() else config.get(
        "model_name", str(model_dir)
    )
    model = PhoBertCRFTokenClassifier.create(
        model_name=model_name,
        num_labels=len(id2label),
        dropout=float(config.get("dropout", 0.1)),
    )
    state_dict = torch.load(weights_path, map_location=run_device)
    model.load_state_dict(state_dict)
    model.to(run_device)
    model.eval()

    transitions_path = model_dir / "crf_transitions.npz"
    np.savez(
        transitions_path,
        start_transitions=model.crf.start_transitions.detach().cpu().numpy(),
        end_transitions=model.crf.end_transitions.detach().cpu().numpy(),
        transitions=model.crf.transitions.detach().cpu().numpy(),
    )

    class _EmissionsOnly(nn.Module):
        def __init__(self, wrapped_model):
            super().__init__()
            self.wrapped_model = wrapped_model

        def forward(self, input_ids, attention_mask, token_positions):
            outputs = self.wrapped_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_positions=token_positions,
                token_mask=None,
            )
            return outputs["emissions"]

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=False)
    input_encoder = PhoBertInputEncoder(
        tokenizer,
        max_length=int(config.get("max_length", 256)),
    )
    sample = input_encoder.collate([input_encoder.encode(["xin", "chào"])])
    input_ids = sample["input_ids"].to(run_device)
    attention_mask = sample["attention_mask"].to(run_device)
    token_positions = sample["token_positions"].to(run_device)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_model = _EmissionsOnly(model)
    export_model.eval()

    torch.onnx.export(
        export_model,
        (input_ids, attention_mask, token_positions),
        str(output_path),
        input_names=["input_ids", "attention_mask", "token_positions"],
        output_names=["emissions"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "input_length"},
            "attention_mask": {0: "batch", 1: "input_length"},
            "token_positions": {0: "batch", 1: "token_length"},
            "emissions": {0: "batch", 1: "token_length"},
        },
        opset_version=opset,
    )

    return output_path


def save_phobert_crf_artifacts(
    output_dir: Union[str, Path],
    model,
    tokenizer,
    config: Dict[str, object],
    label2id: Dict[str, int],
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch, _, _, _ = _require_ml_dependencies()
    model_to_save = model.module if hasattr(model, "module") else model
    torch.save(model_to_save.state_dict(), output_dir / "pytorch_model.bin")
    model_to_save.encoder.save_pretrained(str(output_dir / "encoder"))
    tokenizer.save_pretrained(str(output_dir))

    id2label = {str(index): label for label, index in label2id.items()}
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    with open(output_dir / "label2id.json", "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)
    with open(output_dir / "id2label.json", "w", encoding="utf-8") as f:
        json.dump(id2label, f, ensure_ascii=False, indent=2)


def load_label_maps(labels: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {index: label for label, index in label2id.items()}
    return label2id, id2label
