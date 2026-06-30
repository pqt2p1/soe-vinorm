from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Union

from tqdm import tqdm

from soe_vinorm.nsw_detector import CRFNSWDetector
from soe_vinorm.nsw_expander import RuleBasedNSWExpander
from soe_vinorm.phobert_crf import PhoBertCRFNSWDetector, PhoBertCRFONNXNSWDetector
from soe_vinorm.text_processor import TextPreprocessor
from soe_vinorm.utils import load_abbreviation_dict, load_vietnamese_syllables


ENABLE_PHOBERT_CRF_ONNX_DETECTOR = False


class Normalizer(ABC):
    """
    Abstract base class for text normalizers.
    """

    @abstractmethod
    def normalize(self, text: str) -> str:
        """
        Normalize text to spoken form.

        Args:
            text: Input text to normalize.

        Returns:
            Normalized text in spoken form.
        """
        ...

    @abstractmethod
    def batch_normalize(
        self, texts: List[str], n_jobs: int = 1, show_progress: bool = False
    ) -> List[str]:
        """
        Normalize multiple texts efficiently.

        Args:
            texts: List of input texts to normalize.
            n_jobs: Number of jobs to run in parallel.
            show_progress: Whether to show progress bar.

        Returns:
            List of normalized texts.
        """
        ...


def _worker_initializer(
    vn_dict: Union[List[str], None] = None,
    abbr_dict: Union[Dict[str, List[str]], None] = None,
    detector: str = "crf",
    kwargs: Dict[str, Any] = {},
):
    """Initialize worker instance."""
    global worker_normalizer
    worker_normalizer = SoeNormalizer(
        vn_dict=vn_dict,
        abbr_dict=abbr_dict,
        detector=detector,
        **kwargs,
    )


def _worker_normalize(text: str) -> str:
    """Normalize text in worker instance."""
    global worker_normalizer
    return worker_normalizer.normalize(text)


class SoeNormalizer(Normalizer):
    """
    Effective Vietnamese text normalizer.
    """

    def __init__(
        self,
        vn_dict: Union[List[str], None] = None,
        abbr_dict: Union[Dict[str, List[str]], None] = None,
        detector: str = "crf",
        **kwargs,
    ):
        """
        Initialize the effective Vietnamese normalizer.

        Args:
            vn_dict: List of Vietnamese words for dictionary lookup. If None, use default Vietnamese dictionary.
            abbr_dict: Dictionary of abbreviations and their expansions. If None, use default abbreviation dictionary.
            detector: NSW detector backend. Use "crf" for the default legacy detector,
                "phobert_crf" for a trained PyTorch PhoBERT+CRF detector, or
                "phobert_crf_onnx" for an exported ONNX detector.
        """
        self._vn_dict = vn_dict or load_vietnamese_syllables()
        self._abbr_dict = abbr_dict or load_abbreviation_dict()
        self._detector_name = detector
        self._kwargs = kwargs

        self._preprocessor = TextPreprocessor(self._vn_dict, **kwargs)
        self._nsw_detector = self._create_detector(detector, kwargs)
        self._nsw_expander = RuleBasedNSWExpander(
            vn_dict=self._vn_dict,
            abbr_dict=self._abbr_dict,
            **self._get_expander_kwargs(detector, kwargs),
        )

    def _create_detector(self, detector: str, kwargs: Dict[str, Any]):
        """Create the configured NSW detector backend."""
        if detector == "crf":
            return CRFNSWDetector(
                vn_dict=self._vn_dict,
                abbr_dict=self._abbr_dict,
                **kwargs,
            )

        if detector == "phobert_crf":
            model_path = self._resolve_phobert_model_path(kwargs)
            device = kwargs.get("device")
            return PhoBertCRFNSWDetector(model_path=model_path, device=device)

        if detector == "phobert_crf_onnx":
            model_path = self._resolve_phobert_model_path(kwargs)
            device = kwargs.get("device")
            if not ENABLE_PHOBERT_CRF_ONNX_DETECTOR:
                return PhoBertCRFNSWDetector(model_path=model_path, device=device)
            return PhoBertCRFONNXNSWDetector(model_path=model_path, device=device)

        raise ValueError(
            "detector must be either 'crf', 'phobert_crf', or 'phobert_crf_onnx'"
        )

    def _resolve_phobert_model_path(self, kwargs: Dict[str, Any]) -> Path:
        model_path = kwargs.get("detector_model_path") or kwargs.get("model_path")
        if model_path is None:
            raise ValueError(
                "PhoBERT detector requires model_path or detector_model_path"
            )

        model_path = Path(model_path)
        nested_model_path = model_path / "phobert_crf"
        if nested_model_path.exists():
            return nested_model_path

        return model_path

    def _get_expander_kwargs(self, detector: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        expander_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in {"detector_model_path", "device"}
        }

        if (
            detector in {"phobert_crf", "phobert_crf_onnx"}
            and "model_path" in expander_kwargs
        ):
            model_path = Path(expander_kwargs["model_path"])
            if not (model_path / "abbreviation_expander").exists():
                expander_kwargs.pop("model_path")

        return expander_kwargs

    def normalize(self, text: str) -> str:
        """
        Normalize text to spoken form.

        Args:
            text: Input text to normalize.

        Returns:
            Normalized text in spoken form.
        """
        if not isinstance(text, str):
            raise TypeError("Input must be a string")

        tokens = self._preprocessor(text).split()

        if not tokens:
            return text.strip()

        nsw_tags = self._nsw_detector.detect(tokens)
        expanded_tokens = self._nsw_expander.expand(tokens, nsw_tags)

        return " ".join(expanded_tokens)

    def detect(self, text: str) -> Dict[str, List[str]]:
        """
        Detect NSW labels after preprocessing without expanding the text.

        Args:
            text: Input text to detect.

        Returns:
            Dictionary containing preprocessed tokens and detected labels.
        """
        if not isinstance(text, str):
            raise TypeError("Input must be a string")

        tokens = self._preprocessor(text).split()
        if not tokens:
            return {"tokens": [], "labels": []}

        return {"tokens": tokens, "labels": self._nsw_detector.detect(tokens)}

    def explain(self, text: str) -> Dict[str, Union[List[str], str]]:
        """
        Detect and expand NSWs, returning intermediate values for debugging.

        Args:
            text: Input text to explain.

        Returns:
            Dictionary containing preprocessed tokens, detected labels, expanded
            tokens, and the final normalized text.
        """
        if not isinstance(text, str):
            raise TypeError("Input must be a string")

        tokens = self._preprocessor(text).split()
        if not tokens:
            return {
                "tokens": [],
                "labels": [],
                "expanded_tokens": [],
                "normalized": text.strip(),
            }

        labels = self._nsw_detector.detect(tokens)
        expanded_tokens = self._nsw_expander.expand(tokens, labels)
        return {
            "tokens": tokens,
            "labels": labels,
            "expanded_tokens": expanded_tokens,
            "normalized": " ".join(expanded_tokens),
        }

    def batch_detect(self, texts: List[str]) -> List[Dict[str, List[str]]]:
        """
        Detect NSW labels for multiple texts after preprocessing.

        Args:
            texts: List of input texts to detect.

        Returns:
            List of dictionaries containing preprocessed tokens and detected labels.
        """
        if not isinstance(texts, list) or not all(
            isinstance(text, str) for text in texts
        ):
            raise TypeError("Input must be a list of strings")

        tokenized_texts = [self._preprocessor(text).split() for text in texts]
        non_empty = [
            (index, tokens) for index, tokens in enumerate(tokenized_texts) if tokens
        ]
        labels = [[] for _ in tokenized_texts]
        if non_empty:
            detected = self._nsw_detector.batch_detect(
                [tokens for _, tokens in non_empty]
            )
            for (index, _), detected_labels in zip(non_empty, detected):
                labels[index] = detected_labels

        return [
            {"tokens": tokens, "labels": token_labels}
            for tokens, token_labels in zip(tokenized_texts, labels)
        ]

    def batch_normalize(
        self, texts: List[str], n_jobs: int = 1, show_progress: bool = False
    ) -> List[str]:
        """
        Normalize multiple texts efficiently.

        Args:
            texts: List of input texts to normalize.
            n_jobs: Number of jobs to run in parallel.
            show_progress: Whether to show progress bar.

        Returns:
            List of normalized texts.
        """
        if not isinstance(texts, list) or not all(
            isinstance(text, str) for text in texts
        ):
            raise TypeError("Input must be a list of strings")

        if n_jobs <= 0:
            raise ValueError("Number of jobs must be greater than 0")

        if n_jobs == 1:
            return [
                self.normalize(text)
                for text in tqdm(
                    texts,
                    desc="Normalizing texts",
                    total=len(texts),
                    disable=not show_progress,
                )
            ]

        with ProcessPoolExecutor(
            max_workers=n_jobs,
            initializer=_worker_initializer,
            initargs=(self._vn_dict, self._abbr_dict, self._detector_name, self._kwargs),
        ) as executor:
            return list(
                tqdm(
                    executor.map(_worker_normalize, texts),
                    desc="Normalizing texts",
                    total=len(texts),
                    disable=not show_progress,
                )
            )


def normalize_text(
    text: str,
    vn_dict: Union[List[str], None] = None,
    abbr_dict: Union[Dict[str, List[str]], None] = None,
    **kwargs,
) -> str:
    """
    Quick normalization function.

    Args:
        text: Input text to normalize.
        vn_dict: Optional Vietnamese dictionary. If None, use default Vietnamese dictionary.
        abbr_dict: Optional abbreviation dictionary. If None, use default abbreviation dictionary.

    Returns:
        Normalized text.
    """
    normalizer = SoeNormalizer(
        vn_dict=vn_dict,
        abbr_dict=abbr_dict,
        **kwargs,
    )
    return normalizer.normalize(text)


def batch_normalize_texts(
    texts: List[str],
    vn_dict: Union[List[str], None] = None,
    abbr_dict: Union[Dict[str, List[str]], None] = None,
    n_jobs: int = 1,
    show_progress: bool = False,
    **kwargs,
) -> List[str]:
    """
    Batch normalization function.

    Args:
        texts: List of input texts to normalize.
        vn_dict: Optional Vietnamese dictionary. If None, use default Vietnamese dictionary.
        abbr_dict: Optional abbreviation dictionary. If None, use default abbreviation dictionary.
        n_jobs: Number of jobs to run in parallel.
        show_progress: Whether to show progress bar.

    Returns:
        List of normalized texts.
    """
    normalizer = SoeNormalizer(
        vn_dict=vn_dict,
        abbr_dict=abbr_dict,
        **kwargs,
    )
    return normalizer.batch_normalize(texts, n_jobs=n_jobs, show_progress=show_progress)
