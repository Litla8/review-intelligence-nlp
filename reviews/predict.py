"""Run the fine-tuned DistilBERT on new text: the shared scorer used by Step 8 and the Streamlit app."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Sequence, Tuple

import numpy as np

from reviews.config import AspectConfig

LOGGER = logging.getLogger(__name__)
Scorer = Callable[[Sequence[str]], np.ndarray]


def read_threshold(model_dir: Path, default: float = 0.5) -> float:
    """The probability cut-off tuned in Step 6 (saved next to the model); ``default`` if the file is missing."""
    path = model_dir / "inference_config.json"
    if not path.is_file():
        LOGGER.warning("%s not found; using threshold %.2f", path, default)
        return default
    return float(json.loads(path.read_text(encoding="utf-8")).get("threshold", default))


def label_probabilities(p_positive: np.ndarray, threshold: float, cfg: AspectConfig = AspectConfig()) -> np.ndarray:
    """``positive`` / ``negative`` for confident scores and ``mixed`` for those between the neutral bands (vectorised)."""
    p = np.asarray(p_positive, dtype=float)
    return np.where(p >= cfg.neutral_high, "positive", np.where(p <= cfg.neutral_low, "negative", "mixed"))


def load_scorer(model_dir: Path, max_length: int, batch_size: int, require_cuda: bool = True) -> Scorer:
    """Return ``scorer(texts) -> P(positive)`` for the model saved in ``model_dir``.

    Texts are sorted by length before batching (far less padding) and returned in their original order. Without a
    GPU the scorer runs on the CPU when ``require_cuda`` is False (fine for single reviews, slow for big files).

    Raises:
        FileNotFoundError: If ``model_dir`` is missing.
        ImportError: If torch / transformers are missing.
        RuntimeError: If ``require_cuda`` is True and PyTorch cannot see a GPU.
    """
    if not model_dir.is_dir():
        raise FileNotFoundError(f"{model_dir} not found. Run `python -m reviews.train_transformer` first.")
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise ImportError("This step needs torch and transformers (see the Step 6 install commands).") from exc
    cuda = torch.cuda.is_available()
    if require_cuda and not cuda:
        raise RuntimeError("PyTorch cannot see a CUDA GPU; see the Step 6 install commands.")
    device = "cuda" if cuda else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device).eval()

    def scorer(texts: Sequence[str]) -> np.ndarray:
        order = np.argsort([len(t) for t in texts], kind="stable")
        out = np.zeros(len(texts), dtype=np.float64)
        with torch.no_grad():
            for start in range(0, len(order), batch_size):
                idx = order[start:start + batch_size]
                enc = tokenizer([texts[i] for i in idx], truncation=True, max_length=max_length, padding=True,
                                return_tensors="pt").to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=cuda):
                    logits = model(**enc).logits.float()
                out[idx] = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        return out

    return scorer
