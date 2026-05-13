"""
Transformer-based sentiment scoring with context awareness.

Uses a Twitter-tuned RoBERTa model by default (handles slang like "sick" better than lexicons).
Returns:
  - label: "Positive" | "Neutral" | "Negative"
  - score: float in [-1, 1] where positive values mean positive sentiment
  - confidence: float in [0, 1] (max class probability)
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List, Tuple


DEFAULT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"


def _normalize_label(raw: str) -> str:
    s = (raw or "").strip().lower()
    if "pos" in s:
        return "Positive"
    if "neg" in s:
        return "Negative"
    if "neu" in s:
        return "Neutral"
    # Common id-style labels from some checkpoints.
    if s in {"label_2"}:
        return "Positive"
    if s in {"label_1"}:
        return "Neutral"
    if s in {"label_0"}:
        return "Negative"
    return "Neutral"


@lru_cache(maxsize=1)
def _get_pipeline(model_name: str = DEFAULT_MODEL):
    # Imported lazily to keep app import fast and allow a clean fallback.
    from transformers import pipeline  # type: ignore

    return pipeline(
        "sentiment-analysis",
        model=model_name,
        tokenizer=model_name,
        device=-1,  # CPU
        top_k=None,  # return all scores
    )


def score_texts(
    texts: Iterable[str],
    model_name: str = DEFAULT_MODEL,
    neutral_band: float = 0.10,
) -> List[Tuple[str, float, float]]:
    """
    Returns list of (label, score, confidence) aligned with input.

    score = P(positive) - P(negative) in [-1, 1]
    label uses argmax, with a neutral band around 0 to avoid overconfident polarity.
    """
    items = ["" if t is None else str(t) for t in texts]
    if not items:
        return []

    clf = _get_pipeline(model_name=model_name)
    # transformers pipeline returns: List[List[{label, score}, ...]]
    all_scores = clf(items)
    out: List[Tuple[str, float, float]] = []

    for per_text in all_scores:
        probs = { _normalize_label(d.get("label", "")): float(d.get("score", 0.0)) for d in per_text }
        p_pos = probs.get("Positive", 0.0)
        p_neg = probs.get("Negative", 0.0)
        p_neu = probs.get("Neutral", 0.0)

        score = max(-1.0, min(1.0, p_pos - p_neg))
        confidence = max(p_pos, p_neg, p_neu)

        if abs(score) < neutral_band:
            label = "Neutral"
        else:
            label = "Positive" if p_pos >= p_neg else "Negative"

        out.append((label, score, confidence))

    return out

