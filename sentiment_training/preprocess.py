"""
Text cleaning and tokenization for downstream vectorization / modeling.

Designed to stay lightweight (regex + stdlib). Swap in NLTK/spaCy later if needed.
"""

from __future__ import annotations

import re
from typing import Iterable, List

# URLs, handles, repeated whitespace
_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"@[\w_]+")
_NON_WORD = re.compile(r"[^a-z\s]")
_WS = re.compile(r"\s+")
# Word tokens: lowercase letters, min length 2 (adjust for your task)
_TOKEN = re.compile(r"\b[a-z]{2,}\b")


def clean_text(text: object) -> str:
    """
    Lowercase, remove URLs and @mentions, strip non-letters, collapse whitespace.

    Parameters
    ----------
    text : object
        Raw sentence (often Twitter-style lowercased noisy text).
    """
    try:
        import pandas as pd

        if pd.isna(text):
            return ""
    except Exception:
        pass
    if text is None:
        return ""
    s = str(text).lower()
    s = _URL.sub(" ", s)
    s = _MENTION.sub(" ", s)
    s = _NON_WORD.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s


def tokenize(cleaned: str) -> List[str]:
    """Split cleaned text into lowercase word tokens."""
    if not cleaned:
        return []
    return _TOKEN.findall(cleaned)


def tokenize_batch(texts: Iterable[str]) -> List[List[str]]:
    return [tokenize(t) for t in texts]


def apply_clean_tokenize(df, text_col: str = "sentence") -> "pd.DataFrame":
    """
    Add ``text_clean`` and ``tokens`` columns (tokens as string lists stored as object dtype).

    Returns a copy.
    """
    import pandas as pd

    out = df.copy()
    out["text_clean"] = out[text_col].map(clean_text)
    out["tokens"] = out["text_clean"].map(lambda s: tokenize(s))
    return out
