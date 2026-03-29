from __future__ import annotations

import re

import numpy as np

from app.config import EMBED_DIM

KEYWORDS = {
    "turboagents": 0,
    "turboquant": 0,
    "compressed": 1,
    "compression": 1,
    "retrieval": 2,
    "reranking": 3,
    "surrealdb": 4,
    "agent": 5,
    "pydantic": 6,
    "grounding": 7,
    "neon": 8,
    "fox": 8,
    "742": 8,
    "neon-fox-742": 8,
    "baseline": 9,
    "token": 10,
    "vector": 11,
}


def embed_text(text: str, dim: int = EMBED_DIM) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    tokens = re.findall(r"[a-zA-Z0-9-]+", text.lower())
    for token in tokens:
        if token in KEYWORDS:
            vec[KEYWORDS[token]] += 1.0
        else:
            slot = 12 + (hash(token) % max(1, dim - 12))
            vec[slot] += 0.05
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def embed_texts(texts: list[str], dim: int = EMBED_DIM) -> np.ndarray:
    return np.vstack([embed_text(text, dim=dim) for text in texts]).astype(np.float32)
