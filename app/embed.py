from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import EMBED_DIM, EMBED_MODEL


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    model = SentenceTransformer(EMBED_MODEL, trust_remote_code=True, truncate_dim=EMBED_DIM)
    actual_dim = model.get_sentence_embedding_dimension()
    if actual_dim != EMBED_DIM:
        raise ValueError(
            f"Embedding dimension mismatch: model returned {actual_dim}, expected {EMBED_DIM}."
        )
    return model


def embed_text(text: str, dim: int = EMBED_DIM) -> np.ndarray:
    model = get_embedder()
    vector = model.encode(text, normalize_embeddings=True, truncate_dim=dim)
    arr = np.asarray(vector, dtype=np.float32)
    if arr.shape != (dim,):
        raise ValueError(f"Expected embedding shape {(dim,)}, got {arr.shape}.")
    return arr


def embed_texts(texts: list[str], dim: int = EMBED_DIM) -> np.ndarray:
    model = get_embedder()
    matrix = model.encode(texts, normalize_embeddings=True, truncate_dim=dim)
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != dim:
        raise ValueError(f"Expected embedding matrix shape (n, {dim}), got {arr.shape}.")
    return arr
