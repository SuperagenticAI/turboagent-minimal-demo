from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from surrealdb import AsyncSurreal

from app.config import (
    CORPUS,
    EMBED_DIM,
    SURREAL_AUTH,
    SURREAL_DATABASE,
    SURREAL_NAMESPACE,
    SURREAL_URL,
    TURBO_BITS,
)
from app.embed import embed_text, embed_texts
from turboagents.quant import Config, quantize
from turboagents.rag import TurboSurrealDB


@dataclass
class RetrievalResult:
    snippets: list[str]
    elapsed_ms: float
    raw_vector_bytes: int
    compressed_bytes: int | None
    mode: str


class Retriever(Protocol):
    async def prepare(self) -> None: ...
    async def search(self, question: str, *, k: int = 3) -> RetrievalResult: ...


class BaselineSurrealRetriever:
    def __init__(self) -> None:
        self.client = AsyncSurreal(SURREAL_URL)
        self.table = f"baseline_docs_{uuid.uuid4().hex[:8]}"
        self._prepared = False
        self._vectors = embed_texts([f"{doc.title} {doc.content} {doc.token}" for doc in CORPUS])

    async def prepare(self) -> None:
        if self._prepared:
            return
        await self.client.connect()
        if SURREAL_AUTH:
            await self.client.signin(SURREAL_AUTH)
        await self.client.use(SURREAL_NAMESPACE, SURREAL_DATABASE)
        await self.client.query(f"DEFINE TABLE {self.table} SCHEMALESS;")
        await self.client.query(
            f"DEFINE INDEX {self.table}_embedding_idx ON {self.table} "
            f"FIELDS embedding HNSW DIMENSION {EMBED_DIM} DIST COSINE TYPE F32 EFC 150 M 8;"
        )
        for idx, (doc, vec) in enumerate(zip(CORPUS, self._vectors, strict=True)):
            await self.client.create(
                f"{self.table}:{idx}",
                {
                    "title": doc.title,
                    "content": doc.content,
                    "token": doc.token,
                    "embedding": vec.tolist(),
                },
            )
        self._prepared = True

    async def search(self, question: str, *, k: int = 3) -> RetrievalResult:
        query = embed_text(question)
        started = time.perf_counter()
        rows = await self.client.query(
            (
                f"SELECT id, title, content, token, vector::distance::knn() AS dist "
                f"FROM {self.table} WHERE embedding <|{k},COSINE|> $query;"
            ),
            {"query": query.tolist()},
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        snippets = [
            f"{row['title']} [{row['token']}]: {row['content']}"
            for row in rows[:k]
        ]
        return RetrievalResult(
            snippets=snippets,
            elapsed_ms=elapsed_ms,
            raw_vector_bytes=int(self._vectors[0].nbytes),
            compressed_bytes=None,
            mode="baseline-surrealdb",
        )


class TurboSurrealRetriever:
    def __init__(self) -> None:
        self.store = TurboSurrealDB(
            url=SURREAL_URL,
            namespace=SURREAL_NAMESPACE,
            database=SURREAL_DATABASE,
            dim=EMBED_DIM,
            bits=TURBO_BITS,
            metric="COSINE",
            auth=SURREAL_AUTH,
        )
        self.collection = f"turbo_docs_{uuid.uuid4().hex[:8]}"
        self._prepared = False
        self._vectors = embed_texts([f"{doc.title} {doc.content} {doc.token}" for doc in CORPUS])
        self._compressed_bytes = len(quantize(self._vectors[0], Config(bits=TURBO_BITS, head_dim=EMBED_DIM)).to_bytes())

    async def prepare(self) -> None:
        if self._prepared:
            return
        await self.store.create_collection(self.collection, dim=EMBED_DIM)
        metadata = [
            {"title": doc.title, "content": doc.content, "token": doc.token}
            for doc in CORPUS
        ]
        await self.store.add(self._vectors, metadata=metadata)
        self._prepared = True

    async def search(self, question: str, *, k: int = 3) -> RetrievalResult:
        query = embed_text(question)
        started = time.perf_counter()
        rows = await self.store.search(query, k=k, rerank_top=max(k, 5))
        elapsed_ms = (time.perf_counter() - started) * 1000
        snippets = [
            f"{row['metadata']['title']} [{row['metadata']['token']}]: {row['metadata']['content']}"
            for row in rows[:k]
        ]
        return RetrievalResult(
            snippets=snippets,
            elapsed_ms=elapsed_ms,
            raw_vector_bytes=int(self._vectors[0].nbytes),
            compressed_bytes=self._compressed_bytes,
            mode=f"turbo-surrealdb-{TURBO_BITS}-bits",
        )
