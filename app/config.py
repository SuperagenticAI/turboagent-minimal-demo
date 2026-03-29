from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"
OLLAMA_MODEL = "qwen3.5:9b"
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DATA_DIR = Path(__file__).resolve().parents[1] / "demo_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
SURREAL_URL = f"surrealkv://{DATA_DIR / 'minimal_rag.kv'}"
SURREAL_AUTH = None
SURREAL_NAMESPACE = "turboagent_demo"
SURREAL_DATABASE = "minimal_rag"
EMBED_DIM = 256
TURBO_BITS = 3.5
QUESTION = "What is the seeded retrieval token NEON-FOX-742, and what does TurboAgents change in this SurrealDB-based agent?"


@dataclass(frozen=True)
class Doc:
    title: str
    content: str
    token: str


CORPUS = [
    Doc(
        title="TurboAgents Overview",
        content="TurboAgents adds a TurboQuant-style compressed retrieval and reranking layer on top of an existing vector store.",
        token="TURBO-OVERVIEW-001",
    ),
    Doc(
        title="SurrealDB Baseline",
        content="The baseline demo uses plain SurrealDB vector search so the before-and-after integration seam stays easy to read.",
        token="BASELINE-SEARCH-002",
    ),
    Doc(
        title="Grounding Token: NEON-FOX-742",
        content="NEON-FOX-742 is the seeded retrieval token in this demo. If a user asks what the seeded retrieval token NEON-FOX-742 is, the correct answer is that NEON-FOX-742 is a seeded retrieval token stored in the SurrealDB knowledge base and used to verify grounded answers.",
        token="NEON-FOX-742",
    ),
    Doc(
        title="Why SurrealDB Here",
        content="This demo uses SurrealDB because LanceDB already has its own quantization story. SurrealDB makes the TurboAgents retrieval-layer change easier to isolate, but this document is not the definition of NEON-FOX-742.",
        token="SURREAL-CHOICE-003",
    ),
    Doc(
        title="Minimal Code Change",
        content="The point of the demo is that the agent wiring stays almost the same while the retriever changes from baseline SurrealDB search to TurboAgents-backed SurrealDB search. This document explains the code change, not the meaning of NEON-FOX-742.",
        token="CODE-CHANGE-004",
    ),
    Doc(
        title="Unrelated Note",
        content="This document exists to make retrieval slightly less trivial and give the search path a few distractors.",
        token="DISTRACTOR-005",
    ),
]
