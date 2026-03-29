from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent import DemoDeps, run_grounded
from app.config import QUESTION, TURBO_BITS
from app.retrievers import BaselineSurrealRetriever, RetrievalResult, TurboSurrealRetriever

RESET = "\033[0m"
BOLD = "\033[1m"
BLUE = "\033[94m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"


def style(text: str, *codes: str) -> str:
    return "".join(codes) + text + RESET


def section(title: str) -> None:
    print()
    print(style(f"== {title} ==", BOLD, BLUE))


def display_snippets(snippets: list[str]) -> list[str]:
    prioritized = [snippet for snippet in snippets if "NEON-FOX-742" in snippet]
    remaining = [snippet for snippet in snippets if "NEON-FOX-742" not in snippet]
    return prioritized + remaining


async def run_scenario(label: str, retriever) -> tuple[str, RetrievalResult, float]:
    section(label)
    print(style("Preparing retriever and seeding SurrealDB records...", YELLOW))
    await retriever.prepare()
    deps = DemoDeps(retriever=retriever)
    started = time.perf_counter()
    result = await run_grounded(QUESTION, deps=deps)
    agent_ms = (time.perf_counter() - started) * 1000
    if not deps.metrics:
        raise RuntimeError("The model did not call search_knowledge_base after two attempts.")
    retrieval = deps.metrics[-1]
    print(style("Question:", BOLD, CYAN), QUESTION)
    print(style("Answer:", BOLD, CYAN), result.output)
    print(style("Retriever mode:", BOLD, CYAN), retrieval.mode)
    print(style("Retrieval time:", BOLD, CYAN), f"{retrieval.elapsed_ms:.2f} ms")
    print(style("Agent total time:", BOLD, CYAN), f"{agent_ms:.2f} ms")
    print(style("Top snippets:", BOLD, CYAN))
    for snippet in display_snippets(retrieval.snippets):
        print(f"- {snippet}")
    if retrieval.compressed_bytes is None:
        print(style("Vector storage:", BOLD, CYAN), f"raw float32 only ({retrieval.raw_vector_bytes} bytes per vector)")
    else:
        ratio = retrieval.raw_vector_bytes / retrieval.compressed_bytes
        print(
            style("Vector storage:", BOLD, CYAN),
            f"raw={retrieval.raw_vector_bytes} bytes, turbo={retrieval.compressed_bytes} bytes, compression≈{ratio:.2f}x",
        )
    return result.output, retrieval, agent_ms


async def main() -> None:
    print(style("Minimal Pydantic AI + SurrealDB + TurboAgents demo", BOLD, GREEN))
    print(style("This compares a plain SurrealDB RAG path with a TurboAgents-enhanced SurrealDB path.", YELLOW))

    _, baseline, _ = await run_scenario("Baseline SurrealDB", BaselineSurrealRetriever())
    _, turbo, _ = await run_scenario(f"TurboAgents SurrealDB ({TURBO_BITS} bits)", TurboSurrealRetriever())

    section("Comparison")
    print(style("What changed in the app:", BOLD, CYAN), "only the retriever implementation")
    print(style("Baseline mode:", BOLD, CYAN), baseline.mode)
    print(style("Turbo mode:", BOLD, CYAN), turbo.mode)
    if turbo.compressed_bytes is not None:
        ratio = turbo.raw_vector_bytes / turbo.compressed_bytes
        print(style("Compression gain:", BOLD, CYAN), f"about {ratio:.2f}x smaller rerank payload per vector")
    print(style("Latency note:", BOLD, CYAN), "this small demo is for integration clarity; storage/compression is the main visible win.")
    print(style("Conclusion:", BOLD, GREEN), "same agent flow, compressed retrieval payload, and only a retriever-level code change.")


if __name__ == "__main__":
    asyncio.run(main())
