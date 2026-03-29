from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent import DemoDeps, agent
from app.config import QUESTION, TURBO_BITS
from app.retrievers import TurboSurrealRetriever

RESET = "\033[0m"
BOLD = "\033[1m"
BLUE = "\033[94m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"


def style(text: str, *codes: str) -> str:
    return "".join(codes) + text + RESET


async def main() -> None:
    print(style(f"Pydantic AI + SurrealDB + TurboAgents ({TURBO_BITS} bits)", BOLD, GREEN))
    print(style("This is the same app with the TurboAgents retrieval layer added.", YELLOW))

    retriever = TurboSurrealRetriever()
    print(style("Preparing TurboAgents-backed SurrealDB retriever...", YELLOW))
    await retriever.prepare()

    deps = DemoDeps(retriever=retriever)
    started = time.perf_counter()
    result = await agent.run(QUESTION, deps=deps)
    total_ms = (time.perf_counter() - started) * 1000
    retrieval = deps.metrics[-1]

    print(style("Question:", BOLD, CYAN), QUESTION)
    print(style("Answer:", BOLD, CYAN), result.output)
    print(style("Retriever mode:", BOLD, CYAN), retrieval.mode)
    print(style("Retrieval time:", BOLD, CYAN), f"{retrieval.elapsed_ms:.2f} ms")
    print(style("Agent total time:", BOLD, CYAN), f"{total_ms:.2f} ms")
    if retrieval.compressed_bytes is not None:
        ratio = retrieval.raw_vector_bytes / retrieval.compressed_bytes
        print(
            style("Vector storage:", BOLD, CYAN),
            f"raw={retrieval.raw_vector_bytes} bytes, turbo={retrieval.compressed_bytes} bytes, compression≈{ratio:.2f}x",
        )
    print(style("Top snippets:", BOLD, CYAN))
    for snippet in retrieval.snippets:
        print(f"- {snippet}")


if __name__ == "__main__":
    asyncio.run(main())
