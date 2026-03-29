from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent import DemoDeps, run_grounded
from app.config import QUESTION
from app.retrievers import BaselineSurrealRetriever

RESET = "\033[0m"
BOLD = "\033[1m"
BLUE = "\033[94m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"


def style(text: str, *codes: str) -> str:
    return "".join(codes) + text + RESET


async def main() -> None:
    print(style("Plain Pydantic AI + SurrealDB RAG", BOLD, GREEN))
    print(style("This is the baseline app without TurboAgents.", YELLOW))

    retriever = BaselineSurrealRetriever()
    print(style("Preparing baseline SurrealDB retriever...", YELLOW))
    await retriever.prepare()

    deps = DemoDeps(retriever=retriever)
    started = time.perf_counter()
    result = await run_grounded(QUESTION, deps=deps)
    total_ms = (time.perf_counter() - started) * 1000
    if not deps.metrics:
        raise RuntimeError("The model did not call search_knowledge_base after two attempts.")
    retrieval = deps.metrics[-1]

    print(style("Question:", BOLD, CYAN), QUESTION)
    print(style("Answer:", BOLD, CYAN), result.output)
    print(style("Retriever mode:", BOLD, CYAN), retrieval.mode)
    print(style("Retrieval time:", BOLD, CYAN), f"{retrieval.elapsed_ms:.2f} ms")
    print(style("Agent total time:", BOLD, CYAN), f"{total_ms:.2f} ms")
    print(style("Vector storage:", BOLD, CYAN), f"raw float32 only ({retrieval.raw_vector_bytes} bytes per vector)")
    print(style("Top snippets:", BOLD, CYAN))
    for snippet in retrieval.snippets:
        print(f"- {snippet}")


if __name__ == "__main__":
    asyncio.run(main())
