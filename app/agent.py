from __future__ import annotations

from dataclasses import dataclass, field

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL
from app.retrievers import RetrievalResult, Retriever


@dataclass
class DemoDeps:
    retriever: Retriever
    metrics: list[RetrievalResult] = field(default_factory=list)


model = OpenAIChatModel(
    model_name=OLLAMA_MODEL,
    provider=OllamaProvider(base_url=OLLAMA_BASE_URL),
)

agent = Agent(
    model,
    deps_type=DemoDeps,
    system_prompt=(
        "You are a concise RAG demo assistant. Always use the search_knowledge_base tool before answering. "
        "Answer in plain English using only the retrieved facts. "
        "Do not mention internal document IDs, token labels, or bracketed codes unless the user explicitly asks for them. "
        "Keep the answer to 2 short sentences and explain what TurboAgents changes at the retrieval layer."
    ),
)


@agent.tool
async def search_knowledge_base(ctx: RunContext[DemoDeps], question: str) -> str:
    result = await ctx.deps.retriever.search(question)
    ctx.deps.metrics.append(result)
    return "\n".join(result.snippets)
