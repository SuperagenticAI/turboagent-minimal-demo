# TurboAgents Minimal Demo

A small standalone demo that starts with a plain **Pydantic AI + SurrealDB** RAG app and then adds **TurboAgents** on top of the same retrieval flow.

The point of this repo is simple:

- build a normal agent first
- add TurboAgents second
- keep the code change easy to see
- show what improves

## What you will see

This repo gives you three ways to run the demo:

1. **Plain SurrealDB RAG**
   A simple Pydantic AI agent using a plain SurrealDB retriever.

2. **TurboAgents SurrealDB RAG**
   The same agent, but the retriever is replaced with a TurboAgents-backed SurrealDB path.

3. **Side-by-side comparison**
   A script that runs both versions and prints the difference.

## Before vs After

| Version | What stays the same | What changes |
| --- | --- | --- |
| Plain SurrealDB RAG | Pydantic AI agent, question, documents, local model | Uses plain SurrealDB vector search |
| TurboAgents SurrealDB RAG | Pydantic AI agent, question, documents, local model | Uses TurboAgents for compressed retrieval and reranking on top of SurrealDB |

## Why SurrealDB in this demo

This demo uses **SurrealDB** because it gives a cleaner before-and-after story.

LanceDB is also supported by TurboAgents, but LanceDB already has its own quantization and indexing story. For a minimal integration demo, SurrealDB makes the TurboAgents addition easier to isolate.

## What TurboAgents is doing here

TurboAgents is **not** replacing the whole app.

It is only changing the retrieval layer.

That means the demo goes from this:

```text
Pydantic AI -> SurrealDB search -> answer
```

to this:

```text
Pydantic AI -> TurboAgents + SurrealDB search -> answer
```

In this repo, TurboAgents adds a **TurboQuant-style compressed retrieval and reranking layer** on top of SurrealDB.

## What this demo shows

- a plain SurrealDB retriever feeding a small Pydantic AI agent
- the same agent with a TurboAgents-backed SurrealDB retriever
- the same question asked both times
- retrieval timing for both runs
- raw vector bytes versus TurboAgents compressed payload bytes

## Tools you need

You need these tools installed before running the demo:

- **Python 3.11+**
- **uv**
- **Ollama**
- **Docker**

### What each tool is for

- **uv** installs and runs the Python environment for this repo
- **Ollama** runs the local language model used by the agent
- **Docker** is useful for local database and AI workflows in general, even though this demo uses the embedded `surrealkv://` backend and does not require a separate SurrealDB server process

## Prerequisites

Make sure these local services are available:

- Ollama is running on `http://127.0.0.1:11434`
- the Ollama model `qwen3.5:9b` is already pulled

This demo uses the local embedded `surrealkv://` backend from the SurrealDB Python SDK, so you do **not** need to run a separate SurrealDB server for it.

## Quick start

```bash
git clone <your-demo-repo-url>
cd turboagent-minimal-demo
uv sync
```

If you do not have `uv` yet:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

If you do not have Ollama yet, install it first and then make sure the model is available:

```bash
ollama pull qwen3.5:9b
ollama list
```

## Run

Plain baseline app:

```bash
uv run python scripts/run_plain_rag.py
```

TurboAgents-enhanced app:

```bash
uv run python scripts/run_turbo_rag.py
```

Side-by-side comparison:

```bash
uv run python scripts/run_compare.py
```

## Project layout

- `app/config.py`
  Shared configuration, demo question, and sample documents.

- `app/embed.py`
  A tiny local embedding function used to keep the demo self-contained.

- `app/retrievers.py`
  Contains both retrievers:
  - the plain SurrealDB retriever
  - the TurboAgents-backed SurrealDB retriever

- `app/agent.py`
  The shared Pydantic AI agent used by both versions.

- `scripts/run_plain_rag.py`
  Runs the plain baseline app.

- `scripts/run_turbo_rag.py`
  Runs the TurboAgents-enhanced app.

- `scripts/run_compare.py`
  Runs both versions and prints the comparison.

## What to look for

The app code stays almost the same.

The retriever changes, and the TurboAgents path adds a compressed retrieval and reranking layer.

For this minimal demo, the clearest measurable benefit is:

- **smaller compressed rerank payloads**
- not a blanket claim that every run will be faster end to end

That is why the output shows:

- answer quality
- retrieval mode
- retrieval time
- raw vector bytes
- compressed vector bytes

## Troubleshooting

### `uv: command not found`

Install `uv` first:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### `Connection refused` or Ollama errors

Make sure Ollama is running:

```bash
ollama list
```

If `qwen3.5:9b` is missing:

```bash
ollama pull qwen3.5:9b
```

### First run feels slow

That is normal.

The first model call may take longer because the local model needs to warm up.

### Why does this use fake embeddings?

This repo is a **minimal integration demo**. The point is to make the TurboAgents change easy to understand, not to build a production embedding pipeline.

## Summary

If you want the shortest explanation of this repo, it is this:

- run a simple RAG app
- swap the retriever
- keep the agent the same
- see how TurboAgents adds compressed retrieval on top of SurrealDB
