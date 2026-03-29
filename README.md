# TurboAgents Minimal Demo

A small standalone demo that starts with a plain **Pydantic AI + SurrealDB** RAG app and then adds [TurboAgents](https://github.com/SuperagenticAI/turboagents) on top of the same retrieval flow.

[TurboAgents](https://github.com/SuperagenticAI/turboagents) is a Python package for TurboQuant-style compression, retrieval, and reranking in agent and RAG systems. The main docs are at [superagenticai.github.io/turboagents](https://superagenticai.github.io/turboagents/).

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

## What Changed In The Code

This repo is intentionally small, so the integration seam stays easy to see.

- [`scripts/run_plain_rag.py`](./scripts/run_plain_rag.py) runs the baseline agent with the plain SurrealDB retriever.
- [`scripts/run_turbo_rag.py`](./scripts/run_turbo_rag.py) runs the same agent with the TurboAgents-backed retriever.
- [`app/retrievers.py`](./app/retrievers.py) is where the real swap happens: `BaselineSurrealRetriever` becomes `TurboSurrealRetriever`.
- [`app/agent.py`](./app/agent.py) keeps the same high-level Pydantic AI agent wiring in both versions.

The point of the demo is that TurboAgents changes the retrieval layer, not the rest of the app.

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

### What each tool is for

- **uv** installs and runs the Python environment for this repo
- **Ollama** runs the local language model used by the agent
- **Sentence Transformers** provides the real local embedding model used for retrieval

## Prerequisites

Make sure these local services are available:

- Ollama is running on `http://127.0.0.1:11434`
- the Ollama model `qwen3.5:9b` is already pulled

This demo uses the local embedded `surrealkv://` backend from the SurrealDB Python SDK, so you do **not** need to run a separate SurrealDB server for it.

It also uses the local embedding model `Qwen/Qwen3-Embedding-0.6B`, truncated to `256` dimensions so it stays compatible with TurboAgents. The model will be downloaded on first use.

## Quick start

```bash
git clone https://github.com/SuperagenticAI/turboagent-minimal-demo.git
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

## What happens on first run

When you run the demo for the first time, a few things are created locally:

- the embedding model is downloaded and cached by `sentence-transformers`
- the demo builds embeddings for the small sample corpus
- the embedded `surrealkv://` database file is created under `demo_data/`

Nothing is precomputed in the repo. The demo builds its own local retrieval state so the flow stays easy to inspect and reproduce.

Because of that, the first run may be slower than later runs.

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
  The real local embedding model wrapper used by both retrievers.

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

The first model call may take longer because:

- the local Ollama model may need to warm up
- the embedding model may need to download and load
- the demo is building its local retrieval state for the first time

## Resetting the demo

If you want to rebuild everything from scratch, delete the local demo data and run the scripts again:

```bash
rm -rf demo_data
uv run python scripts/run_compare.py
```

This will rebuild the local SurrealKV database and fresh embeddings for the demo corpus.


## Summary

If you want the shortest explanation of this repo, it is this:

- run a simple RAG app
- swap the retriever
- keep the agent the same
- see how TurboAgents adds compressed retrieval on top of SurrealDB
