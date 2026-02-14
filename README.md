# Petri Example

Minimal examples of [Petri](https://github.com/safety-research/petri), an alignment auditing framework for LLMs.

Petri uses a multi-agent architecture to test whether AI models exhibit concerning behaviors:

- **Auditor** — receives a hypothesis ("seed instruction") and autonomously designs a realistic multi-turn conversation to probe the target
- **Target** — the model being evaluated (doesn't know it's being tested)
- **Judge** — reads the transcript and scores it for concerning behavior

## Setup

```bash
uv sync
cp .env.example .env
# Fill in your API keys in .env
```

## Usage

### Simple audit (`simple_audit.py`)

Runs 3 seed instructions with default settings:

```bash
uv run inspect eval simple_audit.py \
  --model-role auditor=openai/gpt-4o \
  --model-role target=openai/gpt-4o-mini \
  --model-role judge=openai/gpt-4o
```

### Custom audit (`custom_audit.py`)

Supports filtering by sample ID and configuring max turns:

```bash
# Run a single test
uv run inspect eval custom_audit.py \
  --model-role auditor=openai/gpt-4o \
  --model-role target=anthropic/claude-sonnet-4-20250514 \
  --model-role judge=openai/gpt-4o \
  -T sample_ids=sycophancy

# More turns for deeper probing
uv run inspect eval custom_audit.py \
  --model-role auditor=openai/gpt-4o \
  --model-role target=openai/gpt-4o-mini \
  --model-role judge=openai/gpt-4o \
  -T max_turns=20
```

## Viewing results

```bash
uv run inspect view
```

This opens a local web UI to browse transcripts and scores.
