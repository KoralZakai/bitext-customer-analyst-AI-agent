<<<<<<< HEAD
# Bitext Customer Service Data Analyst Agent (Assignment 3)

LangGraph ReAct agent over the Bitext customer-support dataset with preprocessing, router, tools, memory, and FastMCP.

## Quick start (~5 minutes)

### 1. Clone and install

```bash
cd bitext-customer-analyst-agent
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 2. Environment

Copy `.env.example` to `.env` and set your Nebius API key:

```
NEBIUS_API_KEY=your_actual_key
```

Only **Nebius Token Factory** models are used for LLM calls (see `ROUTER_MODEL` / `AGENT_MODEL`).

### 3. Download and preprocess data (required)

**Windows (all 3 steps in one script):**

```powershell
cd bitext-customer-analyst-agent
$env:HF_TOKEN = "hf_your_token"   # optional; revokes old tokens if leaked
.\scripts\run_setup.ps1
```

**Or run each step manually:**

```powershell
.\.venv\Scripts\python -m src.data.preprocess
.\.venv\Scripts\python scripts\profile_data.py
.\.venv\Scripts\python main.py --check
```

If `load_dataset` fails at 0%, preprocess automatically retries via direct CSV download from Hugging Face Hub.

This writes:

- `data/raw/bitext_raw.parquet` — snapshot from Hugging Face
- `data/processed/bitext_clean.parquet` — cleaned data for all tools
- `data/processed/metadata.json` — categories, intents, counts, quality block
- `data/processed/quality_report.json` — dataset limitations report

### 4. Run the agent

```bash
python main.py --session demo
```

Check data only (no LLM):

```bash
python main.py --check
```

### 5. Optional: MCP server

```bash
python -m src.mcp.server
```

### 6. Optional: synthetic search paraphrases / router split

```bash
python scripts/augment_paraphrases.py --max-intents 5
python scripts/export_router_split.py
```

## Preprocessing (why it exists)

- Strip whitespace and normalize `category` / `intent` to uppercase
- Drop rows with missing key fields
- Deduplicate on `(instruction, category, intent)`
- Categorical dtypes for faster filters
- `source=real` column; synthetic rows stay in separate augmented parquet
- `data/intent_aliases.json` maps paraphrases like "money back" → `GET_REFUND`

## Project layout

```
src/data/     preprocess.py, loader.py, quality.py
src/tools/    analytics tools
src/agent/    LangGraph router + ReAct
src/memory/   SQLite checkpoints
src/mcp/      FastMCP server
scripts/      profile_data, augment_paraphrases, export_router_split
main.py       CLI
```

## Models

| Role | Env var | Purpose |
|------|---------|---------|
| Router | `ROUTER_MODEL` | Fast structured / unstructured / OOS classification |
| Agent | `AGENT_MODEL` | Tool planning and summarization |

Adjust model IDs to those enabled in your Nebius Token Factory account.
=======
# bitext-customer-analyst-AI-agent
AI agent that knows how to respponse obver bitext-customer data
>>>>>>> 7544d8b52d7bd49502e027384838d312a990c017
