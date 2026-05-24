# Bitext Customer Service Data Analyst Agent (Assignment 3)

A LangGraph ReAct agent that answers analytical questions about the [Bitext customer-support training dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset).

**Due: 29.5.26**

## Prerequisites

- Python 3.11+
- A [Nebius Token Factory](https://studio.nebius.ai/) API key
- Hugging Face account (optional — speeds up dataset download)

## Quick start (~5 minutes)

### 1. Clone and install

```bash
git clone <repo-url>
cd bitext-customer-analyst-agent
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS
pip install -r requirements.txt
```

### 2. Environment

```bash
cp .env.example .env
# Edit .env and set NEBIUS_API_KEY=your_actual_key
```

### 3. Download and preprocess data (required once)

**Windows (all steps in one script):**

```powershell
$env:HF_TOKEN = "hf_your_token"   # optional but faster
.\scripts\run_setup.ps1
```

**Or manually:**

```bash
python -m src.data.preprocess
python scripts/profile_data.py
python main.py --check
```

If the download stalls at 0%, the preprocessor retries via direct CSV download.

This writes:
- `data/raw/bitext_raw.parquet` — raw snapshot from Hugging Face
- `data/processed/bitext_clean.parquet` — cleaned data (24,635 rows)
- `data/processed/metadata.json` — categories, intents, counts
- `data/processed/quality_report.json` — known dataset limitations

### 4. Run the agent

```bash
python main.py --session demo
```

The CLI drops into an interactive loop. **Reasoning steps (tool calls + observations) are printed before each answer.** Type `quit` to exit.

Check data only (no LLM):

```bash
python main.py --check
```

### 5. Example queries to try

```
What categories exist in the dataset?
How many refund requests did we get?
Show me 5 examples from the SHIPPING category.
Summarize how agents respond to complaint intents.
Show me examples of people wanting their money back.
What is the distribution of intents in the ACCOUNT category?
What's the best CRM software for handling complaints?   ← out-of-scope
Who is the president of France?                         ← out-of-scope
What do you remember about me?                          ← user profile
```

### 6. Optional: Streamlit UI (Bonus A)

A chat interface with visible reasoning steps and session switching:

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501). Use the **Session ID** field in the sidebar to switch between or resume conversations.

### 7. MCP server

Start the server:

```bash
python -m src.mcp.server
```

**Connecting a client (Claude Desktop / Cursor):**

Add to your MCP client config:

```json
{
  "mcpServers": {
    "bitext-analyst": {
      "command": "python",
      "args": ["-m", "src.mcp.server"],
      "cwd": "/absolute/path/to/bitext-customer-analyst-agent"
    }
  }
}
```

**Calling a tool directly with Python:**

```python
from fastmcp import Client
import asyncio

async def main():
    async with Client("python -m src.mcp.server") as client:
        result = await client.call_tool("count_rows_tool", {"intent": "GET_REFUND"})
        print(result)

asyncio.run(main())
```

### 8. Optional: data augmentation scripts

```bash
# Fill {{placeholder}} tokens with realistic Faker values (recommended first step)
python scripts/fill_placeholders.py

# Generate synthetic 3–5 turn conversations from Q&A pairs
python scripts/augment_conversations.py --per-intent 3 --max-intents 10

# Generate paraphrase variants for search
python scripts/augment_paraphrases.py --max-intents 5

# Export router train/test split
python scripts/export_router_split.py
```

---

## Architecture overview

### Models

Two Nebius Token Factory models with different roles:

| Role | Model (`ROUTER_MODEL` / `AGENT_MODEL`) | Why |
|------|----------------------------------------|-----|
| **Router** | `google/gemma-3-27b-it` | Fast, cheap, reliable JSON output at `temperature=0`. Routing is a narrow 4-class task (structured / unstructured / recommend / out_of_scope) — a 70B model would be overkill and 3× slower. |
| **Agent** | `meta-llama/Llama-3.3-70B-Instruct` | Strong multi-step reasoning and tool-use. Handles chained tool calls, interprets varied outputs, and writes coherent summaries. Open weights, 128k context. |
| **Profile updater** | `google/gemma-3-27b-it` (reused) | Profile extraction is a simple JSON rewrite task — same small model is sufficient and keeps costs low. |

See `src/agent/llm.py` for detailed pros/cons of each model choice.  
Run `python scripts/list_nebius_models.py` to list models available in your account.

### Graph (LangGraph)

```
User input
    │
    ▼
[route node]  ← Gemma 3 27B classifies: structured / unstructured / recommend / out_of_scope
    │
    ├── out_of_scope → polite decline (no LLM general-knowledge answer)
    │
    └── structured / unstructured / recommend
            │
            ▼
    [agent node]  ← Llama 3.3 70B ReAct loop (max 12 iterations)
         Tool calls → observations → reasoning → final answer
            │
            ▼
    Print reasoning steps to CLI
    Update user profile (Gemma 27B)
    Persist state via SQLite checkpoint
```

### Tools

All tools have Pydantic input schemas with `Field` descriptions so the LLM knows exactly when and how to use each one.

| Tool | Purpose |
|------|---------|
| `list_categories` | List all 11 dataset categories |
| `list_intents` | List intents (optionally scoped to a category) |
| `count_rows` | Count rows with category / intent / keyword filters |
| `distribution_by_category` | Row-count histogram by category |
| `distribution_by_intent` | Row-count histogram by intent (global or per-category) |
| `filter_records` | Return sample rows (instruction + response) matching filters |
| `search_instructions` | Free-text keyword search over instructions + augmented paraphrases |
| `dataset_summary` | High-level stats + quality limitations summary |
| `suggest_next_query` | **Bonus B** — suggest follow-up queries; waits for user confirmation before executing |

Keyword alias resolution maps natural phrases to canonical values  
(e.g. "money back" → `GET_REFUND`, "cancel" → `CANCEL_ORDER`).

### Memory

- **Episodic (Task 2a)**: LangGraph `SqliteSaver` checkpoints at `data/checkpoints.sqlite`. Pass `--session <id>` to resume the same conversation after restart.
- **User profile (Task 2b)**: Per-session JSON at `data/profiles/<session_id>.json`. Captures name, topics of interest, preferences. Injected into the system prompt each turn and updated after each reply using the router model.

### Bonus

- **Bonus A — Streamlit UI** (`app.py`): `streamlit run app.py` — chat interface with expandable reasoning steps and session ID sidebar.
- **Bonus B — Query Recommender** (`suggest_next_query` tool): type "What should I query next?" — the agent suggests queries based on what you've discussed, waits for confirmation or refinement, then executes only on "yes / go ahead / do it".

---

## Project layout

```
src/data/      preprocess.py, loader.py, quality.py
src/tools/     analytics.py (10 tools with Pydantic schemas)
src/agent/     graph.py (LangGraph), router.py, llm.py
src/memory/    checkpoint.py (SQLite episodic), profile.py (user profile)
src/mcp/       server.py (FastMCP — 10 tools)
scripts/       evaluate.py, fill_placeholders, augment_conversations, augment_paraphrases, ...
data/eval/     golden_set.json (14 test cases), last_run.json (latest eval results)
main.py        CLI entry point
app.py         Streamlit UI (Bonus A)
```

---

## Evaluation (EDD)

The project follows **Evaluation-Driven Development**: define expected behaviour first, then measure it.

### What is evaluated

| Dimension | How | Applies to |
|-----------|-----|-----------|
| **Router accuracy** | Expected vs actual route | All |
| **Tool selection** | At least one expected tool called | Structured / unstructured |
| **Answer content** | Response contains ground-truth strings (e.g. exact counts) | Structured |
| **OOS guard** | Response does NOT contain leaked knowledge | Out-of-scope |
| **LLM judge** | Gemma 27B scores relevance + correctness + helpfulness (0–9) | Unstructured (opt-in) |

### Running the evaluation

```bash
# Quick check — no LLM judge (fast, no extra token cost)
python scripts/evaluate.py

# Full evaluation with LLM judge for unstructured answers
python scripts/evaluate.py --judge

# Smoke test — first 5 cases only
python scripts/evaluate.py --max-cases 5

# Save raw results to a file for analysis
python scripts/evaluate.py --out data/eval/my_run.json
```

### Golden test set (`data/eval/golden_set.json`)

14 cases that cover every query type:

| ID | Query | Type | Ground truth |
|----|-------|------|-------------|
| struct_01 | "How many refund requests?" | structured | 918 rows (GET_REFUND) |
| struct_02 | "What categories exist?" | structured | lists all 11 categories |
| struct_03 | "How many CANCEL_ORDER rows?" | structured | 493 rows |
| struct_04 | "Show 3 examples from SHIPPING" | structured | SHIPPING category examples |
| struct_05 | "Distribution of intents in ACCOUNT?" | structured | 6 ACCOUNT intents |
| struct_06 | "How many intents total?" | structured | 27 |
| alias_01 | "People wanting their money back" | structured | alias → GET_REFUND |
| alias_02 | "Rows mentioning password problems" | structured | keyword → RECOVER_PASSWORD |
| multi_01 | "Complaints + refunds, total?" | structured | 1000 + 918 = 1918 |
| unstruct_01 | "Summarize cancellation responses" | unstructured | LLM judge |
| unstruct_02 | "Summarize FEEDBACK category" | unstructured | LLM judge |
| oos_01 | "Who is the president of France?" | out_of_scope | must decline |
| oos_02 | "Best CRM software?" | out_of_scope | must decline (domain-adjacent trap) |
| recommend_01 | "What should I query next?" | recommend | calls suggest_next_query |

### EDD development loop

```
1. Run evaluate.py  →  identify failing cases
2. Inspect failure  →  is it router? wrong tool? bad alias? missing alias?
3. Fix the issue    →  improve tool description, add alias, fix router prompt
4. Re-run           →  confirm the case now passes, no regressions
5. Add new test cases as you discover new failure modes
```

## Known data limitations

The Bitext dataset is **template-based synthetic data** — not real customer conversations.

| Limitation | Detail |
|------------|--------|
| 43.9% placeholder rate | `{{name}}`, `{{order_id}}` tokens are never filled in |
| Single-turn only | Each row = 1 question + 1 answer, no multi-turn dialogue |
| Intent imbalance | `CANCEL_ORDER` (493 rows) vs `COMPLAINT` (1000) — 2× gap |
| No real user noise | No typos, emotional escalation, or mixed intents |

Optional scripts to improve data quality:
```bash
python scripts/augment_paraphrases.py --max-intents 5   # generate paraphrase variants
python scripts/export_router_split.py                    # train/test split for router
```
