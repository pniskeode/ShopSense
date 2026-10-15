## ShopSense agent

This repository contains the ShopSense capstone implementation through M8.

### M2: tool-enabled single agent

M2 provides one natural-language customer-care agent backed by four read-only LangChain tools. Each tool is declared with LangChain's `@tool` decorator and bound to the chat model with `bind_tools()`:

- `lookup_order(order_ref)` joins order, customer, and product data.
- `track_shipment(order_ref)` returns carrier and latest scan information.
- `refund_or_replace(order_ref, action)` returns every matching refund record or explains that replacement is unavailable.
- `get_refund_amount(order_ref)` reports the approved refund total awaiting payout as `amount_inr`, alongside separate totals for every status.

Refund rows are returned deterministically by `requested_on`, then `refund_id`. `approved` amounts are authorized and awaiting payout; `processed` amounts are already refunded. The calculator's `amount_inr` is the approved total, while the status summary keeps processed, requested, and rejected totals distinct. No tool mutates the mock data.

Install the project and its dependencies into the project virtual environment once:

```powershell
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe -m pip install -e .
```

M1 uses `SHOPSENSE_M1_MODEL` and M2 uses `SHOPSENSE_M2_MODEL` from `.env`. `GROQ_API_KEY` supplies their provider key. Keep the real key out of source control; `.env.example` is only a template.

Ask one question from Command Prompt or PowerShell:

```powershell
& .\.venv\Scripts\python.exe -m shopsense.agent.cli "Where is my order KW-O-000123?"
```

Or start the interactive prompt:

```powershell
& .\.venv\Scripts\python.exe -m shopsense.agent.cli
```

Then type a question at `ShopSense>` and type `exit` or `quit` to stop. The agent uses a maximum of six model/tool iterations per question.

Run the tests with:

```powershell
& .\.venv\Scripts\python.exe -m pytest -q
```

See [docs/m2.md](docs/m2.md) for the flow, tool contracts, and edge-case behavior.

### M3 Stage 1: persistent per-customer memory

Stage 1 adds Supermemory-backed long-term customer memory and the Day 2 rolling-summary pattern. Historical ticket memory is built only from the existing M1 intake records and successful M1 parsed results. It is isolated by the intake record's `customer_ref`; M2 `mock_api` data is not used to infer ownership of historical tickets.

Add these keys to `.env` after copying `.env.example`:

```text
SUPERMEMORY_API_KEY=...
GOOGLE_API_KEY=...
SHOPSENSE_M3_SUMMARY_MODEL="gemini/gemini-3.5-flash-lite"
```

Bootstrap the 200 successful historical M1 tickets once:

```powershell
& .\.venv\Scripts\python.exe -m shopsense.agent.cli --bootstrap-memory
```

Ask a question with memory enabled by supplying the verified customer reference explicitly:

```powershell
& .\.venv\Scripts\python.exe -m shopsense.agent.cli --customer-ref KW-C-11111 "What have I contacted support about before?"
```

Memory retrieval and writes are ordinary application functions, not LangChain tools. Each preference write polls Supermemory's extracted-memory layer for up to about one minute; retrieval uses extracted memory first and falls back to source documents only while extraction is pending. The existing four M2 LangChain tools still handle order, shipment, and refund facts. The rolling summary retains the four most recent turns and uses Gemini to compress older session turns after 220 tokens. See [docs/m3-stage1.md](docs/m3-stage1.md).

### M3 Stage 2: Qdrant policy retrieval

Stage 2 indexes the returns, shipping, warranty, category, escalation, fraud, and refund-authority policy corpus in Qdrant Cloud. Every declared policy is loaded from both Markdown and PDF, split with the Day 2 RAG defaults of 900 characters with 120-character overlap, then embedded by Gemini. The policy collection is `shopsense_policy_documents`; its searchable metadata includes policy name, slug, section, and source format.

Add `QDRANT_URL` and `QDRANT_API_KEY` to `.env`, install the updated project dependencies, and run:

```cmd
python -m shopsense.agent.cli --index-policies
```

Policy questions retrieve Qdrant context before the existing M2 agent responds. Current order, shipment, and refund facts still come from the M2 tools. The approved Grocery rule is non-returnable and overrides the conflicting 14-day category-addendum source. See [docs/m3-stage2.md](docs/m3-stage2.md).

### M4: production hybrid RAG and groundedness

M4 adds a local BM25 ranking over the same policy chunks, fuses it with Qdrant dense retrieval using reciprocal-rank fusion, and reranks the fused candidates with the Colab-aligned `cross-encoder/ms-marco-MiniLM-L-6-v2` model. Answers cite exact retrieved source identifiers. A runtime guard blocks unsupported policy citations and numeric claims; actual customer refund amounts remain exclusive to the M2 refund tools.

Reinstall the editable project, then rebuild the Qdrant collection once because M4 adds citation metadata:

```cmd
python -m pip install -e .
python -m shopsense.agent.cli --index-policies --rebuild-policy-index
```

The 20-case local golden set is at `data/shopsense/evaluation/m4_golden_set.json`. Run the no-model-cost retrieval check first, then the full groundedness run when ready:

```cmd
python -m shopsense.agent.cli --evaluate-policy-rag --retrieval-only
python -m shopsense.agent.cli --evaluate-policy-rag
```

See [docs/m4.md](docs/m4.md) for the complete runtime flow, grounding contract, and golden-set design. M4 does not use Langfuse.

### M8: guarded FastAPI demo chat

M8 packages the existing M4 policy grounding, M6 multi-agent workflow, and M7 reliability boundary behind a local FastAPI service. Deterministic guardrails stop prompt-injection commands, product-safety reports, internal-account-status questions, and explicitly stated refund requests above the policy-derived ₹2,000 automated cap before an order or refund tool can run. Safety, privacy, and high-refund handoffs may add read-only policy citations, but never resume M6 or an MCP action. Every allowed message then reaches M6’s structured triage: general policy questions go directly to grounded RAG and operational requests use MCP only when an order lookup is required. M6 always starts with M4's broad hybrid discovery; canonical source IDs selected by triage may add relevant evidence but never remove a broadly retrieved passage. Policy answers receive one bounded citation-repair attempt; an ungrounded result is replaced with a safe clarification response and exposes no retrieved citations.

The supplied 20-case golden source remains at the data-toolkit path and is loaded read-only; M8 does not modify it, M1 data, M2 mock data, or policy documents.

After activating the virtual environment and installing the editable project, start the local-only server from Command Prompt:

```cmd
python -m shopsense.api
```

Then open `http://127.0.0.1:8000/docs` to use the demo endpoint. `POST /chat` always requires `question`, accepts only live-chat fields, and returns an `input`/`output` response with report fields named without an `_o` suffix: `evaluation_run_id`, `case_id`, `expected`, `category`, `must_cite`, `must_not_contain`, and `expected_route`. Match, verdict, and milestone-execution fields are omitted because live chat has no golden expectation to compare. Golden expectations are evaluator-only data and are never accepted by this endpoint. General policy questions need no reference. Memory and preference requests require `customer_ref`; order, shipment, and refund requests require an `order_ref` in the question. Supplying both validates ownership before operational data is returned. `message_id` is optional and enables a safe idempotent retry, but is not returned publicly. `GET /health` needs no credentials. M8 model and retrieval settings are named `SHOPSENSE_M8_*` in both `.env` and `.env.example`.

### M8 golden and ad-hoc evaluation

The original 20-case golden source is always read without modification. Run an isolated, sequential golden evaluation from Command Prompt with:

```cmd
python -m shopsense.m8.evaluation --golden
```

For one question without golden expectations:

```cmd
python -m shopsense.m8.evaluation --question "What is the return window for electronics?"
```

Each run generates its own case IDs for M7/Langfuse and writes `results.jsonl` plus `summary.json` below `runtime\evaluations\<run-id>`. Each JSONL line preserves the original `input` and adds the agreed `output` fields, including `case_id_o`, direct `_o` comparisons, and `milestones_o` for M1-M8. Deterministic route, citation, and forbidden-text checks are hard gates; the separate `SHOPSENSE_M8_EVALUATION_MODEL` semantic judge evaluates the expected answer. If that judge is unavailable, the affected result is `review`, never a silent pass. Evaluation uses no Supermemory service and separate SQLite runtime state, so it cannot alter customer memory or the live M5/M7/M8 databases.

The supplied `data-toolkit\capstone-data-toolkit\data\shopsense\eval\golden_set.json` remains immutable. `golden_set_shopsense_v2.json` sits beside it as the source-faithful benchmark: it preserves all original IDs and questions while correcting malformed citation lists, policy-fact expectations, and the prompt-injection route. Run it explicitly with `--golden-path`; the default remains the supplied original file.

For Postman, use only `POST /chat` with `{"question": "..."}` (plus optional live-chat context). Use the command-line evaluator for a full sequential run of all 20 source records.
