# ShopSense M3 Stage 1: Persistent Customer Memory

Stage 1 implements the persistent-memory half of M3. Qdrant policy indexing and policy retrieval are Stage 2.

## Data and identity boundary

Historical ticket memory is built from these project-local files:

```text
data/shopsense/intake/records.jsonl
data/shopsense/m1_results/parsed_records.jsonl
```

The bootstrap joins them by `record_id`, stores only successful M1 results, and uses the intake record's `customer_ref` as the Supermemory `container_tag`. It does not derive a historical customer from `mock_api`: those M2 tables are operational data, not the source of historical tickets.

Every stored historical ticket is episodic memory. It contains the ticket identity and date, optional order reference, M1 issue type/sentiment/urgency, and the original ticket text. A local fingerprint ledger makes repeated bootstrap runs idempotent.

## Preferences and isolation

Only a direct phrase such as `I prefer email updates` or `My preference is a refund instead of a replacement` is stored as semantic preference memory. Sentiment, anger, and a single support event never become a preference. Later explicit statements of the same preference category create a new version; recall suppresses older versions.

Supermemory is the sole store for both preference history and the latest preference. After an explicit preference is written, ShopSense polls the extracted-memory layer up to 20 times at three-second intervals (about one minute). Retrieval is memory-first: it searches extracted memories within the customer namespace first, and only searches source documents when no extracted memory is available. Preference recall filters to explicit-preference metadata, then selects the latest version for each preference category before adding other relevant historical records. If extraction is still pending after the polling window, the preference is already safely stored and the document fallback keeps it available.

The CLI requires an explicit `--customer-ref` to use M3 memory. Without it, M2 continues to answer ordinary operational questions but does not read or write per-customer memory. This prevents an unsafe historical-memory mapping from an order reference.

## Session summary

Each verified customer session retains recent turns in memory. When the running session exceeds 220 tokens, the agent uses Gemini to merge older turns into a summary, preserves concrete facts and preferences, and keeps the latest four turns verbatim. The summary is session working memory, not a long-term Supermemory record.

## Commands

```powershell
# One-time, repeat-safe bootstrap of the 200 successful M1 records.
& .\.venv\Scripts\python.exe -m shopsense.agent.cli --bootstrap-memory

# One-shot M3 question.
& .\.venv\Scripts\python.exe -m shopsense.agent.cli --customer-ref KW-C-11111 "What have I contacted support about before?"

# Interactive M3 session.
& .\.venv\Scripts\python.exe -m shopsense.agent.cli --customer-ref KW-C-11111
```

Required `.env` values are `SUPERMEMORY_API_KEY`, `GOOGLE_API_KEY`, and `SHOPSENSE_M3_SUMMARY_MODEL`.

## Test coverage

The M3 tests cover the 200-record M1 join, customer isolation, explicit-preference-only extraction and versioning, the bounded extraction poll, memory-first/document-fallback retrieval, repeat-safe bootstrap, rolling-summary retention, and M2-agent context/persistence integration. They use in-memory fakes and do not call Supermemory or Gemini.
