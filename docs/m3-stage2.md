# ShopSense M3 Stage 2: Qdrant Policy Retrieval

Stage 2 implements the policy-document half of M3. It keeps M3 Stage 1 customer memory in Supermemory and stores policy chunks in one Qdrant Cloud collection: `shopsense_policy_documents`.

## Corpus and ingestion

The project-local source corpus is under `data/shopsense/corpus/`. Its `index.json` declares 14 policies. Every policy is ingested from both its Markdown and PDF source, for 28 policy source files in total.

Markdown is loaded with LangChain's `TextLoader`; PDFs use `PyPDFLoader`. Each source is split with the Day 2 RAG Colab defaults: 900 characters per chunk with a 120-character overlap. Gemini's `gemini-embedding-001` creates embeddings, and Qdrant stores them with cosine distance.

Each Qdrant point retains these filterable fields:

- `policy_name`
- `policy_slug`
- `policy_section`
- `source_format` (`markdown`, `pdf`, or `application_override`)

It also retains `source_file`, `chunk_index`, `content_hash`, and the original chunk text. Point IDs are deterministic, so normal re-indexing updates the same logical points instead of duplicating them. Local embedding cache files prevent repeated Gemini embedding calls for unchanged text.

## Grocery-rule resolution

The supplied Grocery category addendum conflicts with the general returns policy: it says 14 days, while the approved ShopSense rule is that groceries are non-returnable. Both original Markdown and PDF files remain indexed for traceability. An `application_override` point is added during ingestion, and Grocery return queries always receive this authoritative rule first. Retrieval suppresses the conflicting 14-day Grocery chunk from the context supplied to the agent.

## Agent flow

Before the existing M2 tool-calling loop begins, a policy-related question triggers `search_policy_documents`. The function embeds the question, queries Qdrant, formats up to three policy passages as untrusted context, and adds them to the model’s system context. The M2 agent still uses its four operational tools for live order, shipment, and refund facts. Policy passages cannot override tool results or claim that a replacement was created.

## Configuration and commands

Add these values to `.env` after copying the template:

```text
GOOGLE_API_KEY=...
QDRANT_URL=...
QDRANT_API_KEY=...
SHOPSENSE_M3_POLICY_EMBEDDING_MODEL="gemini-embedding-001"
SHOPSENSE_M3_POLICY_COLLECTION="shopsense_policy_documents"
```

Install the added dependencies, then build or update the index:

```cmd
python -m pip install -e .
python -m shopsense.agent.cli --index-policies
```

Use this only when you intentionally want to replace the collection before indexing:

```cmd
python -m shopsense.agent.cli --index-policies --rebuild-policy-index
```

After indexing, start the ordinary CLI. A customer reference remains optional for policy questions and is required only when you also want M3 customer memory:

```cmd
python -m shopsense.agent.cli
python -m shopsense.agent.cli --customer-ref KW-C-11111
```
