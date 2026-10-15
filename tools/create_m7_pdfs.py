"""Create the ShopSense M7 implementation brief and the M1-M7 input reference."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


OUTPUT = Path(r"C:\Users\Prem Niskeode\Desktop\Agents\milestoneimplementations")
BLUE = colors.HexColor("#0B5CAD")
LIGHT_BLUE = colors.HexColor("#EAF3FC")
INK = colors.HexColor("#1F2937")
MUTED = colors.HexColor("#52606D")
LINE = colors.HexColor("#CBD5E1")
GREEN = colors.HexColor("#166534")
AMBER = colors.HexColor("#92400E")


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title", parent=base["Title"], fontName="Helvetica-Bold", fontSize=22,
            leading=27, textColor=BLUE, spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle", parent=base["Normal"], fontName="Helvetica", fontSize=10.5,
            leading=15, textColor=MUTED, spaceAfter=12,
        ),
        "h1": ParagraphStyle(
            "H1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=15,
            leading=19, textColor=BLUE, spaceBefore=10, spaceAfter=7,
        ),
        "h2": ParagraphStyle(
            "H2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=11.5,
            leading=14, textColor=INK, spaceBefore=8, spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "Body", parent=base["BodyText"], fontName="Helvetica", fontSize=9.1,
            leading=13, textColor=INK, spaceAfter=5,
        ),
        "small": ParagraphStyle(
            "Small", parent=base["BodyText"], fontName="Helvetica", fontSize=7.8,
            leading=10.5, textColor=MUTED,
        ),
        "table": ParagraphStyle(
            "Table", parent=base["BodyText"], fontName="Helvetica", fontSize=7.6,
            leading=10, textColor=INK,
        ),
        "table_head": ParagraphStyle(
            "TableHead", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=7.6,
            leading=10, textColor=colors.white,
        ),
        "callout": ParagraphStyle(
            "Callout", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=9,
            leading=13, textColor=GREEN,
        ),
        "code": ParagraphStyle(
            "Code", parent=base["Code"], fontName="Courier", fontSize=7.5,
            leading=10, textColor=INK, leftIndent=5, rightIndent=5,
        ),
    }


S = styles()


def p(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, S[style])


def bullets(items: list[str]) -> list[Paragraph]:
    return [p(f"- {item}") for item in items]


def section(title: str, paragraphs: list[str] | None = None) -> list:
    flow = [p(title, "h1")]
    flow.extend(p(item) for item in paragraphs or [])
    return flow


def table(headers: list[str], rows: list[list[str]], widths: list[float]) -> Table:
    data = [[p(header, "table_head") for header in headers]]
    data.extend([[p(cell, "table") for cell in row] for row in rows])
    result = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, LINE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BLUE]),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return result


def header_footer(label: str):
    def draw(canvas, document) -> None:
        canvas.saveState()
        width, height = A4
        canvas.setStrokeColor(LINE)
        canvas.line(18 * mm, height - 14 * mm, width - 18 * mm, height - 14 * mm)
        canvas.setFont("Helvetica-Bold", 8)
        canvas.setFillColor(BLUE)
        canvas.drawString(18 * mm, height - 10 * mm, label)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawRightString(width - 18 * mm, 10 * mm, f"Page {document.page}")
        canvas.drawString(18 * mm, 10 * mm, "ShopSense capstone - implementation reference")
        canvas.restoreState()
    return draw


def document(path: Path, title: str, story: list) -> None:
    template = SimpleDocTemplate(
        str(path), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=20 * mm, bottomMargin=17 * mm,
        title=title, author="ShopSense",
    )
    template.build(story, onFirstPage=header_footer(title), onLaterPages=header_footer(title))


def build_m7_brief() -> None:
    story: list = [
        Spacer(1, 11 * mm),
        p("ShopSense M7 - Codex Implementation Brief", "title"),
        p("Langfuse tracing and production reliability hardening", "subtitle"),
        p("Purpose: Give Codex the complete M7 guide: M6 runtime tracing, resilient order-system MCP access, retry and circuit-breaker rules, a durable human fallback queue, safe identity handling, fault injection, and Command Prompt operation."),
        p("M7 in one line", "h2"),
        p("Run the existing M6 live-case workflow with safe Langfuse observability and a durable, customer-safe fallback when the order-system MCP boundary is unreliable."),
        p("Source documents", "h2"),
        *bullets([
            "Capstone Problem Statements - ShopSense M7 tracing and reliability requirement.",
            "Day 4 Session 1 - LangFuse Instrumentation, Failure Injection and Production Hardening notebook.",
            "Existing M1-M6 implementation briefs, source boundaries, MCP contract, and workflow decisions.",
        ]),
        Spacer(1, 5),
        p("Prepared for the existing local workspace:", "small"),
        p(r"C:\Users\Prem Niskeode\Desktop\Agents\capstone-project\shopsense-agent", "code"),
        PageBreak(),
    ]

    story += section("1. Executive Summary and Decisions")
    story.append(table(
        ["Area", "M7 decision"],
        [
            ["Customer entry point", "No separate customer command. Every --m6-chat, --m6-email-json, --m6-email-file, and --m6-resume execution receives M7 protection."],
            ["Observability", "One fail-open Langfuse trace per M6 invocation. case_id is the session/correlation ID; customer identity is HMAC-hashed."],
            ["Reliability boundary", "Only the M6-to-MCP order-system boundary is retried and protected by the breaker. M3 memory and M4 policy failures retain their existing safe behavior."],
            ["Retry", "Three total attempts for transient transport failures only: initial call plus two bounded exponential-backoff attempts with jitter."],
            ["Circuit breaker", "One shared SQLite breaker named shopsense-order-mcp. It opens after three consecutive logical failures, remains open 30 seconds, and permits exactly one non-retried half-open probe."],
            ["Fallback queue", "One atomic, idempotent local queue item per case_id for infrastructure failure only. The queue never stores raw message text or memory."],
            ["Human approval", "Verified high-risk refunds still use M6 approve/reject. An order-system outage uses the M7 fallback queue instead; the two paths are mutually exclusive."],
            ["Data writes", "M7 does not edit intake, mock_api, policy documents, or refund records. Queue and breaker SQLite state are operational metadata only."],
        ], [43 * mm, 126 * mm]
    ))
    story += section("2. Input Contracts and Identity", [
        "M7 does not create another customer input schema. It hardens the M6 IncomingMessage envelope after M6 validates the channel and before operational MCP work is performed.",
    ])
    story.append(table(
        ["Channel", "Identity rule", "Case identity and repeat behavior"],
        [
            ["Chat", "--customer-ref plus --m6-chat QUESTION. customer_ref remains the M3 namespace and MCP ownership identity.", "CLI generates CHAT-<uuid> by default. A caller can provide --m6-message-id for a deliberate retry. The same ID, customer, and message returns the checkpointed result."],
            ["Simulated email", "customer_ref and body are required; subject is optional. message_id is used if supplied.", "When message_id is absent, M7 derives EMAIL-<hmac> from normalized customer_ref, subject, and body using SHOPSENSE_CASE_ID_SALT. Same simulated email means same case."],
            ["Human resume", "case_id and decision approve/reject are required only for an M6 refund interrupt.", "The prior checkpoint message supplies channel/customer context. Completed or fallback cases cannot be resumed as refund approval."],
        ], [30 * mm, 63 * mm, 76 * mm]
    ))
    story.append(PageBreak())

    story += section("3. Trace Design and Privacy")
    story += [
        p("M7 tracing is best effort. Missing or unavailable Langfuse never blocks a ticket resolution, retry, breaker transition, fallback queue operation, or CLI shutdown."),
        p("Trace hierarchy", "h2"),
        p("ticket_resolution -> LangGraph node -> worker model call and/or MCP call -> retry and circuit-breaker event -> human-queue handoff when needed", "code"),
    ]
    story.append(table(
        ["Trace item", "Safe metadata", "Never trace"],
        [
            ["Root trace", "case_id, channel, HMAC customer hash, m7 tag", "Raw customer_ref, raw message body, email subject, recalled memory, API keys."],
            ["Node and worker span", "Node/worker name and safe route state.", "Prompts, model completions, policy passage text, M3 memory text."],
            ["MCP span", "Operation name, attempt number, half-open-probe flag.", "customer_ref, order_ref, MCP arguments, customer-safe result payload."],
            ["Retry/breaker span", "Operation, attempt, event, category, probe flag.", "Exception text, stack trace, secrets."],
            ["Queue span", "Failed operation and safe failure category.", "Raw message, memory, source mock rows, customer reference."],
        ], [38 * mm, 59 * mm, 72 * mm]
    ))
    story += section("4. Resilient MCP Boundary", [
        "The existing FastMCP server remains the only operational-data path. Its four read-only customer-safe APIs require customer_ref plus order_ref and enforce ownership before returning data.",
    ])
    story.append(table(
        ["Condition", "M7 behavior", "Customer/workflow result"],
        [
            ["Valid found=true or found=false response", "No retry. The breaker records success because the dependency responded correctly.", "Normal M6 operational result. found=false is a safe not_found result, never a queue item."],
            ["Timeout, connection, transport, temporary-server failure", "Retry up to three total attempts. Retry delay uses bounded exponential backoff plus jitter.", "On exhaustion, record one logical breaker failure and enqueue a fallback item."],
            ["Malformed or schema-invalid MCP result", "No retry. Record one logical breaker failure immediately.", "Fallback queue and human_support_escalation; no unsafe interpretation of result data."],
            ["Known business/validation outcome", "No retry and no queue. The MCP response itself carries the customer-safe result.", "M6 handles missing order_ref, unknown order, or customer-order mismatch normally."],
            ["Open circuit", "Short-circuit without invoking MCP.", "Fallback queue item for the live case. No refund approval interrupt."],
        ], [48 * mm, 65 * mm, 56 * mm]
    ))
    story.append(PageBreak())

    story += section("5. Circuit Breaker State Machine")
    story += [
        p("The breaker is persisted in runtime/shopsense_m7.sqlite, so it is shared across CLI processes and survives restarts."),
        p("CLOSED -- three consecutive logical failures --> OPEN -- 30 seconds --> HALF_OPEN -- one probe success --> CLOSED", "code"),
        p("HALF_OPEN -- probe failure --> OPEN. A second process cannot take another probe while the half-open lease is active. If a process dies while holding the probe, the lease expires after the same timeout."),
    ]
    story.append(table(
        ["State", "Allowed behavior", "Persisted fields"],
        [
            ["closed", "Calls proceed with ordinary retry behavior.", "consecutive_failures"],
            ["open", "Calls fail fast and create/return the idempotent fallback queue item.", "opened_at and failure count"],
            ["half_open", "Exactly one MCP probe runs, with one attempt and no retry. Other calls fail fast until the probe resolves or expires.", "probe_started_at"],
        ], [35 * mm, 88 * mm, 46 * mm]
    ))
    story += section("6. Durable Human Fallback Queue", [
        "M7 uses the same SQLite runtime file for the breaker and queue. SQLite BEGIN IMMEDIATE transactions make enqueue, claim, and resolve transitions durable and cross-process safe.",
    ])
    story.append(table(
        ["Queue field", "Purpose"],
        [
            ["queue_id and case_id", "Operator handle and link to the M6 checkpoint. case_id uniqueness prevents duplicate handoffs."],
            ["customer_ref_hash", "Database-local HMAC hash; raw customer_ref is not stored in the item."],
            ["channel, failed_operation, failure_category", "Safe operational context for a human support operator."],
            ["retry_count and circuit_state", "Explains why automatic service stopped."],
            ["status, timestamps, operator, resolution_note", "Manual lifecycle: queued -> claimed -> resolved. Resolution never replays the case or edits source data."],
        ], [54 * mm, 115 * mm]
    ))
    story.append(PageBreak())

    story += section("7. M6 and M7 Routing Relationship")
    story.append(table(
        ["Situation", "M6/M7 route", "Why"],
        [
            ["Verified high-risk refund", "M6 awaiting_human_approval then approve/reject resume.", "This is a business-risk decision based on verified facts and policy evidence."],
            ["Angry non-refund concern", "M6 human_support_escalation.", "Not a refund approval decision; M7 queue is not required when MCP data is available."],
            ["MCP outage or malformed response", "M7 fallback queue plus M6 human_support_escalation.", "The order system is unsafe/unavailable, so there are no facts on which to make a refund approval decision."],
            ["M4 policy unavailable", "Existing M6 safe policy escalation, not an M7 breaker event.", "M7 deliberately protects the order-system MCP dependency only."],
            ["M3 memory unavailable", "Existing M6 best-effort memory behavior, not an M7 breaker event.", "Memory recall never authorizes operational action."],
        ], [51 * mm, 59 * mm, 59 * mm]
    ))
    story += section("8. Command Prompt Operation")
    story += [
        p("Initial setup", "h2"),
        p(r'cd /d "C:\Users\Prem Niskeode\Desktop\Agents\capstone-project\shopsense-agent"<br/>.\.venv\Scripts\activate.bat<br/>python -m pip install -e .', "code"),
        p("Required M7 .env entries", "h2"),
        p("LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL, SHOPSENSE_CASE_ID_SALT, and SHOPSENSE_M7_TRACE_PII_SALT. M7 also uses its own M7 model, summary, embedding, and reranker entries.", "body"),
        p("Run an M7-enhanced M6 chat", "h2"),
        p('python -m shopsense.agent.cli --customer-ref KW-C-00317 --m6-chat "Where is my order KW-O-000000?"', "code"),
        p("List and operate the fallback queue", "h2"),
        p('python -m shopsense.agent.cli --m7-queue-list<br/>python -m shopsense.agent.cli --m7-queue-claim QUEUE-000001 --operator "Prem"<br/>python -m shopsense.agent.cli --m7-queue-resolve QUEUE-000001 --operator "Prem" --note "Verified after recovery."', "code"),
    ]
    story.append(PageBreak())

    story += section("9. Implementation Map and Configuration")
    story.append(table(
        ["Path", "Responsibility"],
        [
            ["src/shopsense/reliability/case_ids.py", "Stable chat/email identity helpers and salted simulated-email HMAC."],
            ["src/shopsense/reliability/mcp.py", "Transient retry classification, bounded backoff, breaker events, malformed-response handling."],
            ["src/shopsense/reliability/circuit.py", "Cross-process SQLite closed/open/half_open circuit breaker."],
            ["src/shopsense/reliability/queue.py", "Idempotent fallback queue, HMAC customer hash, claim/resolve lifecycle."],
            ["src/shopsense/reliability/tracing.py", "Fail-open Langfuse v4 trace adapter and bounded flush."],
            ["src/shopsense/workflow/m6.py", "M7 trace spans in graph nodes/models, fallback conversion, result reliability metadata."],
            ["src/shopsense/agent/cli.py", "M7-enhanced M6 construction and queue operator commands."],
            ["tests/m7/", "Offline retry, breaker, queue, tracing, identity, fault-injection, and workflow-fallback tests."],
        ], [62 * mm, 107 * mm]
    ))
    story += section("10. Acceptance Criteria and Scope Boundary")
    story += bullets([
        "Every M6 invocation has a safe M7 trace when Langfuse is configured; trace failure never fails customer resolution.",
        "Only transient order-system failures retry, and attempts never exceed three total calls.",
        "Three consecutive logical MCP failures open one shared breaker; half-open permits one non-retried probe.",
        "Unknown or mismatched customer-order pairs remain normal safe not_found results and do not create a fallback queue item.",
        "Fallback queue items are idempotent per case_id, contain no raw message/memory, and require manual claim then resolve.",
        "Fallback queue and refund human approval are distinct, mutually exclusive control paths.",
        "M7 makes no changes to source intake records, policy files, mock API data, or refund rows.",
    ])
    story += [
        p("Out of scope", "h2"),
        p("A live order platform, real email delivery, a payment/refund API, replaying resolved queue items, automatic queue resolution, trace storage of customer content, and applying M7 breaker policy to Supermemory or Qdrant."),
        p("Verification baseline", "h2"),
        p("The complete M1-M7 offline suite passes 78 tests. Live Langfuse dashboard verification depends on valid Langfuse credentials; M7 is intentionally fail-open until they are configured."),
    ]
    document(OUTPUT / "ShopSense_M7_Codex_Implementation_Brief.pdf", "ShopSense M7 - Codex Implementation Brief", story)


def build_input_reference() -> None:
    story: list = [
        Spacer(1, 10 * mm),
        p("ShopSense M1-M7 - Input Data Reference", "title"),
        p("Source data, customer input, identity boundaries, runtime state, and outputs", "subtitle"),
        p("A quick guide to the data accepted by each ShopSense milestone. M1 historical intake and M2 mock operational data remain separate datasets throughout the project."),
        p("Milestone input summary", "h1"),
    ]
    story.append(table(
        ["Milestone", "Input data used", "Boundary / output"],
        [
            ["M1 - Ticket parsing", "data/shopsense/intake/records.jsonl: raw_text, ticket_id, channel, received_at, order_ref, customer_ref, and supplied ground_truth.", "Validated parsed ticket output. M1 data does not establish a safe link to M2 mock data."],
            ["M2 - Support agent and tools", "CLI question plus mock_api orders, customers, products, shipments, and refund rows.", "Python functions return read-only order/shipping/refund information. M1 and M2 remain independent."],
            ["M3 Stage 1 - Customer memory", "Historical intake tickets by customer_ref plus explicit preferences stated during conversation.", "Supermemory holds customer-isolated episodic history/preferences. Rolling summary is separate."],
            ["M3 Stage 2 - Policy index", "Markdown and PDF return, shipping, warranty, refund-authority, and category-policy files in data/shopsense/corpus.", "Qdrant chunks, embeddings, and source metadata for one policy collection."],
            ["M4 - Production RAG", "Customer policy question plus M3 Stage 2 collection.", "Dense plus BM25 hybrid retrieval, reranking, grounded answer validation, and citations."],
            ["M5 - Approval workflow", "One original M1 record using supplied ground_truth, or one M2 order_ref and current order/refund rows; M4 policy evidence.", "Read-only checkpointed approval decision. M3 memory is not decision input; no payout or source-data write."],
            ["M6 - Multi-agent team and MCP", "Live chat: customer_ref plus question. Simulated email: message_id, customer_ref, body, optional subject. M2 data through MCP; M3 memory; M4 policy evidence.", "Validated live case, customer-scoped MCP reads, Supervisor/four workers, optional refund interrupt, and M6 checkpoint."],
            ["M7 - Tracing and reliability", "The normalized M6 IncomingMessage, existing M6 checkpoint/case_id, M6-to-MCP operation results, and safe operational metadata.", "Langfuse trace plus retry/breaker/queue state. No new customer data source and no source-data write."],
        ], [30 * mm, 81 * mm, 58 * mm]
    ))
    story.append(PageBreak())
    story += section("M6/M7 live-case input path")
    story.append(table(
        ["Stage", "Data accepted", "What is returned / persisted"],
        [
            ["Channel normalization", "Chat or simulated email payload.", "IncomingMessage: message_id, customer_ref, body, channel, optional subject."],
            ["Triage", "Current message plus customer-isolated M3 recall when available.", "Intent, explicit order_ref/INR claim, calm/angry sentiment, and policy need. Explicit refund wording is guarded."],
            ["Order-Actions through MCP", "Verified customer_ref plus order_ref, using current M2 mock data only.", "Customer-safe order/shipment/refund information or safe not_found/validation outcome."],
            ["Policy RAG", "Current policy question and verified category when available, over M4 policy index.", "Grounded policy answer, citation IDs, and refund-policy cap/eligibility."],
            ["M6 checkpoint", "Validated graph state and human decision only when refund approval is required.", "M6 case_id, status/message, route reasons, citations, and execution log in runtime/shopsense_m6.sqlite."],
            ["M7 trace", "case_id, channel, HMAC customer hash, node/model/MCP/retry/breaker/queue event metadata.", "One fail-open Langfuse trace per invocation. No raw message, memory, customer_ref, order_ref, secrets, or internal fields."],
            ["M7 breaker", "Logical MCP outcome only: success, transient exhaustion, malformed response, or open/half-open transition.", "Shared shopsense-order-mcp state in runtime/shopsense_m7.sqlite. It is not used for M3/M4 dependencies."],
            ["M7 fallback queue", "case_id, HMAC customer hash, channel, failed operation/category, retry count, circuit state.", "One durable queued/claimed/resolved handoff record per case. It never stores raw message or memory and never replays a case."],
        ], [36 * mm, 71 * mm, 62 * mm]
    ))
    story += section("Key identity and separation rules")
    story += bullets([
        "customer_ref is the M3 memory namespace and the M6/M7 customer identity; order_ref identifies an M2 operational record.",
        "MCP requires both references and rejects unknown or mismatched pairs without disclosing another customer data.",
        "M1 historical ticket identifiers must not be joined, repaired, or mapped to M2 operational records.",
        "M7 case_id correlates one live M6 workflow, its trace, and at most one fallback queue item. It is not a customer identifier.",
        "A generated chat ID is new for each normal CLI command. A supplied --m6-message-id is an explicit idempotency key and must be reused only for the identical message/customer.",
        "A supplied email message_id is preferred. If absent, M7 derives a deterministic HMAC ID using SHOPSENSE_CASE_ID_SALT.",
    ])
    story += section("M7 outcome boundary")
    story.append(table(
        ["Situation", "M7 involvement", "Result"],
        [
            ["Healthy MCP response, including safe not_found", "Trace only; no retry/queue.", "Existing M6 output."],
            ["Transient MCP outage", "Retry, breaker accounting, and one fallback queue handoff on exhaustion.", "M6 human_support_escalation with reliability metadata."],
            ["High-risk verified refund", "Trace only. M6 owns the policy-backed approval interrupt.", "awaiting_human_approval then approve/reject."],
            ["M3 memory or M4 policy problem", "Trace node state only; no M7 breaker/queue.", "Existing M6 safe behavior."],
        ], [48 * mm, 62 * mm, 59 * mm]
    ))
    document(OUTPUT / "ShopSense_M1-M7_Input_Data_Reference.pdf", "ShopSense M1-M7 - Input Data Reference", story)


if __name__ == "__main__":
    OUTPUT.mkdir(parents=True, exist_ok=True)
    build_m7_brief()
    build_input_reference()
