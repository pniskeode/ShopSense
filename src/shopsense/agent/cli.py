"""Interactive and one-shot command-line interface for M2."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from shopsense.memory import (
    MemoryUnavailableError,
    bootstrap_historical_memory,
    create_m6_summary_model,
    create_m7_summary_model,
    create_memory_summary_model,
    create_supermemory_service,
)
from shopsense.llm import LiteLLMClient, ModelConfigurationError, configured_model
from shopsense.policy import (
    PolicyUnavailableError,
    create_m4_policy_service,
    create_m6_policy_service,
    create_m7_policy_service,
    create_policy_service,
    evaluate_policy_rag,
    policy_configuration_is_present,
)
from shopsense.workflow import IncomingMessage, M5Workflow, M5WorkflowError, M6Workflow, M6WorkflowError
from shopsense.workflow.m6 import JsonM6Model, StdioMCPToolClient
from shopsense.reliability import (
    CaseIdConfigurationError,
    CircuitBreaker,
    HumanQueue,
    QueueStateError,
    ResilientMCPToolClient,
    create_m7_tracer,
)

from .runner import ShopSenseAgent


M7_RUNTIME_PATH = Path(__file__).resolve().parents[3] / "runtime" / "shopsense_m7.sqlite"


def main() -> None:
    # Windows consoles may default to cp1252 while the model can return
    # Unicode punctuation. Keep the CLI from failing while printing a valid
    # customer-facing answer.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Ask the ShopSense M2-M7 assistant or run an M5/M6 workflow case")
    parser.add_argument("question", nargs="?", help="one natural-language ShopSense question")
    parser.add_argument(
        "--customer-ref",
        help="verified customer reference used for M3 persistent memory, for example KW-C-11111",
    )
    parser.add_argument(
        "--bootstrap-memory",
        action="store_true",
        help="ingest successful M1 ticket history into Supermemory once",
    )
    parser.add_argument(
        "--index-policies",
        action="store_true",
        help="load the Markdown and PDF policy corpus into the M3 Qdrant collection",
    )
    parser.add_argument(
        "--rebuild-policy-index",
        action="store_true",
        help="replace the M3 Qdrant policy collection before indexing it",
    )
    parser.add_argument(
        "--evaluate-policy-rag",
        action="store_true",
        help="run the checked-in M4 20-case hybrid retrieval and groundedness evaluation",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="with --evaluate-policy-rag, skip paid model answers and measure retrieval only",
    )
    m5_group = parser.add_mutually_exclusive_group()
    m5_group.add_argument(
        "--m5-m1-record",
        metavar="RECORD_ID",
        help="run M5 for one original M1 intake record, for example SHOPSENSE-00000",
    )
    m5_group.add_argument(
        "--m5-m2-order",
        metavar="ORDER_REF",
        help="run M5 for one M2 refund order, for example KW-O-000086",
    )
    m5_group.add_argument(
        "--m5-resume",
        action="store_true",
        help="resume an M5 case paused for human approval",
    )
    parser.add_argument("--case-id", help="M5 or M6 checkpoint case ID used with a resume option")
    parser.add_argument("--decision", choices=("approve", "reject"), help="human decision used with an M5 or M6 resume option")
    m6_group = parser.add_mutually_exclusive_group()
    m6_group.add_argument("--m6-chat", metavar="QUESTION", help="run M6 for a chat message; --customer-ref is optional ownership validation")
    m6_group.add_argument("--m6-email-json", metavar="JSON", help="run M6 for a simulated email JSON object")
    m6_group.add_argument("--m6-email-file", metavar="PATH", help="run M6 for a simulated email JSON file")
    m6_group.add_argument("--m6-resume", action="store_true", help="resume an M6 refund review")
    parser.add_argument("--m6-message-id", help="optional stable chat message ID for a client retry")
    m7_queue_group = parser.add_mutually_exclusive_group()
    m7_queue_group.add_argument("--m7-queue-list", action="store_true", help="list durable M7 fallback queue items")
    m7_queue_group.add_argument("--m7-queue-claim", metavar="QUEUE_ID", help="claim one queued M7 fallback item")
    m7_queue_group.add_argument("--m7-queue-resolve", metavar="QUEUE_ID", help="mark one claimed M7 fallback item resolved")
    parser.add_argument("--m7-queue-status", choices=("queued", "claimed", "resolved"), help="optional queue status filter")
    parser.add_argument("--operator", help="operator name for M7 queue claim or resolution")
    parser.add_argument("--note", help="required resolution note for --m7-queue-resolve")
    args = parser.parse_args()

    if args.bootstrap_memory and (args.index_policies or args.rebuild_policy_index or args.evaluate_policy_rag):
        parser.error("--bootstrap-memory cannot be combined with policy-index or evaluation options")
    if args.rebuild_policy_index and not args.index_policies:
        parser.error("--rebuild-policy-index requires --index-policies")
    if args.retrieval_only and not args.evaluate_policy_rag:
        parser.error("--retrieval-only requires --evaluate-policy-rag")
    if args.evaluate_policy_rag and args.index_policies:
        parser.error("--evaluate-policy-rag cannot be combined with policy-index options")

    m5_requested = bool(args.m5_m1_record or args.m5_m2_order or args.m5_resume)
    m6_requested = bool(args.m6_chat or args.m6_email_json or args.m6_email_file or args.m6_resume)
    m7_queue_requested = bool(args.m7_queue_list or args.m7_queue_claim or args.m7_queue_resolve)
    if m7_queue_requested:
        if (
            args.question or args.customer_ref or args.case_id or args.decision or args.m6_message_id
            or m5_requested or m6_requested or args.bootstrap_memory or args.index_policies or args.evaluate_policy_rag
        ):
            parser.error("M7 queue commands cannot be combined with customer, M1-M6, or policy commands")
        if args.m7_queue_status and not args.m7_queue_list:
            parser.error("--m7-queue-status requires --m7-queue-list")
        if args.m7_queue_claim and not args.operator:
            parser.error("--m7-queue-claim requires --operator")
        if args.m7_queue_resolve and (not args.operator or not args.note):
            parser.error("--m7-queue-resolve requires --operator and --note")
        if args.m7_queue_list and (args.operator or args.note):
            parser.error("--operator and --note are not used with --m7-queue-list")
        if args.m7_queue_claim and args.note:
            parser.error("--note is only used with --m7-queue-resolve")
        try:
            queue = HumanQueue(M7_RUNTIME_PATH)
            if args.m7_queue_list:
                output: object = [item.as_dict() for item in queue.list(args.m7_queue_status)]
            elif args.m7_queue_claim:
                output = queue.claim(args.m7_queue_claim, operator=args.operator).as_dict()
            else:
                output = queue.resolve(args.m7_queue_resolve, operator=args.operator, note=args.note).as_dict()
        except (QueueStateError, sqlite3.Error, OSError) as exc:
            parser.error(str(exc))
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return
    if m6_requested:
        if args.question or args.bootstrap_memory or args.index_policies or args.evaluate_policy_rag or m5_requested:
            parser.error("M6 options cannot be combined with M1-M5 commands or a positional chat question")
        if args.m6_resume and (not args.case_id or not args.decision):
            parser.error("--m6-resume requires --case-id and --decision")
        if not args.m6_resume and (args.case_id or args.decision):
            parser.error("--case-id and --decision require --m6-resume")
        if args.m6_message_id and not args.m6_chat:
            parser.error("--m6-message-id requires --m6-chat")
        tracer = create_m7_tracer()
        try:
            memory = create_supermemory_service()
            summary_model = create_m7_summary_model()
            tool_client = ResilientMCPToolClient(
                StdioMCPToolClient(), CircuitBreaker(M7_RUNTIME_PATH), tracer=tracer
            )
            workflow = M6Workflow(
                policy_service=create_m7_policy_service(),
                memory_service=memory,
                summary_model=summary_model,
                tool_client=tool_client,
                model=JsonM6Model(LiteLLMClient(model=configured_model("SHOPSENSE_M7_MODEL"))),
                fallback_queue=HumanQueue(M7_RUNTIME_PATH),
                tracer=tracer,
            )
            if args.m6_resume:
                result = workflow.resume(args.case_id, args.decision)
            elif args.m6_chat:
                result = workflow.run_message(IncomingMessage.chat(args.customer_ref, args.m6_chat, args.m6_message_id))
            else:
                raw = args.m6_email_json
                if args.m6_email_file:
                    raw = Path(args.m6_email_file).read_text(encoding="utf-8")
                payload = json.loads(raw)
                if not isinstance(payload, dict):
                    parser.error("M6 simulated email JSON must be an object")
                result = workflow.run_message(IncomingMessage.email(payload))
        except (
            CaseIdConfigurationError,
            MemoryUnavailableError,
            ModelConfigurationError,
            PolicyUnavailableError,
            M6WorkflowError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            parser.error(str(exc))
        finally:
            tracer.flush()
        print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
        return

    if m5_requested:
        if args.question or args.customer_ref:
            parser.error("M5 options cannot be combined with a chat question or --customer-ref")
        if args.bootstrap_memory or args.index_policies or args.evaluate_policy_rag:
            parser.error("M5 options cannot be combined with memory, policy-index, or evaluation options")
        if args.m5_resume and (not args.case_id or not args.decision):
            parser.error("--m5-resume requires --case-id and --decision")
        if not args.m5_resume and (args.case_id or args.decision):
            parser.error("--case-id and --decision require --m5-resume")
        try:
            workflow = M5Workflow()
            if args.m5_m1_record:
                result = workflow.run_m1_record(args.m5_m1_record)
            elif args.m5_m2_order:
                result = workflow.run_m2_order(args.m5_m2_order)
            else:
                result = workflow.resume(args.case_id, args.decision)
        except M5WorkflowError as exc:
            parser.error(str(exc))
        print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
        return

    if args.bootstrap_memory:
        try:
            service = create_supermemory_service()
        except MemoryUnavailableError as exc:
            parser.error(str(exc))
        stats = bootstrap_historical_memory(service)
        print(
            "M3 bootstrap complete: "
            f"{stats.stored_tickets} tickets stored, {stats.skipped_tickets} already stored, "
            f"{stats.stored_preferences} explicit preferences stored."
        )
        return

    if args.index_policies:
        try:
            service = create_policy_service()
            stats = service.index_corpus(rebuild=args.rebuild_policy_index)
        except PolicyUnavailableError as exc:
            parser.error(str(exc))
        print(
            "M3 policy index complete: "
            f"{stats.chunk_count} chunks from {stats.source_file_count} source files across "
            f"{stats.policy_count} policies in {stats.collection_name}."
        )
        return

    if args.evaluate_policy_rag:
        try:
            service = create_m4_policy_service()
            answer_fn = None
            if not args.retrieval_only:
                agent = ShopSenseAgent(policy_service=service)

                def answer_fn(question: str):
                    result = agent.run(question)
                    return result.answer, [call.result for call in result.tool_calls]

            report = evaluate_policy_rag(service, answer_fn=answer_fn)
        except (PolicyUnavailableError, ModelConfigurationError) as exc:
            parser.error(str(exc))
        print(report.as_json())
        return

    try:
        service = create_supermemory_service() if args.customer_ref else None
        summary_model = create_memory_summary_model() if args.customer_ref else None
        policy_service = create_m4_policy_service() if policy_configuration_is_present() else None
    except (MemoryUnavailableError, ModelConfigurationError, PolicyUnavailableError) as exc:
        parser.error(str(exc))
    try:
        agent = ShopSenseAgent(
            memory_service=service,
            policy_service=policy_service,
            summary_model=summary_model,
        )
    except ModelConfigurationError as exc:
        parser.error(str(exc))

    if args.question:
        result = agent.run(args.question, customer_ref=args.customer_ref)
        print(result.answer)
        for error in result.memory_errors:
            print(f"M3 warning: {error}", file=sys.stderr)
        return

    memory_note = f" M3 memory: {args.customer_ref}." if args.customer_ref else ""
    print(f"ShopSense assistant.{memory_note} Type 'exit' or 'quit' to stop.")
    while True:
        try:
            question = input("ShopSense> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"exit", "quit"}:
            break
        if question:
            result = agent.run(question, customer_ref=args.customer_ref)
            print(result.answer)
            for error in result.memory_errors:
                print(f"M3 warning: {error}", file=sys.stderr)


if __name__ == "__main__":
    main()
