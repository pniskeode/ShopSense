"""M7 reliability primitives for ShopSense's M6 live-case boundary."""

from .case_ids import CaseIdConfigurationError, derive_email_message_id
from .circuit import CircuitBreaker, CircuitOpenError
from .faults import FaultInjectingMCPToolClient
from .mcp import OrderSystemFailure, ResilientMCPToolClient
from .queue import HumanQueue, QueueItem, QueueStateError
from .tracing import LangfuseTracer, NoopTracer, TraceRecorder, create_m7_tracer

__all__ = [
    "CaseIdConfigurationError", "CircuitBreaker", "CircuitOpenError", "FaultInjectingMCPToolClient", "HumanQueue", "LangfuseTracer",
    "NoopTracer", "OrderSystemFailure", "QueueItem", "QueueStateError", "ResilientMCPToolClient",
    "TraceRecorder", "create_m7_tracer", "derive_email_message_id",
]
