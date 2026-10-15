"""Fail-open, PII-minimising Langfuse tracing for the M7 boundary."""

from __future__ import annotations

import hashlib
import hmac
import os
import threading
from contextlib import contextmanager
from typing import Any, Iterator, Protocol


class TraceRecorder(Protocol):
    @contextmanager
    def run(self, *, case_id: str, channel: str, customer_ref: str) -> Iterator[None]: ...

    @contextmanager
    def span(self, name: str, metadata: dict[str, Any] | None = None) -> Iterator[None]: ...

    def flush(self) -> None: ...


class NoopTracer:
    """A tracing replacement used when Langfuse is not configured."""

    @contextmanager
    def run(self, *, case_id: str, channel: str, customer_ref: str) -> Iterator[None]:
        yield

    @contextmanager
    def span(self, name: str, metadata: dict[str, Any] | None = None) -> Iterator[None]:
        yield

    def flush(self) -> None:
        return None


class LangfuseTracer(NoopTracer):
    """Best-effort trace adapter which never sends raw customer content."""

    def __init__(self, client: Any, *, pii_salt: str, propagate: Any | None = None) -> None:
        self.client = client
        self.pii_salt = pii_salt.encode("utf-8")
        self.propagate = propagate

    def _customer_hash(self, customer_ref: str) -> str:
        return hmac.new(self.pii_salt, customer_ref.encode("utf-8"), hashlib.sha256).hexdigest()

    @contextmanager
    def run(self, *, case_id: str, channel: str, customer_ref: str) -> Iterator[None]:
        metadata = {"case_id": case_id, "channel": channel, "customer_hash": self._customer_hash(customer_ref)}
        with self._safe_context(self._propagation_context(case_id, metadata["customer_hash"], channel)):
            with self.span("shopsense.m7.ticket_resolution", metadata):
                yield

    @contextmanager
    def span(self, name: str, metadata: dict[str, Any] | None = None) -> Iterator[None]:
        # Metadata supplied by callers is deliberately limited to operation and
        # state values. Do not add prompts, tool arguments, email subjects, or IDs.
        factory = lambda: self.client.start_as_current_observation(
            as_type="span", name=name, input=dict(metadata or {})
        )
        with self._safe_context(factory):
            yield

    def flush(self) -> None:
        """Bound shutdown work so a telemetry outage cannot delay the CLI indefinitely."""

        try:
            def flush_safely() -> None:
                try:
                    self.client.flush()
                except Exception:
                    return None

            thread = threading.Thread(target=flush_safely, daemon=True)
            thread.start()
            thread.join(timeout=2.0)
        except Exception:
            return None

    def _propagation_context(self, case_id: str, customer_hash: str, channel: str):
        if self.propagate is None:
            return None
        return lambda: self.propagate(
            session_id=case_id,
            user_id=customer_hash,
            tags=["shopsense", "m7", channel],
        )

    @contextmanager
    def _safe_context(self, factory: Any) -> Iterator[None]:
        if factory is None:
            yield
            return
        try:
            context = factory()
            context.__enter__()
        except Exception:
            yield
            return
        try:
            yield
        except BaseException as exc:
            try:
                context.__exit__(type(exc), exc, exc.__traceback__)
            except Exception:
                pass
            raise
        else:
            try:
                context.__exit__(None, None, None)
            except Exception:
                pass


def create_m7_tracer() -> TraceRecorder:
    """Create Langfuse only when all required M7 telemetry configuration exists."""

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
    pii_salt = os.getenv("SHOPSENSE_M7_TRACE_PII_SALT", "").strip()
    if not (public_key and secret_key and pii_salt):
        return NoopTracer()
    try:
        from langfuse import get_client, propagate_attributes

        # The v4 SDK reads credentials from its standard environment variables.
        # Keep the user-facing M7 name for the optional custom endpoint.
        os.environ.setdefault("LANGFUSE_PUBLIC_KEY", public_key)
        os.environ.setdefault("LANGFUSE_SECRET_KEY", secret_key)
        base_url = os.getenv("LANGFUSE_BASE_URL", "").strip()
        if base_url:
            os.environ.setdefault("LANGFUSE_HOST", base_url)
        return LangfuseTracer(get_client(), pii_salt=pii_salt, propagate=propagate_attributes)
    except Exception:
        return NoopTracer()
