"""Deterministic test-only MCP failure injection for M7 reliability checks."""

from __future__ import annotations

import random
from typing import Any, Literal, Protocol


FaultMode = Literal["none", "connection", "timeout", "malformed", "outage"]


class MCPClient(Protocol):
    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...


class FaultInjectingMCPToolClient:
    """Test helper with no environment or customer-facing CLI configuration."""

    def __init__(self, client: MCPClient, *, mode: FaultMode, seed: int, enabled_for_tests: bool = False) -> None:
        if not enabled_for_tests:
            raise ValueError("M7 fault injection is test-only and must be enabled explicitly by a test.")
        self.client = client
        self.mode = mode
        self.random = random.Random(seed)

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self.mode == "connection" and self.random.randrange(2) == 0:
            raise ConnectionError("Injected M7 connection failure")
        if self.mode == "timeout":
            raise TimeoutError("Injected M7 timeout")
        if self.mode == "malformed":
            return {"found": "injected-invalid"}
        if self.mode == "outage":
            raise ConnectionError("Injected M7 sustained outage")
        return self.client.call(name, arguments)
