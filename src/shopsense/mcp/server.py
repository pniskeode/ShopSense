"""Read-only, customer-safe MCP surface for the M2 mock API tables.

The existing M2 functions remain useful data-layer functions, but are intentionally
not exposed directly: they accept only an order reference and may include internal
customer fields. This module is M6's enforcement boundary. An order reference is
enough for an operational lookup; a supplied customer reference adds ownership
validation before any result is returned.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Literal

from shopsense.tools import get_refund_amount, lookup_order, refund_or_replace, track_shipment
from shopsense.tools.data import MockShopSenseData

CONTRACT_RESOURCE_URI = "shopsense://contracts/mock-api-v1"


class MockApiGateway:
    """Customer-safe facade over the read-only ShopSense mock tables."""

    def __init__(self, data: MockShopSenseData | None = None) -> None:
        self.data = data or MockShopSenseData()

    def lookup_order(self, order_ref: str, customer_ref: str | None = None) -> dict[str, Any]:
        failure = self._verify_ownership(order_ref, customer_ref)
        if failure:
            return failure
        raw = lookup_order(order_ref, self.data)
        return {
            "found": True,
            "order_ref": raw["order_ref"],
            "product": raw.get("product"),
            "quantity": raw.get("quantity"),
            "order_value_inr": raw.get("order_value_inr"),
            "placed_at": raw.get("placed_at"),
            "status": raw.get("status"),
            "delivery_promise_days": raw.get("delivery_promise_days"),
            "actual_delivery_days": raw.get("actual_delivery_days"),
        }

    def track_shipment(self, order_ref: str, customer_ref: str | None = None) -> dict[str, Any]:
        failure = self._verify_ownership(order_ref, customer_ref)
        if failure:
            return failure
        return track_shipment(order_ref, self.data)

    def refund_or_replace(
        self,
        order_ref: str,
        action: Literal["refund", "replace"],
        customer_ref: str | None = None,
    ) -> dict[str, Any]:
        failure = self._verify_ownership(order_ref, customer_ref)
        if failure:
            return failure
        return self._redact_refunds(refund_or_replace(order_ref, action, self.data))

    def get_refund_amount(self, order_ref: str, customer_ref: str | None = None) -> dict[str, Any]:
        failure = self._verify_ownership(order_ref, customer_ref)
        if failure:
            return failure
        return self._redact_refunds(get_refund_amount(order_ref, self.data))

    @staticmethod
    def _redact_refunds(result: dict[str, Any]) -> dict[str, Any]:
        """Do not expose an internal approver identity through a customer channel."""

        redacted = dict(result)
        rows = result.get("refunds")
        if isinstance(rows, list):
            redacted["refunds"] = [
                {key: value for key, value in row.items() if key != "approved_by"}
                for row in rows
                if isinstance(row, dict)
            ]
        return redacted

    def _verify_ownership(self, order_ref: str, customer_ref: str | None) -> dict[str, Any] | None:
        customer_ref = customer_ref.strip() if isinstance(customer_ref, str) else ""
        order_ref = order_ref.strip() if isinstance(order_ref, str) else ""
        if not order_ref:
            return {"found": False, "error": "order_ref_required", "message": "Order reference is required."}
        order = self.data.orders_by_ref.get(order_ref)
        if order is None:
            return {"found": False, "error": "order_not_found", "order_ref": order_ref, "message": "Order was not found."}
        if not customer_ref:
            return None
        if customer_ref not in self.data.customers_by_ref:
            return {"found": False, "error": "customer_not_found", "message": "Customer was not found."}
        if order.get("customer_ref") != customer_ref:
            # Do not reveal whether the order exists for another customer.
            return {
                "found": False,
                "error": "order_not_found_for_customer",
                "order_ref": order_ref,
                "message": "Order was not found for this customer.",
            }
        return None


def mock_api_contract() -> str:
    """Return the application-owned contract for the four customer-safe MCP tools."""

    return json.dumps(
        {
            "version": "mock-api-v1",
            "mode": "read_only",
            "required_identity": ["order_ref"],
            "optional_identity": ["customer_ref"],
            "tools": [
                "lookup_order",
                "track_shipment",
                "refund_or_replace",
                "get_refund_amount",
            ],
            "guardrails": [
                "A supplied customer reference is verified against the order before returning operational data.",
                "Order-only lookups never return customer profile fields.",
                "Internal abuse flags are never returned.",
                "No operation creates or changes orders, shipments, refunds, or customer data.",
            ],
        },
        ensure_ascii=False,
    )


def create_mcp_server(gateway: MockApiGateway | None = None) -> Any:
    """Build the FastMCP server lazily so ordinary M1-M5 imports stay lightweight."""

    try:
        from fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - exercised in installation guidance
        raise RuntimeError("M6 requires fastmcp. Install the ShopSense project dependencies first.") from exc

    safe_gateway = gateway or MockApiGateway()
    server = FastMCP("shopsense-mock-api")

    @server.tool
    def lookup_order_for_customer(order_ref: str, customer_ref: str | None = None) -> dict[str, Any]:
        """Retrieve safe order details; validate ownership when customer_ref is supplied."""

        return safe_gateway.lookup_order(order_ref, customer_ref)

    @server.tool
    def track_shipment_for_customer(order_ref: str, customer_ref: str | None = None) -> dict[str, Any]:
        """Retrieve a shipment scan; validate ownership when customer_ref is supplied."""

        return safe_gateway.track_shipment(order_ref, customer_ref)

    @server.tool
    def refund_or_replace_for_customer(
        order_ref: str,
        action: Literal["refund", "replace"],
        customer_ref: str | None = None,
    ) -> dict[str, Any]:
        """Inspect refund records or replacement availability; validate supplied ownership."""

        return safe_gateway.refund_or_replace(order_ref, action, customer_ref)

    @server.tool
    def get_refund_amount_for_customer(order_ref: str, customer_ref: str | None = None) -> dict[str, Any]:
        """Calculate approved refunds awaiting payout; validate supplied ownership."""

        return safe_gateway.get_refund_amount(order_ref, customer_ref)

    @server.resource(CONTRACT_RESOURCE_URI)
    def contract_resource() -> str:
        """Read-only M6 mock-API contract and guardrails."""

        return mock_api_contract()

    return server


def main() -> None:
    """Run the local MCP server over stdio. stdout is reserved for JSON-RPC."""

    try:
        create_mcp_server().run()
    except Exception as exc:  # pragma: no cover - process-level diagnostics
        print(f"ShopSense MCP server failed: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
