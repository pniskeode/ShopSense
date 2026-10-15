"""LangChain tools exposed to the M2 model."""

from typing import Any, Literal

from langchain_core.tools import BaseTool, tool

from shopsense.tools import (
    get_refund_amount as get_refund_amount_data,
    lookup_order as lookup_order_data,
    refund_or_replace as refund_or_replace_data,
    track_shipment as track_shipment_data,
)


@tool
def lookup_order(order_ref: str) -> dict[str, Any]:
    """Retrieve an order with related customer and product details."""

    return lookup_order_data(order_ref)


@tool
def track_shipment(order_ref: str) -> dict[str, Any]:
    """Retrieve carrier and latest shipment scan information for an order."""

    return track_shipment_data(order_ref)


@tool
def refund_or_replace(order_ref: str, action: Literal["refund", "replace"]) -> dict[str, Any]:
    """Inspect all refund records or report that replacement is unavailable."""

    return refund_or_replace_data(order_ref, action)


@tool
def get_refund_amount(order_ref: str) -> dict[str, Any]:
    """Calculate the approved refund total awaiting payout and show other statuses separately."""

    return get_refund_amount_data(order_ref)


LANGCHAIN_TOOLS: list[BaseTool] = [
    lookup_order,
    track_shipment,
    refund_or_replace,
    get_refund_amount,
]

TOOL_REGISTRY: dict[str, BaseTool] = {item.name: item for item in LANGCHAIN_TOOLS}
