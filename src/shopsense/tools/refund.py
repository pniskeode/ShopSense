"""Transparent, read-only refund and replacement tools."""

from typing import Any, Literal

from .data import MockShopSenseData

RefundAction = Literal["refund", "replace"]
REFUND_STATUSES = ("processed", "approved", "requested", "rejected")


def _sorted_refunds(order_ref: str, data: MockShopSenseData) -> list[dict[str, Any]]:
    rows = data.refunds_by_order.get(order_ref, [])
    return sorted(rows, key=lambda item: (item.get("requested_on", ""), item.get("refund_id", "")))


def _refund_summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        f"{status}_amount_inr": sum(
            row["amount_inr"] for row in rows if row.get("status") == status
        )
        for status in REFUND_STATUSES
    }


def _public_refund(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row[key]
        for key in (
            "refund_id",
            "order_ref",
            "amount_inr",
            "reason_code",
            "status",
            "approved_by",
            "requested_on",
        )
    }


def refund_or_replace(
    order_ref: str,
    action: RefundAction,
    data: MockShopSenseData | None = None,
) -> dict[str, Any]:
    """Return refund records or explain that replacement is unavailable."""

    data = data or MockShopSenseData()
    order_ref = order_ref.strip() if isinstance(order_ref, str) else ""
    if action == "replace":
        return {
            "found": False,
            "action": "replace",
            "order_ref": order_ref,
            "message": "Replacement functionality is not available in the current mock system.",
        }
    if action != "refund":
        return {"found": False, "action": action, "order_ref": order_ref, "message": "Unsupported refund action"}

    rows = _sorted_refunds(order_ref, data)
    if not rows:
        return {
            "found": False,
            "action": "refund",
            "order_ref": order_ref,
            "refunds": [],
            "message": "No refund record is available for this order",
        }
    return {
        "found": True,
        "action": "refund",
        "order_ref": order_ref,
        "refunds": [_public_refund(row) for row in rows],
        "summary": _refund_summary(rows),
    }


def get_refund_amount(order_ref: str, data: MockShopSenseData | None = None) -> dict[str, Any]:
    """Calculate the approved refund total awaiting payout and show all statuses."""

    data = data or MockShopSenseData()
    order_ref = order_ref.strip() if isinstance(order_ref, str) else ""
    rows = _sorted_refunds(order_ref, data)
    summary = _refund_summary(rows)
    if not rows:
        return {
            "found": False,
            "order_ref": order_ref,
            "amount_inr": None,
            "refunds": [],
            "message": "No refund record or refund amount is available for this order",
        }
    return {
        "found": True,
        "order_ref": order_ref,
        "amount_inr": summary["approved_amount_inr"],
        "calculation": "amount_inr is the sum of approved refund records awaiting payout; other statuses are reported separately",
        "summary": summary,
        "refunds": [_public_refund(row) for row in rows],
    }
