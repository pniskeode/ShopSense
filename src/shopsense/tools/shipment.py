"""Shipment tracking tool."""

from typing import Any

from .data import MockShopSenseData


def track_shipment(order_ref: str, data: MockShopSenseData | None = None) -> dict[str, Any]:
    """Retrieve the shipment/carrier scan for an order."""

    data = data or MockShopSenseData()
    order_ref = order_ref.strip() if isinstance(order_ref, str) else ""
    rows = data.shipments_by_order.get(order_ref, [])
    if not rows:
        return {"found": False, "order_ref": order_ref, "message": "Shipment not found"}

    row = sorted(rows, key=lambda item: (item.get("last_scan_at", ""), item.get("tracking_id", "")))[-1]
    return {
        "found": True,
        "tracking_id": row["tracking_id"],
        "order_ref": row["order_ref"],
        "carrier": row["carrier"],
        "last_scan": row["last_scan"],
        "last_scan_at": row["last_scan_at"],
        "scan_location": row["scan_location"],
    }

