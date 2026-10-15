"""Order lookup tool."""

from typing import Any

from .data import MockShopSenseData


def lookup_order(order_ref: str, data: MockShopSenseData | None = None) -> dict[str, Any]:
    """Retrieve order details joined with the related customer and product."""

    data = data or MockShopSenseData()
    order_ref = order_ref.strip() if isinstance(order_ref, str) else ""
    order = data.orders_by_ref.get(order_ref)
    if order is None:
        return {"found": False, "order_ref": order_ref, "message": "Order not found"}

    customer = data.customers_by_ref.get(order["customer_ref"])
    product = data.products_by_sku.get(order["sku"])
    return {
        "found": True,
        "order_ref": order_ref,
        "customer": {
            key: customer[key]
            for key in ("customer_ref", "display_name", "city", "tier", "flagged_for_abuse")
        }
        if customer
        else None,
        "product": {
            key: product[key]
            for key in ("sku", "title", "category", "price_inr", "warranty_months")
        }
        if product
        else None,
        "quantity": order["quantity"],
        "order_value_inr": order["order_value_inr"],
        "placed_at": order["placed_at"],
        "payment_method": order["payment_method"],
        "status": order["status"],
        "delivery_promise_days": order["delivery_promise_days"],
        "actual_delivery_days": order["actual_delivery_days"],
    }

