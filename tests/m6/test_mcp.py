from __future__ import annotations

import json

from shopsense.mcp.server import MockApiGateway, mock_api_contract
from shopsense.tools.data import MockShopSenseData


def test_gateway_order_only_lookup_omits_customer_profile_and_internal_abuse_flag() -> None:
    data = MockShopSenseData()
    order = data.orders[0]
    gateway = MockApiGateway(data)

    result = gateway.lookup_order(order["order_ref"])

    assert result["found"] is True
    assert "customer" not in result


def test_gateway_redacts_internal_refund_approver_and_contract_declares_read_only() -> None:
    data = MockShopSenseData()
    order = next(row for row in data.orders if data.refunds_by_order.get(row["order_ref"]))

    result = MockApiGateway(data).get_refund_amount(order["order_ref"], order["customer_ref"])

    assert all("approved_by" not in refund for refund in result["refunds"])
    contract = json.loads(mock_api_contract())
    assert contract["mode"] == "read_only"
    assert contract["required_identity"] == ["order_ref"]
    assert contract["optional_identity"] == ["customer_ref"]
    assert "get_refund_amount" in contract["tools"]


def test_gateway_does_not_disclose_another_customers_order() -> None:
    data = MockShopSenseData()
    order = data.orders[0]
    different_customer = next(row["customer_ref"] for row in data.customers if row["customer_ref"] != order["customer_ref"])

    result = MockApiGateway(data).lookup_order(order["order_ref"], different_customer)

    assert result == {
        "found": False,
        "error": "order_not_found_for_customer",
        "order_ref": order["order_ref"],
        "message": "Order was not found for this customer.",
    }
