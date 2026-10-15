"""Read-only access to the copied ShopSense mock API tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class MockShopSenseData:
    """Load the five JSON tables once and expose lookup-only indexes."""

    def __init__(self, data_dir: Path | None = None) -> None:
        if data_dir is None:
            data_dir = Path(__file__).resolve().parents[3] / "data" / "shopsense" / "mock_api"
        self.data_dir = Path(data_dir)
        self.customers = self._load("customers.json")
        self.orders = self._load("orders.json")
        self.products = self._load("products.json")
        self.shipments = self._load("shipments.json")
        self.refunds = self._load("refunds.json")

        self.customers_by_ref = {row["customer_ref"]: row for row in self.customers}
        self.orders_by_ref = {row["order_ref"]: row for row in self.orders}
        self.products_by_sku = {row["sku"]: row for row in self.products}
        self.shipments_by_order = {}
        for row in self.shipments:
            self.shipments_by_order.setdefault(row["order_ref"], []).append(row)
        self.refunds_by_order = {}
        for row in self.refunds:
            self.refunds_by_order.setdefault(row["order_ref"], []).append(row)

    def _load(self, filename: str) -> list[dict[str, Any]]:
        path = self.data_dir / filename
        with path.open("r", encoding="utf-8") as stream:
            rows = json.load(stream)
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ValueError(f"Expected a JSON array of objects in {path}")
        return rows

