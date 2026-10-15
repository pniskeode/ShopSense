import hashlib
import json
from pathlib import Path

from shopsense.tools import MockShopSenseData, get_refund_amount, lookup_order, refund_or_replace, track_shipment


DATA_DIR = Path(__file__).parents[2] / "data" / "shopsense" / "mock_api"


def _hashes() -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in DATA_DIR.glob("*.json")
    }


def test_lookup_order_joins_customer_and_product() -> None:
    result = lookup_order("KW-O-000000")
    assert result["found"] is True
    assert result["customer"]["customer_ref"] == "KW-C-00317"
    assert result["product"]["sku"] == "KW-SKU-000508"


def test_order_and_shipment_not_found_are_business_results() -> None:
    assert lookup_order("KW-O-DOES-NOT-EXIST")["found"] is False
    assert track_shipment("KW-O-DOES-NOT-EXIST")["found"] is False


def test_track_shipment_returns_known_fields() -> None:
    result = track_shipment("KW-O-000000")
    assert result["found"] is True
    assert set(result) == {"found", "tracking_id", "order_ref", "carrier", "last_scan", "last_scan_at", "scan_location"}


def test_refund_returns_all_duplicate_rows_and_status_summary() -> None:
    result = refund_or_replace("KW-O-000086", "refund")
    assert result["found"] is True
    assert len(result["refunds"]) == 3
    assert [row["requested_on"] for row in result["refunds"]] == sorted(row["requested_on"] for row in result["refunds"])
    assert result["summary"]["processed_amount_inr"] == 25798
    assert result["summary"]["requested_amount_inr"] == 2999


def test_refund_amount_counts_approved_only_and_shows_other_statuses() -> None:
    result = get_refund_amount("KW-O-000086")
    assert result["found"] is True
    assert result["amount_inr"] == result["summary"]["approved_amount_inr"]
    assert result["summary"]["requested_amount_inr"] == 2999
    assert len(result["refunds"]) == 3


def test_refund_replace_is_explicitly_unavailable_and_absence_is_not_eligibility() -> None:
    replace = refund_or_replace("KW-O-000086", "replace")
    assert replace["found"] is False
    assert "not available" in replace["message"]
    missing = refund_or_replace("KW-O-DOES-NOT-EXIST", "refund")
    assert missing["found"] is False
    assert "No refund record" in missing["message"]


def test_tools_do_not_modify_mock_data() -> None:
    before = _hashes()
    data = MockShopSenseData(DATA_DIR)
    lookup_order("KW-O-000000", data)
    track_shipment("KW-O-000000", data)
    refund_or_replace("KW-O-000086", "refund", data)
    get_refund_amount("KW-O-000086", data)
    after = _hashes()
    assert before == after
