"""The validated structured-intake contract for ShopSense M1."""

from enum import Enum

from pydantic import BaseModel, ConfigDict


class IssueType(str, Enum):
    """Observed ShopSense intent vocabulary used by the M1 application."""

    ACCOUNT_ISSUE = "account_issue"
    CANCEL_ORDER = "cancel_order"
    COMPLAINT = "complaint"
    DAMAGED_DELIVERY = "damaged_delivery"
    DELIVERY_DELAY = "delivery_delay"
    MISSING_ITEM = "missing_item"
    PAYMENT_ISSUE = "payment_issue"
    PRODUCT_INFORMATION = "product_information"
    PRODUCT_SAFETY = "product_safety"
    REFUND_REQUEST = "refund_request"
    RETURN_REQUEST = "return_request"
    TRACK_ORDER = "track_order"
    WARRANTY_CLAIM = "warranty_claim"
    WRONG_ITEM = "wrong_item"


class Sentiment(str, Enum):
    """Observed ShopSense sentiment vocabulary."""

    ANGRY = "angry"
    CALM = "calm"
    DISTRESSED = "distressed"
    FRUSTRATED = "frustrated"


class Urgency(str, Enum):
    """M1 project taxonomy; lowercase output is a design decision."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SupportTicket(BaseModel):
    """Exactly the four fields produced by the M1 structured-intake parser."""

    model_config = ConfigDict(extra="forbid")

    issue_type: IssueType
    order_id: str | None
    sentiment: Sentiment
    urgency: Urgency
