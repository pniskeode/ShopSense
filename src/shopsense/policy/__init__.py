"""Qdrant-backed policy retrieval for ShopSense M3 Stage 2."""

from .service import (
    DEFAULT_POLICY_COLLECTION,
    PolicyChunk,
    PolicyHit,
    PolicyIndexStats,
    PolicyService,
    PolicyUnavailableError,
    create_m4_policy_service,
    create_m5_policy_service,
    create_m6_policy_service,
    create_m7_policy_service,
    create_m8_policy_service,
    create_policy_service,
    is_policy_question,
    load_policy_chunks,
    policy_configuration_is_present,
)
from .evaluation import DEFAULT_GOLDEN_SET, EvaluationReport, evaluate_policy_rag, load_golden_set
from .grounding import GroundingReport, is_safe_policy_abstention, validate_answer_grounding
from .hybrid import CrossEncoderReranker

__all__ = [
    "DEFAULT_POLICY_COLLECTION",
    "PolicyChunk",
    "PolicyHit",
    "PolicyIndexStats",
    "PolicyService",
    "PolicyUnavailableError",
    "create_m4_policy_service",
    "create_m5_policy_service",
    "create_m6_policy_service",
    "create_m7_policy_service",
    "create_m8_policy_service",
    "create_policy_service",
    "is_policy_question",
    "load_policy_chunks",
    "policy_configuration_is_present",
    "CrossEncoderReranker",
    "DEFAULT_GOLDEN_SET",
    "EvaluationReport",
    "GroundingReport",
    "is_safe_policy_abstention",
    "evaluate_policy_rag",
    "load_golden_set",
    "validate_answer_grounding",
]
