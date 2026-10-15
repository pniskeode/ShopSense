"""M4 hybrid policy retrieval: BM25, reciprocal-rank fusion, and reranking."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol

from .service import PolicyChunk, PolicyHit


RRF_K = 60
DENSE_CANDIDATE_LIMIT = 10
LEXICAL_CANDIDATE_LIMIT = 10
RERANK_CANDIDATE_LIMIT = 10


class PolicyReranker(Protocol):
    """A small adapter so unit tests never download the cross-encoder."""

    def score(self, query: str, passages: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    """Lazy Colab-aligned local cross-encoder reranker."""

    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or self.model_name
        self._model: Any | None = None

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:  # pragma: no cover - installation guidance
                raise RuntimeError(
                    "M4 reranking dependencies are missing; run python -m pip install -e ."
                ) from exc
            self._model = CrossEncoder(self.model_name)
        return [float(score) for score in self._model.predict([(query, passage) for passage in passages])]


@dataclass(frozen=True)
class RankedCandidate:
    hit: PolicyHit
    rrf_score: float


def hybrid_search(
    query: str,
    *,
    dense_hits: Iterable[PolicyHit],
    chunks: Iterable[PolicyChunk],
    limit: int,
    reranker: PolicyReranker | None,
    filters: dict[str, str] | None = None,
) -> list[PolicyHit]:
    """Fuse dense and lexical rankings, then rerank a small, deduplicated set."""

    # Old M3 points predate M4's stable citation metadata. They are ignored
    # rather than allowed to produce an uncitable customer-facing assertion;
    # rebuilding the collection restores their dense contribution.
    dense = [hit for hit in dense_hits if hit.metadata.get("citation_id")][:DENSE_CANDIDATE_LIMIT]
    local_chunks = _filter_chunks(chunks, filters)
    lexical = _bm25_hits(query, local_chunks, limit=LEXICAL_CANDIDATE_LIMIT)
    fused = _reciprocal_rank_fusion(dense, lexical)
    if not fused:
        return []

    candidates = fused[:RERANK_CANDIDATE_LIMIT]
    if reranker is not None:
        rerank_scores = reranker.score(query, [candidate.hit.text for candidate in candidates])
        if len(rerank_scores) != len(candidates):
            raise RuntimeError("M4 reranker returned an unexpected number of scores")
        candidates = [
            RankedCandidate(hit=candidate.hit, rrf_score=score)
            for candidate, score in zip(candidates, rerank_scores, strict=True)
        ]
        candidates.sort(key=lambda candidate: candidate.rrf_score, reverse=True)

    return _deduplicate_logical_chunks([candidate.hit for candidate in candidates], limit=limit)


def _filter_chunks(chunks: Iterable[PolicyChunk], filters: dict[str, str] | None) -> list[PolicyChunk]:
    if not filters:
        return list(chunks)
    return [
        chunk
        for chunk in chunks
        if all(str(chunk.metadata.get(key, "")) == value for key, value in filters.items())
    ]


def _bm25_hits(query: str, chunks: list[PolicyChunk], *, limit: int) -> list[PolicyHit]:
    if not chunks:
        return []
    documents = [_tokenize(chunk.text) for chunk in chunks]
    tokens = _tokenize(query)
    try:
        from rank_bm25 import BM25Okapi

        scores = [float(score) for score in BM25Okapi(documents).get_scores(tokens)]
    except ImportError:  # Kept deterministic for source-only inspection before dependencies install.
        query_tokens = set(tokens)
        scores = [float(len(query_tokens.intersection(document))) for document in documents]
    ranked_indices = sorted(range(len(chunks)), key=lambda index: scores[index], reverse=True)
    hits: list[PolicyHit] = []
    query_token_set = set(tokens)
    for index in ranked_indices:
        # With very small corpora BM25's IDF can legitimately be zero even
        # for a lexical overlap. Keep that candidate for fusion/reranking,
        # but never add an entirely unrelated zero-score document.
        if scores[index] <= 0 and not query_token_set.intersection(documents[index]):
            continue
        chunk = chunks[index]
        hits.append(
            PolicyHit(text=chunk.text, score=scores[index], metadata=dict(chunk.metadata), point_id=chunk.point_id)
        )
        if len(hits) >= limit:
            break
    return hits


def _reciprocal_rank_fusion(*rankings: list[PolicyHit]) -> list[RankedCandidate]:
    candidates: dict[str, RankedCandidate] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            identity = hit.point_id or str(hit.metadata.get("chunk_id") or hit.metadata.get("content_hash") or hit.text)
            existing = candidates.get(identity)
            contribution = 1.0 / (RRF_K + rank)
            if existing is None:
                candidates[identity] = RankedCandidate(hit=hit, rrf_score=contribution)
            else:
                candidates[identity] = RankedCandidate(hit=existing.hit, rrf_score=existing.rrf_score + contribution)
    return sorted(candidates.values(), key=lambda candidate: candidate.rrf_score, reverse=True)


def _deduplicate_logical_chunks(hits: Iterable[PolicyHit], *, limit: int) -> list[PolicyHit]:
    unique: list[PolicyHit] = []
    seen: set[str] = set()
    for hit in hits:
        identity = str(hit.metadata.get("logical_chunk_id") or re.sub(r"\s+", " ", hit.text).lower())
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(hit)
        if len(unique) >= limit:
            break
    return unique


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())
