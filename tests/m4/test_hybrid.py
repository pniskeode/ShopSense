from __future__ import annotations

from shopsense.policy.hybrid import hybrid_search
from shopsense.policy.service import PolicyChunk, PolicyHit


class StubReranker:
    def score(self, query: str, passages: list[str]) -> list[float]:
        del query
        return [10.0 if "authoritative" in passage else 1.0 for passage in passages]


def _chunk(point_id: str, text: str, *, source_format: str = "markdown") -> PolicyChunk:
    return PolicyChunk(
        point_id=point_id,
        text=text,
        metadata={
            "policy_slug": "returns-policy",
            "policy_name": "Returns",
            "policy_section": "policy",
            "source_format": source_format,
            "chunk_id": point_id,
            "logical_chunk_id": "same-rule" if "30 days" in text else point_id,
            "citation_id": f"returns-policy:policy:{source_format}:0",
        },
    )


def test_hybrid_search_fuses_dense_and_bm25_then_reranks() -> None:
    chunks = [
        _chunk("dense", "The normal return window is 30 days."),
        _chunk("lexical", "authoritative return policy: custom purchases are non-returnable."),
    ]
    dense = [PolicyHit(chunks[0].text, 0.9, dict(chunks[0].metadata), "dense")]

    hits = hybrid_search(
        "Are custom purchases returnable?",
        dense_hits=dense,
        chunks=chunks,
        limit=2,
        reranker=StubReranker(),
    )

    assert hits[0].point_id == "lexical"
    assert {hit.point_id for hit in hits} == {"dense", "lexical"}


def test_hybrid_search_deduplicates_markdown_and_pdf_logical_chunks() -> None:
    markdown = _chunk("markdown", "The normal return window is 30 days.", source_format="markdown")
    pdf = _chunk("pdf", "The normal return window is 30 days.", source_format="pdf")

    hits = hybrid_search(
        "What is the normal return window?",
        dense_hits=[
            PolicyHit(markdown.text, 0.9, dict(markdown.metadata), "markdown"),
            PolicyHit(pdf.text, 0.8, dict(pdf.metadata), "pdf"),
        ],
        chunks=[markdown, pdf],
        limit=3,
        reranker=None,
    )

    assert len(hits) == 1
    assert hits[0].metadata["source_format"] == "markdown"
