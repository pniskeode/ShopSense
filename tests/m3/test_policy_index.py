from __future__ import annotations

from shopsense.policy.service import PolicyHit, PolicyService, _apply_grocery_return_override, load_policy_chunks


def test_m3_policy_corpus_loads_every_declared_format_with_traceable_metadata() -> None:
    chunks = load_policy_chunks()
    required_metadata = {
        "policy_name",
        "policy_slug",
        "policy_section",
        "source_format",
        "source_file",
        "chunk_index",
        "chunk_id",
        "logical_chunk_id",
        "citation_id",
        "content_hash",
    }

    assert any(chunk.metadata["source_format"] == "markdown" for chunk in chunks)
    assert any(chunk.metadata["source_format"] == "pdf" for chunk in chunks)
    assert any(chunk.metadata["policy_slug"] == "grocery-return-override" for chunk in chunks)
    assert all(required_metadata.issubset(chunk.metadata) for chunk in chunks)
    assert len({chunk.point_id for chunk in chunks}) == len(chunks)
    assert max(len(chunk.text) for chunk in chunks) <= 900


def test_grocery_override_suppresses_space_and_hyphenated_conflicting_windows() -> None:
    hits = [
        PolicyHit("Groceries have a 14 days return window.", 1.0, {"policy_slug": "category-groceries"}),
        PolicyHit("Groceries have a 14-day return window.", 0.9, {"policy_slug": "category-groceries"}),
    ]

    safe_hits = _apply_grocery_return_override("Can I return groceries?", hits, limit=3)

    assert safe_hits[0].metadata["policy_slug"] == "grocery-return-override"
    assert all(hit.metadata.get("policy_slug") != "category-groceries" for hit in safe_hits)


def test_policy_context_renders_a_copyable_canonical_citation() -> None:
    service = object.__new__(PolicyService)
    context = service.format_context(
        [
            PolicyHit(
                "Home & Kitchen items have a 45-day return window.",
                1.0,
                {
                    "citation_id": "category-home-kitchen:page-1:pdf:0",
                    "policy_name": "Category Policy Addendum: Home & Kitchen",
                    "page_number": 1,
                    "source_format": "pdf",
                },
            )
        ]
    )

    assert "Citation: [SOURCE:category-home-kitchen:page-1:pdf:0]" in context
    assert "[SOURCE:category-home-kitchen:page-1:pdf:0;" not in context
