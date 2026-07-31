"""Tests 5-7: the rank convention, the hallucination guard, and the blindfold."""

from __future__ import annotations

import json

import pytest

from geo_probe.extract import (
    ExtractParseError,
    assign_ranks,
    brand_is_cited,
    build_extract_payload,
    normalize_for_span_match,
    parse_extract_response,
    shuffled_brand_names,
    span_match_kind,
    validate_evidence_span,
)
from geo_probe.schemas import Brand, RunRecord

BRANDS = [
    Brand(name="Basecamp", domains=["basecamp.com"]),
    Brand(name="Asana", domains=["asana.com"]),
    Brand(name="Trello", domains=["trello.com", "atlassian.com"]),
    Brand(name="ClickUp", domains=["clickup.com"]),
]
NAMES = [b.name for b in BRANDS]


def _mentions(*present: str) -> dict[str, bool]:
    return {n: (n in present) for n in NAMES}


# --------------------------------------------------------------------------
# Test 5: rank convention, table-driven over handcrafted responses
# --------------------------------------------------------------------------

RANK_CASES = [
    (
        "three brands in order",
        "I'd start with Asana, then look at Trello, and Basecamp if you want simplicity.",
        _mentions("Asana", "Trello", "Basecamp"),
        {"Asana": 1, "Trello": 2, "Basecamp": 3, "ClickUp": None},
    ),
    (
        "single brand mentioned gets rank 1",
        "Honestly, Basecamp is all a team that size needs.",
        _mentions("Basecamp"),
        {"Basecamp": 1, "Asana": None, "Trello": None, "ClickUp": None},
    ),
    (
        "no tracked brand mentioned",
        "Any lightweight kanban board will do; pick whichever your team will open daily.",
        _mentions(),
        {"Asana": None, "Basecamp": None, "Trello": None, "ClickUp": None},
    ),
    (
        "rank uses first mention, not later repeats",
        "Trello is the easy answer. Asana is more structured. Trello again if you want free.",
        _mentions("Trello", "Asana"),
        {"Trello": 1, "Asana": 2, "Basecamp": None, "ClickUp": None},
    ),
    (
        "matching is case-insensitive",
        "we use asana here, though CLICKUP came close and trello was the runner-up.",
        _mentions("Asana", "ClickUp", "Trello"),
        {"Asana": 1, "ClickUp": 2, "Trello": 3, "Basecamp": None},
    ),
]


@pytest.mark.parametrize(
    "label,text,mentioned,expected",
    RANK_CASES,
    ids=[c[0] for c in RANK_CASES],
)
def test_rank_convention(label, text, mentioned, expected):
    assert assign_ranks(text, mentioned) == expected


def test_rank_falls_back_to_evidence_span_position():
    """The brand is mentioned under a variant spelling the literal search misses;
    the verbatim span still pins its position."""
    text = "Click-Up is the heavier option, but Asana is the safer default."
    ranks = assign_ranks(
        text,
        _mentions("ClickUp", "Asana"),
        spans={"ClickUp": "Click-Up is the heavier option", "Asana": "Asana is the safer default"},
    )
    assert ranks["ClickUp"] == 1
    assert ranks["Asana"] == 2


def test_rank_is_never_invented():
    """Mentioned but unlocatable: rank stays None rather than getting a guess."""
    ranks = assign_ranks("no brand names here at all", _mentions("Asana"))
    assert ranks["Asana"] is None


# --------------------------------------------------------------------------
# Test 6: evidence_span validator rejects a span not present in the source
# --------------------------------------------------------------------------


def test_evidence_span_validator_rejects_absent_span():
    source = "Basecamp keeps things simple for small teams."
    assert validate_evidence_span(source, "Basecamp keeps things simple") is True
    # A plausible-looking paraphrase that is not actually in the source.
    assert validate_evidence_span(source, "Basecamp is simple and great for small teams") is False
    assert validate_evidence_span(source, None) is True


def test_evidence_span_validator_rejects_overlong_and_empty_spans():
    source = "x" * 500
    assert validate_evidence_span(source, "x" * 201) is False
    assert validate_evidence_span(source, "") is False


# --------------------------------------------------------------------------
# Fix 5: the guard is invariant to rendering, but not to wording
# --------------------------------------------------------------------------


def test_span_matches_through_markdown_emphasis():
    source = "Jasper is **well suited** for marketing teams"
    span = "Jasper is well suited for marketing teams"
    assert span_match_kind(source, span) == "normalized"
    assert validate_evidence_span(source, span) is True


def test_span_matches_through_inline_citation_markers():
    source = "Basecamp[1] is popular with small teams[2][3] that dislike setup."
    span = "Basecamp is popular with small teams that dislike setup."
    assert span_match_kind(source, span) == "normalized"


def test_span_matches_through_line_breaks_and_curly_punctuation():
    source = "Trello’s board view is simple,\n   and the team—ours—liked it."
    span = "Trello's board view is simple, and the team-ours-liked it."
    assert span_match_kind(source, span) == "normalized"


def test_a_fabricated_span_is_still_rejected_after_normalization():
    """The guard must survive the relaxation: a word absent from the source is
    still a failure, no matter how the text is rendered."""
    source = "Asana is **widely used** by product teams."
    assert span_match_kind(source, "Asana is widely used by engineering teams.") is None
    assert span_match_kind(source, "Asana is the best tool available") is None
    assert validate_evidence_span(source, "Asana dominates the market") is False


def test_an_exact_match_is_not_reported_as_normalized():
    source = "Basecamp keeps things simple for small teams."
    assert span_match_kind(source, "Basecamp keeps things simple") == "exact"


def test_normalization_does_not_casefold():
    """Case is wording, not rendering. Folding it would let the guard through on
    a span the model actually altered."""
    source = "Basecamp keeps things simple."
    assert span_match_kind(source, "basecamp keeps things simple") is None


def test_parse_records_whether_the_match_needed_normalization():
    source = "Asana and **Trello** both work well."
    reply = json.dumps(
        {
            "brands": [
                {"brand": "Asana", "mentioned": True, "sentiment": "positive",
                 "evidence_span": "Asana and"},                       # exact
                {"brand": "Trello", "mentioned": True, "sentiment": "positive",
                 "evidence_span": "Trello both work well."},          # needs normalization
                {"brand": "Basecamp", "mentioned": False, "sentiment": None, "evidence_span": None},
                {"brand": "ClickUp", "mentioned": False, "sentiment": None, "evidence_span": None},
            ]
        }
    )
    verdicts = parse_extract_response(reply, BRANDS, source)
    assert verdicts["Asana"].normalized_match is False
    assert verdicts["Trello"].normalized_match is True
    # The raw span is stored unchanged -- normalization is a matching concession,
    # not a rewrite.
    assert verdicts["Trello"].evidence_span == "Trello both work well."
    assert verdicts["Basecamp"].normalized_match is False


def test_normalizer_output_is_what_it_claims():
    assert normalize_for_span_match("a  **b**\n\nc[12]  `d`") == "a b c d"
    assert normalize_for_span_match("  spaced   out  ") == "spaced out"


def test_parse_rejects_hallucinated_span_so_it_gets_retried():
    source = "Asana and Trello both work well."
    reply = json.dumps(
        {
            "brands": [
                {"brand": "Asana", "mentioned": True, "sentiment": "positive",
                 "evidence_span": "Asana is widely regarded as the best"},
                {"brand": "Trello", "mentioned": False, "sentiment": None, "evidence_span": None},
                {"brand": "Basecamp", "mentioned": False, "sentiment": None, "evidence_span": None},
                {"brand": "ClickUp", "mentioned": False, "sentiment": None, "evidence_span": None},
            ]
        }
    )
    with pytest.raises(ExtractParseError, match="not a substring of the text"):
        parse_extract_response(reply, BRANDS, source)


def test_parse_accepts_a_well_formed_reply():
    source = "Asana and Trello both work well."
    reply = "```json\n" + json.dumps(
        {
            "brands": [
                {"brand": "Asana", "mentioned": True, "sentiment": "positive",
                 "evidence_span": "Asana and Trello both work well."},
                {"brand": "Trello", "mentioned": True, "sentiment": "positive",
                 "evidence_span": "Trello both work well"},
                {"brand": "Basecamp", "mentioned": False, "sentiment": None, "evidence_span": None},
                {"brand": "ClickUp", "mentioned": False, "sentiment": None, "evidence_span": None},
            ]
        }
    ) + "\n```"
    verdicts = parse_extract_response(reply, BRANDS, source)
    assert verdicts["Asana"].mentioned is True
    assert verdicts["Basecamp"].mentioned is False
    assert verdicts["Basecamp"].sentiment is None


def test_parse_rejects_a_missing_brand():
    with pytest.raises(ExtractParseError, match="missing brands"):
        parse_extract_response('{"brands": []}', BRANDS, "text")


# --------------------------------------------------------------------------
# Test 7: the extract stage never receives the prompt (or provider, or rep)
# --------------------------------------------------------------------------


def _run() -> RunRecord:
    return RunRecord(
        run_id="11111111-2222-3333-4444-555555555555",
        batch_id="2026-07-28T10-00-00Z",
        prompt_id="p01",
        prompt_text="ZZQPROMPTSENTINELZZQ what is the best project management software",
        intent="listicle",
        provider="perplexity",
        model_requested="sonar-pro",
        model_returned="sonar-pro",
        rep=3,
        temperature=1.0,
        raw_response="For a small team, Asana and Trello are the usual picks.",
        citation_urls=["https://www.asana.com/pricing"],
        latency_ms=2140,
        error=None,
        ts="2026-07-28T10:04:12Z",
    )


def test_extract_payload_never_contains_prompt_text():
    payload = build_extract_payload(_run(), BRANDS)
    blob = json.dumps(payload)
    assert "ZZQPROMPTSENTINELZZQ" not in blob
    assert "what is the best project management software" not in blob


def test_extract_payload_never_leaks_provider_intent_or_rep():
    run = _run()
    payload = build_extract_payload(run, BRANDS)
    blob = json.dumps(payload).lower()
    assert "perplexity" not in blob
    assert "sonar-pro" not in blob
    assert "listicle" not in blob
    assert "p01" not in blob
    assert '"rep"' not in blob
    # It does carry the two things it is allowed to see.
    assert run.raw_response in json.dumps(payload)
    for name in NAMES:
        assert name in json.dumps(payload)


def test_extract_payload_is_deterministic_and_temperature_zero():
    run = _run()
    assert build_extract_payload(run, BRANDS) == build_extract_payload(run, BRANDS)
    assert build_extract_payload(run, BRANDS)["temperature"] == 0.0


def test_brand_order_is_shuffled_per_run_but_reproducible():
    a = shuffled_brand_names(BRANDS, "run-a")
    b = shuffled_brand_names(BRANDS, "run-b")
    assert sorted(a) == sorted(NAMES)
    assert a == shuffled_brand_names(BRANDS, "run-a")  # seeded by run_id
    assert a != b or len(NAMES) < 2  # different runs get different orders


# --------------------------------------------------------------------------
# Citation matching: exact registrable domain, no fuzzy matching
# --------------------------------------------------------------------------


def test_cited_requires_the_brands_own_domain():
    urls = [
        "https://www.asana.com/pricing",
        "https://blog.example.com/asana-vs-trello",
        "https://support.atlassian.com/trello",
    ]
    assert brand_is_cited(urls, ["asana.com"]) == (True, ["https://www.asana.com/pricing"])
    assert brand_is_cited(urls, ["trello.com", "atlassian.com"]) == (
        True,
        ["https://support.atlassian.com/trello"],
    )
    # A third-party page *about* the brand is not the brand being cited.
    assert brand_is_cited(["https://blog.example.com/asana-review"], ["asana.com"]) == (False, [])
    # And a lookalike domain must not match.
    assert brand_is_cited(["https://notasana.com/x"], ["asana.com"]) == (False, [])
