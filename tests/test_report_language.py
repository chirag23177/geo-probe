"""Fix 1: the generated prose must not lie, in either of two ways.

1a. A number and the label attached to it must come from the same record.
1b. No sentence may assert that a change below the MDE is absent. "Below the
    MDE" means indistinguishable at this sample size; asserting it *is* noise is
    accepting the null hypothesis, which is the overclaim this tool criticises.
"""

from __future__ import annotations

import re

import pytest

from geo_probe.report import render_findings
from geo_probe.schemas import AggRecord

# Constructions that accept the null. Checked case-insensitively against the
# whole generated file.
BANNED_PHRASES = [
    "is sampling noise",
    "is flagging sampling noise",
    "is reporting sampling noise",
    "is just noise",
    "proves no change",
    "shows no change",
    "no change occurred",
    "there was no change",
    "confirms the null",
]


def _record(brand: str, provider: str, rate: float, mde: float, deff: float, **kw) -> AggRecord:
    """An aggregate record with deliberately distinguishable numbers."""
    defaults = dict(
        model_returned="model-x",
        n_prompts=20,
        k=5,
        n_runs_used=100,
        n_runs_excluded=0,
        ci95_cluster=(max(0.0, rate - 0.1), min(1.0, rate + 0.1)),
        ci95_naive_wrong=(max(0.0, rate - 0.05), min(1.0, rate + 0.05)),
        var_between_prompt_share=0.40,
        var_within_prompt_share=0.60,
        flip_rate=0.30,
        mean_rank=2.0,
        rank_stdev=1.0,
        rank_n=50,
        mde_alpha=0.05,
        mde_power=0.80,
        se_naive=0.02,
        mde_abs_pp_naive=mde / 2,
        mde_inflation=2.0,
        headroom_up=1.0 - rate,
        headroom_down=rate,
        mde_interpretable="both",
        ci_at_boundary="none",
        k_effective=5.0,
        deff_predicted=deff,
        deff_residual=0.0,
    )
    defaults.update(kw)
    return AggRecord(
        brand=brand, provider=provider, mention_rate=rate, mde_abs_pp=mde,
        design_effect=deff, **defaults,
    )


def _two_distinguishable_cells() -> list[AggRecord]:
    """Alpha and Bravo share no numeric values, so any cross-pairing is visible."""
    return [
        _record("Alpha", "anthropic", rate=0.20, mde=0.11, deff=1.10,
                mde_abs_pp_naive=0.055, mde_inflation=2.0),
        _record("Bravo", "perplexity", rate=0.70, mde=0.44, deff=3.30,
                mde_abs_pp_naive=0.22, mde_inflation=2.0),
    ]


# --------------------------------------------------------------------------
# 1b: banned language
# --------------------------------------------------------------------------


@pytest.mark.parametrize("phrase", BANNED_PHRASES)
def test_generated_findings_never_accept_the_null(phrase):
    text = render_findings(_two_distinguishable_cells(), "test category", "B1").lower()
    assert phrase not in text, f"generated prose contains banned phrase {phrase!r}"


def test_generated_findings_state_the_correct_form():
    """The replacement must actually be present, not merely the banned form absent."""
    text = render_findings(_two_distinguishable_cells(), "test category", "B1").lower()
    assert "cannot be distinguished from sampling noise" in text
    assert "not evidence that no move occurred" in text


def test_banned_phrase_check_can_actually_fail():
    """Guard against a vacuous test: the detector must fire on a real violation."""
    bad = "a dashboard that flags a 19.0pp shift is flagging sampling noise"
    assert any(p in bad.lower() for p in BANNED_PHRASES)


# --------------------------------------------------------------------------
# 1a: a number must never be paired with another cell's name
# --------------------------------------------------------------------------

_PP = re.compile(r"(\d+\.\d)pp")
_SENTENCE_SPLIT = re.compile(r"(?<=[.;])\s+")


def _prose_sentences(text: str) -> list[str]:
    """Sentences from the narrative section only -- the table is row-structured
    and pairs numbers with names by construction."""
    body = text.split("## Findings", 1)[1].split("## Per-cell numbers", 1)[0]
    return [s for s in _SENTENCE_SPLIT.split(body) if s.strip()]


def test_no_sentence_pairs_one_cells_number_with_another_cells_name():
    records = _two_distinguishable_cells()
    text = render_findings(records, "test category", "B1")

    # Every pp figure each brand is allowed to be quoted with.
    allowed: dict[str, set[str]] = {}
    for r in records:
        allowed[r.brand] = {
            f"{r.mde_abs_pp * 100:.1f}",
            f"{r.mde_abs_pp_naive * 100:.1f}",
            f"{r.headroom_up * 100:.1f}",
            f"{r.headroom_down * 100:.1f}",
        }

    for sentence in _prose_sentences(text):
        named = [b for b in allowed if b in sentence]
        found = set(_PP.findall(sentence))
        if not named or not found:
            continue
        permitted = set().union(*(allowed[b] for b in named))
        stray = found - permitted
        assert not stray, (
            f"sentence quotes {sorted(stray)}pp but only names {named}, "
            f"whose own values are {sorted(permitted)}: {sentence!r}"
        )


def test_cross_pairing_check_can_actually_fail():
    """The detector must fire when a number really does belong to another cell."""
    records = _two_distinguishable_cells()
    allowed = {"Alpha": {"11.0"}, "Bravo": {"44.0"}}
    forged = "a dashboard flagging a 44.0pp move on Alpha is at the limit."
    named = [b for b in allowed if b in forged]
    found = set(_PP.findall(forged))
    assert found - set().union(*(allowed[b] for b in named))
    assert records  # the fixture is the one under test elsewhere


def test_every_mde_in_the_prose_carries_its_own_cell_name():
    """The structural guarantee behind 1a: MDE figures are only ever emitted by
    _Cell.mde_phrase, which always appends `on <brand>/<provider>`."""
    records = _two_distinguishable_cells()
    text = render_findings(records, "test category", "B1")
    body = text.split("## Findings", 1)[1].split("## Per-cell numbers", 1)[0]
    for r in records:
        phrase = f"{r.mde_abs_pp * 100:.1f}pp on {r.brand}/{r.provider}"
        assert phrase in body, f"expected {phrase!r} in the findings paragraph"


# --------------------------------------------------------------------------
# Fix 7: exclusions reported at the top, broken down by provider
# --------------------------------------------------------------------------


def test_exclusions_are_reported_per_provider_near_the_top():
    records = [
        _record("Alpha", "anthropic", 0.20, 0.11, 1.10, n_runs_used=100, n_runs_excluded=0),
        _record("Bravo", "anthropic", 0.70, 0.44, 3.30, n_runs_used=100, n_runs_excluded=0),
        _record("Alpha", "perplexity", 0.20, 0.11, 1.10, n_runs_used=71, n_runs_excluded=29),
        _record("Bravo", "perplexity", 0.70, 0.44, 3.30, n_runs_used=71, n_runs_excluded=29),
    ]
    text = render_findings(records, "test category", "B1")
    head = text.split("## Findings", 1)[0]
    assert "0/100 anthropic runs" in head
    assert "29/100 perplexity runs (58 run-brand pairs)" in head


def test_boundary_cells_are_daggered_with_a_footnote():
    records = [
        _record("Alpha", "anthropic", 0.96, 0.30, 2.0, headroom_up=0.04,
                mde_interpretable="down_only", ci_at_boundary="upper"),
        _record("Bravo", "anthropic", 0.50, 0.10, 2.0),
    ]
    text = render_findings(records, "test category", "B1")
    assert "30.0pp†" in text
    assert "interpret one-sided" in text


# --------------------------------------------------------------------------
# Superlatives must never name a degenerate cell
# --------------------------------------------------------------------------


def test_superlatives_skip_cells_whose_interval_touches_a_boundary():
    """A cell on a bound has a compressed bootstrap, so it can win 'smallest MDE'
    by being the most broken cell in the table. Calling it best-measured inverts
    the truth, so it must not be selected."""
    records = [
        # Smallest MDE in the table, but its interval terminates at 1.00.
        _record("Degenerate", "anthropic", rate=0.96, mde=0.05, deff=1.9,
                ci95_cluster=(0.88, 1.00), ci_at_boundary="upper",
                mde_interpretable="down_only", headroom_up=0.04),
        _record("Clean", "anthropic", rate=0.50, mde=0.21, deff=2.8),
        _record("Worst", "perplexity", rate=0.40, mde=0.35, deff=3.0),
    ]
    body = render_findings(records, "test category", "B1").split("## Per-cell numbers", 1)[0]
    assert "5.0pp on Degenerate/anthropic" not in body
    assert "21.0pp on Clean/anthropic" in body, "best-measured must be the cleanest cell"
    assert "35.0pp on Worst/perplexity" in body


def test_superlative_falls_back_when_every_cell_is_on_a_boundary():
    """Degraded but not silent: if there is no clean cell, one is still named."""
    records = [
        _record("A", "anthropic", rate=0.96, mde=0.05, deff=1.9, ci_at_boundary="upper"),
        _record("B", "anthropic", rate=0.02, mde=0.30, deff=2.0, ci_at_boundary="lower"),
    ]
    body = render_findings(records, "test category", "B1").split("## Per-cell numbers", 1)[0]
    assert "5.0pp on A/anthropic" in body


def test_findings_explains_why_superlatives_exclude_boundary_cells():
    records = [
        _record("Degenerate", "anthropic", rate=0.96, mde=0.05, deff=1.9, ci_at_boundary="upper"),
        _record("Clean", "anthropic", rate=0.50, mde=0.21, deff=2.8),
    ]
    text = render_findings(records, "test category", "B1")
    assert "degenerate at a bound" in text
    assert "most degenerate one" in text


def test_no_footnote_when_every_cell_is_interpretable():
    text = render_findings(_two_distinguishable_cells(), "test category", "B1")
    assert "interpret one-sided" not in text
    assert "†" not in text
