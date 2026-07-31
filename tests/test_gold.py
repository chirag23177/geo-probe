"""Extractor validation scoring.

The central case: Cohen's kappa is undefined when every label falls in one class,
and reporting 1.0 there would dress an uncomputable statistic as a perfect score.
"""

from __future__ import annotations

import csv

import pytest

from geo_probe.extract import GOLD_FIELDS, cohens_kappa, score_gold


def _write_gold(tmp_path, pairs: list[tuple[bool, bool]]):
    """pairs of (extractor, human)."""
    p = tmp_path / "gold.csv"
    with open(p, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=GOLD_FIELDS)
        w.writeheader()
        for i, (machine, human) in enumerate(pairs):
            w.writerow({
                "run_id": f"r{i}", "brand": "Asana",
                "mentioned_extractor": str(machine).lower(),
                "mentioned_human": str(human).lower(),
                "response_text": "Asana is mentioned here.",
            })
    return p


# --------------------------------------------------------------------------
# Kappa
# --------------------------------------------------------------------------


def test_kappa_is_undefined_when_every_label_is_one_class():
    """p_e = 1.0, so there is no agreement-above-chance to measure. This is the
    real case the recovered gold sample hit: a span only exists when the brand
    was mentioned, so that subset can contain nothing but positives."""
    assert cohens_kappa([True] * 10, [True] * 10) is None


def test_kappa_is_undefined_even_when_the_raters_disagree_within_one_class():
    assert cohens_kappa([True] * 10, [True] * 9 + [True]) is None


def test_kappa_is_defined_with_both_classes_present():
    kappa = cohens_kappa([True] * 27 + [False] * 3, [True] * 27 + [False] * 3)
    assert kappa == pytest.approx(1.0)


def test_kappa_is_zero_at_chance():
    """Raters agreeing exactly as often as chance predicts scores 0."""
    machine = [True, True, False, False]
    human = [True, False, True, False]
    assert cohens_kappa(machine, human) == pytest.approx(0.0)


def test_kappa_is_negative_below_chance():
    machine = [True, True, False, False]
    human = [False, False, True, True]
    kappa = cohens_kappa(machine, human)
    assert kappa is not None and kappa < 0


def test_kappa_rejects_mismatched_or_empty_input():
    with pytest.raises(ValueError):
        cohens_kappa([True], [True, False])
    with pytest.raises(ValueError):
        cohens_kappa([], [])


# --------------------------------------------------------------------------
# score_gold
# --------------------------------------------------------------------------


def test_score_gold_reports_class_balance_so_a_degenerate_kappa_is_visible(tmp_path):
    path = _write_gold(tmp_path, [(True, True)] * 10)
    score = score_gold(path)
    assert score.n == 10
    assert score.n_human_true == 10
    assert score.n_human_false == 0
    assert score.raw_agreement == pytest.approx(1.0)
    assert score.kappa is None


def test_score_gold_computes_kappa_when_both_classes_are_present(tmp_path):
    path = _write_gold(tmp_path, [(True, True)] * 27 + [(False, False)] * 3)
    score = score_gold(path)
    assert score.n_human_true == 27 and score.n_human_false == 3
    assert score.kappa == pytest.approx(1.0)


def test_score_gold_collects_disagreements(tmp_path):
    path = _write_gold(tmp_path, [(True, True), (True, False), (False, True)])
    score = score_gold(path)
    assert score.raw_agreement == pytest.approx(1 / 3)
    assert len(score.disagreements) == 2
    assert score.disagreements[0][2] is True and score.disagreements[0][3] is False


def test_score_gold_ignores_unlabelled_rows(tmp_path):
    path = _write_gold(tmp_path, [(True, True)] * 3)
    rows = list(csv.DictReader(open(path, encoding="utf-8", newline="")))
    rows.append({**rows[0], "run_id": "blank", "mentioned_human": ""})
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=GOLD_FIELDS)
        w.writeheader()
        w.writerows(rows)
    assert score_gold(path).n == 3


def test_score_gold_raises_when_nothing_is_labelled(tmp_path):
    path = _write_gold(tmp_path, [])
    with pytest.raises(ValueError, match="no filled mentioned_human"):
        score_gold(path)
