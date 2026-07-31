"""Stage 2: deterministic grading of non-deterministic output.

A separate LLM pass at temperature=0, strict JSON, one call per run scoring all
brands at once. The extractor sees only the raw response text and the brand list.
It never sees the prompt, the provider, the rep index, or which brand is the
focus -- if it did, the grading would be contaminated by the thing being measured.

Rank and `cited` are computed in code, not returned by the model. The extractor
decides *whether* a brand was mentioned and supplies verbatim evidence; the rank
convention is then applied mechanically so it is enforced rather than trusted.
"""

from __future__ import annotations

import csv
import json
import random
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence
from urllib.parse import urlparse

import anthropic

from .providers.base import require_env
from .schemas import (
    Brand,
    ExperimentConfig,
    ExtractRecord,
    RunRecord,
    append_jsonl,
    read_extracts,
    read_runs,
)

EXTRACTS_DIR = Path("data/extracts")
GOLD_DIR = Path("data/gold")
GOLD_TEMPLATE = GOLD_DIR / "gold_template.csv"
GOLD_RECOVERED = GOLD_DIR / "gold_recovered.csv"
"""Runs that only graded after normalization get their own gold file. They are
exactly the runs most likely to be mis-extracted, so their agreement is reported
separately -- merging them into one number would hide whichever is worse."""

MAX_SPAN_CHARS = 200
EXTRACTOR_MAX_TOKENS = 1024

SYSTEM_PROMPT = (
    "You are a strict annotation tool. You are given the text of a single "
    "response from some assistant, and a list of brand names. For each brand in "
    "the list, decide whether that brand is mentioned in the text.\n"
    "\n"
    "Rules:\n"
    "- A brand counts as mentioned if the text refers to that product by name, "
    "including obvious spelling and spacing variants.\n"
    "- Do not infer a mention from a generic description. The name must appear.\n"
    "- sentiment describes how the text portrays the brand: positive, neutral, "
    "or negative. Use null when the brand is not mentioned.\n"
    "- evidence_span must be copied VERBATIM from the text, character for "
    "character, at most 200 characters, and must contain the mention. Use null "
    "when the brand is not mentioned. Do not paraphrase, do not fix typos, do "
    "not add ellipses.\n"
    "\n"
    "Reply with JSON only. No prose, no markdown fences. Shape:\n"
    '{"brands": [{"brand": "<exact name from the list>", "mentioned": <bool>, '
    '"sentiment": "positive"|"neutral"|"negative"|null, '
    '"evidence_span": "<verbatim substring>"|null}]}\n'
    "Include every brand from the list exactly once."
)


def extracts_path(batch_id: str) -> Path:
    return EXTRACTS_DIR / f"{batch_id}.jsonl"


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------


def shuffled_brand_names(brands: Sequence[Brand], run_id: str) -> list[str]:
    """Randomize brand order per call to neutralize position bias in the
    extractor, seeded by run_id so the shuffle is reproducible."""
    names = [b.name for b in brands]
    random.Random(run_id).shuffle(names)
    return names


def build_extract_payload(run: RunRecord, brands: Sequence[Brand]) -> dict:
    """The exact request body sent to the extractor.

    Deliberately excludes prompt_text, intent, provider, rep, and any notion of a
    focus brand. tests/test_extract.py asserts on this.
    """
    if run.raw_response is None:
        raise ValueError("cannot build an extract payload for a failed run")
    names = shuffled_brand_names(brands, run.run_id)
    user = (
        "BRANDS:\n"
        + "\n".join(f"- {n}" for n in names)
        + "\n\nTEXT:\n<<<\n"
        + run.raw_response
        + "\n>>>\n\nReturn the JSON object now."
    )
    return {
        "model": "",  # filled in by the caller
        "max_tokens": EXTRACTOR_MAX_TOKENS,
        "temperature": 0.0,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user}],
    }


# --------------------------------------------------------------------------
# Response parsing and validation
# --------------------------------------------------------------------------


class ExtractParseError(Exception):
    """Malformed extractor output. The message is fed back on retry."""


_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)
_VALID_SENTIMENT = {"positive", "neutral", "negative"}

_CITATION_MARKER = re.compile(r"\[\d+\]")
_MARKDOWN_MARKER = re.compile(r"\*\*|__|[*_`]")
_WHITESPACE = re.compile(r"\s+")
_PUNCT_FOLD = {
    "‘": "'", "’": "'",  # curly single quotes
    "“": '"', "”": '"',  # curly double quotes
    "–": "-", "—": "-",  # en dash, em dash
}


def normalize_for_span_match(text: str) -> str:
    """Strip rendering, keep wording.

    The hallucination guard should be invariant to how text is rendered. A model
    asked to quote `Jasper is **well suited**` will typically return `Jasper is
    well suited` -- it copied what the markdown means, not what it says. That is
    a formatting mismatch, not a fabricated quote, and failing it discards real
    data.

    What this deliberately does NOT do: casefold, fuzzy-match, compute edit
    distance, or accept partial overlap. After normalization the span must still
    be an exact substring. A span containing a word that is not in the source
    still fails, which is the property the guard exists for.
    """
    t = unicodedata.normalize("NFKC", text)
    for src, dst in _PUNCT_FOLD.items():
        t = t.replace(src, dst)
    t = _CITATION_MARKER.sub("", t)
    t = _MARKDOWN_MARKER.sub("", t)
    return _WHITESPACE.sub(" ", t).strip()


def span_match_kind(raw_response: str, span: str | None) -> str | None:
    """How the span matched: "exact", "normalized", or None for no match.

    Length is checked against the raw span, since MAX_SPAN_CHARS is a limit on
    what the model was asked to return, not on the normalized form.
    """
    if span is None:
        return "exact"
    if not span or len(span) > MAX_SPAN_CHARS:
        return None
    if span in raw_response:
        return "exact"
    if normalize_for_span_match(span) in normalize_for_span_match(raw_response):
        return "normalized"
    return None


def validate_evidence_span(raw_response: str, span: str | None) -> bool:
    """A span must be a substring of the source text, modulo rendering.

    Cheap, strong hallucination guard: an extractor that invents a quote fails
    here and gets retried.
    """
    return span_match_kind(raw_response, span) is not None


@dataclass(frozen=True)
class BrandVerdict:
    brand: str
    mentioned: bool
    sentiment: str | None
    evidence_span: str | None
    normalized_match: bool = False
    """True when the span matched only after normalization -- the model copied
    rendered text. Lets the recovered runs be audited as their own subset."""


def parse_extract_response(text: str, brands: Sequence[Brand], raw_response: str) -> dict[str, BrandVerdict]:
    stripped = _FENCE.sub("", text.strip())
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ExtractParseError(f"not valid JSON: {exc}") from exc

    if not isinstance(payload, dict) or not isinstance(payload.get("brands"), list):
        raise ExtractParseError('top level must be an object with a "brands" array')

    expected = {b.name for b in brands}
    verdicts: dict[str, BrandVerdict] = {}
    for item in payload["brands"]:
        if not isinstance(item, dict):
            raise ExtractParseError("each entry in brands must be an object")
        name = item.get("brand")
        if name not in expected:
            raise ExtractParseError(f"unknown brand {name!r}; expected one of {sorted(expected)}")
        if name in verdicts:
            raise ExtractParseError(f"brand {name!r} appears more than once")
        mentioned = item.get("mentioned")
        if not isinstance(mentioned, bool):
            raise ExtractParseError(f"{name}: mentioned must be a boolean")
        sentiment = item.get("sentiment")
        span = item.get("evidence_span")
        normalized_match = False
        if mentioned:
            if sentiment not in _VALID_SENTIMENT:
                raise ExtractParseError(f"{name}: sentiment must be one of {sorted(_VALID_SENTIMENT)}")
            if not isinstance(span, str):
                raise ExtractParseError(f"{name}: evidence_span is required when mentioned is true")
            kind = span_match_kind(raw_response, span)
            if kind is None:
                raise ExtractParseError(
                    f"{name}: evidence_span is not a substring of the text, even after "
                    f"ignoring markdown, citation markers and whitespace (or it exceeds "
                    f"{MAX_SPAN_CHARS} chars). Copy it exactly."
                )
            normalized_match = kind == "normalized"
        else:
            sentiment = None
            span = None
        # The raw span is stored unchanged; normalization is only ever a matching
        # concession, never a rewrite of what the model returned.
        verdicts[name] = BrandVerdict(name, mentioned, sentiment, span, normalized_match)

    missing = expected - set(verdicts)
    if missing:
        raise ExtractParseError(f"missing brands: {sorted(missing)}")
    return verdicts


# --------------------------------------------------------------------------
# Rank convention -- defined once, here, and enforced
# --------------------------------------------------------------------------


def assign_ranks(
    raw_response: str,
    mentioned: Mapping[str, bool],
    spans: Mapping[str, str | None] | None = None,
) -> dict[str, int | None]:
    """rank = ordinal position of the brand's first mention relative to the other
    tracked brands' first mentions.

    - Mentioned, and no other tracked brand is mentioned -> rank 1.
    - Not mentioned -> None.
    - Mentioned but the name cannot be located in the text (and no usable
      evidence span either) -> None. We never invent a rank.
    """
    spans = spans or {}
    positions: dict[str, int] = {}
    lowered = raw_response.lower()
    for brand, is_mentioned in mentioned.items():
        if not is_mentioned:
            continue
        idx = lowered.find(brand.lower())
        if idx < 0:
            span = spans.get(brand)
            if span:
                idx = raw_response.find(span)
        if idx >= 0:
            positions[brand] = idx

    ordered = sorted(positions, key=lambda b: (positions[b], b))
    ranks: dict[str, int | None] = {b: None for b in mentioned}
    for i, brand in enumerate(ordered, start=1):
        ranks[brand] = i
    return ranks


# --------------------------------------------------------------------------
# Citation matching
# --------------------------------------------------------------------------


def _registrable_host(url: str) -> str | None:
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return None
    if not host:
        return None
    return host[4:] if host.startswith("www.") else host


def brand_is_cited(citation_urls: Sequence[str], domains: Sequence[str]) -> tuple[bool, list[str]]:
    """True iff a citation URL resolves to one of the brand's own domains.

    Exact registrable-domain match (or a subdomain of one). No fuzzy matching --
    a page *about* a brand on someone else's site is not the brand being cited.
    """
    wanted = {d.lower().lstrip(".") for d in domains}
    hits = []
    for url in citation_urls:
        host = _registrable_host(url)
        if host is None:
            continue
        if any(host == d or host.endswith("." + d) for d in wanted):
            hits.append(url)
    return (bool(hits), hits)


# --------------------------------------------------------------------------
# Extractor client
# --------------------------------------------------------------------------


class Extractor:
    """Anthropic call at temperature=0 with a strict-JSON wrapper."""

    def __init__(self, model: str, max_parse_retries: int = 3) -> None:
        self.model = model
        self.max_parse_retries = max_parse_retries
        self._client = anthropic.Anthropic(api_key=require_env("ANTHROPIC_API_KEY"))

    def _call(self, payload: dict) -> str:
        resp = self._client.messages.create(**payload)
        return "".join(b.text for b in resp.content if b.type == "text")

    def score_run(self, run: RunRecord, brands: Sequence[Brand]) -> dict[str, BrandVerdict]:
        assert run.raw_response is not None
        payload = build_extract_payload(run, brands)
        payload["model"] = self.model
        # Deterministic grading: temperature 0, no thinking.
        payload["thinking"] = {"type": "disabled"}

        messages = list(payload["messages"])
        last_error: str | None = None
        for _ in range(self.max_parse_retries):
            payload["messages"] = messages
            text = self._call(payload)
            try:
                return parse_extract_response(text, brands, run.raw_response)
            except ExtractParseError as exc:
                last_error = str(exc)
                # Feed the parse error back so the retry has something to act on.
                messages = messages + [
                    {"role": "assistant", "content": text},
                    {
                        "role": "user",
                        "content": (
                            f"That reply was rejected: {last_error}\n"
                            "Return the corrected JSON object only."
                        ),
                    },
                ]
        raise ExtractParseError(last_error or "unknown parse failure")


def extract_batch(
    cfg: ExperimentConfig,
    runs: Sequence[RunRecord],
    batch_id: str,
    extractor: Extractor | None = None,
    on_progress: Callable[[str], None] = lambda _: None,
) -> Path:
    path = extracts_path(batch_id)
    already = {r.run_id for r in read_extracts(path)}
    extractor = extractor or Extractor(cfg.extractor.model, cfg.extractor.max_parse_retries)

    todo = [r for r in runs if r.ok and r.run_id not in already]
    on_progress(f"{len(todo)} runs to extract ({len(already)} already done)")

    for i, run in enumerate(todo, start=1):
        append_jsonl(path, extract_one(run, cfg.brands, extractor))
        on_progress(f"[{i}/{len(todo)}] {run.run_id}")
    return path


def extract_one(run: RunRecord, brands: Sequence[Brand], extractor: Extractor) -> list[ExtractRecord]:
    assert run.raw_response is not None
    try:
        verdicts = extractor.score_run(run, brands)
    except Exception as exc:  # noqa: BLE001 - failure is recorded, never dropped
        return [
            ExtractRecord(
                run_id=run.run_id,
                brand=b.name,
                mentioned=False,
                extract_error=f"{type(exc).__name__}: {exc}",
            )
            for b in brands
        ]

    mentioned = {b.name: verdicts[b.name].mentioned for b in brands}
    spans = {b.name: verdicts[b.name].evidence_span for b in brands}
    ranks = assign_ranks(run.raw_response, mentioned, spans)

    records = []
    for b in brands:
        v = verdicts[b.name]
        cited, urls = brand_is_cited(run.citation_urls, b.domains)
        records.append(
            ExtractRecord(
                run_id=run.run_id,
                brand=b.name,
                mentioned=v.mentioned,
                rank=ranks[b.name],
                sentiment=v.sentiment,  # type: ignore[arg-type]
                cited=cited,
                citation_urls=urls,
                evidence_span=v.evidence_span,
                evidence_span_normalized_match=v.normalized_match,
                extract_error=None,
            )
        )
    return records


# --------------------------------------------------------------------------
# Stage 2b: extractor validation against a human-labelled gold set
# --------------------------------------------------------------------------

GOLD_FIELDS = ["run_id", "brand", "mentioned_extractor", "mentioned_human", "response_text"]


def sample_gold(
    batch_id: str,
    n: int = 30,
    seed: int = 0,
    force: bool = False,
    from_recovered: bool = False,
) -> Path:
    """Pull a random n (run, brand) pairs into a CSV for a human to label.

    With from_recovered=True, draws only from pairs whose evidence span matched
    after normalization -- the subset the original gold set could not describe,
    because those runs produced no verdict at all when it was drawn.
    """
    runs = {r.run_id: r for r in read_runs(f"data/runs/{batch_id}.jsonl")}
    extracts = [e for e in read_extracts(extracts_path(batch_id)) if e.ok and e.run_id in runs]
    if from_recovered:
        extracts = [e for e in extracts if e.evidence_span_normalized_match]
    if not extracts:
        which = "normalization-recovered " if from_recovered else ""
        raise ValueError(f"no usable {which}extracts for batch {batch_id}")

    target = GOLD_RECOVERED if from_recovered else GOLD_TEMPLATE
    if target.exists() and not force:
        raise FileExistsError(f"{target} exists; pass force=True to overwrite")

    chosen = random.Random(seed).sample(extracts, min(n, len(extracts)))
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=GOLD_FIELDS)
        w.writeheader()
        for e in chosen:
            w.writerow(
                {
                    "run_id": e.run_id,
                    "brand": e.brand,
                    "mentioned_extractor": str(e.mentioned).lower(),
                    "mentioned_human": "",
                    "response_text": runs[e.run_id].raw_response or "",
                }
            )
    return target


def _to_bool(value: str) -> bool:
    v = value.strip().lower()
    if v in {"true", "t", "yes", "y", "1"}:
        return True
    if v in {"false", "f", "no", "n", "0"}:
        return False
    raise ValueError(f"cannot read {value!r} as a boolean")


def cohens_kappa(a: Sequence[bool], b: Sequence[bool]) -> float | None:
    """Cohen's kappa for two binary raters, or None when it is undefined.

    Kappa is agreement above chance. When both raters put every item in the same
    single class, chance agreement is 1.0, the denominator vanishes, and there is
    no answer -- the sample contains no information about agreement above chance.

    Returning 1.0 there would be the exact overclaim this tool exists to
    criticise: an uncomputable statistic dressed up as a perfect score. Callers
    must handle None and say "undefined" rather than print a number.
    """
    if len(a) != len(b):
        raise ValueError("rater vectors must be the same length")
    n = len(a)
    if n == 0:
        raise ValueError("need at least one pair")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa_true = sum(a) / n
    pb_true = sum(b) / n
    pe = pa_true * pb_true + (1 - pa_true) * (1 - pb_true)
    if pe >= 1.0:
        return None
    return (po - pe) / (1 - pe)


@dataclass(frozen=True)
class GoldScore:
    n: int
    raw_agreement: float
    kappa: float | None
    """None when every human label falls in one class -- see cohens_kappa."""
    disagreements: list[tuple[str, str, bool, bool, str]]
    n_human_true: int
    n_human_false: int


def score_gold(path: str | Path = GOLD_TEMPLATE) -> GoldScore:
    """Read the human-filled gold file and report agreement with the extractor."""
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if not (row.get("mentioned_human") or "").strip():
                continue
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} has no filled mentioned_human values")

    machine = [_to_bool(r["mentioned_extractor"]) for r in rows]
    human = [_to_bool(r["mentioned_human"]) for r in rows]
    disagreements = [
        (r["run_id"], r["brand"], m, h, (r.get("response_text") or "")[:400])
        for r, m, h in zip(rows, machine, human)
        if m != h
    ]
    return GoldScore(
        n=len(rows),
        raw_agreement=sum(1 for m, h in zip(machine, human) if m == h) / len(rows),
        kappa=cohens_kappa(machine, human),
        disagreements=disagreements,
        n_human_true=sum(human),
        n_human_false=len(human) - sum(human),
    )
