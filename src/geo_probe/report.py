"""Stage 4: one chart, one findings file. matplotlib only.

Two rules govern the generated prose.

First, a number and the label it belongs to are always derived from the same
record. `_Cell` below exposes phrases, not floats -- a call site cannot obtain
`mde_abs_pp` without also obtaining the brand it came from, so a sentence cannot
pair one cell's number with another cell's name.

Second, no sentence may assert that a change below the MDE is absent. "Below the
MDE" means indistinguishable from noise at this sample size. Claiming it *is*
noise is accepting the null hypothesis -- the exact overclaim this tool exists to
criticise. tests/test_report_language.py enforces both rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # no display in CI or on a headless box
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from .schemas import AggRecord  # noqa: E402

REPORTS_DIR = Path("reports")

BOUNDARY_MARK = "†"  # dagger on an MDE that exceeds its headroom


def report_dir(batch_id: str) -> Path:
    return REPORTS_DIR / batch_id


# --------------------------------------------------------------------------
# Formatting primitives
# --------------------------------------------------------------------------


def _fmt_pp(x: float) -> str:
    return f"{x * 100:.1f}pp"


def _fmt_ci(ci: tuple[float, float]) -> str:
    return f"[{ci[0]:.2f}, {ci[1]:.2f}]"


@dataclass(frozen=True)
class _Cell:
    """A record and the phrases derived from it.

    Every property returns a number *together with* its label. This is the whole
    point of the type: there is no accessor that hands back a bare float, so the
    templating bug where a number from one cell lands next to another cell's name
    cannot be written.
    """

    rec: AggRecord

    @property
    def name(self) -> str:
        return f"{self.rec.brand}/{self.rec.provider}"

    @property
    def mde_phrase(self) -> str:
        return f"{_fmt_pp(self.rec.mde_abs_pp)} on {self.name}"

    @property
    def naive_mde_phrase(self) -> str:
        return f"{_fmt_pp(self.rec.mde_abs_pp_naive)} on {self.name}"

    @property
    def deff_phrase(self) -> str:
        return f"{self.rec.design_effect:.2f} on {self.name}"

    @property
    def interval_phrase(self) -> str:
        return (
            f"the cluster interval on {self.name} is {_fmt_ci(self.rec.ci95_cluster)} "
            f"against a naive Wilson interval of {_fmt_ci(self.rec.ci95_naive_wrong)}"
        )

    @property
    def mde_gap_phrase(self) -> str:
        return (
            f"on {self.name} the naive MDE is {_fmt_pp(self.rec.mde_abs_pp_naive)} while the "
            f"cluster MDE is {_fmt_pp(self.rec.mde_abs_pp)}"
        )


# --------------------------------------------------------------------------
# Chart
# --------------------------------------------------------------------------


def _asymmetric_err(point: float, ci: tuple[float, float]) -> tuple[float, float]:
    return (max(0.0, point - ci[0]), max(0.0, ci[1] - point))


def render_chart(records: Sequence[AggRecord], path: Path) -> Path:
    """Mention rate per brand with cluster error bars, faceted by provider.

    The naive Wilson interval is overlaid as a thin inner bar so the gap between
    the two is visible rather than merely tabulated. Cells whose cluster interval
    terminates on the [0, 1] boundary are drawn hollow: there the percentile
    bootstrap is degenerate and the interval end is an artifact of the bound.
    """
    providers = sorted({r.provider for r in records})
    brands = sorted({r.brand for r in records})

    fig, axes = plt.subplots(
        1,
        len(providers),
        figsize=(1.6 + 2.9 * len(providers) + 0.55 * len(brands), 5.2),
        sharey=True,
        squeeze=False,
    )

    for ax, provider in zip(axes[0], providers):
        rows = {r.brand: r for r in records if r.provider == provider}

        for solid in (True, False):
            xs, ys, lo, hi = [], [], [], []
            for i, b in enumerate(brands):
                r = rows.get(b)
                if r is None or (r.ci_at_boundary == "none") != solid:
                    continue
                d, u = _asymmetric_err(r.mention_rate, r.ci95_cluster)
                xs.append(i), ys.append(r.mention_rate), lo.append(d), hi.append(u)
            if not xs:
                continue
            ax.errorbar(
                xs, ys, yerr=[lo, hi],
                fmt="o", color="#1f3a5f", ecolor="#1f3a5f",
                markerfacecolor="#1f3a5f" if solid else "none",
                elinewidth=3.0, capsize=7, capthick=3.0, markersize=7,
                markeredgewidth=1.8, linestyle="none", zorder=2,
            )

        xs, ys, lo, hi = [], [], [], []
        for i, b in enumerate(brands):
            r = rows.get(b)
            if r is None:
                continue
            d, u = _asymmetric_err(r.mention_rate, r.ci95_naive_wrong)
            xs.append(i), ys.append(r.mention_rate), lo.append(d), hi.append(u)
        ax.errorbar(
            xs, ys, yerr=[lo, hi],
            fmt="none", ecolor="#e07a3f",
            elinewidth=1.0, capsize=3, capthick=1.0, zorder=3,
        )

        ax.set_title(provider)
        ax.set_xticks(range(len(brands)))
        ax.set_xticklabels(brands, rotation=20, ha="right")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(axis="y", alpha=0.25)

    axes[0][0].set_ylabel("mention rate (mean over prompts)")
    handles = [
        Line2D([], [], color="#1f3a5f", marker="o", markersize=7, linewidth=3,
               label="95% cluster bootstrap (correct)"),
        Line2D([], [], color="#e07a3f", linewidth=1, label="95% naive Wilson (wrong)"),
        Line2D([], [], color="#1f3a5f", marker="o", markersize=7, markerfacecolor="none",
               markeredgewidth=1.8, linestyle="none",
               label="hollow: interval touches 0 or 1, bootstrap degenerate there"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Brand mention rate: cluster vs naive intervals")
    fig.tight_layout(rect=(0, 0.11, 1, 1))

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# findings.md
# --------------------------------------------------------------------------


def _pick(cells: Sequence[_Cell], key, largest: bool) -> _Cell:
    """Choose a superlative cell, excluding those whose interval touches a bound.

    A cell at 0.96 whose cluster interval terminates at 1.00 has a degenerate
    percentile bootstrap: the interval is compressed by the bound, which makes
    its standard error -- and therefore its MDE -- look smaller than it is. Such
    a cell can win "smallest MDE" by being the most broken one in the table, so
    calling it "best-measured" inverts the truth. Superlatives are drawn from the
    non-degenerate cells; the fallback only fires if every cell is on a bound.
    """
    eligible = [c for c in cells if c.rec.ci_at_boundary == "none"] or list(cells)
    return (max if largest else min)(eligible, key=key)


def _exclusion_line(records: Sequence[AggRecord]) -> str:
    """Exclusions broken down by provider, at the top rather than buried.

    A reader who sees `mean rank (n=67)` next to `n_prompts=20, k=5` should not
    have to scroll to the bottom to find out where the other 33 went.
    """
    parts = []
    for provider in sorted({r.provider for r in records}):
        rows = [r for r in records if r.provider == provider]
        # A failed run drops every brand in that run, so the per-brand counts are
        # equal within a provider; max() is the run count, sum() the pair count.
        excl_runs = max(r.n_runs_excluded for r in rows)
        total = max(r.n_runs_used + r.n_runs_excluded for r in rows)
        pairs = sum(r.n_runs_excluded for r in rows)
        parts.append(f"{excl_runs}/{total} {provider} runs ({pairs} run-brand pairs)")
    return "Excluded: " + ", ".join(parts) + "."


def render_findings(records: Sequence[AggRecord], category: str, batch_id: str) -> str:
    if not records:
        return f"# geo-probe findings - {batch_id}\n\nNo aggregated records.\n"

    cells = [_Cell(r) for r in records]
    widest = _pick(cells, lambda c: c.rec.design_effect, largest=True)
    coarsest = _pick(cells, lambda c: c.rec.mde_abs_pp, largest=True)
    finest = _pick(cells, lambda c: c.rec.mde_abs_pp, largest=False)
    most_inflated = _pick(cells, lambda c: c.rec.mde_inflation, largest=True)
    mean_deff = sum(r.design_effect for r in records) / len(records)
    shares = [r.var_between_prompt_share for r in records if r.var_between_prompt_share is not None]
    mean_between = sum(shares) / len(shares) if shares else None
    residuals = [abs(r.deff_residual) for r in records if r.deff_residual is not None]

    lines: list[str] = []
    lines.append(f"# geo-probe findings - {batch_id}")
    lines.append("")
    lines.append(f"Category: **{category}**")
    lines.append(
        f"Design: {records[0].n_prompts} prompts x k={records[0].k} reps per (brand, provider) "
        f"cell, {len(records)} cells."
    )
    lines.append(_exclusion_line(records))
    lines.append("")
    lines.append("![mention rate by brand and provider](chart.png)")
    lines.append("")

    lines.append("## Findings")
    lines.append("")
    para = []
    para.append(
        f"Treating the runs in a cell as independent trials overstates precision: across the "
        f"{len(records)} (brand, provider) cells the design effect averages {mean_deff:.2f}, "
        f"peaking at {widest.deff_phrase}, where {widest.interval_phrase}."
    )
    if mean_between is not None:
        # The prompts-vs-reps conclusion is derived, not assumed. At a fixed run
        # budget N = m*k, Var(mean p_j) = (k*var_between + var_within)/N, so
        # shifting budget from reps to prompts helps whenever var_between > 0 --
        # a different condition from var_between being the larger share.
        if mean_between > 0.02:
            para.append(
                f"Between-prompt variance accounts for {mean_between * 100:.0f}% of total variance "
                f"on average ({min(shares) * 100:.0f}-{max(shares) * 100:.0f}% across cells). At a "
                f"fixed run budget N the variance of the estimate is "
                f"(k*var_between + var_within)/N, so any non-zero between-prompt component means "
                f"the same spend on more prompts and fewer reps buys a tighter interval."
            )
        else:
            para.append(
                f"Between-prompt variance is negligible here ({mean_between * 100:.0f}% of total "
                f"on average), so prompts and reps are close to interchangeable and the naive "
                f"interval is nearly correct -- the opposite of the usual case."
            )
    para.append(
        f"The headline number is the minimum detectable effect: at this sample size, "
        f"alpha={records[0].mde_alpha}, power={records[0].mde_power}, a week-over-week change has "
        f"to exceed {finest.mde_phrase} in the best-measured cell and {coarsest.mde_phrase} in the "
        f"worst before this design can distinguish it from sampling noise. Both are chosen among "
        f"cells whose interval does not touch 0 or 1, since the bootstrap is degenerate at a bound "
        f"and understates the MDE there."
    )
    para.append(
        f"A change smaller than that cannot be distinguished from sampling noise at this sample "
        f"size; the design has no power to detect a move that small, which is a statement about "
        f"the design and not evidence that no move occurred."
    )
    para.append(
        f"Compared like for like, {most_inflated.mde_gap_phrase} -- a dashboard using the naive "
        f"figure would treat a change in that range as detectable when this design cannot "
        f"detect it."
    )
    lines.append(" ".join(para))
    lines.append("")

    lines.append("## Per-cell numbers")
    lines.append("")
    lines.append(
        "| brand | provider | n used | k_eff | mention rate | cluster 95% | naive 95% (wrong) | "
        "deff | deff pred | resid | between-prompt var share | flip rate | mean rank (n) | "
        "MDE | naive MDE |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(records, key=lambda x: (x.provider, -x.mention_rate)):
        between = "n/a" if r.var_between_prompt_share is None else f"{r.var_between_prompt_share:.2f}"
        pred = "n/a" if r.deff_predicted is None else f"{r.deff_predicted:.2f}"
        resid = "n/a" if r.deff_residual is None else f"{r.deff_residual:+.2f}"
        rank = "n/a" if r.mean_rank is None else f"{r.mean_rank:.2f} (n={r.rank_n})"
        mde = _fmt_pp(r.mde_abs_pp) + (BOUNDARY_MARK if r.mde_interpretable != "both" else "")
        lines.append(
            f"| {r.brand} | {r.provider} | {r.n_runs_used} | {r.k_effective:.2f} | "
            f"{r.mention_rate:.2f} | {_fmt_ci(r.ci95_cluster)} | {_fmt_ci(r.ci95_naive_wrong)} | "
            f"{r.design_effect:.2f} | {pred} | {resid} | {between} | {r.flip_rate:.2f} | "
            f"{rank} | {mde} | {_fmt_pp(r.mde_abs_pp_naive)} |"
        )
    lines.append("")
    if any(r.mde_interpretable != "both" for r in records):
        lines.append(
            f"{BOUNDARY_MARK} MDE exceeds the available headroom in this direction; interpret "
            f"one-sided. A mention rate is bounded in [0, 1], and a symmetric absolute-scale MDE "
            f"cannot describe a move that would take the rate past the bound."
        )
        lines.append("")

    lines.append("## Reading notes")
    lines.append("")
    lines.append(
        "- `mention rate` is the mean of the per-prompt rates, not the pooled run rate. The prompt "
        "is the unit of analysis."
    )
    lines.append(
        "- `naive 95% (wrong)` is the Wilson interval on pooled runs, and `naive MDE` is the "
        "matching detection threshold. Both are printed to be argued with, not used."
    )
    lines.append(
        "- `k_eff` is `n used / n prompts`, the realised cluster size after exclusions. `deff pred` "
        "is `1 + (k_eff - 1) * between-prompt var share`; `resid` is observed minus predicted."
    )
    lines.append(
        "- Rank statistics are conditional on the brand being mentioned, so `mean rank` describes "
        "only the runs where the brand appeared."
    )
    lines.append("- Providers are never pooled: they are different measurement surfaces.")
    lines.append(
        "- Superlatives in the paragraph above are selected among cells whose interval does not "
        "touch 0 or 1. A cell on a bound can win 'smallest MDE' by being the most degenerate one "
        "in the table rather than the best measured."
    )
    lines.append("")

    if residuals:
        lines.append(
            f"Design-effect check: mean |observed - predicted| = {sum(residuals) / len(residuals):.2f} "
            f"across {len(residuals)} cells."
        )
        lines.append("")
    models = sorted({r.model_returned for r in records})
    lines.append(f"Models returned by the APIs: {', '.join(models)}.")
    lines.append("")
    return "\n".join(lines)


def build_report(records: Sequence[AggRecord], category: str, batch_id: str) -> tuple[Path, Path]:
    out = report_dir(batch_id)
    out.mkdir(parents=True, exist_ok=True)
    chart = render_chart(records, out / "chart.png")
    findings = out / "findings.md"
    findings.write_text(render_findings(records, category, batch_id), encoding="utf-8")
    return chart, findings
