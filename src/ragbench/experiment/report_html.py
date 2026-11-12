"""Renders a `Report` as one self-contained HTML page: no scripts, no network, light and dark."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from html import escape
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from ragbench.experiment.report import Report

_STYLE = """
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  --series-1: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a; --good: #006300;
}
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; --good: #0ca30c;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1240px; margin: 0 auto; padding: 32px 16px 64px; }
h1 { font-size: 26px; margin: 0 0 4px; }
h2 { font-size: 19px; margin: 40px 0 4px; }
p { margin: 6px 0; } .lede { color: var(--ink-2); max-width: 78ch; }
.meta { color: var(--ink-2); font-size: 13px; }
code { font: 13px ui-monospace, SFMono-Regular, Menlo, monospace; background: var(--surface);
  border: 1px solid var(--border); border-radius: 4px; padding: 1px 5px; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
  padding: 16px 18px; margin: 14px 0; }
.pick { border-left: 4px solid var(--series-1); }
.pick .name { font-size: 20px; font-weight: 650; margin: 2px 0 6px; }
.label { font-size: 12px; letter-spacing: .04em; text-transform: uppercase; color: var(--muted); }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px;
  margin-top: 14px; }
.tile .value { font-size: 26px; font-weight: 650; line-height: 1.2; }
.tile .note { font-size: 12px; color: var(--ink-2); }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13px;
  font-variant-numeric: tabular-nums; }
th, td { padding: 6px 6px; text-align: right; border-bottom: 1px solid var(--grid);
  white-space: nowrap; }
th { color: var(--ink-2); font-weight: 600; border-bottom: 1px solid var(--axis); }
th:first-child, td:first-child, th.text, td.text { text-align: left; }
td.wrap { white-space: normal; min-width: 160px; max-width: 215px; }
.heat th { white-space: normal; vertical-align: bottom; }
tr.chosen td { font-weight: 650; }
.rank { color: var(--muted); }
.bar { display: flex; align-items: center; gap: 8px; justify-content: flex-start; }
.bar i { display: block; height: 10px; border-radius: 0 4px 4px 0; background: var(--series-1); }
.bar span { min-width: 40px; text-align: left; }
.ci { color: var(--muted); font-size: 12px; }
.tag { display: inline-block; font-size: 11.5px; padding: 0 7px; border-radius: 999px;
  border: 1px solid var(--axis); color: var(--ink-2); }
.heat td { text-align: center; border: 2px solid var(--surface); min-width: 62px; }
.heat td.text { text-align: left; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 13px; color: var(--ink-2);
  margin: 4px 0 8px; }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 50%;
  margin-right: 6px; }
svg { display: block; width: 100%; height: auto; }
svg text { fill: var(--ink-2); font: 12px system-ui, sans-serif; }
svg .title { fill: var(--muted); }
svg .name { fill: var(--ink); }
.notes li { margin: 3px 0; } .notes { color: var(--ink-2); padding-left: 20px; }
"""
_WRAP = ' class="text wrap"'
_CHOSEN = ' class="chosen"'
_SERIES = ("var(--series-1)", "var(--series-2)", "var(--series-3)")


def _percent(value: object) -> str:
    return "-" if _missing(value) else f"{100 * float(value):.1f}%"  # type: ignore[arg-type]


def _missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _money(value: object, digits: int = 2) -> str:
    return "-" if _missing(value) else f"${float(value):,.{digits}f}"  # type: ignore[arg-type]


def _table(headers: Sequence[str], rows: Iterable[Sequence[str]], *, chosen: int = -1) -> str:
    head = "".join(f"<th>{escape(header)}</th>" for header in headers)
    body = ""
    for number, row in enumerate(rows):
        cells = "".join(
            f"<td{_WRAP if column == 1 else ''}>{cell}</td>" for column, cell in enumerate(row)
        )
        body += f"<tr{_CHOSEN if number == chosen else ''}>{cells}</tr>"
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _bar(value: float, largest: float, low: object = None, high: object = None) -> str:
    width = 0 if largest <= 0 else round(64 * value / largest)
    interval = (
        "" if _missing(low) else f' <span class="ci">{_percent(low)} to {_percent(high)}</span>'
    )
    return f'<div class="bar"><i style="width:{width}px"></i><span>{_percent(value)}</span>{interval}</div>'


def _ticks(largest: float, count: int = 5) -> list[float]:
    if largest <= 0:
        return [0.0, 1.0]
    raw = largest / count
    magnitude = 10 ** math.floor(math.log10(raw))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw)
    return [step * number for number in range(math.ceil(largest / step) + 1)]


def _scatter(answers: pl.DataFrame, chosen_key: str) -> str:
    """Quality against cost: one dot per configuration, colour by retrieval configuration."""
    rows = list(answers.iter_rows(named=True))
    retrievals = list(dict.fromkeys(row["retrieval"] for row in rows))
    colour = {
        name: _SERIES[position] if len(retrievals) <= len(_SERIES) else _SERIES[0]
        for position, name in enumerate(retrievals)
    }
    width, height, left, right, top, bottom = 1000, 430, 56, 150, 18, 46
    x_ticks = _ticks(max(row["cost_per_1k_answers_usd"] for row in rows) * 1.05)
    low = min(row["correctness_low"] for row in rows)
    y_low = max(0.0, math.floor(low * 10 - 0.5) / 10)
    y_high = min(1.0, math.ceil(max(row["correctness_high"] for row in rows) * 10 + 0.2) / 10)

    def x(value: float) -> float:
        return float(left + (width - left - right) * value / x_ticks[-1])

    def y(value: float) -> float:
        return float(top + (height - top - bottom) * (1 - (value - y_low) / (y_high - y_low)))

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Answer correctness against cost per 1,000 answers">'
    ]
    steps = round((y_high - y_low) * 10)
    for number in range(steps + 1):
        value = y_low + number / 10
        parts.append(
            f'<line x1="{left}" x2="{width - right}" y1="{y(value):.1f}" y2="{y(value):.1f}" '
            f'stroke="var(--grid)"/><text x="{left - 8}" y="{y(value) + 4:.1f}" '
            f'text-anchor="end">{100 * value:.0f}%</text>'
        )
    for value in x_ticks:
        parts.append(
            f'<text x="{x(value):.1f}" y="{height - bottom + 18}" text-anchor="middle">'
            f"${value:g}</text>"
        )
    parts.append(
        f'<line x1="{left}" x2="{width - right}" y1="{height - bottom}" y2="{height - bottom}" '
        'stroke="var(--axis)"/>'
        f'<text class="title" x="{left}" y="{height - 6}">Cost per 1,000 answers (USD)</text>'
        f'<text class="title" x="{left}" y="{top - 4}">Correctness (judge), 95% interval</text>'
    )
    placed: list[tuple[float, float]] = []
    # The recommended point is drawn first so that its label always has room.
    ordered = sorted(
        rows, key=lambda item: (item["config_key"] != chosen_key, -item["correctness"])
    )
    for row in ordered:
        cx, cy = x(row["cost_per_1k_answers_usd"]), y(row["correctness"])
        tooltip = (
            f"{row['configuration']}\ncorrectness {_percent(row['correctness'])} "
            f"({_percent(row['correctness_low'])} to {_percent(row['correctness_high'])})\n"
            f"{_money(row['cost_per_1k_answers_usd'])} per 1,000 answers, "
            f"median {row['latency_p50_s']:.1f}s"
        )
        is_chosen = row["config_key"] == chosen_key
        ring = (
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="11" fill="none" stroke="var(--ink)" '
            'stroke-width="2"/>'
            if is_chosen
            else ""
        )
        parts.append(
            f"<g><title>{escape(tooltip)}</title>"
            f'<line x1="{cx:.1f}" x2="{cx:.1f}" y1="{y(row["correctness_low"]):.1f}" '
            f'y2="{y(row["correctness_high"]):.1f}" stroke="var(--axis)" stroke-width="2"/>'
            f"{ring}"
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" fill="{colour[row["retrieval"]]}" '
            'stroke="var(--surface)" stroke-width="2"/>'
        )
        # A label is drawn only where it does not sit on top of one already placed; the
        # legend, the tooltip and the table below carry the rest.
        free = all(abs(cy - py) >= 14 or abs(cx - px) >= 190 for px, py in placed)
        if free:
            placed.append((cx, cy))
            suffix = " (recommended)" if is_chosen else ""
            parts.append(
                f'<text class="name" x="{cx + (20 if is_chosen else 14):.1f}" y="{cy + 4:.1f}">'
                f"{escape(row['generator'])}{suffix}</text>"
            )
        parts.append("</g>")
    parts.append("</svg>")
    legend = "".join(
        f'<span><i style="background:{colour[name]}"></i>{escape(name)}</span>'
        for name in retrievals
    )
    return f'<div class="legend">{legend}</div>' + "".join(parts)


def _heatmap(retrieval: pl.DataFrame, primary: str) -> str:
    """Stage-1 grid: chunker rows, embedder and search-mode columns, shaded by the metric."""
    rows = list(retrieval.iter_rows(named=True))
    columns = list(
        dict.fromkeys(
            (row["search_mode"], row["embedder"])
            for row in sorted(
                rows, key=lambda r: (r["search_mode"] != "lexical", r["search_mode"], r["embedder"])
            )
        )
    )
    chunkers = list(dict.fromkeys(row["chunker"] for row in rows))
    values = {(row["chunker"], row["search_mode"], row["embedder"]): row[primary] for row in rows}
    low, high = min(values.values()), max(values.values())
    head = '<th class="text">Chunker</th>' + "".join(
        f"<th>{escape(mode)}<br>{escape(embedder) or '&nbsp;'}</th>" for mode, embedder in columns
    )
    body = ""
    for chunker in chunkers:
        cells = ""
        for mode, embedder in columns:
            value = values.get((chunker, mode, embedder))
            if value is None:
                cells += "<td>-</td>"
                continue
            share = 8 + round(52 * (value - low) / (high - low)) if high > low else 30
            cells += (
                f'<td style="background:color-mix(in srgb, var(--series-1) {share}%, '
                f'var(--surface))">{100 * value:.1f}</td>'
            )
        body += f'<tr><td class="text">{escape(chunker)}</td>{cells}</tr>'
    return (
        f'<div class="scroll"><table class="heat"><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )


def _retrieval_section(report: Report) -> str:
    frame, primary = report.retrieval, report.primary_metric
    others = [c for c in frame.columns if "@" in c and c != primary]
    largest = float(frame[primary].max())  # type: ignore[arg-type]
    rows = [
        [
            f'<span class="rank">{position}</span>',
            escape(row["configuration"]),
            _bar(row[primary], largest, row["primary_low"], row["primary_high"]),
            *[_percent(row[name]) for name in others],
            f"{row['chunks']:,}",
            _money(row["cost_per_1k_queries_usd"], 3),
        ]
        for position, row in enumerate(frame.iter_rows(named=True), start=1)
    ]
    headers = [
        "#",
        "Configuration",
        f"{primary} (95% interval)",
        *others,
        "Chunks",
        "$ / 1k queries",
    ]
    return (
        f"<h2>Stage 1 - Retrieval without a reranker</h2>"
        f'<p class="lede">{frame.height} configurations on {int(str(frame["questions"].max()))} questions. '
        f"The grid shows <code>{escape(primary)}</code> in percent; a stronger blue is better. Lexical "
        "search does not use embeddings, so it has one column.</p>"
        f'<div class="card">{_heatmap(frame, primary)}</div>'
        f'<div class="card">{_table(headers, rows)}</div>'
    )


def _reranking_section(report: Report) -> str:
    frame, primary = report.reranking, report.primary_metric
    if not frame.height:
        return ""
    rows = []
    for position, row in enumerate(frame.iter_rows(named=True), start=1):
        if _missing(row["gain"]):
            gain, verdict = "-", ""
        else:
            gain = (
                f"{100 * row['gain']:+.1f} pt "
                f'<span class="ci">{100 * row["gain_low"]:+.1f} to {100 * row["gain_high"]:+.1f}</span>'
            )
            verdict = "significant" if row["significant"] else "no significant difference"
        rows.append(
            [
                f'<span class="rank">{position}</span>',
                escape(row["configuration"]),
                _percent(row[primary]),
                _percent(row["base"]),
                gain + (f' <span class="tag">{verdict}</span>' if verdict else ""),
                _money(row["cost_per_1k_queries_usd"], 2),
                f"{row['latency_ms']:.0f} ms",
            ]
        )
    headers = [
        "#",
        "Configuration",
        primary,
        "Without reranker",
        "Gain (paired 95% interval)",
        "$ / 1k queries",
        "Median latency",
    ]
    return (
        "<h2>Stage 2 - Reranking</h2>"
        '<p class="lede">The best stage-1 configurations, each with every reranker. The gain is '
        "measured on the same questions, with a paired bootstrap interval; an interval that "
        "contains zero is reported as no significant difference.</p>"
        f'<div class="card">{_table(headers, rows)}</div>'
    )


def _answers_section(report: Report) -> str:
    frame = report.answers
    if not frame.height:
        return ""
    chosen_key = report.recommendation.config_key if report.recommendation else ""
    chosen = -1
    rows = []
    for position, row in enumerate(frame.iter_rows(named=True)):
        if row["config_key"] == chosen_key:
            chosen = position
        gap = (
            "best"
            if position == 0
            else f"{100 * row['gap_to_best']:+.1f} pt "
            f'<span class="ci">{100 * row["gap_low"]:+.1f} to {100 * row["gap_high"]:+.1f}</span>'
        )
        rows.append(
            [
                f'<span class="rank">{position + 1}</span>',
                f"<strong>{escape(row['generator'])}</strong><br>"
                f'<span class="ci">{escape(row["retrieval"])}</span>',
                _bar(row["correctness"], 1.0, row["correctness_low"], row["correctness_high"]),
                gap
                + (' <span class="tag">tied</span>' if row["tied_with_best"] and position else ""),
                _percent(row["faithfulness"]),
                _percent(row["citation"]),
                _percent(row["cited_gold"]),
                _percent(row["gold_in_context"]),
                _money(row["cost_per_1k_answers_usd"]),
                f"{row['latency_p50_s']:.1f}s / {row['latency_p95_s']:.1f}s",
            ]
        )
    headers = [
        "#",
        "Generator and retrieval",
        "Correctness (95% interval)",
        "Gap to best (paired)",
        "Faithful",
        "Citation",
        "Cited = gold",
        "Gold retrieved",
        "$ / 1k answers",
        "p50 / p95",
    ]
    return (
        "<h2>Stage 3 - Answers</h2>"
        f'<p class="lede">The best retrieval configurations, each with every generator, on '
        f"{int(str(frame['questions'].max()))} questions. A judge model scores each answer 0, 1 or 2 on "
        "correctness, faithfulness and citation; scores are shown as a share of the maximum. "
        "<em>Cited = gold</em> is the share of cited passages that contain gold evidence; "
        "<em>Gold retrieved</em> is the share of questions whose context held any. Cost and "
        "latency are those of retrieval plus generation, as billed and timed when each call "
        "was first made. Hover a dot for details.</p>"
        f'<div class="card">{_scatter(frame, chosen_key)}</div>'
        f'<div class="card">{_table(headers, rows, chosen=chosen)}</div>'
    )


def _agreement_section(report: Report) -> str:
    if not report.answers.height:
        return ""
    if not report.agreement:
        return (
            "<h2>Judge validation</h2>"
            '<div class="card"><p><strong>Not done yet.</strong> No human labels have been '
            "imported for this run, so the judge's scores above are unverified. Label the sheet "
            f"written by <code>ragbench judge sheet {escape(report.run.run_id)}</code> and import "
            f"it with <code>ragbench judge agreement {escape(report.run.run_id)} "
            "&lt;sheet.csv&gt;</code>; this section then shows Cohen's kappa and raw agreement "
            "per criterion.</p></div>"
        )
    rows = [
        [
            escape(item.criterion),
            "",
            str(item.items),
            _percent(item.raw),
            f"{item.kappa:.2f}",
            f"{item.weighted_kappa:.2f}",
        ]
        for item in report.agreement
    ]
    headers = [
        "Criterion",
        "",
        "Answers labelled",
        "Raw agreement",
        "Cohen's kappa",
        "Weighted kappa",
    ]
    return (
        "<h2>Judge validation</h2>"
        '<p class="lede">Agreement between the judge model and a person who scored the same '
        "answers with the same rubric, without seeing the judge's scores.</p>"
        f'<div class="card">{_table(headers, rows)}</div>'
    )


def _by_source_section(report: Report) -> str:
    frame = report.by_source
    if not frame.height:
        return ""
    sources = [column for column in frame.columns if column not in ("configuration", "all")]
    rows = [
        [
            "",
            escape(row["configuration"]),
            _percent(row["all"]),
            *[_percent(row[s]) for s in sources],
        ]
        for row in frame.iter_rows(named=True)
    ]
    return (
        "<h2>By source</h2>"
        f'<p class="lede"><code>{escape(report.primary_metric)}</code> of the best retrieval '
        "configurations on each source collection.</p>"
        f'<div class="card">{_table(["", "Configuration", "All", *sources], rows)}</div>'
    )


def _tiles(report: Report) -> str:
    recommendation = report.recommendation
    if recommendation is None:
        return ""
    tiles: list[tuple[str, str, str]] = []
    if report.answers.height:
        row = report.answers.filter(pl.col("config_key") == recommendation.config_key).row(
            0, named=True
        )
        tiles = [
            (
                "Correctness",
                _percent(row["correctness"]),
                f"{_percent(row['correctness_low'])} to {_percent(row['correctness_high'])}",
            ),
            ("Faithfulness", _percent(row["faithfulness"]), "claims supported by the passages"),
            ("Cost", _money(row["cost_per_1k_answers_usd"]), "per 1,000 answers"),
            ("Latency", f"{row['latency_p50_s']:.1f}s", f"median; p95 {row['latency_p95_s']:.1f}s"),
        ]
    else:
        frames = [f for f in (report.retrieval, report.reranking) if f.height]
        row = (
            pl.concat([f.select(["config_key", report.primary_metric]) for f in frames])
            .filter(pl.col("config_key") == recommendation.config_key)
            .row(0, named=True)
        )
        tiles = [(report.primary_metric, _percent(row[report.primary_metric]), "")]
    return (
        '<div class="tiles">'
        + "".join(
            f'<div class="tile"><div class="label">{escape(label)}</div>'
            f'<div class="value">{value}</div><div class="note">{escape(note)}</div></div>'
            for label, value, note in tiles
        )
        + "</div>"
    )


def _corpus_line(corpus: Mapping[str, object]) -> str:
    by_source = corpus.get("questions_by_source")
    licences = corpus.get("licences")
    parts = [f"{corpus.get('questions')} questions over {corpus.get('documents')} documents"]
    if isinstance(by_source, dict):
        parts.append(", ".join(f"{name} {count}" for name, count in by_source.items()))
    line = f"{escape(str(corpus.get('name')))}: " + " · ".join(parts)
    if corpus.get("revision"):
        line += f" · revision <code>{escape(str(corpus['revision'])[:12])}</code>"
    if isinstance(licences, dict) and licences:
        line += " · licences: " + ", ".join(
            f"{escape(str(name))} {escape(str(licence))}" for name, licence in licences.items()
        )
    return line


def render_html(report: Report) -> str:
    """Return the report as a complete HTML document."""
    run = report.run
    recommendation = report.recommendation
    pick = ""
    if recommendation is not None:
        pick = (
            '<div class="card pick"><div class="label">Recommended configuration</div>'
            f'<div class="name">{escape(recommendation.display)}</div>'
            f"<p>{escape(recommendation.reason)}</p>{_tiles(report)}</div>"
        )
    status = (
        f"finished {run.finished_at:%Y-%m-%d %H:%M} UTC"
        if run.finished_at
        else "not finished: some questions have no result yet"
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>ragbench report - {escape(run.run_id)}</title><style>{_STYLE}</style></head>"
        "<body><main>"
        f"<h1>RAG benchmark report</h1>"
        f'<p class="meta">Run <code>{escape(run.run_id)}</code> · configuration hash '
        f"<code>{escape(run.fingerprint[:16])}</code> · started {run.created_at:%Y-%m-%d %H:%M} "
        f"UTC · {status} · report generated {report.generated_at:%Y-%m-%d %H:%M} UTC</p>"
        f'<p class="meta">{_corpus_line(run.corpus)}</p>'
        f"{pick}"
        f"{_answers_section(report)}"
        f"{_agreement_section(report)}"
        f"{_reranking_section(report)}"
        f"{_retrieval_section(report)}"
        f"{_by_source_section(report)}"
        '<h2>How to read this report</h2><ul class="notes">'
        "<li>Every configuration answers the same questions, so comparisons are paired. "
        "A difference is called significant only when its 95% bootstrap interval excludes zero."
        "</li><li><code>char_recall</code> is the share of gold evidence characters that were "
        "retrieved; <code>char_precision</code> is the share of retrieved characters that are "
        "gold. Rank metrics count a passage as relevant when it overlaps a gold span.</li>"
        "<li>Costs are what the provider billed when each call was first made. Repeating the "
        "run reads every call from the cache and costs nothing.</li>"
        f"<li>Regenerate this page with <code>ragbench report {escape(run.run_id)}</code>.</li>"
        "</ul></main></body></html>"
    )
