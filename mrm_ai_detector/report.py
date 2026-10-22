"""Render detection results as text, JSON, Markdown or a self-contained HTML report."""

from __future__ import annotations

import html
import json
import math
from datetime import datetime, timezone
from typing import List, Sequence

from .detector import INCONCLUSIVE, INSUFFICIENT, LIKELY_AI, LIKELY_HUMAN, MIXED, DocumentResult

DISCLAIMER = (
    "Scores are statistical indicators of AI-generated text, not proof of authorship. "
    "Use them to prioritise human review; never as the sole basis for a decision about a person."
)


def _pct(p) -> str:
    return "n/a" if p is None or (isinstance(p, float) and math.isnan(p)) else f"{p * 100:.0f}%"


def _ci(r: DocumentResult) -> str:
    return f" (90% CI {_pct(r.ci_low)}–{_pct(r.ci_high)})" if r.ci_low is not None else ""


def _fmt(v) -> str:
    if v is None:
        return "–"
    return f"{v:.3g}" if isinstance(v, float) else str(v)


# ---------------------------------------------------------------------------
def render_json(results: Sequence[DocumentResult], include_text: bool = False) -> str:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "disclaimer": DISCLAIMER,
        "documents": [r.to_dict(include_text=include_text) for r in results],
    }
    return json.dumps(payload, indent=2, default=lambda o: None)


# ---------------------------------------------------------------------------
_ANSI = {LIKELY_AI: "\033[31m", MIXED: "\033[33m", INCONCLUSIVE: "\033[36m", LIKELY_HUMAN: "\033[32m", INSUFFICIENT: "\033[90m"}


def render_text(results: Sequence[DocumentResult], color: bool = False, detail: bool = True) -> str:
    def c(verdict: str, s: str) -> str:
        return f"{_ANSI.get(verdict, '')}{s}\033[0m" if color else s

    out: List[str] = []
    if len(results) > 1 or not detail:
        width = min(60, max(len(r.source) for r in results))
        out.append(f"{'Document':<{width}}  {'Words':>7}  {'AI prob':>7}  {'AI share':>8}  Verdict")
        out.append("-" * (width + 50))
        for r in results:
            src = r.source if len(r.source) <= width else "…" + r.source[-(width - 1):]
            out.append(f"{src:<{width}}  {r.word_count:>7}  {_pct(r.probability):>7}  {_pct(r.ai_share):>8}  {c(r.verdict, r.verdict)}")
        out.append("")
        if not detail:
            return "\n".join(out)
    for r in results if detail else []:
        out.append(f"== {r.source}")
        out.append(f"Verdict      : {c(r.verdict, r.verdict)}")
        out.append(f"AI probability: {_pct(r.probability)}{_ci(r)}")
        out.append(f"AI-flagged share of text: {_pct(r.ai_share)}  ({sum(s.flagged for s in r.segments)}/{sum(s.scored for s in r.segments)} scored segments)")
        out.append(f"Words: {r.word_count}   Model: {r.model}   Time: {r.elapsed_s:.2f}s")
        for w in r.warnings:
            out.append(f"! {w}")
        if r.provenance:
            out.append("Provenance hints (file traces; not part of the probability):")
            out += [f"  - {h}" for h in r.provenance]
        if r.signals:
            out.append("Top signals (+ towards AI, − towards human):")
            for s in r.signals[:6]:
                arrow = "+" if s.contribution > 0 else "−"
                out.append(f"  {arrow}{abs(s.contribution):.2f}  {s.feature:<26} {_fmt(s.value):>8}  {s.description}")
        flagged = [s for s in r.segments if s.flagged]
        if flagged:
            out.append("Flagged segments:")
            for s in flagged[:10]:
                snippet = " ".join(s.text.split())[:110]
                sec = f"[{s.section}] " if s.section else ""
                out.append(f"  #{s.index:<3} {_pct(s.probability):>4}  {sec}{snippet}…")
        out.append("")
    out.append(DISCLAIMER)
    return "\n".join(out)


# ---------------------------------------------------------------------------
def render_markdown(results: Sequence[DocumentResult]) -> str:
    out = ["# AI Content Detection Report", "", f"_{DISCLAIMER}_", ""]
    out += ["| Document | Words | AI probability | AI-flagged share | Verdict |", "|---|---:|---:|---:|---|"]
    for r in results:
        out.append(f"| {r.source} | {r.word_count} | {_pct(r.probability)}{_ci(r)} | {_pct(r.ai_share)} | {r.verdict} |")
    for r in results:
        out += ["", f"## {r.source}", "", f"**{r.verdict}** — AI probability {_pct(r.probability)}{_ci(r)}", ""]
        if r.provenance:
            out += ["**Provenance hints** (file traces; not part of the probability):", ""] + [f"- {h}" for h in r.provenance] + [""]
        if r.signals:
            out += ["| Signal | Value | Contribution | Meaning |", "|---|---:|---:|---|"]
            out += [f"| `{s.feature}` | {_fmt(s.value)} | {s.contribution:+.2f} | {s.description} |" for s in r.signals]
        out += ["", "| # | Section | Words | AI prob |", "|---:|---|---:|---:|"]
        out += [f"| {s.index} | {s.section or ''} | {s.word_count} | {'**' if s.flagged else ''}{_pct(s.probability) if s.scored else 'not scored (not prose)'}{'**' if s.flagged else ''} |" for s in r.segments]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
_CSS = """
:root{--bg:#fbfbfa;--fg:#1d1f23;--muted:#666b73;--card:#fff;--line:#e3e4e8;--ai:#c2410c;--mix:#b45309;--hum:#15803d;--inc:#0e7490;--hl:255,120,60}
@media (prefers-color-scheme:dark){:root{--bg:#111316;--fg:#e7e8ea;--muted:#9aa0a8;--card:#1a1d21;--line:#2c3036;--ai:#fb923c;--mix:#fbbf24;--hum:#4ade80;--inc:#22d3ee;--hl:255,140,80}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1000px;margin:0 auto;padding:32px 16px}h1{font-size:24px;margin:0 0 4px}h2{font-size:19px;margin:0 0 8px;word-break:break-all}
.muted{color:var(--muted);font-size:13px}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:20px;margin:20px 0}
.stats{display:flex;flex-wrap:wrap;gap:24px;margin:12px 0}.stat b{display:block;font-size:26px;font-variant-numeric:tabular-nums}
.v-ai{color:var(--ai)}.v-mix{color:var(--mix)}.v-hum{color:var(--hum)}.v-inc,.v-ins{color:var(--inc)}
table{width:100%;border-collapse:collapse;font-size:13px}td,th{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
td.n{text-align:right;font-variant-numeric:tabular-nums}.seg{padding:10px 12px;border-radius:6px;margin:6px 0;border-left:4px solid transparent;white-space:pre-wrap}
.seg .meta{font-size:12px;color:var(--muted);margin-bottom:4px}.strip{display:flex;height:14px;border-radius:4px;overflow:hidden;margin:8px 0}
.strip span{display:block;height:100%}details summary{cursor:pointer;color:var(--muted)}
"""


def _verdict_class(v: str) -> str:
    return {LIKELY_AI: "v-ai", MIXED: "v-mix", LIKELY_HUMAN: "v-hum", INCONCLUSIVE: "v-inc"}.get(v, "v-ins")


def render_html(results: Sequence[DocumentResult]) -> str:
    e = html.escape
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<title>AI Content Detection Report</title><style>{_CSS}</style></head><body><main>",
        "<h1>AI Content Detection Report</h1>",
        f"<p class='muted'>Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} · {len(results)} document(s)</p>",
        f"<p class='muted'>{e(DISCLAIMER)}</p>",
    ]
    if len(results) > 1:
        parts.append("<div class='card'><table><tr><th>Document</th><th>Words</th><th>AI prob.</th><th>Flagged share</th><th>Verdict</th></tr>")
        for i, r in enumerate(results):
            parts.append(
                f"<tr><td><a href='#doc{i}'>{e(r.source)}</a></td><td class='n'>{r.word_count}</td><td class='n'>{_pct(r.probability)}</td>"
                f"<td class='n'>{_pct(r.ai_share)}</td><td class='{_verdict_class(r.verdict)}'>{e(r.verdict)}</td></tr>"
            )
        parts.append("</table></div>")
    for i, r in enumerate(results):
        parts.append(f"<section class='card' id='doc{i}'><h2>{e(r.source)}</h2>")
        parts.append(f"<div class='stats'><div class='stat'><span class='muted'>Verdict</span><b class='{_verdict_class(r.verdict)}'>{e(r.verdict)}</b></div>")
        parts.append(f"<div class='stat'><span class='muted'>AI probability</span><b>{_pct(r.probability)}</b><span class='muted'>{e(_ci(r).strip())}</span></div>")
        parts.append(f"<div class='stat'><span class='muted'>AI-flagged share</span><b>{_pct(r.ai_share)}</b></div>")
        parts.append(f"<div class='stat'><span class='muted'>Words</span><b>{r.word_count:,}</b></div></div>")
        for w in r.warnings:
            parts.append(f"<p class='v-mix'>⚠ {e(w)}</p>")
        if r.provenance:
            parts.append("<p><b>Provenance hints</b> <span class='muted'>(file traces; not part of the probability)</span></p><ul>")
            parts += [f"<li>{e(h)}</li>" for h in r.provenance]
            parts.append("</ul>")
        total = sum(s.word_count for s in r.segments) or 1
        parts.append("<div class='strip' title='Segment AI probability across the document'>")
        for s in r.segments:
            parts.append(f"<span style='width:{100 * s.word_count / total:.3f}%;background:rgba(var(--hl),{0.08 + 0.85 * s.probability:.2f})' title='#{s.index}: {_pct(s.probability)}'></span>")
        parts.append("</div>")
        if r.signals:
            parts.append("<table><tr><th>Signal</th><th>Value</th><th>Contribution</th><th>Meaning</th></tr>")
            for s in r.signals:
                parts.append(f"<tr><td><code>{e(s.feature)}</code></td><td class='n'>{_fmt(s.value)}</td><td class='n'>{s.contribution:+.2f}</td><td>{e(s.description)}</td></tr>")
            parts.append("</table>")
        parts.append("<details open><summary>Segments</summary>")
        for s in r.segments:
            alpha = max(0.0, (s.probability - 0.3) / 0.7) * 0.35 if s.scored else 0.0
            border = "var(--ai)" if s.flagged else "var(--line)"
            sig = "; ".join(f"{g.feature} {g.contribution:+.2f}" for g in s.signals)
            parts.append(
                f"<div class='seg' style='background:rgba(var(--hl),{alpha:.2f});border-left-color:{border}'>"
                f"<div class='meta'>#{s.index} · {e(s.section or '—')} · {s.word_count} words · {('AI ' + _pct(s.probability) + ' · ' + e(sig)) if s.scored else 'not scored (table or list)'}</div>{e(s.text)}</div>"
            )
        parts.append("</details></section>")
    parts.append("</main></body></html>")
    return "".join(parts)


RENDERERS = {"text": render_text, "json": render_json, "markdown": render_markdown, "html": render_html}
