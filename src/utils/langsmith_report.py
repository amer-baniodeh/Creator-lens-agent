"""
langsmith_report.py
--------------------
Pulls traced runs from LangSmith and computes cost/latency metrics.

Tracing has been on since notebook 01 (LANGCHAIN_TRACING_V2=true in .env), so
every ingestion, chat turn, and compliance check run through this project has
already been recorded — this just reads and summarizes it, no new instrumentation
needed.

Caveat: embedding calls (text-embedding-3-small) are made via the raw OpenAI
client in src/ingestion/embedder.py, not a LangChain-wrapped call, so they are
NOT traced and NOT included in these cost totals. This only affects GPT-4o-mini
generation cost. Embeddings are priced low enough (~$0.02 / 1M tokens) that the
gap is real but small — worth stating rather than silently ignoring.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from langsmith import Client

from src.utils.config import LANGCHAIN_API_KEY, LANGCHAIN_PROJECT


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(int(len(s) * pct), len(s) - 1)
    return s[idx]


def _stats(values: list[float]) -> dict:
    values = [v for v in values if v is not None]
    if not values:
        return {"n": 0, "avg": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "n": len(values),
        "avg": round(sum(values) / len(values), 3),
        "p50": round(_percentile(values, 0.5), 3),
        "p95": round(_percentile(values, 0.95), 3),
        "max": round(max(values), 3),
    }


def generate_cost_latency_report(project_name: str | None = None) -> dict:
    """
    Pull all runs for the given LangSmith project (defaults to the configured
    LANGCHAIN_PROJECT) and compute cost/latency metrics: overall totals, LLM
    call latency distribution, per-root-operation cost/latency (e.g. average
    cost of one full agent chat turn), and tool-level latency.
    """
    project_name = project_name or LANGCHAIN_PROJECT
    client = Client(api_key=LANGCHAIN_API_KEY)
    runs = list(client.list_runs(project_name=project_name))

    if not runs:
        return {
            "run_at": datetime.now(timezone.utc).isoformat(),
            "project": project_name,
            "error": "No runs found — tracing may not be active, or the project is empty.",
        }

    by_id = {r.id: r for r in runs}
    llm_runs = [r for r in runs if r.run_type == "llm"]
    tool_runs = [r for r in runs if r.run_type == "tool"]
    root_runs = [r for r in runs if r.parent_run_id is None]

    def find_root(run):
        r = run
        seen = set()
        while r.parent_run_id and r.parent_run_id in by_id and r.id not in seen:
            seen.add(r.id)
            r = by_id[r.parent_run_id]
        return r

    # Overall LLM cost/token/latency totals
    # total_cost comes back as decimal.Decimal from the SDK — cast to float throughout.
    total_cost = sum(float(r.total_cost or 0) for r in llm_runs)
    total_prompt_tokens = sum((r.prompt_tokens or 0) for r in llm_runs)
    total_completion_tokens = sum((r.completion_tokens or 0) for r in llm_runs)
    llm_latencies = [r.latency for r in llm_runs if r.latency is not None]

    models = {
        (r.extra or {}).get("invocation_params", {}).get("model")
        or (r.extra or {}).get("invocation_params", {}).get("model_name")
        for r in llm_runs
    }
    models.discard(None)

    # Per-root-run aggregation: attribute each LLM call's cost back to its
    # top-level operation (e.g. one AgentExecutor run = one chat turn).
    root_cost = defaultdict(float)
    root_tokens = defaultdict(int)
    for r in llm_runs:
        root = find_root(r)
        root_cost[root.id] += float(r.total_cost or 0)
        root_tokens[root.id] += (r.total_tokens or 0)

    by_root_name_cost = defaultdict(list)
    by_root_name_latency = defaultdict(list)
    for root in root_runs:
        by_root_name_cost[root.name].append(root_cost.get(root.id, 0.0))
        if root.latency is not None:
            by_root_name_latency[root.name].append(root.latency)

    operation_breakdown = {}
    for name in by_root_name_cost:
        costs = by_root_name_cost[name]
        latencies = by_root_name_latency.get(name, [])
        operation_breakdown[name] = {
            "n": len(costs),
            "avg_cost_usd": round(sum(costs) / len(costs), 6) if costs else 0.0,
            "total_cost_usd": round(sum(costs), 6),
            "latency": _stats(latencies),
        }

    # Tool-level latency (query_corpus, check_compliance, ingest_video)
    tool_latency = defaultdict(list)
    for r in tool_runs:
        if r.latency is not None:
            tool_latency[r.name].append(r.latency)
    tool_breakdown = {name: _stats(lats) for name, lats in tool_latency.items()}

    start_times = [r.start_time for r in runs if r.start_time]

    return {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "project": project_name,
        "trace_window": {
            "earliest": min(start_times).isoformat() if start_times else None,
            "latest": max(start_times).isoformat() if start_times else None,
        },
        "totals": {
            "total_runs": len(runs),
            "llm_calls": len(llm_runs),
            "tool_calls": len(tool_runs),
            "root_operations": len(root_runs),
            "models_used": sorted(models),
            "total_cost_usd": round(total_cost, 6),
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
        },
        "llm_call_latency_seconds": _stats(llm_latencies),
        "operation_breakdown": operation_breakdown,
        "tool_latency_seconds": tool_breakdown,
        "caveat": (
            "Embedding calls (text-embedding-3-small) bypass LangChain and are not "
            "traced, so total_cost_usd covers GPT-4o-mini generation only. "
            "Embedding cost is real but small (~$0.02 / 1M tokens)."
        ),
    }


def save_report(report: dict, metrics_dir: str = "data/metrics") -> tuple[Path, Path]:
    """
    Save the report as both a timestamped archive entry (never overwritten) and
    the "latest" file, mirroring the pattern used for compliance eval runs.

    Returns (latest_path, archive_path).
    """
    metrics_path = Path(metrics_dir)
    metrics_path.mkdir(parents=True, exist_ok=True)
    runs_dir = metrics_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    latest_path = metrics_path / "langsmith_report.json"
    latest_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    label = report["run_at"].replace(":", "").replace(".", "")[:15]
    archive_path = runs_dir / f"{label}.json"
    archive_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    return latest_path, archive_path


def generate_summary_md(metrics_dir: str = "data/metrics") -> Path:
    """
    Regenerate data/metrics/SUMMARY.md — a single readable index of every
    archived cost/latency report, mirroring data/eval/SUMMARY.md's pattern.
    """
    metrics_path = Path(metrics_dir)
    runs_dir = metrics_path / "runs"
    reports = []
    for path in sorted(runs_dir.glob("*.json")):
        reports.append(json.loads(path.read_text(encoding="utf-8")))

    lines = [
        "# Cost & latency — history",
        "",
        "Auto-generated by `src/utils/langsmith_report.py` from LangSmith traces "
        "(`data/metrics/runs/*.json`). Do not edit by hand — re-run the generator "
        "instead (see notebook 07).",
        "",
    ]

    for r in reports:
        if "error" in r:
            continue
        date = r["run_at"].replace("T", " ")[:16]
        totals = r["totals"]
        lines += [
            f"## {date} UTC",
            "",
            f"- Trace window: {r['trace_window']['earliest']} → {r['trace_window']['latest']}",
            f"- Total runs traced: {totals['total_runs']} ({totals['llm_calls']} LLM calls, "
            f"{totals['tool_calls']} tool calls, {totals['root_operations']} top-level operations)",
            f"- Model(s): {', '.join(totals['models_used']) or 'none'}",
            f"- **Total GPT cost: ${totals['total_cost_usd']:.4f}** "
            f"({totals['total_prompt_tokens']:,} prompt + {totals['total_completion_tokens']:,} completion tokens)",
            f"- LLM call latency: avg {r['llm_call_latency_seconds']['avg']}s, "
            f"p95 {r['llm_call_latency_seconds']['p95']}s, max {r['llm_call_latency_seconds']['max']}s",
            "",
            "**Cost/latency per operation type:**",
            "",
            "| Operation | Count | Avg cost | Total cost | Avg latency | p95 latency |",
            "|---|---|---|---|---|---|",
        ]
        for name, stats in r["operation_breakdown"].items():
            lines.append(
                f"| {name} | {stats['n']} | ${stats['avg_cost_usd']:.6f} | "
                f"${stats['total_cost_usd']:.6f} | {stats['latency']['avg']}s | {stats['latency']['p95']}s |"
            )
        lines += ["", f"*{r['caveat']}*", ""]

    summary_path = metrics_path / "SUMMARY.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


if __name__ == "__main__":
    report = generate_cost_latency_report()
    save_report(report)
    generate_summary_md()
    print(json.dumps(report, indent=2, default=str))
