#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Write numbers.tex from the result CSVs in data/. Every number in the paper is a macro defined here.

Usage: scripts/export_numbers.py [--data data] [--out numbers.tex]
G = Gemma 4 26B-A4B, Q = Qwen3.8-27B, C = deterministic classic pipeline.
"""
import argparse
import csv
import pathlib
import re
import sys

MODELS = ("G", "Q")
NAME = re.compile(r"^[A-Za-z]+$")


def rows(path):
    with open(path, newline="") as f:
        return list(csv.reader(f))


def num(s):
    """First numeric token of a cell such as '169 (1.0)' or '0 / None / None'."""
    m = re.match(r"\s*(-?\d+(?:\.\d+)?)", s)
    return float(m.group(1)) if m else None


def paren_num(s):
    """Numeric token inside parentheses such as '162 (0.0771)' -> 0.0771."""
    m = re.search(r"\((-?\d+(?:\.\d+)?)\)", s)
    return float(m.group(1)) if m else num(s)


class Out:
    def __init__(self):
        self.lines = []

    def put(self, name, value, comment=""):
        if not NAME.match(name):
            sys.exit(f"bad macro name {name!r}: letters only")
        self.lines.append(f"\\newcommand{{\\{name}}}{{{value}}}" + (f" % {comment}" if comment else ""))

    def fmt(self, name, x, nd=1, pct=False):
        v = x * 100 if pct else x
        self.put(name, f"{v:.{nd}f}")

    def intn(self, name, x):
        self.put(name, f"{int(round(x)):,}".replace(",", "\\,"))


def area_metric(path):
    """Area,Metric,Gemma,Qwen -> {(area, metric): (g, q)} as raw strings."""
    d = {}
    for r in rows(path)[1:]:
        d[(r[0], r[1])] = (r[2], r[3])
    return d


def by_index(path):
    """First column is the row label; returns {(label, column): cell}."""
    t = rows(path)
    head, d = t[0], {}
    for r in t[1:]:
        for j in range(1, len(head)):
            d[(r[0], head[j])] = r[j]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="numbers.tex")
    a = ap.parse_args()
    D = pathlib.Path(a.data)
    o = Out()

    # --- fidelity (1_fidelity) ---
    fid = area_metric(D / "1_fidelity.csv")
    for tag, key, nd, pct in [
        ("CovMean", ("Prediction coverage", "mean coverage vs classic"), 1, True),
        ("SmapeClassic", ("Forecast accuracy vs observed", "mean SMAPE classic"), 4, False),
        ("SmapeAgent", ("Forecast accuracy vs observed", "mean SMAPE agent"), 4, False),
        ("ExactMatch", ("Prediction agreement", "exact match (|diff| <= tol)"), 1, True),
        ("EvalCov", ("Evaluation correctness", "mean evaluation coverage vs classic"), 1, True),
        ("RunSummaryMatch", ("Evaluation correctness", "run-summary counts match classic"), 1, True),
    ]:
        for m, cell in zip(MODELS, fid[key]):
            o.fmt(f"Res{tag}{m}", num(cell), nd, pct)
    for m, cell in zip(MODELS, fid[("Input correctness", "calls with a wrong payload")]):
        o.fmt(f"ResWrongPayload{m}", paren_num(cell), 1, True)
    for tag, key in [
        ("Sessions", ("Sessions", "agent sessions run")),
        ("Completed", ("Sessions", "completed (run summary + final Ditto logged)")),
        ("ExitZero", ("Sessions", "goose exit code 0")),
        ("NodeHoursClassic", ("Prediction coverage", "node-hours classic predicted")),
        ("NodeHoursAgent", ("Prediction coverage", "node-hours agent predicted")),
        ("MissingNodeHours", ("Prediction coverage", "missing node-hours")),
        ("FullCov", ("Prediction coverage", "sessions with full coverage")),
        ("PredCalls", ("Input correctness", "baseline_predict calls")),
        ("WrongPayloadN", ("Input correctness", "calls with a wrong payload")),
        ("DevCalls", ("Evaluation correctness", "check_deviation calls")),
        ("DevWrongArgs", ("Evaluation correctness", "check_deviation calls with wrong arguments")),
        ("BothCorridors", ("Sessions", "build_summary_table succeeded for both corridors")),
    ]:
        for m, cell in zip(MODELS, fid[key]):
            o.intn(f"Res{tag}{m}", num(cell))
    for m, cell in zip(MODELS, fid[("Sessions", "classic hours in the reference")]):
        o.intn(f"ResClassicHours{m}", num(cell))
    for m, cell in zip(MODELS, fid[("Prediction agreement", "max abs diff (veh/h)")]):
        o.fmt(f"ResMaxDiff{m}", num(cell), 1)
    for m, cell in zip(MODELS, fid[("Evaluation correctness", "overall verdict matches classic")]):
        o.fmt(f"ResVerdictMatch{m}", num(cell), 1, True)

    # --- agreement (2_agreement) ---
    ag = area_metric(D / "2_agreement.csv")
    for lvl, tag in [("Detector (node)", "Node"), ("Corridor", "Cor"), ("Network (overall)", "Net")]:
        for m, cell in zip(MODELS, ag[(lvl, "Cohen's kappa")]):
            o.fmt(f"Kappa{tag}{m}", num(cell), 2)
        for m, cell in zip(MODELS, ag[(lvl, "Cohen's kappa excluding trivial pending")]):
            o.fmt(f"KappaNoPend{tag}{m}", num(cell), 2)
        for m, cell in zip(MODELS, ag[(lvl, "raw agreement")]):
            o.fmt(f"Agree{tag}{m}", num(cell), 1, True)
        for m, cell in zip(MODELS, ag[(lvl, "rows compared")]):
            o.intn(f"Rows{tag}{m}", num(cell))
        for m, cell in zip(MODELS, ag[(lvl, "agent verdict missing (not_reported)")]):
            o.intn(f"Missing{tag}{m}", num(cell))
        key = (lvl, "localized_event: classic count / recall / precision")
        for m, cell in zip(MODELS, ag[key]):
            parts = [p.strip() for p in cell.split("/")]
            o.intn(f"LocCount{tag}{m}", float(parts[0]))
            o.fmt(f"LocRecall{tag}{m}", float(parts[1]), 1, True)
            o.fmt(f"LocPrec{tag}{m}", float(parts[2]), 1, True)
    for lvl, tag in [("Detector (node)", "Node"), ("Corridor", "Cor"), ("Network (overall)", "Net")]:
        for m, cell in zip(MODELS, ag[(lvl, "macro_drift: classic count / recall / precision")]):
            o.intn(f"DriftCount{tag}{m}", float(cell.split("/")[0]))
    dp_q = {(r[0], r[1]): float(r[2]) for r in rows(D / "3_disagreement_pairs_Qwen.csv")[1:]}
    dp_g = {(r[0], r[1]): float(r[2]) for r in rows(D / "3_disagreement_pairs_Gemma.csv")[1:]}
    o.intn("ConfBaseToPendQ", dp_q[("within_baseline", "pending")])
    o.intn("ConfLocToPendQ", dp_q[("localized_event", "pending")])
    o.intn("ConfBaseToPendG", dp_g[("within_baseline", "pending")])
    o.intn("ConfBaseToMissG", dp_g[("within_baseline", "not_reported")])
    o.intn("ConfLocToMissG", dp_g[("localized_event", "not_reported")])
    malformed_g = sum(v for (k1, k2), v in dp_g.items() if "'" in k2)
    o.intn("ConfMalformedG", malformed_g)
    o.intn("ResNoCorridorG", num(fid[("Sessions", "build_summary_table succeeded for no corridor")][0]))
    o.intn("ResOneCorridorG", num(fid[("Sessions", "agent sessions run")][0]) - num(fid[("Sessions", "build_summary_table succeeded for both corridors")][0]) - num(fid[("Sessions", "build_summary_table succeeded for no corridor")][0]))
    o.intn("TrivPendNode", num(ag[("Detector (node)", "rows compared")][0]) - num(ag[("Detector (node)", "rows excluding trivial pending")][0]))

    # --- failures (3_failure_summary) ---
    fs = by_index(D / "3_failure_summary.csv")
    cols = {"G": "Gemma", "Q": "Qwen"}
    for tag, label in [
        ("NoSummary", "no run summary logged"),
        ("MissingPredSess", "sessions with missing predictions"),
        ("IncompleteEvalSess", "sessions with incomplete evaluations"),
        ("BadPayloadSess", "sessions with wrong baseline_predict payloads"),
        ("BadArgsSess", "sessions with wrong check_deviation arguments"),
        ("SlowReqSess", "sessions with a request slower than 300s"),
        ("ToolErrSess", "sessions with tool errors"),
        ("CleanSess", "sessions with no problem at all"),
    ]:
        for m in MODELS:
            o.intn(f"Fail{tag}{m}", num(fs[(label, cols[m])]))

    # --- delegation ---
    for m in MODELS:
        dl = {r[0]: r[1] for r in rows(D / f"3_delegation_{cols[m]}.csv")[1:]}
        o.intn(f"DelegateCalls{m}", num(dl["delegate calls"]))
        o.intn(f"CoordDirect{m}", num(dl["sessions where the coordinator called pipeline tools itself"]))
        o.intn(f"SubagentLoad{m}", num(dl["subagent `load` instead of tools"]))

    # --- disagreement causes ---
    for m in MODELS:
        for r in rows(D / f"3_disagreement_causes_{cols[m]}.csv")[1:]:
            tag = {"agent reported no verdict": "NoVerdict", "previous-hour prediction differed": "OtherBaseline",
                   "different verdict although inputs matched": "RuleDiff", "session had wrong check_deviation": "BadArgs"}
            for k, t in tag.items():
                if r[0].startswith(k):
                    o.intn(f"CauseN{t}{m}", float(r[1]))
                    o.fmt(f"Cause{t}{m}", float(r[2]), 1, True)

    # --- payload issues & tool errors ---
    pi_g = {r[0]: float(r[1]) for r in rows(D / "3_payload_issues_Gemma.csv")[1:]}
    pi_q = {r[0]: float(r[1]) for r in rows(D / "3_payload_issues_Qwen.csv")[1:]}
    o.intn("PayMissPrecipG", pi_g["missing_field:precipitation"])
    o.intn("PayMissLagDayG", pi_g["missing_field:flows_24"])
    o.intn("PayWrongTsG", pi_g["wrong_timestamp"])
    o.intn("PayWrongLagDayQ", pi_q["wrong_value:flows_24"])
    o.intn("PayWrongLagWeekQ", pi_q["wrong_value:flows_168"])
    o.intn("PayWrongLagHourQ", pi_q["wrong_value:flows_1"])
    te_g = {r[0]: float(r[2]) for r in rows(D / "3_tool_errors_Gemma.csv")[1:]}
    te_q = {r[0]: float(r[2]) for r in rows(D / "3_tool_errors_Qwen.csv")[1:]}
    o.intn("ToolErrPredictG", te_g["baseline_predict"])
    o.intn("ToolErrPredictQ", te_q["baseline_predict"])
    o.intn("ToolErrFlowsQ", te_q["get_flows"])
    o.intn("ToolErrPendingQ", te_q["get_pending_prediction"])

    # --- execution time (4_*) ---
    et = {r[0]: r[1:] for r in rows(D / "4_execution_time.csv")[1:]}
    head = rows(D / "4_execution_time.csv")[0][1:]
    for lab, tag in [("Gemma (all sessions)", "G"), ("Qwen (all sessions)", "Q"), ("Classic (deterministic)", "C"),
                     ("Gemma (typical, no stalled request)", "Gtyp")]:
        for col, ct in [("mean", "Mean"), ("median", "Median"), ("p95", "PNinetyFive"), ("max", "Max"), ("std", "Std"),
                        ("total_hours", "TotalHours")]:
            o.fmt(f"Time{ct}{tag}", float(et[lab][head.index(col)]), 1)
    tr = {r[0]: r[1:] for r in rows(D / "4_time_ratio.csv")[1:]}
    for m in MODELS:
        o.fmt(f"TimeRatioMean{m}", float(tr[cols[m]][0]), 1)
        o.fmt(f"TimeRatioMedian{m}", float(tr[cols[m]][1]), 0)
    fe = by_index(D / "4_feasibility.csv")
    for m in MODELS:
        o.fmt(f"SlowestMin{m}", float(fe[(cols[m], "slowest (min)")]), 1)
        o.fmt(f"WithinTen{m}", float(fe[(cols[m], "finished within 10 min")]), 1, True)
        o.fmt(f"WithinSixty{m}", float(fe[(cols[m], "finished within 60 min")]), 0, True)

    # --- resources (5_resources) ---
    rs = area_metric(D / "5_resources.csv")
    for tag, key, nd in [
        ("ReqMean", ("LLM requests per session (turns incl. subagents)", "mean"), 1),
        ("ReqMax", ("LLM requests per session (turns incl. subagents)", "max"), 0),
        ("InMean", ("input tokens per session", "mean"), 0),
        ("OutMean", ("output tokens per session", "mean"), 0),
        ("OutMeanClean", ("output tokens per session excl. responses > 20000", "mean"), 0),
        ("MaxPromptPct", ("context (limit 24000)", "largest prompt as % of limit"), 0),
        ("MaxPrompt", ("context (limit 24000)", "largest prompt (tokens)"), 0),
        ("OverEighty", ("context (limit 24000)", "requests above 80% of limit"), 0),
        ("OverLimit", ("context (limit 24000)", "requests above the limit"), 0),
        ("OverLimitSess", ("context (limit 24000)", "sessions with a request above the limit"), 0),
        ("CacheShare", ("totals", "prompt tokens served from cache (if reported)"), 1),
        ("Retried", ("errors and stalls", "requests retried after 429/503"), 0),
        ("LlmFailed", ("errors and stalls", "LLM requests failed or errored"), 0),
        ("Runaway", ("errors and stalls", "responses cut off (finish_reason=length)"), 0),
        ("RunawayShare", ("errors and stalls", "share of all output tokens"), 1),
        ("ReqLatency", ("errors and stalls", "mean request latency (s)"), 1),
        ("TotalRequests", ("totals", "LLM requests"), 0),
        ("TokPerReqIn", ("tokens per request", "mean input"), 0),
        ("ToolErrors", ("errors and stalls", "MCP tool errors (total)"), 0),
        ("TurnCapSess", ("turns (max 150)", "sessions with requests >= max turns"), 0),
    ]:
        for m, cell in zip(MODELS, rs[key]):
            v = num(cell)
            if tag in ("CacheShare", "RunawayShare"):
                o.fmt(f"Res{tag}{m}", v, nd, True)
            elif nd == 0:
                o.intn(f"Res{tag}{m}", v)
            else:
                o.fmt(f"Res{tag}{m}", v, nd)

    # --- cost (6_cost): rows keyed by model, scenario, provider ---
    cost = rows(D / "6_cost.csv")
    ch = cost[0]
    scen_map = {"observed": "Obs", "runaways removed": "NoRun", "output capped at 4096": "Cap"}
    for r in cost[1:]:
        m = "G" if r[0] == "Gemma" else "Q"
        sc = scen_map[r[1]]
        pr = "G" if r[2] == "Gemma" else "Q"
        if r[0] == r[2]:
            o.fmt(f"CostRunApi{sc}{m}", float(r[ch.index("API cost per run (USD)")]), 4)
            o.intn(f"CostFiveApi{sc}{m}", float(r[ch.index("API cost, 5 years (USD)")]))
            o.fmt(f"BreakEven{sc}{m}", float(r[ch.index("break-even (years at 1 run/hour)")]), 1)
        if r[1] == "observed" and r[0] == r[2]:
            o.fmt(f"CostRunApi{m}", float(r[ch.index("API cost per run (USD)")]), 4)
            o.intn(f"CostFiveApi{m}", float(r[ch.index("API cost, 5 years (USD)")]))
            o.intn(f"CostFiveLocal{m}", float(r[ch.index("local cost, 5 years (USD)")]))
            o.fmt(f"CostRunLocal{m}", float(r[ch.index("local cost per run (USD)")]), 3)
            o.fmt(f"BreakEvenYears{m}", float(r[ch.index("break-even (years at 1 run/hour)")]), 1)
        o.intn(f"CostCrossFive{sc}{m}at{pr}", float(r[ch.index("API cost, 5 years (USD)")]))
        o.fmt(f"BreakEvenCross{sc}{m}at{pr}", float(r[ch.index("break-even (years at 1 run/hour)")]), 1)
    be = [float(r[cost[0].index("break-even (years at 1 run/hour)")]) for r in cost[1:]]
    o.fmt("BreakEvenMin", min(be), 1)
    o.fmt("BreakEvenMax", max(be), 1)
    o.intn("CostRunsFiveYears", 24 * 365 * 5)

    # --- tokens (6_tokens) ---
    tok = rows(D / "6_tokens.csv")
    th = tok[0]
    for r in tok[1:]:
        m = "G" if r[0] == "Gemma" else "Q"
        sc = scen_map[r[1]]
        o.intn(f"TokInRun{sc}{m}", float(r[th.index("input tokens / run")]))
        o.intn(f"TokOutRun{sc}{m}", float(r[th.index("output tokens / run")]))
        o.fmt(f"TokInFiveM{sc}{m}", float(r[th.index("input tokens, 5 years (millions)")]), 1)
        o.fmt(f"TokOutFiveM{sc}{m}", float(r[th.index("output tokens, 5 years (millions)")]), 1)

    # --- forecasting (forecast_test.csv, transcribed from the dissertation) ---
    for r in rows(D / "forecast_test.csv")[1:]:
        tag = {"Random Forest": "RF", "XGBoost": "XGB", "LSTM": "LSTM", "SARIMA": "SARIMA"}[r[0]]
        o.fmt(f"FcRmse{tag}", float(r[1]), 1)
        o.fmt(f"FcMae{tag}", float(r[2]), 1)
        o.fmt(f"FcSmape{tag}", float(r[3]), 4)
        o.fmt(f"FcRsq{tag}", float(r[4]), 3)

    header = "% Generated by scripts/export_numbers.py from data/*.csv. Do not edit.\n"
    pathlib.Path(a.out).write_text(header + "\n".join(o.lines) + "\n")
    print(f"wrote {a.out}: {len(o.lines)} macros")


if __name__ == "__main__":
    main()
