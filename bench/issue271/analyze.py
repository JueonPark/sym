"""Score the frozen small-region policies against the exhaustive measured oracle."""
import argparse
import gzip
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from .region_cost import plan_id


def controls(plans, placement, groups):
    arms = {name: [] for name in ("fixed-cpu", "fixed-gpu", "fixed-forward", "fixed-fused", "resident",
                                  "per-transfer", "placement-only", "schedule-only",
                                  "buffer-only", "joint")}
    for plan in plans:
        name = plan_id(plan)
        serial = plan["chunk"] == plan["depth"] == 1
        ordinary = not plan["reuse"]
        offload = plan["resident"] == 0
        placement_matches = plan["placement"] == placement
        arms["joint"].append(name)
        if serial and ordinary:
            arms["placement-only"].append(name)
            if plan["resident"] == groups:
                arms["resident"].append(name)
            if offload:
                arms["fixed-" + plan["placement"]].append(name)
                if placement_matches:
                    arms["per-transfer"].append(name)
        if offload and placement_matches:
            if ordinary:
                arms["schedule-only"].append(name)
            if serial:
                arms["buffer-only"].append(name)
    return arms


def paced_metrics(trace):
    arrivals = trace["arrival_ms"]
    starts = trace["start_ms"]
    finishes = trace["finish_ms"]
    if not arrivals or len(arrivals) != len(starts) or len(starts) != len(finishes):
        raise ValueError("invalid paced request count")
    if arrivals != sorted(arrivals) or finishes != sorted(finishes):
        raise ValueError("invalid serialized trace")
    if not all(math.isfinite(t) for t in (*arrivals, *starts, *finishes)):
        raise ValueError("nonfinite paced timestamp")
    if (any(a > s or s >= f for a, s, f in zip(arrivals, starts, finishes))
            or any(s < f for s, f in zip(starts[1:], finishes))):
        raise ValueError("invalid arrival/start/completion")
    sojourns = sorted(f - a for a, f in zip(arrivals, finishes))
    p95 = sojourns[math.ceil(.95 * len(arrivals)) - 1]
    return p95, 1000 * len(arrivals) / (finishes[-1] - arrivals[0])


def actual(row, objective, slo_ms=None, load_index=None):
    if row["status"] in ("refused", "oom"):
        return None
    if row["status"] != "ok":
        raise ValueError(f"unexpected measurement failure {row['status']}")
    if objective == "warm":
        return statistics.median(row["samples_ms"])
    if objective == "lifecycle":
        return row["lifecycle5_ms"]
    if objective == "slo":
        p95, rate = paced_metrics(row["paced"][load_index])
        return rate if p95 <= slo_ms else None
    raise ValueError(f"unknown objective {objective}")


def compare(selected, rows, objective, slo_ms=None, load_index=None):
    feasible = {}
    for name, row in rows.items():
        loads = range(len(row.get("paced", []))) if objective == "slo" else [None]
        for i in loads:
            value = actual(row, objective, slo_ms, i)
            if value is not None:
                if not math.isfinite(value) or value <= 0:
                    raise ValueError("expected a positive finite measurement")
                feasible[name, i] = value
    best = max if objective == "slo" else min
    winner = best(feasible, key=feasible.get) if feasible else None
    chosen = feasible.get((selected, load_index))
    result = dict(selected=selected, load_index=load_index, actual=chosen, oracle_plan=winner,
                  oracle=feasible.get(winner), feasible_pairs=len(feasible), regret_pct=None)
    if selected is None:
        outcome = "missed_feasible" if feasible else "no_oracle"
    elif chosen is None:
        outcome = "slo_miss" if objective == "slo" and rows[selected]["status"] == "ok" else "infeasible"
    else:
        oracle = feasible[winner]
        ratio = oracle / chosen if objective == "slo" else chosen / oracle
        result["regret_pct"] = 100 * (ratio - 1)
        outcome = "ok"
    return dict(result, outcome=outcome)


def read_evidence(path):
    opener = gzip.open if Path(path).suffix == ".gz" else open
    with opener(path, "rt") as stream:
        return json.load(stream)


def index_rows(raw_rows, work):
    from .measure import plans

    names = {plan_id(p) for p in plans(work)}
    matrix = {budget: {} for budget in work["budgets_bytes"]}
    periods = work["arrivals_ms"]
    for row in raw_rows:
        if row["workload"] != work["id"]:
            continue
        cells = matrix[row["budget_bytes"]]
        name = plan_id(row["plan"])
        if name in cells or name not in names:
            raise ValueError(f"duplicate or unknown candidate {name}")
        cells[name] = row
        if row["status"] == "ok":
            checks = row["correctness"]
            if (not checks["exact_int16"] or not checks["first_and_last"]
                    or checks["previous_output_checked"] != bool(work["retained_outputs"])):
                raise ValueError(f"failed output check in {name}")
            mem = row["memory"]
            if not 0 < mem["peak_live_bytes"] <= mem["arena_peak_address"] <= mem["arena_bytes"] == row["budget_bytes"]:
                raise ValueError(f"over-budget successful row {name}")
            if [point["period_ms"] for point in row["paced"]] != periods:
                raise ValueError(f"incomplete measured load grid for {name}")
            for trace in row["paced"]:
                grid = [i * trace["period_ms"] for i in range(work["paced_requests"])]
                if trace["arrival_ms"] != grid:
                    raise ValueError(f"incomplete arrival trace for {name}")
                paced_metrics(trace)
            lifecycle = row["lifecycle5"]
            if len(lifecycle["request_ms"]) != 5:
                raise ValueError(f"incomplete lifecycle for {name}")
            stages = [lifecycle["setup_ms"], lifecycle["close_ms"], *lifecycle["request_ms"]]
            if sum(stages) > row["lifecycle5_ms"] + 1e-6:
                raise ValueError(f"lifecycle stages exceed wall time for {name}")
            samples = [row["setup_ms"], row["first_ms"], row["close_ms"], row["lifecycle5_ms"],
                       *stages, *row["samples_ms"]]
            if len(row["samples_ms"]) < 4 or any(not math.isfinite(x) or x <= 0 for x in samples):
                raise ValueError(f"invalid completed-request measurements for {name}")
        elif row["status"] not in ("oom", "refused"):
            raise ValueError(f"unknown qualification result for {name}")
    if any(cells.keys() != names for cells in matrix.values()):
        raise ValueError(f"incomplete oracle grid for {work['id']}")
    return matrix


def score_campaign(data):
    from time import perf_counter_ns
    from .measure import plans, workloads
    from .region_cost import ComponentProfile, catalog, select

    started = perf_counter_ns()
    profile = ComponentProfile(data["calibration"])
    fit_us = (perf_counter_ns() - started) / 1000
    reports, diagnostics = [], []
    if {w["id"] for w in data["workloads"]} != {w["id"] for w in workloads()}:
        raise ValueError("unexpected workload set")
    if len({w["id"] for w in data["workloads"]}) != len(data["workloads"]):
        raise ValueError("duplicate workload")
    if set(r["workload"] for r in data["rows"]) != {w["id"] for w in workloads()}:
        raise ValueError("unexpected workload in measured oracle")
    for work in workloads():
        measured = next(w for w in data["workloads"] if w["id"] == work["id"])
        if any(measured[k] != v for k, v in work.items()):
            raise ValueError("workload contract differs from the recorded experiment")
        native = measured["selector"]["placement"]
        if native not in ("cpu", "gpu"):
            raise ValueError("invalid native selector decision")
        policies = controls(list(plans(work)), native, work["groups"])
        start = perf_counter_ns()
        candidates = {c["name"]: c for c in catalog(work, profile)}
        build_us = (perf_counter_ns() - start) / 1000
        # No candidate timing, feasibility label or holdout sample enters select().
        matrix = index_rows(data["rows"], work)
        for budget, cells in matrix.items():
            base = dict(workload=work["id"], split=work["split"], budget_bytes=budget)
            for candidate, row in cells.items():
                pred = select([candidates[candidate]], work, budget, "warm")["prediction"]
                seen = actual(row, "warm")
                err = None if seen is None or pred is None else 100 * abs(pred["warm_us"]/1000 - seen) / seen
                diagnostics.append(dict(base, plan=candidate, observed=row["status"],
                                        admitted=pred is not None, error_pct=err,
                                        predicted_peak=pred["peak_bytes"] if pred else None,
                                        measured_peak=row["memory"].get("peak_live_bytes")))
            objectives = ["warm", "lifecycle"]
            if work["arrivals_ms"]:
                objectives.append("slo")
            for objective in objectives:
                for policy, ids in policies.items():
                    result = select([candidates[key] for key in ids], work, budget, objective)
                    pred = result["prediction"]
                    load = pred["load_index"] if pred else None
                    scored = compare(result["selected"], cells, objective, work["slo_ms"], load)
                    estimate = None
                    tail = rate = None
                    if pred:
                        if objective == "slo":
                            estimate = pred["throughput_rps"]
                            tail, rate = paced_metrics(cells[result["selected"]]["paced"][load]) if (
                                cells[result["selected"]]["status"] == "ok") else (None, None)
                        else:
                            estimate = pred[objective + "_us"] / 1000
                    reports.append(dict(base, objective=objective, policy=policy, **scored,
                                        predicted=estimate, planning_us=result["planning_us"],
                                        catalog_us=build_us, predicted_peak=pred["peak_bytes"] if pred else None,
                                        predicted_p95_ms=pred["p95_us"]/1000 if pred and objective=="slo" else None,
                                        measured_p95_ms=tail, measured_trace_rps=rate, examined=result["examined"],
                                        pruned=len(result["pruned"]), rejected=len(result["rejected"])))
    return dict(decisions=reports, diagnostics=diagnostics, component_fit_us=fit_us)


def number(value, places=2):
    return "—" if value is None else f"{value:.{places}f}"


def median(xs):
    return statistics.median(xs) if xs else None


def report(data, scored):
    rows, fits = scored["decisions"], scored["diagnostics"]
    lines = ["# Joint region planner results", "",
             "These results score the complete measured catalog. The oracle can choose any executable "
             "plan in the catalog. For the arrival experiment it can also choose the offered load.",
             "", "## Warm request, five-call lifecycle and paced throughput", "",
             "| Objective | Workload | Budget MiB | Joint plan | Predicted | Measured | Oracle | Regret % | Outcome |",
             "|---|---|---:|---|---:|---:|---:|---:|---|"]
    for row in rows:
        if row["policy"] != "joint":
            continue
        name = row["selected"] or "—"
        if row["load_index"] is not None:
            name += f" @ load {row['load_index']}"
        lines.append(f"| {row['objective']} | {row['workload']} | {row['budget_bytes']/2**20:g} | "
                     f"{name} | {number(row['predicted'])} | {number(row['actual'])} | "
                     f"{number(row['oracle'])} | {number(row['regret_pct'])} | {row['outcome']} |")
    lines.extend(["", "Latency is in ms. Paced throughput is in completed requests/s. A failed deadline "
                  "renders a selection infeasible at that load. Infeasible and refused selections have "
                  "no numerical regret.", "", "## Same-engine controls", "",
                  "| Objective | Policy | Scored / total | Invalid admissions | Missed feasible | No oracle | Median regret % | Worst regret % |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"])
    for obj in ("warm", "lifecycle", "slo"):
        for policy in dict.fromkeys(r["policy"] for r in rows):
            group = [r for r in rows if r["objective"] == obj and r["policy"] == policy]
            regrets = [r["regret_pct"] for r in group if r["outcome"] == "ok"]
            count = Counter(r["outcome"] for r in group)
            lines.append(f"| {obj} | {policy} | {len(regrets)} / {len(group)} | "
                         f"{count['infeasible']+count['slo_miss']} | {count['missed_feasible']} | "
                         f"{sum(r['oracle'] is None for r in group)} | {number(median(regrets))} | {number(max(regrets, default=None))} |")
    lines.extend(["", "## Held-out prediction and feasibility", "",
                  "Warm error is the absolute percentage error for each admitted, successful catalog row. "
                  "The accuracy columns also count refusals against the independent execution result.",
                  "", "| Workload | Catalog rows | Successful | Arena refusals | Driver OOM | False admit | False reject | Peak mismatches | Median error % | Worst error % |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for work in data["workloads"]:
        group = [r for r in fits if r["workload"] == work["id"]]
        errs = [r["error_pct"] for r in group if r["error_pct"] is not None]
        count = Counter(r["observed"] for r in group)
        lines.append(f"| {work['id']} | {len(group)} | {count['ok']} | {count['refused']} | {count['oom']} | "
                     f"{sum(r['admitted'] and r['observed']!='ok' for r in group)} | "
                     f"{sum(not r['admitted'] and r['observed']=='ok' for r in group)} | "
                     f"{sum(r['predicted_peak'] != r['measured_peak'] for r in group if r['admitted'] and r['observed']=='ok')} | "
                     f"{number(median(errs))} | {number(max(errs, default=None))} |")
    lines.extend(["", "## Paced sojourn", "",
                  "| Workload | Budget MiB | Joint period ms | Predicted p95 ms | Measured p95 ms | Completed/s at selected load | Feasible pairs | Feasible oracle plan / period ms |",
                  "|---|---:|---:|---:|---:|---:|---:|---|"])
    for r in rows:
        if r["policy"] == "joint" and r["objective"] == "slo":
            w = next(w for w in data["workloads"] if w["id"] == r["workload"])
            period = w["arrivals_ms"][r["load_index"]] if r["load_index"] is not None else None
            winner = r["oracle_plan"]
            oracle = f"{winner[0]} / {w['arrivals_ms'][winner[1]]}" if winner else "—"
            lines.append(f"| {w['id']} | {r['budget_bytes']/2**20:g} | {number(period)} | "
                         f"{number(r['predicted_p95_ms'])} | {number(r['measured_p95_ms'])} | "
                         f"{number(r['measured_trace_rps'])} | {r['feasible_pairs']} | {oracle} |")
    lines.extend(["", "The no-oracle count includes every cell with zero measured feasible choices, "
                  "including a cell in which the policy admitted a plan that missed its deadline.",
                  "", "## Warm amortization against the current per-transfer selector", "",
                  "This estimate charges all independent component calibration to a single deployment "
                  "plus the observed difference in setup, first completion and release. Subsequent calls "
                  "use the measured warm medians. The estimate assumes stationary service on this host. "
                  "The last column additionally charges the full cold Triton import and qualification. A dash "
                  "means the selected plan has no warm saving. CPU search and common CUDA initialization "
                  "are itemized in the [planner report](README.md).", "",
                  "| Workload | Budget MiB | Per-transfer ms | Joint ms | Saved per warm call ms | Calls to recover calibration and lifecycle difference | Calls including cold qualification |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    lookup = {(r["workload"], r["budget_bytes"], plan_id(r["plan"])): r for r in data["rows"]}
    by_cell = {(r["workload"], r["budget_bytes"], r["policy"]): r for r in rows if r["objective"] == "warm"}
    for joint in (r for r in rows if r["objective"] == "warm" and r["policy"] == "joint"):
        key = joint["workload"], joint["budget_bytes"]
        base = by_cell[(*key, "per-transfer")]
        if joint["actual"] is None or base["actual"] is None:
            continue
        saving = base["actual"] - joint["actual"]
        def one_time(name):
            raw = lookup[(*key, name)]
            return raw["setup_ms"] + raw["first_ms"] + raw["close_ms"]
        initial_ms = data["calibration_wall_ms"] + one_time(joint["selected"]) - one_time(base["selected"])
        calls = 1 + max(0, math.ceil(initial_ms / saving)) if saving > 0 else None
        cold_ms = initial_ms + data["fused_qualification"]["import_and_qualification_ms"]
        cold_calls = 1 + max(0, math.ceil(cold_ms / saving)) if saving > 0 else None
        lines.append(f"| {key[0]} | {key[1]/2**20:g} | {number(base['actual'])} | {number(joint['actual'])} | "
                     f"{number(saving)} | {number(calls, 0)} | {number(cold_calls, 0)} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--json", action="store_true", help="emit every decision, error and planning time")
    args = parser.parse_args()
    data = read_evidence(args.evidence)
    scored = score_campaign(data)
    print(json.dumps(scored) if args.json else report(data, scored), end="\n" if args.json else "")


if __name__ == "__main__":
    main()
