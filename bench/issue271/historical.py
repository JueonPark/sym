"""CPU-only replay of deployment decisions against published completed requests.

Run from the repository root with python -m bench.issue271.historical.
"""
import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import median

from bench.issue266.support import matrix, policy_configuration
from bench.issue271 import planner


ROOT = Path(__file__).resolve().parents[2]
MIB = 1 << 20
SOURCES = {
    "bench/issue263/RESULTS.md": "af55323a4f7511f9b799cdc819c070eee0888aae08b624ac4697664909eb9cd5",
    "bench/issue266/TABLES.md": "be9c41a9c368ffb82514a63ce6473a4aff4a6a980995cb4700e0c632a31799b7",
    "bench/issue269/TABLES.md": "22dbf10305b7835637aedbfa5ef5715c2dc9f814581a9c622edb117781290d44",
    "bench/issue266/support.py": "e76487c09505e2ffe38eccdaaa69160432d0bd43e01cf86f18851724e7119449",
    "libreloc/python/reloc_torch/memory.py": "31976d8bf26c04c6c528b830142adbe93581a7b06b7008af39918fe4699dc899",
}


def read_table(source, header, *, section=None):
    """Header prefix selects one table in a pinned document."""
    content = (ROOT / source).read_text()
    if section:
        content = content.split(section, 1)[1]
    start = content.index(header)
    table = content[start:].split("\n\n", 1)[0].splitlines()[2:]
    return [line.strip("|").strip().split(" | ") for line in table]


@dataclass(frozen=True)
class Arm:
    ms: float | None
    peak_mib: float | None = None
    oom_rounds: int = 0
    native_contexts: int | None = None
    native_retained_mib: float | None = None
    torch_reserved_mib: float | None = None
    arena_mib: float | None = None


def select(prices, *, budget_mib, margin_mib, context_limit, objective):
    """Encode completed calibration endpoints as indivisible planner events."""
    candidates = []
    for name, price in prices.items():
        if price.ms is None or price.oom_rounds:
            continue
        if budget_mib is not None and price.peak_mib is None:
            raise ValueError(f"missing physical projection for {name}")
        floor = math.ceil((price.peak_mib + margin_mib) * MIB) if budget_mib is not None else 0
        candidates.append(dict(
            name=name, qualified=context_limit is None or price.native_contexts <= context_limit,
            setup_us=0, events=[dict(id="completed", us=price.ms * 1000, deps=[], resources=["executor"])],
            memory=dict(resident_bytes=0, retained_bytes=0, context_bytes=0, intervals=[],
                        setup_peak_bytes=0, warm_floor_bytes=floor)))
    # Folds without a physical constraint have zero modeled byte cost.
    result = planner.choose_plan(candidates, budget_bytes=(budget_mib or 0) * MIB, objective=objective, reuses=1)
    result.pop("planning_us")  # Emit reproducible evidence independent of CPU clock noise.
    return result


def exhaustive_calibration(prices, *, budget_mib, margin_mib, context_limit):
    """Rank original price rows directly, independent of the planner/catalog adapter."""
    ranking = []
    for name, arm in prices.items():
        if arm.ms is None or arm.oom_rounds or (context_limit is not None and arm.native_contexts > context_limit):
            continue
        peak_mib = 0 if budget_mib is None else arm.peak_mib + margin_mib
        if budget_mib is None or peak_mib <= budget_mib:
            ranking.append((arm.ms, math.ceil(peak_mib * MIB), name))
    return min(ranking)[2] if ranking else None


def score(case, prices, observed, *, budget_mib=None, margin_mib=0, context_limit=None, objective="warm"):
    limits = dict(budget_mib=budget_mib, margin_mib=margin_mib, context_limit=context_limit)
    plan = select(prices, objective=objective, **limits)
    selected = plan["selected"]
    expected = exhaustive_calibration(prices, **limits)
    if selected != expected:
        raise ValueError(f"{case}: planner chose {selected}, exhaustive calibration chose {expected}")
    # The campaign physically imposed its own budget. A sampled peak cannot
    # prove success. Only actual completed, non-OOM arms enter this oracle.
    eligible = [name for name, arm in observed.items()
                if arm.ms is not None and not arm.oom_rounds
                and (context_limit is None or arm.native_contexts <= context_limit)]
    best = min(eligible, key=lambda name: (observed[name].ms, observed[name].peak_mib or 0, name), default=None)
    target = observed[selected] if selected is not None else None
    oracle_ms = observed[best].ms if best is not None else None
    actual_ms = target.ms if target is not None else None
    prediction = plan["prediction"][objective + "_us"] / 1000 if selected is not None else None
    return dict(case=case, selected=selected, oracle=best, predicted_ms=prediction, actual_ms=actual_ms,
                oracle_ms=oracle_ms, regret_pct=100 * (actual_ms / oracle_ms - 1) if actual_ms else None,
                abs_prediction_error_pct=100 * abs(prediction / actual_ms - 1) if actual_ms else None,
                selected_oom_rounds=target.oom_rounds if target is not None else 0,
                feasible_arms=len(eligible), evaluated_arms=len(observed), limits=limits,
                core_plan=plan, objective=objective, exhaustive_calibration_selection=expected,
                predictions={name: asdict(arm) for name, arm in prices.items()},
                observations={name: asdict(arm) for name, arm in observed.items()})


def matched_requests():
    source = "bench/issue263/RESULTS.md"
    sections = (
        ("E2 warm", "| Workload | Compute | Torch p50 ms", 1, 8),
        ("E5 lifecycle", "| Workload | Compute | Torch total s", 1000, 4),
    )
    decisions = []
    for label, header, units, size in sections:
        data = {(work, compute): {"torch": Arm(float(torch) * units), "sym": Arm(float(sym) * units)}
                for work, compute, torch, sym, *_ in read_table(source, header)}
        if len(data) != size:
            raise ValueError(f"unexpected #263 {label} coverage")
        for (work, compute), measured in data.items():
            if label == "E2 warm":
                train = (work, "inductor" if compute == "eager" else "eager")
            else:
                train = ("moe_sparse" if work == "llm_decode" else "llm_decode", compute)
            decisions.append(score(f"{label} {work} {compute}", data[train], measured,
                                   objective="lifecycle" if label == "E5 lifecycle" else "warm"))
    return decisions


def representation():
    source = "bench/issue269/TABLES.md"
    blocks = defaultdict(lambda: defaultdict(list))
    for _, case, compute, replay, runtime, variant, p50, *_ in read_table(source, "| Round | Case | Compute | Replay"):
        if replay == "false" and variant != "resident":
            blocks[case, compute][f"{runtime} {variant}"].append(float(p50))
    if len(blocks) != 6 or any(len(arms) != 6 or any(len(times) != 3 for times in arms.values())
                               for arms in blocks.values()):
        raise ValueError("unexpected #269 no-replay offload coverage")
    costs = {case: {arm: Arm(median(times)) for arm, times in arms.items()} for case, arms in blocks.items()}
    return [score(f"E2 layout {case} {compute}", costs["decode8", compute], costs[case, compute])
            for case in ("decode64", "prefill256") for compute in ("eager", "inductor")]


def physical_cases():
    source = "bench/issue266/TABLES.md"
    e4, e5 = defaultdict(dict), defaultdict(dict)
    header = "| Arm | Success/OOM/admission | Median ms | P95 ms | Success physical MiB"
    for label, outcomes, ms, _, success_peak, failed_peak, _ in read_table(source, header + " | OOM"):
        shape, name = label.split(" ", 1)
        layers, context, budget = map(int, re.fullmatch(r"L(\d+)/S(\d+)/B(\d+)", shape).groups())
        if outcomes not in ("3/0/0", "0/3/0"):
            raise ValueError(f"unexpected E4 outcomes {label} {outcomes}")
        arm = Arm(float(ms), float(success_peak)) if ms != "—" else Arm(None, float(failed_peak), oom_rounds=3)
        e4[layers, context, budget][name] = arm

    section = "## E5: completed timing and physical memory"
    retained_header = "| Arm | Torch allocated MiB | Torch reserved MiB | Native device MiB"
    retention = {label: (float(native), float(reserved), float(arena))
                 for label, _, reserved, native, _, arena, _, _ in read_table(source, retained_header, section=section)}
    # Use original campaign metadata for executable arms / native transfer
    # contexts. Torch does not create a Sym transfer owner.
    jobs = {f"{j['case']} C{j['callers']} {j['runtime']} {j['policy']}": j
            for j in matrix() if j["kind"] == "e5" and j["round"] == 1}
    for label, outcomes, _, tail, peak in read_table(source, header, section=section):
        job = jobs[label]
        if outcomes != "3/0/0":
            raise ValueError(f"unexpected E5 outcomes {label} {outcomes}")
        native_contexts = job["callers"] * policy_configuration(job)["contexts"] if job["runtime"] == "sym" else 0
        name = f"{job['runtime']} {job['policy']}"
        e5[job["case"], job["callers"]][name] = Arm(float(tail), float(peak), 0, native_contexts, *retention[label])
    if (sum(map(len, e4.values())), sum(arm.oom_rounds for arms in e4.values() for arm in arms.values()),
            sum(map(len, e5.values()))) != (72, 24, 32):
        raise ValueError("unexpected #266 matrix or missing actual OOMs")
    return e4, e5, jobs


def budget_decisions(e4, margin_mib):
    decisions = []
    for layers in (16, 32):
        for context in (16, 512):
            prices = e4[layers, context, 1024]
            observed = e4[layers, context, 512]
            decisions.append(score(f"E4 L{layers}/S{context} 512 MiB", prices, observed,
                                   budget_mib=512, margin_mib=margin_mib))
    return decisions


def burst_prices(history, target_jobs):
    prices = {}
    previous = history[-1]
    for job in target_jobs:
        name = f"{job['runtime']} {job['policy']}"
        recent = previous[name]
        prior_callers = job["callers"] // 2
        contexts = job["callers"] * policy_configuration(job)["contexts"] if job["runtime"] == "sym" else 0
        if len(history) == 1:
            peak = 2 * recent.peak_mib
        else:
            peak = recent.peak_mib + prior_callers * max(0, recent.peak_mib - history[0][name].peak_mib)
        # Correct only the projected change in retained native scratch when
        # the deployed context count changes. The output arena is nested in
        # Torch's allocations, so adding it here would double count it.
        if recent.native_contexts:
            per_context = recent.native_retained_mib / recent.native_contexts
            peak += (contexts - 2 * recent.native_contexts) * per_context
        prices[name] = Arm(recent.ms, peak, native_contexts=contexts)
    return prices


def bursts(e5, jobs):
    regular, quota = [], []
    for work in ("llm_decode", "moe_sparse"):
        history = [e5[work, 1]]
        for callers in (2, 4):
            target_jobs = [j for j in jobs.values() if j["case"] == work and j["callers"] == callers]
            prices = burst_prices(history, target_jobs)
            target = e5[work, callers]
            case = f"E5 burst {work} C{callers}"
            regular.append(score(case, prices, target, budget_mib=1024))
            if callers == 4:
                quota.append(score(case + " quota 4", prices, target, budget_mib=1024, context_limit=4))
            history.append(target)
    return regular, quota


def evaluate():
    for name, expected in SOURCES.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"source hash mismatch: {name}")
    e4, e5, jobs = physical_cases()
    regular, quota = bursts(e5, jobs)
    return {"main": matched_requests() + representation() + budget_decisions(e4, 32) + regular,
            "zero_margin": budget_decisions(e4, 0), "native_context_quota": quota}


def table(rows):
    lines = ["| Target | Selected | Feasible oracle | Pred. ms | Measured ms | Best ms | Regret |",
             "|---|---|---|---:|---:|---:|---:|"]
    for row in rows:
        amounts = ["—" if row[k] is None else f"{row[k]:.3f}" for k in ("predicted_ms", "actual_ms", "oracle_ms")]
        if row["selected_oom_rounds"]:
            amounts[1] = "OOM"
        regret = "—" if row["regret_pct"] is None else f"{row['regret_pct']:.2f}%"
        lines.append("| " + " | ".join([row["case"], row["selected"] or "no admission",
                                        row["oracle"] or "no feasible arm", *amounts, regret]) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    outputs = parser.add_mutually_exclusive_group()
    outputs.add_argument("--json", action="store_true", help="include every prediction and historical target arm")
    outputs.add_argument("--check", action="store_true", help="compare the report tables with a fresh CPU reanalysis")
    args = parser.parse_args()
    results = evaluate()
    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True, allow_nan=False))
        return
    report = Path(__file__).with_name("HISTORICAL.md").read_text() if args.check else ""
    for name, rows in results.items():
        rendered = table(rows)
        if args.check:
            if report.count(rendered) != 1:
                raise ValueError(f"{name} decisions differ from HISTORICAL.md")
        else:
            print(rendered, end="\n\n")
    if args.check:
        print(f"Verified {sum(map(len, results.values()))} planner decisions against exhaustive calibration rankings, "
              f"the published report tables, and {len(SOURCES)} input hashes.")


if __name__ == "__main__":
    main()
