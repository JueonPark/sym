"""Choose an executable plan from component calibration on a recorded host."""
import argparse
import gzip
import json
from pathlib import Path
from time import perf_counter_ns

from .measure import plans, workloads
from .region_cost import ComponentProfile, catalog, plan_id, select


def recommend(calibration, work, budget, objective):
    """Accept component rows only. No oracle or end-to-end timings enter selection."""
    start = perf_counter_ns()
    profile = ComponentProfile(calibration)
    candidates = catalog(work, profile)
    catalog_us = (perf_counter_ns() - start) / 1000
    choice = select(candidates, work, budget, objective)
    plan = next((p for p in plans(work) if plan_id(p) == choice["selected"]), None)
    prediction = choice["prediction"]
    return dict(plan=plan, reason=choice["reason"], objective=objective,
                prediction={k: v for k, v in prediction.items() if k != "schedule"} if prediction else None,
                examined=choice["examined"], pruned=len(choice["pruned"]),
                rejected=choice["rejected"], catalog_us=catalog_us, planning_us=choice["planning_us"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path, help="calibration-only or complete measure.py JSON[.gz]")
    by_name = {w["id"]: w for w in workloads()}
    parser.add_argument("--workload", choices=by_name, required=True)
    parser.add_argument("--budget-bytes", type=int, required=True)
    parser.add_argument("--objective", choices=("warm", "lifecycle", "slo"), default="warm")
    parser.add_argument("--plan-only", action="store_true", help="emit the executor's --plan-json argument")
    args = parser.parse_args()
    work = by_name[args.workload]
    if args.objective == "slo" and not work["arrivals_ms"]:
        parser.error("this workload has no qualified arrival train")
    if args.budget_bytes <= 0:
        parser.error("budget must be positive")
    opener = gzip.open if args.profile.suffix == ".gz" else open
    with opener(args.profile, "rt") as stream:
        data = json.load(stream)
    if data["version"] != 1:
        parser.error("unknown calibration format")
    choice = recommend(data["calibration"], work, args.budget_bytes, args.objective)
    if not choice["plan"]:
        if args.plan_only:
            parser.exit(1, choice["reason"] + "\n")
        print(json.dumps(choice, indent=2))
        raise SystemExit(1)
    print(json.dumps(choice["plan"] if args.plan_only else choice, indent=2))


if __name__ == "__main__":
    main()
