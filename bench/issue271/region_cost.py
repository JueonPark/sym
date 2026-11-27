"""Instantiate the bounded planner from executable region facts and component probes."""
import bisect
import math
import statistics
from collections import defaultdict

from .measure import byte_terms, plans
from .planner import choose_plan


class MissingCalibration(ValueError):
    pass


class ComponentProfile:
    """Median microseconds at each calibrated size, interpolated inside coverage."""
    def __init__(self, rows):
        self.points = defaultdict(dict)
        for row in rows:
            kind, size = row["component"], row["elements"]
            values = row["samples_ms"]
            if not values or not all(math.isfinite(v) and v > 0 for v in values):
                raise ValueError(f"invalid {kind} calibration samples")
            if size in self.points[kind]:
                raise ValueError(f"duplicate {kind} calibration at {size}")
            self.points[kind][size] = 1000 * statistics.median(values)

    def fixed(self, kind):
        costs = self.points.get(kind)
        if not costs:
            raise MissingCalibration(kind)
        return statistics.median(costs.values())

    def at(self, kind, size):
        points = self.points.get(kind, {})
        if size in points:
            return points[size]
        sizes = sorted(points)
        k = bisect.bisect_left(sizes, size)
        if not 0 < k < len(sizes):
            raise MissingCalibration(f"{kind}: {size} elements outside measured coverage")
        a, b = sizes[k-1:k+1]
        return points[a] + (points[b] - points[a]) * (size - a) / (b - a)


def plan_id(plan):
    return "{}-r{}-c{}-d{}-{}".format(
        plan["placement"], plan["resident"], plan["chunk"], plan["depth"],
        "reuse" if plan["reuse"] else "fresh",
    )


def describe(work, plan, profile):
    name = plan_id(plan)
    try:
        return _describe(name, work, plan, profile)
    except MissingCalibration as exc:
        return dict(name=name, qualified=False, exclusion=str(exc))


def _describe(name, work, plan, profile):
    size, groups, passes = work["elements"], work["groups"], work["passes"]
    placement, prefix = plan["placement"], plan["resident"]
    piece, depth = size // plan["chunk"], plan["depth"]
    terms = byte_terms(work, plan)
    events = []

    def event(label, lane, cost, deps=()):
        ident = f"{len(events)}.{label}"
        events.append(dict(id=ident, resources=[lane], us=cost, deps=list(deps)))
        return ident

    # Distinct streams keep FIFO order. A CPU slot wait prevents the submitter
    # from outrunning either the source lease or the last GPU read of that slot.
    output_elements = size * groups
    start = event("open-output", "cpu", profile.at("allocate_f16", output_elements))
    last_consumer = start
    for _ in range(prefix):
        submit = event("submit-resident", "cpu", (1 + passes) * profile.fixed("launch"), [start])
        last_consumer = event("resident-consume", "gpu", profile.at("device_copy", size)
                              + passes * profile.at("consumer", size), [submit, last_consumer])

    consumers = []
    leases = []
    for i in range((groups - prefix) * plan["chunk"]):
        deps = [consumers[i-depth]] if i >= depth else [start]
        if i >= depth:
            deps = [event("retire-slot", "cpu", profile.fixed("sync"), deps)]
        slot_us = 0
        if placement == "cpu":
            slot_us = profile.at("slot_setup", piece)  # Compact view plus pinned host slot.
        else:
            slot_us = profile.at("allocate_f32", piece)
            if placement == "gpu":
                slot_us += profile.at("allocate_f16", piece)
        if not plan["reuse"]:
            acquire = event("acquire-slot", "cpu", slot_us, deps)
            deps = [acquire]
        if placement == "cpu":
            deps = [event("pack", "cpu", profile.at("cpu_pack", piece), deps)]
        submit = event("submit-copy", "cpu", profile.fixed("launch"), deps)
        link = "h2d_f16" if placement == "cpu" else "h2d_f32"
        ready = event("h2d", "link", profile.at(link, piece), [submit])
        # CUDA submissions are asynchronous on the host. Transfer and kernel
        # execution can begin once their submission and data dependencies finish.
        kernels = 1 if placement == "fused" else passes + int(placement != "cpu") + int(placement != "forward")
        submit_gpu = event("submit-kernels", "cpu", kernels * profile.fixed("launch"), [submit])
        if placement == "fused":
            gpu_us = profile.at(f"fused_consumer_{passes}", piece)
        else:
            gpu_us = passes * profile.at("consumer", piece)
            if placement != "cpu":
                gpu_us += profile.at("gpu_cast", piece)
            if placement != "forward":
                gpu_us += profile.at("device_copy", piece)
        last_consumer = event("convert-consume", "gpu", gpu_us, [ready, submit_gpu, last_consumer])
        consumers.append(last_consumer)
        if not plan["reuse"]:
            leases.append(dict(begin=acquire, end=last_consumer, bytes=terms["slot"]))
    # The executor joins the compute stream and retires the last occupied slots.
    end = event("complete", "cpu", (1 + terms["inflight"]) * profile.fixed("sync"), [last_consumer])

    setup = profile.fixed("setup_fixed") + profile.fixed("close_fixed")
    if prefix:
        setup += prefix * profile.at("resident_preload", size)
    if plan["reuse"] and terms["inflight"]:
        setup += terms["inflight"] * slot_us
    # The executor reserves an exact scratch window of depth slots. Even fresh
    # leases must have space for that window before work can be submitted.
    reserved_slots = terms["scratch"] if plan["reuse"] else 0
    memory = dict(resident_bytes=terms["resident"], retained_bytes=terms["retained"],
                  context_bytes=reserved_slots, setup_peak_bytes=terms["context_retained"],
                  warm_floor_bytes=terms["live_peak"],
                  intervals=[dict(begin=start, end=end, bytes=terms["output"]), *leases])
    slowdowns = []
    if groups > prefix and (depth > 1 or prefix > 0):
        # The probe co-runs raw H2D and one consumer on independent streams.
        copy_us = profile.at("h2d_f32", piece)
        compute_us = profile.at("consumer", piece)
        # With equal slowdown only during overlap, T=max(a,b)+(factor-1)*min(a,b).
        extra = max(0., profile.at("paired_link_compute", piece) - max(copy_us, compute_us))
        factor = 1 + extra / min(copy_us, compute_us)
        slowdowns.append(dict(resources=["link", "gpu"], factor=factor))
    loads = [dict(arrivals_us=[i * period * 1000 for i in range(work["paced_requests"])])
             for period in work["arrivals_ms"]]
    return dict(name=name, qualified=True, events=events, memory=memory, setup_us=setup,
                overlap_slowdowns=slowdowns, loads=loads)


def catalog(work, profile):
    return [describe(work, plan, profile) for plan in plans(work)]


def select(candidates, work, budget, objective):
    return choose_plan(candidates, budget_bytes=budget, objective=objective,
                       reuses=5, slo_us=work["slo_ms"] * 1000 if objective == "slo" else None)
