"""Bounded, opt-in search over qualified, executable relocation region schedules.

Events appear in the executor's submission order. Edges describe data, stream
and reuse dependencies. Each event holds all its named resources exclusively
for its entire duration. Shared resources guarantee FIFO order, even when an
earlier submission waits for an input. Independent resources can overlap.
"""
from collections import defaultdict
from dataclasses import dataclass
import math
from time import perf_counter_ns


def _number(value, field, minimum=0):
    if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
        raise ValueError(f'invalid {field}')
    return float(value)


def _integer(value, field, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'invalid {field}')
    return value


def _rows(value, field, maximum, minimum=0):
    if not isinstance(value, (list, tuple)) or not minimum <= len(value) <= maximum:
        raise ValueError(f'invalid or out-of-bounds {field}')
    return value


def _name(value):
    if not isinstance(value, str) or not 0 < len(value) <= 256:
        raise ValueError('missing or invalid name')
    return value


def _completion(start, duration):
    # A positive sub-ULP residual still owns its resources and live storage.
    # Round forward by a tick when adding it would erase the interval.
    end = start + duration
    if end == start:
        end = math.nextafter(start, math.inf)
    return _number(end, 'event completion')


@dataclass
class Event:
    name: str
    us: float
    lanes: tuple
    after: set


def _dag(raw):
    events = []
    ends = {}
    previous = {}
    baseline = {}
    for row in _rows(raw, 'events', 1024, 1):
        name = _name(row['id'])
        if name in ends:
            raise ValueError('duplicate event')
        us = _number(row['us'], 'event duration')
        if us == 0:
            raise ValueError('event duration must be positive')
        lanes = tuple(_name(lane) for lane in _rows(row['resources'], 'resources', 16, 1))
        after = set(_rows(row['deps'], 'dependencies', 1024))
        if not after.issubset(ends) or len(set(lanes)) != len(lanes):
            raise ValueError('invalid dependency or resource')
        after.update(previous[lane] for lane in lanes if lane in previous)
        start = max((ends[parent] for parent in after), default=0.)
        ends[name] = _completion(start, us)
        baseline[name] = dict(start_us=start, end_us=ends[name])
        previous.update(dict.fromkeys(lanes, name))
        events.append(Event(name, us, lanes, after))
    if len(previous) > 64:
        raise ValueError('resource count exceeds limit')
    return events, baseline


def _rules(raw):
    rules = {}
    for rule in _rows(raw, 'overlap slowdowns', 128):
        pair = tuple(sorted(_name(r) for r in _rows(rule['resources'], 'overlap pair', 2, 2)))
        if pair[0] == pair[1] or pair in rules:
            raise ValueError('ambiguous overlap pair')
        rules[pair] = _number(rule['factor'], 'overlap slowdown', 1)
    return rules


def _contended_schedule(events, rules):
    """Integrate calibrated service rates between starts and completions."""
    children = defaultdict(list)
    pending = {}
    remaining = {}
    lanes = {}
    for e in events:
        pending[e.name] = len(e.after)
        remaining[e.name] = e.us
        lanes[e.name] = e.lanes
        for parent in e.after:
            children[parent].append(e.name)
    ready = [e.name for e in events if not e.after]
    active = set()
    schedule = {}
    now = 0.
    while ready or active:
        for name in ready:
            active.add(name)
            schedule[name] = dict(start_us=now)
        ready = []
        factors = dict.fromkeys(active, 1.)
        owners = {lane: name for name in active for lane in lanes[name]}
        for (first, second), factor in rules.items():
            a, b = owners.get(first), owners.get(second)
            if a is not None and b is not None and a != b:
                factors[a] = max(factors[a], factor)
                factors[b] = max(factors[b], factor)
        times = {name: remaining[name] * factors[name] for name in active}
        step = min(times.values())
        end = _completion(now, step)
        for name, delta in times.items():
            if delta > step:
                remaining[name] = (delta - step) / factors[name]
                continue
            active.remove(name)
            schedule[name]['end_us'] = end
            for child in children[name]:
                pending[child] -= 1
                if pending[child] == 0:
                    ready.append(child)
        now = end
    return schedule


def _memory_bound(memory):
    persistent = sum(_integer(memory[key], key)
                     for key in ('resident_bytes', 'retained_bytes', 'context_bytes'))
    observed = max(_integer(memory[key], key) for key in ('setup_peak_bytes', 'warm_floor_bytes'))
    return persistent, max(persistent, observed)


def _peak_bytes(memory, schedule, persistent, bound):
    # Half-open lifetimes allow a buffer to be reused at completion.
    changes = defaultdict(int)
    for row in _rows(memory['intervals'], 'memory intervals', 8192):
        first, last = schedule[row['begin']], schedule[row['end']]
        if last['end_us'] < first['end_us']:
            raise ValueError('release before allocation event completes')
        size = _integer(row['bytes'], 'interval bytes')
        changes[first['start_us']] += size
        changes[last['end_us']] -= size
    live, peak = persistent, bound
    for t in sorted(changes):
        live += changes[t]
        peak = max(peak, live)
    return peak


def _load_point(load, service):
    arrivals = [_number(a, 'arrival') for a in _rows(load['arrivals_us'], 'arrivals', 16384, 1)]
    if any(b < a for a, b in zip(arrivals, arrivals[1:])):
        raise ValueError('arrivals must be in request order')
    if 'completions_us' in load:
        values = _rows(load['completions_us'], 'completions', len(arrivals), len(arrivals))
        completions = [_number(v, 'completion') for v in values]
        if any(b <= a for a, b in zip(completions, completions[1:])):
            raise ValueError('calibration completions violate single-executor FCFS order')
    else:
        # Open-loop arrivals continue independently while the single executor is busy.
        completions = []
        previous = arrivals[0]
        for arrival in arrivals:
            previous = _number(max(previous, arrival) + service, 'modeled completion')
            completions.append(previous)
    sojourns = sorted(c - a for a, c in zip(arrivals, completions))
    if sojourns[0] <= 0:
        raise ValueError('invalid request completion')
    p95 = sojourns[(95 * len(sojourns) + 99) // 100 - 1]
    rate = _number(1e6 * len(arrivals) / (max(completions) - arrivals[0]), 'completed throughput')
    return p95, rate


def _slo(loads, service, limit):
    points = []
    for i, load in enumerate(_rows(loads, 'load points', 128, 1)):
        tail, rate = _load_point(load, service)
        if tail <= limit:
            points.append((-rate, tail, i))
    if not points:
        raise ValueError('no offered load meets the sojourn SLO')
    negative_rate, tail, index = min(points)
    return dict(throughput_rps=-negative_rate, p95_us=tail, load_index=index)


def choose_plan(candidates, *, budget_bytes, objective='warm', reuses=1,
                slo_us=None, max_candidates=256):
    """Select by (objective, peak bytes, name) over a finite catalog.

    Each row must explicitly attest legality/executability with qualified=True.
    Capacity is the larger of the absolute setup/warm byte floors and
    persistent (resident + retained + context) plus simultaneous temporary
    byte intervals. Those persistent byte counts must be disjoint. Floors
    may be derived from a compile-time allocation contract or observed in a
    separate physical calibration. The caller records their provenance.
    A contract-derived floor alone does not establish a physical peak.

    Warm cost is completed makespan. Lifecycle cost includes setup once and N
    completed requests, including request one, with calibration excluded. The
    SLO objective maximizes completions per second at offered loads meeting a
    nearest-rank p95 sojourn limit. Supply arrival timestamps to model one FCFS
    executor, or supply its independent calibration completions at matching loads.
    Never pass held-out evaluation completions to the planner.

    The byte floor and unloaded critical path are admissible bounds. Candidates
    are examined in name order. Strictly dominated warm/lifecycle bounds skip
    the slowdown and lifetime sweeps. Bounds on catalog, events, lanes, loads
    and samples limit work independently of timing noise. An oversized catalog
    returns no selection, so enumeration order cannot hide a better candidate.
    """
    started = perf_counter_ns()
    _integer(budget_bytes, 'budget')
    _integer(reuses, 'reuses', 1)
    _integer(max_candidates, 'max_candidates', 1)
    if objective not in ('warm', 'lifecycle', 'slo'):
        raise ValueError('unknown objective')
    if objective == 'slo':
        _number(slo_us, 'sojourn SLO')
    result = dict(selected=None, prediction=None, fallback=True, reason='no qualified feasible plan',
                  examined=0, rejected={}, pruned={})
    best = None
    try:
        catalog = {}
        for i, p in enumerate(_rows(candidates, 'catalog', min(max_candidates, 1024))):
            try:
                name = _name(p['name'])
            except (KeyError, TypeError, ValueError) as err:
                result['rejected'][f'row {i}'] = str(err)
                continue
            if name in catalog:
                raise ValueError(f'duplicate executable plan name: {name}')
            catalog[name] = p
        for name, p in sorted(catalog.items()):
            result['examined'] += 1
            try:
                if p['qualified'] is not True:
                    raise ValueError('missing execution qualification')
                setup = _number(p['setup_us'], 'setup')
                persistent, floor = _memory_bound(p['memory'])
                if floor > budget_bytes:
                    raise ValueError(f'byte floor {floor} exceeds budget')
                events, schedule = _dag(p['events'])
                rules = _rules(p.get('overlap_slowdowns', []))
                warm = max(t['end_us'] for t in schedule.values())
                lower = setup + reuses * warm if objective == 'lifecycle' else warm
                if objective != 'slo' and best is not None and lower > best[0]:
                    result['pruned'][name] = f'critical path lower bound {lower} us exceeds incumbent'
                    continue
                if rules:
                    schedule = _contended_schedule(events, rules)
                    warm = max(t['end_us'] for t in schedule.values())
                peak = _peak_bytes(p['memory'], schedule, persistent, floor)
                if peak > budget_bytes:
                    raise ValueError(f'live peak {peak} exceeds budget')
                lifecycle = _number(setup + reuses * warm, 'lifecycle cost')
                pred = dict(warm_us=warm, setup_us=setup, lifecycle_us=lifecycle,
                            amortized_us=lifecycle / reuses, peak_bytes=peak, schedule=schedule,
                            throughput_rps=None, p95_us=None, load_index=None)
                value = lifecycle if objective == 'lifecycle' else warm
                if objective == 'slo':
                    pred.update(_slo(p['loads'], warm, slo_us))
                    value = -pred['throughput_rps']
            except (TypeError, KeyError, ValueError, OverflowError) as err:
                result['rejected'][name] = str(err)
                continue
            key = (value, peak, name)
            if best is None or key < best:
                best = key
                result.update(selected=name, prediction=pred, fallback=False,
                              reason=f'best feasible {objective} objective within {budget_bytes} bytes')
    except ValueError as err:
        result['reason'] = str(err)
    result['planning_us'] = (perf_counter_ns() - started) / 1000.
    return result
