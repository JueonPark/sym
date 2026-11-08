"""Reject incomplete or incomparable measurements rather than silently average."""
import copy
import pytest


def record():
    return dict(job=dict(kind='e2', case='llm_decode', compute='eager', callers=1,
                        round=1, runtime='sym', mode='on', id='fixture'),
                protocol=dict(dtype='float16', kernel='native', grouping='grouped',
                    prefetch='whole', slots=2, tf32=False, samples=60, warmup_requests_per_caller=5),
                warmup_requests_per_caller=[5], outcome='success', checks=60, retained_checks=60, held_after_close=1,
                completed_ms=[1.] * 60, submission_ms=[.5] * 60,
                first_ms=5., preparation_ms=2., retirement_ms=.2, warmup_ms=[3.], initial_nvml={'used': 10},
                compile_before=[{'compile_callbacks':0}], compile_after=[{'compile_callbacks':0}], input_sha256=['frozen'],
                warm_start_ms=10., warm_end_ms=80.,
                replay_before=[dict(captures=2, replays=3, frozen=True, retained_bytes=1024, max_bytes=4096)],
                replay_after=[dict(captures=2, replays=63, frozen=True, retained_bytes=1024, max_bytes=4096)],
                torch_memory=dict(allocated=1024, reserved=2048, max_allocated=4096, max_reserved=4096),
                environment=dict(torch='2.14.0+cu126', python=[3,14,7], gpu='NVIDIA GeForce RTX 2080 Ti',
                                 affinity=[4,5,6,7,20,21,22,23], threads=8, interop=1))


def validate(value):
    from bench.issue268.analyze import validate_record
    validate_record(value, record()['job'])


def test_accept_complete_record():
    validate(record())


@pytest.mark.parametrize('change', [
    lambda r: r['protocol'].update(tf32=True),
    lambda r: r['protocol'].update(grouping='split'),
    lambda r: r.update(checks=59),
    lambda r: r.update(retained_checks=0),
    lambda r: r.update(held_after_close=0),
    lambda r: r['completed_ms'].pop(),
    lambda r: r['completed_ms'].__setitem__(0, float('nan')),
    lambda r: r['submission_ms'].__setitem__(0, -1),
    lambda r: r.update(first_ms=None),
    lambda r: r['replay_after'][0].update(captures=3),
    lambda r: r['replay_after'][0].update(retained_bytes=8192),
    lambda r: r['environment'].update(threads=16),
    lambda r: r['torch_memory'].update(max_reserved=0),
    lambda r: r.update(warm_end_ms=9.),
])
def test_reject_bad_record(change):
    value = record(); change(value)
    with pytest.raises((ValueError, KeyError, TypeError)):
        validate(value)


def test_audit_requires_measured_quiet_interval():
    from bench.issue268.analyze import validate_audit
    good = dict(exit_code=0, foreign_busy_processes=[], observations=2,
                preflight=dict(quiet_seconds=60.1, quiet_observations=21))
    validate_audit(good)
    for bad in (dict(quiet_seconds=59.9, quiet_observations=21),
                dict(quiet_seconds=60., quiet_observations=0), {}):
        with pytest.raises((ValueError, KeyError)):
            validate_audit(good | {'preflight': bad})
    with pytest.raises(ValueError):
        validate_audit(good | {'foreign_busy_processes': [[1, 'compiler']]})


def test_memory_requires_warm_observations():
    from bench.issue268.analyze import memory_peak
    assert memory_peak([[1.,100,20],[20.,120,30],[90.,110,30]], 10.,80.) == 120
    with pytest.raises(ValueError):
        memory_peak([[1.,100,20],[90.,110,30]],10.,80.)
    with pytest.raises(ValueError):
        memory_peak([[20.,100,20],[19.,110,30]],10.,80.)


def test_exact_membership():
    from bench.issue268.analyze import membership
    membership(['a','b'], ['a','b'])
    for rows in (['a'], ['a','b','b'], ['a','c']):
        with pytest.raises(ValueError):
            membership(rows, ['a','b'])


def test_saved_campaign_checks_hashes_and_provenance(tmp_path, monkeypatch):
    import gzip
    import json
    from bench.issue268 import contract
    from bench.issue268.analyze import analyze
    row=record(); job=row['job']; row['provenance']={'source': {'runtime.py':'frozen'}}
    trace=tmp_path/'memory.json.gz'
    trace.write_bytes(gzip.compress(json.dumps({'samples': [[20.,100,20],[40.,120,20]]}).encode()))
    row['memory']=dict(file=trace.name,sha256=contract.sha(trace))
    contract.save(tmp_path/'record.json',row)
    contract.save(tmp_path/'audit.json',dict(exit_code=0,foreign_busy_processes=[],observations=2,
                   preflight=dict(quiet_seconds=60.,quiet_observations=21)))
    saved=dict(path='record.json',sha256=contract.sha(tmp_path/'record.json'),
               audit_path='audit.json',audit_sha256=contract.sha(tmp_path/'audit.json'))
    manifest=dict(qualification=True,jobs=[job],records={job['id']:saved},provenance=row['provenance'],excluded=[])
    contract.save(tmp_path/'manifest.json',manifest)
    monkeypatch.setattr(contract,'jobs',lambda:[job])
    assert analyze(tmp_path)[0]['p50_ms']==1.
    manifest['provenance']={'source':{'runtime.py':'changed'}}
    contract.save(tmp_path/'manifest.json',manifest)
    with pytest.raises(ValueError,match='provenance'): analyze(tmp_path)
    manifest['provenance']=row['provenance']; contract.save(tmp_path/'manifest.json',manifest)
    (tmp_path/'record.json').write_text('{}')
    with pytest.raises(ValueError,match='checksum'): analyze(tmp_path)


def micro_record():
    rows=[]
    for scenario in ('stable','varying','pressure'):
        for mode in ('plain','off','on'):
            stats=None if mode=='plain' else dict(captures=2,replays=100 if mode=='on' else 0,
                 frozen=True,retained_bytes=1024,max_bytes=4096,fallbacks={})
            rows.append(dict(scenario=scenario,mode=mode,checks=100,retained_checks=100,
                completed_ms=[1.]*100,submission_ms=[.5]*100,preparation_ms=3.,first_ms=2.,
                warm_start_ms=10.,warm_end_ms=50.,before=copy.deepcopy(stats),after=stats,
                closed=None if mode=='plain' else dict(retained_bytes=0,closed=True)))
    return dict(qualification=True,compute='eager',round=2,rows=rows,
                environment=record()['environment'],preflight=dict(quiet_seconds=60.,quiet_observations=21),
                audit=dict(observations=2,foreign_busy_processes=[]))


def test_supplement_requires_complete_shapes_budget_and_checks():
    from bench.issue268.analyze import validate_micro
    value=micro_record(); validate_micro(value)
    for change in (lambda r:r['rows'].pop(),lambda r:r['rows'][0].update(checks=99),
                   lambda r:r.update(qualification=False),lambda r:r['audit'].update(foreign_busy_processes=[1]),
                   lambda r:r['rows'][1]['closed'].update(retained_bytes=100)):
        bad=copy.deepcopy(value);change(bad)
        with pytest.raises(ValueError):validate_micro(bad)


def test_runner_waits_for_full_quiet_interval_even_without_wait_flag(monkeypatch):
    from bench.issue268 import run
    now=[0.]
    monkeypatch.setattr(run,'audit',lambda allowed:[])
    monkeypatch.setattr(run.time,'monotonic',lambda:now[0])
    monkeypatch.setattr(run.time,'sleep',lambda seconds:now.__setitem__(0,now[0]+seconds))
    observed=run.quiet(False,False)
    assert observed['quiet_seconds']>=60.
    assert observed['quiet_observations']>=20


def test_runner_restarts_quiet_interval_after_interference(monkeypatch):
    from bench.issue268 import run
    now=[0.]
    monkeypatch.setattr(run,'audit',lambda allowed:[(99,'compiler')] if now[0]==30 else [])
    monkeypatch.setattr(run.time,'monotonic',lambda:now[0])
    monkeypatch.setattr(run.time,'sleep',lambda seconds:now.__setitem__(0,now[0]+seconds))
    observed=run.quiet(True,False)
    assert now[0]>=90.
    assert observed['quiet_seconds']>=60.


def test_reject_warm_inductor_recompile():
    value=record();value['compile_before']=[{'compile_callbacks':1}];value['compile_after']=[{'compile_callbacks':2}]
    with pytest.raises(ValueError,match='compile'):validate(value)


def test_reject_too_few_executed_warmups():
    value=record();value['warmup_requests_per_caller']=[3]
    with pytest.raises(ValueError,match='warmup'):validate(value)


def test_parent_win_requires_each_round_below_both_controls():
    from bench.issue268.analyze import parent_targets
    rows=[]
    for round_ in (1,2,3):
        for runtime,compute,value in (('torch','eager',10.),('torch','inductor',8.),
                ('sym','eager',7.5 if round_<3 else 7.7),('sym','inductor',7.4)):
            rows.append(dict(kind='e2',case='llm_decode',callers=1,mode='on',runtime=runtime,
                             compute=compute,round=round_,outcome='success',p50_ms=value))
    values={r['compute']:r for r in parent_targets(rows)}
    assert values['eager']['passed'] is False
    assert values['eager']['rounds_passed']==2
    assert values['inductor']['passed'] is True


def test_summary_counts_only_warm_replays():
    from bench.issue268.analyze import summarize
    assert summarize(record(),120)['warm_replays']==60
