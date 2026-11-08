"""The qualification gate must fail closed on incomplete or altered evidence."""
import copy
import pytest
from bench.issue269 import contract


def analyzer():
    from bench.issue269 import analyze
    return analyze


def row():
    job=next(j for j in contract.jobs() if j['kind']=='model')
    return job,dict(job=job,protocol=contract.protocol(job),environment=contract.expected_environment(job),
        outcome='success',warmups=5,checks=30,retained_checks=30,held_after_close=1,
        completed_ms=[1.]*30,submission_ms=[.9]*30,preparation_ms=3.,first_ms=2.,
        warmup_ms=5.,retirement_ms=.2,compile_before=4,compile_after=4,
        replay_before=[],replay_after=[],warm_start_ms=10.,warm_end_ms=40.,
        phases=[dict(name=n,start_ms=a,end_ms=b) for n,a,b in
            [('fixture',0.,1.),('prepare',1.,4.),('first',4.,6.),('warmup',6.,10.),('warm',10.,40.),('retirement',40.,41.)]],
        torch_memory=dict(allocated=100,max_allocated=200,reserved=300,max_reserved=400),
        input_sha256=['0'*64]*8,transfer_before={},transfer_after={},lowering=[],
        compile_ms=2.,compiler_graphs=[])


def test_declared_model_membership_and_counterbalance():
    jobs=contract.jobs()
    assert len(jobs)==168 and len({j['id'] for j in jobs})==168
    assert sum(j['kind']=='model' for j in jobs)==162
    assert sum(j['variant']=='resident' for j in jobs)==18
    assert sum(j['replay'] for j in jobs)==36
    assert next(j for j in jobs if j['kind']=='model' and j['round']==2)['runtime']=='sym'


def test_valid_record():
    job,value=row();analyzer().validate_record(value,job)


@pytest.mark.parametrize('field,value',[
    ('checks',29),('retained_checks',29),('held_after_close',0),('warmups',3),
    ('completed_ms',[1.]*29),('submission_ms',[1.]*29),('compile_after',5),
    ('first_ms',-1.),('preparation_ms',float('nan')),('warm_end_ms',10.),
    ('input_sha256',[]),('outcome','unknown')])
def test_bad_record_rejected(field,value):
    job,data=row();data[field]=value
    with pytest.raises(ValueError):analyzer().validate_record(data,job)


def test_protocol_and_environment_must_match():
    job,data=row();data['protocol']['tf32']=True
    with pytest.raises(ValueError):analyzer().validate_record(data,job)
    job,data=row();data['environment']['threads']=16
    with pytest.raises(ValueError):analyzer().validate_record(data,job)


def test_missing_duplicate_matrix_and_quiet_gate():
    with pytest.raises(ValueError):analyzer().membership(['a','a'],['a','b'])
    valid=dict(exit_code=0,foreign_busy_processes=[],observations=2,
               preflight=dict(quiet_seconds=60.,quiet_observations=21))
    analyzer().validate_audit(valid)
    for patch in ({'foreign_busy_processes':[[1,'build']]},{'exit_code':1},{'observations':0},
                  {'preflight':dict(quiet_seconds=59.,quiet_observations=21)}):
        with pytest.raises(ValueError):analyzer().validate_audit(valid|patch)


def test_no_unsampled_memory_peak_invention():
    with pytest.raises(ValueError):analyzer().memory_peak([[0.,100,200]],1.,2.)
    assert analyzer().phase_peaks([[0.,100,200]],[dict(name='first',start_ms=1.,end_ms=2.)])[0]['peak_bytes'] is None


def test_oom_requires_actual_phase_and_error():
    job,data=row();data.update(outcome='oom',phase='first',error='CUDA out of memory')
    analyzer().validate_record(data,job)
    data['error']=''
    with pytest.raises(ValueError):analyzer().validate_record(data,job)


def test_report_separates_finite_warm_slope_from_setup():
    job,data=row();out=analyzer().summarize(data,400)
    assert out['p50_ms']==1. and out['setup_ms']==10.
    assert out['total_ms']['100']==110.


def micro_row():
    job=next(j for j in contract.jobs() if j['kind']=='micro')
    rows=[]
    for case in contract.MICROS:
        for variant in contract.VARIANTS:
            rows.append(dict(case=case,variant=variant,outcome='success',checks=100,retained_checks=100,
                held_after_close=1,warmups=5,compile_before=1,compile_after=1,
                completed_ms=[1.]*100,submission_ms=[.9]*100,consumer_ms=[.4]*100,transfer_ms=[.2]*100,
                warm_start_ms=1.,warm_end_ms=2.,preparation_ms=1.,first_ms=1.,warmup_ms=1.,retirement_ms=1.,
                budget=dict(requested_free_budget=128<<20,after=dict(free=128<<20)),input_sha256=['1'*64]*4,
                lowering=[dict(selected=True,eliminated_bytes=case['length']*case['heads']*case['dim']*2)]*2
                         if variant=='lowered' and not case['visible'] else []))
    return job,dict(job=job,environment=contract.expected_environment(job),protocol=contract.protocol(job),rows=rows)


@pytest.mark.parametrize('change',['consumer','transfer','selected','mixed','inputs','budget_oom','missing','duplicate'])
def test_micro_fails_closed(change):
    job,record=micro_row();analyzer().validate_micro(record,job)
    rows=record['rows']
    if change=='consumer':rows[0]['consumer_ms']=[]
    elif change=='transfer':rows[0]['transfer_ms'][0]=float('nan')
    elif change=='selected':rows[2]['lowering']=[]
    elif change=='mixed':next(r for r in rows if r['case']['visible'] and r['variant']=='lowered')['lowering']=[dict(selected=True)]
    elif change=='inputs':rows[0]['input_sha256']=[]
    elif change=='budget_oom':
        rows[-1].update(outcome='oom',phase='warm',error='CUDA out of memory');rows[-1]['budget']['after']['free']=500<<20
    elif change=='missing':rows.pop()
    elif change=='duplicate':rows[-1]=rows[0]
    with pytest.raises(ValueError):analyzer().validate_micro(record,job)


def test_generated_evidence_hashes(tmp_path):
    p=tmp_path/'graph.py';p.write_text('original')
    row=dict(compiler_graphs=[dict(file=p.name,sha256=contract.sha(p))])
    analyzer().validate_files(tmp_path,row)
    p.write_text('changed')
    with pytest.raises(ValueError):analyzer().validate_files(tmp_path,row)
    row=dict(profile=dict(file='missing.gz',sha256='0'*64))
    with pytest.raises((ValueError,FileNotFoundError)):analyzer().validate_files(tmp_path,row)
