import pytest
from bench.issue263.analyze import pair_rows, lifecycle_metrics


def test_pairing_rejects_missing_round_and_preserves_round_identity():
    a=[{'round':1,'value':100},{'round':2,'value':200}]
    b=[{'round':2,'value':180},{'round':1,'value':90}]
    assert pair_rows(a,b,'value',2)==[{'round':1,'baseline':100,'candidate':90},{'round':2,'baseline':200,'candidate':180}]
    with pytest.raises(ValueError,match='round'): pair_rows(a,b[:1],'value',2)


def test_lifecycle_uses_accounted_envelope_not_oracle_wall_time():
    phases={k:1 for k in ('cuda_init_ms','fixture_ms','snapshot_total_ms','request_owner_ms','shape_update_ms','scale_reprepare_ms','owner_retirement_ms','caller_release_and_cache_clear_ms')}
    x=dict(launch_to_pre_cuda_ms=2,phases=phases,requests=[{'completed_ms':3,'phase':'initial'}]*100+
           [{'completed_ms':5,'phase':'shape_change'}]*10+[{'completed_ms':7,'phase':'scale_change'}]*10)
    m=lifecycle_metrics(x)
    assert m['full120_accounted_ms']==430
    assert m['warm_p50_ms']==3 and m['warm_p95_ms']==3
    assert m['shape_first_ms']==5 and m['scale_first_ms']==7


def test_complete_flag_cannot_hide_a_missing_model_case(tmp_path,monkeypatch):
    import json
    from bench.issue263.analyze import collect
    # This fixture isolates coverage validation; execution qualification has
    # separate mixed-platform/source tests in test_contract.py.
    monkeypatch.setattr('bench.issue263.analyze.load_contract',lambda _:({},False))
    monkeypatch.setattr('bench.issue263.analyze.validate_record',lambda *a,**kw:None)
    job=dict(name='e2',kind='models',round=1,variant='baseline',mode='off',profile=False,accepted=True)
    names=('support.py','instrument.py','interventions.py','models.py','e5.py','lifecycle.py','concurrency.py')
    x=dict(complete=True,isolation=dict(valid=True,audit='audit.json'),campaign_job=job,host_phases=[],
        metadata=dict(baseline_hash_mismatches=[],harness_sha256=dict.fromkeys(names,'hash'),runtime_sha256={},artifact_sha256={}),
        configuration=dict(samples=60,warmup=3,computes=['eager','inductor'],cases=['llm_decode','llm_prefill','moe_sparse','moe_dense']),cases={})
    (tmp_path/'matrix.json').write_text(json.dumps([job]))
    (tmp_path/'e2.json').write_text(json.dumps(x))
    (tmp_path/'audit.json').write_text(json.dumps(dict(returncode=0,observations=[])))
    with pytest.raises(AssertionError,match='case coverage'):collect(tmp_path)


def test_dependency_witness_excludes_kernel_enqueued_before_stream_wait():
    from bench.issue263.analyze import dependency_witnesses
    row=dict(dependencies=[dict(eventId=7,streamId=2,end=21,wait_api=dict(start=20,end=21),
            record=dict(streamId=1,api_start=10))],activities=[
        dict(streamId=1,launch_start=9,start=25,end=30,kind='KERNEL',group='g0',name='producer'),
        dict(streamId=2,launch_start=15,start=31,end=34,kind='KERNEL',group=None,name='queued_before_wait'),
        dict(streamId=2,launch_start=22,start=35,end=38,kind='KERNEL',group=None,name='queued_after_wait')])
    assert dependency_witnesses(row)[0]['consumer_next_kernel']['name']=='queued_after_wait'
