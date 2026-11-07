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
