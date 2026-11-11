import pytest
from bench.issue263.paired import paired_summary, percentile


def test_process_pairs_and_descriptive_quantiles():
    r=paired_summary([{'round':1,'baseline':100,'candidate':90}, {'round':2,'baseline':200,'candidate':180}])
    assert r['reduction_percent'] == 10
    assert r['ci95_percent'] == [10,10]
    assert r['process_pairs'] == 2
    assert percentile([1,2,3,4],95) == pytest.approx(3.85)


@pytest.mark.parametrize('rows', [[],[{'round':1,'baseline':0,'candidate':1}],
    [{'round':1,'baseline':1}], [{'round':1,'baseline':2,'candidate':1}]*2])
def test_invalid_pairs_fail(rows):
    with pytest.raises(ValueError): paired_summary(rows)
