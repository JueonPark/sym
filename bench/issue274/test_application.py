import pytest

from bench.issue274.application import nearest_rank, paired_ratio


def test_nearest_rank_keeps_tail_observation():
    assert nearest_rank(range(1, 21), .50) == 10
    assert nearest_rank(range(1, 21), .95) == 19


def test_paired_interval_uses_round_ratios():
    result = paired_ratio([10, 200, 150], [20, 400, 300])
    assert result['ratios_by_round'] == [.5, .5, .5]
    assert result['geomean'] == pytest.approx(.5)
    assert result['bootstrap_95'] == pytest.approx([.5, .5])
