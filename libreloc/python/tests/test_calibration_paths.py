"""The committed cost-advice keys parse and price the fast lane's paths."""
from pathlib import Path

import pyreloc

ROOT = Path(__file__).resolve().parents[3]
KEYS = ('overhead.direct.pooled_ms', 'overhead.dispatch.pooled_ms', 'overhead.direct.indexed_rows_ms',
        'overhead.dispatch.indexed_rows_ms', 'overhead.direct.dequant_ms', 'overhead.dispatch.dequant_ms',
        'overhead.dispatch.layout_ms', 'overhead.group_ms', 'overhead.group_member_ms',
        'cpu.t8.contiguous.pooled_sum_f16_gbps', 'cpu.t8.indexed_rows.gather_f16_gbps',
        'recv.m.dequant_s8_f16', 'hbm.m.indexed_rows')


def test_epyc_calibration_carries_and_prices_every_advice_key():
    cal = pyreloc.load_calibration(str(ROOT / 'calibration/epyc7351-2080ti.cal'))
    for key in KEYS:
        assert cal.has(key), key
        assert cal.value(key) >= 0, key
    for kind in ('pooled', 'indexed_rows', 'dequant'):
        # The premise of the direct path: it has the smaller fixed cost.
        assert cal.value(f'overhead.direct.{kind}_ms') < cal.value(f'overhead.dispatch.{kind}_ms'), kind
    costs = pyreloc.price_advice(cal, [dict(name='direct', form='serial', intercept='overhead.direct.pooled_ms',
                                            member_intercept=None, chunks=None,
                                            terms=[[1 << 20, 'cpu.t8.contiguous.pooled_sum_f16_gbps', None],
                                                   [1 << 20, 'pcie.h2d_gbps', None],
                                                   [1 << 20, 'hbm.bw_gbps', 'hbm.m.contiguous']])])
    assert costs[0]['missing_key'] == '' and costs[0]['ms'] > 0


def test_gen4_calibration_omits_the_keys_until_measured():
    cal = pyreloc.load_calibration(str(ROOT / 'calibration/7800x3d-4070tis.cal'))
    assert not any(cal.has(key) for key in KEYS)
