import json

import pytest


def test_matched_estimators_keep_arm_medians_and_paired_pct_distinct():
    from bench.issue274.baseline_analysis import contrast

    a = {1: 1, 2: 10, 3: 20}
    b = {3: 1, 1: 4, 2: 9}
    row = contrast(a, b)
    assert row['candidate_over_control'] == 0.4
    assert row['paired_effect_pct'] == pytest.approx(10)
    assert row['paired_bootstrap95_pct'] == pytest.approx([-300, 95])
    assert row['paired_rounds'] == [1, 2, 3]
    assert row['classification'] == 'inconclusive'


def test_contrast_rejects_nonpaired_inputs():
    from bench.issue274.baseline_analysis import contrast

    with pytest.raises(ValueError, match='same rounds'):
        contrast({1: 1, 2: 2}, {1: 1, 3: 2})


def test_all_oom_pairs_have_no_latency_ratio():
    from bench.issue274.baseline_analysis import contrast, paired_cell

    assert contrast({}, {}) == {'paired_rounds': [], 'classification': 'no paired success'}
    assert paired_cell(contrast({}, {})) == 'OOM'


def test_percentiles_keep_original_tail_convention():
    from bench.issue274.baseline_analysis import tail

    values = list(range(1, 21))
    assert tail(values, interpolate=False) == 19
    assert tail(values, interpolate=True) == pytest.approx(19.05)


def test_archive_checks_hash_and_inventory(tmp_path):
    from bench.issue274.baseline_analysis import hash_file, verify_archive

    sample = tmp_path / 'sample'
    sample.write_text('a')
    manifest = tmp_path / 'SHA256-MANIFEST.json'
    manifest.write_text(json.dumps({'algorithm': 'sha256', 'files': [
        {'path': 'sample', 'bytes': 1, 'sha256': hash_file(sample)},
    ]}))
    expected = hash_file(manifest)
    assert verify_archive(tmp_path, expected=expected)['files'] == 1
    sample.write_text('b')
    with pytest.raises(ValueError, match='changed'):
        verify_archive(tmp_path, expected=expected)
    sample.write_text('a')
    extra = tmp_path / 'extra'
    extra.touch()
    with pytest.raises(ValueError, match='inventory'):
        verify_archive(tmp_path, expected=expected)
    extra.unlink()
    sample.unlink()
    with pytest.raises(ValueError, match='inventory'):
        verify_archive(tmp_path, expected=expected)
