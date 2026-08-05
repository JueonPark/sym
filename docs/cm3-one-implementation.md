# CM3 — one cost-model implementation (issue #111)

`bench/rtrack/figure_rstar.py`'s standalone roofline model is retired; predictions now
come exclusively from `pyreloc.predict` (`reloc::costmodel`, the implementation
`decide()` uses). The four `b_fair` r* artifacts are regenerated alongside their
originals with the `_pyreloc` suffix; originals are unchanged (`gates.py`'s R2-G5 keeps
reading them; `r2_rstar_gen4.json` is the pre-`b_fair` R2-era record and is not
regenerated). Three distinct effects separate the old stored predictions from the new
files — conflating them would misattribute the change:

## Axis 1 — V3-era implementation divergence (the issue's "1.6–2.9×")

`figure_rstar.py`'s internal model had no intercepts (`overhead.{a,b}_ms`), no B-side
HBM term (`hbm.m.*`, CM1 #109), and no K/broadcast handling. Documented at
`docs/v3-costmodel.md` "Two-sided implementation note": stored gen4
`rstar_predicted` 0.1807 (blocked_transpose) / 0.1640 (quant) vs the V3-era
`pyreloc.predict` 0.2914 / 0.4806 on the same family/box.

## Axis 2 — model drift since V3 (CM1 + CM2)

The maintained model itself moved after the V3 snapshot: CM2's
`cpu_pipe.t8.contiguous.convert_f32_f16_gbps` keys shift every contiguous r=0.5
prediction (gen3 quant overlapped r* 0.9989 → 0.9966), and on gen4 the quant
crossing **disappears entirely** (r=0.5 predicted speedup 0.978 → 1.195 lifts the
whole grid above 1.0 → `rstar_predicted = None`). That is a live instance of the
`both_exist` → `mismatch_one_sided` reclassification fragility `docs/v3-costmodel.md`
§5 flagged; **CM4's tightened RSTAR rule must handle it** (hand-off).

## Axis 3 — the placement axis (new in these files)

All four artifacts were measured with `--b-method b_fair` — serial-B. The new files
therefore predict with `b_placement="serial"` (CM1's term; `--b-method` maps
b→overlapped, b_fair→serial). The frozen `v3_prediction_report.json` computed the
same families with the overlapped default. The numbers legitimately differ; neither
is wrong — they answer different questions (which B implementation is being priced).
For the three gen4-box arms (v1_gen4, avx2, avx512), Axis 2's model drift alone
already pushes the overlapped-placement `quant` curve entirely above 1.0
(r=0.125/0.25/0.5/1.0 speedup 1.187/1.366/1.195/1.002), so their new
`rstar_predicted = None` is primarily an Axis-2 effect, not a placement effect. Only
the gen3/epyc `quant` case is genuinely placement-caused: its overlapped curve still
crosses at r*=0.9966, and serial placement raises r=1's predicted speedup to 1.0234,
which removes that crossing and yields `rstar_predicted = None` — an explained
consequence of the placement switch there, not a missing measurement.

## Old → new, per artifact (rstar_predicted)

| artifact | family | stored (internal model) | new `_pyreloc` (serial) | rstar_measured |
|---|---|---|---|---|
| v1_gen4_rstar_bfair | blocked_transpose | 0.180653 | 0.324666 | 0.373802 |
| v1_gen4_rstar_bfair | quant | 0.164007 | None | 0.540927 |
| v2_isa_rstar_avx2 | quant | 0.497078 | None | 0.579018 |
| v2_isa_rstar_avx512 | blocked_transpose | 0.333408 | 0.324666 | 0.395783 |
| v2_isa_rstar_avx512 | quant | 1.0 | None | 0.597010 |
| v2_isa_gen3_rstar_avx2_epyc… | quant | 1.0 | None | 0.635650 |

(Families whose stored and new values are both None are omitted. `speedup_serial` —
the internal model's A-side serial bound — has no pyreloc counterpart and is absent
from the new files; CM1's `Serial` placement serializes the *B* side.)

Also note: the gen4 arms (v1/avx2/avx512) now share one prediction per family — the
maintained model prices the box (its calibration), not the per-artifact roofline
JSONs the internal model consumed. Per-arm measured curves still differ.

## Measured-side caveat (v1_gen4 pair only)

The original `v1_gen4_rstar_bfair.json` was assembled with an out-of-band per-point
stabler-preference merge (`bench/rtrack/README.md:225-228`,
`docs/r2-exp2-gen4-crossover.md:48-58`): six N=16384 best-chunk points with
IQR/median > 5% were flagged and replaced by their stabler rerun rows before the
figure was generated. `figure_rstar.py` never implemented that merge rule itself, so
regenerating `speedup_measured` directly from the committed CSVs (as
`v1_gen4_rstar_bfair_pyreloc.json` does) reproduces the merge at three of the sixteen
measured points only:

| family | r | original `speedup_measured` | new `_pyreloc` `speedup_measured` | delta |
|---|---|---|---|---|
| blocked_transpose | 0.125 | 1.253238 | 1.255733 | 0.00249 |
| quant | 0.125 | 1.248677 | 1.288218 | 0.03954 |
| transpose_quant | 1.0 | 0.137376 | 0.146687 | 0.00931 |

These three deltas are the full extent of the divergence — every other measured point
across all four families is bit-exact between the two files. One `unstable` flag also
flips as a side effect: `quant`'s `unstable` field goes `false` → `true`. Despite the
`speedup_measured` deltas, `rstar_measured` itself is bit-exact in all four families
(`blocked_transpose` 0.3738022702156795, `nchw_nhwc_quant` `None`, `quant`
0.5409268961815304, `transpose_quant` `None`) — the crossing interpolation happens to
land on the same value either side of the three perturbed points. The three v2
artifacts (`v2_isa_rstar_avx2`, `v2_isa_rstar_avx512`,
`v2_isa_gen3_rstar_avx2_epyc7351-2080ti`) were not subject to this merge and are fully
bit-exact, `speedup_measured` included, between original and `_pyreloc`.
