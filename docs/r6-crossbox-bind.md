# R6 — cross-box no-recompile bind demo (issue #87)

**Claim demonstrated**: the same MLIR-folded symbolic relocation plan
(`libreloc/test/corpus/blocked_transpose_sym.bin`, the V3 wire-row plan:
chain → `sym-opt --reloc-fold` → `encodePlan`), symbol-bound at runtime
(`pyreloc.bind(plan, {"N": N}, model=<box calibration>, wire_ratio=r)`),
gets its placement chosen automatically by the calibrated cost model at
bind time — and the choice is correct on both boxes, with zero per-box
recompilation or refitting. Cross-box generalization of V3's wire row
(`docs/v3-costmodel.md` §6: one box, one decision).

Narrowing provenance: issue #107 "Reconciliation with #87" — the
held-out prediction-accuracy half of #87 went to CM4 (#112/#121) and
CM5 (#113/#126); this demo is the remaining half.

## Setup

- Grid: r ∈ {0.25, 0.5, 1.0} (the measured rsweep wire tiers s8/f16/f32)
  × N ∈ {2048, 4096, 8192, 16384} × both boxes = 24 cells. r = 0.125 is
  excluded: neither calibration carries a blocked s4 CPU key, so
  `predict` raises and `bind` leaves the decision unset (opt-in advice,
  never a bind failure) — a CM-track model-coverage boundary, not a
  machinery limit.
- Decision: bind-time, t8 + Overlapped placement by construction
  (`libreloc/src/Bind.cpp` step 8 — the deployment default; the
  library's real Method B is the double-buffered pipeline). A
  serial-priced (`b_fair`) `predict` check rides along per cell.
- Ground truth: stabler-preference-merged BP rsweep medians
  (`bench/results/bp_rsweep{,_rerun}_{box}.csv`; merge rule =
  `cm5_eval.merge_points`, the CM5 rule). Cell pairing follows the
  rsweep design: `a` at tier r vs `b_pipelined` (or `b_fair`) at r=1 —
  B ships f32 regardless of A's wire ratio.
- Artifacts: `bench/results/r6_bind_demo_{epyc7351-2080ti,
  7800x3d-4070tis}.json` — fully deterministic (sorted keys, no
  timestamps/hostnames/revs, inputs sha256-pinned). Each was produced
  on its own box; CI regenerates both in-process and byte-compares
  (`libreloc/python/tests/test_r6_crossbox_bind.py`). Byte-equality is
  the no-recompile portability bar: the artifact's only inputs are the
  committed calibration and blob, so a box cannot have "adapted"
  anything and still match.

## Verdict

- **23/24 bind-time decisions match the measured winner** under the
  overlapped pairing; the serial pairing gives the same 23/24.
- **The flip row, r = 0.25 — all 8 cells correct**: Gen3 chooses `b`
  at every N (its host gather is slow relative to its gen3 link;
  shipping raw f32 wins ~4×), Gen4 chooses `a` at every N (host
  gather+quantize wins 1.32–1.73×). Same plan bytes, same shapes,
  opposite placements, both correct — the boundary law
  (`docs/claim-ledger.md`) deciding placement per host at bind time.
- r = 1.0 row: both boxes choose `b` at every N, matching measurement —
  the machinery is not hardwired to flip (and this is the r=1.0
  pure-relocation loss both boxes' ledger rows record).
- **The 1 miss, disclosed**: (Gen4, r=0.5, N=2048) — measured `a` by
  4.4% (0.669 vs 0.700 ms), model says `b`. Small-N WSL2 cell — the
  caveat class BP's admissibility gate excludes below N=8192; CM5's
  MISCLASS handling is the authority on model quality. Not excluded
  from this table; counted, named, and left standing.

### epyc7351-2080ti

| r | N | a (ms) | b_pipelined (ms) | measured | bind decision | threshold_bytes | verdict |
|---|---|---|---|---|---|---|---|
| 0.25 | 2048 | 5.569 | 1.348 | b | b | -1 | match |
| 0.25 | 4096 | 21.157 | 5.242 | b | b | -1 | match |
| 0.25 | 8192 | 83.074 | 20.693 | b | b | -1 | match |
| 0.25 | 16384 | 329.246 | 82.497 | b | b | -1 | match |
| 0.5 | 2048 | 2.050 | 1.348 | b | b | -1 | match |
| 0.5 | 4096 | 7.556 | 5.242 | b | b | -1 | match |
| 0.5 | 8192 | 44.087 | 20.693 | b | b | -1 | match |
| 0.5 | 16384 | 174.143 | 82.497 | b | b | -1 | match |
| 1.0 | 2048 | 1.726 | 1.348 | b | b | -1 | match |
| 1.0 | 4096 | 5.929 | 5.242 | b | b | -1 | match |
| 1.0 | 8192 | 32.819 | 20.693 | b | b | -1 | match |
| 1.0 | 16384 | 127.499 | 82.497 | b | b | -1 | match |

### 7800x3d-4070tis

| r | N | a (ms) | b_pipelined (ms) | measured | bind decision | threshold_bytes | verdict |
|---|---|---|---|---|---|---|---|
| 0.25 | 2048 | 0.482 | 0.700 | a | a | -1 | match |
| 0.25 | 4096 | 1.384 | 2.397 | a | a | -1 | match |
| 0.25 | 8192 | 6.526 | 9.537 | a | a | -1 | match |
| 0.25 | 16384 | 27.495 | 36.380 | a | a | -1 | match |
| 0.5 | 2048 | 0.669 | 0.700 | a | b | 1.41362e+06 | **MISS** (disclosed) |
| 0.5 | 4096 | 2.577 | 2.397 | b | b | 1.41362e+06 | match |
| 0.5 | 8192 | 10.113 | 9.537 | b | b | 1.41362e+06 | match |
| 0.5 | 16384 | 51.152 | 36.380 | b | b | 1.41362e+06 | match |
| 1.0 | 2048 | 0.788 | 0.700 | b | b | 3.38132e+06 | match |
| 1.0 | 4096 | 3.077 | 2.397 | b | b | 3.38132e+06 | match |
| 1.0 | 8192 | 13.955 | 9.537 | b | b | 3.38132e+06 | match |
| 1.0 | 16384 | 56.986 | 36.380 | b | b | 3.38132e+06 | match |

## Why there is no #73 gate registration for this demo

Nothing here is a new stochastic measurement. The decision tables are
deterministic functions of committed calibrations + committed code; the
measured winners were committed by BP3 (#116/#124) and merged by the
CM5 rule (#113/#126). The acceptance bars (23/24 with the pinned miss;
artifact byte-equality) are enforced continuously in CI, which is
stronger than a one-shot pre-registered run. The expected outcomes were
still written into the spec before the code existed
(`docs/superpowers/specs/2026-08-12-r6-crossbox-bind-demo-design.md`).

## Reproduction

    PYTHONPATH=build/sym/python python3 bench/rtrack/r6_bind_demo.py \
        --machine <epyc7351-2080ti|7800x3d-4070tis>
    PYTHONPATH=build/sym/python python3 -m pytest \
        libreloc/python/tests/test_r6_crossbox_bind.py -v

Gen4 session: see "R6 Gen4 runbook" in `bench/rtrack/README.md`.

## Cross-links

#87 (this demo) · #107 reconciliation (scope split) · CM4/CM5 (held-out
model-quality gates, `bench/results/cm5_eval_report.json`) · V3 §6 wire
row (single-box precedent) · BP3 (`bp_rsweep*` ground truth) ·
`docs/claim-ledger.md` "Machinery" row.
