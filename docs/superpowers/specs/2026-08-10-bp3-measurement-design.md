# BP3 — Measurement: both boxes, the first `bp_*` artifacts

**Issue**: #116 (BP3), parent tracking #108; blocked-by #BP1 ✓ / #BP2 ✓ / #CM4 ✓ (all
merged). **Date**: 2026-08-10.
**Deliverable shape**: **draft PR** on branch `bp3-measurement` (CM1's pattern — Gen3
measured now on this box, Gen4 completed at home, then undraft); ~1 day. This PR creates
the FIRST `bench/results/bp_*` artifacts — the commit-order acceptance of CM4 (#112) and
BP2 (#115) is thereby finalized (both merged before this branch's data commits).

## Context

BP1 shipped `Method::BPipelined` (two-stream, verified); BP2 registered the gates and
per-cell predictions; CM4 registered the model predictions. BP3 produces the dataset both
tracks are judged on (#CM5 formal evaluation; #BP4 restatement). The measurement follows
the pre-declared protocol exactly — the M0/R1 session ritual, fresh per-session
calibration with link-state-under-load, the full matrix + r-sweep with `--method all`,
the V1 stabler-preference rerun rule, and one Nsight capture of a `b_pipelined` T1 run.

Decisions taken with the user: (a) **draft-PR pattern** — Gen3 data + all four configs +
the Gen4 runbook land now; Gen4 completes at home; (b) **Nsight digest-only** — the
`.stats.txt` digest and the capture command are committed/recorded; the `.nsys-rep` stays
local (D3's policy and the standing `.gitignore` rule; R4's force-add is noted as the
exception, not the precedent).

## §1 Configs, build, Gen3 session (this box)

**Configs** — four files under `bench/rtrack/configs/`, modeled on `v1_gen3.json` /
`v1_gen4_rsweep.json`:

- `bp_matrix_nsweep_gen3.json`: machine `epyc7351-2080ti`, `transforms: "matrix"`,
  `methods: "all"`, `n: [2048, 4096, 8192, 16384]`, `chunk_mib: [4, 16, 64, 256]`,
  `threads: [8]`, warmup 5 / iters 30, `bin: "./bench-rtrack"`,
  `out_csv: "bench/results/bp_matrix_nsweep_epyc7351-2080ti.csv"`.
- `bp_rsweep_gen3.json`: same but `transforms: "rsweep"`,
  `out_csv: ".../bp_rsweep_epyc7351-2080ti.csv"`.
- `bp_matrix_nsweep_gen4.json` / `bp_rsweep_gen4.json`: machine `7800x3d-4070tis`,
  otherwise identical, out_csv `.../bp_{matrix_nsweep|rsweep}_7800x3d-4070tis.csv`.

Notes pinned by exploration: `--transform all` ≡ matrix in the current harness, so matrix
and rsweep are separate configs (rsweep is REQUIRED — BP-G1's kernel-bearing r=1.0 cells
exist only there); the `machine` value must be exactly the gate-constant keys
(`BP_G1_STRICT_MIN_N` / `BP_G3_DIVISOR`); the filename slug equals the machine tag
verbatim (a deliberate, self-consistent convention for the bp\_ series).

**Build**: the repo-root `./bench-rtrack` is stale (predates BPipelined) — rebuild with
the README's standalone recipe (`bench/rtrack/README.md:140-159`; per-file ISA `.o`s for
QuantAVX2/512, `nvcc -ccbin g++ -arch=sm_75`, CUDA 12.5) and place it at the repo root so
`config.bin` records `./bench-rtrack`. Sanity: `--method bpipe` accepted; a small-N smoke
row carries `h2d_occupancy` (proves the BP1-era binary).

**Session ritual** (pre-declared; record as-executed state in the PR's Session section):
governor `performance` (already set — verify), persistence mode on (verify), process
pinned to GPU0's affinity cores `taskset -c 4-7,20-23` (never run unpinned — M0's
bimodality), THP recorded by calibrate (not set). Fresh calibration:
`python3 bench/rtrack/calibrate.py --out calibration.json --load-bin ./bench-rtrack`
(samples the PCIe link under load); the session file is committed as
`bench/results/bp_calibration_epyc7351-2080ti.json`.

**Runs**: `taskset -c 4-7,20-23 python3 bench/rtrack/run_rtrack.py --config
bench/rtrack/configs/bp_matrix_nsweep_gen3.json`, then the rsweep config. Every emitted
row must print `[verified]` (bit-exact gate; zero failures aborts otherwise); the T2
`b_pipelined N/A` skip lines are expected and counted.

**Unstable handling** (the pre-declared V1 rule, executed as V1 did): after both runs,
enumerate best-chunk rows with `unstable=1` (IQR/median > 5%); re-measure exactly those
(transform, N, method) points by direct targeted `bench-rtrack` invocations; commit the
rerun rows as `bench/results/bp_{matrix_nsweep|rsweep}_rerun_epyc7351-2080ti.csv`
(filename labelling, V1 convention — rerun files carry no session header); list the
flagged points verbatim in the PR. The stabler-preference merge itself happens at
analysis time (CM5/BP4), never by editing the original CSVs.

**Nsight capture** (one, `b_pipelined` T1): scratchpad `.nsys-rep` via
`/usr/local/cuda-12.5/bin/nsys profile -t cuda,osrt --force-overwrite true -o <scratch>
./bench-rtrack --transform T1 --method bpipe --n 8192 --chunk-mib 16 --warmup 2 --iters 8
--no-verify --csv /dev/null` (D3's command shape), then
`nsys stats --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum` → committed digest
`bench/results/bp_bpipelined_trace_2080ti.stats.txt` with a NOTICE line recording the
command + nsys version. The `.nsys-rep` is NOT committed. The digest should visibly show
copy/kernel overlap consistent with the occupancy column (kernels a small fraction of
span, memcpy dominating).

## §2 Verification, Gen4 runbook, deliverable

**Verification**:
- Acceptance line 1: every config bit-exact — all rows `[verified]`, zero VERIFY FAILED
  (quote counts in the PR, PR #104's format).
- CSV `machine` column equals the gate-constant key exactly (scripted check).
- `h2d_occupancy` ∈ (0, 1.05] on all rows; `b_pipelined` occupancy > `b_fair` on
  kernel-bearing matrix families at n_chunks > 1 (the BP1 semantic, now on real
  full-protocol data).
- **Informational gate run** (not the formal evaluation — that is CM5's): `gates.py
  --exp bp --csv <the bp CSVs + reruns>` output quoted verbatim in the PR. Expected
  per the pre-registrations: BP-G1/G2 plausibly PASS; BP-G3 Gen3 cells near the fair
  ÷1.06 predictions; the #108 "Gen3 ≥1.5×" headline MISS appearing exactly as the #108
  correction comment predicted. Data is committed as measured regardless of verdicts.
- Suites + `v3_gate.py` REGEN checks stay green (nothing in this PR touches code,
  calibrations, or frozen artifacts — data + configs + one digest only).

**Gen4 runbook** (PR body + a short section appended to `bench/rtrack/README.md`'s
session-ritual area): WSL2 caveats recorded not controlled (governor unreadable, no
`-lgc` without the Windows-side lock — use the CM1 lesson: Windows `nvidia-smi -lgc 2610
-lmc 10251` before the session); build with `-arch=sm_89`; run the two gen4 configs;
apply the same unstable→targeted-rerun rule (`_rerun_` filenames); fresh calibration →
`bp_calibration_7800x3d-4070tis.json`; commit and undraft. The stabler-preference rule
citation (`docs/r2-exp2-gen4-crossover.md:48-58`) rides in the runbook.

**Deliverable**: branch `bp3-measurement`, draft PR containing — 4 configs, 2 Gen3 CSVs
(+ any `_rerun_` files), `bp_calibration_epyc7351-2080ti.json`, the nsys digest, the
README runbook addition, and the Session section in the PR body (box, ritual as-executed,
verified counts, T2 N/A counts, flagged/rerun list, gate first-look, nsys command +
version). No code changes. Gen4 artifacts arrive before undraft.

**Acceptance mapping** (#116): Gen3 bullet = §1 (ritual + calibration + both sweeps +
nsys); Gen4 bullet = the runbook + home-box completion pre-undraft; data + configs under
`bp_*` = the artifact list; "every config bit-exact" = the verified counts; "both
calibrations committed" = the two `bp_calibration_*.json`; "unstable rows handled per the
pre-declared rule with reruns labelled" = the `_rerun_` files + PR list.
