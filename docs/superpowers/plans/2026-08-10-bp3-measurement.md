# BP3 — Both-Box Measurement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement issue #116 (BP3): the Gen3 measurement session on this box (ritual + calibration + matrix & r-sweep `--method all` + unstable reruns + one b_pipelined nsys capture), all four bp configs, the Gen4 runbook, and the draft PR carrying the first `bench/results/bp_*` artifacts.

**Architecture:** The spec is `docs/superpowers/specs/2026-08-10-bp3-measurement-design.md` — read it first. Pure measurement: no code changes anywhere. Task 1 builds the binary and commits configs; Task 2 runs the Gen3 session; Task 3 handles unstable reruns + the Nsight digest; Task 4 verifies, adds the Gen4 runbook, and opens the draft PR.

**Tech Stack:** standalone nvcc build (sm_75, CUDA 12.5), `run_rtrack.py` + config JSON, `calibrate.py`, `nsys` 2024.2.3 at `/usr/local/cuda-12.5/bin/nsys`.

## Global Constraints

- **No code changes** — this branch commits only: 4 config JSONs, 2 (+rerun) Gen3 CSVs, 1 session calibration JSON, 1 nsys digest, 1 README runbook section, the spec/plan docs. Frozen artifacts and calibrations untouched.
- The CSV `machine` column must be EXACTLY `epyc7351-2080ti` (Gen3) / `7800x3d-4070tis` (Gen4) — the BP2 gate constants key on these strings.
- **Long-running commands**: the Bash tool's hard cap is 600s; each measurement run can take 30–90+ minutes. Launch them with `run_in_background: true` and wait for the completion notification (never poll with foreground sleeps; check interim progress by reading the output file / `wc -l` on the CSV in separate quick calls).
- Every measurement runs pinned: `taskset -c 4-7,20-23` (GPU0's affinity cores — unpinned runs are bimodal on this box; M0 rule). Verify persistence mode + performance governor before measuring; record, never assume.
- Reruns are targeted direct `bench-rtrack` invocations committed as `_rerun_` files (bare column header, no session comments — V1 convention); originals are NEVER edited; the stabler-preference merge is analysis-time (CM5/BP4).
- Nsight: digest-only policy — commit `bench/results/bp_bpipelined_trace_2080ti.stats.txt`; the `.nsys-rep` stays in the scratchpad (gitignored anyway).
- Branch: `bp3-measurement` (exists, spec committed). Deliverable: **draft PR**.
- Commit messages end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: Build the BP1-era binary + commit the four configs

**Files:**
- Create: `bench/rtrack/configs/bp_matrix_nsweep_gen3.json`, `bench/rtrack/configs/bp_rsweep_gen3.json`, `bench/rtrack/configs/bp_matrix_nsweep_gen4.json`, `bench/rtrack/configs/bp_rsweep_gen4.json`
- Create (untracked, repo root): `./bench-rtrack` (rebuilt; gitignored)

**Interfaces:**
- Produces: the four configs Task 2/4 use; a repo-root `./bench-rtrack` binary that accepts `--method bpipe` and emits `h2d_occupancy`.

- [ ] **Step 1: Rebuild** (the stale root binary predates `Method::BPipelined` — never use it):

```bash
SCRATCH=build/cm1-tools && mkdir -p "$SCRATCH"
g++ -O3 -DNDEBUG -std=c++17 -fPIC -DRELOC_QUANT_HAVE_X86_SIMD=1 \
  -Ilibreloc/include -c libreloc/quant/QuantAVX2.cpp \
  -mavx2 -mfma -mf16c -o "$SCRATCH/QuantAVX2.o"
g++ -O3 -DNDEBUG -std=c++17 -fPIC -DRELOC_QUANT_HAVE_X86_SIMD=1 \
  -Ilibreloc/include -c libreloc/quant/QuantAVX512.cpp \
  -mavx512f -mavx512bw -o "$SCRATCH/QuantAVX512.o"
/usr/local/cuda-12.5/bin/nvcc -ccbin g++ -O3 -DNDEBUG -std=c++17 -arch=sm_75 \
  -DRELOC_ENABLE_CUDA=1 -DRELOC_QUANT_HAVE_X86_SIMD=1 \
  -Ilibreloc/include -Ibench \
  bench/rtrack/rtrack_bench.cu libreloc/src/*.cpp libreloc/quant/Quant.cpp \
  "$SCRATCH/QuantAVX2.o" "$SCRATCH/QuantAVX512.o" \
  libreloc/cuda/CudaBackend.cu libreloc/cuda/CudaKernels.cu \
  -o bench-rtrack -Xcompiler -pthread
```

- [ ] **Step 2: BP1-era smoke** (proves bpipe + occupancy):

```bash
taskset -c 4-7,20-23 ./bench-rtrack --transform T3 --method bpipe --n 2048 \
  --chunk-mib 4 --warmup 1 --iters 3 --machine smoke --csv-header --csv - | head -3
```

Expected: a header line ending `,h2d_occupancy` and one `b_pipelined` data row marked from a `[verified]` run (stderr). If `--csv-header`/flag spellings differ, check the usage string first.

- [ ] **Step 3: Write the four configs.** `bp_matrix_nsweep_gen3.json`:

```json
{
  "machine": "epyc7351-2080ti",
  "bin": "./bench-rtrack",
  "numactl": "",
  "transforms": "matrix",
  "methods": "all",
  "n": [2048, 4096, 8192, 16384],
  "chunk_mib": [4, 16, 64, 256],
  "threads": [8],
  "warmup": 5,
  "iters": 30,
  "calibration": "calibration.json",
  "out_csv": "bench/results/bp_matrix_nsweep_epyc7351-2080ti.csv"
}
```

`bp_rsweep_gen3.json`: identical except `"transforms": "rsweep"` and
`"out_csv": "bench/results/bp_rsweep_epyc7351-2080ti.csv"`.
`bp_matrix_nsweep_gen4.json` / `bp_rsweep_gen4.json`: identical shapes with
`"machine": "7800x3d-4070tis"` and out_csv `bench/results/bp_matrix_nsweep_7800x3d-4070tis.csv` / `bench/results/bp_rsweep_7800x3d-4070tis.csv`.

- [ ] **Step 4: Commit (configs only — no data yet)**

```bash
git add bench/rtrack/configs/bp_*.json
git commit -m "bench(rtrack): BP3 measurement configs, both boxes (#116)"
```

---

### Task 2: Gen3 session — ritual, calibration, both sweeps

**Files:**
- Create: `bench/results/bp_matrix_nsweep_epyc7351-2080ti.csv`, `bench/results/bp_rsweep_epyc7351-2080ti.csv`, `bench/results/bp_calibration_epyc7351-2080ti.json`

**Interfaces:**
- Consumes: Task 1's binary + gen3 configs.
- Produces: the two Gen3 CSVs + session calibration Task 3/4 analyze.

- [ ] **Step 1: Ritual check (record, don't assume):**

```bash
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor        # expect performance
nvidia-smi --query-gpu=persistence_mode --format=csv,noheader | head -1  # expect Enabled
nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l        # expect 0 (idle)
cat /sys/kernel/mm/transparent_hugepage/enabled                   # record (madvise)
```

If governor is not `performance` or persistence is off, fix (`sudo cpupower frequency-set -g performance`, `sudo nvidia-smi -pm 1`) or STOP and report if no permission. Record all outputs in your report.

- [ ] **Step 2: Fresh session calibration (link state under load):**

```bash
rm -f calibration.json
taskset -c 4-7,20-23 python3 bench/rtrack/calibrate.py --out calibration.json \
  --load-bin ./bench-rtrack
python3 -c "import json; d=json.load(open('calibration.json')); print(d['pcie_under_load'], d['triad_gbps'], d['gpus'][0]['name'])"
```

Expected: `pcie_under_load` gen "3" width "16"; triad ≈ 26 under the pinned-session
protocol (v1: 26.07, v2_gen3: 26.14; the ≈44 figure was the R1-era pre-pinning value);
2080 Ti. Copy it now:
`cp calibration.json bench/results/bp_calibration_epyc7351-2080ti.json`.

- [ ] **Step 3: Matrix run** — LAUNCH IN BACKGROUND (this exceeds the 600s Bash cap):

```bash
# Bash tool call with run_in_background: true
taskset -c 4-7,20-23 python3 bench/rtrack/run_rtrack.py \
  --config bench/rtrack/configs/bp_matrix_nsweep_gen3.json \
  > /tmp/bp3_matrix_gen3.log 2>&1
```

While waiting (quick foreground checks only): `wc -l bench/results/bp_matrix_nsweep_epyc7351-2080ti.csv` and `tail -2 /tmp/bp3_matrix_gen3.log`. On completion: `grep -c "VERIFY FAILED" /tmp/bp3_matrix_gen3.log` (must be 0), `grep -c "\[verified\]" /tmp/bp3_matrix_gen3.log` (record), `grep -c "b_pipelined N/A" /tmp/bp3_matrix_gen3.log` (record — T2 skips).

- [ ] **Step 4: R-sweep run** — same background pattern with `bp_rsweep_gen3.json`, log `/tmp/bp3_rsweep_gen3.log`, same post-checks.

- [ ] **Step 5: Commit the session data:**

```bash
git add bench/results/bp_matrix_nsweep_epyc7351-2080ti.csv \
        bench/results/bp_rsweep_epyc7351-2080ti.csv \
        bench/results/bp_calibration_epyc7351-2080ti.json
git commit -m "bench(results): BP3 Gen3 session -- matrix + r-sweep, --method all (#116)"
```

---

### Task 3: Unstable scan → targeted reruns; Nsight capture → digest

**Files:**
- Create (if any flagged): `bench/results/bp_matrix_nsweep_rerun_epyc7351-2080ti.csv` and/or `bench/results/bp_rsweep_rerun_epyc7351-2080ti.csv`
- Create: `bench/results/bp_bpipelined_trace_2080ti.stats.txt`

**Interfaces:**
- Consumes: Task 2's CSVs, Task 1's binary.
- Produces: the rerun files (V1 labelling) + the nsys digest.

- [ ] **Step 1: Unstable scan** — best-chunk rows (min `median_ms` per (transform, N, method, variant)) with `unstable == 1`:

```bash
python3 - <<'EOF'
import csv, collections
flagged = []
for path in ("bench/results/bp_matrix_nsweep_epyc7351-2080ti.csv",
             "bench/results/bp_rsweep_epyc7351-2080ti.csv"):
    rows = [r for r in csv.DictReader(l for l in open(path) if not l.startswith("#"))]
    best = {}
    for r in rows:
        key = (r["transform"], r["N"], r["method"], r["variant"], r["r"])
        if key not in best or float(r["median_ms"]) < float(best[key]["median_ms"]):
            best[key] = r
    for key, r in sorted(best.items()):
        if r["unstable"] == "1":
            flagged.append((path.split("/")[-1],) + key +
                           (r["chunk_req_mib"], r["iqr_over_median_pct"]))
for f in flagged:
    print(f)
print(f"total flagged best-chunk points: {len(flagged)}")
EOF
```

Record the list verbatim (it goes in the PR). If empty: skip Step 2, note "0 flagged" — a legitimate outcome.

- [ ] **Step 2: Targeted reruns** (only flagged points; workload-id mapping — matrix: transpose=T1, blocked_transpose=T1b, transpose_quant=T2, quant=T3, nchw_nhwc_quant=T4, convert_f16=T5; rsweep: `<matrix-id>R100/R050/R025/R0125` by r=1.0/0.5/0.25/0.125, e.g. blocked_transpose r=0.5 → T1bR050). One invocation per flagged (id, N), full chunk sweep, ALL methods at that point re-measured together (the file is self-consistent per point); first invocation carries `--csv-header`, later ones append:

```bash
# example for a flagged matrix point (T1b, N=16384); repeat per point, >> after the first
taskset -c 4-7,20-23 ./bench-rtrack --transform T1b --n 16384 --method all \
  --chunk-mib 4,16,64,256 --threads 8 --warmup 5 --iters 30 \
  --machine epyc7351-2080ti --csv-header \
  --csv bench/results/bp_matrix_nsweep_rerun_epyc7351-2080ti.csv
# --threads 8 is MANDATORY: it must match the originals' threads column (the
# binary defaults to 1; a threads=1 rerun measures a different machine state —
# method a ~5x slower single-threaded — and cannot be merged with the originals).
```

(Use `run_in_background` if a point exceeds ~5 minutes.) Confirm rerun rows print `[verified]`; confirm the rerun file starts with the bare column header (no `#` lines — matching V1's rerun convention). Commit whatever rerun files exist:

```bash
git add bench/results/bp_*_rerun_epyc7351-2080ti.csv 2>/dev/null
git commit -m "bench(results): BP3 Gen3 targeted reruns -- flagged unstable best-chunk points (#116)" || echo "no reruns needed"
```

- [ ] **Step 3: Nsight capture** (one b_pipelined T1 run; digest-only policy):

```bash
NSYS=/usr/local/cuda-12.5/bin/nsys
SCRATCHDIR=/tmp/claude-2017/-home-jueonpark-sym/e33a9c45-e759-40e3-b905-3b7cf69f69c6/scratchpad
mkdir -p "$SCRATCHDIR"
taskset -c 4-7,20-23 "$NSYS" profile -t cuda,osrt --force-overwrite true \
  -o "$SCRATCHDIR/bp_bpipelined_trace" \
  ./bench-rtrack --transform T1 --method bpipe --n 8192 --chunk-mib 16 \
  --warmup 2 --iters 8 --no-verify --csv /dev/null
"$NSYS" stats --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum \
  "$SCRATCHDIR/bp_bpipelined_trace.nsys-rep" > /tmp/bp3_nsys_stats.txt
"$NSYS" --version
```

Build the committed digest with a NOTICE header (R4's shape):

```bash
{ echo "# NOTICE: digest of a b_pipelined T1 N=8192 chunk=16MiB trace (issue #116)."
  echo "# capture: nsys profile -t cuda,osrt -o <scratchpad>/bp_bpipelined_trace \\"
  echo "#   ./bench-rtrack --transform T1 --method bpipe --n 8192 --chunk-mib 16 --warmup 2 --iters 8 --no-verify --csv /dev/null"
  echo "# nsys version: $(/usr/local/cuda-12.5/bin/nsys --version | head -1)"
  echo "# policy: .nsys-rep not committed (digest-only, D3 precedent; R4's force-add was the exception)."
  cat /tmp/bp3_nsys_stats.txt
} > bench/results/bp_bpipelined_trace_2080ti.stats.txt
```

Sanity: the kern_sum/mem_time_sum tables must show memcpy time dominating kernel time (overlap consistent with h2d_occupancy ≈ 0.99 on this row class). Commit:

```bash
git add bench/results/bp_bpipelined_trace_2080ti.stats.txt
git commit -m "bench(results): BP3 nsys digest -- b_pipelined T1 overlap trace (#116)"
```

---

### Task 4: Verification, Gen4 runbook, draft PR

**Files:**
- Modify: `bench/rtrack/README.md` (Gen4 BP session runbook subsection near the session-ritual block)

- [ ] **Step 1: Data verification suite:**

```bash
python3 - <<'EOF'
import csv
ok = True
for path in ("bench/results/bp_matrix_nsweep_epyc7351-2080ti.csv",
             "bench/results/bp_rsweep_epyc7351-2080ti.csv"):
    rows = [r for r in csv.DictReader(l for l in open(path) if not l.startswith("#"))]
    assert rows, path
    machines = {r["machine"] for r in rows}
    assert machines == {"epyc7351-2080ti"}, (path, machines)  # gate-constant key, exact
    for r in rows:
        occ = float(r["h2d_occupancy"])
        assert 0.0 < occ <= 1.05, (path, r["transform"], r["N"], r["method"], occ)
        assert r["verified"] == "1", (path, r["transform"], r["N"], r["method"])
    print(path, len(rows), "rows OK")
# BP1 semantic on real full-protocol data: bpipe occupancy > b_fair on
# kernel-bearing matrix families at n_chunks > 1 (best-eff rows).
rows = [r for r in csv.DictReader(l for l in open(
    "bench/results/bp_matrix_nsweep_epyc7351-2080ti.csv") if not l.startswith("#"))]
best = {}
for r in rows:
    if int(r["n_chunks"]) <= 1 or float(r["gpu_kernel_ms"]) <= 0:
        continue
    key = (r["transform"], r["N"], r["method"])
    if key not in best or float(r["effective_input_GBps"]) > float(best[key]["effective_input_GBps"]):
        best[key] = r
pairs = 0
for (t, n, m), r in sorted(best.items()):
    if m != "b_pipelined":
        continue
    bf = best.get((t, n, "b_fair"))
    if bf is None:
        continue
    pairs += 1
    bp_occ, bf_occ = float(r["h2d_occupancy"]), float(bf["h2d_occupancy"])
    print(f"{t} N={n}: bpipe occ {bp_occ:.3f} vs b_fair {bf_occ:.3f}"
          f"{'' if bp_occ >= bf_occ else '  <-- INVESTIGATE'}")
print(f"{pairs} pairs compared")
EOF
```

Expected: both files' rows OK; every bpipe/b_fair pair shows bpipe ≥ b_fair (a violation is an INVESTIGATE, not an auto-fail — report it honestly if it appears).

- [ ] **Step 2: Informational gate run** (formal evaluation is CM5's; data stands as measured regardless):

```bash
python3 bench/rtrack/gates.py --exp bp \
  --csv bench/results/bp_matrix_nsweep_epyc7351-2080ti.csv \
        bench/results/bp_rsweep_epyc7351-2080ti.csv \
        $(ls bench/results/bp_*_rerun_epyc7351-2080ti.csv 2>/dev/null) \
  > /tmp/bp3_gates_gen3.txt 2>&1
cat /tmp/bp3_gates_gen3.txt
```

Record verbatim. Expected shape (not required): BP-G1/BP-G2 PASS; BP-G3 Gen3 cells near the fair ÷1.06 predictions; the #108-predicted "Gen3 ≥1.5×" headline MISS visible in the quant ratios (~1.35–1.45). Any surprise (e.g. a BP-G1 FAIL) is investigated and reported, never hidden.

- [ ] **Step 3: README Gen4 runbook** — append a short subsection after the session-ritual block:

```markdown
### BP3 Gen4 session runbook (issue #116; run at the home box, then undraft PR)

WSL2 caveats are recorded, not controlled (cpufreq unreadable; `nvidia-smi -lgc`
refused in WSL) — lock clocks from the WINDOWS side first: `nvidia-smi -lgc 2610
-lmc 10251` (the CM1 session's lesson; unlocked WDDM P-states corrupt medians).
1. Build `bench-rtrack` with the standalone recipe above, `-arch=sm_89`.
2. Fresh calibration: `python3 bench/rtrack/calibrate.py --out calibration.json
   --load-bin ./bench-rtrack`; commit as
   `bench/results/bp_calibration_7800x3d-4070tis.json`.
3. Run `bench/rtrack/configs/bp_matrix_nsweep_gen4.json` then
   `bp_rsweep_gen4.json` via `run_rtrack.py` — every row `[verified]`.
4. Unstable best-chunk points (IQR/median > 5%): targeted `bench-rtrack` reruns
   into `bench/results/bp_{matrix_nsweep|rsweep}_rerun_7800x3d-4070tis.csv`
   (bare header, no session comments) — the pre-declared stabler-preference rule
   (docs/r2-exp2-gen4-crossover.md:48-58) applies at analysis time.
5. Commit CSVs + calibration, list flagged points in the PR, undraft.
```

- [ ] **Step 4: Suites + hygiene**

```bash
build/sym/libreloc/test/libreloc-test 2>&1 | tail -3
PYTHONPATH=build/sym/python python3 -m pytest libreloc/python/tests/ -q
PYTHONPATH=build/sym/python python3 bench/rtrack/v3_gate.py --report bench/results/v3_prediction_report.json 2>&1 | grep -E 'REGEN.*(PASS|FAIL)'
git status --short   # only expected additions
```

- [ ] **Step 5: Commit + push + DRAFT PR**

```bash
git add bench/rtrack/README.md
git commit -m "docs(bench): BP3 Gen4 session runbook (#116)"
git push -u origin bp3-measurement
gh pr create --draft --title "bench(results): BP3 — both-box BP measurement, Gen3 session + Gen4 runbook (#116)" --body "<body>"
```

PR body, in order — the **Session section** (PR #104's format) leads: (1) box + ritual as-executed (governor/persistence/pinning/THP values recorded in Task 2), fresh calibration summary (pcie_under_load, triad), row counts + `[verified]` counts + T2 N/A counts, flagged/rerun point list (or "0 flagged"); (2) the informational `--exp bp` output verbatim, with one sentence: formal evaluation is #CM5's — data committed as measured; the #108-predicted Gen3 headline MISS called out if present; (3) the nsys digest summary (command + version + the memcpy-vs-kernel numbers); (4) why draft: Gen4 pending at the home box per the README runbook (link it); (5) note this PR creates the first `bp_*` artifacts — CM4's and BP2's commit-order acceptance is hereby finalized (both merged prior); (6) `Refs #116, #108, #115, #112`; (7) the standard generated-with footer.

---

## Verification (end-to-end, after all tasks)

1. Both Gen3 CSVs: all rows `verified=1`, machine column exact, occupancy in range, bpipe ≥ b_fair occupancy pairs (or investigated).
2. Zero VERIFY FAILED across both run logs; T2 N/A counts recorded.
3. Flagged-point list + `_rerun_` files consistent (every flagged point has rerun rows; no rerun without a flag).
4. Informational gate output recorded; suites + REGEN checks green; no code/frozen-artifact diffs (`git diff main --stat` shows only the expected additions + README + docs).
5. Draft PR open with the Session section; Gen4 runbook in README; #116's Gen3 bullet fully discharged, Gen4 bullet pending-by-design.
