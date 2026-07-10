# D3 Re-measurement Implementation Plan (issue #67)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-measure the C7 PoC (32768² fp32 transpose) under D1+D2+Release build against issue #67's three pre-registered acceptance criteria, and land superseding v2 result docs.

**Architecture:** This is a measurement campaign, not a code change: a fresh Release build tree, four `bench-poc-transpose` runs (headline + size sweep), one Nsight capture with a small committed analysis script, one `bench-gather-bw` run, and two superseding markdown docs. The only new code is the trace-analysis script.

**Tech Stack:** CMake/Ninja, CUDA 13.2 `nsys`, Python 3 (csv/statistics stdlib only).

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-11-d3-remeasurement-design.md`. Read it before starting.
- Base: branch `d3-remeasurement` off main `28a0743` (contains D1 #68 + D2 #69).
- **Pre-registered, binding (issue #67):** no tuning — no chunk-size, buffer-count, stream-count, iteration-count, or method changes in response to any measured number. Criterion 3 (≥1×) is reported as-is either way.
- **Never edit v1 artifacts:** `docs/poc-reproduction.md`, `bench/results/poc_overlap_summary.md`, `bench/results/poc_transpose_n32768.json`, `bench/results/gather_bw_n4096.json`, and the old `build/cuda` tree all stay byte-identical.
- All measurements from `build/cuda-release` (CMAKE_BUILD_TYPE=Release). The v1 numbers were `-O0`; every v1-vs-v2 comparison in the docs must say so.
- Frozen bench protocol: defaults from `bench/protocol.h` (10 warmup + 50 timed × 3 re-runs, median + IQR, < 5% rerun-spread stability bar), `--verify` on every `bench-poc-transpose` run.
- Machine: WSL2, RTX 4070 Ti SUPER, CUDA 13.2 — recorded as the standing caveat; the v2 doc carries a marked empty bare-metal slot.
- Binary Nsight artifacts (`.nsys-rep`, `.sqlite`, extracted CSV) are NOT committed.
- **Failure gates:** `--verify` failure at any size → stop, report as a correctness regression. Criterion 1 failure (trace still gather-bound) → stop after Task 4, report trace evidence, quote no headline. Rerun spread ≥ 5% → re-run once; if still unstable, report the instability instead of the number.

---

### Task 1: Release build + preflight

**Files:**
- Create: `build/cuda-release/` (build tree, not committed)

**Interfaces:**
- Produces: `build/cuda-release/bench/bench-poc-transpose`, `build/cuda-release/bench/bench-gather-bw`, `build/cuda-release/libreloc/test/libreloc-test` (exact test-binary path may differ; find with `find build/cuda-release -name 'libreloc-test' -type f`). All later tasks run these binaries.

- [ ] **Step 1: Configure the Release tree**

```bash
cd /home/jueonpark/sym
cmake -G Ninja -S . -B build/cuda-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR=/home/jueonpark/sym/build/llvm-project/build/lib/cmake/mlir \
  -DRELOC_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89
```

Expected: configure completes; summary shows CUDA enabled. If MLIR_DIR is rejected, compare against `build/cuda/CMakeCache.txt` (LLVM_DIR=`/home/jueonpark/sym/build/llvm-project/build/lib/cmake/llvm`) and pass that too.

- [ ] **Step 2: Verify Release flags are live**

```bash
grep -A6 "Execute.cpp.o: CXX_COMPILER" build/cuda-release/build.ninja | grep -o '\-O3'
```

Expected: `-O3`. If absent, STOP — the whole point of this tree is optimized host code; debug the configure before proceeding.

- [ ] **Step 3: Build the three targets**

```bash
ninja -C build/cuda-release bench-poc-transpose bench-gather-bw libreloc-test
```

Expected: builds cleanly (warnings OK, errors not).

- [ ] **Step 4: Correctness gate under Release**

```bash
find build/cuda-release -name 'libreloc-test' -type f -exec {} \;
```

Expected: all tests pass (98/98 as of D2's merge; exact count may grow). A Release-only failure (e.g. a race the -O0 build hid) is a STOP: report it, do not measure.

- [ ] **Step 5: Preflight environment**

```bash
nvidia-smi --query-compute-apps=pid,name --format=csv
free -g
```

Expected: no other CUDA compute processes; `available` ≥ 14 GiB. If not, resolve before measuring (close consumers, or report blockage).

No commit — this task produces only uncommitted build artifacts.

---

### Task 2: Headline run, N = 32768 (criterion 3)

**Files:**
- Create: `bench/results/poc_transpose_n32768_v2.json`

**Interfaces:**
- Consumes: `build/cuda-release/bench/bench-poc-transpose` (Task 1).
- Produces: the headline JSON. Task 6 reads its `methods.ours_pipeline_2buf_2stream.wall_ms.median`, `methods.baseline_pinned_memcpy_naive_kernel.wall_ms.median`, `speedup_wall_median`, `config.chunk_bytes`, and rerun spreads.

- [ ] **Step 1: Run the frozen protocol with verification**

```bash
build/cuda-release/bench/bench-poc-transpose --n 32768 --verify \
  --json bench/results/poc_transpose_n32768_v2.json
```

Defaults apply: 10 warmup + 50 timed × 3 re-runs. Expected runtime: minutes (v1 took ~50 s of pure timed loop per rerun pair at -O0; Release is faster). Expected output: verification PASS for both methods, then timing summary.

- [ ] **Step 2: Check the three gates in the JSON**

```bash
python3 - <<'EOF'
import json
d = json.load(open('bench/results/poc_transpose_n32768_v2.json'))
print(json.dumps(d, indent=2)[:2000])
EOF
```

Confirm by eye: (a) `chunk_bytes` = 67108864 (64 MiB — D2 live; if 268435456 the old heuristic is somehow in play: STOP), (b) both methods' `median_spread_pct` < 5, (c) `speedup_wall_median` present. Record the number; do NOT react to it (no-tuning rule).

- [ ] **Step 3: Commit**

```bash
git add bench/results/poc_transpose_n32768_v2.json
git commit -m "bench: D3 headline re-measurement at N=32768, Release build (#67)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Size sweep, N ∈ {4096, 8192, 16384}

**Files:**
- Create: `bench/results/poc_transpose_n4096_v2.json`, `bench/results/poc_transpose_n8192_v2.json`, `bench/results/poc_transpose_n16384_v2.json`

**Interfaces:**
- Consumes: `build/cuda-release/bench/bench-poc-transpose` (Task 1).
- Produces: three JSONs. Task 6 builds its size-scaling table from each file's two `wall_ms.median` values and `speedup_wall_median`.

- [ ] **Step 1: Run all three sizes, same protocol**

```bash
for n in 4096 8192 16384; do
  build/cuda-release/bench/bench-poc-transpose --n $n --verify \
    --json bench/results/poc_transpose_n${n}_v2.json
done
```

Expected: verification PASS at every size; three JSONs written.

- [ ] **Step 2: Spot-check spreads and chunking**

```bash
python3 - <<'EOF'
import json
for n in (4096, 8192, 16384):
    d = json.load(open(f'bench/results/poc_transpose_n{n}_v2.json'))
    print(n, json.dumps(d.get('config', d), indent=0)[:300])
EOF
```

Confirm each size's `chunk_bytes` matches the D2 heuristic — clamp(totalBytes/16, 4 MiB, 64 MiB) with 2 buffers: N=4096 (64 MiB total) → 4 MiB floor; N=8192 (256 MiB) → 16 MiB; N=16384 (1 GiB) → 64 MiB — and spreads < 5%. Sweep results are supporting data, not a tuning input.

- [ ] **Step 3: Commit**

```bash
git add bench/results/poc_transpose_n4096_v2.json bench/results/poc_transpose_n8192_v2.json bench/results/poc_transpose_n16384_v2.json
git commit -m "bench: D3 size sweep N=4096/8192/16384, Release build (#67)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Nsight capture + trace analysis at N = 16384 (criterion 1)

**Files:**
- Create: `bench/analyze_overlap.py` (committed — v1's analysis was ad-hoc; this makes v2 reproducible)
- Create: `bench/results/poc_overlap_summary_v2.md` (committed)
- Create: `bench/results/poc_overlap_v2.nsys-rep`, `.sqlite`, `_cuda_gpu_trace.csv` (NOT committed)

**Interfaces:**
- Consumes: `build/cuda-release/bench/bench-poc-transpose` (Task 1).
- Produces: `bench/analyze_overlap.py` (CLI: `python3 bench/analyze_overlap.py <trace.csv> --warmup-iters 2 --timed-iters 5`) printing utilization/period/duration stats; `poc_overlap_summary_v2.md` with the criterion-1 verdict Task 6 quotes.

- [ ] **Step 1: Capture**

```bash
/usr/local/cuda/bin/nsys profile -t cuda,osrt --force-overwrite true \
  -o bench/results/poc_overlap_v2 \
  build/cuda-release/bench/bench-poc-transpose --n 16384 --warmup 2 --iters 5 --reruns 1 --json -
```

Expected: bench JSON on stdout (save the `speedup_wall_median` line for the summary doc) and `bench/results/poc_overlap_v2.nsys-rep` written. nsys attached cleanly under WSL2 for C7; if it fails here, record the failure mode and retry once before reporting blockage.

- [ ] **Step 2: Extract the GPU trace CSV**

```bash
/usr/local/cuda/bin/nsys stats --report cuda_gpu_trace --format csv \
  --output bench/results/poc_overlap_v2 bench/results/poc_overlap_v2.nsys-rep
head -3 bench/results/poc_overlap_v2_cuda_gpu_trace.csv
```

Expected: header contains `Start (ns)`, `Duration (ns)`, `Strm`, `Name`; H2D rows named `[CUDA memcpy Host-to-Device]` (v1's documented quirk — filter on lowercase substring `host-to-device`).

- [ ] **Step 3: Write the analysis script**

Create `bench/analyze_overlap.py`:

```python
#!/usr/bin/env python3
"""Overlap-trace analysis for bench-poc-transpose nsys captures (issues #47, #67).

Reads a `nsys stats --report cuda_gpu_trace --format csv` export and computes,
for the pipeline phase (the two streams carrying the chunked H2D copies) and
the baseline phase (the single-stream whole-tensor copies):

  - H2D copy-engine utilization over the timed span (busy / span)
  - chunk-issue period (start-to-start gap; the producer cadence)
  - per-chunk copy duration
  - per-iteration wall from first-chunk starts

Warmup iterations are excluded. Chunks per iteration is inferred from the
copy count: pipeline copies = (warmup + timed) * chunks_per_iter.
"""
import argparse
import csv
import statistics
from collections import defaultdict


def load_h2d(path):
    by_stream = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if "host-to-device" not in row["Name"].lower():
                continue
            by_stream[row["Strm"]].append(
                (int(row["Start (ns)"]), int(row["Duration (ns)"])))
    return {s: sorted(v) for s, v in by_stream.items()}


def phase_stats(copies, label):
    busy = sum(d for _, d in copies)
    span = copies[-1][0] + copies[-1][1] - copies[0][0]
    print(f"{label}: {len(copies)} H2D copies, busy {busy/1e6:.2f} ms over "
          f"span {span/1e6:.2f} ms -> utilization {100*busy/span:.1f}%")
    return busy, span


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--warmup-iters", type=int, default=2)
    ap.add_argument("--timed-iters", type=int, default=5)
    args = ap.parse_args()

    by_stream = load_h2d(args.csv_path)
    if len(by_stream) < 3:
        raise SystemExit(f"expected >=3 streams with H2D copies "
                         f"(2 pipeline + 1 baseline), got {list(by_stream)}")
    # Pipeline = the two busiest streams; baseline = the next one.
    ranked = sorted(by_stream, key=lambda s: -len(by_stream[s]))
    pipe = sorted(by_stream[ranked[0]] + by_stream[ranked[1]])
    base = by_stream[ranked[2]]

    total_iters = args.warmup_iters + args.timed_iters
    if len(pipe) % total_iters:
        raise SystemExit(f"pipeline copy count {len(pipe)} not divisible by "
                         f"{total_iters} iterations")
    chunks = len(pipe) // total_iters
    print(f"inferred chunks/iteration: {chunks} "
          f"(streams: pipeline {ranked[0]},{ranked[1]}; baseline {ranked[2]})")

    timed = pipe[args.warmup_iters * chunks:]
    phase_stats(timed, "pipeline (timed)")

    periods = [b[0] - a[0] for a, b in zip(timed, timed[1:])]
    durs = [d for _, d in timed]
    print(f"chunk-issue period ms: mean {statistics.mean(periods)/1e6:.3f}, "
          f"median {statistics.median(periods)/1e6:.3f}, "
          f"min {min(periods)/1e6:.3f}, max {max(periods)/1e6:.3f}")
    print(f"copy duration ms: mean {statistics.mean(durs)/1e6:.3f}, "
          f"median {statistics.median(durs)/1e6:.3f}")

    starts = [timed[i * chunks][0] for i in range(args.timed_iters)]
    walls = [(b - a) / 1e6 for a, b in zip(starts, starts[1:])]
    print("per-iteration wall ms (first-chunk-start deltas): "
          + ", ".join(f"{w:.2f}" for w in walls))

    base_timed = base[args.warmup_iters:]
    phase_stats(base_timed, "baseline (timed)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the analysis**

```bash
python3 bench/analyze_overlap.py bench/results/poc_overlap_v2_cuda_gpu_trace.csv \
  --warmup-iters 2 --timed-iters 5
```

Expected shape: 112 pipeline copies (7 iters × 16 chunks of 64 MiB at N=16384), 80 timed. **Criterion 1 verdict rule (pre-registered):** PASS iff (a) pipeline timed utilization is near-continuous (≳ 80%, vs 25.6% in v1) AND (b) mean copy duration ≥ mean chunk-issue-period − copy-duration slack, i.e. the copy — not the gather — is the long pole (in v1 gather was ~43 ms vs ~10.6 ms copies; the flip means the period collapses to ≈ the copy duration). If either fails → STOP after committing the summary: criterion 1 failed, no headline is quoted, #67 goes back to design.

If stream identification misbehaves (e.g. the pool's pinned allocations add H2D traffic on another stream), inspect stream copy-counts printed in the error, adjust the *interpretation* (which streams are the pipeline) in the summary doc — do NOT massage the numbers.

- [ ] **Step 5: Write `bench/results/poc_overlap_summary_v2.md`**

Supersedes (does not edit) `poc_overlap_summary.md`. Same structure as v1: capture command, environment (nsys version from `nsys --version`, driver from `nvidia-smi`, **Release build** noted), the driver-JSON lines from Step 1, the analysis output verbatim, computed-numbers narrative, and an **Interpretation** section that states the criterion-1 verdict explicitly and contrasts v1's numbers (25.6% utilization, ~43 ms gather vs ~10.6 ms copy) with v2's, noting the v1 numbers were `-O0`. Close with the local-artifacts note (binaries uncommitted, same as v1).

- [ ] **Step 6: Commit**

```bash
git add bench/analyze_overlap.py bench/results/poc_overlap_summary_v2.md
git commit -m "bench: D3 overlap trace v2 + committed analysis script (#67)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Gather scaling (criterion 2)

**Files:**
- Create: `bench/results/gather_bw_n32768_v2.json`

**Interfaces:**
- Consumes: `build/cuda-release/bench/bench-gather-bw` (Task 1).
- Produces: JSON with `methods.gather_1thread.gb_per_s` and `methods.gather_multithread.gb_per_s` (arrays, one per rerun) plus `config.threads_multi`. Task 6 quotes the pair and the ratio.

- [ ] **Step 1: Check headroom, then run at N = 32768**

Peak host footprint is ~12 GiB (4 GiB src + 4 GiB dst + 4 GiB reference during the built-in correctness gate). `free -g` must show ≥ 13 GiB available; if it doesn't, drop to `--n 16384` and state the substitution in the doc.

```bash
build/cuda-release/bench/bench-gather-bw --n 32768 \
  --json bench/results/gather_bw_n32768_v2.json
```

Expected stderr: `gather_bw: N=32768 1T X GB/s, TT Y GB/s (Zx), rerun spread a% / b%` — the bench self-gates correctness (parallel gather vs `executeH2D` memcmp) before timing; a mismatch error is a STOP. Runtime: a few minutes (each timed iteration gathers 4 GiB per thread config; defaults 10+50×3).

- [ ] **Step 2: Commit**

```bash
git add bench/results/gather_bw_n32768_v2.json
git commit -m "bench: D3 gather scaling at N=32768, Release build (#67)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: `docs/poc-reproduction-v2.md`

**Files:**
- Create: `docs/poc-reproduction-v2.md`

**Interfaces:**
- Consumes: all four `poc_transpose_n*_v2.json` (Tasks 2–3), `poc_overlap_summary_v2.md` verdict (Task 4), `gather_bw_n32768_v2.json` (Task 5).
- Produces: the superseding result doc Task 7's issue comments link to.

- [ ] **Step 1: Write the doc**

Structure (mirroring v1's voice — plain, honest-reporting first):

1. **Headline** — `ours X ms vs baseline Y ms (wall, median of 3×50) → Z× speedup`, from `poc_transpose_n32768_v2.json`; rerun spreads; explicit sentence: "This supersedes `docs/poc-reproduction.md` (0.36×, measured at `-O0`)."
2. **What changed since v1** — D1 parallel per-chunk gather (#65/#68), D2 chunk heuristic 256 MiB → 64 MiB (#66/#69), Release build (`-O0` discovery reported plainly: the v1 tree had empty `CMAKE_BUILD_TYPE`; both methods re-measured under Release so the v2 ratio is internally consistent; the v1↔v2 delta spans a build-config change and is not attributable to D1+D2 alone). Protocol unchanged: same flags, same 10+50×3, `--verify` in-invocation, same golden reference plan.
3. **Acceptance criteria verdicts (issue #67, priority order)** — three explicit lines: criterion 1 verdict + one-line trace evidence (link `poc_overlap_summary_v2.md`); criterion 2 the GB/s pair + ratio; criterion 3 the measured speedup as-is. If 1–2 hold and 3 fails, add the pre-registered framing: genuine crossover result, E4/E5 input, not a tuning trigger.
4. **Numbers** — v1-style table (method / wall median / IQR / rerun spread) for N=32768, then the **size-scaling table**: N, total GiB, chunk_bytes, ours median, baseline median, speedup — with the old results (0.13–0.23× smokes at 4096/8192, 0.36× at 32768, all `-O0`) as a comparison column or adjacent rows, clearly labeled.
5. **Gather scaling** — 1T vs multi-T GB/s, thread count, `min_rows_per_worker`, note it feeds E2/P3 and supersedes the `-O0` `gather_bw_n4096.json`.
6. **Honest-reporting notes** — deviations (or none), stability spreads vs the 5% bar, verification result, the no-tuning statement, WSL2 standing caveat.
7. **Bare-metal confirmation** — marked empty slot: "*Pending: no bare-metal box was available at measurement time (2026-07-11). To be appended if one becomes available; WSL2 remains the standing caveat until then.*"
8. Raw-data pointers to all committed v2 JSONs.

- [ ] **Step 2: Self-check against the spec**

Re-read the spec's step-6 bullet list; confirm every bullet has a section, all three criteria are explicit, and no v1 file was modified (`git status` shows only new files).

- [ ] **Step 3: Commit**

```bash
git add docs/poc-reproduction-v2.md
git commit -m "docs: poc-reproduction-v2 superseding the 0.36x doc (#67)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: PR + issue updates (user gate)

**Files:**
- None (GitHub operations only)

**Interfaces:**
- Consumes: everything committed in Tasks 2–6; criteria verdicts from Task 6.

- [ ] **Step 1: Push and open the PR**

```bash
git push -u origin d3-remeasurement
gh pr create --title "docs(bench): D3 C7 re-measurement + result-doc supersession (#67)" --body "..."
```

PR body: implements #67; one line per criterion verdict; the `-O0` discovery and Release-only decision (user-approved) stated up front; list of superseded-not-edited v1 artifacts; note that Nsight binaries are uncommitted. End with the standard generated-with footer.

- [ ] **Step 2: Draft issue comments — SHOW TO USER BEFORE POSTING**

Draft, then present to the user for approval (hard gate — do not `gh issue comment` without it):
- **#47 comment:** v2 outcome summary, link to `docs/poc-reproduction-v2.md` and the PR; explicit "supersedes the 0.36× result posted here; that number was measured at `-O0`".
- **#63 comment:** same summary angled at the pipelined-transfer design question; close #63 **only if criteria 1–2 both hold** (`gh issue close 63 --comment "..."`), otherwise comment without closing and say which criterion failed.

- [ ] **Step 3: After user approval, post the comments (and close #63 if warranted)**

```bash
gh issue comment 47 --body-file <draft>
gh issue comment 63 --body-file <draft>   # or: gh issue close 63 --comment "$(cat <draft>)"
```

---

## Verification (whole-plan)

- `git log --oneline main..d3-remeasurement` shows: spec, plan, 4 result commits, doc commit — no source-code changes outside `bench/analyze_overlap.py`.
- `git diff main --stat -- docs/poc-reproduction.md bench/results/poc_overlap_summary.md bench/results/poc_transpose_n32768.json bench/results/gather_bw_n4096.json` is empty (v1 untouched).
- Every number quoted in `poc-reproduction-v2.md` traces to a committed JSON or the v2 overlap summary.
