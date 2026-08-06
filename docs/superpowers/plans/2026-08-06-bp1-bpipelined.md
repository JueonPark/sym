# BP1 — Method::BPipelined Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement issue #114 (BP1): `Method::BPipelined` in `bench/rtrack/rtrack_bench.cu` — b_fair's pinned-source chunked DMA with the per-chunk transform kernel issued in-stream, per-chunk events on both legs, an `h2d_occupancy` CSV column, bit-exact verification riding the existing gate, and T2 recorded N/A per the audit on issue #114.

**Architecture:** The spec is `docs/superpowers/specs/2026-08-06-bp1-bpipelined-design.md`; the chunkability audit is the comment on issue #114 — read both first. Three pieces: (1) the harness core — enum/CLI/fixture predicates, `Pipeline` per-chunk kernel events, the per-family chunk-kernel dispatch helper with alignment guards, `runMethodBPipelined`, the T2 loud skip; (2) the `h2d_occupancy` CSV column; (3) smoke verification + b/b_fair-intact proof + PR.

**Tech Stack:** CUDA C++ (sm_75 on this box), standalone nvcc build (no CUDA cmake tree here).

## Global Constraints

- `runMethodB` (BStaged) and `runMethodBFair` bodies are NOT modified — three baseline generations coexist. The proof is `git diff`: zero hunks inside those two functions.
- T2's `b_pipelined` cell is N/A — a loud stderr skip, never a silent omission, never a relocate-only hybrid. Alignment violations hard-fail (`std::exit(1)` with a message naming the workload), never approximate.
- No measurement artifacts: nothing lands in `bench/results/` (bp_* files are #BP3's). No libreloc file changes.
- The CSV gains exactly one column, appended after `wire`: `h2d_occupancy` (median of per-iteration `t.h2d / t.gpu`, both event-derived), populated for every method.
- Build (standalone nvcc; the CM1-era recipe — per-file ISA flags are mandatory, a flat union SIGILLs on this AVX2-only EPYC):

```bash
SCRATCH=build/cm1-tools && mkdir -p "$SCRATCH"
g++ -O3 -DNDEBUG -std=c++17 -DRELOC_QUANT_HAVE_X86_SIMD=1 -Ilibreloc/include \
  -mavx2 -mfma -mf16c -pthread -c libreloc/quant/QuantAVX2.cpp -o "$SCRATCH/QuantAVX2.o"
g++ -O3 -DNDEBUG -std=c++17 -DRELOC_QUANT_HAVE_X86_SIMD=1 -Ilibreloc/include \
  -mavx512f -mavx512bw -pthread -c libreloc/quant/QuantAVX512.cpp -o "$SCRATCH/QuantAVX512.o"
/usr/local/cuda-12.5/bin/nvcc -ccbin g++ -O3 -DNDEBUG -std=c++17 -arch=sm_75 \
  -DRELOC_ENABLE_CUDA=1 -DRELOC_QUANT_HAVE_X86_SIMD=1 -Ilibreloc/include -Ibench \
  bench/rtrack/rtrack_bench.cu libreloc/src/*.cpp libreloc/cuda/*.cu libreloc/quant/Quant.cpp \
  "$SCRATCH/QuantAVX2.o" "$SCRATCH/QuantAVX512.o" \
  -o "$SCRATCH/rtrack-bp1" -Xcompiler -pthread
```

- Smoke runs on GPU0 with affinity pinning: `taskset -c 4-7,20-23 "$SCRATCH/rtrack-bp1" …`.
- clang-format clean on the touched files (`build/cm1-tools/fmt/bin/clang-format`).
- Branch: `bp1-bpipelined` (exists, spec committed). Deliverable: regular PR.
- Commit messages end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: Harness core — BPipelined end to end

**Files:**
- Modify: `bench/rtrack/rtrack_bench.cu` (enum/tag ~:181-195, Options usage, CLI ~:1108-1121 + usage string ~:1086, buildFixture call site in `run()`, `Pipeline` ~:474-560, new helpers before `runConfig`, `runConfig` ~:865-903, `run()` method loop ~:1019-1024)

**Interfaces:**
- Consumes: existing `Fixture`, `ByteChunks`/`planByteChunks`, `methodBDmaDst`, `launchConvertF32F16` (bench-local, ~:78-92 — match its exact dst parameter type when transcribing), `reloc::cuda::{relocateF32, quantizeF32S8}`.
- Produces: `Method::BPipelined` / tag `"b_pipelined"` / CLI `bpipe`; `Pipeline::kernBeg/kernEnd` + `sumKernMs()` (gated by a new `withKern` ctor param, default false); `launchBPipeChunkKernel(const Fixture&, int64_t byteOff, int64_t bytes, cudaStream_t)`; `runMethodBPipelined(const Fixture&, const ByteChunks&, Pipeline&)`. Task 2 fills the occupancy from the `t.h2d`/`t.gpu` this task already populates.

- [ ] **Step 1: Enum, tag, CLI, usage.** Extend `Method { A, BStaged, BFair, BPipelined }`; `methodTag` gains `case Method::BPipelined: return "b_pipelined";` (comment: the tag `cm4_registered_predictions.json`'s `placement_map` reserves for the Overlapped placement). CLI: `else if (m == "bpipe") opt.methods = {Method::BPipelined};` and `all` becomes `{Method::A, Method::BStaged, Method::BFair, Method::BPipelined}`; usage string gains `bpipe`. Update the file-top method comment (~:175-180) with one sentence for BPipelined.

- [ ] **Step 2: Fixture predicate.** In `run()`, add `const bool needBPipe = hasMethod(opt.methods, Method::BPipelined);` next to `needBFair`, and pass `needBFair || needBPipe` where `buildFixture` takes its `needBFair` argument (that parameter gates exactly the two things BPipelined also needs: `pinnedSrc`, and via `needB || needBFair`, `dLin`/`dTmp`). Add a comment at the call site: `// BPipelined rides the BFair predicate: same pinned source, same dLin/dTmp.`

- [ ] **Step 3: Pipeline kernel events.** Add to `Pipeline`: `std::vector<cudaEvent_t> kernBeg, kernEnd; // BPipelined per-chunk kernel leg` and a `bool withKern = false` ctor parameter after `allocStaging`; allocate/destroy exactly like `recvBeg/recvEnd`; add:

```cpp
  double sumKernMs() const {
    double total = 0;
    for (size_t c = 0; c < kernBeg.size(); ++c) {
      float ms = 0;
      CUDA_CHECK(cudaEventElapsedTime(&ms, kernBeg[c], kernEnd[c]));
      total += ms;
    }
    return total;
  }
```

- [ ] **Step 4: The per-chunk kernel dispatch helper** (place after `methodBDmaDst`, before `runMethodA`):

```cpp
[[noreturn]] void bpipeMisaligned(const char *id, int64_t off) {
  std::fprintf(stderr,
               "error: b_pipelined chunk misaligned for %s at byte %lld -- "
               "refusing to approximate (issue #114 chunkability audit)\n",
               id, static_cast<long long>(off));
  std::exit(1);
}

// b_pipelined's per-chunk kernel (issue #114): the chunk is a contiguous
// slab [byteOff, byteOff+bytes) of the linear fp32 source that just
// landed in dLin; launch the workload's transform over exactly that
// slab. Slicing per the chunkability audit recorded on issue #114:
// identity plans use pointer/channel offsets; blocked/nchw plans slice
// whole 64-row groups / images (a rectangular sub-plan); transposePlan
// slabs are dst column bands (legal relocateF32 input, but no longer
// isTranspose2D-shaped, so the SMEM tile path falls back to the naive
// kernel -- the audit's recorded perf-class caveat). T2 (RelocateQuant
// on transposePlan) never reaches here: run() skips it as N/A.
void launchBPipeChunkKernel(const Fixture &f, int64_t byteOff, int64_t bytes,
                            cudaStream_t stream) {
  const Workload &w = *f.w;
  const int64_t elemOff = byteOff / 4;
  const int64_t elems = bytes / 4;
  switch (w.gpuStage) {
  case GpuStage::None:
    return; // DMA-only row (T3R100): no kernel leg by construction
  case GpuStage::ConvertF16:
    launchConvertF32F16(f.dLin + elemOff,
                        static_cast<uint16_t *>(f.dOut) + elemOff, elems,
                        stream);
    return;
  case GpuStage::Quantize: {
    if (elemOff % f.channelSize != 0 || elems % f.channelSize != 0)
      bpipeMisaligned(w.id, byteOff);
    const int64_t c0 = elemOff / f.channelSize;
    reloc::cuda::quantizeF32S8(f.dLin + elemOff,
                               static_cast<int8_t *>(f.dOut) + elemOff,
                               elems / f.channelSize, f.channelSize,
                               f.dInv + c0, stream);
    return;
  }
  case GpuStage::Relocate:
  case GpuStage::RelocateQuant: {
    reloc::BoundPlan sub = f.bound;
    const float *src = f.dLin + elemOff;
    float *relocDst = nullptr;
    int64_t chanBegin = 0, chanCount = 0;
    const size_t rank = f.bound.extents.size();
    if (rank == 3) {
      // blockedTransposePlan {64, m, n}: slab = whole 64-src-row groups
      // (group = 64*n elems); slice axis 1, shift dst by j0*n.
      const int64_t n = f.bound.extents[2];
      const int64_t group = 64 * n;
      if (elemOff % group != 0 || elems % group != 0)
        bpipeMisaligned(w.id, byteOff);
      const int64_t j0 = elemOff / group;
      sub.extents[1] = elems / group;
      relocDst = static_cast<float *>(f.dOut) + j0 * n;
    } else if (rank == 4) {
      // nchwToNhwcPlan {b, H, W, C}: slab = whole images (64*n elems);
      // slice axis 0 -- the dst channel axis, so the quantize leg
      // chunks with dInv + b0.
      const int64_t image = f.channelSize; // = 64*n
      if (elemOff % image != 0 || elems % image != 0)
        bpipeMisaligned(w.id, byteOff);
      chanBegin = elemOff / image;
      chanCount = elems / image;
      sub.extents[0] = chanCount;
      relocDst = (w.gpuStage == GpuStage::RelocateQuant ? f.dTmp
                                                        : static_cast<float *>(
                                                              f.dOut)) +
                 elemOff;
    } else {
      // transposePlan {n, n}, srcStrides {1, n}: slab = axis-1 band
      // (dst column band j0..j0+nj). RelocateQuant on this shape is T2,
      // which run() already skipped as N/A -- guard defensively.
      if (w.gpuStage == GpuStage::RelocateQuant) {
        std::fprintf(stderr,
                     "error: %s: RelocateQuant on a rank-2 transpose plan "
                     "is N/A for b_pipelined (issue #114 audit)\n",
                     w.id);
        std::exit(1);
      }
      const int64_t n = f.bound.extents[0];
      if (elemOff % n != 0 || elems % n != 0)
        bpipeMisaligned(w.id, byteOff);
      const int64_t j0 = elemOff / n;
      sub.extents[1] = elems / n;
      relocDst = static_cast<float *>(f.dOut) + j0;
    }
    if (w.gpuStage == GpuStage::Relocate) {
      reloc::cuda::relocateF32(sub, src, relocDst, stream);
      return;
    }
    // RelocateQuant (rank-4 / T4 family only): relocate the image slab
    // into dTmp, then per-channel quantize exactly that channel range.
    reloc::cuda::relocateF32(sub, src, relocDst, stream);
    reloc::cuda::quantizeF32S8(f.dTmp + chanBegin * f.channelSize,
                               static_cast<int8_t *>(f.dOut) +
                                   chanBegin * f.channelSize,
                               chanCount, f.channelSize, f.dInv + chanBegin,
                               stream);
    return;
  }
  }
}
```

(Note the rank-3 blocked case writes `f.dOut` directly — T1b-family Relocate has no dTmp; the rank-4 Relocate case (`T4R100`) likewise goes straight to `f.dOut` via the ternary, while rank-4 RelocateQuant (T4) relocates into `f.dTmp + elemOff` and quantizes exactly the `[chanBegin, chanBegin+chanCount)` channel range.)

- [ ] **Step 5: `runMethodBPipelined`** (place after `runMethodBFair`):

```cpp
// Method B (pipelined) -- issue #114's overlap-fair baseline: b_fair's
// DMA path (pinned source, no staging, no reuse gate) with the per-chunk
// transform kernel issued in-stream after each chunk's copy -- Method
// A's loop shape on B's buffer model. Chunk c's kernel is ordered after
// chunk c's copy by the stream; dLin chunk regions are disjoint across
// c. Per-family slicing: see launchBPipeChunkKernel and the audit on
// issue #114.
StageTimes runMethodBPipelined(const Fixture &f, const ByteChunks &ck,
                               Pipeline &pl) {
  const char *src = reinterpret_cast<const char *>(f.pinnedSrc);
  const bool hasKern = f.w->gpuStage != GpuStage::None;
  StageTimes t;
  const double w0 = nowMs();
  CUDA_CHECK(cudaEventRecord(pl.evStart, pl.stream));
  for (int64_t c = 0; c < ck.nChunks; ++c) {
    const int64_t off = c * ck.bytesPerChunk;
    const int64_t bytes = std::min(ck.bytesPerChunk, f.inBytes - off);
    CUDA_CHECK(cudaEventRecord(pl.h2dBeg[static_cast<size_t>(c)], pl.stream));
    CUDA_CHECK(cudaMemcpyAsync(methodBDmaDst(f) + off, src + off,
                               static_cast<size_t>(bytes),
                               cudaMemcpyHostToDevice, pl.stream));
    CUDA_CHECK(cudaEventRecord(pl.h2dEnd[static_cast<size_t>(c)], pl.stream));
    if (hasKern) {
      CUDA_CHECK(
          cudaEventRecord(pl.kernBeg[static_cast<size_t>(c)], pl.stream));
      launchBPipeChunkKernel(f, off, bytes, pl.stream);
      CUDA_CHECK(
          cudaEventRecord(pl.kernEnd[static_cast<size_t>(c)], pl.stream));
    }
  }
  CUDA_CHECK(cudaEventRecord(pl.evStop, pl.stream));
  CUDA_CHECK(cudaStreamSynchronize(pl.stream));
  t.wall = nowMs() - w0;
  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, pl.evStart, pl.evStop));
  t.gpu = ms;
  t.h2d = pl.sumH2dMs();
  t.kern = pl.sumKernMs(); // 0 when hasKern is false (empty vectors)
  t.cpu = 0.0;             // pinned source: no host staging copy
  return t;
}
```

- [ ] **Step 6: Wire runConfig + the T2 skip.** In `runConfig`: the Pipeline construction becomes

```cpp
  Pipeline pl(stagingBytes, nChunks,
              /*withRecv=*/methodA && w.recvStage != RecvStage::None,
              /*allocStaging=*/method != Method::BFair &&
                  method != Method::BPipelined,
              /*withKern=*/method == Method::BPipelined &&
                  w.gpuStage != GpuStage::None);
```

and the dispatch switch gains `case Method::BPipelined: return runMethodBPipelined(f, bck, pl);`. In `run()`'s method loop, immediately after the existing `if (m != Method::A && !w->methodB) continue;`:

```cpp
        // T2's quantize leg cannot run per chunk (column bands are not
        // channel-contiguous): N/A per the chunkability audit recorded
        // on issue #114 -- a loud skip, never a silent omission.
        if (m == Method::BPipelined && std::strcmp(w->id, "T2") == 0) {
          std::fprintf(stderr,
                       "rtrack: T2   b_pipelined N/A (chunkability audit, "
                       "issue #114); skipped\n");
          continue;
        }
```

- [ ] **Step 7: Build + smoke.** Build per Global Constraints, then:

```bash
taskset -c 4-7,20-23 "$SCRATCH/rtrack-bp1" --n 2048 --method bpipe \
  --warmup 1 --iters 3 --machine smoke --csv /tmp/bp1_smoke_bpipe.csv --csv-header 2>/tmp/bp1_smoke_bpipe.log
grep -c "VERIFY FAILED" /tmp/bp1_smoke_bpipe.log; grep "b_pipelined N/A" /tmp/bp1_smoke_bpipe.log | head -2
grep -c "\[verified\]" /tmp/bp1_smoke_bpipe.log
```

(Adjust flag spellings to the usage string — e.g. the CSV-header flag — by checking `--help`/usage output first.) Expected: zero VERIFY FAILED; the T2 N/A line once per surviving chunk-sweep point; every emitted row `[verified]`; b_pipelined rows present for T1, T1b, T3, T4, T5 families and the rsweep R100 rows.

- [ ] **Step 8: Commit**

```bash
git add bench/rtrack/rtrack_bench.cu
git commit -m "bench(rtrack): Method::BPipelined -- per-chunk in-stream B kernel, T2 N/A (#114)"
```

---

### Task 2: h2d_occupancy CSV column

**Files:**
- Modify: `bench/rtrack/csv.h` (CsvRow ~:25-37, header ~:39-44, row line ~:53-72)
- Modify: `bench/rtrack/rtrack_bench.cu` (`runConfig` sample loop ~:923-934 and row fill ~:936-960)
- Modify: `bench/rtrack/README.md` (the CSV column documentation block)

**Interfaces:**
- Consumes: `StageTimes.h2d`/`.gpu` per iteration (Task 1 populates them for BPipelined; a/b/b_fair already do).
- Produces: CSV column 26, `h2d_occupancy`, appended after `wire`.

- [ ] **Step 1: csv.h.** `CsvRow` gains `RStats h2dOcc;` (after `gpuRecv` in the RStats list is fine — field order in the struct is not the emit order). `csvHeaderLine()` gains `",h2d_occupancy"` at the end. `csvRowLine` appends `out += ',' + num(r.h2dOcc.median);` as the final field, after the `wire` append.

- [ ] **Step 2: runConfig.** Add `occ` to the sample vectors:

```cpp
  std::vector<double> wall, gpu, cpu, h2d, kern, recv, occ;
  ...
    occ.push_back(t.gpu > 0 ? t.h2d / t.gpu : 0.0);
  ...
  row.h2dOcc = summarizeSamples(occ);
```

with a comment at the push: `// h2d-busy / pipeline-span, both event-derived -- the issue #114 overlap-occupancy figure (WSL2-compatible substitute for a trace).`

- [ ] **Step 3: README.** In `bench/rtrack/README.md`'s CSV column list, append the `h2d_occupancy` description: "median over iterations of sum(per-chunk h2d event time) / (evStart→evStop span); ~1.0 = the pipeline is DMA-saturated (kernels fully hidden), lower = exposed kernel or gaps. Populated for every method (issue #114)."

- [ ] **Step 4: Build + smoke both a column check and a semantics check:**

```bash
# rebuild (same recipe), then:
taskset -c 4-7,20-23 "$SCRATCH/rtrack-bp1" --n 2048 --method all \
  --warmup 1 --iters 3 --machine smoke --csv /tmp/bp1_smoke_all.csv --csv-header 2>/tmp/bp1_smoke_all.log
python3 - <<'EOF'
import csv
rows = list(csv.DictReader(l for l in open("/tmp/bp1_smoke_all.csv") if not l.startswith("#")))
assert rows and all("h2d_occupancy" in r for r in rows)
for r in rows:
    v = float(r["h2d_occupancy"])
    assert 0.0 < v <= 1.05, (r["method"], r["transform"], v)  # small event jitter above 1.0 would be a bug flag
bp = {r["transform"]: float(r["h2d_occupancy"]) for r in rows
      if r["method"] == "b_pipelined" and r["transform"] in ("blocked_transpose", "quant")}
bf = {r["transform"]: float(r["h2d_occupancy"]) for r in rows
      if r["method"] == "b_fair" and r["transform"] in ("blocked_transpose", "quant")}
print("b_pipelined occ:", bp, " b_fair occ:", bf)
for k in bp:
    assert bp[k] >= bf.get(k, 0), (k, bp[k], bf.get(k))  # pipelined hides the kernel -> higher h2d share of span
EOF
```

Expected: all rows carry the column in (0, 1.05]; b_pipelined occupancy ≥ b_fair on kernel-bearing families (b_fair's span includes the serial kernel tail, so its h2d share is lower). If the ≥ check fails, investigate before proceeding — it is the semantic core of the column.

- [ ] **Step 5: Commit**

```bash
git add bench/rtrack/csv.h bench/rtrack/rtrack_bench.cu bench/rtrack/README.md
git commit -m "bench(rtrack): h2d_occupancy CSV column -- event-derived overlap figure (#114)"
```

---

### Task 3: Intact-baselines proof, spec amendment, clang-format, PR

**Files:**
- Modify: `docs/superpowers/specs/2026-08-06-bp1-bpipelined-design.md` (one amendment)
- Possible clang-format reflows on the two touched C++ files.

**Interfaces:** consumes everything above; produces the PR.

- [ ] **Step 1: b/b_fair-intact proof.** The spec's §2 says "b/b_fair rows byte-compare against a pre-change smoke CSV" — that check is wrong as written (timing medians vary run to run; CSV rows can never byte-match across runs). The correct proof, run it:

```bash
git diff main -- bench/rtrack/rtrack_bench.cu | grep -E '^@@' | head -20
# manually confirm no hunk header falls inside runMethodB (:777-822) or runMethodBFair (:824-861 pre-change line ranges);
# stronger: extract both function bodies at main and at HEAD and diff them:
for fn in runMethodB runMethodBFair; do
  for ref in main HEAD; do
    git show $ref:bench/rtrack/rtrack_bench.cu | awk "/^StageTimes $fn\(/,/^}/" > /tmp/${fn}_${ref}.txt
  done
  diff /tmp/${fn}_main.txt /tmp/${fn}_HEAD.txt && echo "$fn: identical"
done
```

Expected: both functions byte-identical (note `runMethodB(` must not also match `runMethodBPipelined(`/`runMethodBFair(` — the awk pattern `runMethodB\(` with the escaped paren handles it). Then append to the spec's §2 verification bullet an amendment section:

```markdown
## Amendment (2026-08-06, during implementation)

§2's "b/b_fair rows byte-compare against a pre-change smoke CSV" was uncheckable as
written — timing medians differ across runs. The three-generations-intact proof used
instead: the `runMethodB` and `runMethodBFair` function bodies are byte-identical
between main and this branch (extracted and diffed), plus both methods still print
`[verified]` in the --method all smoke.
```

- [ ] **Step 2: clang-format + full suites**

```bash
build/cm1-tools/fmt/bin/clang-format -i bench/rtrack/rtrack_bench.cu bench/rtrack/csv.h
git diff --stat   # if reflows occurred, rebuild + re-run the Task 2 smoke before committing
build/sym/libreloc/test/libreloc-test 2>&1 | tail -3
PYTHONPATH=build/sym/python python3 -m pytest libreloc/python/tests/ -q
git status --short bench/results/   # MUST print nothing (no artifacts)
```

- [ ] **Step 3: Commit + push + PR (regular)**

```bash
git add -A bench/rtrack docs/superpowers/specs/2026-08-06-bp1-bpipelined-design.md
git commit -m "docs(specs), style: BP1 intact-baselines amendment + clang-format (#114)"
git push -u origin bp1-bpipelined
gh pr create --title "bench(rtrack): BP1 — Method::BPipelined, overlap-fair Method B (#114)" --body "<body>"
```

PR body, in order: (1) verdict-first — b_pipelined implemented per the chunkability audit (link the #114 comment), all runnable configs `[verified]` in the smoke, T2 N/A loud-skip in place, occupancy column live with the b_pipelined ≥ b_fair semantic check passing (quote the smoke numbers); (2) what changed per file, incl. the Pipeline withKern events and the per-family slicing helper with the transposePlan naive-fallback caveat restated; (3) three-generations-intact proof (function-body diff result) + the spec amendment note; (4) explicitly: no measurement artifacts in this PR — `bp_*` is #BP3, `gates.py --exp bp` and `figure_rstar --b-method b_pipelined` are #BP2; (5) `Refs #114, #108`; (6) the standard generated-with footer.

---

## Verification (end-to-end, after all tasks)

1. Smoke `--method all --n 2048`: zero VERIFY FAILED, every row `[verified]`, T2 N/A line present, b_pipelined rows for T1/T1b/T3/T4/T5 + R100 rsweep rows.
2. `h2d_occupancy` ∈ (0, 1.05] everywhere; b_pipelined ≥ b_fair on kernel-bearing families.
3. `runMethodB`/`runMethodBFair` bodies byte-identical vs main; both still verify.
4. Full C++ + pytest suites green; `git status --short bench/results/` empty.
5. Issue #114 acceptance: all configs `[verified]` ✓ (smoke evidence); stage split + occupancy in CSV schema ✓ (`h2d_ms`, `gpu_kernel_ms` per-chunk sum, `h2d_occupancy`); N/A cells listed ✓ (T2 loud skip + audit comment, posted before any implementation commit — verify the comment timestamp precedes the first code commit).
