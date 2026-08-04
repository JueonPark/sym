#!/usr/bin/env python3
"""Calibration assembler (issue #97): committed bench/results artifacts ->
a deterministic `.cal` file for `reloc::costmodel::CostModel`.

  python3 bench/rtrack/make_calibration.py \
      --machine epyc7351-2080ti --out calibration/epyc7351-2080ti.cal
  python3 bench/rtrack/make_calibration.py \
      --machine 7800x3d-4070tis --out calibration/7800x3d-4070tis.cal

Run from the repo root (relative paths below resolve against CWD, the
same convention as run_rtrack.py/gates.py). Every emitted `key value`
line carries a trailing `# <source>` provenance comment; nothing but the
output path is printed on success. Hard-fails (nonzero exit, message on
stderr) if a source file this machine's SOURCES map lists is missing --
a (pattern, kernel) cell with no measured file is a deliberate omission
(the model returns nullopt for it), not a missing-file error.

SOURCES per machine (frozen provenance -- see task-6 brief / design doc
`docs/superpowers/specs/2026-07-29-v3-costmodel-design.md` S1):

  epyc7351-2080ti:
    - pcie.h2d_gbps                <- bench/results/v1_gate_report.txt
    - cpu.t{8,1}.{pattern}.{kernel}_gbps
                                   <- bench/results/r1_rooflines/r1_roofline_*_t{T}.json
      contig_read/quantize_pack/convert_f32_f16 (no plan tag) -> contiguous;
      gather_f32/gather_quantize {blocked,transpose,nchw} -> {blocked,
      single_element,tiled}. blocked/single_element/tiled reuse
      contiguous's convert_f32_f16 (second pass runs on the already-
      gathered contiguous buffer). pack_s8_s4: NO valid epyc source
      exists (see NOTE below) -- omitted entirely (r=0.125 tier
      unmodelable on this box).
    - hbm.bw_gbps, hbm.m.{contiguous,blocked,single_element,tiled},
      hiding.ratio            <- bench/results/r4_hiding_ratio_epyc_2080ti.json
    - multigpu.delivery_gbps.k{2,4}
                                   <- bench/results/m0_multigpu_h2d_{pair01,all4}.json
    - prefold.alloc_ms_per_gib     <- bench/results/v4_scatter_n8192_epyc_2080ti.json
    - overhead.{a,b}_ms            <- bench/results/v1_gen3_nsweep_epyc_2080ti.csv
    - strategy.*                   <- P2 defaults (no small-size sweep yet)
    - pipeline.chunks_per_buffer: NOT emitted in CM1 -- activating the
      Overlapped fill/drain term re-scores the frozen V3 prediction
      report (issue #97 pre-registered verdicts, which stand as measured
      per #107); the key lands in CM5 together with the CM4
      re-registered gate rule.
    - recv.m.{convert_f16_f32,dequant_s8_f32}
                                     <- bench/results/v2_isa_gen3_rsweep_avx2_epyc7351-2080ti.csv
      (gpu_recv_ms, method=a transform=quant, min over chunk sweep, vs
      the R4 copy_f32 ceiling at the m-table N)
    - recv.m.unpack_dequant_s4       <- bench/results/cm1_recv_kernel_bw_epyc_2080ti.json
      (CM1 targeted isolated run; same-run copy_f32 ceiling)

  7800x3d-4070tis: same key shapes, sourced from
    bench/results/v1_gen4_gate_report.txt, bench/results/r2_rooflines/*.json
    (per-plan files carrying ALL kernels; identity/blocked/transpose/nchw
    -> contiguous/blocked/single_element/tiled; pack_s8_s4 measured
    directly per-plan here, no reuse needed for it), and
    bench/results/v1_gen4_matrix_nsweep_7800x3d_4070tis.csv (224-row
    original) MERGED with its
    bench/results/v1_gen4_matrix_nsweep_rerun_7800x3d_4070tis.csv
    companion for overheads. No multigpu or HBM sweep exists on this box:
    hbm.bw_gbps and hbm.m.* are PROXIED from the 2080 Ti computation with
    an explicit note; multigpu.*, hiding.ratio and prefold.* are omitted
    outright.
    - pipeline.chunks_per_buffer: NOT emitted in CM1 -- activating the
      Overlapped fill/drain term re-scores the frozen V3 prediction
      report (issue #97 pre-registered verdicts, which stand as measured
      per #107); the key lands in CM5 together with the CM4
      re-registered gate rule.
    - recv.m.*                       <- bench/results/cm1_recv_kernel_bw_7800x3d_4070tis.json
      when it exists (issue #109 runbook); omitted until then. The
      "relocate/transpose recv" m values from the issue's list are the
      existing hbm.m.{pattern} keys -- not duplicated under recv.m.*.

    Fix-report addenda (post-review, see task-6-report.md "Fix report"):
    - Gen4 rooflines are pinned at N=8192 for BOTH t1 and t8 (not
      "largest N wins"): only t8 has N=16384 files, so the old rule
      compared t1@8192 against t8@16384 and produced a V-cache-residency
      inversion (t1 pack_s8_s4 35.0 > t8's 26.5) that is an artifact of
      the N mismatch, not a real T1-vs-T8 effect. N=8192 exists for every
      plan at both thread counts, so it is used uniformly.
    - Gen4 overhead.{a,b}_ms fit over the MERGED original+rerun T3 quant
      rows (stabler-of-the-two-files best-chunk per (method, N), per the
      pre-declared rule in docs/r2-exp2-gen4-crossover.md:48-58), across
      the full N span the original file covers (2048..16384) -- not just
      the rerun file's partial 2048/4096 span used previously.

  NOTE on pack_s8_s4 provenance (inspected, not assumed): the only
  pack_s8_s4 measurement committed outside r2_rooflines is
  bench/results/quant_bw_n8192_t{8,1}*.json (PR #74, R0.1). Its t8 file
  resolves variant "avx512" for every quant kernel via
  reloc::quant::cpuSupports(AVX512), a runtime __builtin_cpu_supports
  check -- impossible on the EPYC 7351 (Zen 1, no AVX-512F; see
  memory/v2-isolation.md). Its numbers also match the Gen4
  r2_rooflines/identity_n8192_t8.json roofline (quantize_pack 36.6,
  convert_f32_f16 ~27, gather_f32 ~16-17) far more closely than EPYC's
  r1_rooflines (quantize_pack 23.0, convert 17.4). Conclusion: the
  quant_bw_*.json files are Gen4 (7800X3D) data, not EPYC -- despite
  lacking a machine suffix in their filename (pre-dating the multi-
  machine naming convention). They are therefore NOT used as an epyc
  source; epyc's pack_s8_s4 (and hence its whole r=0.125 dtype tier) is
  simply unmodelable and the corresponding keys are omitted. Gen4 does
  not need this file either -- its own r2_rooflines per-plan files
  already carry a directly-measured pack_s8_s4 kernel.
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

RESULTS = "bench/results"
ENCODING = "utf-8"  # explicit: em-dashes in header/comments crash under
                    # a plain-C locale's default open() encoding otherwise.

# ---------------------------------------------------------------- helpers --

def must_read_text(path):
    p = Path(path)
    if not p.exists():
        sys.exit(f"error: missing required source file: {path}")
    return p.read_text(encoding=ENCODING)


def must_read_json(path):
    return json.loads(must_read_text(path))


def read_json_optional(path):
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding=ENCODING))


def load_csv_rows(path, machine):
    """Load non-comment CSV rows and assert every row's `machine` column
    matches --machine (M5): a source file mismatched to the wrong box
    would silently miscalibrate it."""
    text = must_read_text(path)
    rows = list(csv.DictReader(l for l in text.splitlines() if not l.startswith("#")))
    bad = {r.get("machine") for r in rows} - {machine}
    if bad:
        sys.exit(f"error: {path} has machine column value(s) {sorted(bad)}, "
                 f"expected only {machine!r}")
    return rows


def must_read_gate_report(path, machine):
    """Parse a `v1*_gate_report.txt`: assert its `=== <slug> ===` line
    matches --machine (M5), then return the pinned H2D GB/s figure."""
    text = must_read_text(path)
    slug_m = re.search(r"^=== (\S+) ===\s*$", text, re.MULTILINE)
    if not slug_m:
        sys.exit(f"error: could not find '=== <machine> ===' line in {path}")
    if slug_m.group(1) != machine:
        sys.exit(f"error: {path} is for machine {slug_m.group(1)!r}, "
                 f"expected {machine!r}")
    return parse_pinned_h2d(text, path)


def fmt_num(x):
    """Shortest round-tripping text for x; whole numbers print without a
    decimal point (needed for exact byte-count constants)."""
    xf = float(x)
    if xf.is_integer():
        return str(int(xf))
    return repr(xf)


def parse_pinned_h2d(text, path):
    m = re.search(r"pinned H2D \(DMA leg\):\s*([0-9.]+)\s*GB/s", text)
    if not m:
        sys.exit(f"error: could not find 'pinned H2D (DMA leg):' line in {path}")
    return float(m.group(1))


class Emitter:
    """Accumulates `key value  # source` lines; rejects duplicate keys."""

    def __init__(self, machine):
        self.machine = machine
        self.lines = []
        self.keys = set()

    def emit(self, key, value, source, note=None):
        if key in self.keys:
            raise AssertionError(f"duplicate calibration key: {key}")
        self.keys.add(key)
        comment = source if note is None else f"{source} -- {note}"
        self.lines.append(f"{key} {fmt_num(value)}  # {comment}")

    def text(self):
        header = [
            "# costmodel calibration v0",
            f"# machine: {self.machine}",
            "# generated by make_calibration.py from committed bench/results "
            "— regenerate, do not hand-edit",
        ]
        return "\n".join(header + self.lines) + "\n"


# Gather-kernel plan tag (as it appears in filenames / the r2 "plan"
# config field) -> Pattern name the C++ classifier/model uses.
GATHER_PLAN_TO_PATTERN = {
    "blocked": "blocked",
    "transpose": "single_element",
    "nchw": "tiled",
}
CONTIG_KERNELS = ("contig_read", "quantize_pack", "convert_f32_f16")
GATHER_KERNELS = ("gather_f32", "gather_quantize")
THREADS = (8, 1)


# ------------------------------------------------------------- epyc box ---

def build_epyc(e):
    r1 = f"{RESULTS}/r1_rooflines"

    # pcie.h2d_gbps <- V1 admissible anchor.
    gate_path = f"{RESULTS}/v1_gate_report.txt"
    h2d = must_read_gate_report(gate_path, e.machine)
    e.emit("pcie.h2d_gbps", h2d, gate_path)

    # contiguous-pattern kernels: files without a plan tag.
    contig_val = {t: {} for t in THREADS}
    for kernel in CONTIG_KERNELS:
        for t in THREADS:
            path = f"{r1}/r1_roofline_{kernel}_t{t}.json"
            doc = must_read_json(path)
            val = doc["kernels"][kernel]["in_gb_per_s"]
            e.emit(f"cpu.t{t}.contiguous.{kernel}_gbps", val, path)
            contig_val[t][kernel] = val

    # gather-pattern kernels: blocked/single_element/tiled.
    for plan_tag, pattern in GATHER_PLAN_TO_PATTERN.items():
        for kernel in GATHER_KERNELS:
            for t in THREADS:
                path = f"{r1}/r1_roofline_{kernel}_{plan_tag}_t{t}.json"
                doc = read_json_optional(path)
                if doc is None:
                    continue  # no measured file for this cell: OMIT (nullopt)
                val = doc["kernels"][kernel]["in_gb_per_s"]
                e.emit(f"cpu.t{t}.{pattern}.{kernel}_gbps", val, path)

        # second-pass reuse: convert_f32_f16 runs on the already-gathered
        # contiguous buffer regardless of the first pass's access pattern.
        for t in THREADS:
            if "convert_f32_f16" in contig_val[t]:
                src = f"{r1}/r1_roofline_convert_f32_f16_t{t}.json"
                e.emit(f"cpu.t{t}.{pattern}.convert_f32_f16_gbps",
                       contig_val[t]["convert_f32_f16"], src,
                       note="second pass is contiguous by construction")
        # pack_s8_s4: no valid epyc source (see module docstring NOTE) ->
        # omitted for contiguous AND, by construction, for every reused
        # pattern too.

    # HBM roofline + hiding ratio.
    r4_path = f"{RESULTS}/r4_hiding_ratio_epyc_2080ti.json"
    r4 = must_read_json(r4_path)
    by_n = r4["by_n"]
    ns = sorted(int(k) for k in by_n)
    n_head, n_m = ns[0], ns[-1]  # headline (bw/ratio) vs m-table reference N
    copy_head = by_n[str(n_head)]["copy_f32"]["gb_per_s"]
    copy_m = by_n[str(n_m)]["copy_f32"]["gb_per_s"]
    relocate_m = by_n[str(n_m)]["relocate_naive_f32"]["gb_per_s"]
    transpose_m = by_n[str(n_m)]["transpose_smem_padded"]["gb_per_s"]

    bw_gbps = int(copy_head)  # measured HBM ceiling, whole GB/s
    m_blocked = round(copy_m / relocate_m, 2)
    m_single = round(copy_m / transpose_m, 2)
    e.emit("hbm.bw_gbps", bw_gbps, r4_path,
           note=f"copy_f32 ceiling at N={n_head}")
    e.emit("hbm.m.contiguous", 1.0, r4_path, note="copy_f32 by construction")
    e.emit("hbm.m.blocked", m_blocked, r4_path,
           note=f"copy_f32/relocate_naive_f32 at N={n_m}")
    e.emit("hbm.m.single_element", m_single, r4_path,
           note=f"copy_f32/transpose_smem_padded at N={n_m} -- T1's exact-"
                "transpose SMEM path")
    e.emit("hbm.m.tiled", m_blocked, r4_path,
           note="no separate tiled GPU-receive kernel measured; reuses "
                "relocate_naive_f32's ratio")
    e.emit("hiding.ratio", round(copy_head / h2d, 1), r4_path,
           note=f"copy_f32(N={n_head}) / pcie.h2d_gbps")

    # pipeline.chunks_per_buffer deliberately NOT emitted here (see
    # SOURCES docstring / module-level note): activating the Overlapped
    # fill/drain term would re-score the frozen V3 prediction report.

    # Recv-kernel multipliers (issue #109/CM1): Method A's post-DMA GPU
    # decompress kernels. f16/s8 derive from the committed V2 rsweep's
    # per-chunk gpu_recv_ms ("from committed artifacts where possible"):
    # traffic = r*S read + S written; min gpu_recv_ms over the chunk
    # sweep (largest chunk = least launch-diluted, closest to isolated);
    # divided into copy_f32 at the m-table N (same rule as hbm.m.*).
    rsweep_path = f"{RESULTS}/v2_isa_gen3_rsweep_avx2_epyc7351-2080ti.csv"
    rrows = [row for row in load_csv_rows(rsweep_path, e.machine)
             if row["method"] == "a" and row["transform"] == "quant"]
    for key, r_wire in (("recv.m.convert_f16_f32", 0.5),
                        ("recv.m.dequant_s8_f32", 0.25)):
        cells = [row for row in rrows if float(row["r"]) == r_wire]
        if not cells:
            sys.exit(f"error: no method=a quant r={r_wire} rows in "
                     f"{rsweep_path}")
        best = min(cells, key=lambda row: float(row["gpu_recv_ms"]))
        s = source_bytes(int(best["N"]))
        bw = (1.0 + r_wire) * s / (float(best["gpu_recv_ms"]) * 1e-3) / 1e9
        e.emit(key, round(copy_m / bw, 2), rsweep_path,
               note=f"copy_f32(N={n_m})/recv BW; traffic=(1+{r_wire})*S, "
                    f"min gpu_recv_ms over chunks, N={best['N']} (issue #109)")
    # s4 has no committed Gen3 measurement -> the CM1 targeted run.
    emit_recv_from_cm1_run(
        e, f"{RESULTS}/cm1_recv_kernel_bw_epyc_2080ti.json",
        ["unpack_dequant_s4"])

    # Multi-GPU aggregate delivery.
    pair01_path = f"{RESULTS}/m0_multigpu_h2d_pair01.json"
    all4_path = f"{RESULTS}/m0_multigpu_h2d_all4.json"
    pair01 = must_read_json(pair01_path)
    all4 = must_read_json(all4_path)
    e.emit("multigpu.delivery_gbps.k2",
           round(pair01["h2d"]["aggregate_gbps"], 2), pair01_path)
    e.emit("multigpu.delivery_gbps.k4",
           round(all4["h2d"]["aggregate_gbps"], 2), all4_path)

    # Prefold cold-allocation cost.
    v4_path = f"{RESULTS}/v4_scatter_n8192_epyc_2080ti.json"
    v4 = must_read_json(v4_path)
    reuse_k1 = sorted(
        (r for r in v4["reuse_rows"] if r["mode"] == "reuse" and r["K"] == 1),
        key=lambda r: r["n_reuse"])
    if not reuse_k1:
        sys.exit(f"error: no mode=reuse, K=1 rows in {v4_path}")
    row = reuse_k1[0]
    n_v4 = v4["config"]["N"]
    v4_r = 0.25  # quantize scatter's dtype ratio (K=1 reuse rows); not in
                 # the JSON's config, so named here rather than folded into
                 # a bare byte constant (M2: r*S derives from config.N).
    r_times_s_bytes = v4_r * source_bytes(n_v4)
    r_times_s_gib = r_times_s_bytes / 2 ** 30
    alloc_ms_per_gib = (row["t_prefold_cold_ms"] - row["t_transform_ms"]) \
        / r_times_s_gib
    e.emit("prefold.alloc_ms_per_gib", round(alloc_ms_per_gib, 2), v4_path,
           note=f"(t_prefold_cold_ms-t_transform_ms)/(r*S GiB), reuse rows, "
                f"K=1, r*S = {v4_r}*config.N^2*4 (N={n_v4}) = "
                f"{r_times_s_bytes / 2**20:g} MiB")

    # Overhead intercepts.
    nsweep_path = f"{RESULTS}/v1_gen3_nsweep_epyc_2080ti.csv"
    fit_overheads(e, nsweep_path)

    # Strategy thresholds: P2 seed, no small-size sweep exists yet.
    emit_strategy_defaults(e)


# ------------------------------------------------------------- gen4 box ---

def build_gen4(e):
    r2 = f"{RESULTS}/r2_rooflines"

    gate_path = f"{RESULTS}/v1_gen4_gate_report.txt"
    h2d = must_read_gate_report(gate_path, e.machine)
    e.emit("pcie.h2d_gbps", h2d, gate_path)

    PLAN_TO_PATTERN = {"identity": "contiguous", **GATHER_PLAN_TO_PATTERN}

    # I2 fix: pin N=8192 uniformly across every plan and BOTH thread
    # counts, rather than "largest N wins" (only t8 has N=16384 files;
    # comparing t1@8192 vs t8@16384 produced a spurious V-cache-residency
    # inversion). N=8192 exists for all four plans at both t1 and t8.
    GEN4_ROOFLINE_N = 8192
    n_pin_note = (f"N pinned at {GEN4_ROOFLINE_N} across all plans/threads "
                  "(I2 fix: avoids a t1-vs-t8 N mismatch/V-cache-residency "
                  "inversion; matches epyc's single-N r1 rooflines)")

    plan_files = {}  # (plan, t) -> (path, doc)
    for plan_tag in PLAN_TO_PATTERN:
        for t in THREADS:
            path = f"{r2}/{plan_tag}_n{GEN4_ROOFLINE_N}_t{t}.json"
            if not Path(path).exists():
                sys.exit(f"error: missing required gen4 roofline source "
                         f"(pinned N={GEN4_ROOFLINE_N}): {path}")
            plan_files[(plan_tag, t)] = (path, must_read_json(path))

    def pf(plan_tag, t):
        """Guarded plan_files accessor (M4): a clear error instead of a
        bare KeyError if a (plan, threads) cell was never populated."""
        key = (plan_tag, t)
        if key not in plan_files:
            sys.exit(f"error: no roofline source loaded for plan={plan_tag!r} "
                     f"threads={t} (expected {r2}/{plan_tag}_n"
                     f"{GEN4_ROOFLINE_N}_t{t}.json)")
        return plan_files[key]

    contig_val = {t: {} for t in THREADS}
    for t in THREADS:
        path, doc = pf("identity", t)
        for kernel in ("contig_read", "quantize_pack", "convert_f32_f16",
                       "pack_s8_s4"):
            val = doc["kernels"][kernel]["in_gb_per_s"]
            e.emit(f"cpu.t{t}.contiguous.{kernel}_gbps", val, path,
                   note=n_pin_note)
            contig_val[t][kernel] = val

    for plan_tag, pattern in GATHER_PLAN_TO_PATTERN.items():
        for t in THREADS:
            path, doc = pf(plan_tag, t)
            for kernel in ("gather_f32", "gather_quantize"):
                val = doc["kernels"][kernel]["in_gb_per_s"]
                e.emit(f"cpu.t{t}.{pattern}.{kernel}_gbps", val, path,
                       note=n_pin_note)
        for t in THREADS:
            src_path, _ = pf("identity", t)
            e.emit(f"cpu.t{t}.{pattern}.convert_f32_f16_gbps",
                   contig_val[t]["convert_f32_f16"], src_path,
                   note="second pass is contiguous by construction")
            e.emit(f"cpu.t{t}.{pattern}.pack_s8_s4_gbps",
                   contig_val[t]["pack_s8_s4"], src_path,
                   note="second pass is contiguous by construction")

    # No Gen4 HBM/hiding sweep exists (R4 ran on the 2080 Ti only): proxy
    # the 2080 Ti's computed values through, honestly labelled.
    r4_path = f"{RESULTS}/r4_hiding_ratio_epyc_2080ti.json"
    r4 = must_read_json(r4_path)
    by_n = r4["by_n"]
    ns = sorted(int(k) for k in by_n)
    n_head, n_m = ns[0], ns[-1]
    copy_head = by_n[str(n_head)]["copy_f32"]["gb_per_s"]
    copy_m = by_n[str(n_m)]["copy_f32"]["gb_per_s"]
    relocate_m = by_n[str(n_m)]["relocate_naive_f32"]["gb_per_s"]
    transpose_m = by_n[str(n_m)]["transpose_smem_padded"]["gb_per_s"]
    proxy_note = ("proxy: no Gen4 HBM sweep exists (R4 ran on 2080 Ti); B "
                  "is link-bound whenever m < ratio, so this only matters "
                  "if m/BW_hbm exceeds 1/BW_link")
    e.emit("hbm.bw_gbps", int(copy_head), r4_path, note=proxy_note)
    e.emit("hbm.m.contiguous", 1.0, r4_path, note=proxy_note)
    e.emit("hbm.m.blocked", round(copy_m / relocate_m, 2), r4_path,
           note=proxy_note)
    e.emit("hbm.m.single_element", round(copy_m / transpose_m, 2), r4_path,
           note=proxy_note)
    e.emit("hbm.m.tiled", round(copy_m / relocate_m, 2), r4_path,
           note=proxy_note)
    # multigpu.*, hiding.ratio, prefold.*: no Gen4 data -- omitted.

    # pipeline.chunks_per_buffer deliberately NOT emitted here (see
    # SOURCES docstring / module-level note): activating the Overlapped
    # fill/drain term would re-score the frozen V3 prediction report.

    # Recv-kernel multipliers (issue #109/CM1): the committed Gen4
    # pipeline CSVs are too noisy for an R4-style derivation (in-pipeline
    # event timings collapse to 23-52 GB/s on some cells) and no Gen4
    # copy_f32 ceiling exists at all -- ALL THREE keys wait for the
    # targeted run (runbook: bench/rtrack/README.md, issue #109).
    emit_recv_from_cm1_run(
        e, f"{RESULTS}/cm1_recv_kernel_bw_7800x3d_4070tis.json",
        ["convert_f16_f32", "dequant_s8_f32", "unpack_dequant_s4"])

    nsweep_path = f"{RESULTS}/v1_gen4_matrix_nsweep_7800x3d_4070tis.csv"
    nsweep_rerun_path = (f"{RESULTS}/v1_gen4_matrix_nsweep_rerun_"
                         "7800x3d_4070tis.csv")
    fit_overheads(e, nsweep_path, rerun_path=nsweep_rerun_path)

    emit_strategy_defaults(e)


# --------------------------------------------------------- shared pieces --

def source_bytes(n, elem_bytes=4):
    """Total f32-source byte count for an NxN transform: S = N^2 * 4."""
    return n * n * elem_bytes


def _quant_rows(csv_path, machine):
    rows = load_csv_rows(csv_path, machine)
    return [r for r in rows
            if r["transform"] == "quant"
            and (r.get("variant") or "matrix") == "matrix"]


def _best_chunk(rows, method, n):
    cand = [r for r in rows if r["method"] == method and int(r["N"]) == n]
    if not cand:
        return None
    return min(cand, key=lambda r: float(r["median_ms"]))


def _merged_endpoint(orig_rows, rerun_rows, method, n):
    """The (row, which_file) for (method, N): the stabler (lower IQR) of
    the original and rerun files' independently-recomputed best-chunk
    rows, per the pre-declared merge rule in
    docs/r2-exp2-gen4-crossover.md:48-58. Falls back to whichever file
    actually has rows for this point."""
    o = _best_chunk(orig_rows, method, n)
    r = _best_chunk(rerun_rows, method, n) if rerun_rows is not None else None
    if o is None and r is None:
        return None
    if o is None:
        return r, "rerun"
    if r is None:
        return o, "original"
    o_iqr = float(o["iqr_over_median_pct"])
    r_iqr = float(r["iqr_over_median_pct"])
    return (r, "rerun") if r_iqr < o_iqr else (o, "original")


def fit_overheads(e, csv_path, rerun_path=None):
    """overhead.{a,b}_ms: intercept of a two-point affine fit (in source
    bytes S) over the T3 (transform=='quant') rows, best chunk per
    (method, N), at the min and max N present; clamp >= 0.

    When `rerun_path` is given (I1 fix, gen4 only), the best-chunk row
    per (method, N) is the stabler (lower IQR) of the two files'
    independently-recomputed best-chunk rows -- the merge rule
    pre-declared in docs/r2-exp2-gen4-crossover.md:48-58 -- and the N
    span is the union of both files' N values (so the endpoints reach
    whatever the ORIGINAL file's full sweep covers, not just the
    (possibly partial) rerun file's span)."""
    orig_rows = _quant_rows(csv_path, e.machine)
    if not orig_rows:
        sys.exit(f"error: no transform=quant rows in {csv_path}")
    rerun_rows = _quant_rows(rerun_path, e.machine) if rerun_path else None

    all_rows = orig_rows + (rerun_rows or [])
    ns = sorted({int(r["N"]) for r in all_rows})
    n_lo, n_hi = ns[0], ns[-1]

    for method, key in (("a", "overhead.a_ms"), ("b_fair", "overhead.b_ms")):
        points, which = [], []
        for n in (n_lo, n_hi):
            result = _merged_endpoint(orig_rows, rerun_rows, method, n)
            if result is None:
                sys.exit(f"error: no method={method} N={n} rows in "
                         f"{csv_path}" + (f" or {rerun_path}" if rerun_path
                                          else ""))
            row, src = result
            points.append((source_bytes(n), float(row["median_ms"])))
            which.append(src)
        (s1, t1), (s2, t2) = points
        slope = (t2 - t1) / (s2 - s1)
        intercept = max(0.0, t1 - slope * s1)
        if rerun_path is None:
            source = csv_path
            note = f"two-point fit, T3 quant, best chunk, N={n_lo},{n_hi}"
        else:
            source = f"{csv_path} + {rerun_path}"
            note = (f"two-point fit, T3 quant, best chunk, MERGED original+"
                    f"rerun (stabler-IQR preference per docs/r2-exp2-gen4-"
                    f"crossover.md:48-58), N={n_lo}({which[0]}),{n_hi}"
                    f"({which[1]})")
        e.emit(key, round(intercept, 4), source, note=note)


def emit_strategy_defaults(e):
    seed_note = "P2 default seed (no small-size sweep exists yet)"
    e.emit("strategy.single_thread_max_bytes", 262144,
           "libreloc/src/Bind.cpp", note=seed_note)
    e.emit("strategy.multi_thread_max_bytes", 268435456,
           "libreloc/src/Bind.cpp", note=seed_note)


def emit_recv_from_cm1_run(e, path, kernels):
    """recv.m.* from a CM1 targeted isolated-kernel run (issue #109),
    R4-style: m = copy_f32 / kernel BW at the largest measured N, both
    from the SAME run (session self-consistency: a different session's
    ceiling would bias every m derived against it). Artifact absent ->
    keys omitted (loud omission, the pack_s8_s4 precedent)."""
    doc = read_json_optional(path)
    if doc is None:
        return
    by_n = doc["by_n"]
    n_ref = str(max(int(k) for k in by_n))
    copy = by_n[n_ref]["copy_f32"]["gb_per_s"]
    for kern in kernels:
        bw = by_n[n_ref][kern]["gb_per_s"]
        e.emit(f"recv.m.{kern}", round(copy / bw, 2), path,
               note=f"copy_f32/{kern} at N={n_ref}, same-run ceiling")


BUILDERS = {
    "epyc7351-2080ti": build_epyc,
    "7800x3d-4070tis": build_gen4,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--machine", required=True, choices=sorted(BUILDERS))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    e = Emitter(args.machine)
    BUILDERS[args.machine](e)

    Path(args.out).write_text(e.text(), encoding=ENCODING)
    print(args.out)


if __name__ == "__main__":
    main()
