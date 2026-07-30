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

  7800x3d-4070tis: same key shapes, sourced from
    bench/results/v1_gen4_gate_report.txt, bench/results/r2_rooflines/*.json
    (per-plan files carrying ALL kernels; identity/blocked/transpose/nchw
    -> contiguous/blocked/single_element/tiled; pack_s8_s4 measured
    directly per-plan here, no reuse needed for it), and
    bench/results/v1_gen4_matrix_nsweep_rerun_7800x3d_4070tis.csv for
    overheads. No multigpu or HBM sweep exists on this box: hbm.bw_gbps
    and hbm.m.* are PROXIED from the 2080 Ti computation with an explicit
    note; multigpu.*, hiding.ratio and prefold.* are omitted outright.

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

# ---------------------------------------------------------------- helpers --

def must_read_text(path):
    p = Path(path)
    if not p.exists():
        sys.exit(f"error: missing required source file: {path}")
    return p.read_text()


def must_read_json(path):
    return json.loads(must_read_text(path))


def read_json_optional(path):
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def load_csv_rows(path):
    text = must_read_text(path)
    return list(csv.DictReader(l for l in text.splitlines() if not l.startswith("#")))


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
    h2d = parse_pinned_h2d(must_read_text(gate_path), gate_path)
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
    r_times_s_bytes = 64 * 2 ** 20  # r*S at K=1: N=8192, r=0.25 -> 64 MiB
    r_times_s_gib = r_times_s_bytes / 2 ** 30
    alloc_ms_per_gib = (row["t_prefold_cold_ms"] - row["t_transform_ms"]) \
        / r_times_s_gib
    e.emit("prefold.alloc_ms_per_gib", round(alloc_ms_per_gib, 2), v4_path,
           note="(t_prefold_cold_ms-t_transform_ms)/(r*S GiB), reuse rows, "
                "K=1, r*S=64 MiB")

    # Overhead intercepts.
    nsweep_path = f"{RESULTS}/v1_gen3_nsweep_epyc_2080ti.csv"
    fit_overheads(e, nsweep_path)

    # Strategy thresholds: P2 seed, no small-size sweep exists yet.
    emit_strategy_defaults(e)


# ------------------------------------------------------------- gen4 box ---

def build_gen4(e):
    r2 = f"{RESULTS}/r2_rooflines"

    gate_path = f"{RESULTS}/v1_gen4_gate_report.txt"
    h2d = parse_pinned_h2d(must_read_text(gate_path), gate_path)
    e.emit("pcie.h2d_gbps", h2d, gate_path)

    PLAN_TO_PATTERN = {"identity": "contiguous", **GATHER_PLAN_TO_PATTERN}
    KERNELS_BY_PATTERN = {
        "contiguous": ("contig_read", "quantize_pack", "convert_f32_f16",
                       "pack_s8_s4"),
        "blocked": ("gather_f32", "gather_quantize"),
        "single_element": ("gather_f32", "gather_quantize"),
        "tiled": ("gather_f32", "gather_quantize"),
    }

    contig_val = {t: {} for t in THREADS}
    plan_files = {}  # (plan, t) -> (path, doc)
    for plan_tag in PLAN_TO_PATTERN:
        for t in THREADS:
            candidates = sorted(
                Path(r2).glob(f"{plan_tag}_n*_t{t}.json"),
                key=lambda p: must_read_json(p)["config"]["N"])
            if not candidates:
                continue
            path = candidates[-1]  # largest N wins
            plan_files[(plan_tag, t)] = (str(path), must_read_json(path))

    for t in THREADS:
        path, doc = plan_files[("identity", t)]
        for kernel in ("contig_read", "quantize_pack", "convert_f32_f16",
                       "pack_s8_s4"):
            val = doc["kernels"][kernel]["in_gb_per_s"]
            e.emit(f"cpu.t{t}.contiguous.{kernel}_gbps", val, path)
            contig_val[t][kernel] = val

    for plan_tag, pattern in GATHER_PLAN_TO_PATTERN.items():
        for t in THREADS:
            path, doc = plan_files[(plan_tag, t)]
            for kernel in ("gather_f32", "gather_quantize"):
                val = doc["kernels"][kernel]["in_gb_per_s"]
                e.emit(f"cpu.t{t}.{pattern}.{kernel}_gbps", val, path)
        for t in THREADS:
            src_path, _ = plan_files[("identity", t)]
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

    nsweep_path = (f"{RESULTS}/v1_gen4_matrix_nsweep_rerun_7800x3d_4070tis"
                   ".csv")
    fit_overheads(e, nsweep_path)

    emit_strategy_defaults(e)


# --------------------------------------------------------- shared pieces --

def fit_overheads(e, csv_path):
    """overhead.{a,b}_ms: intercept of a two-point affine fit (in source
    bytes S) over the T3 (transform=='quant') rows, best chunk per
    (method, N), at this file's min and max N; clamp >= 0."""
    rows = load_csv_rows(csv_path)
    rows = [r for r in rows
            if r["transform"] == "quant"
            and (r.get("variant") or "matrix") == "matrix"]
    if not rows:
        sys.exit(f"error: no transform=quant rows in {csv_path}")
    ns = sorted({int(r["N"]) for r in rows})
    n_lo, n_hi = ns[0], ns[-1]
    for method, key in (("a", "overhead.a_ms"), ("b_fair", "overhead.b_ms")):
        points = []
        for n in (n_lo, n_hi):
            cand = [r for r in rows
                    if r["method"] == method and int(r["N"]) == n]
            if not cand:
                sys.exit(f"error: no method={method} N={n} rows in "
                         f"{csv_path}")
            best = min(cand, key=lambda r: float(r["median_ms"]))
            points.append((n * n * 4, float(best["median_ms"])))
        (s1, t1), (s2, t2) = points
        slope = (t2 - t1) / (s2 - s1)
        intercept = max(0.0, t1 - slope * s1)
        e.emit(key, round(intercept, 4), csv_path,
               note=f"two-point fit, T3 quant, best chunk, N={n_lo},{n_hi}")


def emit_strategy_defaults(e):
    seed_note = "P2 default seed (no small-size sweep exists yet)"
    e.emit("strategy.single_thread_max_bytes", 262144,
           "libreloc/src/Bind.cpp", note=seed_note)
    e.emit("strategy.multi_thread_max_bytes", 268435456,
           "libreloc/src/Bind.cpp", note=seed_note)


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

    Path(args.out).write_text(e.text())
    print(args.out)


if __name__ == "__main__":
    main()
