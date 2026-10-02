#!/usr/bin/env python3
"""Render #189 qualification measurements; never substitute profiled timings."""
import argparse
import base64
from collections import defaultdict
import html
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

MODES = ["pageable", "pinned", "auto_default", "auto_configured", "torch"]
LABELS = ["Forced pageable", "Forced pinned", "Auto, unconfigured", "Auto, 8 MiB configured", "Torch CPU transpose"]
COLORS = ["#3274a1", "#d17a22", "#8092a4", "#268773", "#7758a6"]


def summarize_sweep(data):
    grouped = defaultdict(list)
    for row in data["rows"]:
        grouped[(row["bytes"], row["variant"], row["reuse"])].append(row)
    result = {}
    for (wire, mode, reuse), rows in grouped.items():
        assert len(rows) == data["rounds"]
        for phase in ["first", "repeated"]:
            values = [statistics.median(c["ms"] for c in row["calls"] if c["phase"] == phase) for row in rows]
            result[(wire, mode, reuse, phase)] = dict(median=statistics.median(values),
                low=min(values), high=max(values), round_medians=values)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    args = p.parse_args()
    root = args.results
    sweeps = {d: json.loads((root / f"sweep-{d}.json").read_text()) for d in ("h2d", "d2h")}
    assert all(d["correct"] for d in sweeps.values())
    summaries = {d: summarize_sweep(data) for d, data in sweeps.items()}
    sizes = sorted({row["bytes"] for row in sweeps["h2d"]["rows"]})
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white"})
    pdf = PdfPages(root / "figures.pdf")
    def save(fig, name):
        pdf.savefig(fig, bbox_inches="tight")
        for extension in ("png", "svg"):
            fig.savefig(root / f"{name}.{extension}", dpi=170, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), layout="constrained")
    for axis_row, (direction, summary) in zip(axes, summaries.items()):
        for ax, (reuse, phase, title) in zip(axis_row, [(True, "first", "First use of new owner"),
                (True, "repeated", "Warm retained buffers"), (False, "repeated", "Per-call allocation")]):
            for mode, label, color in zip(MODES, LABELS, COLORS):
                values = [summary[(size, mode, reuse if mode != "torch" else False, phase)] for size in sizes]
                x = np.array(sizes) / (1 << 20)
                y = np.array([v["median"] for v in values])
                ax.plot(x, y, marker="o", ms=3, color=color, label=label,
                        linestyle="--" if mode.startswith("auto") else "-")
                ax.fill_between(x, [v["low"] for v in values], [v["high"] for v in values], color=color, alpha=.10)
            ax.axvline(8, color="#777", linestyle=":", lw=1)
            ax.set(xscale="log", yscale="log", xlabel="Wire size (MiB)", ylabel="Completed transfer (ms)",
                   title=f"{direction.upper()}: {title}")
            ax.grid(alpha=.15)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5)
    fig.suptitle("Pinning gains depend on reuse: keep first-allocation costs visible", fontsize=17)
    save(fig, "pinning-lifecycle")

    traces = {name: json.loads((root / "traces" / f"{name}.analysis.json").read_text())
              for name in ["configured-4", "configured-1", "unconfigured-4"]}
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), layout="constrained")
    overlap_rows = []
    chunk_colors = ["#2878a4", "#b66b27", "#2c8c6b", "#865eac"]
    for ax, (name, trace) in zip(axes, traces.items()):
        req = trace["requests"][1]  # fixed middle request; not chosen by fastest/most overlap
        threads = sorted({g["thread"] for c in req["chunks"] for g in c["gather"]})
        streams = sorted({c["stream"] for c in req["chunks"]})
        lanes = {t: i for i, t in enumerate(threads)}
        origin = req["start_ns"]
        for chunk in req["chunks"]:
            color = chunk_colors[chunk["chunk"] % len(chunk_colors)]
            for work in chunk["gather"]:
                ax.broken_barh([((work["start_ns"] - origin) / 1e6,
                                 (work["end_ns"] - work["start_ns"]) / 1e6)],
                               (lanes[work["thread"]] - .35, .7), facecolors=color)
            lane = len(threads) + streams.index(chunk["stream"])
            begin = (chunk["dma_start_ns"] - origin) / 1e6
            duration = (chunk["dma_end_ns"] - chunk["dma_start_ns"]) / 1e6
            ax.broken_barh([(begin, duration)], (lane - .35, .7), facecolors=color)
            ax.text(begin + duration / 2, lane, str(chunk["chunk"]), color="white", ha="center", va="center", fontsize=8)
        overlap = [sum(c["next_chunk_work_overlap_ns"] for c in r["chunks"]) / 1e6 for r in trace["requests"]]
        overlap_rows.append(dict(case=name, status=trace["status"], overlap_ms_by_request=overlap,
                                 overlap_observed=[r["observed_overlap"] for r in trace["requests"]]))
        kind = "pinned" if name.startswith("configured") else "pageable"
        ax.set_yticks(range(len(threads) + len(streams)),
            [f"CPU worker {i}" for i in range(len(threads))] + [f"GPU H2D stream {s}" for s in streams])
        ax.invert_yaxis()
        ax.set(xlabel="Milliseconds from annotated request start", title=f"{name}: {kind}; gather(n+1)/DMA(n) overlap {overlap[1]:.3f} ms",
               xlim=(0, (req["end_ns"] - origin) / 1e6))
        ax.grid(axis="x", alpha=.15)
    fig.suptitle("Actual GPU DMA correlated with native CPU gather work (16 MiB, four equal chunks)", fontsize=15)
    save(fig, "pinning-overlap")

    matrix = json.loads((root / "matrix/summary.json").read_text())
    assert len(matrix) == 48, "complete three rounds of four policies and four workloads"
    variants = ["auto_default", "auto_configured", "pinned", "pageable"]
    workloads = ["dlrm", "gnn", "llm", "moe"]
    workload_summary = {}
    for example in workloads:
        workload_summary[example] = {}
        for variant in variants:
            runs = [json.loads(path.read_text()) for path in (root / "matrix").glob(f"{variant}-{example}-*.json")]
            assert len(runs) == 3 and all(r["ok"] for r in runs)
            columns = {}
            for who in ["sym", "torch"]:
                all_calls = [sum(sum(paths[who]) for paths in run["raw_transfer_samples_ms"].values()) for run in runs]
                repeated = [run["summary"]["steady_transfer_ms"][who] for run in runs]
                first = [sum(paths[who][0] for paths in run["raw_transfer_samples_ms"].values()) for run in runs]
                columns[who] = {phase: dict(median=statistics.median(values), low=min(values), high=max(values), samples=values)
                    for phase, values in [("all", all_calls), ("after_first", repeated), ("first", first)]}
            workload_summary[example][variant] = columns
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), layout="constrained")
    for column, example in enumerate(workloads):
        data = workload_summary[example]
        for ax, phase, title in [(axes[0, column], "after_first", "After first of each kind"),
                                 (axes[1, column], "all", "All calls, including setup")]:
            values = [data[v]["sym"][phase] for v in variants] + [data["auto_default"]["torch"][phase]]
            y = np.array([v["median"] for v in values])
            bars = ax.barh(range(5), y, color=[COLORS[2], COLORS[3], COLORS[1], COLORS[0], COLORS[4]])
            ax.errorbar(y, range(5), xerr=[y - [v["low"] for v in values], [v["high"] for v in values] - y],
                        fmt="none", ecolor="#333", capsize=2)
            for index, value in enumerate(values):
                ax.text(value["high"] + max(v["high"] for v in values) * .02, index,
                        f'{value["median"]:.1f}', va="center", fontsize=8)
            ax.set_yticks(range(5), ["Auto default", "Auto 8 MiB", "Pinned", "Pageable", "Torch"])
            ax.invert_yaxis()
            ax.set(xlabel="Completed transfer total (ms)", title=f"{example.upper()} — {title}", xlim=(0, max(v["high"] for v in values) * 1.25))
            ax.grid(axis="x", alpha=.12)
    fig.suptitle("Unchanged #162 workloads: policy comparison on the latest implementation", fontsize=16)
    save(fig, "pinning-workloads")
    pdf.close()

    summary = dict(overlap=overlap_rows, workloads=workload_summary,
                   sweeps={direction: [dict(bytes=key[0], variant=key[1], reuse=key[2], phase=key[3], **value)
                            for key, value in data.items()] for direction, data in summaries.items()})
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    def embedded(name):
        data = base64.b64encode((root / f"{name}.png").read_bytes()).decode()
        return f'<img src="data:image/png;base64,{data}" alt="{name}"><p><a href="{name}.png">PNG</a> · <a href="{name}.svg">SVG</a></p>'
    table_rows = []
    for direction, values in summaries.items():
        for wire in (8 << 20, 16 << 20, 32 << 20):
            fields = [values[(wire, mode, True, phase)]["median"] for phase in ("first", "repeated") for mode in ("pinned", "pageable")]
            table_rows.append(f'<tr><th>{direction.upper()} {wire >> 20} MiB</th>' + ''.join(f'<td>{v:.3f}</td>' for v in fields) + '</tr>')
    overlap_table = ''.join('<tr><th>' + row['case'] + '</th><td>' + ', '.join(f'{v:.3f}' for v in row['overlap_ms_by_request']) + '</td><td>' + row['status'] + '</td></tr>' for row in overlap_rows)
    document = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Sym #189 pinning qualification</title>
<style>body{font:17px/1.6 system-ui;max-width:1400px;margin:32px auto;padding:0 24px;color:#18303c}h1,h2{line-height:1.25}img{width:100%;height:auto}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ddd;padding:9px;text-align:right}th:first-child{text-align:left}.note{background:#edf5f2;padding:18px;border-left:4px solid #268773}a{color:#246185}</style>
<h1>Sym #189: pinning by an explicitly configured size gate</h1>
<p class="note">Unconfigured auto stays pageable. Explicit thresholds enable retained pinned staging only within the caller's calibrated scope. The evidence separates first use, warmed retention and per-call allocation; a warm win does not guarantee a first-call win. This qualifies the initial gate, not an adaptive latency model.</p>
<p>EPYC 7351 / RTX 2080 Ti (GPU 0), CUDA 12.6, Torch 2.14.0+cu126, eight Torch/gather threads, one interop thread, CPU affinity 4–7,20–23. GPU clocks are not locked. Three shuffled rounds per size/policy, ten repeated samples per round, two intervening warmups; all samples and first calls are preserved. The size sweep uses a 512-column FP32 transpose. Source inputs, compiler and CUDA initialization precede timing. Every output is checked outside the timer.</p>
<h2>Completed transfer latency and allocation lifecycle</h2>
<p>Timing includes prepare/bind, fresh output allocation, CPU transform, transfer and completion. Input creation and result destruction are outside timing. First use is a new Sym owner in an initialized process: global Torch/CUDA caches can already be warm. Torch performs CPU transpose then H2D, or D2H then CPU transpose; it is not a benchmark of every possible Torch/GPU transpose placement. Lines are medians of three round medians; shading shows their full range, not a confidence interval. The 8 MiB marker is an experiment setting, not a runtime default.</p>'''
    document += embedded("pinning-lifecycle")
    document += '<table><tr><th>Transfer</th><th>First pinned</th><th>First pageable</th><th>Warm pinned</th><th>Warm pageable</th></tr>' + ''.join(table_rows) + '</table>'
    document += '''<h2>CPU transpose and actual DMA overlap</h2><p>NVTX-enabled runtime, Nsight Systems CUDA/NVTX capture. Native per-worker gather intervals are correlated through CUDA API correlation IDs to actual GPU H2D activity. Every case uses four 4 MiB chunks. The chart shows the fixed middle request of three captured requests. Per-worker overlap is unioned within each chunk before aggregation; it is not multiplied by worker count. Gather spans measure wall time and can include CPU preemption. These profiled durations are separate from the uninstrumented latency sweep.</p>'''
    document += embedded("pinning-overlap")
    document += '<table><tr><th>Case</th><th>Overlap ms, all three requests</th><th>Result</th></tr>' + overlap_table + '</table>'
    document += '''<p>The single-buffer pinned control must show no gather(n+1)/DMA(n) overlap; the four-buffer configured-auto path must show it. Pageable behavior is characterized, not assumed: some overlap can occur despite driver staging/blocking. This does not guarantee overlap on every driver or workload. The typed CPU reference still transforms a whole buffer before copying; direct dense uploads have no Sym CPU gather to overlap.</p>
<h2>Unchanged workload bodies, four pinning policies</h2><p>48 fresh-process runs: four workloads × four policies × three rounds. All retain direct typed resources explicitly; DLRM/GNN retain frontend AUTO owners. Original bodies, shapes, GPU computation and correctness checks are unchanged. Each process runs Torch before Sym, as in #162; policy/workload process order is shuffled. Error bars span the three process totals. The plotted Torch column uses the paired auto-default runs; every policy's paired Torch values are in summary.json. First calls can include graph compilation and setup. Neither row includes model computation, so these are not whole-model throughput measurements.</p>'''
    document += embedded("pinning-workloads")
    document += '''<h2>Scope and reproduction</h2><p>Calibration applies to the measured configuration only. Shape, direction, CPU/NUMA placement, threads, budgets, warm capacity and copy schedule can change the crossover. In particular, cold allocation and per-call pinned allocation can lose even when warm retention wins. The runtime cannot detect hardware-profile mismatches; outside a qualified profile, omit the threshold. No universal cutoff or end-to-end model speedup is claimed.</p><p><a href="summary.json">Numerical summary</a> · <a href="sweep-h2d.json">H2D raw calls</a> · <a href="sweep-d2h.json">D2H raw calls</a> · <a href="figures.pdf">All figures (PDF)</a> · <a href="README.md">Commands, provenance and validation</a>. The download is standalone: all chart images are embedded. Separate PNG/SVG files are suitable for issue and PR comments.</p></html>'''
    (root / "report.html").write_text(document)


if __name__ == "__main__":
    main()
