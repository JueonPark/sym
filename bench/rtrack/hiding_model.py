#!/usr/bin/env python3
"""R4 / EXP-4 (issue #85): the hiding-ratio model over hiding_ratio.cu's
measured kernel bandwidths.

Model: a Method-B GPU transform moves m tensor-passes through HBM, so its
time is m*S/HBM_BW; the PCIe transfer that feeds it takes S/PCIe_BW. The
transform hides under the transfer iff

    m < ratio = HBM_BW / PCIe_BW.

HBM_BW is the measured copy_f32 ceiling; the per-kernel multiplier is
m = HBM_BW / kernel_BW on the same read+write traffic basis (so copy_f32
itself lands at m = 1.0 by construction, i.e. one full HBM round-trip).
PCIe_BW defaults to the M0/R1 measured Gen3 pinned figure (13.06 GB/s).

Validation (the issue's ±20% bar): for each kernel the model predicts an
exposed GPU time in the full Method-B pipeline of max(0, t_kernel - t_pcie)
per tensor; --pipeline-csv cross-checks that against the measured
gpu_kernel_ms / h2d_ms of the matching R0.3 rows.

  python3 bench/rtrack/hiding_model.py --json r4_hiding_ratio.json \
      [--pcie-gbps 13.06] [--pipeline-csv r1_gen3_nsweep.csv]
"""

import argparse
import csv
import json
import sys

# hiding_ratio kernel -> the rtrack transform whose Method-B GPU stage runs
# the IDENTICAL kernel on the IDENTICAL plan, for the ±20% prediction-error
# cross-check. Only transpose_smem_padded qualifies: pipeline T1 (transpose)
# Method B calls relocateF32 on the same rank-2 transpose plan, hitting the
# same padded tiled kernel. relocate_naive here runs on the rank-2 transpose
# (single-element gather) which the pipeline never uses for Method B (T1
# takes the tiled path, T1b's blocked plan is a different access pattern),
# so it is a roofline-floor data point only, not a pipeline match.
KERNEL_TO_TRANSFORM = {
    "transpose_smem_padded": "transpose",
}
VALIDATION_BAR_PCT = 20.0


def load_pipeline(path):
    # (transform, N) -> best row by median_ms, keeping its kernel/h2d split.
    best = {}
    with open(path) as f:
        for row in csv.DictReader(l for l in f if not l.startswith("#")):
            if row["method"] != "b":
                continue
            key = (row["transform"], int(row["N"]))
            ms = float(row["median_ms"])
            if key not in best or ms < best[key][0]:
                best[key] = (ms, float(row["gpu_kernel_ms"]),
                             float(row["h2d_ms"]))
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", required=True)
    ap.add_argument("--pcie-gbps", type=float, default=13.06)
    ap.add_argument("--pipeline-csv", default=None)
    args = ap.parse_args()

    doc = json.load(open(args.json))
    gpu = doc["config"]["gpu"]
    hbm_peak = doc["config"]["hbm_peak_gb_per_s"]
    pipeline = load_pipeline(args.pipeline_csv) if args.pipeline_csv else {}

    print(f"# R4 hiding-ratio model — {gpu}")
    print(f"HBM theoretical peak: {hbm_peak:.0f} GB/s  |  "
          f"PCIe (Gen3 measured): {args.pcie_gbps:.2f} GB/s\n")

    for n_str, kernels in doc["by_n"].items():
        n = int(n_str)
        S = n * n * 4
        hbm = kernels["copy_f32"]["gb_per_s"]  # measured ceiling
        ratio = hbm / args.pcie_gbps
        t_pcie_ms = S / (args.pcie_gbps * 1e9) * 1e3
        print(f"## N={n}  (S={S/2**20:.0f} MiB)")
        print(f"measured HBM ceiling (copy_f32): {hbm:.1f} GB/s "
              f"({100*hbm/hbm_peak:.0f}% of peak)")
        print(f"hiding ratio = HBM/PCIe = {ratio:.1f}  "
              f"(a transform hides iff its multiplier m < {ratio:.1f})")
        print(f"PCIe transfer time for S: {t_pcie_ms:.2f} ms\n")
        print(f"| kernel | GB/s | m = HBM/BW | t_kernel ms | hides? "
              f"(m<ratio) |")
        print("|---|---|---|---|---|")
        for name, k in kernels.items():
            bw = k["gb_per_s"]
            m = hbm / bw if bw > 0 else float("inf")
            t_kernel = k["median_ms"]
            hides = "yes" if m < ratio else "NO"
            print(f"| {name} | {bw:.1f} | {m:.2f} | {t_kernel:.3f} | {hides} |")
        print()

        # Pipeline cross-check. Two claims:
        #  (1) qualitative: a kernel the model says hides shows up in the
        #      Method-B pipeline as gpu_kernel_ms << h2d_ms;
        #  (2) quantitative (the issue's ±20% bar): the isolated kernel time
        #      predicts the pipeline's measured gpu_kernel_ms within ±20%,
        #      for the kernel that is identical in both.
        if pipeline:
            print("pipeline overlap cross-check (Method B, R0.3 CSV):")
            for name, transform in KERNEL_TO_TRANSFORM.items():
                if name not in kernels or (transform, n) not in pipeline:
                    continue
                _, gpu_ms, h2d_ms = pipeline[(transform, n)]
                t_iso = kernels[name]["median_ms"]
                err = 100 * abs(t_iso - gpu_ms) / gpu_ms if gpu_ms > 0 else \
                    float("inf")
                verdict = "PASS" if err <= VALIDATION_BAR_PCT else "FAIL"
                frac = gpu_ms / h2d_ms if h2d_ms > 0 else float("inf")
                print(f"  {name} == pipeline '{transform}': isolated "
                      f"{t_iso:.3f} ms vs pipeline gpu_kernel {gpu_ms:.3f} ms "
                      f"-> prediction error {err:.1f}% (bar ±"
                      f"{VALIDATION_BAR_PCT:.0f}%)  [{verdict}]")
                print(f"      hidden: gpu_kernel is {100*frac:.0f}% of the "
                      f"{h2d_ms:.1f} ms transfer (model: m<ratio -> hides)")
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
