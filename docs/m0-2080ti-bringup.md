# M0 — 2080 Ti machine bring-up (issue #73)

Measured 2026-07-20 on `rebel-gpu1`. Reproduce with
`bench-multigpu-h2d` (bench/rtrack/multigpu_h2d.cu, this PR) plus
`bench/rtrack/calibrate.py`; raw JSONs in `bench/results/m0_*.json`.

## Checklist verdicts

| M0 item | verdict |
|---|---|
| Bare-metal Linux | **yes** — Ubuntu, kernel 6.8.0-124, no hypervisor. This box fills the bare-metal slot; the WSL2 caveat applies only to the Gen4 (7800X3D) numbers. |
| Topology | 4 Naples dies, one PCIe root complex each, **no PLX switch**. GPU0+GPU1 share die 1's root; GPU2 and GPU3 have dies to themselves. All four slots electrically **gen3 x16**. |
| Per-GPU pinned H2D ~12–13 GB/s | **confirmed on all four**: 13.08 GB/s H2D / 13.18 GB/s D2H — no downtrained slot, no x8 slot. Requires the placement rules below. |
| Aggregate 4-GPU H2D | 41.96 GB/s vs 50.2 sum-of-singles = **3.35× of 4× < the pre-registered 3.5× bar → EXP-3 (contention regime) is LIVE**. |
| Host CPU | EPYC 7351 (Zen1), 16C/32T, **AVX2-only** (no AVX-512), 4 NUMA nodes, DRAM on nodes 1 and 3 only (2 × 62 GiB, 4 of 8 channels populated), triad **44 GB/s** (parallel first-touch; 34 GB/s if serially touched). |

## Topology detail

```
die/node 0  cores 0-3,16-19    no local DRAM   no GPU
die/node 1  cores 4-7,20-23    62 GiB          GPU0 (21:00.0) + GPU1 (22:00.0)  <- shared root
die/node 2  cores 8-11,24-27   no local DRAM   GPU2 (41:00.0)
die/node 3  cores 12-15,28-31  62 GiB          GPU3 (61:00.0)
```

(`nvidia-smi topo -m`: GPU0–GPU1 = PHB, everything else = SYS.
`lspci -tv`: each GPU under its die's Family-17h root complex; no bridges
in between.) All links report 8 GT/s x16 capable and idle at 2.5 GT/s —
gen3 speed only appears under load, so record link state during a copy
loop (calibrate.py does).

## Bandwidth results (256 MiB pinned, 20 iters, best of 3 rounds)

Per-GPU alone, process pinned to the GPU's affinity cores: every GPU
13.08 H2D / 13.18 D2H GB/s.

Concurrent (barrier-started, one thread+stream per GPU):

| set | H2D aggregate | scaling | D2H aggregate | scaling |
|---|---|---|---|---|
| {0,1,2,3} | 41.96 GB/s | 3.35× of 4× | 38.85 GB/s | 3.28× |
| {0,1} same die | 21.02 GB/s | 1.61× of 2× | 19.66 GB/s | 1.49× |
| {2,3} separate dies | 23.69 GB/s | **1.98× of 2×** | 22.53 GB/s | 2.07× |

Structure of the contention:

- **Cross-die pairs scale perfectly** (1.98×) — PCIe roots are not shared
  across dies and Infinity Fabric handles two streams.
- **The shared-die pair {0,1} caps at ~21 GB/s** (1.6×) — die 1's
  host-side path saturates below 2 × 13.1. This is the EXP-3 contention
  pair.
- **The 4-GPU aggregate (~42 GB/s) sits at the host-DRAM roofline**
  (triad 44 GB/s with only 4 memory channels populated) — with all four
  engines pulling, the bottleneck moves from any PCIe link to host DRAM.
  EXP-3 on this box therefore probes BOTH regimes: root-port contention
  ({0,1}) and DRAM-bandwidth contention (K=4).

## NUMA placement rules (feed into every E/R-track driver on this box)

Found empirically (see PR discussion; reproducible with taskset):

1. **Enable persistence mode** (`sudo nvidia-smi -pm 1`, done this
   session, resets at reboot). Without it, links/clocks ramp per process
   and solo measurements read 5–9 GB/s until warm.
2. **Pin each measurement process to its GPU's affinity cores**
   (`nvidia-smi topo -m` column): GPU0/1 → 4-7,20-23; GPU2 → 8-11,24-27;
   GPU3 → 12-15,28-31. Pinned-buffer pages then land right for that GPU.
3. GPU2's die has **no local DRAM**: from a single memory node it reads
   only ~5.7–6.6 GB/s (one IF hop), but pinned to its own (memory-less)
   die's cores the fallback placement stripes pages across both memory
   nodes and it reaches full 13.08 GB/s. Unpinned runs are bimodal
   (6.3 / 13.1) at the scheduler's whim — never run unpinned.
4. GPU0 reading node-3 memory (one hop, local DRAM exists): 11.25 GB/s —
   a mild ~15% remote penalty; the catastrophic case is only the
   memory-less-die + single-remote-node combination.
5. `numactl` is not installed (no root apt); `taskset` + first-touch
   discipline substitutes. Install numactl when convenient.

## Gate implications for EXP-1 (pre-registration inputs)

- Host is **AVX2-only**: Method A's quant tier here is AVX2 (measured
  ~13–16 GB/s effective input on T3 at T=8, bench/results/
  rtrack_smoke_n8192_epyc_2080ti.csv), i.e. the CPU transform roughly
  MATCHES the 13.1 GB/s gen3 link rather than exceeding it as on the
  7800X3D. G2/G3 bars must be re-derived from these measured rooflines
  before EXP-1 runs, per the issue-#73 risk register.
- G1 (Gen3 link floor 11–14 GB/s): satisfied — 13.08 pinned H2D.
- Strided-gather roofline (T1b Method A): ~7 GB/s at any thread count
  (memory-bound), consistent with the post-#63-fix figure.
