# R5 / EXP-5 — DMA-engine negative result (issue #86)

**Neither the copy engine's 2-D path nor UVA zero-copy is a viable
substitute for the CPU-staged gather + pinned H2D. Zero-copy strided pull
collapses to ~1 GB/s (13× below the pinned-copy line); the legacy
`cudaMemcpy2D` small-pitch floor does NOT reproduce on Turing/CUDA 12.5
(it holds ~13 GB/s down to 64-byte rows) — a negative result in the other
direction, and still no win over a plain pinned copy.**

Measured 2026-07-27 on `rebel-gpu1` (RTX 2080 Ti, PCIe gen3, CUDA 12.5),
performance governor, persistence mode, pinned to GPU0's NUMA cores. Tool:
`bench-dma-engine` (`bench/rtrack/dma_engine.cu`); 256 MiB, 3 warmup + 20
timed, each config byte-exact / exact-sum verified before timing. Data:
`bench/results/r5_dma_engine_epyc_2080ti.json`.

## Table

### `cudaMemcpy2DAsync` H2D, effective BW vs row width

| row width | dense (spitch=W) | strided (spitch=2·W) |
|---|---|---|
| 64 B | 11.36 GB/s | 11.35 GB/s |
| 256 B | 13.08 | 13.08 |
| 1 KiB | 13.08 | 13.08 |
| 4 KiB | 13.08 | 13.08 |
| 16 KiB | 13.08 | 13.08 |
| 64 KiB | 13.08 | 13.08 |
| 256 KiB | 13.08 | 13.08 |
| 1 MiB | 13.08 | 13.08 |

Total useful bytes fixed at 256 MiB; strided reads `W` bytes then skips
`W`. (`cudaMemcpy3DAsync` shares the same copy-descriptor path and was not
separately swept in the 2-day box.)

### UVA zero-copy, delivered BW vs element stride

| stride (elems) | delivered BW | vs pinned copy (13.08) |
|---|---|---|
| 1 (contiguous) | 9.55 GB/s | 0.73× |
| 2 | 4.78 | 0.37× |
| 4 | 2.39 | 0.18× |
| 8 | 1.63 | 0.12× |
| 16 | 1.03 | 0.08× |
| 32 | 1.05 | 0.08× |

Delivered BW = (elements read × 4) / time — the *useful* rate; the PCIe
bus moves the full sector per access, so the wasted traffic grows with
stride.

## Two paragraphs (for the paper)

**The copy engine's 2-D path is not the trap the folklore warns of — but
it is also not a shortcut.** On the 2080 Ti under CUDA 12.5,
`cudaMemcpy2DAsync` sustains the full ~13 GB/s Gen3 line rate down to
256-byte rows and only dips to 11.4 GB/s at 64-byte rows, whether the
source rows are dense or strided at twice the copied width. The classic
"~300 MB/s floor at small pitch" from older GPUs does not reproduce: the
modern copy engine pipelines row descriptors well enough that a
million-row, 64-byte-per-row transfer still runs within 15 % of line rate.
This is a negative result in the unexpected direction, but it does not
change the design conclusion — a 2-D strided copy at best *matches* a plain
pinned 1-D copy and never exceeds it, so there is no way to fold a
relocation into the transfer for free by choosing a clever pitch.

**UVA zero-copy is the genuine trap.** Reading host-pinned mapped memory
directly from a kernel delivers only 9.55 GB/s even contiguously (0.73× of
the pinned-copy line, because there is no staging or copy-engine
pipelining), and it collapses as soon as the access is strided — 2.39 GB/s
at stride 4, ~1 GB/s at stride ≥ 16 — because each strided read still pulls
a full PCIe sector and discards all but one element of it. Since any
relocation is strided by construction, zero-copy pull delivers 5–13× less
useful bandwidth than the CPU-staged gather + pinned H2D that Method A and
Method B both use. Zero-copy is therefore ruled out as a relocation
transport: it trades the CPU gather for a far more expensive on-device
strided PCIe read.

## Caveat

Single 2080 Ti, Gen3, CUDA 12.5. The 2-D non-reproduction is
driver/architecture-dependent; the headline the paper should carry is the
zero-copy collapse (robust and mechanism-clear), with the 2-D result as a
"modern drivers fixed this, but it still doesn't beat a pinned copy" note.
