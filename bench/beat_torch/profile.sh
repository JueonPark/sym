#!/bin/bash
# Nsight Systems capture of one regime: both the eager Torch control and the
# sym_fast arm run in one process under NVTX ranges (compare.py --trace).
#   bench/beat_torch/profile.sh <build> <python> <case> <out-prefix> [samples]
# Afterwards: nsys stats --report cuda_gpu_trace / nvtx_sum, or export sqlite
# and run bench/beat_torch/analyze.py.
set -euo pipefail
BUILD=$1; PY=$2; CASE=$3; OUT=$4; SAMPLES=${5:-3}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
NSYS=${NSYS:-/usr/local/cuda-12.5/bin/nsys}
export PYTHONPATH=$BUILD/python LD_LIBRARY_PATH=$BUILD/libreloc \
  SYM_RELOC_EXPORT=$BUILD/sym/tools/sym-reloc-export SYM_OPT=$BUILD/sym/tools/sym-opt \
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0 \
  TORCHINDUCTOR_COMPILE_THREADS=1 TORCHINDUCTOR_CACHE_DIR=${OUT}-cache
cd "$ROOT"
"$NSYS" profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi \
  --capture-range-end=stop --force-overwrite=true -o "$OUT" \
  taskset -c 4-7,20-23 "$PY" bench/beat_torch/compare.py --output "${OUT}.json" \
  --cases "$CASE" --paths torch sym_fast --samples "$SAMPLES" --warmup 2 --trace
"$NSYS" export --type sqlite --force-overwrite=true -o "${OUT}.sqlite" "${OUT}.nsys-rep"
