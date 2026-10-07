#!/usr/bin/env bash
# Pin to GPU 0's local cores (EPYC 7351 box: 4-7,20-23) and compile Inductor
# in-process: its background worker pool otherwise steals the measured cores.
set -euo pipefail
export TORCHINDUCTOR_COMPILE_THREADS=1
exec taskset -c "${STACK_FUSION_CPUS:-4-7,20-23}" "${SYM_PYTHON:?set SYM_PYTHON to the cp314 cu126 python}" \
  "$(dirname "$0")/stack_fusion.py" "$@"
