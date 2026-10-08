#!/usr/bin/env python3
"""Isolated Torch allocator diagnostics; run separately from latency measurements."""
import argparse
import json
from pathlib import Path
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=['small','cast','transpose','split'],required=True)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    torch.set_num_threads(8); torch.set_num_interop_threads(1); torch.cuda.init()
    shape = (128,128,1024) if args.case=='transpose' else ((65539,) if args.case=='small' else (16777219,))
    source = torch.ones(shape)
    before = torch.cuda.memory.host_memory_stats()
    pinned = torch.empty(shape,dtype=torch.float16,pin_memory=True)
    pinned.copy_(source.transpose(0,1) if args.case=='transpose' else source)
    after = torch.cuda.memory.host_memory_stats()
    torch.cuda.reset_peak_memory_stats()
    wire = pinned.to('cuda:0',non_blocking=True)
    out = wire.float() if args.case=='split' else wire
    del wire
    torch.cuda.synchronize()
    result = dict(case=args.case,torch=torch.__version__,
        pinned_logical_bytes=pinned.numel()*pinned.element_size(),
        pinned_active_capacity_bytes=after['active_bytes.current']-before['active_bytes.current'],
        pinned_reserved_capacity_bytes=after['allocated_bytes.current']-before['allocated_bytes.current'],
        gpu_output_logical_bytes=out.numel()*out.element_size(),
        gpu_allocated_bytes=torch.cuda.memory_allocated(),gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated())
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(result)


if __name__=='__main__': main()
