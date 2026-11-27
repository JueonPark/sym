"""Executable research model for a snapshot, a bound H2D artifact and a fresh cat."""
import copy
from dataclasses import dataclass

import pyreloc
import torch
from reloc_torch import compat
from reloc_torch.transport import _storage_view


@dataclass
class Invocation:
    """One immutable native parameter binding, plus observations of the last dispatch."""
    bound: object
    device: torch.device
    implementation: str = "cpu_reference"
    source_address: int = 0
    destination_address: int = 0
    report: dict | None = None

    @property
    def dtype(self):
        return getattr(torch, self.bound.result_dtype)

    def write(self, source, dst):
        if (type(source) is not torch.Tensor or source.device.type != "cpu"
                or source.shape != tuple(self.bound.source_extents)
                or source.dtype != getattr(torch, self.bound.source_dtype)
                or not source.is_contiguous() or source.is_neg() or source.is_conj()
                or source.requires_grad):
            raise ValueError("expected a dense CPU source matching the bound artifact")
        on_gpu = dst.is_cuda
        destination = _storage_view(dst, "cuda" if on_gpu else "host", dst.device.index if on_gpu else -1)
        request = pyreloc.prepare_dispatch(
            self.bound, _storage_view(source, "host", -1), destination, "h2d",
            implementation=self.implementation,
        )
        self.report = pyreloc.execute_dispatch(
            request, caller_stream=compat.cuda_stream_handle(dst.device) if on_gpu else None,
            owners=(source, dst),
        )
        self.source_address = source.data_ptr()
        self.destination_address = dst.data_ptr()


def transfer(call, source):
    dst = torch.empty(call.bound.result_extents, dtype=call.dtype, device=call.device)
    call.write(source, dst)
    return dst


def transfer_cat(call, source, prefix):
    shape = call.bound.result_extents
    out = prefix.new_empty((prefix.size(0) + shape[0], *shape[1:]))
    # The allocation is still private. Initialize through a separate version counter
    # so publishing it exposes the same fresh tensor version as functional cat.
    out.data[:prefix.size(0)].copy_(prefix)
    call.write(source, out[prefix.size(0):])
    return out


def materialized(call, source, prefix):
    producer = source.clone(memory_format=torch.contiguous_format)
    value = transfer(call, producer)
    return torch.cat((prefix, value), dim=0)


def capture(fn):
    tracer = torch.fx.Tracer(autowrap_functions=(transfer,))
    graph = tracer.trace(fn)
    return torch.fx.GraphModule(tracer.root, graph)


def _plain_dense(t):
    return (type(t) is torch.Tensor and t.layout == torch.strided
            and not t.is_quantized and not t.is_nested and not torch._is_zerotensor(t)
            and not torch._is_functional_tensor(t) and t.is_contiguous()
            and not t.is_conj() and not t.is_neg() and not t.requires_grad)


def _guard(call, source, prefix):
    # A Python dispatch override can observe the eliminated ops even with plain inputs.
    if (type(call) is not Invocation or torch._C._len_torch_dispatch_stack()
            or torch._C._len_torch_function_stack() or torch._C._are_functorch_transforms_active()
            or torch.autograd.forward_ad._current_level >= 0):
        return False
    if not (_plain_dense(source) and _plain_dense(prefix)):
        return False
    shape = tuple(call.bound.result_extents)
    return (source.device.type == "cpu" and not source.is_pinned()
            and source.data_ptr() % source.element_size() == 0
            and source.shape == tuple(call.bound.source_extents)
            and source.dtype == getattr(torch, call.bound.source_dtype)
            and prefix.device == call.device and prefix.dtype == call.dtype
            and prefix.dim() == len(shape) and 1 <= prefix.dim() <= 3 and prefix.size(0) > 0
            and prefix.shape[1:] == shape[1:] and not torch.is_autocast_enabled(prefix.device.type))


def lower(original):
    """Match the closed region grammar. Any other op may write or observe aliases."""
    graph = copy.deepcopy(original.graph)
    nodes = list(graph.nodes)
    if len(nodes) != 7:
        return original
    call, source, prefix, producer, value, cat, ret = nodes
    if not (all(n.op == "placeholder" for n in (call, source, prefix)) and ret.op == "output"
            and producer.op == "call_method" and producer.target == "clone"
            and producer.args == (source,) and producer.kwargs == {"memory_format": torch.contiguous_format}
            and value.op == "call_function" and value.target is transfer
            and value.args == (call, producer) and not value.kwargs
            and cat.op == "call_function" and cat.target is torch.cat
            and cat.args == ((prefix, value),) and cat.kwargs == {"dim": 0}):
        return original

    # FX output edges count as uses. Keep each independently observable allocation.
    forward_source = set(producer.users) == {value}
    forward_destination = set(value.users) == {cat}
    if not (forward_source or forward_destination):
        return original
    if forward_source:
        producer.replace_all_uses_with(source)
        graph.erase_node(producer)
    if forward_destination:
        cat.target = transfer_cat
        cat.args = (*value.args, prefix)
        cat.kwargs = {}
        graph.erase_node(value)
    graph.lint()
    rewritten = torch.fx.GraphModule(original, graph)

    def run(call, source, prefix):
        graph_module = rewritten if _guard(call, source, prefix) else original
        return graph_module(call, source, prefix)

    return run
