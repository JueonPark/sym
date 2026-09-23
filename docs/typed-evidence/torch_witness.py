"""Characterize the pinned PyTorch build on the C1 witness vectors (issue #141).

Evidence for docs/reloc-typed-semantics.md section 7. Run from either qualified
environment (see docs/torch-integration.md):

    "$TORCH_PYTHON" -W ignore docs/typed-evidence/torch_witness.py [--cuda]

Prints exact outputs for f32->f16, f16->f32, f32->s8 (several PyTorch
routes) and s8->f32 so the semantics document can cite observed behavior
rather than assumed behavior. Runs on CPU always and on CUDA when present.
"""
import struct
import sys

import torch


def bits16(t):
    return [f"0x{v & 0xFFFF:04x}" for v in t.view(torch.int16).tolist()] if t.numel() else []


def bits32(t):
    return [f"0x{struct.unpack('<I', struct.pack('<f', v))[0]:08x}" for v in t.tolist()]


def run(device):
    print(f"== device {device} torch {torch.__version__} ==")
    ties = torch.tensor([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], device=device)
    limits = torch.tensor([-129.0, -128.0, -127.0, 126.0, 127.0, 128.0], device=device)
    specials = torch.tensor([float("nan"), float("inf"), float("-inf")], device=device)
    big = torch.tensor([1e9, -1e9, 3e9, -3e9], device=device)

    # f32 -> s8 via plain .to(int8): truncation toward zero, UB-style wrap on range.
    print("to(int8) ties      :", ties.to(torch.int8).tolist())
    print("to(int8) limits    :", limits.to(torch.int8).tolist())
    print("to(int8) specials  :", specials.to(torch.int8).tolist())
    print("to(int8) big       :", big.to(torch.int8).tolist())
    # torch.round is RNE (half to even).
    print("round ties         :", torch.round(ties).tolist())
    # clamp/round/to path (the reloc_torch reference_q shape)
    clamped = torch.clamp(torch.cat([ties, limits, specials]), -128.0, 127.0)
    clamped = torch.where(torch.isnan(torch.cat([ties, limits, specials])), torch.full_like(clamped, -128.0), clamped)
    print("clamp+round+to int8:", torch.round(clamped).to(torch.int8).tolist())
    # torch.quantize_per_tensor (qint8): uses round-half-to-even then clamp; zero_point added.
    if device == "cpu":
        for zp in (0, 127, -128):
            try:
                q = torch.quantize_per_tensor(torch.cat([ties, limits]), scale=1.0, zero_point=zp, dtype=torch.qint8)
                print(f"quantize_per_tensor zp={zp} int_repr:", q.int_repr().tolist())
            except Exception as error:  # pragma: no cover - reporting only
                print(f"quantize_per_tensor zp={zp} raised: {type(error).__name__}: {error}")
        try:
            q = torch.quantize_per_tensor(specials, scale=1.0, zero_point=0, dtype=torch.qint8)
            print("quantize_per_tensor specials int_repr:", q.int_repr().tolist())
        except Exception as error:  # pragma: no cover
            print(f"quantize_per_tensor specials raised: {type(error).__name__}: {error}")
        x = torch.arange(24.0).reshape(2, 3, 4) - 11.5
        scales = torch.tensor([0.5, 1.0, 2.0])
        try:
            q = torch.quantize_per_channel(x, scales, torch.zeros(3, dtype=torch.int64), axis=1, dtype=torch.qint8)
            print("quantize_per_channel axis=1 int_repr[0]:", q.int_repr()[0].tolist())
        except Exception as error:  # pragma: no cover
            print(f"quantize_per_channel raised: {type(error).__name__}: {error}")
    # x / scale vs x * (1/scale) are not bit-equivalent in general.
    xs = torch.tensor([0.1, 0.3, 0.7, 1.1, 2.9, 10.1, 100.3, 1000.7], device=device)
    scale = torch.tensor(0.3, device=device)
    div = xs / scale
    mul = xs * (1.0 / scale)
    print("x/scale vs x*inv equal bits:", torch.equal(div.view(torch.int32), mul.view(torch.int32)), "diff count:", int((div.view(torch.int32) != mul.view(torch.int32)).sum()))
    # f32 -> f16
    f16_in = torch.tensor([0.0, -0.0, 2.0 ** -24, 2.0 ** -25, 2.0 ** -14, 65504.0, 65520.0, 65519.99, 1e5, float("nan"), float("inf"), float("-inf"), 1.0 + 2.0 ** -11, 1.0 + 3 * 2.0 ** -12], device=device)
    f16 = f16_in.to(torch.float16)
    print("f32->f16 in        :", f16_in.tolist())
    print("f32->f16 bits      :", bits16(f16.cpu()))
    print("f32->f16 values    :", f16.float().tolist())
    nan_payload = torch.tensor([struct.unpack("<f", struct.pack("<I", 0x7fc12345))[0], struct.unpack("<f", struct.pack("<I", 0xffa00000))[0]], device=device)
    print("nan payload f32 bits:", bits32(nan_payload.cpu()), "-> f16 bits:", bits16(nan_payload.to(torch.float16).cpu()))
    # f16 -> f32 (exact)
    h16 =torch.tensor([0x0001, 0x0400, 0x7bff, 0x7c00, -0x0400, 0x7e00, -0x8000], dtype=torch.int16).view(torch.float16).to(device)
    print("f16->f32 bits      :", bits32(h16.float().cpu()))
    # s8 -> f32 dequant: exact for |q| <= 127 with power-of-two scale, else one rounding.
    q = torch.tensor([-128, -1, 0, 1, 127], dtype=torch.int8, device=device)
    print("s8->f32 * 0.3      :", bits32((q.float() * 0.3).cpu()))
    print("s8->f32 (int8*scale in f32 vs f64):", torch.equal((q.float() * 0.3).double(), q.double() * 0.3))
    if device == "cpu":
        try:
            dq = torch.dequantize(torch.quantize_per_tensor(torch.tensor([-38.4, 0.3, 38.1]), 0.3, 0, torch.qint8))
            print("torch.dequantize(0.3):", bits32(dq))
        except Exception as error:  # pragma: no cover
            print(f"dequantize raised: {type(error).__name__}: {error}")


run("cpu")
if torch.cuda.is_available() and "--cuda" in sys.argv:
    run("cuda:0")
