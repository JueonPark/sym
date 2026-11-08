"""Composed chains and independent eager semantics for the folding experiment."""
from dataclasses import dataclass
import torch
from reloc_torch.recipe import Cast, Dequantize, Quantize, BindingParam, Fill, Pad, Recipe, TensorSpec, Transpose
from reloc_torch.symbolic import Const, dense_strides

def spec(shape,dtype):
    shape=tuple(Const(i) for i in shape)
    return TensorSpec(shape,dense_strides(shape),Const(0),dtype)

@dataclass
class Case:
    family: str
    n: int
    m: int
    source: torch.Tensor
    scales: torch.Tensor
    parts: list
    shapes: list
    dtypes: list

    @property
    def recipe(self):
        return Recipe(self.parts[0].source,tuple(op for p in self.parts for op in p.operations),
                      self.parts[-1].destination,'h2d')

    def reference(self,x):
        if self.family == 'bf16_roundtrip':
            return x.t().contiguous().bfloat16().float()
        if self.family == 'dequant_roundtrip':
            return (x.float()*self.scales).t().contiguous().half().float()
        if self.family == 'quant_roundtrip':
            scaled = x * self.scales.reciprocal()
            q = torch.where(torch.isnan(scaled), -128., scaled).clamp(-128, 127).round().to(torch.int8)
            return (q.float()*self.scales).t().contiguous()
        if self.family.startswith('roundtrip'):
            for _ in range(int(self.family.removeprefix('roundtrip'))):
                x=x.t().contiguous().half().t().contiguous().float()
            return x
        if self.family=='rank3_chain':
            return x.permute(1,0,2).contiguous().half().permute(0,2,1).contiguous().float()
        if self.family=='dequant_transpose':
            return (x.float()*self.scales).t().contiguous().half()
        x=x.t().contiguous().half()
        return torch.nn.functional.pad(x,(2,7,3,5),value=-0.).float()

def make_case(family,n,m=128,seed=261104,outer=8):
    gen=torch.Generator().manual_seed(seed+n)
    dtype=torch.int8 if family.startswith('dequant_') else torch.float32
    shape=(outer,n,m) if family=='rank3_chain' else (n,m)
    source=(torch.randint(-128,128,shape,dtype=dtype,generator=gen) if dtype==torch.int8
            else torch.randn(shape,generator=gen)*7)
    scales=torch.rand(m,generator=gen)*.03+.001
    parts=[];shapes=[shape];dtypes=[str(dtype).split('.')[-1]]
    def add(ops,out_shape,out_dtype):
        for op in ops:
            shape,dtype=shapes[-1],dtypes[-1]
            if isinstance(op,Transpose):
                next_shape=tuple(shape[i] for i in op.perm);next_dtype=dtype
            elif isinstance(op,Pad):
                next_shape=list(shape);next_shape[op.axis]+=op.lo.value+op.hi.value
                next_shape=tuple(next_shape);next_dtype=dtype
            else:
                next_shape=shape;next_dtype=op.dtype
            parts.append(Recipe(spec(shape,dtype),(op,),spec(next_shape,next_dtype),'h2d'))
            shapes.append(next_shape);dtypes.append(next_dtype)
        assert shapes[-1]==out_shape and dtypes[-1]==out_dtype
    if family.startswith('roundtrip'):
        for _ in range(int(family.removeprefix('roundtrip'))):
            add([Transpose((1,0)),Cast('float16','ieee_rne')],(m,n),'float16')
            add([Transpose((1,0)),Cast('float32','exact')],(n,m),'float32')
    elif family=='rank3_chain':
        add([Transpose((1,0,2)),Cast('float16','ieee_rne')],(n,outer,m),'float16')
        add([Transpose((0,2,1)),Cast('float32','exact')],(n,m,outer),'float32')
    elif family.startswith('dequant_'):
        add([Dequantize('float32',BindingParam('scale','float32',(Const(m),)),None,1,'affine')],(n,m),'float32')
        add([Transpose((1,0)),Cast('float16','ieee_rne')],(m,n),'float16')
        if family == 'dequant_roundtrip':
            add([Cast('float32','exact')],(m,n),'float32')
    elif family == 'bf16_roundtrip':
        add([Transpose((1,0)),Cast('bfloat16','ieee_rne'),Cast('float32','exact')],(m,n),'float32')
    elif family == 'quant_roundtrip':
        scale = BindingParam('scale','float32',(Const(m),))
        add([Quantize('int8',scale,None,1,'symmetric_rne'),
             Dequantize('float32',scale,None,1,'affine'),Transpose((1,0))],(m,n),'float32')
    elif family=='padded_roundtrip':
        add([Transpose((1,0)),Cast('float16','ieee_rne')],(m,n),'float16')
        add([Pad(0,Const(3),Const(5),Fill('float16',0x8000)),
             Pad(1,Const(2),Const(7),Fill('float16',0x8000)),Cast('float32','exact')],(m+8,n+9),'float32')
    else:
        raise ValueError(family)
    return Case(family,n,m,source,scales,parts,shapes,dtypes)

FAMILIES=['roundtrip1','roundtrip2','roundtrip4','rank3_chain','dequant_transpose','padded_roundtrip']
