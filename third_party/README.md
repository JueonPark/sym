# Notices for embedded dependencies

Sym compiler executables incorporate LLVM/MLIR code from repository pin
`2078da43e25a4623cab2d0d60decddf709aaea28`. The Python extension incorporates
pybind11 3.0.4 headers. Their unmodified license notices are installed beside
native binaries in both the wheel and SDK.

- [LLVM notice at the pinned revision](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/LICENSE.TXT)
- [pybind11 3.0.4 notice](https://github.com/pybind/pybind11/blob/v3.0.4/LICENSE)

Release repair may add shared libraries. Inventory those actual dependencies and
include their notices before distributing a repaired package; these notices do
not certify an arbitrary native dependency closure.
