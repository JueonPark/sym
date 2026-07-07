// RUN: sym-opt %s | sym-opt | FileCheck %s
// RUN: sym-opt --show-dialects | FileCheck %s --check-prefix=DIALECTS

// CHECK: module
// DIALECTS: reloc

module {}
