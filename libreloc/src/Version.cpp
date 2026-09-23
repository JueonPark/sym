//===- Version.cpp - libreloc version identification ----------------------===//

#include "reloc/Version.h"

const char *reloc::versionString() {
  return "libreloc 0.2 (wire format v0, typed wire format v1)";
}
