//===- VersionTest.cpp - libreloc scaffolding smoke tests -----------------===//
//
// Seed tests for the standalone runtime: the library links, the wire
// format version matches the frozen v0 spec (docs/reloc-plan-format.md),
// and the version string is queryable. #C2's decoder tests replace these
// as the real surface grows.
//
//===----------------------------------------------------------------------===//

#include "reloc/Version.h"
#include "gtest/gtest.h"

#include <cstring>

namespace {

TEST(LibrelocScaffolding, WireFormatVersionIsFrozenV0) {
  EXPECT_EQ(reloc::kWireFormatVersion, 0u);
}

TEST(LibrelocScaffolding, VersionStringMentionsWireFormat) {
  const char *version = reloc::versionString();
  ASSERT_NE(version, nullptr);
  EXPECT_NE(std::strstr(version, "wire format v0"), nullptr);
}

} // namespace
