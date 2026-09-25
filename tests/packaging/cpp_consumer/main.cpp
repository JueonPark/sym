#include <cstring>
#include <reloc/Version.h>
int main() { return std::strlen(reloc::versionString()) == 0; }
