#include <reloc/Version.h>
#include <cstring>
int main() { return std::strlen(reloc::versionString()) == 0; }
