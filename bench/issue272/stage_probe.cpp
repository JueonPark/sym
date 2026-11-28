// Untimed diagnostic for the GNU Linux x86-64 shared runtime. Compile against
// the same checkout as libreloc_runtime.so. The wrapper forwards all arguments
// to the real implementation and counts only calls that return true.
#include "TypedStageKernels.h"
#include "reloc/Quant.h"

#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>

#if !defined(__linux__) || !defined(__x86_64__)
#error "The issue #272 probe requires the GNU Linux x86-64 ELF C++ ABI"
#endif

namespace {

using reloc::typed::detail::StageRun;
constexpr size_t boundaries = 9;
constexpr size_t kinds = 3;
constexpr size_t slots = kinds * boundaries * boundaries;
constexpr char stageSymbol[] = "_ZN5reloc5typed6detail11runStageRunERKNS0_"
                               "7ProgramERKNS1_8StageRunEPKhPhll";
constexpr char isaSymbol[] = "_ZN5reloc5quant11cpuSupportsENS0_7VariantE";

static_assert(sizeof(int64_t) == sizeof(long));
static_assert(static_cast<size_t>(StageRun::Kind::Buffered) == 0);
static_assert(static_cast<size_t>(StageRun::Kind::CastChain) == 1);
static_assert(static_cast<size_t>(StageRun::Kind::DequantCastChain) == 2);

struct Counts {
  std::atomic<uint64_t> calls{0};
  std::atomic<uint64_t> elements{0};
};

std::array<Counts, slots> totals;
std::atomic<uint64_t> outOfRange{0};

[[noreturn]] void fail(const char *message) {
  const char *loaderError = dlerror();
  std::fprintf(stderr, "issue272 stage probe: %s%s%s\n", message,
               loaderError ? ": " : "", loaderError ? loaderError : "");
  std::abort();
}

void *runtimeHandle() {
  static void *handle = [] {
    const char *path = std::getenv("SYM_ISSUE272_STAGE_RUNTIME");
    if (!path || !*path)
      fail("SYM_ISSUE272_STAGE_RUNTIME did not name the loaded runtime");
    // Python may load the extension and its dependency with RTLD_LOCAL.
    // Query the already-loaded DSO directly, without making it global.
    void *loaded = dlopen(path, RTLD_NOLOAD | RTLD_NOW);
    if (!loaded)
      fail("the named runtime DSO is not loaded");
    return loaded;
  }();
  return handle;
}

template <typename Symbol>
Symbol resolve(const char *name) {
  dlerror();
  auto symbol = reinterpret_cast<Symbol>(dlsym(runtimeHandle(), name));
  if (!symbol)
    fail(name);
  return symbol;
}
} // namespace

namespace reloc::typed::detail {

bool runStageRun(const Program &program, const StageRun &run,
                 const uint8_t *src, uint8_t *dst, int64_t count,
                 int64_t offset) {
  static auto original = [] {
    auto function = resolve<decltype(&runStageRun)>(stageSymbol);
    if (function == &runStageRun)
      fail("resolved the interposer instead of the runtime symbol");
    return function;
  }();
  if (!original(program, run, src, dst, count, offset))
    return false;
  const size_t kind = static_cast<size_t>(run.kind);
  if (kind >= kinds || run.from >= boundaries || run.to >= boundaries ||
      count < 0) {
    outOfRange.fetch_add(1, std::memory_order_relaxed);
    return true;
  }
  auto &slot = totals[(kind * boundaries + run.from) * boundaries + run.to];
  slot.calls.fetch_add(1, std::memory_order_relaxed);
  slot.elements.fetch_add(static_cast<uint64_t>(count),
                          std::memory_order_relaxed);
  return true;
}
} // namespace reloc::typed::detail

extern "C" unsigned sym_issue272_stage_probe_abi() { return 1; }

extern "C" int sym_issue272_stage_cpu_eligible() {
  auto cpuSupports = resolve<decltype(&reloc::quant::cpuSupports)>(isaSymbol);
  return cpuSupports(reloc::quant::Variant::AVX2);
}

extern "C" int sym_issue272_stage_counts(uint64_t *values, size_t size) {
  if (size != 2 * slots + 1)
    return -1;
  for (size_t i = 0; i < slots; ++i) {
    values[2 * i] = totals[i].calls.load(std::memory_order_relaxed);
    values[2 * i + 1] = totals[i].elements.load(std::memory_order_relaxed);
  }
  values[2 * slots] = outOfRange.load(std::memory_order_relaxed);
  return 0;
}
