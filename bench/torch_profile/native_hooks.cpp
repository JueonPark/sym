// Diagnostic-only ELF interposition for the exact installed PyTorch ABI.
// Each wrapper forwards once, unchanged, to the original shared-library method.
// Never preload this library for headline performance measurements.
#include <ATen/TensorIterator.h>
#include <ATen/record_function.h>
#include <nvtx3/nvToolsExt.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>

namespace {
std::atomic<bool> enabled{false};
bool detailed_ranges = true;
std::atomic<unsigned long long> calls{0};
std::atomic<unsigned long long> cpu_copies{0};
at::CallbackHandle callback = 0;
struct CopyContext : at::ObserverContext {
  nvtxRangeId_t range;
  explicit CopyContext(const char *name) : range(nvtxRangeStartA(name)) {}
};
std::unique_ptr<at::ObserverContext> copyStart(const at::RecordFunction &fn) {
  if (std::strcmp(fn.name(), "aten::copy_") != 0) return nullptr;
  const auto inputs = fn.inputs();
  if (inputs.size() < 2 || !inputs[0].isTensor() || !inputs[1].isTensor()) return nullptr;
  const auto &dst = inputs[0].toTensor();
  const auto &src = inputs[1].toTensor();
  // A CPU copy region includes operator setup, checks and its CPU kernel.
  // Analysis subtracts instrumented preparation from it; residual checks remain.
  if (dst.device().is_cpu() && src.device().is_cpu()) {
    cpu_copies.fetch_add(1, std::memory_order_relaxed);
    return std::make_unique<CopyContext>("torch.native.cpu_copy_operator");
  }
  return nullptr;
}
void copyEnd(const at::RecordFunction &, at::ObserverContext *ctx) {
  if (ctx) nvtxRangeEnd(static_cast<CopyContext *>(ctx)->range);
}
void *resolve(const char *symbol) {
  static void *handle = dlopen("libtorch_cpu.so", RTLD_NOW | RTLD_NOLOAD);
  void *address = handle ? dlsym(handle, symbol) : nullptr;
  if (!address) {
    std::fprintf(stderr, "symprof: cannot resolve original Torch symbol %s: %s\n",
                 symbol, dlerror());
    std::abort();
  }
  return address;
}
struct SymprofScope {
  bool active = enabled.load(std::memory_order_relaxed);
  explicit SymprofScope(const char *name) {
    active = active && (detailed_ranges || std::strcmp(name, "torch.native.ti.build") == 0 ||
                        std::strcmp(name, "torch.native.cpu_loop") == 0);
    if (active) {
      calls.fetch_add(1, std::memory_order_relaxed);
      nvtxRangePushA(name);
    }
  }
  ~SymprofScope() { if (active) nvtxRangePop(); }
};
}
extern "C" void symprof_enable_native(int value) {
  if (value && !callback) {
    const char *mode = std::getenv("SYMPROF_NATIVE_DETAIL");
    detailed_ranges = !mode || std::strcmp(mode, "build") != 0;
    callback = at::addThreadLocalCallback(at::RecordFunctionCallback(copyStart, copyEnd).needsInputs(true));
  }
  if (!value && callback) {
    at::removeCallback(callback);
    callback = 0;
  }
  enabled.store(value != 0);
}
extern "C" unsigned long long symprof_native_calls() { return calls.load(); }
extern "C" unsigned long long symprof_cpu_copies() { return cpu_copies.load(); }

#define CONFIG_METHOD(NAME, QUAL, SYMBOL) \
void at::TensorIteratorBase::NAME(QUAL at::TensorIteratorConfig &config) { \
  using Fn = void (*)(at::TensorIteratorBase *, QUAL at::TensorIteratorConfig &); \
  static Fn original = reinterpret_cast<Fn>(resolve(SYMBOL)); \
  SymprofScope scope("torch.native.ti." #NAME); \
  original(this, config); \
}
#define NOARG_METHOD(NAME, SYMBOL) \
void at::TensorIteratorBase::NAME() { \
  using Fn = void (*)(at::TensorIteratorBase *); \
  static Fn original = reinterpret_cast<Fn>(resolve(SYMBOL)); \
  SymprofScope scope("torch.native.ti." #NAME); \
  original(this); \
}

CONFIG_METHOD(build, , "_ZN2at18TensorIteratorBase5buildERNS_20TensorIteratorConfigE")
CONFIG_METHOD(populate_operands, , "_ZN2at18TensorIteratorBase17populate_operandsERNS_20TensorIteratorConfigE")
CONFIG_METHOD(compute_mem_overlaps, const, "_ZN2at18TensorIteratorBase20compute_mem_overlapsERKNS_20TensorIteratorConfigE")
CONFIG_METHOD(compute_shape, const, "_ZN2at18TensorIteratorBase13compute_shapeERKNS_20TensorIteratorConfigE")
CONFIG_METHOD(mark_resize_outputs, const, "_ZN2at18TensorIteratorBase19mark_resize_outputsERKNS_20TensorIteratorConfigE")
CONFIG_METHOD(compute_types, const, "_ZN2at18TensorIteratorBase13compute_typesERKNS_20TensorIteratorConfigE")
CONFIG_METHOD(compute_strides, const, "_ZN2at18TensorIteratorBase15compute_stridesERKNS_20TensorIteratorConfigE")
NOARG_METHOD(mark_outputs, "_ZN2at18TensorIteratorBase12mark_outputsEv")
NOARG_METHOD(reorder_dimensions, "_ZN2at18TensorIteratorBase18reorder_dimensionsEv")
NOARG_METHOD(allocate_or_resize_outputs, "_ZN2at18TensorIteratorBase26allocate_or_resize_outputsEv")
NOARG_METHOD(coalesce_dimensions, "_ZN2at18TensorIteratorBase19coalesce_dimensionsEv")

bool at::TensorIteratorBase::fast_set_up(const at::TensorIteratorConfig &config) {
  using Fn = bool (*)(at::TensorIteratorBase *, const at::TensorIteratorConfig &);
  static Fn original = reinterpret_cast<Fn>(resolve("_ZN2at18TensorIteratorBase11fast_set_upERKNS_20TensorIteratorConfigE"));
  SymprofScope scope("torch.native.ti.fast_set_up");
  return original(this, config);
}
void at::TensorIteratorBase::for_each(loop2d_t loop, int64_t grain_size) {
  using Fn = void (*)(at::TensorIteratorBase *, loop2d_t, int64_t);
  static Fn original = reinterpret_cast<Fn>(resolve("_ZN2at18TensorIteratorBase8for_eachEN3c1012function_refIFvPPcPKlllEEEl"));
  SymprofScope scope("torch.native.cpu_loop");
  original(this, loop, grain_size);
}
