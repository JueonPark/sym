//===- Ablation.cpp - Runtime knobs that disable one optimization ---------===//

#include "reloc/Ablation.h"

#include <array>
#include <atomic>
#include <cstdlib>
#include <mutex>
#include <string>

namespace reloc {
namespace ablation {
namespace {

constexpr size_t kCount = static_cast<size_t>(Knob::Count);
constexpr std::array<const char *, kCount> kNames = {"scalar_dequant"};

std::array<std::atomic<bool>, kCount> &flags() {
  static std::array<std::atomic<bool>, kCount> values{};
  static std::once_flag initialized;
  // Environment knobs apply once, before the first query; set() afterwards
  // overrides them for the process.
  std::call_once(initialized, [] {
    if (const char *env = std::getenv("RELOC_ABLATE")) {
      std::string list(env);
      size_t start = 0;
      while (start <= list.size()) {
        size_t end = list.find(',', start);
        if (end == std::string::npos)
          end = list.size();
        if (auto knob =
                parse(std::string_view(list).substr(start, end - start)))
          values[static_cast<size_t>(*knob)].store(true,
                                                   std::memory_order_relaxed);
        start = end + 1;
      }
    }
  });
  return values;
}

} // namespace

bool enabled(Knob knob) {
  return flags()[static_cast<size_t>(knob)].load(std::memory_order_relaxed);
}

void set(Knob knob, bool on) {
  flags()[static_cast<size_t>(knob)].store(on, std::memory_order_relaxed);
}

std::optional<Knob> parse(std::string_view text) {
  for (size_t i = 0; i < kCount; ++i)
    if (text == kNames[i])
      return static_cast<Knob>(i);
  return std::nullopt;
}

const char *name(Knob knob) { return kNames[static_cast<size_t>(knob)]; }

} // namespace ablation
} // namespace reloc
