# V3 Cost-Model Component Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `reloc::costmodel` (issue #97 / P3b): a calibrated two-path cost model in libreloc, consumed by `bind`, exposed through pyreloc, validated by a pre-registered prediction test over committed R-track data, plus one measured row driven by a compiler-emitted plan.

**Architecture:** One C++ component (flat-text calibration parser → pattern classifier → path costs → decision + single-symbol threshold), a thin pybind surface so the Python prediction test calls the identical implementation, a deterministic calibration assembler over committed `bench/results` artifacts, and a `--plan-wire` bench row that closes the compiler→measurement gap. Spec: `docs/superpowers/specs/2026-07-29-v3-costmodel-design.md`.

**Tech Stack:** C++17 (libreloc, MLIR-free), pybind11 (pyreloc), Python 3 stdlib (+pytest for tests), gtest via the `build/sym` Ninja tree, CUDA 12.5 / sm_75 for the one measured row on this box (`rebel-gpu1`).

## Global Constraints

- libreloc is MLIR-free: no `mlir/`/`llvm/` includes under `libreloc/` (CTest contract tests enforce).
- Preconditions checked unconditionally, never bare `assert` (benchmark builds compile `-DNDEBUG`).
- Gate bars are fixed in code and committed BEFORE the prediction test first runs against data (the `a051a5a` discipline).
- Prediction-test cells that the model gets wrong are REPORTED and explained, never refit.
- All model arithmetic exists only in C++; Python calls it through pyreloc (no mirror implementations).
- Milliseconds for times, GB/s (=1e9 bytes/s) for bandwidths, bytes for sizes — everywhere.
- Plans in tests are hand-authored `BoundPlan`s or corpus-generated blobs — never `bench/reference_plan.h` (issue #63).
- Code style: LLVM-ish, 80-column comments, `//===- file - purpose -===//` banners; run `clang-format` (pip `clang-format==21.1.8`, matching CI) on every touched `.cpp/.h` before committing.
- Branch: `v3-costmodel` (exists; spec committed as `5c00490`).
- The gtest tree: `ninja -C build/sym libreloc-test`; binary via `find build/sym -name libreloc-test -type f`. Python tests: `PYTHONPATH=build/sym/python <venv-python> -m pytest libreloc/python/tests/...` (matplotlib/pytest venv at `/tmp/claude-2017/-home-jueonpark-sym/595debf3-ca62-43ba-a9fe-225c7908aed7/scratchpad/fmt-venv`; `build/sym` is already configured with pybind11).

**One spec refinement discovered during planning (apply in Task 3, note in the report):** the pure-bandwidth cost forms are linear in `S` with **no S-dependent min/max arm switches**, so an S-threshold is degenerate (always-A or always-B) unless the model carries per-transfer fixed costs. The calibration therefore includes `overhead.a_ms` / `overhead.b_ms` intercepts (fitted deterministically from the committed N=2048 vs N=16384 medians), and the threshold is the argmin-boundary of the affine cost lines. The spec file is amended by this plan's Task 9.

---

### Task 1: Calibration parser — `CostModel::load` (TDD)

**Files:**
- Create: `libreloc/include/reloc/CostModel.h`
- Create: `libreloc/src/CostModel.cpp`
- Create: `libreloc/test/CostModelTest.cpp`
- Modify: `libreloc/CMakeLists.txt` (add `src/CostModel.cpp` after `src/Prefold.cpp` in `reloc_runtime`)
- Modify: `libreloc/test/CMakeLists.txt` (add `CostModelTest.cpp` after `PrefoldTest.cpp`)

**Interfaces:**
- Consumes: nothing.
- Produces: `reloc::costmodel::CostModel` with `static std::variant<CostModel, std::string> load(const std::string &path)` and `static std::variant<CostModel, std::string> parse(const std::string &text)`; `bool has(const std::string &key) const`; `double at(const std::string &key) const` (precondition: `has`); `double get(const std::string &key, double fallback) const`; `const std::string &machine() const`. Format: first non-blank line must be `# costmodel calibration v0`; optional `# machine: <slug>` comment sets `machine()`; data lines are `key value` with dotted keys; `#` starts a comment (full-line or trailing); duplicate keys, non-numeric values, and malformed lines are errors naming the line number.

- [ ] **Step 1: Write the failing tests**

Create `libreloc/test/CostModelTest.cpp`:

```cpp
//===- CostModelTest.cpp - P3b cost-model component (issue #97) -----------===//

#include "reloc/CostModel.h"

#include "gtest/gtest.h"

#include <string>
#include <variant>

namespace {

using reloc::costmodel::CostModel;

const char *kMinimal = R"(# costmodel calibration v0
# machine: testbox
pcie.h2d_gbps 13.07
cpu.t8.contiguous.quantize_pack_gbps 22.76  # trailing comment
hbm.bw_gbps 544
)";

CostModel mustParse(const std::string &text) {
  auto r = CostModel::parse(text);
  EXPECT_TRUE(std::holds_alternative<CostModel>(r))
      << std::get<std::string>(r);
  return std::get<CostModel>(r);
}

std::string mustFail(const std::string &text) {
  auto r = CostModel::parse(text);
  EXPECT_TRUE(std::holds_alternative<std::string>(r));
  return std::holds_alternative<std::string>(r) ? std::get<std::string>(r)
                                                : std::string();
}

TEST(CostModelParse, MinimalFileRoundTrips) {
  CostModel m = mustParse(kMinimal);
  EXPECT_EQ(m.machine(), "testbox");
  EXPECT_TRUE(m.has("pcie.h2d_gbps"));
  EXPECT_DOUBLE_EQ(m.at("pcie.h2d_gbps"), 13.07);
  EXPECT_DOUBLE_EQ(m.at("cpu.t8.contiguous.quantize_pack_gbps"), 22.76);
  EXPECT_FALSE(m.has("absent.key"));
  EXPECT_DOUBLE_EQ(m.get("absent.key", -1.0), -1.0);
}

TEST(CostModelParse, RejectsMissingOrWrongVersionHeader) {
  EXPECT_NE(mustFail("pcie.h2d_gbps 13.07\n").find("version"),
            std::string::npos);
  EXPECT_NE(mustFail("# costmodel calibration v1\nk 1\n").find("version"),
            std::string::npos);
}

TEST(CostModelParse, RejectsMalformedLines) {
  const std::string base = "# costmodel calibration v0\n";
  // No value.
  EXPECT_NE(mustFail(base + "pcie.h2d_gbps\n").find("line 2"),
            std::string::npos);
  // Non-numeric value.
  EXPECT_NE(mustFail(base + "pcie.h2d_gbps fast\n").find("line 2"),
            std::string::npos);
  // Trailing junk after the value that is not a comment.
  EXPECT_NE(mustFail(base + "pcie.h2d_gbps 13.07 junk\n").find("line 2"),
            std::string::npos);
}

TEST(CostModelParse, RejectsDuplicateKeys) {
  const std::string text = "# costmodel calibration v0\n"
                           "pcie.h2d_gbps 13.07\n"
                           "pcie.h2d_gbps 26.79\n";
  EXPECT_NE(mustFail(text).find("duplicate"), std::string::npos);
}

TEST(CostModelParse, LoadReportsUnreadablePath) {
  auto r = CostModel::load("/nonexistent/path.cal");
  ASSERT_TRUE(std::holds_alternative<std::string>(r));
}

} // namespace
```

- [ ] **Step 2: Add to CMake, run to verify failure**

Add `CostModelTest.cpp` to `libreloc/test/CMakeLists.txt` and `src/CostModel.cpp` to `libreloc/CMakeLists.txt`.
Run: `ninja -C build/sym libreloc-test 2>&1 | tail -3`
Expected: FAIL — `reloc/CostModel.h: No such file or directory`.

- [ ] **Step 3: Write the header**

Create `libreloc/include/reloc/CostModel.h`:

```cpp
//===- CostModel.h - P3b calibrated transfer cost model (#97) ---*- C++ -*-===//
//
// Consolidates the R-track's analytic model into a component: a flat-text
// calibration file (per machine, assembled deterministically from
// committed bench/results artifacts) feeds a two-path cost model
//   T_A = overhead.a_ms + S * max(1/BW_cpu(pattern), wireBytes/S/BW_link)
//   T_B = overhead.b_ms + S * max(1/BW_link, m/BW_hbm)
// (all affine in S), a gather-pattern classifier over BoundPlan
// properties, and a decision with a single-free-symbol threshold
// precompute so bind() only compares the bound size against a stored
// boundary. The prefold arm delegates to reloc::prefold::prefoldWins.
// Units: ms, GB/s (1e9 bytes/s), bytes.
//
//===----------------------------------------------------------------------===//

#ifndef RELOC_COSTMODEL_H
#define RELOC_COSTMODEL_H

#include "reloc/Bind.h"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <variant>

namespace reloc {
namespace costmodel {

/// Host-side access pattern of Method A's transform, classified from the
/// coalesced BoundPlan alone. Misclassifying the pattern dominates every
/// other modelling error (R1 measured a >20x spread across these).
enum class Pattern { Contiguous, Blocked, SingleElement, Tiled };

const char *patternName(Pattern p);

/// L == totalElems -> Contiguous; L == 1 -> SingleElement;
/// L >= kBlockedRunFloor -> Blocked; else Tiled.
constexpr int64_t kBlockedRunFloor = 64; // elements
Pattern classify(const BoundPlan &b);

class CostModel {
public:
  /// Parse a "# costmodel calibration v0" flat text file. Returns the
  /// model or a diagnostic naming the offending line. Checked
  /// unconditionally (-DNDEBUG builds keep every check).
  static std::variant<CostModel, std::string> parse(const std::string &text);
  static std::variant<CostModel, std::string> load(const std::string &path);

  bool has(const std::string &key) const { return values_.count(key) != 0; }
  double at(const std::string &key) const { return values_.at(key); }
  double get(const std::string &key, double fallback) const {
    auto it = values_.find(key);
    return it == values_.end() ? fallback : it->second;
  }
  const std::string &machine() const { return machine_; }

private:
  std::map<std::string, double> values_;
  std::string machine_;
};

/// Source-normalized CPU GB/s of Method A's transform stage for this
/// pattern at wire ratio r (the figure_rstar composition: two-pass
/// stages compose harmonically; pack's source-normalized BW is 4x its
/// input BW). nullopt when the calibration lacks a needed key.
std::optional<double> cpuBw(const CostModel &m, Pattern p, double r,
                            int threads);

struct PathCosts {
  double tAMs = 0, tBMs = 0;
  // Affine decomposition (per-method): t = interceptMs + slopeMsPerByte*S.
  double aInterceptMs = 0, aSlopeMsPerByte = 0;
  double bInterceptMs = 0, bSlopeMsPerByte = 0;
};

/// Both path costs for shipping one source tensor of S bytes with wire
/// ratio r to K receivers (broadcast: every receiver gets the whole
/// tensor; scatter: receivers partition it). nullopt on missing keys.
std::optional<PathCosts> pathCosts(const CostModel &m, Pattern p,
                                   int64_t srcBytes, double r, int threads,
                                   int K = 1, bool broadcast = false);

struct MethodDecision {
  enum class Method { A, B, APrefold };
  Method method = Method::B;
  double tAMs = 0, tBMs = 0;   // chosen-arm A is APrefold's when nReuse>0
  double thresholdBytes = -1;  // argmin boundary nearest the bound S;
                               // -1 = decision is size-independent
  Pattern pattern = Pattern::Contiguous;
  int k = 1;
  int64_t nReuse = -1;
};

const char *methodName(MethodDecision::Method m);

/// The decision. nReuse < 0 disables the prefold arm; nReuse >= 1
/// enables it via reloc::prefold::prefoldWins (V4's validated rule).
std::optional<MethodDecision> decide(const CostModel &m, Pattern p,
                                     int64_t srcBytes, double r,
                                     int threads = 8, int K = 1,
                                     int64_t nReuse = -1,
                                     bool broadcast = false);

} // namespace costmodel
} // namespace reloc

#endif // RELOC_COSTMODEL_H
```

- [ ] **Step 4: Implement the parser (only) in `libreloc/src/CostModel.cpp`**

```cpp
//===- CostModel.cpp - P3b calibrated transfer cost model (#97) -----------===//

#include "reloc/CostModel.h"

#include "reloc/Prefold.h"

#include <cmath>
#include <fstream>
#include <sstream>

namespace reloc {
namespace costmodel {

static std::string stripComment(const std::string &line) {
  size_t h = line.find('#');
  return h == std::string::npos ? line : line.substr(0, h);
}

static std::string trim(const std::string &s) {
  size_t b = s.find_first_not_of(" \t\r\n");
  if (b == std::string::npos)
    return "";
  size_t e = s.find_last_not_of(" \t\r\n");
  return s.substr(b, e - b + 1);
}

std::variant<CostModel, std::string>
CostModel::parse(const std::string &text) {
  CostModel m;
  std::istringstream in(text);
  std::string line;
  int lineNo = 0;
  bool sawVersion = false;
  while (std::getline(in, line)) {
    ++lineNo;
    const std::string raw = trim(line);
    if (raw.empty())
      continue;
    if (!sawVersion) {
      if (raw != "# costmodel calibration v0")
        return std::string("calibration: first non-blank line must be the "
                           "v0 version header (line " +
                           std::to_string(lineNo) + ")");
      sawVersion = true;
      continue;
    }
    if (raw[0] == '#') {
      const std::string kMachine = "# machine:";
      if (raw.rfind(kMachine, 0) == 0)
        m.machine_ = trim(raw.substr(kMachine.size()));
      continue;
    }
    std::istringstream ls(stripComment(raw));
    std::string key, value, extra;
    ls >> key >> value;
    if (key.empty() || value.empty())
      return std::string("calibration: expected 'key value' at line " +
                         std::to_string(lineNo));
    if (ls >> extra)
      return std::string("calibration: trailing junk at line " +
                         std::to_string(lineNo));
    char *end = nullptr;
    const double v = std::strtod(value.c_str(), &end);
    if (end == value.c_str() || *end != '\0' || !std::isfinite(v))
      return std::string("calibration: non-numeric value at line " +
                         std::to_string(lineNo));
    if (!m.values_.emplace(key, v).second)
      return std::string("calibration: duplicate key '" + key +
                         "' at line " + std::to_string(lineNo));
  }
  if (!sawVersion)
    return std::string("calibration: empty file (no version header)");
  return m;
}

std::variant<CostModel, std::string>
CostModel::load(const std::string &path) {
  std::ifstream f(path);
  if (!f)
    return std::string("calibration: cannot read " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return parse(ss.str());
}

} // namespace costmodel
} // namespace reloc
```

(The header declares `classify`/`cpuBw`/`pathCosts`/`decide`/`patternName`/`methodName`; they are implemented in Tasks 2–3. To keep this task linking, add temporary definitions NOTHING — the test file only references parse/load/has/at/get/machine, and unreferenced declarations don't need definitions.)

- [ ] **Step 5: Run tests, verify pass**

Run: `ninja -C build/sym libreloc-test && $(find build/sym -name libreloc-test -type f | head -1) --gtest_filter='CostModelParse.*'`
Expected: 5 tests PASS.

- [ ] **Step 6: clang-format + commit**

```bash
SP=/tmp/claude-2017/-home-jueonpark-sym/595debf3-ca62-43ba-a9fe-225c7908aed7/scratchpad
$SP/fmt-venv/bin/clang-format -i libreloc/include/reloc/CostModel.h \
  libreloc/src/CostModel.cpp libreloc/test/CostModelTest.cpp
ninja -C build/sym libreloc-test   # recheck after format
git add libreloc/include/reloc/CostModel.h libreloc/src/CostModel.cpp \
  libreloc/test/CostModelTest.cpp libreloc/CMakeLists.txt \
  libreloc/test/CMakeLists.txt
git commit -m "feat(libreloc): costmodel calibration parser (#97)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Classifier + path costs (TDD)

**Files:**
- Modify: `libreloc/src/CostModel.cpp`
- Modify: `libreloc/test/CostModelTest.cpp`

**Interfaces:**
- Consumes: Task 1's `CostModel`; `reloc::BoundPlan` (`reloc/Bind.h`: `extents`, `totalBytes`, `elementSize`, `L`).
- Produces: `patternName(Pattern) -> const char*` (`"contiguous"|"blocked"|"single_element"|"tiled"`); `classify(const BoundPlan&) -> Pattern`; `cpuBw(model, pattern, r, threads) -> optional<double>`; `pathCosts(model, pattern, srcBytes, r, threads, K, broadcast) -> optional<PathCosts>` with the affine decomposition filled. Calibration keys used: `cpu.t{threads}.{pattern}.{kernel}_gbps` for kernels `contig_read, gather_f32, convert_f32_f16, quantize_pack, gather_quantize, pack_s8_s4`; `pcie.h2d_gbps`; `multigpu.delivery_gbps.k{K}` (K>1); `hbm.bw_gbps`; `hbm.m.{pattern}`; `overhead.a_ms`, `overhead.b_ms` (default 0 via `get`).

- [ ] **Step 1: Write the failing tests** (append to `CostModelTest.cpp`)

```cpp
using reloc::costmodel::classify;
using reloc::costmodel::cpuBw;
using reloc::costmodel::pathCosts;
using reloc::costmodel::Pattern;

reloc::BoundPlan planWithL(int64_t elems, int64_t L) {
  reloc::BoundPlan b;
  b.extents = {elems / L > 0 ? elems / L : 1, L};
  b.srcStrides = {1, elems / L > 0 ? elems / L : 1}; // shape-only fixture
  b.dstStrides = {L, 1};
  b.elementSize = 4;
  b.totalBytes = elems * 4;
  b.L = L;
  return b;
}

TEST(CostModelClassify, MapsLToPattern) {
  EXPECT_EQ(classify(planWithL(4096, 4096)), Pattern::Contiguous);
  EXPECT_EQ(classify(planWithL(4096, 1)), Pattern::SingleElement);
  EXPECT_EQ(classify(planWithL(4096, 64)), Pattern::Blocked);   // == floor
  EXPECT_EQ(classify(planWithL(4096, 512)), Pattern::Blocked);
  EXPECT_EQ(classify(planWithL(4096, 8)), Pattern::Tiled);
}

// Synthetic calibration with easy numbers: link 10 GB/s, contiguous CPU
// kernels all 20 GB/s, blocked gather 5 GB/s, HBM m=2 with BW 100.
const char *kSynth = R"(# costmodel calibration v0
pcie.h2d_gbps 10
cpu.t8.contiguous.contig_read_gbps 20
cpu.t8.contiguous.convert_f32_f16_gbps 20
cpu.t8.contiguous.quantize_pack_gbps 20
cpu.t8.contiguous.pack_s8_s4_gbps 20
cpu.t8.blocked.gather_f32_gbps 5
cpu.t8.blocked.convert_f32_f16_gbps 20
cpu.t8.blocked.gather_quantize_gbps 4
cpu.t8.blocked.pack_s8_s4_gbps 20
hbm.bw_gbps 100
hbm.m.contiguous 1
hbm.m.blocked 2
multigpu.delivery_gbps.k4 30
overhead.a_ms 0.5
overhead.b_ms 0.1
)";

TEST(CostModelCpuBw, FigureRstarComposition) {
  CostModel m = mustParse(kSynth);
  // Contiguous r=1.0 -> contig_read.
  EXPECT_DOUBLE_EQ(*cpuBw(m, Pattern::Contiguous, 1.0, 8), 20.0);
  // Contiguous r=0.25 -> quantize_pack.
  EXPECT_DOUBLE_EQ(*cpuBw(m, Pattern::Contiguous, 0.25, 8), 20.0);
  // Strided r=0.5 -> harmonic(gather_f32, convert): 1/(1/5+1/20) = 4.
  EXPECT_DOUBLE_EQ(*cpuBw(m, Pattern::Blocked, 0.5, 8), 4.0);
  // Strided r=0.125 -> harmonic(gather_quantize, 4*pack):
  // 1/(1/4 + 1/80) = 80/21.
  EXPECT_NEAR(*cpuBw(m, Pattern::Blocked, 0.125, 8), 80.0 / 21.0, 1e-12);
  // Missing tier -> nullopt.
  EXPECT_FALSE(cpuBw(m, Pattern::Blocked, 1.0, 1).has_value());
  // Unmodelled r -> nullopt (not a crash).
  EXPECT_FALSE(cpuBw(m, Pattern::Blocked, 0.3, 8).has_value());
}

TEST(CostModelPathCosts, AffineFormsAndK) {
  CostModel m = mustParse(kSynth);
  const int64_t S = 1000000000; // 1 GB -> 1 s per 1 GB/s: easy arithmetic
  // Contiguous r=0.25, K=1: A slope = max(1/20, 0.25/10)=0.05 ms/MB ->
  // tA = 0.5 + 1000*0.05... in ms: S/1e9 * 1e3 * max(1/20, 0.025) = 50ms.
  auto pc = pathCosts(m, Pattern::Contiguous, S, 0.25, 8, 1, false);
  ASSERT_TRUE(pc.has_value());
  EXPECT_NEAR(pc->tAMs, 0.5 + 50.0, 1e-9);
  // B: max(1/10, 1/100) = 0.1 s -> 100 ms + 0.1 intercept.
  EXPECT_NEAR(pc->tBMs, 0.1 + 100.0, 1e-9);
  // Affine decomposition consistent: t = intercept + slope*S.
  EXPECT_NEAR(pc->aInterceptMs + pc->aSlopeMsPerByte * S, pc->tAMs, 1e-9);
  EXPECT_NEAR(pc->bInterceptMs + pc->bSlopeMsPerByte * S, pc->tBMs, 1e-9);
  // K=4 scatter: delivery 30 GB/s aggregate. B ships S at 30 -> 33.3ms;
  // A CPU still 50ms (flat in K), A DMA r*S at 30 -> 8.3ms -> max = CPU.
  auto pc4 = pathCosts(m, Pattern::Contiguous, S, 0.25, 8, 4, false);
  ASSERT_TRUE(pc4.has_value());
  EXPECT_NEAR(pc4->tBMs, 0.1 + 1000.0 / 30.0, 1e-6);
  EXPECT_NEAR(pc4->tAMs, 0.5 + 50.0, 1e-6);
  // K=4 broadcast: B ships 4S -> 133.3ms; A ships 4*r*S=1S -> 33.3ms DMA,
  // CPU 50 -> max 50.
  auto pb4 = pathCosts(m, Pattern::Contiguous, S, 0.25, 8, 4, true);
  ASSERT_TRUE(pb4.has_value());
  EXPECT_NEAR(pb4->tBMs, 0.1 + 4000.0 / 30.0, 1e-6);
  EXPECT_NEAR(pb4->tAMs, 0.5 + 50.0, 1e-6);
  // Missing K key -> nullopt.
  EXPECT_FALSE(pathCosts(m, Pattern::Contiguous, S, 0.25, 8, 2, false)
                   .has_value());
}
```

- [ ] **Step 2: Run to verify failure**

Run: `ninja -C build/sym libreloc-test 2>&1 | tail -5`
Expected: link errors — `classify`, `cpuBw`, `pathCosts` undefined.

- [ ] **Step 3: Implement** (append to `CostModel.cpp`, above the closing namespaces)

```cpp
const char *patternName(Pattern p) {
  switch (p) {
  case Pattern::Contiguous:
    return "contiguous";
  case Pattern::Blocked:
    return "blocked";
  case Pattern::SingleElement:
    return "single_element";
  case Pattern::Tiled:
    return "tiled";
  }
  return "?";
}

Pattern classify(const BoundPlan &b) {
  const int64_t totalElems =
      b.elementSize ? b.totalBytes / b.elementSize : 0;
  if (b.L >= totalElems && totalElems > 0)
    return Pattern::Contiguous;
  if (b.L <= 1)
    return Pattern::SingleElement;
  return b.L >= kBlockedRunFloor ? Pattern::Blocked : Pattern::Tiled;
}

// Key helper: cpu.t{threads}.{pattern}.{kernel}_gbps
static std::optional<double> kern(const CostModel &m, Pattern p, int threads,
                                  const char *kernel) {
  const std::string key = "cpu.t" + std::to_string(threads) + "." +
                          patternName(p) + "." + kernel + "_gbps";
  if (!m.has(key))
    return std::nullopt;
  return m.at(key);
}

static double harmonic(double a, double b) {
  return 1.0 / (1.0 / a + 1.0 / b);
}

std::optional<double> cpuBw(const CostModel &m, Pattern p, double r,
                            int threads) {
  const bool contig = p == Pattern::Contiguous;
  auto need = [&](const char *k) { return kern(m, p, threads, k); };
  if (r == 1.0)
    return contig ? need("contig_read") : need("gather_f32");
  if (r == 0.5) {
    auto conv = need("convert_f32_f16");
    if (contig)
      return conv;
    auto g = need("gather_f32");
    if (!g || !conv)
      return std::nullopt;
    return harmonic(*g, *conv);
  }
  if (r == 0.25)
    return contig ? need("quantize_pack") : need("gather_quantize");
  if (r == 0.125) {
    auto base = contig ? need("quantize_pack") : need("gather_quantize");
    auto pack = need("pack_s8_s4");
    if (!base || !pack)
      return std::nullopt;
    // pack reads S/4 bytes: source-normalized BW is 4x its input BW.
    return harmonic(*base, 4.0 * *pack);
  }
  return std::nullopt; // only the measured r grid is modelled in v0
}

static std::optional<double> deliveryGbps(const CostModel &m, int K) {
  if (K <= 1)
    return m.has("pcie.h2d_gbps") ? std::optional<double>(m.at(
                                        "pcie.h2d_gbps"))
                                  : std::nullopt;
  const std::string key = "multigpu.delivery_gbps.k" + std::to_string(K);
  if (!m.has(key))
    return std::nullopt;
  return m.at(key);
}

std::optional<PathCosts> pathCosts(const CostModel &m, Pattern p,
                                   int64_t srcBytes, double r, int threads,
                                   int K, bool broadcast) {
  if (srcBytes <= 0 || r <= 0 || K < 1)
    return std::nullopt;
  auto bwCpu = cpuBw(m, p, r, threads);
  auto bwDel = deliveryGbps(m, K);
  if (!bwCpu || !bwDel || *bwCpu <= 0 || *bwDel <= 0)
    return std::nullopt;
  if (!m.has("hbm.bw_gbps"))
    return std::nullopt;
  const std::string mKey = std::string("hbm.m.") + patternName(p);
  if (!m.has(mKey))
    return std::nullopt;
  const double bwHbm = m.at("hbm.bw_gbps");
  const double mm = m.at(mKey);
  if (bwHbm <= 0 || mm <= 0)
    return std::nullopt;

  // Bytes on the wire per method (whole delivery, all K receivers).
  const double kMult = broadcast ? static_cast<double>(K) : 1.0;
  // ms per byte at BW gbps: 1e3 / (BW * 1e9) = 1e-6 / BW.
  auto msPerByteAt = [](double gbps) { return 1e-6 / gbps; };

  PathCosts pc;
  // A: pipelined max(CPU pass over S source bytes, DMA of kMult*r*S).
  const double aCpuSlope = msPerByteAt(*bwCpu);            // per source byte
  const double aDmaSlope = kMult * r * msPerByteAt(*bwDel); // per source byte
  pc.aSlopeMsPerByte = std::max(aCpuSlope, aDmaSlope);
  pc.aInterceptMs = m.get("overhead.a_ms", 0.0);
  // B: max(DMA of kMult*S, GPU transform m*kMult*S over HBM).
  const double bDmaSlope = kMult * msPerByteAt(*bwDel);
  const double bHbmSlope = kMult * mm * msPerByteAt(bwHbm);
  pc.bSlopeMsPerByte = std::max(bDmaSlope, bHbmSlope);
  pc.bInterceptMs = m.get("overhead.b_ms", 0.0);
  pc.tAMs = pc.aInterceptMs + pc.aSlopeMsPerByte * srcBytes;
  pc.tBMs = pc.bInterceptMs + pc.bSlopeMsPerByte * srcBytes;
  return pc;
}
```

Add `#include <algorithm>` if `std::max` needs it.

- [ ] **Step 4: Run tests, verify pass**

Run: `$(find build/sym -name libreloc-test -type f | head -1) --gtest_filter='CostModel*'`
Expected: all pass (Parse 5 + Classify 1 + CpuBw 1 + PathCosts 1).

- [ ] **Step 5: clang-format + commit**

```bash
$SP/fmt-venv/bin/clang-format -i libreloc/src/CostModel.cpp libreloc/test/CostModelTest.cpp
git add -u && git commit -m "feat(libreloc): costmodel classifier + two-path costs (#97)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: `decide()` + single-symbol threshold (TDD, brute-force agreement)

**Files:**
- Modify: `libreloc/src/CostModel.cpp`
- Modify: `libreloc/test/CostModelTest.cpp`

**Interfaces:**
- Consumes: Task 2's `pathCosts`; `reloc::prefold::prefoldWins(int64_t, double, double, double)` (`reloc/Prefold.h`).
- Produces: `methodName(MethodDecision::Method) -> "a"|"b"|"a_prefold"`; `decide(model, pattern, srcBytes, r, threads, K, nReuse, broadcast) -> optional<MethodDecision>`. Semantics: arms are affine lines `t(S)`; A-arm is amortized when `nReuse >= 1` — per-load `tAPre(S) = overhead.a_ms + dmaSlope*S + (cpuSlope*S + allocMs(r*S))/nReuse` with `allocMs = prefold.alloc_ms_per_gib * (r*S/2^30)` (key default 0) — and the A-vs-APrefold pick delegates to `prefoldWins(nReuse, tTransform, tTransform + allocMs, 0)`; `thresholdBytes` = the S where the argmin over the active arms changes, chosen as the boundary nearest the bound `srcBytes`; `-1` when the decision is size-independent.

- [ ] **Step 1: Write the failing tests** (append)

```cpp
using reloc::costmodel::decide;
using reloc::costmodel::MethodDecision;

TEST(CostModelDecide, PicksWinnerAndThreshold) {
  CostModel m = mustParse(kSynth);
  // Contiguous r=0.25: slopes A=0.05ms/MB? (see PathCosts test) -- A slope
  // 5e-8 ms/B, B slope 1e-7 ms/B; intercepts A=0.5, B=0.1. Lines cross at
  // S* = (0.5-0.1)/(1e-7-5e-8) = 8e6 bytes. Below: B wins; above: A.
  auto small = decide(m, Pattern::Contiguous, 1 << 20, 0.25, 8);
  ASSERT_TRUE(small.has_value());
  EXPECT_EQ(small->method, MethodDecision::Method::B);
  EXPECT_NEAR(small->thresholdBytes, 8e6, 1.0);
  auto big = decide(m, Pattern::Contiguous, 1 << 30, 0.25, 8);
  ASSERT_TRUE(big.has_value());
  EXPECT_EQ(big->method, MethodDecision::Method::A);
  EXPECT_NEAR(big->thresholdBytes, 8e6, 1.0);
}

TEST(CostModelDecide, SizeIndependentDecisionHasNoThreshold) {
  // Same slopes ordering as intercepts ordering -> no crossing.
  const char *cal = R"(# costmodel calibration v0
pcie.h2d_gbps 10
cpu.t8.contiguous.quantize_pack_gbps 40
hbm.bw_gbps 100
hbm.m.contiguous 1
overhead.a_ms 0.05
overhead.b_ms 0.1
)";
  CostModel m = mustParse(cal);
  // A slope max(1/40, .25/10)=0.025 < B slope 0.1; A intercept smaller too.
  auto d = decide(m, Pattern::Contiguous, 1 << 20, 0.25, 8);
  ASSERT_TRUE(d.has_value());
  EXPECT_EQ(d->method, MethodDecision::Method::A);
  EXPECT_DOUBLE_EQ(d->thresholdBytes, -1);
}

TEST(CostModelDecide, PrefoldArmDelegatesToV4Rule) {
  CostModel m = mustParse(kSynth);
  // nReuse=16 amortizes the 50ms CPU pass to ~3.1ms/load: APrefold wins.
  auto d = decide(m, Pattern::Contiguous, 1000000000, 0.25, 8, 1, 16);
  ASSERT_TRUE(d.has_value());
  EXPECT_EQ(d->method, MethodDecision::Method::APrefold);
  EXPECT_LT(d->tAMs, 30.0); // amortized per-load, way under B's 100ms
  // nReuse=1 (cold single-use): prefoldWins says no -> plain A vs B.
  auto d1 = decide(m, Pattern::Contiguous, 1000000000, 0.25, 8, 1, 1);
  ASSERT_TRUE(d1.has_value());
  EXPECT_NE(d1->method, MethodDecision::Method::APrefold);
}

TEST(CostModelDecide, ThresholdAgreesWithBruteForce) {
  // Issue #97 acceptance: threshold precompute vs brute-force agreement.
  CostModel m = mustParse(kSynth);
  for (double r : {1.0, 0.5, 0.25, 0.125}) {
    for (Pattern p : {Pattern::Contiguous, Pattern::Blocked}) {
      auto probe = decide(m, p, 1 << 20, r, 8);
      if (!probe.has_value())
        continue;
      const double thr = probe->thresholdBytes;
      for (int64_t S = 1 << 12; S <= (1ll << 34); S <<= 1) {
        auto d = decide(m, p, S, r, 8);
        ASSERT_TRUE(d.has_value());
        auto pc = pathCosts(m, p, S, r, 8, 1, false);
        const auto brute = pc->tAMs <= pc->tBMs
                               ? MethodDecision::Method::A
                               : MethodDecision::Method::B;
        EXPECT_EQ(d->method, brute)
            << patternName(p) << " r=" << r << " S=" << S;
        if (thr > 0) {
          // The stored boundary itself separates the regimes.
          const bool aboveThr = static_cast<double>(S) > thr;
          EXPECT_EQ(d->method == MethodDecision::Method::A, aboveThr ||
                        (pc->aSlopeMsPerByte < pc->bSlopeMsPerByte) ==
                            aboveThr)
              << "boundary inconsistent at S=" << S;
        }
      }
    }
  }
}
```

(The last EXPECT in the brute-force loop is awkward as written — implementer: replace it with the direct check `EXPECT_EQ(d->method == MethodDecision::Method::A, pc->tAMs <= pc->tBMs)` and separately assert that scanning S across `thr` flips the method exactly once. Keep the loop bounds.)

- [ ] **Step 2: Run to verify failure** — link errors for `decide`/`methodName`.

- [ ] **Step 3: Implement** (append to `CostModel.cpp`)

```cpp
const char *methodName(MethodDecision::Method m) {
  switch (m) {
  case MethodDecision::Method::A:
    return "a";
  case MethodDecision::Method::B:
    return "b";
  case MethodDecision::Method::APrefold:
    return "a_prefold";
  }
  return "?";
}

std::optional<MethodDecision> decide(const CostModel &m, Pattern p,
                                     int64_t srcBytes, double r, int threads,
                                     int K, int64_t nReuse, bool broadcast) {
  auto pc = pathCosts(m, p, srcBytes, r, threads, K, broadcast);
  if (!pc)
    return std::nullopt;

  MethodDecision d;
  d.pattern = p;
  d.k = K;
  d.nReuse = nReuse;

  // Active A-arm: plain A, or amortized prefold when the V4 rule says the
  // fold pays for itself (delegation, not reimplementation).
  double aInt = pc->aInterceptMs, aSlope = pc->aSlopeMsPerByte;
  bool prefold = false;
  if (nReuse >= 1) {
    // Decompose A's slope back into its cpu and dma parts.
    auto bwCpuV = cpuBw(m, p, r, threads);
    const double cpuSlope = 1e-6 / *bwCpuV; // ms per source byte
    const double dmaSlope = pc->aSlopeMsPerByte; // max(cpu, dma)...
    // Recompute the dma-only slope explicitly:
    const double kMult = broadcast ? static_cast<double>(K) : 1.0;
    auto bwDel = K <= 1 ? m.at("pcie.h2d_gbps")
                        : m.at("multigpu.delivery_gbps.k" +
                               std::to_string(K));
    const double dmaOnly = kMult * r * 1e-6 / bwDel;
    (void)dmaSlope;
    const double tTransform = cpuSlope * srcBytes;
    const double allocMs =
        m.get("prefold.alloc_ms_per_gib", 0.0) *
        (r * static_cast<double>(srcBytes) / (1ll << 30));
    if (reloc::prefold::prefoldWins(nReuse, tTransform,
                                    tTransform + allocMs, 0.0)) {
      prefold = true;
      aSlope = dmaOnly + (cpuSlope +
                          (allocMs / static_cast<double>(srcBytes))) /
                             static_cast<double>(nReuse);
      aInt = m.get("overhead.a_ms", 0.0);
    }
  }

  const double tA = aInt + aSlope * srcBytes;
  const double tB = pc->tBMs;
  d.tAMs = tA;
  d.tBMs = tB;
  d.method = tA <= tB ? (prefold ? MethodDecision::Method::APrefold
                                 : MethodDecision::Method::A)
                      : MethodDecision::Method::B;

  // Single-symbol threshold: the two active arms are affine in S; the
  // boundary is their crossing when it lies in (0, inf) and the slopes
  // actually cross (otherwise the decision is size-independent).
  const double dSlope = aSlope - pc->bSlopeMsPerByte;
  const double dInt = pc->bInterceptMs - aInt;
  if (dSlope != 0.0) {
    const double sStar = dInt / dSlope;
    d.thresholdBytes = sStar > 0 ? sStar : -1;
  }
  return d;
}
```

(Implementer note: simplify/clean the decomposition — the intent is exact: `aSlope` for the prefold arm = per-load DMA slope + (cpu slope + alloc-per-byte)/nReuse. Keep unconditional key checks: `m.at` calls above are guarded because `pathCosts` succeeded, which required those keys.)

- [ ] **Step 4: Run tests, verify all `CostModel*` pass.**

- [ ] **Step 5: Run the FULL suite** (`ctest --test-dir build/sym -R 'libreloc-test|reloc-runtime' --output-on-failure`) — MLIR-free contract must stay green.

- [ ] **Step 6: clang-format + commit**

```bash
git add -u && git commit -m "feat(libreloc): costmodel decide() + single-symbol threshold (#97)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: `bind()` integration

**Files:**
- Modify: `libreloc/include/reloc/Bind.h`
- Modify: `libreloc/src/Bind.cpp`
- Modify: `libreloc/test/BindTest.cpp`
- Modify: `libreloc/include/reloc/CostModel.h` (only if a forward-decl shim is needed)

**Interfaces:**
- Consumes: Tasks 1–3 (`CostModel`, `classify`, `decide`, `MethodDecision`).
- Produces: `bind(const RelocationPlan &, const SymbolMap &, Strategy override = Strategy::Auto, const costmodel::CostModel *model = nullptr, double wireRatio = 1.0, int K = 1, int64_t nReuse = -1)` — extra params defaulted so every existing caller compiles unchanged; `BoundPlan` gains `std::optional<costmodel::MethodDecision> decision;`. Layering: `Bind.h` must NOT include `CostModel.h` (cycle). Move `MethodDecision` + `Pattern` + `patternName` + `methodName` into a new tiny header `libreloc/include/reloc/MethodDecision.h` (no includes beyond `<cstdint>`), have both `Bind.h` and `CostModel.h` include it, and forward-declare `namespace costmodel { class CostModel; }` in `Bind.h`. Strategy-Auto resolution reads `strategy.single_thread_max_bytes` / `strategy.multi_thread_max_bytes` from the model when present (falling back to the P2 constants `kL2Bytes`/`kMultiThreadMaxBytes`).

- [ ] **Step 1: Write the failing test** (append to `libreloc/test/BindTest.cpp`; follow that file's existing fixture style — it builds `RelocationPlan`s from hex or via helpers; use the identity-plan golden already present in the file, `kIdentityHex`-style — read the file first and reuse its smallest bindable fixture)

```cpp
TEST(Bind, CostModelDecisionPopulatedWhenModelPassed) {
  // Reuse the file's existing smallest decodable plan fixture; bind it
  // with a synthetic calibration and assert the decision rides along.
  const char *cal = "# costmodel calibration v0\n"
                    "pcie.h2d_gbps 10\n"
                    "cpu.t8.contiguous.contig_read_gbps 20\n"
                    "hbm.bw_gbps 100\n"
                    "hbm.m.contiguous 1\n";
  auto cmv = reloc::costmodel::CostModel::parse(cal);
  ASSERT_TRUE(std::holds_alternative<reloc::costmodel::CostModel>(cmv));
  const auto &cm = std::get<reloc::costmodel::CostModel>(cmv);
  // <fixture>: decode + bind the identity golden with model attached.
  auto result = reloc::bind(<decoded identity plan>, {{"N", 64}},
                            reloc::Strategy::Auto, &cm, /*wireRatio=*/1.0);
  auto *bound = std::get_if<reloc::BoundPlan>(&result);
  ASSERT_NE(bound, nullptr);
  ASSERT_TRUE(bound->decision.has_value());
  EXPECT_GT(bound->decision->tBMs, 0.0);
}

TEST(Bind, NoModelLeavesDecisionEmptyAndStrategyUnchanged) {
  auto result = reloc::bind(<decoded identity plan>, {{"N", 64}});
  auto *bound = std::get_if<reloc::BoundPlan>(&result);
  ASSERT_NE(bound, nullptr);
  EXPECT_FALSE(bound->decision.has_value());
}
```

(`<decoded identity plan>` is a placeholder for the file's existing fixture variable — the implementer substitutes the real one after reading `BindTest.cpp:30-110`; both bind-error tests there show the invocation shape.)

- [ ] **Step 2: Run to verify failure** (no `decision` member, no 4-arg bind).

- [ ] **Step 3: Implement**

1. Create `libreloc/include/reloc/MethodDecision.h` containing `Pattern`, `patternName`, `kBlockedRunFloor`, `MethodDecision`, `methodName` (moved verbatim from `CostModel.h`; namespace `reloc::costmodel`; include guard `RELOC_METHODDECISION_H`; only `<cstdint>` included).
2. `CostModel.h`: include `reloc/MethodDecision.h`, drop the moved decls.
3. `Bind.h`: include `reloc/MethodDecision.h`; add `namespace costmodel { class CostModel; }` forward decl; add to `BoundPlan`: `std::optional<costmodel::MethodDecision> decision;` (include `<optional>`); extend `bind`'s signature with the defaulted params listed in Interfaces.
4. `Bind.cpp`: include `reloc/CostModel.h`. At the end of a successful bind (after strategy resolution), when `model != nullptr`:

```cpp
  if (model) {
    const costmodel::Pattern pat = costmodel::classify(bound);
    if (auto d = costmodel::decide(*model, pat, bound.totalBytes, wireRatio,
                                   /*threads=*/8, K, nReuse))
      bound.decision = *d;
    // Strategy-Auto thresholds come from the calibration when present.
  }
```

   And in the Auto-strategy resolution replace the two constants:

```cpp
  const double singleMax =
      model ? model->get("strategy.single_thread_max_bytes",
                         static_cast<double>(kL2Bytes))
            : static_cast<double>(kL2Bytes);
  const double multiMax =
      model ? model->get("strategy.multi_thread_max_bytes",
                         static_cast<double>(kMultiThreadMaxBytes))
            : static_cast<double>(kMultiThreadMaxBytes);
```

   (Adapt to the actual local structure at `Bind.cpp:315-327`; the strategy order and `noCopy` short-circuit stay exactly as they are.)

- [ ] **Step 4: Run BindTest + full libreloc suite; verify pass.**

- [ ] **Step 5: Build the CUDA-off whole tree** (`ninja -C build/sym 2>&1 | tail -3`) — catches pyreloc/binding compile breakage from the `BoundPlan` change.

- [ ] **Step 6: clang-format + commit**

```bash
git add -u libreloc && git add libreloc/include/reloc/MethodDecision.h
git commit -m "feat(libreloc): bind() consumes the cost model; decision on BoundPlan (#97)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: pyreloc surface — `load_calibration` + `predict`

**Files:**
- Modify: `libreloc/python/PyReloc.cpp`
- Create: `libreloc/python/tests/test_costmodel.py`
- Modify: `libreloc/python/pyreloc/__init__.py` (re-export the two names, matching existing style)

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces (Python): `pyreloc.load_calibration(path: str) -> Calibration` (opaque class; `.machine` property; raises `ValueError` with the parser diagnostic on failure); `pyreloc.predict(calibration, *, pattern: str, src_bytes: int, r: float, threads: int = 8, k: int = 1, n_reuse: int = -1, broadcast: bool = False) -> dict` with keys `method` (`"a"|"b"|"a_prefold"`), `t_a_ms`, `t_b_ms`, `threshold_bytes`, `pattern`; raises `ValueError` when the calibration lacks needed keys (message names the pattern/r). Pattern strings = `patternName` values.

- [ ] **Step 1: Write the failing test**

Create `libreloc/python/tests/test_costmodel.py`:

```python
"""V3 (issue #97): cost-model bindings — the prediction test's foundation.
The Python surface is a thin veneer; all arithmetic is the C++ component."""
import pathlib

import pytest

pyreloc = pytest.importorskip("pyreloc")

CAL = """# costmodel calibration v0
# machine: testbox
pcie.h2d_gbps 10
cpu.t8.contiguous.contig_read_gbps 20
cpu.t8.contiguous.convert_f32_f16_gbps 20
cpu.t8.contiguous.quantize_pack_gbps 20
cpu.t8.contiguous.pack_s8_s4_gbps 20
hbm.bw_gbps 100
hbm.m.contiguous 1
overhead.a_ms 0.5
overhead.b_ms 0.1
"""


@pytest.fixture()
def cal(tmp_path):
    p = tmp_path / "test.cal"
    p.write_text(CAL)
    return pyreloc.load_calibration(str(p))


def test_load_calibration_machine(cal):
    assert cal.machine == "testbox"


def test_load_calibration_rejects_bad_file(tmp_path):
    p = tmp_path / "bad.cal"
    p.write_text("no header\n")
    with pytest.raises(ValueError):
        pyreloc.load_calibration(str(p))


def test_predict_matches_cpp_arithmetic(cal):
    # Mirrors CostModelPathCosts: S=1e9, r=0.25 -> A 50.5ms, B 100.1ms.
    d = pyreloc.predict(cal, pattern="contiguous", src_bytes=10**9, r=0.25)
    assert d["method"] == "a"
    assert d["t_a_ms"] == pytest.approx(50.5)
    assert d["t_b_ms"] == pytest.approx(100.1)
    assert d["threshold_bytes"] == pytest.approx(8e6, rel=1e-6)
    assert d["pattern"] == "contiguous"


def test_predict_prefold_arm(cal):
    d = pyreloc.predict(cal, pattern="contiguous", src_bytes=10**9, r=0.25,
                        n_reuse=16)
    assert d["method"] == "a_prefold"


def test_predict_missing_keys_raise(cal):
    with pytest.raises(ValueError):
        pyreloc.predict(cal, pattern="blocked", src_bytes=10**9, r=0.25)
```

- [ ] **Step 2: Run to verify failure**

```bash
SP=/tmp/claude-2017/-home-jueonpark-sym/595debf3-ca62-43ba-a9fe-225c7908aed7/scratchpad
PYTHONPATH=/home/jueonpark/sym/build/sym/python $SP/fmt-venv/bin/python3 -m pytest \
  libreloc/python/tests/test_costmodel.py -q
```
Expected: FAIL — `pyreloc` has no attribute `load_calibration`.

- [ ] **Step 3: Implement the bindings** (in `PyReloc.cpp`, after the `bind` registration; include `"reloc/CostModel.h"` at the top)

```cpp
  namespace cmns = reloc::costmodel;
  py::class_<cmns::CostModel>(m, "Calibration",
                              "Parsed costmodel calibration (issue #97).")
      .def_property_readonly("machine", &cmns::CostModel::machine);

  m.def(
      "load_calibration",
      [](const std::string &path) {
        auto r = cmns::CostModel::load(path);
        if (auto *err = std::get_if<std::string>(&r))
          throw py::value_error(*err);
        return std::get<cmns::CostModel>(r);
      },
      py::arg("path"),
      "Load a costmodel calibration (.cal). Raises ValueError with the "
      "parser diagnostic on invalid input.");

  m.def(
      "predict",
      [](const cmns::CostModel &cal, const std::string &pattern,
         int64_t srcBytes, double r, int threads, int k, int64_t nReuse,
         bool broadcast) {
        cmns::Pattern p;
        if (pattern == "contiguous")
          p = cmns::Pattern::Contiguous;
        else if (pattern == "blocked")
          p = cmns::Pattern::Blocked;
        else if (pattern == "single_element")
          p = cmns::Pattern::SingleElement;
        else if (pattern == "tiled")
          p = cmns::Pattern::Tiled;
        else
          throw py::value_error("unknown pattern '" + pattern + "'");
        auto d = cmns::decide(cal, p, srcBytes, r, threads, k, nReuse,
                              broadcast);
        if (!d)
          throw py::value_error(
              "calibration lacks keys for pattern '" + pattern +
              "' at r=" + std::to_string(r) +
              " t" + std::to_string(threads) + " k" + std::to_string(k));
        py::dict out;
        out["method"] = std::string(cmns::methodName(d->method));
        out["t_a_ms"] = d->tAMs;
        out["t_b_ms"] = d->tBMs;
        out["threshold_bytes"] = d->thresholdBytes;
        out["pattern"] = std::string(cmns::patternName(d->pattern));
        return out;
      },
      py::arg("calibration"), py::kw_only(), py::arg("pattern"),
      py::arg("src_bytes"), py::arg("r"), py::arg("threads") = 8,
      py::arg("k") = 1, py::arg("n_reuse") = -1,
      py::arg("broadcast") = false,
      "Run the C++ cost model. Returns {method, t_a_ms, t_b_ms, "
      "threshold_bytes, pattern}.");
```

Re-export in `pyreloc/__init__.py` following its existing list style: add `load_calibration`, `predict`, `Calibration`.

- [ ] **Step 4: Rebuild + run; verify all `test_costmodel.py` pass, then the whole pytest suite** (`... -m pytest libreloc/python/tests -q`) — no regressions.

- [ ] **Step 5: clang-format the .cpp + commit**

```bash
git add libreloc/python && git commit -m "feat(pyreloc): load_calibration + predict over the C++ cost model (#97)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Calibration assembler + committed `.cal` files

**Files:**
- Create: `bench/rtrack/make_calibration.py`
- Create: `calibration/epyc7351-2080ti.cal` (generated, committed)
- Create: `calibration/7800x3d-4070tis.cal` (generated, committed)

**Interfaces:**
- Consumes: committed `bench/results` artifacts (exact sources below); Task 1's format.
- Produces: `python3 bench/rtrack/make_calibration.py --machine epyc7351-2080ti --out calibration/epyc7351-2080ti.cal` — deterministic; every emitted line carries a trailing `# <source-file>` provenance comment. Key set (used by Tasks 2–3 and 7): `pcie.h2d_gbps`, `cpu.t{8,1}.{pattern}.{kernel}_gbps`, `hbm.bw_gbps`, `hbm.m.{pattern}`, `hiding.ratio`, `multigpu.delivery_gbps.k{2,4}` (epyc only), `overhead.{a,b}_ms`, `prefold.alloc_ms_per_gib` (epyc only, from V4), `strategy.single_thread_max_bytes`, `strategy.multi_thread_max_bytes` (seeded with the P2 defaults 262144 / 268435456 and commented as such — no small-size sweep exists yet).

**Source map (embed as a dict in the script; this is the frozen provenance):**

- epyc7351-2080ti:
  - `pcie.h2d_gbps 13.07` ← `bench/results/v1_gate_report.txt` (the V1 admissible anchor)
  - pattern rooflines T=8/T=1 ← `bench/results/r1_rooflines/r1_roofline_{kernel}[_{plan}]_t{T}.json` with plan→pattern mapping: files without a plan tag (contig_read, quantize_pack, convert_f32_f16) → `contiguous`; `gather_f32_blocked`/`gather_quantize_blocked?` → `blocked` (gather_quantize has transpose/nchw variants only — map `transpose`→`single_element`, `nchw`→`tiled`); read `kernels.<name>.in_gb_per_s`. Where a (pattern, kernel) pair has no measured file (e.g. `blocked.gather_quantize`), OMIT the key — the model returns nullopt and the gate reports the cell as unmodelable. `pack_s8_s4` exists only for `contiguous` (quant_bw measured it contiguously; note in provenance).
  - For `blocked.convert_f32_f16` and other second-pass kernels reuse the contiguous value with a provenance note (`# second pass is contiguous by construction`): the gather pass is the strided one; the convert/pack pass reads the gathered contiguous buffer.
  - `hbm.bw_gbps 544`, `hbm.m.contiguous 1.0` (copy_f32), `hbm.m.blocked 2.97` (relocate_naive), `hbm.m.single_element 1.43` (transpose_smem_padded — T1's exact-transpose SMEM path), `hbm.m.tiled 2.97` ← `bench/results/r4_hiding_ratio_epyc_2080ti.json`
  - `hiding.ratio 41.7` ← same
  - `multigpu.delivery_gbps.k2 21.02`, `.k4 41.96` ← `docs/m0-2080ti-bringup.md` numbers as recorded in `bench/results/m0_multigpu_h2d_*.json`
  - `prefold.alloc_ms_per_gib` ← derived from `bench/results/v4_scatter_n8192_epyc_2080ti.json` reuse rows: `(t_prefold_cold_ms − t_transform_ms) / (r·S in GiB)` at K=1 (r·S = 64 MiB → value ≈ (72.94−12.35)/0.0625 — the script computes it, does not hardcode)
  - `overhead.a_ms` / `overhead.b_ms` ← intercept of a two-point linear fit over the committed `v1_gen3_nsweep_epyc_2080ti.csv` T3 rows (best-chunk medians at N=2048 and N=16384, method `a` and `b_fair`); clamp at ≥0.
- 7800x3d-4070tis: same structure from `bench/results/v1_gen4_gate_report.txt` (pinned H2D), `bench/results/r2_rooflines/*.json` (per-plan files: `identity`→contiguous, `blocked`→blocked; plan→pattern as above; only t8 + t1 where present), `v1_gen4_matrix_nsweep_rerun_*.csv` for overheads. No multigpu/hbm data exists for that box — omit those keys (single-GPU cells only; the gate reports what was unmodelable). `hbm.m.*`/`hbm.bw_gbps`: copy the 2080 Ti values with an explicit provenance note `# proxy: no Gen4 HBM sweep exists (R4 ran on 2080 Ti); B is link-bound whenever m < ratio, so this only matters if m/BW_hbm exceeds 1/BW_link` — honest and stated.

- [ ] **Step 1: Write the script** (stdlib only, deterministic; `--machine`, `--out`; hard-fails if a listed source file is missing; prints nothing but the output path). Structure: `SOURCES = {machine: {...}}` dict at top; helper `emit(key, value, source)` accumulating lines; writes header (`# costmodel calibration v0`, `# machine:`, `# generated by make_calibration.py from committed bench/results — regenerate, do not hand-edit`).

- [ ] **Step 2: Generate both files; parse them back through the C++ parser via pyreloc as the round-trip check**

```bash
python3 bench/rtrack/make_calibration.py --machine epyc7351-2080ti --out calibration/epyc7351-2080ti.cal
python3 bench/rtrack/make_calibration.py --machine 7800x3d-4070tis --out calibration/7800x3d-4070tis.cal
PYTHONPATH=build/sym/python $SP/fmt-venv/bin/python3 - <<'EOF'
import pyreloc
for p in ("calibration/epyc7351-2080ti.cal", "calibration/7800x3d-4070tis.cal"):
    c = pyreloc.load_calibration(p)
    print(p, "->", c.machine)
EOF
```
Expected: both load; machines echo. Re-run the generator and `git diff --exit-code calibration/` to prove determinism.

- [ ] **Step 3: Commit**

```bash
git add bench/rtrack/make_calibration.py calibration/
git commit -m "feat(bench): calibration assembler + committed .cal for both boxes (#97)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Pre-registered prediction gate + pytest over committed cells

**Files:**
- Create: `bench/rtrack/v3_gate.py`
- Create: `libreloc/python/tests/test_prediction.py`

**Interfaces:**
- Consumes: Task 5's `pyreloc.predict`, Task 6's `.cal` files, committed CSVs/JSONs.
- Produces: `python3 bench/rtrack/v3_gate.py --report <json>` printing the three verdicts; `test_prediction.py` builds the report JSON at `bench/results/v3_prediction_report.json` (single source both consume). **Ordering discipline: `v3_gate.py` (with its bars) is committed in a standalone commit BEFORE `test_prediction.py` first runs.**

**Bars (verbatim from #97, fixed in `v3_gate.py`):** `MISCLASS_BAR = 0.15` (winner misclassification rate over all modelable cells); `RSTAR_ABS_BAR = 0.15` (|r*_pred − r*_meas|, families where both exist); `REGRET_P90_BAR = 0.20` (regret = (T_chosen − T_oracle)/T_oracle from measured medians).

**Cell inventory (frozen in `test_prediction.py`):**

1. Gen3 single-GPU matrix — `bench/results/v1_gen3_nsweep_epyc_2080ti.csv`: per (transform, N), best-chunk medians of `a` vs `b_fair`; model inputs: S=N²·4, family map r/pattern: `transpose`→(1.0, single_element), `blocked_transpose`→(1.0, blocked), `transpose_quant`→(0.25, single_element), `nchw_nhwc_quant`→(0.25, tiled), `quant`→(0.25, contiguous), `convert_f16`→(0.5, contiguous); threads=8; cal=epyc.
2. Gen4 single-GPU matrix — `bench/results/v1_gen4_matrix_nsweep_rerun_7800x3d_4070tis.csv`, same treatment, cal=7800x3d.
3. r\* — `bench/results/v1_gen4_rstar_bfair.json` (quant 0.597-ish, blocked measured) + `bench/results/v2_isa_gen3_rstar_avx2_epyc7351-2080ti.json` (quant 0.636): model r\*_pred = crossing of predicted speedup curve over the same r grid (compute via `predict` at each measured r, speedup = t_b(r=1)/t_a(r); reuse figure_rstar's log-interp crossing formula inline).
4. Multi-GPU — `bench/results/r3_scatter_n8192_epyc_2080ti.json`, `r3_scatter_n16384_epyc_2080ti.json`, `r3_broadcast_n8192_epyc_2080ti.json`, plus `v5_broadcast_contig_epyc_2080ti.json`: winner a-vs-bxk per (scenario, K, N); model with k=K, broadcast flag, pattern: scatter→contiguous, broadcast→blocked, broadcast_contig→contiguous; wall medians as T_oracle inputs.
5. Ablation: from the same cells — mean regret of {model, always-A, always-B} vs oracle.

Cells whose keys are missing from the calibration are counted and listed as `unmodelable` (reported, not silently dropped; they do NOT count toward misclassification but their count appears in the report and doc).

- [ ] **Step 1: Write `v3_gate.py`** (stdlib only; reads the report JSON; prints per-bar verdict table + the misclassified cell list + ablation table; exit 0 always — verdicts are output). Structure mirrors `exp4v_gate.py`.

- [ ] **Step 2: COMMIT the gate before any prediction run**

```bash
git add bench/rtrack/v3_gate.py
git commit -m "bench: pre-register V3 prediction bars (#97)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 3: Write `test_prediction.py`** — walks the cell inventory, calls `pyreloc.predict`, writes `bench/results/v3_prediction_report.json` with per-cell rows `{cell_id, machine, family, N|K, winner_measured, winner_predicted, t_a_meas, t_b_meas, t_a_pred, t_b_pred, regret, modelable}` + the r\* block + ablation block; asserts only structural sanity (≥30 modelable cells, report written) — the BARS are judged by `v3_gate.py`, keeping measurement and judgment separate.

- [ ] **Step 4: Run it; then run the gate**

```bash
PYTHONPATH=build/sym/python $SP/fmt-venv/bin/python3 -m pytest \
  libreloc/python/tests/test_prediction.py -q
python3 bench/rtrack/v3_gate.py --report bench/results/v3_prediction_report.json | tee bench/results/v3_gate_report.txt
```
Expected: report JSON + verdicts. **Whatever they say is the result** — misses get explained in Task 9's doc, never refit.

- [ ] **Step 5: Commit report + test**

```bash
git add libreloc/python/tests/test_prediction.py bench/results/v3_prediction_report.json bench/results/v3_gate_report.txt
git commit -m "update(bench): V3 prediction test over committed R-track cells — verdicts recorded (#97)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: Compiler-emitted plan row (title-gap closure)

**Files:**
- Modify: `libreloc/test/corpus/generate_corpus.py` (add one curated case)
- Create: `libreloc/test/corpus/blocked_transpose_sym.bin` + `.json` (generated, committed)
- Modify: `bench/rtrack/rtrack_bench.cu` (`--plan-wire` source)
- Create: `bench/results/v3_wire_row_epyc_2080ti.csv` (measured)

**Interfaces:**
- Consumes: `sym-opt` at `build/sym/sym/tools/sym-opt`; corpus generator flow; Task 4's bind-with-model (bench uses model=nullptr — the model prediction for this row is checked in `test_prediction.py`'s cell list by adding the wire row's CSV to inventory... keep simpler: the row's decision is exercised via pyreloc in the corpus test below).
- Produces: corpus case `blocked_transpose_sym` — `src_shape=["N","N"]`, `ops=[reshape(["N floordiv 64", 64, "N floordiv 64", 64]), transpose([2,0,1,3])]`, `symbols={"N": {"multiple_of": 64, "min_factor": 1, "max_factor": 3}}` (mirrors the committed concrete `blocked_reference` with 2 → N/64); `bench-rtrack --plan-wire <path.bin>` runs the standard T1b-shaped measurement (GatherF32 / Relocate / F32, r=1.0) on the decoded+bound plan at `--n`.

- [ ] **Step 1: Add the corpus case + regenerate**

Append to `curated_cases()` in `generate_corpus.py`:

```python
        dict(name="blocked_transpose_sym", dtype="f32",
             src_shape=["N", "N"],
             ops=[r(["N floordiv 64", 64, "N floordiv 64", 64]),
                  t([2, 0, 1, 3])],
             symbols=div64),
```

Regenerate (the script's documented invocation — read its `--help`; it needs `sym-opt` on PATH or via flag): commit ONLY the new pair (`git status` must show no diff to existing corpus entries — determinism check).

- [ ] **Step 2: Verify the new entry against the numpy oracle + the plan verifier**

```bash
PYTHONPATH=build/sym/python $SP/fmt-venv/bin/python3 -m pytest \
  libreloc/python/tests/test_corpus.py libreloc/python/tests/test_oracle.py -q
```
Expected: pass, with the new entry picked up by the corpus fixtures (oracle byte-compares the fold-produced plan against a numpy replay — the #63-class check, by construction).

- [ ] **Step 3: Add `--plan-wire` to `rtrack_bench.cu`**

- Options: `const char *planWire = nullptr;` + CLI `--plan-wire <path>` + usage line.
- In `buildFixture` (or a small pre-step in `run()`): when set, read the file bytes, `reloc::decodePlan` (error → exit 1 with the diagnostic), `reloc::bind(plan, {{"N", opt.n}})` (BindError → exit 1), and use the resulting `BoundPlan` in place of `w.makePlan(n)` for a synthetic workload entry with `id="TW"`, `transform="blocked_transpose_wire"`, `cpuStage=GatherF32`, `gpuStage=Relocate`, `dtypeOut=F32`, `r=1.0`. Everything downstream (verify gate, chunking, CSV) is unchanged — the wire plan must pass the existing packed-dst fixture checks (it does: fold output is packed).
- Constraint: `--plan-wire` is only valid with `--transform TW` (reject other combinations with a usage error) — keeps the flag from silently interacting with the matrix sweeps.

- [ ] **Step 4: Build (standalone nvcc recipe from `bench/rtrack/README.md`, sm_75) and measure the row on this box**

```bash
# session ritual: governor performance, persistence on, taskset 4-7,20-23
taskset -c 4-7,20-23 ./bench-rtrack --plan-wire libreloc/test/corpus/blocked_transpose_sym.bin \
  --transform TW --method all --n 8192 --chunk-mib 4,16,64,256 --threads 8 \
  --warmup 5 --iters 30 --machine epyc7351-2080ti \
  --csv bench/results/v3_wire_row_epyc_2080ti.csv --csv-header
```
Expected: `[verified]` on every row (bit-exact against the scalar reference of the DECODED plan — the verify gate itself is the correctness statement).

- [ ] **Step 5: Acceptance comparison** (script inline, no new file): best-chunk medians of the wire row vs the committed hand-authored T1b cell at N=8192/T=8 from `v1_gen3_nsweep_epyc_2080ti.csv` — PASS iff each method's median falls within the committed row's [min, p95]. Record the comparison numbers for Task 9's doc. If it fails, that is a finding to report (same-day rerun of the hand-authored row as control), not to hide.

- [ ] **Step 6: Commit**

```bash
git add libreloc/test/corpus/blocked_transpose_sym.* libreloc/test/corpus/generate_corpus.py \
  bench/rtrack/rtrack_bench.cu bench/results/v3_wire_row_epyc_2080ti.csv
git commit -m "feat(bench, corpus): compiler-emitted symbolic plan row — fold->wire->bind->measure (#97)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 9: Report, spec amendment, README, PR handoff

**Files:**
- Create: `docs/v3-costmodel.md`
- Modify: `docs/superpowers/specs/2026-07-29-v3-costmodel-design.md` (record the overhead-intercept refinement from this plan's header note, replacing the "breakpoints where the min/max arms switch" sentence)
- Modify: `bench/rtrack/README.md` (calibration assembler + v3_gate + --plan-wire, one entry each)
- Modify: `README.md` only if it lists components (check; likely no change)

**Steps:**

- [ ] **Step 1: Write `docs/v3-costmodel.md`** following the R-track report structure: verdict-first (the three bars' verdicts); component description + calibration provenance; the prediction table (per-cell winners, misclassified cells EXPLAINED, unmodelable cells listed); r\* comparison table; ablation (model vs always-A vs always-B vs oracle); the compiler-emitted row comparison; caveats (Gen4 HBM proxy keys, overhead intercepts from two-point fits, r grid restricted to measured points, strategy thresholds seeded with P2 defaults).
- [ ] **Step 2: Amend the spec's threshold paragraph** (cite this plan; two sentences).
- [ ] **Step 3: Full verification**: `ninja -C build/sym && ctest --test-dir build/sym --output-on-failure` and the full pytest suite; clang-format check over all touched `.cpp/.h` (`--dry-run --Werror`).
- [ ] **Step 4: Commit docs; DO NOT push or open a PR** — report to the controller/user with the gate verdicts; push/PR happens via finishing-a-development-branch with user authorization.

```bash
git add docs/v3-costmodel.md docs/superpowers/specs/2026-07-29-v3-costmodel-design.md bench/rtrack/README.md
git commit -m "docs: V3 cost-model report — prediction verdicts + wire-row closure (#97)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Self-Review Notes

- **Spec coverage**: calibration schema+assembler (T1, T6), classifier/model/multi-GPU/prefold arm (T2, T3), threshold precompute + brute-force acceptance (T3), bind hook incl. strategy-constant sourcing (T4), pyreloc single-source-of-truth (T5), pre-registered bars committed before running + ablation + wrong-cells-reported (T7), compiler-emitted row (T8), report + spec amendment (T9). The spec's degenerate-threshold issue is called out in the header and fixed via overhead intercepts.
- **Placeholder scan**: Task 4's `<decoded identity plan>` is an explicit fixture-substitution instruction with the file/lines to read — the only intentional one; Task 6's script body is specified by its SOURCES contract rather than full code (the emit/loop skeleton is 40 trivial lines; all domain content — keys, sources, derivations — is spelled out above).
- **Type consistency**: `CostModel::parse/load/has/at/get/machine`, `Pattern`, `classify`, `cpuBw`, `PathCosts{tAMs,tBMs,aInterceptMs,aSlopeMsPerByte,bInterceptMs,bSlopeMsPerByte}`, `MethodDecision{method,tAMs,tBMs,thresholdBytes,pattern,k,nReuse}`, `decide(model, pattern, srcBytes, r, threads, K, nReuse, broadcast)`, python `predict(calibration, *, pattern, src_bytes, r, threads, k, n_reuse, broadcast)` — used identically across T1–T8. Calibration keys in T2's code match T6's emitted set and T5/T7's test files.
