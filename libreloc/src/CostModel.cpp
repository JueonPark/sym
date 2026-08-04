//===- CostModel.cpp - P3b calibrated transfer cost model (#97) -----------===//

#include "reloc/CostModel.h"

#include "reloc/Prefold.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
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

std::variant<CostModel, std::string> CostModel::parse(const std::string &text) {
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
      return std::string("calibration: duplicate key '" + key + "' at line " +
                         std::to_string(lineNo));
  }
  if (!sawVersion)
    return std::string("calibration: empty file (no version header)");
  return m;
}

std::variant<CostModel, std::string> CostModel::load(const std::string &path) {
  std::ifstream f(path);
  if (!f)
    return std::string("calibration: cannot read " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return parse(ss.str());
}

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

const char *placementName(BPlacement p) {
  switch (p) {
  case BPlacement::Serial:
    return "serial";
  case BPlacement::Overlapped:
    return "overlapped";
  }
  return "?";
}

Pattern classify(const BoundPlan &b) {
  const int64_t totalElems = b.elementSize ? b.totalBytes / b.elementSize : 0;
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

static double harmonic(double a, double b) { return 1.0 / (1.0 / a + 1.0 / b); }

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
    return m.has("pcie.h2d_gbps") ? std::optional<double>(m.at("pcie.h2d_gbps"))
                                  : std::nullopt;
  const std::string key = "multigpu.delivery_gbps.k" + std::to_string(K);
  if (!m.has(key))
    return std::nullopt;
  return m.at(key);
}

std::optional<PathCosts> pathCosts(const CostModel &m, Pattern p,
                                   int64_t srcBytes, double r, int threads,
                                   int K, bool broadcast, BPlacement bPlace) {
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
  const double aCpuSlope = msPerByteAt(*bwCpu);             // per source byte
  const double aDmaSlope = kMult * r * msPerByteAt(*bwDel); // per source byte
  pc.aSlopeMsPerByte = std::max(aCpuSlope, aDmaSlope);
  pc.aInterceptMs = m.get("overhead.a_ms", 0.0);
  // B: DMA of kMult*S vs GPU transform m*kMult*S over HBM (issue #109).
  const double bDmaSlope = kMult * msPerByteAt(*bwDel);
  const double bHbmSlope = kMult * mm * msPerByteAt(bwHbm);
  if (bPlace == BPlacement::Serial) {
    // b_fair: the kernel starts only after the whole transfer landed --
    // the stages ADD, so B's slope strictly exceeds the bare DMA slope
    // for any m > 0 and the r=1 A/B slope tie (v3-costmodel.md S4) is
    // dead by construction.
    pc.bSlopeMsPerByte = bDmaSlope + bHbmSlope;
  } else {
    // b_pipelined: chunks overlap; the hidden stage still pays one
    // chunk of fill/drain. n must be the TOTAL chunk count across the
    // schedule (ChunkSchedule's mid-range: kChunksPerBuffer x nBuffers
    // = 16 at the pipeline's 2 buffers), so chunk = S/n folds into the
    // slope (min/n) and stays affine in S.
    // n absent or <= 0 -> term 0 -> exactly the V3 formula.
    const double nChunks = m.get("pipeline.chunks_per_buffer", 0.0);
    const double fillDrain =
        nChunks > 0 ? std::min(bDmaSlope, bHbmSlope) / nChunks : 0.0;
    pc.bSlopeMsPerByte = std::max(bDmaSlope, bHbmSlope) + fillDrain;
  }
  pc.bInterceptMs = m.get("overhead.b_ms", 0.0);
  pc.tAMs = pc.aInterceptMs + pc.aSlopeMsPerByte * srcBytes;
  pc.tBMs = pc.bInterceptMs + pc.bSlopeMsPerByte * srcBytes;
  return pc;
}

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
                                     int K, int64_t nReuse, bool broadcast,
                                     BPlacement bPlace) {
  auto pc = pathCosts(m, p, srcBytes, r, threads, K, broadcast, bPlace);
  if (!pc)
    return std::nullopt;

  MethodDecision d;
  d.pattern = p;
  d.k = K;
  d.nReuse = nReuse;
  d.bPlacement = bPlace;

  // Plain A is always a candidate arm.
  const double aIntPlain = pc->aInterceptMs;
  const double aSlopePlain = pc->aSlopeMsPerByte;
  const double tAPlain =
      aIntPlain + aSlopePlain * static_cast<double>(srcBytes);

  // Prefold is an ADDITIONAL arm, never a silent replacement for plain
  // A: when A is DMA-bound (aSlopePlain picked the dma side of its
  // max(cpu, dma)), prefold's additive per-load slope (dma + cpu /
  // nReuse) can be *worse* than plain A's pipelined max, even though
  // prefoldWins (the fold-pays-for-itself gate) says the fold is worth
  // taking on its own terms. So prefoldWins alone cannot select the
  // arm: it must also beat plain A at the bound S being decided here.
  double aSlope = aSlopePlain;
  double tA = tAPlain;
  bool prefold = false;
  if (nReuse >= 1) {
    // pathCosts already required cpuBw(m, p, r, threads) and the K
    // delivery-BW key (via the same deliveryGbps helper it uses) to
    // succeed above, so both are guaranteed present; still checked
    // defensively rather than dereferencing blindly.
    auto bwCpuOpt = cpuBw(m, p, r, threads);
    auto bwDelOpt = deliveryGbps(m, K);
    if (bwCpuOpt && bwDelOpt) {
      const double cpuSlope = 1e-6 / *bwCpuOpt; // ms per source byte
      const double kMult = broadcast ? static_cast<double>(K) : 1.0;
      const double dmaOnlySlope = kMult * r * 1e-6 / *bwDelOpt;
      const double tTransform = cpuSlope * static_cast<double>(srcBytes);
      // allocMs is strictly linear in S with no constant term -- that
      // is what keeps the single-boundary threshold model below valid:
      // prefoldWins' activation (gated on nReuse/tTransform/allocMs)
      // can then never flip as S sweeps, only the arm-vs-B crossing
      // point can move. A constant term in the alloc cost would need a
      // second boundary to represent correctly.
      const double allocMs = m.get("prefold.alloc_ms_per_gib", 0.0) *
                             (r * static_cast<double>(srcBytes) / (1ll << 30));
      if (reloc::prefold::prefoldWins(nReuse, tTransform, tTransform + allocMs,
                                      0.0)) {
        // Per-load slope: DMA is paid every load; the CPU pass and the
        // fold's staging allocation are amortized over nReuse loads.
        // The intercept is unchanged from plain A -- prefolding only
        // changes the per-byte slope, not the fixed per-call setup
        // overhead -- so there is nothing to re-read from `m` here.
        const double aSlopeFold =
            dmaOnlySlope +
            (cpuSlope + allocMs / static_cast<double>(srcBytes)) /
                static_cast<double>(nReuse);
        const double tAFold =
            aIntPlain + aSlopeFold * static_cast<double>(srcBytes);
        if (tAFold < tAPlain) {
          prefold = true;
          aSlope = aSlopeFold;
          tA = tAFold;
        }
      }
    }
  }

  const double tB = pc->tBMs;
  d.tAMs = tA;
  d.tBMs = tB;
  d.method = tA <= tB ? (prefold ? MethodDecision::Method::APrefold
                                 : MethodDecision::Method::A)
                      : MethodDecision::Method::B;

  // Single-symbol threshold precompute: the active A-side arm (plain or
  // prefold, whichever was adopted at this bound S above) and B are
  // both affine in S, so their crossing point is the boundary bind()
  // can compare S against directly later. No real crossing (parallel
  // slopes, or the crossing falls at S <= 0) means the decision never
  // flips with S.
  const double dSlope = aSlope - pc->bSlopeMsPerByte;
  if (dSlope != 0.0) {
    const double sStar = (pc->bInterceptMs - aIntPlain) / dSlope;
    d.thresholdBytes = sStar > 0 ? sStar : -1;
  }
  return d;
}

} // namespace costmodel
} // namespace reloc
