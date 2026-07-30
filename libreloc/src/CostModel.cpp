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

} // namespace costmodel
} // namespace reloc
