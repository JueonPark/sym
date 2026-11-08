// Experimental runner, not a production runtime change.
// Both Sym arms execute all their host plans inside one Python/native call.
#include "reloc/GatherPool.h"
#include "reloc/TypedExecute.h"
#include "reloc/Execute.h"
#include "TransferInternal.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <immintrin.h>
#include <algorithm>
#include <memory>
#include <stdexcept>

namespace py = pybind11;
using Program = reloc::typed::Program;

struct Runner {
  reloc::GatherPool pool;
  struct Step {
    std::optional<Program> typed;
    reloc::BoundPlan layout;
    int64_t sourceBytes, destinationBytes;
  };
  std::vector<Step> programs;
  Runner(const std::vector<py::object>& plans, unsigned threads): pool(threads) {
    if (threads == 0) throw std::invalid_argument("threads must be positive");
    for (const auto& b: plans) {
      if (py::isinstance<reloc::TypedBoundPlan>(b)) {
        auto p = reloc::typed::prepareProgram(b.cast<reloc::TypedBoundPlan>());
        if (auto* e = std::get_if<reloc::typed::ExecutionError>(&p))
          throw std::runtime_error(e->message);
        auto value=std::get<Program>(std::move(p));
        auto sb=reloc::typed::bytesAt(value,0,false), db=reloc::typed::bytesAt(value,value.stages.size(),true);
        programs.push_back({std::move(value),{},sb,db});
      } else {
        auto layout=b.cast<reloc::BoundPlan>();
        int64_t last=0; // all admitted recipes have zero source storage offset
        for(size_t i=0;i<layout.extents.size();++i) last+=(layout.extents[i]-1)*layout.srcStrides[i];
        programs.push_back({std::nullopt,layout,(last+1)*layout.elementSize,layout.totalBytes});
      }
    }
  }
  void run(std::vector<py::array> buffers) {
    if (buffers.size() != programs.size()+1) throw std::invalid_argument("buffer count");
    for (size_t i=0; i<programs.size(); ++i) {
      const auto& p=programs[i];
      if (!(buffers[i].flags() & py::array::c_style) ||
          !(buffers[i+1].flags() & py::array::c_style) || !buffers[i+1].writeable() ||
          buffers[i].nbytes()!=p.sourceBytes || buffers[i+1].nbytes()!=p.destinationBytes)
        throw std::invalid_argument("buffer capacity or layout");
    }
    py::gil_scoped_release release;
    for (size_t i=0; i<programs.size(); ++i) {
      const auto& p=programs[i];
      if(p.typed) {
        if (auto e=reloc::typed::executeHost(*p.typed,0,p.typed->stages.size(),buffers[i].data(),
                                           buffers[i+1].mutable_data(),&pool))
          throw std::runtime_error(e->message);
      } else {
        reloc::TransferOptions options;
        options.gather=&pool;
        options.gatherThreads=pool.threadCount();
        // Use the actual production implementation, including byte-based
        // scheduling grain and conservative disjoint-output-row qualification.
        reloc::detail::forwardHostGather(p.layout,buffers[i].data(),
                                         buffers[i+1].mutable_data(),options);
      }
    }
  }
  std::vector<std::string> kernels() const {
    std::vector<std::string> out;
    for (const auto& p:programs)
      out.emplace_back(p.typed?reloc::typed::hostKernelName(reloc::typed::hostKernel(*p.typed,0,p.typed->stages.size())):"layout_gather");
    return out;
  }
  // Compiler-free, fixed-family address arithmetic. F16C specifies RNE,
  // preserves the intermediate f16 rounding, and emits no approximate math.
  void manual(const std::string& family, py::array src, py::array dst,
              py::array_t<float,py::array::c_style> scales, int64_t n, int64_t m) {
    const bool dq=family=="dequant_transpose";
    const bool rank3=family=="rank3_chain";
    const bool pad=family=="padded_roundtrip";
    const bool rt=family.rfind("roundtrip",0)==0;
    if ((!dq && !rank3 && !pad && !rt) || n<1 || m<1 ||
        !(src.flags()&py::array::c_style) || !(dst.flags()&py::array::c_style) || !dst.writeable() ||
        src.nbytes()!=n*m*(dq?1:4)*(rank3?8:1) ||
        dst.nbytes()!=(pad?(m+8)*(n+9)*4:n*m*(dq?2:4)*(rank3?8:1)) ||
        (dq && scales.size()!=m)) throw std::invalid_argument("manual shape/capacity");
    const auto* s=static_cast<const float*>(src.data());
    const auto* q=static_cast<const int8_t*>(src.data());
    auto* d=static_cast<float*>(dst.mutable_data());
    auto* h=static_cast<uint16_t*>(dst.mutable_data());
    const float* scale=scales.data();
    py::gil_scoped_release release;
    if (rt) {
      const int64_t blocks=(n*m+7)/8;
      pool.parallelFor(0,blocks,2048,[&](int64_t a,int64_t b){
        for(int64_t k=a;k<b;++k) {
          int64_t i=k*8;
          if(i+8<=n*m) _mm256_storeu_ps(d+i,_mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_loadu_ps(s+i),0)));
          else for(;i<n*m;++i) d[i]=_cvtsh_ss(_cvtss_sh(s[i],0));
        }
      });
    } else if(rank3) {
      pool.parallelFor(0,n,1,[&](int64_t a,int64_t b){
        for(int64_t r=a;r<b;++r) for(int64_t c=0;c<m;++c) for(int64_t t=0;t<8;++t)
          d[(r*m+c)*8+t]=_cvtsh_ss(_cvtss_sh(s[(t*n+r)*m+c],0));
      });
    } else {
      // Partition output column tiles; tiles write disjoint destination rows.
      const int64_t rows=pad?m+8:m;
      pool.parallelFor(0,(rows+31)/32,1,[&](int64_t a,int64_t b){
        for(int64_t tile=a;tile<b;++tile)
          for(int64_t rb=0;rb<(pad?n+9:n);rb+=32)
            for(int64_t c=tile*32;c<std::min(rows,(tile+1)*32);++c)
              for(int64_t r=rb;r<std::min<int64_t>(pad?n+9:n,rb+32);++r) {
                if(dq) { float v=static_cast<float>(q[r*m+c])*scale[c]; h[c*n+r]=_cvtss_sh(v,0); }
                else d[c*(n+9)+r]=(c<3 || c>=m+3 || r<2 || r>=n+2)?-0.0f:
                  _cvtsh_ss(_cvtss_sh(s[(r-2)*m+c-3],0));
              }
      });
    }
  }
};

PYBIND11_MODULE(_folding_native,m) {
  py::module_::import("pyreloc");
  py::class_<Runner>(m,"Runner")
    .def(py::init<const std::vector<py::object>&,unsigned>())
    .def("run",&Runner::run).def("kernels",&Runner::kernels).def("manual",&Runner::manual);
}
