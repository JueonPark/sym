// Build-only faults/gates for Python ownership and GIL tests. Not installed.
#include "../../test/TransferTestSupport.h"
#include "../PyTransfer.h"
#include <pybind11/stl.h>

namespace py = pybind11;
namespace {
struct Control : std::enable_shared_from_this<Control> {
  explicit Control(std::string mode, bool gated) : mode(std::move(mode)) {
    if (this->mode != "healthy" && this->mode != "allocation" &&
        this->mode != "complete_failure" && this->mode != "unknown" &&
        this->mode != "copy" && this->mode != "throw_copy" &&
        this->mode != "wait")
      throw std::invalid_argument("unknown test failure mode");
    if (gated)
      gate = std::make_shared<BlockingGate>();
  }
  std::string mode;
  std::shared_ptr<BlockingGate> gate;
  std::shared_ptr<transfer_test::Metrics> metrics =
      std::make_shared<transfer_test::Metrics>();
  std::atomic<transfer_test::Backend *> backend{nullptr};

  std::shared_ptr<reloc::TransferResourceCache>
  cache(std::optional<int64_t> timeout) {
    reloc::TransferResourceLimits limits;
    limits.maxContexts = 1;
    if (timeout)
      limits.acquireTimeout = std::chrono::milliseconds(*timeout);
    return reloc_python::makeResourceCache(
        limits, [self = shared_from_this()](const auto &config) {
          auto b = std::make_unique<transfer_test::Backend>(self->metrics,
                                                            config.streams);
          b->failAllocation = self->mode == "allocation" ? 1 : 0;
          b->failEvent =
              self->mode == "complete_failure" || self->mode == "unknown";
          b->unknown = self->mode == "unknown";
          b->failCopy = self->mode == "copy";
          b->throwCopy = self->mode == "throw_copy";
          b->failWait = self->mode == "wait";
          if (self->gate) {
            b->gate = self->gate;
            b->setCopyHook([gate = self->gate] { gate->arriveAndWait(); });
          }
          self->backend = b.get();
          return b;
        });
  }
};
} // namespace

PYBIND11_MODULE(_reloc_transfer_test, m) {
  py::module_::import("pyreloc");
  py::class_<Control, std::shared_ptr<Control>>(m, "Control")
      .def(py::init<std::string, bool>(), py::arg("mode") = "healthy",
           py::arg("gated") = true)
      .def("cache", &Control::cache, py::arg("timeout_ms") = py::none())
      .def(
          "wait_for_copy",
          [](Control &c) { return c.gate && c.gate->waitForArrivals(1); },
          py::call_guard<py::gil_scoped_release>())
      .def("release",
           [](Control &c) {
             if (c.gate)
               c.gate->release();
           })
      .def(
          "drain_unknown",
          [](Control &c) {
            if (c.mode != "unknown" || !c.backend.load())
              throw std::runtime_error(
                  "drain requires an existing quarantined backend");
            c.backend.load()->HostBackend::quiesce();
          },
          py::call_guard<py::gil_scoped_release>())
      .def("stats", [](const Control &c) {
        py::dict d;
        d["allocations"] = c.metrics->allocations.load();
        d["frees"] = c.metrics->frees.load();
        d["destroyed"] = c.metrics->destroyed.load();
        d["copies"] = c.metrics->copies.load();
        return d;
      });
}
