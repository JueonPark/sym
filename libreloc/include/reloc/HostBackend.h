//===- HostBackend.h - CI-testable CopyBackend ------------------*- C++ -*-===//
//
// Staging is malloc'd host memory; each queue is a FIFO drained in order by
// one worker thread; events are monotonic-counter flags. A test copy-hook can
// artificially slow copies so the event-gating tests can observe a pending
// event. Everything the CUDA pipeline does is exercisable through this.
//
//===----------------------------------------------------------------------===//

#ifndef RELOC_HOSTBACKEND_H
#define RELOC_HOSTBACKEND_H

#include "reloc/Backend.h"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace reloc {

class HostBackend : public CopyBackend {
public:
  explicit HostBackend(int numQueues = 1);
  ~HostBackend() override;

  HostBackend(const HostBackend &) = delete;
  HostBackend &operator=(const HostBackend &) = delete;

  void *allocStaging(size_t bytes) override;
  void freeStaging(void *p) override;
  int numQueues() const override { return static_cast<int>(queues_.size()); }
  void copyAsync(int queue, void *dst, const void *src, size_t bytes,
                 CopyDir dir) override;
  EventHandle recordEvent(int queue) override;
  void waitEvent(EventHandle ev) override;
  bool queryEvent(EventHandle ev) override;

  /// Test hook: invoked on the worker thread at the START of every copy. Used
  /// to artificially slow the backend for the event-gating tests.
  void setCopyHook(std::function<void()> hook);

private:
  struct Task {
    enum Kind { Copy, Event } kind;
    void *dst = nullptr;
    const void *src = nullptr;
    size_t bytes = 0;
    EventHandle ev = 0;
  };
  struct Queue {
    std::deque<Task> tasks;
    std::thread worker;
    bool stop = false;
  };

  void workerLoop(Queue &q);

  std::vector<std::unique_ptr<Queue>> queues_;
  std::mutex mu_;              // guards every task deque, done_, hook_, stop
  std::condition_variable cv_; // signals workers and waitEvent waiters
  std::unordered_map<EventHandle, bool> done_; // event -> completed
  std::function<void()> hook_;
  uint64_t nextEvent_ = 1;
};

} // namespace reloc

#endif // RELOC_HOSTBACKEND_H
