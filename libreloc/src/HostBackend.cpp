//===- HostBackend.cpp - CI-testable CopyBackend --------------------------===//

#include "reloc/HostBackend.h"

#include <cstdlib>
#include <cstring>

namespace reloc {

HostBackend::HostBackend(int numQueues) {
  if (numQueues < 1)
    numQueues = 1;
  for (int i = 0; i < numQueues; ++i)
    queues_.push_back(std::make_unique<Queue>());
  // Start workers only after every Queue exists so their addresses are stable
  // (unique_ptr keeps the Queue objects fixed even as the vector grows).
  for (auto &q : queues_) {
    Queue *qp = q.get();
    qp->worker = std::thread([this, qp] { workerLoop(*qp); });
  }
}

HostBackend::~HostBackend() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    for (auto &q : queues_)
      q->stop = true;
  }
  cv_.notify_all();
  for (auto &q : queues_)
    if (q->worker.joinable())
      q->worker.join();
}

void *HostBackend::allocStaging(size_t bytes) { return std::malloc(bytes); }
void HostBackend::freeStaging(void *p) { std::free(p); }

void HostBackend::copyAsync(int queue, void *dst, const void *src, size_t bytes,
                            CopyDir /*dir*/) {
  // Direction is irrelevant for a pure host memcpy; it exists so CudaBackend
  // can pick the cudaMemcpyKind.
  Task t;
  t.kind = Task::Copy;
  t.dst = dst;
  t.src = src;
  t.bytes = bytes;
  {
    std::lock_guard<std::mutex> lk(mu_);
    queues_[queue]->tasks.push_back(t);
  }
  cv_.notify_all();
}

EventHandle HostBackend::recordEvent(int queue) {
  Task t;
  t.kind = Task::Event;
  EventHandle ev;
  {
    std::lock_guard<std::mutex> lk(mu_);
    ev = nextEvent_++;
    done_[ev] = false;
    t.ev = ev;
    queues_[queue]->tasks.push_back(t);
  }
  cv_.notify_all();
  return ev;
}

void HostBackend::waitEvent(EventHandle ev) {
  if (ev == 0)
    return;
  std::unique_lock<std::mutex> lk(mu_);
  cv_.wait(lk, [&] {
    auto it = done_.find(ev);
    return it != done_.end() && it->second;
  });
}

bool HostBackend::queryEvent(EventHandle ev) {
  if (ev == 0)
    return true;
  std::lock_guard<std::mutex> lk(mu_);
  auto it = done_.find(ev);
  return it != done_.end() && it->second;
}

void HostBackend::setCopyHook(std::function<void()> hook) {
  std::lock_guard<std::mutex> lk(mu_);
  hook_ = std::move(hook);
}

void HostBackend::workerLoop(Queue &q) {
  for (;;) {
    Task t;
    {
      std::unique_lock<std::mutex> lk(mu_);
      cv_.wait(lk, [&] { return q.stop || !q.tasks.empty(); });
      if (q.tasks.empty()) // implies q.stop
        return;
      t = q.tasks.front();
      q.tasks.pop_front();
    }
    if (t.kind == Task::Copy) {
      std::function<void()> hook;
      {
        std::lock_guard<std::mutex> lk(mu_);
        hook = hook_;
      }
      if (hook)
        hook();
      std::memcpy(t.dst, t.src, t.bytes);
    } else { // Task::Event
      std::lock_guard<std::mutex> lk(mu_);
      done_[t.ev] = true;
      cv_.notify_all();
    }
  }
}

} // namespace reloc
