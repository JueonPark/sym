// Test barrier: keep work pending until explicitly released, without sleeps.
#ifndef RELOC_TEST_BLOCKINGGATE_H
#define RELOC_TEST_BLOCKINGGATE_H

#include <chrono>
#include <condition_variable>
#include <mutex>

class BlockingGate {
public:
  void arriveAndWait() {
    std::unique_lock<std::mutex> lock(mu_);
    ++arrivals_;
    cv_.notify_all();
    cv_.wait(lock, [&] { return released_; });
  }

  bool waitForArrivals(int count) {
    std::unique_lock<std::mutex> lock(mu_);
    return cv_.wait_for(lock, std::chrono::seconds(5),
                        [&] { return arrivals_ >= count; });
  }

  void release() {
    std::lock_guard<std::mutex> lock(mu_);
    released_ = true;
    cv_.notify_all();
  }

private:
  std::mutex mu_;
  std::condition_variable cv_;
  int arrivals_ = 0;
  bool released_ = false;
};

#endif
