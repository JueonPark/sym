//===- TransferResources.cpp - native transfer resource lifetime ----------===//
#include "reloc/TransferResources.h"

#include "TransferResourcePolicy.h"
#include "reloc/GatherPool.h"
#include "reloc/PinnedBufferPool.h"

#include <limits>
#include <stdexcept>

namespace reloc {

struct TransferContext::Impl : detail::QuarantineNode {
  explicit Impl(std::unique_ptr<CopyBackend> backend)
      : backend(std::move(backend)) {}
  // Reverse destruction: jobs before staging, staging before its allocator.
  std::unique_ptr<CopyBackend> backend;
  std::unique_ptr<PinnedBufferPool> staging;
  std::unique_ptr<GatherPool> workers;
  std::shared_ptr<void> owners;
  TransferDirection direction = TransferDirection::HostToDevice;
  TransferCompletion completion = TransferCompletion::NotLaunched;
};

TransferContext::TransferContext(std::unique_ptr<CopyBackend> backend) {
  if (!backend)
    throw std::invalid_argument("transfer context requires a backend");
  impl_ = std::make_unique<Impl>(std::move(backend));
  stats_.reusable = !impl_->backend->failed();
}

TransferContext::~TransferContext() { close(); }

TransferCompletion TransferContext::close() noexcept {
  stats_.reusable = false;
  if (impl_) {
    if (impl_->completion == TransferCompletion::Unknown) {
      stats_.quarantined = true;
      detail::quarantine(std::move(impl_));
    } else {
      impl_.reset();
      stats_.stagingBytes = stats_.slotBytes = 0;
      stats_.activeSlots = 0;
      stats_.backgroundWorkers = 0;
    }
  }
  return stats_.quarantined ? TransferCompletion::Unknown
                            : TransferCompletion::Complete;
}

TransferContextStats TransferContext::stats() const {
  auto result = stats_;
  if (impl_ && (impl_->completion == TransferCompletion::Unknown ||
                impl_->backend->failed()))
    result.reusable = false;
  return result;
}

TransferCompletion
detail::TransferContextAccess::completion(const TransferContext &context) {
  if (context.impl_)
    return context.impl_->completion;
  return context.stats_.quarantined ? TransferCompletion::Unknown
                                    : TransferCompletion::NotLaunched;
}

void detail::TransferContextAccess::prepare(TransferContext &context,
                                            TransferDirection direction,
                                            const TransferRequirements &r,
                                            size_t capacity) {
  if (!context.impl_ || !context.stats().reusable)
    throw std::runtime_error("transfer context is unavailable");
  if (r.activeSlots < 1 || capacity < r.slotBytes ||
      capacity > std::numeric_limits<size_t>::max() / size_t(r.activeSlots))
    throw std::invalid_argument("invalid prepared staging capacity");
  auto &state = *context.impl_;
  auto &stats = context.stats_;
  state.completion = TransferCompletion::NotLaunched;
  if (!state.staging || state.direction != direction ||
      state.staging->nBuffers() != r.activeSlots ||
      state.staging->bufferBytes() < capacity) {
    // Release the old idle allocation first. The cache holds the full target
    // charge across this gap; capacity rounding never changes the schedule.
    state.staging.reset();
    stats.stagingBytes = stats.slotBytes = 0;
    stats.activeSlots = 0;
    state.staging = std::make_unique<PinnedBufferPool>(*state.backend,
                                                       r.activeSlots, capacity);
    if (!state.staging->valid() || state.backend->failed())
      throw std::runtime_error("pinned staging allocation failed: " +
                               state.backend->error());
    ++stats.stagingPoolCreations;
    stats.slotBytes = capacity;
    stats.activeSlots = r.activeSlots;
    stats.stagingBytes = capacity * size_t(r.activeSlots);
    state.direction = direction;
  }
  if (state.workers &&
      unsigned(state.workers->threadCount()) != r.gatherThreads) {
    state.workers.reset();
    stats.backgroundWorkers = 0;
  }
  if (!state.workers && r.gatherThreads > 1) {
    state.workers = std::make_unique<GatherPool>(r.gatherThreads);
    ++stats.workerPoolCreations;
    stats.backgroundWorkers = r.gatherThreads - 1;
  }
}

TransferOutcome detail::TransferContextAccess::execute(
    TransferRequest &request, TransferContext &context,
    const TransferOptions &options, const TransferRequirements &requirements,
    std::shared_ptr<void> bufferOwners) {
  auto &state = *context.impl_;
  state.completion = TransferCompletion::NotLaunched;
  state.owners = std::move(bufferOwners);
  std::optional<TransferError> error;
  try {
    error = executePreparedTransfer(
        request, requirements, options, *state.backend, *state.staging,
        options.gather ? options.gather : state.workers.get(),
        state.completion);
  } catch (const std::exception &e) {
    error = TransferError{"backend_failure", e.what()};
  } catch (...) {
    error = TransferError{"backend_failure", "unexpected transfer failure"};
  }
  const auto completion = state.completion;
  if (completion != TransferCompletion::Unknown && state.staging)
    state.staging->markComplete();
  if (completion == TransferCompletion::Unknown) {
    // Publish persistent ownership before constructing an error diagnostic.
    context.close();
    error =
        TransferError{"completion_unknown",
                      error ? error->message : "completion not established"};
  } else if (error) {
    context.close();
  } else {
    state.owners.reset();
  }
  return {std::move(error), completion};
}

TransferOutcome executeTransfer(TransferRequest &request,
                                TransferContext &context,
                                const TransferOptions &options,
                                std::shared_ptr<void> bufferOwners) {
  auto rejected = [](TransferError error) {
    return TransferOutcome{std::move(error), TransferCompletion::NotLaunched};
  };
  if (!bufferOwners)
    return rejected({"invalid_options", "a buffer owner token is required"});
  if (context.impl_ && context.impl_->completion == TransferCompletion::Unknown)
    context.close();
  if (!context.impl_)
    return rejected({"resources_closed", "transfer context is closed"});
  try {
    if (auto error =
            detail::checkTransferBackend(request, *context.impl_->backend))
      return rejected(*error);
    auto described = detail::describeTransfer(request, options);
    if (auto *error = std::get_if<TransferError>(&described))
      return rejected(*error);
    const auto &r = std::get<detail::TransferRequirements>(described);
    request.consumed = true;
    detail::TransferContextAccess::prepare(context, request.direction, r,
                                           r.slotBytes);
    return detail::TransferContextAccess::execute(request, context, options, r,
                                                  std::move(bufferOwners));
  } catch (const std::exception &e) {
    const auto completion = context.impl_ ? context.impl_->completion
                                          : TransferCompletion::NotLaunched;
    context.close();
    return {TransferError{completion == TransferCompletion::Unknown
                              ? "completion_unknown"
                              : "backend_failure",
                          e.what()},
            completion};
  } catch (...) {
    const auto completion = context.impl_ ? context.impl_->completion
                                          : TransferCompletion::NotLaunched;
    context.close();
    return {TransferError{completion == TransferCompletion::Unknown
                              ? "completion_unknown"
                              : "backend_failure",
                          "unexpected transfer failure"},
            completion};
  }
}

} // namespace reloc
