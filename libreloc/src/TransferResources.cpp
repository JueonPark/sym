//===- TransferResources.cpp - native transfer resource lifetime ----------===//
#include "reloc/TransferResources.h"

#include "TransferInternal.h"
#include "reloc/GatherPool.h"
#include "reloc/PinnedBufferPool.h"

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
  auto &state = *context.impl_;
  auto &stats = context.stats_;
  state.completion = TransferCompletion::NotLaunched;
  std::optional<TransferError> error;
  try {
    if (auto error = detail::checkTransferBackend(request, *state.backend))
      return rejected(*error);
    auto described = detail::describeTransfer(request, options);
    if (auto *error = std::get_if<TransferError>(&described))
      return rejected(*error);
    const auto &r = std::get<detail::TransferRequirements>(described);
    request.consumed = true;
    state.completion = TransferCompletion::NotLaunched;
    state.owners = std::move(bufferOwners);

    if (!state.staging || state.direction != request.direction ||
        state.staging->nBuffers() != r.activeSlots ||
        state.staging->bufferBytes() < r.slotBytes) {
      // Previous execution completed. Release old storage before growth so
      // the upcoming cache can reserve a delta without a double-allocation
      // peak.
      state.staging.reset();
      stats.stagingBytes = stats.slotBytes = 0;
      stats.activeSlots = 0;
      state.staging = std::make_unique<PinnedBufferPool>(
          *state.backend, r.activeSlots, r.slotBytes);
      if (!state.staging->valid() || state.backend->failed())
        throw std::runtime_error("pinned staging allocation failed: " +
                                 state.backend->error());
      ++stats.stagingPoolCreations;
      stats.slotBytes = r.slotBytes;
      stats.activeSlots = r.activeSlots;
      stats.stagingBytes = r.stagingBytes;
      state.direction = request.direction;
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
    error = detail::executePreparedTransfer(
        request, r, options, *state.backend, *state.staging,
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

} // namespace reloc
