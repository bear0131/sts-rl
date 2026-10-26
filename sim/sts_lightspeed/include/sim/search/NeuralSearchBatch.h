#ifndef STS_LIGHTSPEED_NEURAL_SEARCH_BATCH_H
#define STS_LIGHTSPEED_NEURAL_SEARCH_BATCH_H

#include "sim/search/NoslCombatTrainer.h"
#include <atomic>
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>

namespace sts::search {

// Independent trees run concurrently; one tree always has one pending leaf.
// The owning binding releases the GIL while select/backup execute. Threads
// persist across roots and GPU batches instead of spawning per simulation.
class NeuralSearchBatch {
public:
    explicit NeuralSearchBatch(int threads);
    ~NeuralSearchBatch();
    NeuralSearchBatch(const NeuralSearchBatch &) = delete;
    NeuralSearchBatch &operator=(const NeuralSearchBatch &) = delete;

    void begin(std::vector<std::shared_ptr<NoslNeuralSearchSession>> sessions);
    void advance(const float *values, const float *logits, std::size_t rows,
                 std::size_t width);
    void clear();
    // Binding holds this across GIL release and subsequent result encoding.
    // Reject reentry instead of blocking with the GIL held.
    std::unique_lock<std::mutex> acquire();
    const auto &requests() const { return requests_; }
    const auto &owners() const { return owners_; }
    const auto &sessions() const { return sessions_; }
    int threadCount() const { return static_cast<int>(threads_.size()); }
    // Reuse the same resident workers for model-free independent root jobs.
    // Caller owns acquire() across this call and result encoding.
    void runIndependent(std::size_t count, std::function<void(std::size_t)> work);
    // Read-only conversion of numeric rows into disjoint caller-owned
    // rows. The binding retains acquire() and all buffers until this returns.
    void encodeRows(std::size_t count, std::function<void(std::size_t)> encode);

private:
    void parallel(std::size_t count, std::function<void(std::size_t)> work);
    void select();
    void gather();
    void worker();
    std::vector<std::thread> threads_;
    std::mutex mutex_, ownerMutex_;
    std::condition_variable ready_, finished_;
    std::function<void(std::size_t)> work_;
    std::atomic<std::size_t> next_{0};
    std::size_t count_ = 0, generation_ = 0, remaining_ = 0;
    bool stopping_ = false;
    std::exception_ptr failure_;
    std::vector<std::shared_ptr<NoslNeuralSearchSession>> sessions_;
    std::vector<std::vector<NoslNeuralSearchSession::LeafRequest>> nextRequests_;
    std::vector<NoslNeuralSearchSession::LeafRequest> requests_;
    std::vector<std::size_t> owners_;
};

} // namespace sts::search
#endif
