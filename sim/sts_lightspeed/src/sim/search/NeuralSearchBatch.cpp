#include "sim/search/NeuralSearchBatch.h"
#include <algorithm>
#include <stdexcept>
#include <unordered_set>

namespace sts::search {

NeuralSearchBatch::NeuralSearchBatch(int threads) {
    if (threads < 1 || threads > 256)
        throw std::invalid_argument("native search threads must be in [1, 256]");
    try {
        for (int i = 0; i < threads; ++i)
            threads_.emplace_back([this] { worker(); });
    } catch (...) {
        { std::lock_guard<std::mutex> lock(mutex_); stopping_ = true; }
        ready_.notify_all();
        for (auto &thread : threads_) thread.join();
        throw;
    }
}

NeuralSearchBatch::~NeuralSearchBatch() {
    { std::lock_guard<std::mutex> lock(mutex_); stopping_ = true; }
    ready_.notify_all();
    for (auto &thread : threads_) thread.join();
}

void NeuralSearchBatch::worker() {
    std::size_t seen = 0;
    std::unique_lock<std::mutex> lock(mutex_);
    while (true) {
        ready_.wait(lock, [&] { return stopping_ || generation_ != seen; });
        if (stopping_) return;
        seen = generation_;
        lock.unlock();
        try {
            for (auto index = next_.fetch_add(1); index < count_; index = next_.fetch_add(1))
                work_(index);
        } catch (...) {
            std::lock_guard<std::mutex> guard(mutex_);
            if (!failure_) failure_ = std::current_exception();
        }
        lock.lock();
        if (--remaining_ == 0) finished_.notify_one();
    }
}

void NeuralSearchBatch::parallel(std::size_t count, std::function<void(std::size_t)> work) {
    if (!count) return;
    if (threads_.size() == 1) {
        for (std::size_t i = 0; i < count; ++i) work(i);
        return;
    }
    std::unique_lock<std::mutex> lock(mutex_);
    if (remaining_) throw std::logic_error("native search batch is already running");
    work_ = std::move(work);
    count_ = count;
    next_ = 0;
    failure_ = nullptr;
    remaining_ = threads_.size();
    ++generation_;
    ready_.notify_all();
    finished_.wait(lock, [&] { return remaining_ == 0; });
    work_ = {};
    if (failure_) std::rethrow_exception(failure_);
}

std::unique_lock<std::mutex> NeuralSearchBatch::acquire() {
    std::unique_lock<std::mutex> lock(ownerMutex_, std::try_to_lock);
    if (!lock.owns_lock()) throw std::logic_error("native search batch is already running");
    return lock;
}

void NeuralSearchBatch::clear() {
    auto lock = acquire();
    requests_.clear();
    owners_.clear();
    nextRequests_.clear();
    sessions_.clear();
}

void NeuralSearchBatch::begin(std::vector<std::shared_ptr<NoslNeuralSearchSession>> sessions) {
    if (!requests_.empty()) throw std::logic_error("finish or clear the previous search batch");
    std::unordered_set<const NoslNeuralSearchSession *> distinct;
    for (const auto &session : sessions)
        if (!session || !distinct.insert(session.get()).second)
            throw std::invalid_argument("native search batch requires distinct nonnull trees");
    sessions_ = std::move(sessions);
    select();
}

void NeuralSearchBatch::select() {
    nextRequests_.clear();
    nextRequests_.resize(sessions_.size());
    parallel(sessions_.size(), [&](std::size_t i) { nextRequests_[i] = sessions_[i]->nextBatch(1); });
    gather();
}

void NeuralSearchBatch::gather() {
    requests_.clear();
    owners_.clear();
    for (std::size_t i = 0; i < nextRequests_.size(); ++i) {
        auto &next = nextRequests_[i];
        if (next.size() > 1) throw std::logic_error("more than one pending leaf per tree");
        if (!next.empty()) {
            owners_.push_back(i);
            requests_.push_back(std::move(next[0]));
            next.clear();
        } else if (!sessions_[i]->done()) {
            throw std::logic_error("native tree has no leaf but is not complete");
        }
    }
}

void NeuralSearchBatch::advance(const float *values, const float *logits,
                              std::size_t rows, std::size_t width) {
    if (rows != requests_.size()) throw std::invalid_argument("prediction rows must match pending leaves");
    parallel(rows, [&](std::size_t row) {
        const auto &request = requests_[row];
        const auto count = request.needsPolicy ? width : 0;
        std::vector<double> policy(logits + row * width, logits + row * width + count);
        const auto owner = owners_[row];
        sessions_[owner]->backprop(request.requestId, values[row], policy);
        nextRequests_[owner] = sessions_[owner]->nextBatch(1);
    });
    gather();
}

} // namespace sts::search
