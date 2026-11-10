#pragma once

#include <deque>
#include <exception>
#include <memory>
#include <torch/torch.h>

namespace rtp_llm {

// GPU completion, rather than a fixed number of CPU rounds, determines when
// staging storage may be released. The bound limits memory if submission runs
// ahead; only the oldest completion is waited at that boundary.
class DecodeBufferRetirement {
public:
    static constexpr size_t kMaxPendingRounds = 4;

    ~DecodeBufferRetirement() {
        try {
            drain();
        } catch (...) {
            std::terminate();
        }
    }

    void reap() {
        while (!pending_.empty() && pending_.front().completion->query()) {
            pending_.pop_front();
        }
    }

    void retire(const std::shared_ptr<torch::Event>& completion, const std::shared_ptr<void>& owner) {
        // Caller keeps its owner alive through this call, including allocation
        // failures. Never unwind with unfinished GPU work losing its storage.
        try {
            reap();
            if (pending_.size() == kMaxPendingRounds) {
                pending_.front().completion->synchronize();
                pending_.pop_front();
            }
            pending_.push_back({completion, owner});
        } catch (...) {
            try {
                completion->synchronize();
            } catch (...) {
                std::terminate();
            }
            throw;
        }
    }

    void drain() {
        while (!pending_.empty()) {
            pending_.front().completion->synchronize();
            pending_.pop_front();
        }
    }

    size_t size() const {
        return pending_.size();
    }

private:
    struct Pending {
        std::shared_ptr<torch::Event> completion;
        std::shared_ptr<void>         owner;
    };
    std::deque<Pending> pending_;
};

}  // namespace rtp_llm
