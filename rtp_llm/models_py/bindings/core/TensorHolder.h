#pragma once

#include <deque>
#include <vector>

#include <torch/extension.h>

namespace rtp_llm {

struct TensorHolder {
    static constexpr size_t kReleasedHoldRounds = 2;

    std::vector<torch::Tensor>              tensors;
    std::deque<std::vector<torch::Tensor>>  clear_tensors;

    void hold_host(const torch::Tensor& tensor) {
        if (tensor.defined() && tensor.device().is_cpu()) {
            tensors.push_back(tensor);
        }
    }

    void hold(const torch::Tensor& tensor) {
        if (tensor.defined()) {
            tensors.push_back(tensor);
        }
    }

    void release() {
        // Move the current hold set into clear_tensors. Keep two released
        // rounds alive so tensors created for async H2D/D2H copies or CUDA
        // kernels are not freed until the third release point.
        clear_tensors.push_back(std::move(tensors));
        tensors.clear();
        while (clear_tensors.size() > kReleasedHoldRounds) {
            clear_tensors.pop_front();
        }
    }

    // Transfer every retained source to an event-owned retirement batch.
    // Allocate before moving anything: an allocation failure leaves this
    // holder intact, including the historical async-copy sources.
    std::vector<torch::Tensor> takeAll() {
        size_t count = tensors.size();
        for (const auto& round : clear_tensors) {
            count += round.size();
        }
        std::vector<torch::Tensor> result;
        result.reserve(count);
        for (auto& tensor : tensors) {
            result.push_back(std::move(tensor));
        }
        tensors.clear();
        while (!clear_tensors.empty()) {
            for (auto& tensor : clear_tensors.front()) {
                result.push_back(std::move(tensor));
            }
            clear_tensors.pop_front();
        }
        return result;
    }
};

}  // namespace rtp_llm
