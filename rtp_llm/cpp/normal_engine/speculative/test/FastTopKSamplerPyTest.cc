#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>

#include <optional>
#include <utility>

#include "rtp_llm/cpp/normal_engine/speculative/SpeculativeSampler.h"

namespace py = pybind11;

namespace {
std::pair<torch::Tensor, torch::Tensor>
runSampler(const torch::Tensor& logits, const std::optional<torch::Tensor>& mapping, bool tail) {
    // Match the engine's GIL-free call site. forwardTail acquires the GIL for
    // the real AITER bridge; this wrapper does not reproduce its route guard.
    py::gil_scoped_release                release;
    rtp_llm::speculative::FastTopKSampler sampler(mapping.value_or(torch::Tensor()));
    auto                                  output = tail ? sampler.forwardTail(logits) : sampler.forward(logits);
    return {output.all_probs, output.token_ids};
}
}  // namespace

PYBIND11_MODULE(libtest_fast_topk_sampler, module) {
    module.def("sample", &runSampler, py::arg("logits"), py::arg("mapping") = py::none(), py::arg("tail") = true);
}
