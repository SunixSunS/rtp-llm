#include "rtp_llm/models_py/bindings/cuda/kernels/attention_input_metadata.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <c10/util/Exception.h>
#include <gtest/gtest.h>

#if USING_ROCM
#define METADATA_GPU(name) hip##name
#else
#include <cuda_runtime.h>
#define METADATA_GPU(name) cuda##name
#endif

namespace rtp_llm {
namespace {

void checkGpu(METADATA_GPU(Error_t) error) {
    if (error != METADATA_GPU(Success)) {
        throw std::runtime_error(METADATA_GPU(GetErrorString)(error));
    }
}

struct Stream {
    AttentionInputMetadataStream value{};

    Stream() {
        checkGpu(METADATA_GPU(StreamCreateWithFlags)(&value, METADATA_GPU(StreamNonBlocking)));
    }
    ~Stream() {
        METADATA_GPU(StreamSynchronize)(value);
        METADATA_GPU(StreamDestroy)(value);
    }
    Stream(const Stream&)            = delete;
    Stream& operator=(const Stream&) = delete;
};

at::TensorOptions gpuInt32Options() {
    return at::TensorOptions().dtype(at::kInt).device(at::kCUDA);
}

void upload(const std::vector<int32_t>& values, at::Tensor& tensor, AttentionInputMetadataStream stream) {
    if (!values.empty()) {
        checkGpu(METADATA_GPU(MemcpyAsync)(tensor.data_ptr<int32_t>(),
                                           values.data(),
                                           values.size() * sizeof(int32_t),
                                           METADATA_GPU(MemcpyHostToDevice),
                                           stream));
    }
}

std::vector<int32_t> download(const at::Tensor& tensor, AttentionInputMetadataStream stream) {
    std::vector<int32_t> result(tensor.numel());
    if (!result.empty()) {
        checkGpu(METADATA_GPU(MemcpyAsync)(result.data(),
                                           tensor.data_ptr<int32_t>(),
                                           result.size() * sizeof(int32_t),
                                           METADATA_GPU(MemcpyDeviceToHost),
                                           stream));
    }
    checkGpu(METADATA_GPU(StreamSynchronize)(stream));
    return result;
}

// Independent packed-to-padded index oracle. It does not emulate the kernel's
// running padding offset: each packed token maps to its rectangular row slot.
std::vector<int32_t> paddingReference(const std::vector<int32_t>& queries) {
    if (queries.empty()) {
        return {};
    }
    const auto           width = *std::max_element(queries.begin(), queries.end());
    std::vector<int32_t> result;
    for (size_t row = 0; row < queries.size(); ++row) {
        for (int32_t column = 0; column < queries[row]; ++column) {
            const int64_t padded_index = static_cast<int64_t>(row) * width + column;
            result.push_back(static_cast<int32_t>(padded_index - static_cast<int64_t>(result.size())));
        }
    }
    return result;
}

void checkMetadata(const std::vector<int32_t>& queries, const std::vector<int32_t>& prefixes) {
    SCOPED_TRACE(::testing::Message() << "queries=" << ::testing::PrintToString(queries)
                                      << ", prefixes=" << ::testing::PrintToString(prefixes));
    ASSERT_TRUE(prefixes.empty() || prefixes.size() == queries.size());
    constexpr int32_t guard  = -1234567;
    const int64_t     batch  = queries.size();
    const int64_t     tokens = std::accumulate(queries.begin(), queries.end(), int64_t{0});

    auto                 input_lengths  = at::empty({batch}, gpuInt32Options());
    auto                 prefix_lengths = prefixes.empty() ? at::Tensor() : at::empty({batch}, gpuInt32Options());
    auto                 q_storage      = at::empty({batch + 3}, gpuInt32Options());
    auto                 kv_storage     = at::empty({batch + 3}, gpuInt32Options());
    auto                 pad_storage    = at::empty({tokens + 2}, gpuInt32Options());
    auto                 cu_seqlens     = q_storage.narrow(0, 1, batch + 1);
    auto                 cu_kv_seqlens  = kv_storage.narrow(0, 1, batch + 1);
    auto                 padding        = pad_storage.narrow(0, 1, tokens);
    std::vector<int32_t> q_guard(batch + 3, guard), kv_guard(batch + 3, guard), pad_guard(tokens + 2, guard);
    // Destroy/drain the stream before its input and output allocations, also
    // if a failed runtime call unwinds the test early.
    Stream stream;
    ASSERT_NE(stream.value, nullptr);

    // Allocation is the only PyTorch GPU operation. Every write, the tested
    // launch, and every read are ordered on this explicit non-default stream.
    upload(queries, input_lengths, stream.value);
    if (!prefixes.empty()) {
        upload(prefixes, prefix_lengths, stream.value);
    }
    upload(q_guard, q_storage, stream.value);
    upload(kv_guard, kv_storage, stream.value);
    upload(pad_guard, pad_storage, stream.value);
    invokeBuildAttentionInputMetadata(input_lengths, prefix_lengths, cu_seqlens, cu_kv_seqlens, padding, stream.value);

    std::vector<int32_t> expected_q(batch + 3, guard), expected_kv(batch + 3, guard);
    expected_q[1] = expected_kv[1] = 0;
    std::partial_sum(queries.begin(), queries.end(), expected_q.begin() + 2);
    std::vector<int32_t> kv_lengths = queries;
    if (!prefixes.empty()) {
        std::transform(queries.begin(), queries.end(), prefixes.begin(), kv_lengths.begin(), std::plus<int32_t>());
    }
    std::partial_sum(kv_lengths.begin(), kv_lengths.end(), expected_kv.begin() + 2);
    auto expected_padding = paddingReference(queries);
    expected_padding.insert(expected_padding.begin(), guard);
    expected_padding.push_back(guard);

    EXPECT_EQ(download(q_storage, stream.value), expected_q);
    EXPECT_EQ(download(kv_storage, stream.value), expected_kv);
    EXPECT_EQ(download(pad_storage, stream.value), expected_padding);
}

TEST(AttentionInputMetadataTest, ExactEmptySingletonAndRaggedBatchesOnExplicitStream) {
    checkMetadata({}, {});
    checkMetadata({0}, {8192});
    checkMetadata({1}, {0});
    checkMetadata({8192}, {6144});
    checkMetadata({2, 0, 4, 1}, {});
    checkMetadata({0, 3, 1, 7, 2, 0, 4, 1, 6, 3, 0, 2, 7, 1, 5, 0},
                  {0, 63, 64, 65, 8191, 8192, 8193, 1, 127, 128, 129, 4095, 4096, 4097, 31, 32});
}

// Capturing a rejected call must produce an empty graph. Checking unchanged
// outputs alone would miss a launch that happens to write the original values.
void expectRejectedBeforeEnqueue(const std::function<void()>& invoke, AttentionInputMetadataStream stream) {
    checkGpu(METADATA_GPU(StreamBeginCapture)(stream, METADATA_GPU(StreamCaptureModeThreadLocal)));
    EXPECT_THROW(invoke(), c10::Error);
    METADATA_GPU(Graph_t) graph{};
    checkGpu(METADATA_GPU(StreamEndCapture)(stream, &graph));
    size_t     nodes  = 0;
    const auto result = METADATA_GPU(GraphGetNodes)(graph, nullptr, &nodes);
    METADATA_GPU(GraphDestroy)(graph);
    checkGpu(result);
    EXPECT_EQ(nodes, 0U) << "invalid tensor descriptors enqueued GPU work";
}

TEST(AttentionInputMetadataTest, RejectInvalidDescriptorsBeforeEnqueue) {
    auto   input   = at::empty({2}, gpuInt32Options());
    auto   prefix  = at::empty({2}, gpuInt32Options());
    auto   q       = at::empty({3}, gpuInt32Options());
    auto   kv      = at::empty({3}, gpuInt32Options());
    auto   padding = at::empty({4}, gpuInt32Options());
    Stream stream;

    auto reject = [&](const at::Tensor& query,
                      const at::Tensor& prefix_arg,
                      at::Tensor        q_arg,
                      at::Tensor        kv_arg,
                      at::Tensor        padding_arg) {
        expectRejectedBeforeEnqueue(
            [&] { invokeBuildAttentionInputMetadata(query, prefix_arg, q_arg, kv_arg, padding_arg, stream.value); },
            stream.value);
    };

    reject(at::Tensor(), prefix, q, kv, padding);
    reject(at::empty({2}, at::TensorOptions().dtype(at::kInt)), prefix, q, kv, padding);
    reject(at::empty({2}, gpuInt32Options().dtype(at::kLong)), prefix, q, kv, padding);
    reject(at::empty({}, gpuInt32Options()), prefix, q, kv, padding);
    reject(input.reshape({1, 2}), prefix, q, kv, padding);
    reject(at::empty({4}, gpuInt32Options()).slice(0, 0, 4, 2), prefix, q, kv, padding);
    reject(input, prefix.narrow(0, 0, 1), q, kv, padding);
    reject(input, prefix.reshape({1, 2}), q, kv, padding);
    reject(input, at::empty({2}, gpuInt32Options().dtype(at::kLong)), q, kv, padding);
    reject(input, at::empty({4}, gpuInt32Options()).slice(0, 0, 4, 2), q, kv, padding);
    reject(input, prefix, q.narrow(0, 0, 2), kv, padding);
    reject(input, prefix, q, kv.narrow(0, 0, 2), padding);
    reject(input, prefix, at::empty({3}, gpuInt32Options().dtype(at::kLong)), kv, padding);
    reject(input, prefix, q.reshape({1, 3}), kv, padding);
    reject(input, prefix, at::empty({6}, gpuInt32Options()).slice(0, 0, 6, 2), kv, padding);
    reject(input, prefix, q, kv, at::empty({4}, gpuInt32Options().dtype(at::kByte)));
    reject(input, prefix, q, kv, padding.reshape({2, 2}));
    reject(input, prefix, q, kv, at::empty({8}, gpuInt32Options()).slice(0, 0, 8, 2));
}

}  // namespace
}  // namespace rtp_llm

#undef METADATA_GPU
