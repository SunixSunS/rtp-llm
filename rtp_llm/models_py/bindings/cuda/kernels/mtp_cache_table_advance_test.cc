#include "rtp_llm/models_py/bindings/cuda/kernels/mtp_cache_table_advance.h"
#include "rtp_llm/cpp/utils/LinearBlocksUtil.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

#include <c10/util/Exception.h>
#include <gtest/gtest.h>

#if USING_ROCM
#define MTP_GPU(name) hip##name
#else
#include <cuda_runtime.h>
#define MTP_GPU(name) cuda##name
#endif

namespace rtp_llm {
namespace {

void checkGpu(MTP_GPU(Error_t) error) {
    if (error != MTP_GPU(Success)) {
        throw std::runtime_error(MTP_GPU(GetErrorString)(error));
    }
}

struct Stream {
    MtpCacheTableAdvanceStream value{};
    Stream() {
        checkGpu(MTP_GPU(StreamCreateWithFlags)(&value, MTP_GPU(StreamNonBlocking)));
    }
    ~Stream() {
        MTP_GPU(StreamSynchronize)(value);
        MTP_GPU(StreamDestroy)(value);
    }
    Stream(const Stream&)            = delete;
    Stream& operator=(const Stream&) = delete;
};

struct Graph {
    MTP_GPU(Graph_t) graph {};
    MTP_GPU(GraphExec_t) executable {};
    ~Graph() {
        if (executable) {
            MTP_GPU(GraphExecDestroy)(executable);
        }
        if (graph) {
            MTP_GPU(GraphDestroy)(graph);
        }
    }
};

at::TensorOptions gpuOptions() {
    return at::TensorOptions().dtype(at::kInt).device(at::kCUDA);
}

void upload(const std::vector<int32_t>& source, at::Tensor& destination, MtpCacheTableAdvanceStream stream) {
    if (!source.empty()) {
        checkGpu(MTP_GPU(MemcpyAsync)(destination.data_ptr<int32_t>(),
                                      source.data(),
                                      source.size() * sizeof(int32_t),
                                      MTP_GPU(MemcpyHostToDevice),
                                      stream));
    }
}

std::vector<int32_t> download(const at::Tensor& source, MtpCacheTableAdvanceStream stream) {
    std::vector<int32_t> result(source.numel());
    if (!result.empty()) {
        checkGpu(MTP_GPU(MemcpyAsync)(result.data(),
                                      source.data_ptr<int32_t>(),
                                      result.size() * sizeof(int32_t),
                                      MTP_GPU(MemcpyDeviceToHost),
                                      stream));
    }
    checkGpu(MTP_GPU(StreamSynchronize)(stream));
    return result;
}

struct Buffers {
    static constexpr int32_t guard = -123456789;
    at::Tensor               table;
    at::Tensor               previous;
    at::Tensor               accepted;
    at::Tensor               output_storage;
    at::Tensor               output;
    // Drain first on destruction, before releasing any device allocation.
    Stream stream;

    Buffers(int64_t groups, int64_t batch, int64_t blocks):
        table(at::empty({groups, batch, blocks}, gpuOptions())),
        previous(at::empty({batch}, gpuOptions())),
        accepted(at::empty({batch}, gpuOptions())),
        output_storage(at::empty({groups * batch * blocks + 2}, gpuOptions())),
        output(output_storage.narrow(0, 1, groups * batch * blocks).view({groups, batch, blocks})) {}

    void invoke(uint64_t mask, int64_t span) {
        invokeMtpCacheTableAdvance(table, previous, accepted, mask, span, output, stream.value);
    }
};

std::vector<int32_t> pattern(int64_t count, int32_t round) {
    std::vector<int32_t> values(count);
    for (int64_t index = 0; index < count; ++index) {
        values[index] = static_cast<int32_t>(index * 37 + round * 100003 + 11);
    }
    return values;
}

// Apply the existing CPU utility's swaps to mutable rows, in forward order.
// This oracle does not construct or invert the GPU kernel's index mapping.
std::vector<int32_t> reference(const std::vector<int32_t>& table,
                               const std::vector<int32_t>& previous,
                               const std::vector<int32_t>& accepted,
                               int64_t                     groups,
                               int64_t                     blocks,
                               uint64_t                    mask,
                               int                         span) {
    auto          result = table;
    const int64_t batch  = previous.size();
    for (int64_t group = 0; group < groups; ++group) {
        if (((mask >> group) & uint64_t{1}) == 0) {
            continue;
        }
        for (int64_t row = 0; row < batch; ++row) {
            if (accepted[row] <= 1) {
                continue;
            }
            const int  cached      = previous[row] - 1;
            const int  next        = cached + accepted[row];
            const auto cached_swap = getCachedTokenBlockSwapIdx(cached, next, span);
            const auto final_swap  = getFinalTokenBlockSwapIdx(cached, next, span);
            auto       apply       = [&](const std::pair<int, int>& swap) {
                if (swap.first < 0 || swap.second < 0 || swap.first >= blocks || swap.second >= blocks) {
                    throw std::runtime_error("test oracle generated a swap outside reserved table capacity");
                }
                const int64_t offset = (group * batch + row) * blocks;
                std::swap(result[offset + swap.first], result[offset + swap.second]);
            };
            apply(cached_swap);
            apply(final_swap);
        }
    }
    return result;
}

void expectOutput(Buffers& buffers, const std::vector<int32_t>& expected, const std::vector<int32_t>& source) {
    auto guarded = expected;
    guarded.insert(guarded.begin(), Buffers::guard);
    guarded.push_back(Buffers::guard);
    EXPECT_EQ(download(buffers.output_storage, buffers.stream.value), guarded);
    EXPECT_EQ(download(buffers.table, buffers.stream.value), source) << "current_table was modified";
}

void runCase(int64_t                     groups,
             const std::vector<int32_t>& previous,
             const std::vector<int32_t>& accepted,
             int64_t                     blocks,
             uint64_t                    mask,
             int                         span) {
    SCOPED_TRACE(::testing::Message() << "groups=" << groups << ", B=" << previous.size() << ", blocks=" << blocks
                                      << ", span=" << span << ", mask=" << mask);
    ASSERT_EQ(previous.size(), accepted.size());
    Buffers                    buffers(groups, previous.size(), blocks);
    const auto                 source = pattern(buffers.table.numel(), 0);
    const std::vector<int32_t> guards(buffers.output_storage.numel(), Buffers::guard);
    upload(source, buffers.table, buffers.stream.value);
    upload(previous, buffers.previous, buffers.stream.value);
    upload(accepted, buffers.accepted, buffers.stream.value);
    upload(guards, buffers.output_storage, buffers.stream.value);
    buffers.invoke(mask, span);
    expectOutput(buffers, reference(source, previous, accepted, groups, blocks, mask, span), source);
    EXPECT_EQ(download(buffers.previous, buffers.stream.value), previous);
    EXPECT_EQ(download(buffers.accepted, buffers.stream.value), accepted);
}

TEST(MtpCacheTableAdvanceTest, BoundarySwapsMixedGroupsAndOddWidths) {
    const std::array<int32_t, 10> boundaries{1, 62, 63, 64, 65, 126, 127, 128, 129, 8192};
    for (int batch : {1, 4, 16, 33}) {
        for (int round = 0; round < 3; ++round) {
            std::vector<int32_t> previous(batch), accepted(batch);
            for (int row = 0; row < batch; ++row) {
                previous[row] = boundaries[(row + round) % boundaries.size()];
                accepted[row] = 1 + (row + round) % 3;
            }
            runCase(5, previous, accepted, 131, 0b01010, 64);
        }
    }
    // Overlap forms a three-cycle; previous=64/accept=2 has a same-index final swap.
    runCase(3, {63, 64, 65, 127}, {3, 2, 1, 3}, 5, 0b101, 64);
    runCase(2, {1, 2, 3, 4}, {1, 2, 3, 2}, 9, 0b11, 1);
    runCase(64, {63}, {3}, 5, (uint64_t{1} << 63) | 1, 64);
    runCase(3, {63, 127}, {3, 2}, 5, 0, 64);
}

TEST(MtpCacheTableAdvanceTest, Int32LengthsUseInt64Arithmetic) {
    Buffers                    buffers(1, 1, 3);
    const std::vector<int32_t> source{17, 23, 31};
    const std::vector<int32_t> previous{std::numeric_limits<int32_t>::max()};
    const std::vector<int32_t> accepted{3};
    const std::vector<int32_t> guards(5, Buffers::guard);
    upload(source, buffers.table, buffers.stream.value);
    upload(previous, buffers.previous, buffers.stream.value);
    upload(accepted, buffers.accepted, buffers.stream.value);
    upload(guards, buffers.output_storage, buffers.stream.value);
    buffers.invoke(1, std::numeric_limits<int32_t>::max());
    // next_cached exceeds INT32_MAX. There is no cached swap and final swaps 2/1.
    expectOutput(buffers, {17, 31, 23}, source);
}

TEST(MtpCacheTableAdvanceTest, GraphReplayRefreshesTableAndBothLengthInputs) {
    constexpr int64_t  groups = 3, batch = 16, blocks = 131;
    constexpr uint64_t mask = 0b101;
    Buffers            buffers(groups, batch, blocks);
    Graph              graph;
    checkGpu(MTP_GPU(StreamBeginCapture)(buffers.stream.value, MTP_GPU(StreamCaptureModeThreadLocal)));
    buffers.invoke(mask, 64);
    checkGpu(MTP_GPU(StreamEndCapture)(buffers.stream.value, &graph.graph));
    size_t nodes = 0;
    checkGpu(MTP_GPU(GraphGetNodes)(graph.graph, nullptr, &nodes));
    ASSERT_EQ(nodes, 1U);
    checkGpu(MTP_GPU(GraphInstantiateWithFlags)(&graph.executable, graph.graph, 0));
    const std::array<int32_t, 8> boundaries{62, 63, 64, 65, 127, 128, 129, 8192};
    for (int round = 0; round < 6; ++round) {
        const auto           source = pattern(buffers.table.numel(), round + 1);
        std::vector<int32_t> previous(batch), accepted(batch);
        for (int row = 0; row < batch; ++row) {
            previous[row] = boundaries[(row + round) % boundaries.size()];
            accepted[row] = 1 + (row + round) % 3;
        }
        const std::vector<int32_t> guards(buffers.output_storage.numel(), Buffers::guard);
        upload(source, buffers.table, buffers.stream.value);
        upload(previous, buffers.previous, buffers.stream.value);
        upload(accepted, buffers.accepted, buffers.stream.value);
        upload(guards, buffers.output_storage, buffers.stream.value);
        checkGpu(MTP_GPU(GraphLaunch)(graph.executable, buffers.stream.value));
        expectOutput(buffers, reference(source, previous, accepted, groups, blocks, mask, 64), source);
        EXPECT_EQ(download(buffers.previous, buffers.stream.value), previous);
        EXPECT_EQ(download(buffers.accepted, buffers.stream.value), accepted);
    }
}

TEST(MtpCacheTableAdvanceTest, EmptyDimensionsCaptureNoGpuWork) {
    for (const auto shape : {std::array<int64_t, 3>{3, 0, 5}, {3, 4, 0}, {0, 4, 5}}) {
        Buffers buffers(shape[0], shape[1], shape[2]);
        Graph   graph;
        checkGpu(MTP_GPU(StreamBeginCapture)(buffers.stream.value, MTP_GPU(StreamCaptureModeThreadLocal)));
        buffers.invoke(0, 64);
        checkGpu(MTP_GPU(StreamEndCapture)(buffers.stream.value, &graph.graph));
        size_t nodes = 0;
        checkGpu(MTP_GPU(GraphGetNodes)(graph.graph, nullptr, &nodes));
        EXPECT_EQ(nodes, 0U);
    }
}

TEST(MtpCacheTableAdvanceTest, InvalidDescriptorsAndAliasesRejectBeforeEnqueue) {
    Buffers buffers(2, 4, 5);
    auto    reject = [&](const at::Tensor& table,
                      const at::Tensor& previous,
                      const at::Tensor& accepted,
                      uint64_t          mask,
                      int64_t           span,
                      at::Tensor        output) {
        Graph graph;
        checkGpu(MTP_GPU(StreamBeginCapture)(buffers.stream.value, MTP_GPU(StreamCaptureModeThreadLocal)));
        EXPECT_THROW(invokeMtpCacheTableAdvance(table, previous, accepted, mask, span, output, buffers.stream.value),
                     c10::Error);
        checkGpu(MTP_GPU(StreamEndCapture)(buffers.stream.value, &graph.graph));
        size_t nodes = 0;
        checkGpu(MTP_GPU(GraphGetNodes)(graph.graph, nullptr, &nodes));
        EXPECT_EQ(nodes, 0U);
    };
    const auto& table    = buffers.table;
    const auto& previous = buffers.previous;
    const auto& accepted = buffers.accepted;
    const auto& output   = buffers.output;
    reject(table, previous, accepted, 4, 64, output);
    reject(table, previous, accepted, 1, 0, output);
    reject(table, previous, accepted, 1, 64, table);
    reject(table.transpose(1, 2), previous, accepted, 1, 64, output);
    reject(table.view({8, 5}), previous, accepted, 1, 64, output);
    reject(table, previous.narrow(0, 0, 3), accepted, 1, 64, output);
    reject(table, at::empty({8}, gpuOptions()).slice(0, 0, 8, 2), accepted, 1, 64, output);
    reject(table, previous, at::empty({4}, gpuOptions().dtype(at::kLong)), 1, 64, output);
    reject(table, previous, accepted, 1, 64, output.narrow(2, 0, 4));
    reject(table, previous, accepted, 1, 64, at::empty({2, 4, 5}, gpuOptions().dtype(at::kLong)));
    reject(at::empty({65, 4, 5}, gpuOptions()), previous, accepted, 0, 64, at::empty({65, 4, 5}, gpuOptions()));
    // Partial overlap is just as unsafe as output==input. The two table views
    // cover shifted ranges of one allocation and both remain contiguous.
    auto shared = at::empty({41}, gpuOptions());
    reject(shared.narrow(0, 0, 40).view({2, 4, 5}), previous, accepted, 1, 64, shared.narrow(0, 1, 40).view({2, 4, 5}));
    reject(table, output.view({-1}).narrow(0, 0, 4), accepted, 1, 64, output);
    reject(table, previous, output.view({-1}).narrow(0, 0, 4), 1, 64, output);
}

}  // namespace
}  // namespace rtp_llm

#undef MTP_GPU
