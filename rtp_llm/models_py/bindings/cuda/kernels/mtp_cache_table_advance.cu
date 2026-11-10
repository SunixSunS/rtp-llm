#include "rtp_llm/models_py/bindings/cuda/kernels/mtp_cache_table_advance.h"

#include <ATen/MemoryOverlap.h>
#include <algorithm>
#include <c10/util/Exception.h>
#if USING_ROCM
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

namespace rtp_llm {
namespace {

__device__ int64_t swappedIndex(int64_t index, int64_t first, int64_t second) {
    return index == first ? second : (index == second ? first : index);
}

__global__ void advanceMtpCacheTableKernel(const int32_t* __restrict__ current_table,
                                           const int32_t* __restrict__ previous_seq_lengths,
                                           const int32_t* __restrict__ accept_lengths,
                                           uint64_t linear_group_mask,
                                           int64_t  seq_size_per_block,
                                           int64_t  batch_size,
                                           int64_t  block_count,
                                           int64_t  row_count,
                                           int32_t* __restrict__ output) {
    // Compute the row's integer divisions once. Every thread then copies its
    // columns directly from the immutable source, including non-LINEAR groups.
    __shared__ int64_t cached_source;
    __shared__ int64_t cached_destination;
    __shared__ int64_t final_source;
    __shared__ int64_t final_destination;
    for (int64_t row_index = blockIdx.x; row_index < row_count; row_index += gridDim.x) {
        if (threadIdx.x == 0) {
            cached_source = cached_destination = final_source = final_destination = 0;
            const int64_t group                                                   = row_index / batch_size;
            const int64_t row                                                     = row_index % batch_size;
            const int64_t accept                                                  = accept_lengths[row];
            if (((linear_group_mask >> group) & uint64_t{1}) && accept > 1) {
                const int64_t current_cached = static_cast<int64_t>(previous_seq_lengths[row]) - 1;
                const int64_t next_cached    = current_cached + accept;
                // (next_cached - 1) % span is equivalent to the utility's
                // (next_cached + span - 1) % span without a potentially large
                // addition when the host span is near the int64 limit.
                if ((current_cached + 1) % seq_size_per_block > (next_cached - 1) % seq_size_per_block) {
                    cached_source = current_cached / seq_size_per_block
                                    + (seq_size_per_block - current_cached % seq_size_per_block - 1);
                    cached_destination = next_cached / seq_size_per_block - 1;
                }
                final_source      = current_cached / seq_size_per_block + accept - 1;
                final_destination = (next_cached - 1) / seq_size_per_block;
            }
        }
        __syncthreads();

        const int64_t row_offset = row_index * block_count;
        for (int64_t column = threadIdx.x; column < block_count; column += blockDim.x) {
            // output = final_swap(cached_swap(input)); an output lookup must
            // undo final first, then cached. The order matters for overlapping
            // swaps, which form a three-element cycle rather than two pairs.
            int64_t source              = swappedIndex(column, final_source, final_destination);
            source                      = swappedIndex(source, cached_source, cached_destination);
            output[row_offset + column] = current_table[row_offset + source];
        }
        // A grid-stride iteration must not overwrite shared swap indices while
        // another warp still uses them for the preceding row's final columns.
        __syncthreads();
    }
}

void checkTensor(const at::Tensor& tensor, const char* name, const at::Device& device, int64_t rank) {
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(tensor.is_cuda() && tensor.device() == device, name, " must be on the current_table GPU");
    TORCH_CHECK(tensor.scalar_type() == at::kInt, name, " must be int32");
    TORCH_CHECK(tensor.dim() == rank && tensor.is_contiguous(), name, " has invalid rank or strides");
}

}  // namespace

void invokeMtpCacheTableAdvance(const at::Tensor&          current_table,
                                const at::Tensor&          previous_seq_lengths,
                                const at::Tensor&          accept_lengths,
                                uint64_t                   linear_group_mask,
                                int64_t                    seq_size_per_block,
                                at::Tensor&                output,
                                MtpCacheTableAdvanceStream stream) {
    TORCH_CHECK(current_table.defined(), "current_table must be defined");
    checkTensor(current_table, "current_table", current_table.device(), 3);
    checkTensor(previous_seq_lengths, "previous_seq_lengths", current_table.device(), 1);
    checkTensor(accept_lengths, "accept_lengths", current_table.device(), 1);
    checkTensor(output, "output", current_table.device(), 3);
    const int64_t group_count = current_table.size(0);
    const int64_t batch_size  = current_table.size(1);
    const int64_t block_count = current_table.size(2);
    TORCH_CHECK(group_count <= 64, "MTP table advance supports at most 64 groups");
    TORCH_CHECK(group_count == 64 || (linear_group_mask >> group_count) == 0,
                "linear_group_mask contains a group outside current_table");
    TORCH_CHECK(seq_size_per_block > 0, "seq_size_per_block must be positive");
    TORCH_CHECK(previous_seq_lengths.numel() == batch_size && accept_lengths.numel() == batch_size,
                "MTP table advance length batch does not match current_table");
    TORCH_CHECK(output.sizes() == current_table.sizes(), "output shape must match current_table");
    at::assert_no_overlap(output, current_table);
    at::assert_no_overlap(output, previous_seq_lengths);
    at::assert_no_overlap(output, accept_lengths);
    if (current_table.numel() == 0) {
        return;
    }

    const int64_t row_count = group_count * batch_size;
    const int     blocks    = static_cast<int>(std::min<int64_t>(row_count, 65535));
    advanceMtpCacheTableKernel<<<blocks, 128, 0, stream>>>(current_table.data_ptr<int32_t>(),
                                                           previous_seq_lengths.data_ptr<int32_t>(),
                                                           accept_lengths.data_ptr<int32_t>(),
                                                           linear_group_mask,
                                                           seq_size_per_block,
                                                           batch_size,
                                                           block_count,
                                                           row_count,
                                                           output.data_ptr<int32_t>());
#if USING_ROCM
    const auto result = hipGetLastError();
    TORCH_CHECK(result == hipSuccess, "MTP cache table advance failed: ", hipGetErrorString(result));
#else
    const auto result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess, "MTP cache table advance failed: ", cudaGetErrorString(result));
#endif
}

}  // namespace rtp_llm
