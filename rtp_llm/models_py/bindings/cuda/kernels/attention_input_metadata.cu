#include "rtp_llm/models_py/bindings/cuda/kernels/attention_input_metadata.h"

#include <c10/util/Exception.h>
#if USING_ROCM
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif
#include <cstdint>
#include <limits>

namespace rtp_llm {

namespace {

void checkInt32Vector(const at::Tensor& tensor, const char* name, const at::Device& device) {
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(tensor.is_cuda(), name, " must be a GPU tensor");
    TORCH_CHECK(tensor.device() == device, name, " must be on the input_lengths device");
    TORCH_CHECK(tensor.scalar_type() == at::kInt, name, " must be int32");
    TORCH_CHECK(tensor.dim() == 1, name, " must be one-dimensional");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void zeroAsync(at::Tensor& tensor, AttentionInputMetadataStream stream) {
    if (tensor.numel() == 0) {
        return;
    }
#if USING_ROCM
    const auto result = hipMemsetAsync(tensor.data_ptr<int32_t>(), 0, tensor.nbytes(), stream);
    TORCH_CHECK(result == hipSuccess, "attention input metadata zero failed: ", hipGetErrorString(result));
#else
    const auto result = cudaMemsetAsync(tensor.data_ptr<int32_t>(), 0, tensor.nbytes(), stream);
    TORCH_CHECK(result == cudaSuccess, "attention input metadata zero failed: ", cudaGetErrorString(result));
#endif
}

__global__ void buildAttentionInputMetadataKernel(const int32_t* __restrict__ input_lengths,
                                                  const int32_t* __restrict__ prefix_lengths,
                                                  int32_t* __restrict__ cu_seqlens,
                                                  int32_t* __restrict__ cu_kv_seqlens,
                                                  int32_t* __restrict__ padding_offset,
                                                  int32_t batch_size,
                                                  int32_t total_tokens) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    int32_t max_input_len = 0;
    int32_t q_acc         = 0;
    int32_t kv_acc        = 0;
    cu_seqlens[0]         = 0;
    cu_kv_seqlens[0]      = 0;

    for (int32_t b = 0; b < batch_size; ++b) {
        const int32_t input_len  = input_lengths[b];
        const int32_t prefix_len = prefix_lengths ? prefix_lengths[b] : 0;
        max_input_len            = max_input_len > input_len ? max_input_len : input_len;
        q_acc += input_len;
        kv_acc += input_len + prefix_len;
        cu_seqlens[b + 1]    = q_acc;
        cu_kv_seqlens[b + 1] = kv_acc;
    }

    if (!padding_offset || total_tokens <= 0) {
        return;
    }

    int32_t out_idx    = 0;
    int32_t cum_offset = 0;
    for (int32_t b = 0; b < batch_size; ++b) {
        const int32_t input_len = input_lengths[b];
        for (int32_t j = 0; j < input_len && out_idx < total_tokens; ++j) {
            padding_offset[out_idx++] = cum_offset;
        }
        cum_offset += max_input_len - input_len;
    }
}

}  // namespace

void invokeBuildAttentionInputMetadata(const at::Tensor&            input_lengths,
                                       const at::Tensor&            prefix_lengths,
                                       at::Tensor&                  cu_seqlens,
                                       at::Tensor&                  cu_kv_seqlens,
                                       at::Tensor&                  padding_offset,
                                       AttentionInputMetadataStream stream) {
    TORCH_CHECK(input_lengths.defined(), "input_lengths must be defined");
    checkInt32Vector(input_lengths, "input_lengths", input_lengths.device());
    checkInt32Vector(cu_seqlens, "cu_seqlens", input_lengths.device());
    checkInt32Vector(cu_kv_seqlens, "cu_kv_seqlens", input_lengths.device());
    TORCH_CHECK(input_lengths.numel() < std::numeric_limits<int32_t>::max(), "input_lengths batch is too large");
    TORCH_CHECK(cu_seqlens.numel() >= input_lengths.numel() + 1, "cu_seqlens must cover batch_size + 1 entries");
    TORCH_CHECK(cu_kv_seqlens.numel() >= input_lengths.numel() + 1, "cu_kv_seqlens must cover batch_size + 1 entries");
    if (padding_offset.defined()) {
        checkInt32Vector(padding_offset, "padding_offset", input_lengths.device());
        TORCH_CHECK(padding_offset.numel() <= std::numeric_limits<int32_t>::max(), "padding_offset is too large");
    }
    const int32_t* prefix_ptr = nullptr;
    if (prefix_lengths.defined() && prefix_lengths.numel() > 0) {
        checkInt32Vector(prefix_lengths, "prefix_lengths", input_lengths.device());
        TORCH_CHECK(prefix_lengths.numel() >= input_lengths.numel(), "prefix_lengths size must cover input_lengths");
        prefix_ptr = prefix_lengths.data_ptr<int32_t>();
    }

    const auto batch_size   = static_cast<int32_t>(input_lengths.size(0));
    const auto total_tokens = padding_offset.defined() ? static_cast<int32_t>(padding_offset.numel()) : 0;
    if (batch_size == 0) {
        // Use the explicit stream, including when it differs from PyTorch's
        // current stream. zero_() would otherwise enqueue on the latter.
        zeroAsync(cu_seqlens, stream);
        zeroAsync(cu_kv_seqlens, stream);
        if (padding_offset.defined()) {
            zeroAsync(padding_offset, stream);
        }
        return;
    }

    buildAttentionInputMetadataKernel<<<1, 1, 0, stream>>>(
        input_lengths.data_ptr<int32_t>(),
        prefix_ptr,
        cu_seqlens.data_ptr<int32_t>(),
        cu_kv_seqlens.data_ptr<int32_t>(),
        padding_offset.defined() && padding_offset.numel() > 0 ? padding_offset.data_ptr<int32_t>() : nullptr,
        batch_size,
        total_tokens);
#if USING_ROCM
    const auto result = hipGetLastError();
    TORCH_CHECK(result == hipSuccess, "build attention input metadata kernel failed: ", hipGetErrorString(result));
#else
    const auto result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess, "build attention input metadata kernel failed: ", cudaGetErrorString(result));
#endif
}

}  // namespace rtp_llm
