#pragma once

#include <ATen/ATen.h>
#if USING_ROCM
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime_api.h>
#endif

namespace rtp_llm {

#if USING_ROCM
using AttentionInputMetadataStream = hipStream_t;
#else
using AttentionInputMetadataStream = cudaStream_t;
#endif

// Length values are non-negative int32; cumulative lengths and padded offsets
// must fit in int32. The caller supplies sum(input_lengths) padding entries when needed.
// Validation only inspects tensor descriptors, so this operation never reads
// device values on the host or synchronizes the supplied stream.
void invokeBuildAttentionInputMetadata(const at::Tensor&            input_lengths,
                                       const at::Tensor&            prefix_lengths,
                                       at::Tensor&                  cu_seqlens,
                                       at::Tensor&                  cu_kv_seqlens,
                                       at::Tensor&                  padding_offset,
                                       AttentionInputMetadataStream stream);

}  // namespace rtp_llm
