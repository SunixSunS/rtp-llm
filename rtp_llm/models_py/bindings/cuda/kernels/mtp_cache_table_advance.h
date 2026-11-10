#pragma once

#include <ATen/ATen.h>
#include <cstdint>
#if USING_ROCM
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime_api.h>
#endif

namespace rtp_llm {

#if USING_ROCM
using MtpCacheTableAdvanceStream = hipStream_t;
#else
using MtpCacheTableAdvanceStream = cudaStream_t;
#endif

// Immutable-input form of LinearBlocksUtil's cached-token then final-token
// swaps. All tensors are contiguous GPU int32 on one device. current_table and
// output are [group,batch,blocks], with group <= 64; lengths are [batch]. output
// must not overlap an input. The caller owns every allocation until stream work
// completes and orders producers/consumers on this explicit stream.
//
// Device-value contract: previous_seq_lengths >= 1, accept_lengths >= 1, and
// the caller's reserve-step policy keeps active swap indices within blocks.
// These values are never read on the CPU. Descriptor errors throw before any
// launch; empty dimensions enqueue nothing. The operation allocates no tensor
// storage and never synchronizes the host.
void invokeMtpCacheTableAdvance(const at::Tensor&          current_table,
                                const at::Tensor&          previous_seq_lengths,
                                const at::Tensor&          accept_lengths,
                                uint64_t                   linear_group_mask,
                                int64_t                    seq_size_per_block,
                                at::Tensor&                output,
                                MtpCacheTableAdvanceStream stream);

}  // namespace rtp_llm
