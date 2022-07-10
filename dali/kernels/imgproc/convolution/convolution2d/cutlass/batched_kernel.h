#pragma once

#include <vector>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"

namespace cutlass {

template <typename Operator>
struct BatchedKernel {

    using Params = typename Operator::Params*;
    using SharedStorage = typename Operator::SharedStorage;

    CUTLASS_HOST_DEVICE
    BatchedKernel() {}

    CUTLASS_DEVICE
    void operator()(typename Operator::Params* params_batch, SharedStorage &shared_storage) {
        int sample_idx = gemm::threadblock::RematerializeBlockIdxZ();
        typename Operator::Params params = params_batch[sample_idx];
        Operator op;
        op(params, shared_storage);
    }

};
} /// namespace cutlass
