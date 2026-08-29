// Minimal standalone probe: run one flash_fwd kernel instantiation with
// selectable traits on sm75 and dump O + LSE. Non-varlen, single batch.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "namespace_config.h"
#include "flash.h"
#include "static_switch.h"
#include "flash_fwd_launch_template.h"

using namespace FLASH_NAMESPACE;

// Traits chosen at compile time via -DPROBE_* defines.
#ifndef PROBE_HEADDIM
#define PROBE_HEADDIM 128
#endif
#ifndef PROBE_BLOCK_M
#define PROBE_BLOCK_M 128
#endif
#ifndef PROBE_BLOCK_N
#define PROBE_BLOCK_N 64
#endif

std::vector<torch::Tensor> probe_fwd(torch::Tensor q, torch::Tensor k,
                                     torch::Tensor v, double softmax_scale) {
    // shapes: q (b, sq, h, d), k/v (b, sk, hk, d), fp16, contiguous
    auto b = q.size(0), sq = q.size(1), h = q.size(2), d = q.size(3);
    auto sk = k.size(1), hk = k.size(2);
    TORCH_CHECK(d == PROBE_HEADDIM, "headdim mismatch");

    auto out = torch::zeros_like(q);
    auto lse = torch::empty({b, h, sq}, q.options().dtype(torch::kFloat32));

    Flash_fwd_params params{};
    params.is_bf16 = false;
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    params.o_ptr = out.data_ptr();
    params.softmax_lse_ptr = lse.data_ptr();
    params.q_row_stride = q.stride(1);
    params.k_row_stride = k.stride(1);
    params.v_row_stride = v.stride(1);
    params.o_row_stride = out.stride(1);
    params.q_head_stride = q.stride(2);
    params.k_head_stride = k.stride(2);
    params.v_head_stride = v.stride(2);
    params.o_head_stride = out.stride(2);
    params.q_batch_stride = q.stride(0);
    params.k_batch_stride = k.stride(0);
    params.v_batch_stride = v.stride(0);
    params.o_batch_stride = out.stride(0);
    params.b = b;
    params.h = h;
    params.h_k = hk;
    params.h_h_k_ratio = h / hk;
    params.seqlen_q = sq;
    params.seqlen_k = sk;
    params.seqlen_q_rounded = (sq + 127) / 128 * 128;
    params.seqlen_k_rounded = (sk + 127) / 128 * 128;
    params.d = d;
    params.d_rounded = d;
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;
    params.softcap = 0.f;
    params.p_dropout = 1.f;  // keep-prob 1 => no dropout
    params.p_dropout_in_uint8_t = 255;
    params.rp_dropout = 1.f;
    params.scale_softmax_rp_dropout = params.scale_softmax;
    params.is_causal = false;
    params.window_size_left = -1;
    params.window_size_right = -1;
    params.is_seqlens_k_cumulative = true;
    params.unpadded_lse = false;

    using T = cutlass::half_t;
    run_flash_fwd<
        Flash_fwd_kernel_traits<PROBE_HEADDIM, PROBE_BLOCK_M, PROBE_BLOCK_N,
                                4, false, false, T>,
        /*Is_dropout=*/false, /*Is_causal=*/false>(
        params, at::cuda::getCurrentCUDAStream());
    return {out, lse};
}

std::vector<torch::Tensor> probe_splitkv(torch::Tensor q, torch::Tensor k,
                                         torch::Tensor v, double softmax_scale,
                                         int64_t num_splits) {
    auto b = q.size(0), sq = q.size(1), h = q.size(2), d = q.size(3);
    auto sk = k.size(1), hk = k.size(2);
    TORCH_CHECK(d == PROBE_HEADDIM, "headdim mismatch");

    auto out = torch::zeros_like(q);
    auto lse = torch::empty({b, h, sq}, q.options().dtype(torch::kFloat32));
    auto lse_accum = torch::zeros({num_splits, b, h, sq},
                                  q.options().dtype(torch::kFloat32));
    auto out_accum = torch::zeros({num_splits, b, h, sq, d},
                                  q.options().dtype(torch::kFloat32));

    Flash_fwd_params params{};
    params.is_bf16 = false;
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    params.o_ptr = out.data_ptr();
    params.softmax_lse_ptr = lse.data_ptr();
    params.softmax_lseaccum_ptr = lse_accum.data_ptr();
    params.oaccum_ptr = out_accum.data_ptr();
    params.q_row_stride = q.stride(1);
    params.k_row_stride = k.stride(1);
    params.v_row_stride = v.stride(1);
    params.o_row_stride = out.stride(1);
    params.q_head_stride = q.stride(2);
    params.k_head_stride = k.stride(2);
    params.v_head_stride = v.stride(2);
    params.o_head_stride = out.stride(2);
    params.q_batch_stride = q.stride(0);
    params.k_batch_stride = k.stride(0);
    params.v_batch_stride = v.stride(0);
    params.o_batch_stride = out.stride(0);
    params.b = b;
    params.h = h;
    params.h_k = hk;
    params.h_h_k_ratio = h / hk;
    params.seqlen_q = sq;
    params.seqlen_k = sk;
    params.seqlen_q_rounded = (sq + 127) / 128 * 128;
    params.seqlen_k_rounded = (sk + 127) / 128 * 128;
    params.d = d;
    params.d_rounded = d;
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;
    params.softcap = 0.f;
    params.p_dropout = 1.f;
    params.p_dropout_in_uint8_t = 255;
    params.rp_dropout = 1.f;
    params.scale_softmax_rp_dropout = params.scale_softmax;
    params.is_causal = false;
    params.window_size_left = -1;
    params.window_size_right = -1;
    params.is_seqlens_k_cumulative = true;
    params.unpadded_lse = false;
    params.num_splits = num_splits;

    using T = cutlass::half_t;
    run_mha_fwd_splitkv_dispatch<T, PROBE_HEADDIM, /*Is_causal=*/false>(
        params, at::cuda::getCurrentCUDAStream());
    return {out, lse};
}

std::vector<torch::Tensor> probe_varlen(torch::Tensor q, torch::Tensor k,
                                        torch::Tensor v,
                                        torch::Tensor cu_seqlens_q,
                                        torch::Tensor cu_seqlens_k,
                                        torch::Tensor seqused_k,
                                        int64_t max_seqlen_q,
                                        int64_t max_seqlen_k,
                                        double softmax_scale,
                                        bool unpadded_lse,
                                        bool use_seqused) {
    // q (total_q, h, d); k/v (total_k, hk, d)
    auto total_q = q.size(0), h = q.size(1), d = q.size(2);
    auto hk = k.size(1);
    auto b = cu_seqlens_q.size(0) - 1;
    TORCH_CHECK(d == PROBE_HEADDIM, "headdim mismatch");

    auto out = torch::zeros_like(q);
    auto lse = unpadded_lse
        ? torch::empty({h, total_q}, q.options().dtype(torch::kFloat32))
        : torch::empty({b, h, max_seqlen_q}, q.options().dtype(torch::kFloat32));

    Flash_fwd_params params{};
    params.is_bf16 = false;
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    params.o_ptr = out.data_ptr();
    params.softmax_lse_ptr = lse.data_ptr();
    params.cu_seqlens_q = static_cast<int*>(cu_seqlens_q.data_ptr());
    params.cu_seqlens_k = static_cast<int*>(cu_seqlens_k.data_ptr());
    params.seqused_k = use_seqused ? static_cast<int*>(seqused_k.data_ptr()) : nullptr;
    params.q_row_stride = q.stride(0);
    params.k_row_stride = k.stride(0);
    params.v_row_stride = v.stride(0);
    params.o_row_stride = out.stride(0);
    params.q_head_stride = q.stride(1);
    params.k_head_stride = k.stride(1);
    params.v_head_stride = v.stride(1);
    params.o_head_stride = out.stride(1);
    params.b = b;
    params.h = h;
    params.h_k = hk;
    params.h_h_k_ratio = h / hk;
    params.seqlen_q = max_seqlen_q;
    params.seqlen_k = max_seqlen_k;
    params.seqlen_q_rounded = (max_seqlen_q + 127) / 128 * 128;
    params.seqlen_k_rounded = (max_seqlen_k + 127) / 128 * 128;
    params.d = d;
    params.d_rounded = d;
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;
    params.softcap = 0.f;
    params.p_dropout = 1.f;
    params.p_dropout_in_uint8_t = 255;
    params.rp_dropout = 1.f;
    params.scale_softmax_rp_dropout = params.scale_softmax;
    params.is_causal = false;
    params.window_size_left = -1;
    params.window_size_right = -1;
    params.is_seqlens_k_cumulative = true;
    params.unpadded_lse = unpadded_lse;

    using T = cutlass::half_t;
    run_flash_fwd<
        Flash_fwd_kernel_traits<PROBE_HEADDIM, PROBE_BLOCK_M, PROBE_BLOCK_N,
                                4, false, false, T>,
        /*Is_dropout=*/false, /*Is_causal=*/false>(
        params, at::cuda::getCurrentCUDAStream());
    return {out, lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("probe_fwd", &probe_fwd, "flash fwd probe");
    m.def("probe_splitkv", &probe_splitkv, "flash splitkv probe");
    m.def("probe_varlen", &probe_varlen, "flash varlen fwd probe");
}
