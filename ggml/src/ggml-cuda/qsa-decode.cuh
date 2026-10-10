#pragma once
#include "common.cuh"
// the per-query decode kernels take 1..512 queries; larger ubatches go to qsa_prefill. Up to 512 queries the
// per-query gathers cost less than qsa_prefill's pack of the whole K/V cache, which grows with the depth
#define QSA_DECODE_MAX_QUERIES 512
bool ggml_cuda_flash_attn_ext_qsa_decode_supported(int device, const ggml_tensor * dst);
void ggml_cuda_flash_attn_ext_qsa_decode(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// K/V rows the QSA kernels (decode and prefill) read: f16, or Q8_0 that they dequantize to f16 on load
static inline bool qsa_kv_type_supported(const ggml_tensor * t) {
    return (t->type == GGML_TYPE_F16 && t->nb[0] == 2) || (t->type == GGML_TYPE_Q8_0 && t->nb[0] == sizeof(block_q8_0));
}

// one K/V stream per sequence (no -kvu): q, K/V, mask and ids carry the streams in ne[3]. The launchers run once per
// stream on views at s*nb[3] (ggml_cuda_flash_attn_ext), so every stream must keep the alignment of stream 0
static inline bool qsa_streams_supported(const ggml_tensor * q, const ggml_tensor * k, const ggml_tensor * v,
        const ggml_tensor * m, const ggml_tensor * ids) {
    const int64_t ns = q->ne[3];
    if (k->ne[3] != ns || v->ne[3] != ns || ids->ne[3] != ns || (m && m->ne[3] != ns)) { return false; }
    return ns == 1 || (q->nb[3] % 16 == 0 && k->nb[3] % 16 == 0 && v->nb[3] % 16 == 0 && ids->nb[3] % 4 == 0);
}

#if defined(GGML_USE_HIP)
// one Q8_0 value as the f16 the kernels compute with: d*q in f32 as dequantize_q8_0 does, rounded once. Decode and
// the prefill pack both use it, so both paths see the same K/V
static __device__ __forceinline__ uint16_t qsa_q8_0_h(const block_q8_0 * b, const int i) {
    return __builtin_bit_cast(uint16_t, (_Float16) ((float) b->qs[i] * __half2float(b->d)));
}
#endif
