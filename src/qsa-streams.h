#pragma once

#include "llama-batch.h"
#include "qsa-prefix-state.h"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

// Incremental QSA bookkeeping (ratio 4 blocks), one tracked prefix per KV stream.
// A cache with several streams keeps sequence s in stream s only (llama_kv_cache::seq_to_stream), so each
// stream can hold the prefix of its sequence. A unified cache has one stream and tracks one sequence.
// A ubatch over several streams has n_seqs_unq equal slices (split_equal): slice k is the tokens
// [k*n, (k+1)*n) of one sequence, so of one stream.
struct qsa_streams {
    std::vector<qsa_prefix_state> prefix;   // per stream
    std::vector<uint8_t>          pending;  // per stream: rebuild the lost prefix from the cells (recover in apply)
    mutable std::vector<int64_t>  ready;    // [stream*n_layer + il]: complete blocks whose pooled keys the cache holds
    int64_t n_layer = 0;

    void init(uint32_t n_stream, size_t capacity, int64_t layers) {
        prefix.assign(n_stream, qsa_prefix_state(capacity));
        pending.assign(n_stream, 0);
        ready.assign(size_t(n_stream)*layers, 0);
        n_layer = layers;
    }

    uint32_t n_stream() const { return (uint32_t) prefix.size(); }

    // the streams a sequence operation touches: all of them for seq_id < 0
    template <typename F> void for_seq(llama_seq_id seq_id, F && f) {
        for (uint32_t st = 0; st < n_stream(); ++st) {
            if (seq_id < 0 || n_stream() == 1 || uint32_t(seq_id) == st) { f(st); }
        }
    }

    void reset() {
        for (auto & p : prefix) { p.reset(); }
        std::fill(pending.begin(), pending.end(), 0);
        std::fill(ready.begin(), ready.end(), 0);
    }

    // the prefix is lost: the next ubatch of the stream rebuilds it from the cells
    void invalidate(uint32_t st) {
        prefix[st].invalidate(); pending[st] = 1;
        std::fill(ready.begin() + st*n_layer, ready.begin() + (st+1)*n_layer, 0);
    }

    void invalidate_seq(llama_seq_id seq_id) { for_seq(seq_id, [&](uint32_t st) { invalidate(st); }); }

    // the ubatch does not fit the tracking: the stream stays lost until a sequence operation marks it again
    void reject(uint32_t st) { invalidate(st); pending[st] = 0; }

    void seq_rm(llama_seq_id seq_id, int32_t p0, int32_t p1) {
        for_seq(seq_id, [&](uint32_t st) {
            auto & p = prefix[st];
            if (!p.valid) { pending[st] = 1; return; }
            if (seq_id >= 0 && seq_id != p.sequence) { return; }
            if (p1 < 0 || size_t(p1) >= p.cells.size()) {
                p.truncate(std::max(0, p0));
                for (int64_t il = 0; il < n_layer; ++il) {
                    auto & r = ready[st*n_layer + il]; r = std::min<int64_t>(r, p.cells.size()/4);
                }
            } else {
                invalidate(st);
            }
        });
    }

    // the other streams lose their sequences (llama_kv_cache::seq_keep)
    void seq_keep(llama_seq_id seq_id) {
        for (uint32_t st = 0; st < n_stream(); ++st) {
            const auto & p = prefix[st];
            const bool kept = n_stream() == 1 ? p.valid && (p.sequence < 0 || p.sequence == seq_id) : st == uint32_t(seq_id) && p.valid;
            if (!kept) { invalidate(st); }
        }
    }

    // slices of the ubatch, 0 when it has none
    uint32_t slices(const llama_ubatch & u) const {
        if (!u.token || !u.pos || !u.n_tokens || !u.n_pos || !u.seq_id || !u.n_seq_id) { return 0; }
        const uint32_t ns = n_stream() == 1 ? 1 : u.n_seqs_unq;
        return ns > 0 && u.n_tokens % ns == 0 ? ns : 0;
    }

    // stream of slice k with n tokens, -1 when its first token has no single sequence of this cache
    int32_t stream(const llama_ubatch & u, uint32_t k, uint32_t n) const {
        if (n_stream() == 1) { return 0; }
        const uint32_t i = k*n;
        if (u.n_seq_id[i] != 1 || !u.seq_id[i]) { return -1; }
        const llama_seq_id seq = u.seq_id[i][0];
        return seq >= 0 && uint32_t(seq) < n_stream() ? seq : -1;
    }

    // tokens [first, first+n) are one sequence at consecutive positions, all position axes equal
    static bool slice_linear(const llama_ubatch & u, uint32_t first, uint32_t n) {
        for (uint32_t i = first; i < first + n; ++i) {
            if (u.n_seq_id[i] != 1 || !u.seq_id[i] || u.seq_id[i][0] != u.seq_id[first][0] || u.pos[i] != u.pos[first] + int32_t(i - first)) {
                return false;
            }
            for (uint32_t a = 1; a < u.n_pos; ++a) {
                if (u.pos[i + a*u.n_tokens] != u.pos[i]) { return false; }
            }
        }
        return true;
    }

    // every slice is the last ubatch applied to its stream, so the prefix describes the cells it reads
    bool matches(const llama_ubatch & u) const {
        const uint32_t ns = slices(u);
        if (ns == 0) { return false; }
        const uint32_t n = u.n_tokens/ns;
        for (uint32_t k = 0; k < ns; ++k) {
            const int32_t st = stream(u, k, n);
            if (st < 0) { return false; }
            const auto & p = prefix[st];
            if (!p.valid || u.pos[k*n] != p.begin || int64_t(n) != p.end - p.begin || !u.seq_id[k*n] ||
                u.seq_id[k*n][0] != p.sequence || !slice_linear(u, k*n, n)) {
                return false;
            }
        }
        return true;
    }

    // prefix of the stream of a single-slice ubatch that matches
    const qsa_prefix_state & state(const llama_ubatch & u) const { return prefix[stream(u, 0, u.n_tokens)]; }

    // re-pool only the (n+3)/4+2 blocks each slice touches: the cache holds every block that was complete
    // before this ubatch, and the stream has enough complete blocks for the fillers
    bool fast(int il, const llama_ubatch & u, int max_tokens) const {
        if (!matches(u)) { return false; }
        const uint32_t ns = slices(u), n = u.n_tokens/ns;
        if (int(n) > max_tokens || n > 512) { return false; }
        for (uint32_t k = 0; k < ns; ++k) {
            const int32_t st = stream(u, k, n);
            const auto & p = prefix[st];
            if (ready[st*n_layer + il] < int64_t(p.previous_size/4) || p.cells.size()/4 < (n+3)/4+1) { return false; }
        }
        return true;
    }

    // per slice: the spare row (the incomplete block), the blocks the ubatch changed, then unchanged fillers.
    // members I32 [4*count, ns], positions I32 [4 sections][ns*count], rows I64 [count, ns]
    void fill_updates(const llama_ubatch & u, int32_t * members, int32_t * positions, int64_t * rows, int count) const {
        const uint32_t ns = slices(u), n = u.n_tokens/ns;
        std::vector<int32_t> ids;
        for (uint32_t k = 0; k < ns; ++k) {
            const auto & p = prefix[stream(u, k, n)];
            const int complete = p.cells.size()/4, first = p.begin/4, last = std::min(complete, (p.end+3)/4);
            GGML_ASSERT(last-first < count && complete >= count - 1);
            ids.assign(1, complete);
            for (int b = first; b < last; ++b) { ids.push_back(b); }
            for (int b = 0; int(ids.size()) < count; ++b) { if (b < first || b >= last) { ids.push_back(b); } }
            for (int i = 0; i < count; ++i) {
                const int64_t row = int64_t(k)*count + i;
                rows[row] = ids[i];
                for (int j = 0; j < 4; ++j) {
                    members[4*row + j] = ids[i] == complete ? 0 : p.cells[ids[i]*4 + j];
                    positions[j*int64_t(ns)*count + row] = ids[i] == complete ? 0 : ids[i]*4;
                }
            }
        }
    }

    // the graph pools the keys of every complete block of each slice's stream into the cache
    void commit(int il, const llama_ubatch & u) const {
        const uint32_t ns = slices(u), n = u.n_tokens/ns;
        for (uint32_t k = 0; k < ns; ++k) {
            const int32_t st = stream(u, k, n);
            GGML_ASSERT(prefix[st].valid);
            ready[st*n_layer + il] = prefix[st].cells.size()/4;
        }
    }

    // the cells of the ubatch are placed: slice k went to stream strm[k], cells idxs[k] (llama_kv_cache::slot_info).
    // recover(st, seq) rebuilds the lost prefix of stream st from the cells, the ubatch included.
    template <typename F>
    void apply(const llama_ubatch & u, const std::vector<llama_seq_id> & strm, const std::vector<std::vector<uint32_t>> & idxs, F && recover) {
        const uint32_t ns = slices(u);
        if (ns == 0 || strm.size() != ns || idxs.size() != ns) {
            for (llama_seq_id st : strm) { if (st >= 0 && uint32_t(st) < n_stream()) { reject(st); } }
            if (strm.empty()) { for (uint32_t st = 0; st < n_stream(); ++st) { reject(st); } }
            return;
        }
        const uint32_t n = u.n_tokens/ns;
        for (uint32_t k = 0; k < ns; ++k) {
            if (strm[k] < 0 || uint32_t(strm[k]) >= n_stream()) { continue; }
            const uint32_t st = strm[k];
            if (idxs[k].size() != n || (n_stream() > 1 && stream(u, k, n) != int32_t(st)) || !slice_linear(u, k*n, n)) {
                reject(st); continue;
            }
            const llama_seq_id seq = u.seq_id[k*n][0];
            auto & p = prefix[st];
            if (!p.valid) {
                const bool rebuilt = pending[st] && recover(st, seq);
                pending[st] = 0;
                if (!rebuilt) { continue; }
            }
            pending[st] = 0;
            if (!p.apply(seq, u.pos[k*n], idxs[k])) { reject(st); }
        }
    }

    // block selection inputs of every slice from the tracked prefixes, the values the cell scan
    // (llama_memory_hybrid_idx::set_input_qsa_scan) gives for a stream with one sequence at positions 0..L-1:
    // complete block b holds the cells of positions 4b..4b+3 at position 4b; a query sees the complete blocks
    // before its own block, whose cells it reads from the tail (-1 padded); all else is -inf.
    // cells I32 [4*n_blocks, ns], pos I32 [4 sections][ns*n_blocks], bias F32 [n_blocks, n, ns], tails I32 [3, n, ns]
    bool block_inputs(const llama_ubatch & u, int64_t n_blocks, int32_t * cells, int32_t * pos, float * bias, int32_t * tails) const {
        if (!cells || !pos || !bias || !tails || !matches(u)) { return false; }
        const uint32_t ns = slices(u), n = u.n_tokens/ns;
        for (uint32_t k = 0; k < ns; ++k) {
            if (int64_t(prefix[stream(u, k, n)].cells.size()/4) > n_blocks) { return false; }
        }
        std::fill(pos, pos + 4*n_blocks*ns, 0);
        for (uint32_t k = 0; k < ns; ++k) {
            const auto & p = prefix[stream(u, k, n)];
            const int64_t complete = p.cells.size()/4;
            int32_t * c = cells + k*4*n_blocks;
            std::copy_n(p.cells.data(), complete*4, c);
            std::fill(c + complete*4, c + 4*n_blocks, 0);
            for (int64_t sec = 0; sec < 4; ++sec) {
                std::copy_n(p.block_positions.data(), complete, pos + sec*n_blocks*ns + k*n_blocks);
            }
            for (uint32_t i = 0; i < n; ++i) {
                const int64_t q = u.pos[k*n + i], tail_start = (q + 1)/4*4;
                const int64_t visible = std::min(complete, tail_start/4);
                float * row = bias + (int64_t(k)*n + i)*n_blocks;
                std::fill(row, row + visible, 0.0f);
                std::fill(row + visible, row + n_blocks, -INFINITY);
                int32_t * t = tails + (int64_t(k)*n + i)*3;
                std::fill(t, t + 3, -1);
                if (tail_start/4 < n_blocks) {
                    for (int64_t j = 0; j < q + 1 - tail_start; ++j) { t[j] = p.cells[tail_start + j]; }
                }
            }
        }
        return true;
    }
};
