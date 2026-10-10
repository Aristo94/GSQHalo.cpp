#include "../src/qsa-streams.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

// Host-side checks of the incremental QSA bookkeeping per KV stream (qsa_streams). No model or backend.

// equal slices as split_equal builds them: slice k holds n tokens of sequence seqs[k] from position starts[k]
struct qsa_slices {
    std::vector<llama_token> tok; std::vector<llama_pos> pos; std::vector<int32_t> n_seq_id;
    std::vector<llama_seq_id> ids, unq; std::vector<llama_seq_id *> seq_id; llama_ubatch u{};
    qsa_slices(std::vector<llama_seq_id> seqs, std::vector<llama_pos> starts, uint32_t n) : unq(seqs) {
        const uint32_t total = seqs.size()*n;
        tok.assign(total, 0); pos.resize(total); ids.resize(total);
        for (size_t k=0; k<seqs.size(); ++k) {
            for (uint32_t i=0; i<n; ++i) { pos[k*n+i] = starts[k]+i; ids[k*n+i] = seqs[k]; }
        }
        for (uint32_t i=0; i<total; ++i) { n_seq_id.push_back(1); seq_id.push_back(&ids[i]); }
        u.n_tokens = total; u.n_seq_tokens = n; u.n_seqs = seqs.size(); u.n_seqs_unq = seqs.size(); u.n_pos = 1;
        u.token = tok.data(); u.pos = pos.data(); u.n_seq_id = n_seq_id.data(); u.seq_id = seq_id.data(); u.seq_id_unq = unq.data();
    }
    qsa_slices(const qsa_slices &) = delete;
};

// slice k goes to stream strm[k], which keeps position p in cell p + offset[strm[k]]
static void place(qsa_streams & q, const qsa_slices & b, std::vector<llama_seq_id> strm, std::vector<uint32_t> offset = {},
        std::vector<std::pair<uint32_t, llama_seq_id>> * recovered = nullptr) {
    const auto shift = [&](uint32_t st) { return offset.empty() ? 0u : offset[st]; };
    std::vector<std::vector<uint32_t>> idxs(strm.size());
    const uint32_t n = b.u.n_tokens/strm.size();
    for (size_t k=0; k<strm.size(); ++k) {
        for (uint32_t i=0; i<n; ++i) { idxs[k].push_back(b.u.pos[k*n+i] + shift(strm[k])); }
    }
    q.apply(b.u, strm, idxs, [&](uint32_t st, llama_seq_id seq) {
        if (!recovered) { return false; }
        recovered->push_back({st, seq});
        // the cells hold positions 0..end-1 of seq in order, the ubatch included
        int32_t end = 0;
        for (uint32_t i=0; i<b.u.n_tokens; ++i) { if (b.ids[i] == seq) { end = std::max(end, b.u.pos[i] + 1); } }
        std::vector<uint32_t> all(end);
        for (int32_t p=0; p<end; ++p) { all[p] = p + shift(st); }
        q.prefix[st].reset();
        return q.prefix[st].apply(seq, 0, all);
    });
}

// a sequence in its own stream keeps its prefix while other streams take ubatches, and a ubatch over
// several streams takes the fast path with each stream's blocks
static void test_streams_are_independent() {
    qsa_streams q; q.init(2, 4096, 1);
    qsa_slices a({0}, {0}, 510); place(q, a, {0});
    GGML_ASSERT(q.matches(a.u) && !q.fast(0, a.u, 512)); q.commit(0, a.u);
    qsa_slices b({1}, {0}, 300); place(q, b, {1}, {0, 7});
    GGML_ASSERT(q.matches(b.u)); q.commit(0, b.u);
    GGML_ASSERT(q.prefix[0].valid && q.prefix[0].cells.size() == 510);

    // two tokens each: stream 0 completes block 127, stream 1 starts block 75
    qsa_slices d({0, 1}, {510, 300}, 2); place(q, d, {0, 1}, {0, 7});
    GGML_ASSERT(q.matches(d.u) && q.fast(0, d.u, 512));
    GGML_ASSERT(!q.fast(0, d.u, 1));

    const int count = (2+3)/4+2;
    std::vector<int32_t> c(4*count*2, -7), p(4*count*2, -7); std::vector<int64_t> w(count*2, -7);
    q.fill_updates(d.u, c.data(), p.data(), w.data(), count);
    // the spare row first (the incomplete block), then the changed blocks, then unchanged fillers
    const std::vector<int64_t> rows = { 128, 127, 0,   75, 0, 1 };
    GGML_ASSERT(w == rows);
    for (int k=0; k<2; ++k) {
        for (int i=0; i<count; ++i) {
            const int64_t id = rows[k*count+i];
            const bool spare = id == (k == 0 ? 128 : 75);
            for (int j=0; j<4; ++j) {
                GGML_ASSERT(c[4*(k*count+i)+j] == (spare ? 0 : int32_t(4*id+j + (k == 0 ? 0 : 7))));
                GGML_ASSERT(p[j*(2*count) + k*count + i] == (spare ? 0 : int32_t(4*id)));
            }
        }
    }

    q.commit(0, d.u);
    GGML_ASSERT(q.ready[0] == 128 && q.ready[1] == 75);
    // the next token of stream 1 alone still finds stream 0 untouched
    qsa_slices e({1}, {302}, 1); place(q, e, {1}, {0, 7});
    GGML_ASSERT(q.matches(e.u) && q.fast(0, e.u, 512) && q.prefix[0].cells.size() == 512);
}

// a unified cache has one stream for all sequences: a second sequence ends the tracked prefix
static void test_unified_tracks_one_sequence() {
    qsa_streams q; q.init(1, 4096, 1);
    qsa_slices a({0}, {0}, 64); place(q, a, {0});
    GGML_ASSERT(q.matches(a.u));
    qsa_slices b({1}, {0}, 64); place(q, b, {0}, {64});
    GGML_ASSERT(!q.prefix[0].valid && !q.matches(b.u));
}

static void test_seq_rm_per_stream() {
    qsa_streams q; q.init(2, 4096, 1);
    qsa_slices a({0}, {0}, 100); place(q, a, {0}); q.commit(0, a.u);
    qsa_slices b({1}, {0}, 100); place(q, b, {1}); q.commit(0, b.u);

    q.seq_rm(1, 60, -1);   // a truncation keeps the prefix
    GGML_ASSERT(q.prefix[1].valid && q.prefix[1].cells.size() == 60 && q.ready[1] == 15);
    GGML_ASSERT(q.prefix[0].cells.size() == 100 && q.ready[0] == 25);

    q.seq_rm(0, 10, 20);   // a hole loses the prefix of that stream only
    GGML_ASSERT(!q.prefix[0].valid && q.pending[0] && q.prefix[1].valid);

    q.seq_rm(-1, 0, -1);   // all sequences
    GGML_ASSERT(q.prefix[1].valid && q.prefix[1].cells.empty() && q.ready[1] == 0);
}

// a ubatch over a tracked and a lost stream: no fast path, the tracked stream still advances, and only
// the lost stream is rebuilt once it is marked for recovery
static void test_lost_stream() {
    qsa_streams q; q.init(2, 4096, 1);
    qsa_slices a({0}, {0}, 40); place(q, a, {0}); q.commit(0, a.u);
    qsa_slices b({1}, {0}, 40); place(q, b, {1}); q.commit(0, b.u);
    q.invalidate(1); q.pending[1] = 0;

    std::vector<std::pair<uint32_t, llama_seq_id>> recovered;
    qsa_slices d({0, 1}, {40, 40}, 1); place(q, d, {0, 1}, {}, &recovered);
    GGML_ASSERT(!q.matches(d.u) && !q.fast(0, d.u, 512));
    GGML_ASSERT(recovered.empty() && q.prefix[0].cells.size() == 41 && !q.prefix[1].valid);

    q.invalidate(1);
    qsa_slices e({0, 1}, {41, 41}, 1); place(q, e, {0, 1}, {}, &recovered);
    GGML_ASSERT(recovered.size() == 1 && recovered[0].first == 1 && recovered[0].second == 1);
    GGML_ASSERT(q.matches(e.u) && q.prefix[1].cells.size() == 42);
    // the rebuilt stream has no pooled keys yet: one full pass first
    GGML_ASSERT(!q.fast(0, e.u, 512));
}

// the block selection inputs of a ubatch over two tracked streams (layouts of llm_graph_input_qsa)
static void test_block_inputs() {
    qsa_streams q; q.init(2, 4096, 1);
    qsa_slices a({0}, {0}, 13); place(q, a, {0}, {0, 0});
    qsa_slices b({1}, {0}, 21); place(q, b, {1}, {0, 5});
    // three tokens each, as with a draft: stream 0 at 13..15, stream 1 at 21..23
    qsa_slices d({0, 1}, {13, 21}, 3); place(q, d, {0, 1}, {0, 5});

    const int64_t nb = 8, n = 3, ns = 2;
    std::vector<int32_t> cells(4*nb*ns, -7), pos(4*nb*ns, -7), tails(3*n*ns, -7);
    std::vector<float> bias(nb*n*ns, 7.0f);
    GGML_ASSERT(q.block_inputs(d.u, nb, cells.data(), pos.data(), bias.data(), tails.data()));
    GGML_ASSERT(!q.block_inputs(d.u, 5, cells.data(), pos.data(), bias.data(), tails.data()));

    const int64_t complete[2] = { 4, 6 };
    const int32_t offset[2] = { 0, 5 };
    for (int64_t k=0; k<ns; ++k) {
        for (int64_t blk=0; blk<nb; ++blk) {
            for (int64_t j=0; j<4; ++j) {
                GGML_ASSERT(cells[k*4*nb + 4*blk + j] == (blk < complete[k] ? int32_t(4*blk + j) + offset[k] : 0));
                GGML_ASSERT(pos[j*nb*ns + k*nb + blk] == (blk < complete[k] ? int32_t(4*blk) : 0));
            }
        }
        for (int64_t i=0; i<n; ++i) {
            const int64_t qp = d.u.pos[k*n+i], start = (qp+1)/4*4;
            for (int64_t blk=0; blk<nb; ++blk) {
                const float want = blk < complete[k] && 4*blk < start ? 0.0f : -INFINITY;
                GGML_ASSERT(bias[(k*n+i)*nb + blk] == want);
            }
            for (int64_t j=0; j<3; ++j) {
                GGML_ASSERT(tails[(k*n+i)*3 + j] == (j < qp+1-start ? int32_t(start+j) + offset[k] : -1));
            }
        }
    }
}

int main() {
    test_streams_are_independent();
    test_unified_tracks_one_sequence();
    test_seq_rm_per_stream();
    test_lost_stream();
    test_block_inputs();
    return 0;
}
