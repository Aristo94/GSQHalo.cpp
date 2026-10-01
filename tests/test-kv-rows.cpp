// disk tier (ported from StrixLlama, MIT, (c) 2026 Victor Shaw): llama_strix_kv_* round trip
//
// Decodes a prompt into seq 0, reads its rows by position in runs (as the server's disk tier does) and, for a model
// with a recurrent state, the LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY state at its end; then greedily decodes KVR_GEN
// tokens as the reference. Removes seq 0, allocates its cells again, writes the rows and the partial state back and
// decodes the same tokens: the logits must match the reference (bit for bit when the cells land where they were),
// and the rows read back must be the bytes written.
//
// KVR_MODE=interleaved: a second conversation (seq 1) is decoded alongside, chunk by chunk, so seq 0's cells are
// scattered among seq 1's in a unified cache, and seq 1 stays resident through the restore.
//
// usage: test-kv-rows -m MODEL [common flags, e.g. -c 32768 -np 2 -fa on -ub 4096]; the cache is always unified
// env:   KVR_N (prompt tokens, 8192), KVR_GEN (32), KVR_RUN (positions per run, 4096), KVR_CHUNK (512),
//        KVR_MODE (single|interleaved), KVR_RESTORE (rows|state: the control, a whole state of seq 0 instead of the
//        rows - what any restore gives)

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int env_int(const char * name, int def) {
    const char * v = getenv(name);
    return v ? atoi(v) : def;
}

static double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// deterministic text, different per seed, long enough for n tokens (token ids from a fixed sequence for a model
// without a tokenizer, as the generated test models are)
static std::vector<llama_token> make_prompt(llama_context * ctx, int n, int seed) {
    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
    if (llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_NONE) {
        std::vector<llama_token> toks(n);
        uint32_t h = 2166136261u + (uint32_t) seed;
        for (auto & t : toks) {
            h = h * 1664525u + 1013904223u;
            t = (llama_token) (h >> 8) % llama_vocab_n_tokens(vocab);
        }
        return toks;
    }
    static const char * colors[]  = { "red", "amber", "teal", "violet", "grey", "golden", "pale", "dark" };
    static const char * animals[] = { "heron", "otter", "lynx", "badger", "falcon", "marten", "ibex", "crane" };
    static const char * places[]  = { "mill", "harbour", "quarry", "orchard", "bridge", "chapel", "market", "ridge" };
    std::string text;
    std::vector<llama_token> toks;
    for (int i = 0; (int) toks.size() < n; ++i) {
        for (int k = 0; k < 64; ++k, ++i) {
            const int h = (i + 1) * 2654435761u % 1000003 + seed * 7919;
            text += "Entry " + std::to_string(i) + ": the " + colors[h % 8] + " " + animals[(h / 8) % 8] + " counted " +
                    std::to_string(h % 10007) + " stones near the " + places[(h / 64) % 8] + ".\n";
        }
        toks = common_tokenize(ctx, text, true);
    }
    toks.resize(n);
    return toks;
}

static bool decode_tokens(llama_context * ctx, llama_batch & batch, const std::vector<llama_token> & toks, int p0, int p1,
                          llama_seq_id seq, bool last_logits) {
    common_batch_clear(batch);
    for (int i = p0; i < p1; ++i) {
        common_batch_add(batch, toks[i], i, { seq }, last_logits && i == p1 - 1);
    }
    return llama_decode(ctx, batch) == 0;
}

static int argmax(const float * v, int n) {
    return (int) (std::max_element(v, v + n) - v);
}

int main(int argc, char ** argv) {
    common_params params;
    params.n_ctx      = 16384;
    params.kv_unified = true;                       // rows need one stream (-kvu is not a common flag)

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    const int  N     = env_int("KVR_N", 8192);
    const int  G     = env_int("KVR_GEN", 32);
    const int  RUN   = env_int("KVR_RUN", 4096);
    const int  CHUNK = env_int("KVR_CHUNK", 512);
    const bool inter = getenv("KVR_MODE") && strcmp(getenv("KVR_MODE"), "interleaved") == 0;
    const bool whole = getenv("KVR_RESTORE") && strcmp(getenv("KVR_RESTORE"), "state") == 0;

    llama_backend_init();

    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model   * model = llama_init->model();
    llama_context * ctx   = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        fprintf(stderr, "%s: failed to init\n", __func__);
        return 1;
    }

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);
    const size_t        row     = llama_strix_kv_row_size(ctx);
    const bool          partial = llama_model_is_recurrent(model) || llama_model_is_hybrid(model);
    llama_memory_t      mem     = llama_get_memory(ctx);

    fprintf(stderr, "kv-rows: mode %s, restore %s, N %d, gen %d, run %d, chunk %d, row size %zu bytes, recurrent state %s\n",
            inter ? "interleaved" : "single", whole ? "state" : "rows", N, G, RUN, CHUNK, row, partial ? "yes" : "no");
    if (row == 0) {
        fprintf(stderr, "kv-rows: FAIL - this context serves no rows (V transposed? -fa on)\n");
        return 1;
    }

    const std::vector<llama_token> P = make_prompt(ctx, N, 0);
    const std::vector<llama_token> Q = inter ? make_prompt(ctx, N, 1) : std::vector<llama_token>();

    llama_batch batch = llama_batch_init(std::max(CHUNK, (int) params.n_batch), 0, 1);

    // prompt(s): seq 0, and seq 1 chunk by chunk alongside it
    auto t0 = std::chrono::steady_clock::now();
    llama_token first = LLAMA_TOKEN_NULL;
    for (int p = 0; p < N; p += CHUNK) {
        const int e = std::min(N, p + CHUNK);
        if (!decode_tokens(ctx, batch, P, p, e, 0, e == N)) {
            fprintf(stderr, "kv-rows: FAIL - decode of seq 0 [%d, %d)\n", p, e);
            return 1;
        }
        if (e == N) {
            first = argmax(llama_get_logits_ith(ctx, -1), n_vocab);
        }
        if (inter && !decode_tokens(ctx, batch, Q, p, e, 1, false)) {
            fprintf(stderr, "kv-rows: FAIL - decode of seq 1 [%d, %d)\n", p, e);
            return 1;
        }
    }
    fprintf(stderr, "kv-rows: prompt decoded in %.0f ms\n", ms_since(t0));

    // rows by run, then the recurrent state at N
    std::vector<std::vector<uint8_t>> runs;
    double t_get = 0;
    for (int p = 0; p < N; p += RUN) {
        const int n = std::min(RUN, N - p);
        runs.emplace_back((size_t) n * row);
        t0 = std::chrono::steady_clock::now();
        if (!llama_strix_kv_get_rows(ctx, 0, p, n, runs.back().data(), runs.back().size())) {
            fprintf(stderr, "kv-rows: FAIL - get_rows [%d, %d)\n", p, p + n);
            return 1;
        }
        t_get += ms_since(t0);
    }
    std::vector<uint8_t> state;
    if (partial) {
        state.resize(llama_state_seq_get_size_ext(ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
        if (llama_state_seq_get_data_ext(ctx, state.data(), state.size(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != state.size()) {
            fprintf(stderr, "kv-rows: FAIL - partial state get\n");
            return 1;
        }
    }
    std::vector<uint8_t> full;
    if (whole) {
        full.resize(llama_state_seq_get_size_ext(ctx, 0, 0));
        if (llama_state_seq_get_data_ext(ctx, full.data(), full.size(), 0, 0) != full.size()) {
            fprintf(stderr, "kv-rows: FAIL - state get\n");
            return 1;
        }
    }
    fprintf(stderr, "kv-rows: get_rows %.1f ms total (%.1f ms per %d positions), %.1f MiB rows, partial state %.1f MiB\n",
            t_get, t_get * RUN / N, RUN, (double) N * row / (1 << 20), (double) state.size() / (1 << 20));

    // reference: greedy from the prompt's last logits
    std::vector<llama_token> gen = { first };
    std::vector<std::vector<float>> ref;
    for (int g = 0; g < G; ++g) {
        common_batch_clear(batch);
        common_batch_add(batch, gen[g], N + g, { 0 }, true);
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "kv-rows: FAIL - reference decode %d\n", g);
            return 1;
        }
        const float * l = llama_get_logits_ith(ctx, -1);
        ref.emplace_back(l, l + n_vocab);
        gen.push_back(argmax(l, n_vocab));
    }

    // remove, allocate, write back
    llama_memory_seq_rm(mem, 0, -1, -1);
    if (whole) {
        if (llama_state_seq_set_data_ext(ctx, full.data(), full.size(), 0, 0) != full.size()) {
            fprintf(stderr, "kv-rows: FAIL - state set\n");
            return 1;
        }
        runs.clear();                               // nothing of the rows path to check
    }
    t0 = std::chrono::steady_clock::now();
    if (!whole && !llama_strix_kv_alloc(ctx, 0, P.data(), N)) {
        fprintf(stderr, "kv-rows: FAIL - alloc\n");
        return 1;
    }
    const double t_alloc = ms_since(t0);
    double t_set = 0;
    for (size_t k = 0; k < runs.size(); ++k) {
        const int p = (int) k * RUN;
        const int n = std::min(RUN, N - p);
        t0 = std::chrono::steady_clock::now();
        if (!llama_strix_kv_set_rows(ctx, 0, p, n, runs[k].data(), n)) {
            fprintf(stderr, "kv-rows: FAIL - set_rows [%d, %d)\n", p, p + n);
            return 1;
        }
        t_set += ms_since(t0);
    }
    if (!whole && partial && llama_state_seq_set_data_ext(ctx, state.data(), state.size(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != state.size()) {
        fprintf(stderr, "kv-rows: FAIL - partial state set\n");
        return 1;
    }
    fprintf(stderr, "kv-rows: alloc %.1f ms, set_rows %.1f ms total; seq 0 now [%d, %d]\n", t_alloc, t_set,
            llama_memory_seq_pos_min(mem, 0), llama_memory_seq_pos_max(mem, 0));

    // the rows read back are the bytes written
    int n_row_diff = 0;
    for (size_t k = 0; k < runs.size(); ++k) {
        const int p = (int) k * RUN;
        const int n = std::min(RUN, N - p);
        std::vector<uint8_t> back((size_t) n * row);
        if (!llama_strix_kv_get_rows(ctx, 0, p, n, back.data(), back.size()) || back != runs[k]) {
            ++n_row_diff;
        }
    }

    // replay the reference tokens
    int n_bit = 0, n_top = 0;
    double max_abs = 0, sum_sq = 0, sum_ref = 0;
    for (int g = 0; g < G; ++g) {
        common_batch_clear(batch);
        common_batch_add(batch, gen[g], N + g, { 0 }, true);
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "kv-rows: FAIL - replay decode %d\n", g);
            return 1;
        }
        const float * l = llama_get_logits_ith(ctx, -1);
        n_bit += memcmp(l, ref[g].data(), sizeof(float) * n_vocab) == 0;
        n_top += argmax(l, n_vocab) == argmax(ref[g].data(), n_vocab);
        for (int i = 0; i < n_vocab; ++i) {
            const double d = (double) l[i] - ref[g][i];
            max_abs = std::max(max_abs, std::fabs(d));
            sum_sq += d * d;
            sum_ref += (double) ref[g][i] * ref[g][i];
        }
    }
    const double nmse = sum_ref > 0 ? sum_sq / sum_ref : 0;

    fprintf(stderr, "kv-rows: rows read back differ in %d of %zu runs\n", n_row_diff, runs.size());
    fprintf(stderr, "kv-rows: replay of %d tokens: %d bit-identical, %d same top token, max |diff| %.3g, NMSE %.3g\n",
            G, n_bit, n_top, max_abs, nmse);

    const bool ok = n_row_diff == 0 && n_top == G && nmse < 1e-6;
    fprintf(stderr, "kv-rows: %s\n", ok ? (n_bit == G ? "PASS (bit-identical)" : "PASS (within tolerance)") : "FAIL");

    llama_batch_free(batch);
    return ok ? 0 : 1;
}
