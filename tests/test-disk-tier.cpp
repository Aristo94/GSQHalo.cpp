// disk tier (ported from StrixLlama, MIT, (c) 2026 Victor Shaw): the prompt cache's store on disk, without a model
//
// Writes runs and checkpoints through the writer thread, extends an entry, opens the store again, and checks what a
// damaged, truncated or older file does to it, eviction, and a job that names a run the store no longer has.
// The directory is DISK_TIER_TEST_DIR (default: a fresh one under the temp directory); run it on the filesystem the
// server will use, with STRIX_DISK_DIRECT=0 and without, to cover both ways the bytes reach the disk.

#include "server-task.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#endif

namespace fs = std::filesystem;
using spc = server_prompt_cache;

static int n_fail = 0;

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++n_fail; } } while (0)

static constexpr uint64_t ROW = 512;     // bytes per position
static constexpr int32_t  RUN = 256;     // positions per run

static llama_tokens make_tokens(size_t n, uint32_t seed) {
    llama_tokens t(n);
    std::mt19937 rng(seed);
    for (auto & x : t) {
        x = (llama_token) (rng() % 30000);
    }
    return t;
}

static std::vector<uint8_t> make_rows(int32_t n, uint32_t seed) {
    std::vector<uint8_t> v((size_t) n * ROW);
    std::mt19937 rng(seed);
    for (auto & b : v) {
        b = (uint8_t) rng();
    }
    return v;
}

static common_prompt_checkpoint make_ckpt(int64_t n_tokens, uint32_t seed) {
    common_prompt_checkpoint c;
    c.n_tokens = n_tokens;
    c.pos_min  = (llama_pos) n_tokens - 1;
    c.pos_max  = (llama_pos) n_tokens - 1;
    c.data_tgt = make_rows(64, seed);          // stands in for a recurrent state
    return c;
}

static void open_store(spc & c, const fs::path & dir, size_t limit_mib = 64) {
    c.disk_run = RUN;
    c.set_disk(dir.string(), limit_mib, false, "model", ROW, 0);
}

// a conversation of n tokens written through: runs [0, n) (those below `have` named only), its end checkpoint
static bool persist(spc & c, const llama_tokens & text, int64_t n, int64_t have, std::vector<spc::disk_run_ref> & runs, uint32_t seed) {
    server_tokens tokens(llama_tokens(text.begin(), text.begin() + n), false);
    std::vector<std::pair<std::pair<uint64_t, uint64_t>, std::vector<uint8_t>>> data;
    for (int64_t p = have; p < n; p += RUN) {
        const int32_t len = (int32_t) std::min<int64_t>(RUN, n - p);
        std::vector<uint8_t> rows = make_rows(len, seed + (uint32_t) p);
        const spc::disk_run_ref r = spc::make_run_ref(rows, (int32_t) p, len);
        runs.push_back(r);
        data.push_back({ { r.hi, r.lo }, std::move(rows) });
    }
    const common_prompt_checkpoint end = make_ckpt(n, seed);
    const bool ok = c.persist_runs(0, tokens, {}, {}, nullptr, &end, ROW, 0, runs, {}, std::move(data), true);
    return ok && c.disk_drain(30000);
}

static size_t count_files(const fs::path & dir, const std::string & ext) {
    size_t n = 0;
    std::error_code ec;
    for (auto it = fs::recursive_directory_iterator(dir, ec); !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
        n += it->path().extension() == ext;
    }
    return n;
}

int main() {
    fs::path root;
    if (const char * d = getenv("DISK_TIER_TEST_DIR"); d && *d) {
        root = d;
    } else {
#ifdef _WIN32
        root = fs::temp_directory_path() / "test-disk-tier";
#else
        root = fs::temp_directory_path() / ("test-disk-tier-" + std::to_string(getpid()));
#endif
    }
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root, ec);
    const fs::path dir = root / "model";
    fprintf(stderr, "test-disk-tier: store in %s, STRIX_DISK_DIRECT=%s\n", dir.string().c_str(),
            getenv("STRIX_DISK_DIRECT") ? getenv("STRIX_DISK_DIRECT") : "(default)");

    const llama_tokens text = make_tokens(4 * RUN + 100, 1);
    std::vector<spc::disk_run_ref> runs;

    // 1. one conversation, two and a half runs, then extended to four and a half: one entry, the runs only once
    {
        spc c(0, 0);
        open_store(c, root);
        const int64_t n1 = 2 * RUN + 100;
        CHECK(persist(c, text, n1, 0, runs, 7));
        server_tokens t1(llama_tokens(text.begin(), text.begin() + n1), false);
        CHECK(c.disk_covered(t1) == (size_t) n1);
        CHECK(c.disk_index.size() == 1);
        CHECK(count_files(dir, ".run") == 3);

        // the short run stays as it is; the next one starts where it ends
        const int64_t n2 = 4 * RUN + 100;
        CHECK(persist(c, text, n2, n1, runs, 7));
        server_tokens t2(llama_tokens(text.begin(), text.begin() + n2), false);
        CHECK(c.disk_covered(t2) == (size_t) n2);
        CHECK(c.disk_index.size() == 1);            // the shorter entry went
        CHECK(count_files(dir, ".run") == 5);       // 256, 256, 100, then 256, 256
        CHECK(count_files(dir, ".ckp") == 1);       // its end checkpoint went with it
        CHECK(count_files(dir, ".part") == 0);
    }

    // 2. opened again: the same entry, nothing swept
    {
        spc c(0, 0);
        open_store(c, root);
        CHECK(c.disk_index.size() == 1);
        if (!c.disk_index.empty()) {
            CHECK(c.disk_index[0].n_exact == 4 * RUN + 100);
            CHECK(c.disk_index[0].runs_tgt.size() == 5);
        }
        CHECK(count_files(dir, ".run") == 5);
    }

    // 3. rows of another size are of no use to this server: the entry and its objects go
    {
        spc c(0, 0);
        c.disk_run = RUN;
        c.set_disk(root.string(), 64, false, "model", ROW * 2, 0);
        CHECK(c.disk_index.empty());
        CHECK(count_files(dir, ".run") == 0);
        CHECK(count_files(dir, ".ckp") == 0);
    }

    // 4. a truncated run: the entry that names it cannot be restored and goes at startup
    {
        runs.clear();
        {
            spc c(0, 0);
            open_store(c, root);
            CHECK(persist(c, text, 2 * RUN, 0, runs, 11));
        }
        fs::path victim;
        for (auto it = fs::recursive_directory_iterator(dir / "runs", ec); !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
            if (it->path().extension() == ".run") {
                victim = it->path();
                break;
            }
        }
        CHECK(!victim.empty());
        fs::resize_file(victim, fs::file_size(victim) - 1, ec);
        spc c(0, 0);
        open_store(c, root);
        CHECK(c.disk_index.empty());
        CHECK(count_files(dir, ".run") == 0);
    }

    // 5. a manifest of an older format goes, with nothing else; one of a newer format is left alone
    {
        {
            std::ofstream f(dir / "0000000000000000-1.spc", std::ios::binary);
            const uint32_t v = 2, m = 0;
            f.write("STRIXSPC", 8);
            f.write((const char *) &v, 4);
            f.write((const char *) &m, 4);
        }
        {
            std::ofstream f(dir / "1111111111111111-1.spc", std::ios::binary);
            const uint32_t v = 9, m = 0;
            f.write("STRIXSPC", 8);
            f.write((const char *) &v, 4);
            f.write((const char *) &m, 4);
        }
        spc c(0, 0);
        open_store(c, root);
        CHECK(!fs::exists(dir / "0000000000000000-1.spc"));
        CHECK(fs::exists(dir / "1111111111111111-1.spc"));
        fs::remove(dir / "1111111111111111-1.spc", ec);
    }

    // 6. a job naming a run the store does not have (an eviction took it): refused, and the slot told to start over
    {
        spc c(0, 0);
        open_store(c, root);
        server_tokens t(llama_tokens(text.begin(), text.begin() + RUN), false);
        const spc::disk_run_ref ghost = spc::make_run_ref(make_rows(RUN, 99), 0, RUN);
        const common_prompt_checkpoint end = make_ckpt(RUN, 99);
        CHECK(!c.persist_runs(3, t, {}, {}, nullptr, &end, ROW, 0, { ghost }, {}, {}, true));
        CHECK(c.runs_lost(3));
        CHECK(!c.runs_lost(3));                     // and cleared
    }

    // 7. eviction: three conversations of ~0.4 MiB each under a 1 MiB limit leave the latest two
    {
        fs::remove_all(dir, ec);
        spc c(0, 0);
        open_store(c, root, 1);
        for (uint32_t i = 0; i < 3; ++i) {
            std::vector<spc::disk_run_ref> r;
            CHECK(persist(c, make_tokens(3 * RUN, 100 + i), 3 * RUN, 0, r, 100 + i));
        }
        CHECK(c.disk_index.size() == 2);
        CHECK(c.disk_size() <= 1024ull * 1024ull);
        server_tokens first(make_tokens(3 * RUN, 100), false);
        CHECK(c.disk_covered(first) == 0);
    }

    // 8. checkpoints paged out and back in: the bytes come back as they were written
    {
        fs::remove_all(dir, ec);
        spc c(0, 0);
        open_store(c, root);
        runs.clear();
        CHECK(persist(c, text, 2 * RUN, 0, runs, 21));
        server_tokens t(llama_tokens(text.begin(), text.begin() + 2 * RUN), false);
        std::list<common_prompt_checkpoint> ckpts = { make_ckpt(2 * RUN, 21) };
        const std::vector<uint8_t> orig = ckpts.front().data_tgt;
        spc::ckpt_paged_map paged;
        CHECK(c.ckpt_page_out(t, ckpts, paged) > 0);
        CHECK(ckpts.front().data_tgt.empty());
        CHECK(c.ckpt_page_in(t, paged, ckpts.front()));
        CHECK(ckpts.front().data_tgt == orig);
        c.ckpt_release(paged);
    }

    fs::remove_all(root, ec);
    fprintf(stderr, "test-disk-tier: %s (%d failed)\n", n_fail == 0 ? "PASS" : "FAIL", n_fail);
    return n_fail == 0 ? 0 : 1;
}
