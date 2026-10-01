// disk tier (ported from StrixLlama, MIT, Portions Copyright (c) 2026 Victor Shaw): a store for the prompt
// cache on disk, version 3 only. Declared in server-task.h (server_prompt_cache::disk_*).
//
// A conversation's attention rows (KV cache and indexer) are kept by position (llama_strix_kv_*), in runs of up
// to disk_run positions that are written once, when a run fills up or the conversation leaves its slot, and never
// again - the next run starts where the last one ends. The recurrent state is a checkpoint, and the part that
// changes: it is written only as the conversation leaves its slot (the state at the end and its last prompt's
// latest checkpoints), so a store entry for a conversation still in a slot restores only as far as it did the
// last time it left one. A restore takes the latest checkpoint the runs reach: cells for its positions, the rows
// a run at a time, then the recurrent state; the rest of the prompt is processed as any other.
//
//   <xxh64>-<n>.spc      manifest: tokens, the checkpoints it uses, the runs of the target's and the draft's rows
//   ckpt/<key>.ckp       one context checkpoint. Immutable once made, and keyed by the token prefix it covers, so
//                        every later save of the same conversation reuses it
//   runs/xx/<h>.run      rows of up to disk_run positions, named by their XXH3-128
//
// One background thread writes everything and is the only thing that deletes: the main loop reads the rows out
// of the KV cache and hands them over. Files appear as .part and are renamed once complete, and the manifest goes
// last, so a process killed mid-write leaves nothing a later start would trust - stray .part files and objects no
// manifest references are swept at startup. A store holds this format and nothing older: an entry of another
// version is deleted at startup (one newer than this build knows is left alone).
//
// manifest, version 3:
//   "STRIXSPC" u32 version u32 has_mtmd
//   u64 n_token_bytes, bytes                          (server_tokens::serialize)
//   u32 n_ckpt, per checkpoint:
//       u64 key i64 n_tokens i32 pos_min i32 pos_max u64 size_tgt u64 size_dft u64 size_spec
//   u64 row_tgt u32 n_runs, per run: u64 xxh_high u64 xxh_low i32 pos0 i32 n i32 n_use
//   u64 row_dft u32 n_runs, the same
// checkpoint file, version 2:
//   "STRIXCKP" u32 version i64 n_tokens i32 pos_min i32 pos_max, then tgt, dft, spec as u64 n + bytes,
//   then u64 xxh_high u64 xxh_low: XXH3-128 of the three payloads
// run file, version 1:
//   "STRIXRUN" u32 version u64 row_size u32 n, then n rows, then u64 xxh_high u64 xxh_low: XXH3-128 of the rows,
//   which is also the name
//
// The formats are StrixLlama's, byte for byte. What differs here is how the bytes reach the disk (see "files").

#include "server-task.h"
#include "server-common.h"

#include "llama.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <future>
#include <set>
#include <system_error>

#ifdef _WIN32
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#else
#  include <fcntl.h>
#  include <unistd.h>
#endif

// XXH3 names the runs and checkpoints; inlined here, so nothing new to link
#define XXH_INLINE_ALL
#include "vendor/hash/xxhash/xxhash.h"

namespace {

namespace fs = std::filesystem;
using disk_entry   = server_prompt_cache::disk_entry;
using disk_ckpt_ref = server_prompt_cache::disk_ckpt_ref;
using disk_run_ref = server_prompt_cache::disk_run_ref;

const char     SPC_MAGIC[8] = {'S','T','R','I','X','S','P','C'};
const char     CKP_MAGIC[8] = {'S','T','R','I','X','C','K','P'};
const char     RUN_MAGIC[8] = {'S','T','R','I','X','R','U','N'};
const uint32_t SPC_V3 = 3;
const uint32_t CKP_V2 = 2;
const uint32_t RUN_V1 = 1;

std::string hex128(uint64_t hi, uint64_t lo) {
    char b[33];
    snprintf(b, sizeof(b), "%016llx%016llx", (unsigned long long) hi, (unsigned long long) lo);
    return b;
}

std::string run_path(const std::string & dir, uint64_t hi, uint64_t lo) {
    const std::string hex = hex128(hi, lo);
    return (fs::path(dir) / "runs" / hex.substr(0, 2) / (hex + ".run")).string();
}

uint64_t run_file_bytes(uint64_t row, int32_t n) {
    return 8 + sizeof(uint32_t) + sizeof(uint64_t) + sizeof(uint32_t) + row * (uint64_t) n + 2*sizeof(uint64_t);
}

std::string ckpt_path(const std::string & dir, uint64_t key) {
    char b[32];
    snprintf(b, sizeof(b), "%016llx.ckp", (unsigned long long) key);
    return (fs::path(dir) / "ckpt" / b).string();
}

int64_t runs_cover(const std::vector<disk_run_ref> & runs) {
    int64_t n = 0;
    for (const auto & r : runs) {
        n += r.n_use;
    }
    return n;
}

// the tokens an entry brings back exactly: its latest checkpoint that the runs reach
int64_t runs_exact(const server_tokens & tokens, const std::vector<disk_run_ref> & tgt, const std::vector<disk_run_ref> & dft,
                   bool has_dft, const std::vector<disk_ckpt_ref> & ckpts) {
    int64_t cover = std::min<int64_t>(runs_cover(tgt), (int64_t) tokens.size());
    if (has_dft) {
        cover = std::min(cover, runs_cover(dft));
    }
    int64_t best = 0;
    for (const auto & k : ckpts) {
        if (k.n_tokens <= cover) {
            best = std::max(best, k.n_tokens);
        }
    }
    return best;
}

// a checkpoint is the recurrent state after the first n_tokens of a conversation, so the tokens it covers
// name it; the rest only guards against two different things claiming one name
uint64_t ckpt_key(const llama_tokens & text, int64_t n_tokens, int32_t pos_min, int32_t pos_max,
                  uint64_t size_tgt, uint64_t size_dft, uint64_t size_spec) {
    const size_t n = (size_t) std::clamp<int64_t>(n_tokens, 0, (int64_t) text.size());
    const uint64_t seed = XXH3_64bits(text.data(), n * sizeof(llama_token));
    const uint64_t meta[6] = { (uint64_t) n_tokens, (uint64_t) (uint32_t) pos_min, (uint64_t) (uint32_t) pos_max,
                               size_tgt, size_dft, size_spec };
    return XXH3_64bits_withSeed(meta, sizeof(meta), seed);
}

disk_ckpt_ref ckpt_ref_of(const llama_tokens & text, const common_prompt_checkpoint & c) {
    return { ckpt_key(text, c.n_tokens, c.pos_min, c.pos_max, c.data_tgt.size(), c.data_dft.size(), c.data_spec.size()),
             c.n_tokens, c.pos_min, c.pos_max, c.data_tgt.size(), c.data_dft.size(), c.data_spec.size() };
}

// --- files ------------------------------------------------------------------------------------------
// Entries are read once and handed to the GPU, and written once and not read again until a conversation returns,
// so the page cache only costs here, and on a machine whose RAM is also the GPU's memory it costs twice: every
// cached page is one the model cannot have, and dirty pages under the low dirty thresholds of a full machine are
// what stalled halobox PR #81 for minutes (its device-to-file copy went into a MAP_SHARED mapping, and writeback
// re-dirtied the same pages again and again). So nothing here is mapped: files are moved through one aligned
// staging buffer that the caller owns, unbuffered where the system and the filesystem allow it - O_DIRECT on Linux
// (ext4, xfs, btrfs), FILE_FLAG_NO_BUFFERING on Windows. Where that is refused (tmpfs, some FUSE, macOS) they go
// through the page cache, are synced as they close and then dropped from it. STRIX_DISK_DIRECT=0 turns it off.
#if defined(O_DIRECT) && !defined(_WIN32)
constexpr int SPC_O_DIRECT = O_DIRECT;
#else
constexpr int SPC_O_DIRECT = 0;
#endif

bool disk_direct() {
    static const bool on = [] {
        const char * v = getenv("STRIX_DISK_DIRECT");
        return !(v && *v && atoi(v) == 0);
    }();
#ifdef _WIN32
    return on;
#else
    return on && SPC_O_DIRECT != 0;
#endif
}

#ifndef _WIN32
// the pages of a buffered file are no use once it is written or read
void spc_drop_cache(int fd) {
#ifdef POSIX_FADV_DONTNEED
    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
#else
    (void) fd;
#endif
}

int spc_sync(int fd) {
#if defined(__APPLE__)
    return fsync(fd);
#else
    return fdatasync(fd);
#endif
}
#endif

struct spc_stage {
    static constexpr size_t SIZE  = 8u << 20;
    static constexpr size_t ALIGN = 64u << 10;      // covers every sector and logical block size in use, 4Kn included
    uint8_t * p = nullptr;
    spc_stage() {
#ifdef _WIN32
        p = (uint8_t *) VirtualAlloc(nullptr, SIZE, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
#else
        p = (uint8_t *) std::aligned_alloc(4096, SIZE);
#endif
        if (!p) {
            throw std::bad_alloc();
        }
    }
    ~spc_stage() {
#ifdef _WIN32
        VirtualFree(p, 0, MEM_RELEASE);
#else
        std::free(p);
#endif
    }
    spc_stage(const spc_stage &) = delete;
    spc_stage & operator=(const spc_stage &) = delete;
};

struct spc_file {
    explicit spc_file(spc_stage & s) : st(s) {}
    ~spc_file() { close(); }
    spc_file(const spc_file &) = delete;
    spc_file & operator=(const spc_file &) = delete;

    bool open(const std::string & path) {
        close();
        have = used = 0;
        bad  = false;
#ifdef _WIN32
        const std::wstring w = fs::path(path).wstring();
        h = INVALID_HANDLE_VALUE;
        if (disk_direct()) {
            h = CreateFileW(w.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                            FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
        }
        if (h == INVALID_HANDLE_VALUE) {
            // not every volume serves unbuffered reads; a buffered handle is slower, not broken
            h = CreateFileW(w.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                            FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
        }
        return h != INVALID_HANDLE_VALUE;
#else
        off    = 0;
        path_  = path;
        direct = false;
        if (disk_direct()) {
            fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | SPC_O_DIRECT);
            direct = fd >= 0;
        }
        if (fd < 0) {
            fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        }
        return fd >= 0;
#endif
    }

    void close() {
#ifdef _WIN32
        if (h != INVALID_HANDLE_VALUE) {
            CloseHandle(h);
            h = INVALID_HANDLE_VALUE;
        }
#else
        if (fd >= 0) {
            if (!direct) {
                spc_drop_cache(fd);
            }
            ::close(fd);
            fd = -1;
        }
#endif
    }

    // sequential; dst may be null to skip
    bool read(void * dst, size_t n) {
        uint8_t * out = (uint8_t *) dst;
        while (n > 0) {
            if (used == have && !refill()) {
                return false;
            }
            const size_t take = std::min(n, have - used);
            if (out) {
                memcpy(out, st.p + used, take);
                out += take;
            }
            used += take;
            n    -= take;
        }
        return true;
    }

private:
    spc_stage & st;
    size_t have = 0;
    size_t used = 0;
    bool   bad  = false;
#ifdef _WIN32
    HANDLE h = INVALID_HANDLE_VALUE;
#else
    int         fd     = -1;
    bool        direct = false;
    off_t       off    = 0;                         // file offset of the next refill
    std::string path_;
#endif

    bool refill() {
        if (bad) {
            return false;
        }
#ifdef _WIN32
        DWORD got = 0;
        if (h == INVALID_HANDLE_VALUE || !ReadFile(h, st.p, (DWORD) spc_stage::SIZE, &got, nullptr)) {
            bad = true;
            return false;
        }
        have = got;
#else
        if (fd < 0) {
            bad = true;
            return false;
        }
        ssize_t got;
        do {
            got = ::read(fd, st.p, spc_stage::SIZE);
        } while (got < 0 && errno == EINTR);
        if (got < 0 && direct && errno == EINVAL) {
            // a short read left the offset unaligned (it should not, before the end of a regular file): go on buffered
            ::close(fd);
            fd = ::open(path_.c_str(), O_RDONLY | O_CLOEXEC);
            direct = false;
            if (fd < 0 || lseek(fd, off, SEEK_SET) != off) {
                bad = true;
                return false;
            }
            do {
                got = ::read(fd, st.p, spc_stage::SIZE);
            } while (got < 0 && errno == EINTR);
        }
        if (got < 0) {
            bad = true;
            return false;
        }
        have = (size_t) got;
        off += got;
#endif
        used = 0;
        return have > 0;
    }
};

struct spc_out {
    explicit spc_out(spc_stage & s) : st(s) {}
    ~spc_out() { abort(); }
    spc_out(const spc_out &) = delete;
    spc_out & operator=(const spc_out &) = delete;

    uint64_t total = 0;

    bool open(const std::string & path) {
        abort();
        path_  = path;
        used   = 0;
        total  = 0;
        bad    = false;
        direct = false;
#ifdef _WIN32
        const std::wstring w = fs::path(path).wstring();
        if (disk_direct()) {
            h = CreateFileW(w.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
            direct = h != INVALID_HANDLE_VALUE;
        }
        if (h == INVALID_HANDLE_VALUE) {
            h = CreateFileW(w.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        }
        return h != INVALID_HANDLE_VALUE;
#else
        if (disk_direct()) {
            fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC | SPC_O_DIRECT, 0644);
            direct = fd >= 0;
        }
        if (fd < 0) {
            fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
        }
        return fd >= 0;
#endif
    }

    bool write(const void * src, size_t n) {
        const uint8_t * in = (const uint8_t *) src;
        while (n > 0 && !bad) {
            const size_t take = std::min(n, spc_stage::SIZE - used);
            memcpy(st.p + used, in, take);
            used  += take;
            in    += take;
            n     -= take;
            total += take;
            if (used == spc_stage::SIZE) {
                flush(spc_stage::SIZE);
            }
        }
        return !bad;
    }

    // unbuffered, the tail goes out padded to the block size and the file is then cut back to its real length.
    // Either way the data is on the device when this returns true, and none of it is left in the page cache.
    bool close() {
        if (!is_open()) {
            return false;
        }
        if (used > 0 && !bad) {
            const size_t padded = direct ? (used + spc_stage::ALIGN - 1) & ~(spc_stage::ALIGN - 1) : used;
            memset(st.p + used, 0, padded - used);
            flush(padded);
        }
#ifdef _WIN32
        if (!bad && direct) {
            FILE_END_OF_FILE_INFO eof {};
            eof.EndOfFile.QuadPart = (LONGLONG) total;
            if (!SetFileInformationByHandle(h, FileEndOfFileInfo, &eof, sizeof(eof))) {
                bad = true;
            }
        }
        if (!bad && !FlushFileBuffers(h)) {
            bad = true;
        }
        CloseHandle(h);
        h = INVALID_HANDLE_VALUE;
#else
        if (!bad && direct && ftruncate(fd, (off_t) total) != 0) {
            bad = true;
        }
        if (!bad && spc_sync(fd) != 0) {
            bad = true;
        }
        if (!direct) {
            spc_drop_cache(fd);                    // clean after the sync, so they go
        }
        ::close(fd);
        fd = -1;
#endif
        return !bad;
    }

    // close and remove what was written
    void abort() {
        if (!is_open()) {
            return;
        }
#ifdef _WIN32
        CloseHandle(h);
        h = INVALID_HANDLE_VALUE;
#else
        ::close(fd);
        fd = -1;
#endif
        std::error_code ec;
        fs::remove(path_, ec);
    }

private:
    spc_stage & st;
    size_t      used   = 0;
    bool        bad    = false;
    bool        direct = false;
    std::string path_;
#ifdef _WIN32
    HANDLE h = INVALID_HANDLE_VALUE;
    bool is_open() const { return h != INVALID_HANDLE_VALUE; }
#else
    int fd = -1;
    bool is_open() const { return fd >= 0; }
#endif

    void flush(size_t n) {
#ifdef _WIN32
        DWORD put = 0;
        if (!WriteFile(h, st.p, (DWORD) n, &put, nullptr) || put != n) {
            bad = true;
        }
#else
        size_t done = 0;
        while (done < n && !bad) {
            const ssize_t put = ::write(fd, st.p + done, n - done);
            if (put < 0 && errno == EINTR) {
                continue;
            }
            if (put <= 0) {
                bad = true;
                break;
            }
            done += (size_t) put;
        }
#endif
        used = 0;
    }
};

template <typename T> bool put(spc_out & o, const T & v) { return o.write(&v, sizeof(v)); }
template <typename T> bool get(spc_file & f, T & v) { return f.read(&v, sizeof(v)); }

bool put_blob(spc_out & o, const std::vector<uint8_t> & v) {
    return put(o, (uint64_t) v.size()) && (v.empty() || o.write(v.data(), v.size()));
}

// a load splits "read" into the file and the buffer it goes into
struct spc_read_cost { int64_t alloc_us; int64_t read_us; uint64_t bytes; };
thread_local spc_read_cost g_spc_cost = {};

bool get_blob(spc_file & f, std::vector<uint8_t> & v, uint64_t limit) {
    uint64_t n = 0;
    if (!get(f, n) || n > limit) {
        return false;
    }
    const int64_t t0 = ggml_time_us();
    v.resize(n);
    const int64_t t1 = ggml_time_us();
    const bool ok = n == 0 || f.read(v.data(), n);
    g_spc_cost.alloc_us += t1 - t0;
    g_spc_cost.read_us  += ggml_time_us() - t1;
    g_spc_cost.bytes    += n;
    return ok;
}

bool put_tokens(spc_out & o, const server_tokens & t) {
    const std::vector<char> b = t.serialize();
    return put(o, (uint64_t) b.size()) && (b.empty() || o.write(b.data(), b.size()));
}

// tokens as the serialized bytes, reinterpreted the way slot restore does
bool get_tokens(spc_file & f, server_tokens & out, bool has_mtmd) {
    std::vector<uint8_t> raw;
    if (!get_blob(f, raw, 1ull << 30) || raw.size() % sizeof(llama_token) != 0) {
        return false;
    }
    llama_tokens packed(raw.size() / sizeof(llama_token));
    if (!packed.empty()) {
        memcpy(packed.data(), raw.data(), raw.size());
    }
    out = server_tokens::deserialize(packed, has_mtmd);
    return true;
}

// write a file as <path>.part; nothing is left behind on failure
template <typename F> bool write_part(spc_stage & st, const std::string & part, F && body, uint64_t & bytes) {
    spc_out o(st);
    if (!o.open(part)) {
        return false;
    }
    if (!body(o) || !o.close()) {
        o.abort();
        std::error_code ec;
        fs::remove(part, ec);
        return false;
    }
    bytes = o.total;
    return true;
}

bool place(const std::string & part, const std::string & path) {
    std::error_code ec;
    fs::rename(part, path, ec);
    if (ec) {
        fs::remove(part, ec);
        return false;
    }
    return true;
}

std::string manifest_path(const std::string & dir, const server_tokens & tokens) {
    const std::vector<char> b = tokens.serialize();
    char name[64];
    snprintf(name, sizeof(name), "%016llx-%d.spc", (unsigned long long) XXH3_64bits(b.data(), b.size()), (int) tokens.size());
    return (fs::path(dir) / name).string();
}

bool write_manifest_part(spc_stage & st, const std::string & part, const disk_entry & e, bool has_mtmd, uint64_t & bytes) {
    return write_part(st, part, [&](spc_out & o) {
        bool ok = o.write(SPC_MAGIC, 8) && put(o, SPC_V3) && put(o, (uint32_t) (has_mtmd ? 1 : 0))
               && put_tokens(o, e.tokens) && put(o, (uint32_t) e.ckpts.size());
        for (const auto & k : e.ckpts) {
            ok = ok && put(o, k.key) && put(o, k.n_tokens) && put(o, k.pos_min) && put(o, k.pos_max)
                    && put(o, k.size_tgt) && put(o, k.size_dft) && put(o, k.size_spec);
        }
        for (int b = 0; b < 2; ++b) {
            const auto & runs = b == 0 ? e.runs_tgt : e.runs_dft;
            ok = ok && put(o, b == 0 ? e.row_tgt : e.row_dft) && put(o, (uint32_t) runs.size());
            for (const auto & r : runs) {
                ok = ok && put(o, r.hi) && put(o, r.lo) && put(o, r.pos0) && put(o, r.n) && put(o, r.n_use);
            }
        }
        return ok;
    }, bytes);
}

// a run file: the rows as the context gave them, checked against their name on the way back
bool write_run_body(spc_out & o, uint64_t row, const disk_run_ref & r, const std::vector<uint8_t> & rows) {
    return o.write(RUN_MAGIC, 8) && put(o, RUN_V1) && put(o, row) && put(o, (uint32_t) r.n)
        && o.write(rows.data(), rows.size()) && put(o, r.hi) && put(o, r.lo);
}

// into buf, resized to the run's rows; false for a missing, damaged or foreign file. Throws std::bad_alloc when
// there is no memory for it, which says nothing about the file.
bool read_run(spc_file & f, const std::string & path, const disk_run_ref & r, uint64_t row, std::vector<uint8_t> & buf) {
    if (!f.open(path)) {
        return false;
    }
    char magic[8];
    uint32_t version = 0, n = 0;
    uint64_t row_ref = 0, hi = 0, lo = 0;
    const size_t size = (size_t) row * (size_t) r.n;
    bool ok = f.read(magic, 8) && memcmp(magic, RUN_MAGIC, 8) == 0 && get(f, version) && version == RUN_V1
           && get(f, row_ref) && row_ref == row && get(f, n) && (int32_t) n == r.n;
    if (ok) {
        const int64_t t0 = ggml_time_us();
        buf.resize(size);
        const int64_t t1 = ggml_time_us();
        ok = f.read(buf.data(), size) && get(f, hi) && get(f, lo);
        g_spc_cost.alloc_us += t1 - t0;
        g_spc_cost.read_us  += ggml_time_us() - t1;
        g_spc_cost.bytes    += size;
    }
    f.close();
    if (ok) {
        const XXH128_hash_t h = XXH3_128bits(buf.data(), size);
        ok = h.high64 == r.hi && h.low64 == r.lo && hi == r.hi && lo == r.lo;
    }
    return ok;
}

// the header of an entry: its tokens and the objects it names. `foreign` says the file is not to be deleted
// for failing - it could not be opened (a lock, a scanner) or a later build wrote it; `old` that it is of an
// earlier format, which this store does not hold
bool scan_entry(spc_stage & st, disk_entry & e, bool has_mtmd, std::string & why, bool & foreign, bool & old) {
    foreign = false;
    old     = false;
    spc_file f(st);
    if (!f.open(e.path)) {
        why = "cannot open it";
        foreign = true;
        return false;
    }
    char magic[8];
    uint32_t version = 0, mtmd = 0;
    if (!f.read(magic, 8) || memcmp(magic, SPC_MAGIC, 8) != 0 || !get(f, version) || !get(f, mtmd)) {
        why = "not a cache file";
        return false;
    }
    if (version > SPC_V3) {
        why = "format version " + std::to_string(version);
        foreign = true;
        return false;
    }
    if (version < SPC_V3) {
        why = "format version " + std::to_string(version);
        old = true;
        return false;
    }
    try {
        if (!get_tokens(f, e.tokens, has_mtmd)) {
            why = "truncated or malformed token state";
            return false;
        }
    } catch (const std::exception & ex) {
        why = ex.what();
        return false;
    }
    // a manifest is named after its tokens: a mismatch is a damaged file, found for free
    {
        const std::vector<char> b = e.tokens.serialize();
        char name[32];
        snprintf(name, sizeof(name), "%016llx-", (unsigned long long) XXH3_64bits(b.data(), b.size()));
        if (fs::path(e.path).filename().string().rfind(name, 0) != 0) {
            why = "its tokens do not match its name";
            return false;
        }
    }
    uint32_t n_ckpt = 0;
    if (!get(f, n_ckpt) || n_ckpt > 4096) {
        why = "bad checkpoint list";
        return false;
    }
    for (uint32_t i = 0; i < n_ckpt; ++i) {
        disk_ckpt_ref r {};
        if (!get(f, r.key) || !get(f, r.n_tokens) || !get(f, r.pos_min) || !get(f, r.pos_max) ||
            !get(f, r.size_tgt) || !get(f, r.size_dft) || !get(f, r.size_spec)) {
            why = "truncated checkpoint list";
            return false;
        }
        e.ckpts.push_back(r);
    }
    // runs from position 0, one after another: a gap or an overlap is a damaged manifest
    for (int b = 0; b < 2; ++b) {
        auto & runs = b == 0 ? e.runs_tgt : e.runs_dft;
        uint64_t & row = b == 0 ? e.row_tgt : e.row_dft;
        uint32_t n = 0;
        if (!get(f, row) || !get(f, n) || n > (1u << 20) || row > (1u << 24)) {
            why = "bad run list";
            return false;
        }
        int64_t next = 0;
        for (uint32_t i = 0; i < n; ++i) {
            disk_run_ref r {};
            if (!get(f, r.hi) || !get(f, r.lo) || !get(f, r.pos0) || !get(f, r.n) || !get(f, r.n_use)) {
                why = "truncated run list";
                return false;
            }
            if (r.pos0 != next || r.n <= 0 || r.n_use <= 0 || r.n_use > r.n || row == 0) {
                why = "runs out of order";
                return false;
            }
            next += r.n_use;
            runs.push_back(r);
        }
    }
    e.n_exact = runs_exact(e.tokens, e.runs_tgt, e.runs_dft, e.row_dft > 0, e.ckpts);
    return true;
}

// a checkpoint is more than half of an entry's bytes, and one wrong byte in a recurrent state restores nonsense
// or trips the size assert in the restore, so it is checked like a run
XXH128_hash_t ckpt_hash(const common_prompt_checkpoint & c) {
    XXH3_state_t hs;
    XXH3_128bits_reset(&hs);
    for (const auto * v : { &c.data_tgt, &c.data_dft, &c.data_spec }) {
        if (!v->empty()) {
            XXH3_128bits_update(&hs, v->data(), v->size());
        }
    }
    return XXH3_128bits_digest(&hs);
}

bool write_ckpt_body(spc_out & o, const common_prompt_checkpoint & c) {
    const XXH128_hash_t h = ckpt_hash(c);
    return o.write(CKP_MAGIC, 8) && put(o, CKP_V2) && put(o, (int64_t) c.n_tokens) && put(o, (int32_t) c.pos_min)
        && put(o, (int32_t) c.pos_max) && put_blob(o, c.data_tgt) && put_blob(o, c.data_dft) && put_blob(o, c.data_spec)
        && put(o, h.high64) && put(o, h.low64);
}

// every blob is read against the size the manifest recorded, so a damaged length cannot ask for a TiB
bool read_ckpt(spc_file & f, const std::string & path, const disk_ckpt_ref & r, common_prompt_checkpoint & c) {
    if (!f.open(path)) {
        return false;
    }
    char magic[8];
    uint32_t version = 0;
    int64_t n_tokens = 0;
    int32_t pmin = 0, pmax = 0;
    uint64_t hi = 0, lo = 0;
    bool ok = f.read(magic, 8) && memcmp(magic, CKP_MAGIC, 8) == 0 && get(f, version) && version == CKP_V2
        && get(f, n_tokens) && get(f, pmin) && get(f, pmax)
        && n_tokens == r.n_tokens && pmin == r.pos_min && pmax == r.pos_max
        && get_blob(f, c.data_tgt,  r.size_tgt)  && c.data_tgt.size()  == r.size_tgt
        && get_blob(f, c.data_dft,  r.size_dft)  && c.data_dft.size()  == r.size_dft
        && get_blob(f, c.data_spec, r.size_spec) && c.data_spec.size() == r.size_spec
        && get(f, hi) && get(f, lo);
    f.close();
    if (ok) {
        const XXH128_hash_t h = ckpt_hash(c);
        ok = h.high64 == hi && h.low64 == lo;
    }
    if (ok) {
        c.n_tokens = n_tokens;
        c.pos_min  = pmin;
        c.pos_max  = pmax;
    }
    return ok;
}

// what a failed read blames, so the writer can retire it along with every entry that names it
struct read_fault {
    bool     ckpt = false;
    uint64_t key  = 0;
    bool     run  = false;
    uint64_t hi = 0, lo = 0;
};

int64_t mtime_ms(const fs::directory_entry & e) {
    std::error_code ec;
    return std::chrono::duration_cast<std::chrono::milliseconds>(e.last_write_time(ec).time_since_epoch()).count();
}

// the files in a directory, without the throwing overloads: one bad directory entry must not take the
// server down at startup
std::vector<fs::directory_entry> list_dir(const fs::path & dir, bool recursive) {
    std::vector<fs::directory_entry> out;
    std::error_code ec;
    if (recursive) {
        for (auto it = fs::recursive_directory_iterator(dir, ec); !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
            std::error_code ec2;
            if (it->is_regular_file(ec2)) {
                out.push_back(*it);
            }
        }
    } else {
        for (auto it = fs::directory_iterator(dir, ec); !ec && it != fs::directory_iterator(); it.increment(ec)) {
            std::error_code ec2;
            if (it->is_regular_file(ec2)) {
                out.push_back(*it);
            }
        }
    }
    return out;
}

constexpr double GIB = 1024.0 * 1024.0 * 1024.0;

} // namespace

server_prompt_cache::~server_prompt_cache() {
    if (disk_thread.joinable()) {
        {
            std::lock_guard<std::mutex> lk(disk_mu);
            disk_stop = true;
            disk_on_written = nullptr;     // nothing to wake once the server is going away
        }
        disk_cv.notify_all();
        disk_thread.join();
    }
}

void server_prompt_cache::set_disk(const std::string & root, size_t limit_mib, bool has_mtmd, const std::string & model_tag,
                                   uint64_t row_tgt, uint64_t row_dft) {
    std::error_code ec;
    // one directory per model: a state is only meaningful to the model that computed it, and two quants of one
    // model have identical state shapes, so the restore would take the other's without a word
    const fs::path dir = model_tag.empty() ? fs::path(root) : fs::path(root) / model_tag;
    fs::create_directories(dir / "ckpt", ec);
    fs::create_directories(dir / "runs", ec);
    disk_dir      = dir.string();
    disk_limit    = (uint64_t) limit_mib * 1024ull * 1024ull;
    disk_has_mtmd = has_mtmd;
    if (const char * r = getenv("STRIX_PROMPT_CACHE_RUN"); r && *r) {
        disk_run = std::max(0, atoi(r));
    }

    std::lock_guard<std::mutex> lk(disk_mu);
    disk_index.clear();
    disk_ckpts.clear();
    disk_runs.clear();
    disk_bytes = 0;

    // 1. what an interrupted write left
    for (const auto & de : list_dir(dir, true)) {
        if (de.path().extension() == ".part") {
            fs::remove(de.path(), ec);
        }
    }

    // 2. entries, and a reference for each object they name. One of another format goes, and the objects only it
    // named are swept below as orphans. A file this build cannot open or does not know the version of is left
    // alone - a later build may have written it - and then nothing is swept, because its objects cannot be told
    // apart from orphans.
    spc_stage stage;
    int n_skipped = 0;
    int n_old = 0;
    uint64_t old_bytes = 0;
    for (const auto & de : list_dir(dir, false)) {
        if (de.path().extension() != ".spc") {
            continue;
        }
        disk_entry e;
        e.path  = de.path().string();
        e.bytes = (uint64_t) de.file_size(ec);
        e.order = mtime_ms(de);
        std::string why;
        bool foreign = false, old = false;
        bool ok = scan_entry(stage, e, disk_has_mtmd, why, foreign, old);
        // rows of another size - a K/V cache of another type, or no draft where this server drafts - are as
        // unusable as another format, and go the same way
        if (ok && row_tgt > 0 && (e.row_tgt != row_tgt || (row_dft > 0 && e.row_dft != row_dft))) {
            ok = false;
            old = true;
        }
        if (!ok) {
            if (old) {
                fs::remove(de.path(), ec);
                ++n_old;
                old_bytes += e.bytes;
            } else if (foreign) {
                SRV_WRN(" - disk cache: leaving %s alone: %s\n", de.path().filename().string().c_str(), why.c_str());
                ++n_skipped;
            } else {
                SRV_WRN(" - disk cache: dropping %s (%.2f GiB): %s\n", de.path().filename().string().c_str(), e.bytes / GIB, why.c_str());
                fs::remove(de.path(), ec);
            }
            continue;
        }
        disk_seq = std::max(disk_seq, e.order);
        for (const auto * runs : { &e.runs_tgt, &e.runs_dft }) {
            for (const auto & r : *runs) {
                disk_runs[{ r.hi, r.lo }].refs++;
            }
        }
        for (const auto & k : e.ckpts) {
            disk_ckpts[k.key].refs++;
        }
        disk_index.push_back(std::move(e));
    }

    // 3. objects: size what is referenced, remove what is not (unless a skipped entry might name it)
    size_t n_orphans = 0;
    uint64_t orphan_bytes = 0;
    for (const auto & de : list_dir(dir / "ckpt", false)) {
        if (de.path().extension() != ".ckp") {
            continue;
        }
        char * end = nullptr;
        const uint64_t key = strtoull(de.path().stem().string().c_str(), &end, 16);
        auto o = *end == 0 ? disk_ckpts.find(key) : disk_ckpts.end();
        if (o != disk_ckpts.end()) {
            o->second.bytes = (uint64_t) de.file_size(ec);
        } else if (n_skipped == 0) {
            orphan_bytes += (uint64_t) de.file_size(ec);
            fs::remove(de.path(), ec);
            ++n_orphans;
        }
    }
    for (const auto & de : list_dir(dir / "runs", true)) {
        if (de.path().extension() != ".run") {
            continue;
        }
        const std::string hex = de.path().stem().string();
        uint64_t hi = 0, lo = 0;
        bool named = hex.size() == 32;
        if (named) {
            char * end = nullptr;
            hi = strtoull(hex.substr(0, 16).c_str(), &end, 16); named = named && *end == 0;
            lo = strtoull(hex.substr(16).c_str(), &end, 16);    named = named && *end == 0;
        }
        auto o = named ? disk_runs.find({ hi, lo }) : disk_runs.end();
        if (o != disk_runs.end()) {
            o->second.bytes = (uint64_t) de.file_size(ec);
        } else if (n_skipped == 0) {
            orphan_bytes += (uint64_t) de.file_size(ec);
            fs::remove(de.path(), ec);
            ++n_orphans;
        }
    }

    // 4. an entry whose objects are gone, or not the size it recorded, cannot be restored
    for (auto it = disk_index.begin(); it != disk_index.end();) {
        bool whole = true;
        for (int b = 0; b < 2; ++b) {
            for (const auto & r : b == 0 ? it->runs_tgt : it->runs_dft) {
                whole = whole && disk_runs[{ r.hi, r.lo }].bytes == run_file_bytes(b == 0 ? it->row_tgt : it->row_dft, r.n);
            }
        }
        for (const auto & k : it->ckpts) {
            whole = whole && disk_ckpts[k.key].bytes > 0;
        }
        if (whole) {
            ++it;
            continue;
        }
        SRV_WRN(" - disk cache: dropping %s: objects it names are missing or damaged\n", fs::path(it->path).filename().string().c_str());
        disk_drop(*it);
        it = disk_index.erase(it);
    }
    disk_bytes = 0;
    for (const auto & e : disk_index) {
        disk_bytes += e.bytes;
    }
    for (const auto & [key, o] : disk_ckpts) {
        disk_bytes += o.bytes;
    }
    for (const auto & [key, o] : disk_runs) {
        disk_bytes += o.bytes;
    }
    disk_evict("");

    // 5. the writer
    disk_stop = false;
    disk_thread = std::thread([this] { disk_writer(); });

    if (n_old > 0) {
        SRV_INF(" - disk cache: %d entries of another format or row size removed, %.2f GiB with the objects no entry names any more\n",
                n_old, (old_bytes + orphan_bytes) / GIB);
    }
    char orphans[64] = "";
    if (n_orphans > 0) {
        snprintf(orphans, sizeof(orphans), ", %zu orphans removed (%.2f GiB)", n_orphans, orphan_bytes / GIB);
    }
    SRV_INF("disk prompt cache: %zu entries, %.1f GiB in %s (limit %.1f GiB, %zu runs, %zu checkpoints, runs of %d positions, "
            "%s I/O%s%s)\n",
            disk_index.size(), disk_bytes / GIB, disk_dir.c_str(), disk_limit / GIB, disk_runs.size(), disk_ckpts.size(), disk_run,
            disk_direct() ? "unbuffered" : "buffered", orphans,
            n_skipped ? (", " + std::to_string(n_skipped) + " entries left alone").c_str() : "");
}

uint64_t server_prompt_cache::disk_size() {
    std::lock_guard<std::mutex> lk(disk_mu);
    return disk_bytes;
}

bool server_prompt_cache::disk_busy() {
    std::lock_guard<std::mutex> lk(disk_mu);
    return !disk_queue.empty();
}

bool server_prompt_cache::disk_drain(int64_t timeout_ms) {
    std::unique_lock<std::mutex> lk(disk_mu);
    return disk_cv.wait_for(lk, std::chrono::milliseconds(timeout_ms), [&] { return disk_queue.empty() && disk_current == nullptr; });
}

// caller holds disk_mu. The entry's objects lose a reference each; those nobody holds any more go, and so does
// the entry's own file unless `remove_file` is false (its path now holds a newer manifest). Only the writer thread
// calls this (and set_disk, before the writer starts).
void server_prompt_cache::disk_drop(const disk_entry & e, bool remove_file) {
    std::error_code ec;
    for (const auto & k : e.ckpts) {
        auto it = disk_ckpts.find(k.key);
        if (it == disk_ckpts.end()) {
            continue;
        }
        if (--it->second.refs <= 0) {
            fs::remove(ckpt_path(disk_dir, k.key), ec);
            disk_bytes -= std::min(disk_bytes, it->second.bytes);
            disk_ckpts.erase(it);
        }
    }
    for (const auto * runs : { &e.runs_tgt, &e.runs_dft }) {
        for (const auto & r : *runs) {
            auto it = disk_runs.find({ r.hi, r.lo });
            if (it == disk_runs.end()) {
                continue;
            }
            if (--it->second.refs <= 0) {
                fs::remove(run_path(disk_dir, r.hi, r.lo), ec);
                disk_bytes -= std::min(disk_bytes, it->second.bytes);
                disk_runs.erase(it);
            }
        }
    }
    if (remove_file) {
        fs::remove(e.path, ec);
    }
    disk_bytes -= std::min(disk_bytes, e.bytes);
}

// caller holds disk_mu: least recently written or used goes first, never `keep`
void server_prompt_cache::disk_evict(const std::string & keep) {
    while (disk_index.size() > 1 && disk_bytes > disk_limit) {
        auto oldest = disk_index.end();
        for (auto it = disk_index.begin(); it != disk_index.end(); ++it) {
            if (it->path != keep && (oldest == disk_index.end() || it->order < oldest->order)) {
                oldest = it;
            }
        }
        if (oldest == disk_index.end()) {
            break;
        }
        SRV_INF(" - disk cache: evicting %d tokens (%s) to stay under %.1f GiB\n", (int) oldest->tokens.size(),
                fs::path(oldest->path).filename().string().c_str(), disk_limit / GIB);
        disk_drop(*oldest);
        disk_index.erase(oldest);
    }
}

// caller holds disk_mu: the most of `tokens` that an entry - or a job, with_jobs - brings back exactly
size_t server_prompt_cache::disk_covered_locked(const server_tokens & tokens, bool with_jobs) {
    size_t best = 0;
    auto consider = [&](const server_tokens & t, int64_t exact) {
        const size_t lcp = t.get_common_prefix(tokens);
        const size_t e   = std::min((size_t) std::max<int64_t>(exact, 0), tokens.size());
        if (lcp >= e) {
            best = std::max(best, e);
        }
    };
    for (const auto & e : disk_index) {
        consider(e.tokens, e.n_exact);
        if (best == tokens.size()) {
            return best;
        }
    }
    if (with_jobs) {
        for (const auto & j : disk_queue) {
            consider(j->tokens, j->n_exact);
        }
        if (disk_current) {
            consider(disk_current->tokens, disk_current->n_exact);
        }
    }
    return best;
}

size_t server_prompt_cache::disk_covered(const server_tokens & tokens, bool with_jobs) {
    if (disk_limit == 0) {
        return 0;
    }
    std::lock_guard<std::mutex> lk(disk_mu);
    return disk_covered_locked(tokens, with_jobs);
}

size_t server_prompt_cache::disk_exact(const server_tokens & tokens) {
    return disk_covered(tokens, /* with_jobs = */ true);
}

server_prompt_cache::disk_run_ref server_prompt_cache::make_run_ref(const std::vector<uint8_t> & rows, int32_t pos0, int32_t n) {
    const XXH128_hash_t h = XXH3_128bits(rows.data(), rows.size());
    return { h.high64, h.low64, pos0, n, n };
}

bool server_prompt_cache::runs_lost(int32_t id_slot) {
    std::lock_guard<std::mutex> lk(disk_mu);
    return disk_runs_lost.erase(id_slot) > 0;
}

// caller holds disk_mu: a job that is queued or being written and would be a better start for this prompt than
// what the slot or the RAM tier already offers (f_sim_base). Anything less is not worth waiting for - notably the
// slot's own previous conversation, queued a moment ago on its way out.
bool server_prompt_cache::disk_in_flight(const server_tokens & tokens, float f_sim_base) {
    if (!disk_thread.joinable() || disk_stop || tokens.size() == 0) {
        return false;
    }
    auto relevant = [&](const disk_job & j) {
        const size_t lcp = j.tokens.get_common_prefix(tokens);
        return lcp > 0 && lcp * 4 >= j.tokens.size() && float(lcp) / tokens.size() > f_sim_base;
    };
    for (const auto & j : disk_queue) {
        if (relevant(*j)) {
            return true;
        }
    }
    return disk_current && relevant(*disk_current);
}

// caller holds disk_mu
bool server_prompt_cache::disk_ckpt_in_flight(uint64_t key) {
    auto carries = [&](const disk_job & j) {
        return std::any_of(j.ckpt_data.begin(), j.ckpt_data.end(), [&](const auto & kc) { return kc.first == key; });
    };
    for (const auto & j : disk_queue) {
        if (carries(*j)) {
            return true;
        }
    }
    return disk_current && carries(*disk_current);
}

bool server_prompt_cache::persist_runs(int32_t id_slot, const server_tokens & tokens, const std::list<common_prompt_checkpoint> & ckpts,
                                       const std::set<const common_prompt_checkpoint *> & writable,
                                       const ckpt_paged_map * paged, const common_prompt_checkpoint * end, uint64_t row_tgt, uint64_t row_dft,
                                       const std::vector<disk_run_ref> & runs_tgt, const std::vector<disk_run_ref> & runs_dft,
                                       std::vector<std::pair<std::pair<uint64_t, uint64_t>, std::vector<uint8_t>>> && run_data, bool wait) {
    if (disk_limit == 0 || tokens.size() == 0 || !disk_thread.joinable() || row_tgt == 0) {
        return false;
    }
    const llama_tokens text = tokens.get_text_tokens();
    if (text.size() != tokens.size()) {
        return false;                              // media: never stored
    }
    if (!wait && disk_busy()) {
        return false;                              // the next round tries again
    }

    auto job = std::make_unique<disk_job>();
    job->id_slot  = id_slot;
    job->tokens   = tokens.clone();
    job->row_tgt  = row_tgt;
    job->row_dft  = row_dft;
    job->runs_tgt = runs_tgt;
    job->runs_dft = runs_dft;
    job->run_data = std::move(run_data);

    // the checkpoints up to its end, and the one at its end when the caller made one; a checkpoint with bytes that
    // is not the caller's to write counts as one handed back (named when stored)
    const int64_t n = (int64_t) tokens.size();
    std::vector<std::pair<disk_ckpt_ref, const common_prompt_checkpoint *>> refs;
    auto add = [&](const common_prompt_checkpoint & c, bool may_write) {
        if (c.n_tokens > n) {
            return;
        }
        disk_ckpt_ref r {};
        const common_prompt_checkpoint * data = nullptr;
        if (c.data_tgt.empty()) {
            const auto p = paged ? paged->find({ c.n_tokens, c.pos_min, c.pos_max }) : ckpt_paged_map::const_iterator();
            if (!paged || p == paged->end() ||
                ckpt_key(text, p->second.n_tokens, p->second.pos_min, p->second.pos_max,
                         p->second.size_tgt, p->second.size_dft, p->second.size_spec) != p->second.key) {
                return;
            }
            r = p->second;
        } else {
            r = ckpt_ref_of(text, c);
            data = may_write ? &c : nullptr;
        }
        for (const auto & x : refs) {
            if (x.first.key == r.key) {
                return;
            }
        }
        refs.push_back({ r, data });
    };
    for (const auto & c : ckpts) {
        add(c, writable.count(&c) > 0);
    }
    if (end) {
        add(*end, true);
    }

    std::vector<bool> need(refs.size(), false);
    auto unpin_job = [&]() {
        std::lock_guard<std::mutex> lk(disk_mu);
        disk_unpin.insert(disk_unpin.end(), job->pinned.begin(), job->pinned.end());
        disk_unpin_runs.insert(disk_unpin_runs.end(), job->run_pinned.begin(), job->run_pinned.end());
        job->pinned.clear();
        job->run_pinned.clear();
        disk_cv.notify_all();
    };
    {
        std::lock_guard<std::mutex> lk(disk_mu);
        for (size_t i = 0; i < refs.size(); ++i) {
            auto o = disk_ckpts.find(refs[i].first.key);
            const bool stored = o != disk_ckpts.end() || disk_ckpt_in_flight(refs[i].first.key);
            if (refs[i].second) {
                need[i] = !stored;
                job->ckpts.push_back(refs[i].first);
            } else if (stored) {
                job->ckpts.push_back(refs[i].first);
            }
            // named but not carried, and on disk: held until the job's entry counts it, so an eviction in the
            // meantime cannot take it
            if (!need[i] && o != disk_ckpts.end() && (refs[i].second || stored)) {
                o->second.refs++;
                job->pinned.push_back(refs[i].first.key);
            }
        }
        // the runs it names without their rows are in the store, pinned until the entry counts them, or on their way
        // there in a job ahead of this one. One that is neither went with an evicted entry: the slot starts over.
        std::set<std::pair<uint64_t, uint64_t>> carried;
        for (const auto & d : job->run_data) {
            carried.insert(d.first);
        }
        auto in_flight = [&](const std::pair<uint64_t, uint64_t> & k) {
            auto carries = [&](const disk_job & j) {
                return std::any_of(j.run_data.begin(), j.run_data.end(), [&](const auto & d) { return d.first == k; });
            };
            return std::any_of(disk_queue.begin(), disk_queue.end(), [&](const auto & j) { return carries(*j); }) ||
                   (disk_current && carries(*disk_current));
        };
        for (const auto * runs : { &job->runs_tgt, &job->runs_dft }) {
            for (const auto & r : *runs) {
                const std::pair<uint64_t, uint64_t> k { r.hi, r.lo };
                if (carried.count(k)) {
                    continue;
                }
                auto o = disk_runs.find(k);
                if (o != disk_runs.end()) {
                    o->second.refs++;
                    job->run_pinned.push_back(k);
                } else if (!in_flight(k)) {
                    disk_runs_lost.insert(id_slot);
                    disk_unpin.insert(disk_unpin.end(), job->pinned.begin(), job->pinned.end());
                    disk_unpin_runs.insert(disk_unpin_runs.end(), job->run_pinned.begin(), job->run_pinned.end());
                    disk_cv.notify_all();
                    return false;
                }
            }
        }
    }
    try {
        for (size_t i = 0; i < refs.size(); ++i) {
            if (need[i]) {
                job->ckpt_data.emplace_back(refs[i].first.key, *refs[i].second);
            }
        }
    } catch (const std::bad_alloc &) {
        SRV_WRN("%s", " - disk cache: not enough memory to copy the checkpoints, not saving\n");
        unpin_job();
        return false;
    }
    job->n_exact = runs_exact(job->tokens, job->runs_tgt, job->runs_dft, row_dft > 0, job->ckpts);

    std::unique_lock<std::mutex> lk(disk_mu);
    // a run is ~110 MB and a checkpoint about as much: one job waiting besides the one being written
    if (!disk_queue.empty()) {
        if (!wait || disk_stop) {
            lk.unlock();
            unpin_job();
            return false;
        }
        disk_cv.wait(lk, [&] { return disk_queue.empty() || disk_stop; });
        if (disk_stop) {
            lk.unlock();
            unpin_job();
            return false;
        }
    }
    disk_queue.push_back(std::move(job));
    disk_cv.notify_all();
    return true;
}

uint64_t server_prompt_cache::ckpt_page_out(const server_tokens & tokens, std::list<common_prompt_checkpoint> & ckpts, ckpt_paged_map & paged) {
    if (disk_limit == 0) {
        return 0;
    }
    const llama_tokens text = tokens.get_text_tokens();
    if (text.size() != tokens.size()) {
        return 0;                                  // media: never stored
    }
    uint64_t freed = 0;
    std::lock_guard<std::mutex> lk(disk_mu);
    for (auto & c : ckpts) {
        if (c.data_tgt.empty()) {
            continue;                              // handed back already
        }
        const disk_ckpt_ref r = ckpt_ref_of(text, c);
        auto stored = disk_ckpts.find(r.key);
        if (stored == disk_ckpts.end()) {
            continue;                              // not stored, or only on its way: keep the bytes
        }
        // a pin: the store keeps the file while this slot may rewind to it, whatever becomes of its entry
        const std::tuple<int64_t, int32_t, int32_t> id { c.n_tokens, c.pos_min, c.pos_max };
        auto p = paged.find(id);
        if (p == paged.end() || p->second.key != r.key) {
            if (p != paged.end()) {
                disk_unpin.push_back(p->second.key);
            }
            stored->second.refs++;
        }
        paged[id] = r;
        freed += c.size();
        std::vector<uint8_t>().swap(c.data_tgt);
        std::vector<uint8_t>().swap(c.data_dft);
        std::vector<uint8_t>().swap(c.data_spec);
    }
    return freed;
}

bool server_prompt_cache::ckpt_page_in(const server_tokens & tokens, const ckpt_paged_map & paged, common_prompt_checkpoint & c) {
    const auto it = paged.find({ c.n_tokens, c.pos_min, c.pos_max });
    if (it == paged.end() || disk_limit == 0) {
        return false;
    }
    const disk_ckpt_ref & r = it->second;
    // the table outlives a change of conversation in the slot: the key says whether this is still the same one
    const llama_tokens text = tokens.get_text_tokens();
    if (ckpt_key(text, r.n_tokens, r.pos_min, r.pos_max, r.size_tgt, r.size_dft, r.size_spec) != r.key) {
        return false;
    }
    std::lock_guard<std::mutex> lk(disk_mu);       // the writer cannot delete it while it is read
    if (!disk_ckpts.count(r.key)) {
        return false;
    }
    common_prompt_checkpoint tmp;
    bool ok = false;
    try {
        spc_stage st;
        spc_file f(st);
        ok = read_ckpt(f, ckpt_path(disk_dir, r.key), r, tmp);
    } catch (const std::bad_alloc &) {
        return false;                              // no memory this time; the file is not to blame
    }
    if (!ok) {
        disk_bad_ckpts.push_back(r.key);           // retire it, and every entry that names it
        disk_cv.notify_all();
        return false;
    }
    c.data_tgt  = std::move(tmp.data_tgt);
    c.data_dft  = std::move(tmp.data_dft);
    c.data_spec = std::move(tmp.data_spec);
    return true;
}

void server_prompt_cache::ckpt_release(ckpt_paged_map & paged) {
    if (paged.empty()) {
        return;
    }
    std::lock_guard<std::mutex> lk(disk_mu);
    for (const auto & [id, r] : paged) {
        disk_unpin.push_back(r.key);
    }
    paged.clear();
    disk_cv.notify_all();
}

// caller holds disk_mu: pins let go of; an object nobody holds any more goes
void server_prompt_cache::disk_release_pins() {
    std::error_code ec;
    for (const uint64_t key : disk_unpin) {
        auto o = disk_ckpts.find(key);
        if (o != disk_ckpts.end() && --o->second.refs <= 0) {
            fs::remove(ckpt_path(disk_dir, key), ec);
            disk_bytes -= std::min(disk_bytes, o->second.bytes);
            disk_ckpts.erase(o);
        }
    }
    disk_unpin.clear();
    for (const auto & k : disk_unpin_runs) {
        auto o = disk_runs.find(k);
        if (o != disk_runs.end() && --o->second.refs <= 0) {
            fs::remove(run_path(disk_dir, k.first, k.second), ec);
            disk_bytes -= std::min(disk_bytes, o->second.bytes);
            disk_runs.erase(o);
        }
    }
    disk_unpin_runs.clear();
}

// caller holds disk_mu: objects a read found missing or damaged go, with every entry that names them
void server_prompt_cache::disk_retire_bad() {
    if (disk_bad_ckpts.empty() && disk_bad_runs.empty()) {
        return;
    }
    auto bad_ckpt = [&](const disk_ckpt_ref & k) {
        return std::find(disk_bad_ckpts.begin(), disk_bad_ckpts.end(), k.key) != disk_bad_ckpts.end();
    };
    auto bad_run = [&](const disk_run_ref & r) {
        return std::find(disk_bad_runs.begin(), disk_bad_runs.end(), std::make_pair(r.hi, r.lo)) != disk_bad_runs.end();
    };
    for (auto it = disk_index.begin(); it != disk_index.end();) {
        const bool names_bad = std::any_of(it->ckpts.begin(), it->ckpts.end(), bad_ckpt)
                            || std::any_of(it->runs_tgt.begin(), it->runs_tgt.end(), bad_run)
                            || std::any_of(it->runs_dft.begin(), it->runs_dft.end(), bad_run);
        if (names_bad) {
            SRV_WRN(" - disk cache: dropping %s: it names a damaged object\n", fs::path(it->path).filename().string().c_str());
            disk_doomed.push_back(std::move(*it));
            it = disk_index.erase(it);
        } else {
            ++it;
        }
    }
    std::error_code ec;
    for (const uint64_t key : disk_bad_ckpts) {
        auto o = disk_ckpts.find(key);
        if (o != disk_ckpts.end()) {
            fs::remove(ckpt_path(disk_dir, key), ec);
            disk_bytes -= std::min(disk_bytes, o->second.bytes);
            disk_ckpts.erase(o);
        }
    }
    for (const auto & k : disk_bad_runs) {
        auto o = disk_runs.find(k);
        if (o != disk_runs.end()) {
            fs::remove(run_path(disk_dir, k.first, k.second), ec);
            disk_bytes -= std::min(disk_bytes, o->second.bytes);
            disk_runs.erase(o);
        }
    }
    disk_bad_ckpts.clear();
    disk_bad_runs.clear();
}

void server_prompt_cache::disk_writer() {
    std::unique_lock<std::mutex> lk(disk_mu);
    // the main loop sleeps when every slot is idle; after each job it is woken once, so a save that found the queue
    // full is retried and the checkpoints that just landed can leave memory
    auto wake = [&]() {
        std::function<void()> cb = disk_on_written;
        lk.unlock();
        if (cb) {
            cb();
        }
        lk.lock();
    };
    while (true) {
        disk_retire_bad();
        disk_release_pins();
        // entries the main loop found unreadable: only this thread deletes
        while (!disk_doomed.empty()) {
            disk_drop(disk_doomed.back());
            disk_doomed.pop_back();
        }
        if (!disk_queue.empty()) {
            std::unique_ptr<disk_job> job = std::move(disk_queue.front());
            disk_queue.pop_front();
            disk_current = job.get();
            disk_cv.notify_all();                  // room in the queue
            lk.unlock();
            try {
                disk_write_runs(*job);
            } catch (const std::exception & ex) {
                SRV_ERR(" - disk cache: writing %d tokens failed: %s\n", (int) job->tokens.size(), ex.what());
            }
            lk.lock();
            disk_current = nullptr;
            disk_cv.notify_all();                  // a load may be waiting for exactly this entry
            lk.unlock();
            job.reset();                           // hundreds of MB: freed without the lock held
            lk.lock();
            wake();
            continue;
        }
        if (disk_stop) {
            break;                                 // saves drain first
        }
        disk_cv.wait(lk, [&] {
            return disk_stop || !disk_queue.empty() || !disk_doomed.empty() ||
                   !disk_bad_ckpts.empty() || !disk_bad_runs.empty() ||
                   !disk_unpin.empty() || !disk_unpin_runs.empty();
        });
    }
}

// the writer thread, without disk_mu: the runs this job carries (never one the store has), the checkpoints it
// carries, then the manifest naming all of them, then - under the lock - the index. A run the job names without its
// rows must still be in the store; if an eviction took it, the entry would have a hole, so the job goes and its
// slot starts its runs over.
void server_prompt_cache::disk_write_runs(disk_job & job) {
    const int64_t t0 = ggml_time_us();
    // the stored objects the job names are pinned until its entry counts them; however this ends, they go back
    // (the writer's next round releases them)
    struct pins_back {
        server_prompt_cache & c;
        disk_job & j;
        ~pins_back() {
            std::lock_guard<std::mutex> lk(c.disk_mu);
            c.disk_unpin.insert(c.disk_unpin.end(), j.pinned.begin(), j.pinned.end());
            c.disk_unpin_runs.insert(c.disk_unpin_runs.end(), j.run_pinned.begin(), j.run_pinned.end());
            j.pinned.clear();
            j.run_pinned.clear();
        }
    } release_on_exit { *this, job };
    if (job.run_data.empty()) {
        std::lock_guard<std::mutex> lk(disk_mu);
        if ((int64_t) disk_covered_locked(job.tokens, false) >= job.n_exact) {
            return;                                // no new runs, and the store brings back as much already
        }
    }

    spc_stage st;
    disk_entry e;
    e.tokens   = job.tokens.clone();
    e.ckpts    = job.ckpts;
    e.row_tgt  = job.row_tgt;
    e.row_dft  = job.row_dft;
    e.runs_tgt = job.runs_tgt;
    e.runs_dft = job.runs_dft;

    std::map<std::pair<uint64_t, uint64_t>, uint64_t> fresh_run;   // runs this job writes -> file bytes
    std::map<uint64_t, uint64_t>                      fresh_ckpt;  // checkpoints this job writes -> file bytes
    uint64_t wrote = 0, wrote_ckpt = 0;
    bool lost = false;
    const std::string path = manifest_path(disk_dir, e.tokens);

    // whatever this job wrote is unreferenced until the manifest is placed: on any failure, it goes
    auto discard = [&]() {
        std::error_code ec;
        fs::remove(path + ".part", ec);
        for (const auto & [k, b] : fresh_run) {
            fs::remove(run_path(disk_dir, k.first, k.second), ec);
        }
        for (const auto & [key, b] : fresh_ckpt) {
            fs::remove(ckpt_path(disk_dir, key), ec);
        }
    };

    bool ok = true;
    uint64_t mbytes = 0;
    try {
        for (const auto & d : job.run_data) {
            if (!ok) {
                break;
            }
            const std::pair<uint64_t, uint64_t> & k    = d.first;
            const std::vector<uint8_t> &          rows = d.second;
            bool have;
            {
                std::lock_guard<std::mutex> lk(disk_mu);
                have = disk_runs.count(k) > 0;
            }
            if (have || fresh_run.count(k)) {
                continue;
            }
            const disk_run_ref * r = nullptr;
            uint64_t row = 0;
            for (int b = 0; b < 2 && !r; ++b) {
                for (const auto & x : b == 0 ? e.runs_tgt : e.runs_dft) {
                    if (x.hi == k.first && x.lo == k.second) {
                        r   = &x;
                        row = b == 0 ? e.row_tgt : e.row_dft;
                        break;
                    }
                }
            }
            if (r == nullptr || rows.size() != row * (uint64_t) r->n) {
                SRV_ERR("%s", " - disk cache: a job's run does not match its list; not writing it\n");
                ok = false;
                break;
            }
            const std::string p = run_path(disk_dir, k.first, k.second);
            std::error_code ec;
            fs::create_directories(fs::path(p).parent_path(), ec);
            uint64_t b = 0;
            ok = write_part(st, p + ".part", [&](spc_out & o) { return write_run_body(o, row, *r, rows); }, b) && place(p + ".part", p);
            if (ok) {
                fresh_run[k] = b;
                wrote += b;
            }
        }

        for (const auto & d : job.ckpt_data) {
            if (!ok) {
                break;
            }
            const uint64_t                   key = d.first;
            const common_prompt_checkpoint & c   = d.second;
            bool have;
            {
                std::lock_guard<std::mutex> lk(disk_mu);
                have = disk_ckpts.count(key) > 0;
            }
            if (have || fresh_ckpt.count(key)) {
                continue;
            }
            const std::string p = ckpt_path(disk_dir, key);
            uint64_t b = 0;
            ok = write_part(st, p + ".part", [&](spc_out & o) { return write_ckpt_body(o, c); }, b) && place(p + ".part", p);
            if (ok) {
                fresh_ckpt[key] = b;
                wrote_ckpt += b;
            }
        }

        // only this thread deletes, so what is present now stays present until the index is updated below
        if (ok) {
            std::lock_guard<std::mutex> lk(disk_mu);
            for (const auto * runs : { &e.runs_tgt, &e.runs_dft }) {
                for (const auto & r : *runs) {
                    if (!disk_runs.count({ r.hi, r.lo }) && !fresh_run.count({ r.hi, r.lo })) {
                        lost = true;
                    }
                }
            }
            e.ckpts.erase(std::remove_if(e.ckpts.begin(), e.ckpts.end(), [&](const disk_ckpt_ref & k) {
                return !disk_ckpts.count(k.key) && !fresh_ckpt.count(k.key);
            }), e.ckpts.end());
            ok = !lost;
        }
        e.n_exact = runs_exact(e.tokens, e.runs_tgt, e.runs_dft, e.row_dft > 0, e.ckpts);
        ok = ok && write_manifest_part(st, path + ".part", e, disk_has_mtmd, mbytes);
    } catch (...) {
        discard();
        throw;
    }
    if (!ok) {
        discard();
        if (lost) {
            std::lock_guard<std::mutex> lk(disk_mu);
            disk_runs_lost.insert(job.id_slot);
            SRV_WRN(" - disk cache: runs of %d tokens went while they were being extended; slot %d writes them again\n",
                    (int) job.tokens.size(), job.id_slot);
        } else {
            SRV_WRN(" - disk cache: could not write %d tokens\n", (int) job.tokens.size());
        }
        return;
    }

    std::lock_guard<std::mutex> lk(disk_mu);
    if (!place(path + ".part", path)) {
        discard();
        return;
    }
    // the new entry's references first, so nothing it shares with an entry dropped below reaches zero
    for (const auto * runs : { &e.runs_tgt, &e.runs_dft }) {
        for (const auto & r : *runs) {
            auto & o = disk_runs[{ r.hi, r.lo }];
            const auto f = fresh_run.find({ r.hi, r.lo });
            if (o.bytes == 0 && f != fresh_run.end()) {
                o.bytes = f->second;
                disk_bytes += f->second;
            }
            o.refs++;
        }
    }
    for (const auto & [key, b] : fresh_ckpt) {
        auto & o = disk_ckpts[key];
        o.bytes = b;
        disk_bytes += b;
    }
    for (const auto & k : e.ckpts) {
        disk_ckpts[k.key].refs++;
    }
    // an entry at the same path was just replaced by this manifest: its references go, its file stays
    for (auto it = disk_index.begin(); it != disk_index.end(); ++it) {
        if (it->path == path) {
            disk_drop(*it, /* remove_file = */ false);
            disk_index.erase(it);
            break;
        }
    }
    // an entry this one extends goes, if this one brings back at least as much: a conversation written through
    // leaves one entry, the latest, whose runs are all of the earlier ones' and more
    for (auto it = disk_index.begin(); it != disk_index.end();) {
        if (it->tokens.get_common_prefix(e.tokens) == it->tokens.size() && e.n_exact >= it->n_exact) {
            disk_drop(*it);
            it = disk_index.erase(it);
        } else {
            ++it;
        }
    }
    e.path  = path;
    e.bytes = mbytes;
    e.order = ++disk_seq;
    disk_bytes += mbytes;
    const int     n_tokens = (int) e.tokens.size();
    const int64_t n_exact  = e.n_exact;
    const size_t  n_runs   = e.runs_tgt.size() + e.runs_dft.size();
    disk_index.push_back(std::move(e));
    disk_evict(path);

    SRV_INF(" - disk cache: wrote %d tokens in %.0f ms: %zu of %zu runs new (%.3f GiB), %zu checkpoints new (%.3f GiB); "
            "brings back %lld tokens - %zu entries, %.1f GiB on disk\n",
            n_tokens, (ggml_time_us() - t0) / 1000.0, fresh_run.size(), n_runs, wrote / GIB, fresh_ckpt.size(), wrote_ckpt / GIB,
            (long long) n_exact, disk_index.size(), disk_bytes / GIB);
}

// pick the entry, pin what it names up to the restore point, let the lock go (making room may save a conversation,
// which takes it), then cells, rows a run at a time - the next run read while this one goes in - and the recurrent
// state from the checkpoint at the restore point. Two runs in memory at most, never the state.
bool server_prompt_cache::load_runs(server_prompt & prompt, const server_tokens & tokens_new, llama_context * ctx_tgt, llama_context * ctx_dft,
                                    int32_t id_slot, float & f_keep_best, float & f_sim_best, const std::function<void(size_t)> & before_restore,
                                    ckpt_paged_map * paged_out, disk_runs_state * runs_out, bool & restored) {
    restored = false;
    const uint64_t row_tgt = llama_strix_kv_row_size(ctx_tgt);
    const uint64_t row_dft = ctx_dft ? llama_strix_kv_row_size(ctx_dft) : 0;
    if (disk_limit == 0 || row_tgt == 0 || (ctx_dft && row_dft == 0) || tokens_new.size() == 0 || paged_out == nullptr) {
        return false;
    }
    if (tokens_new.get_text_tokens().size() != tokens_new.size()) {
        return false;                              // media: never stored
    }

    server_tokens               tokens;            // the entry's, copied: the store can change once the lock goes
    std::string                 path;
    std::vector<disk_run_ref>   runs_tgt, runs_dft;
    std::vector<disk_ckpt_ref>  ckpts;             // the ones up to the restore point, pinned
    std::vector<std::pair<uint64_t, uint64_t>> run_pins;
    disk_ckpt_ref at {};
    int64_t n = 0;
    float keep = 0.0f;
    {
        std::unique_lock<std::mutex> lk(disk_mu);
        const float f_sim_base = f_sim_best;
        disk_cv.wait(lk, [&] { return !disk_in_flight(tokens_new, f_sim_base); });

        auto best = disk_index.end();
        for (auto it = disk_index.begin(); it != disk_index.end(); ++it) {
            if (it->row_tgt != row_tgt || (ctx_dft && it->row_dft != row_dft)) {
                continue;                          // another model's rows, or no draft rows for a server that drafts
            }
            const int64_t lcp = it->tokens.get_common_prefix(tokens_new);
            int64_t reach = std::min<int64_t>(lcp, runs_cover(it->runs_tgt));
            if (ctx_dft) {
                reach = std::min(reach, runs_cover(it->runs_dft));
            }
            const disk_ckpt_ref * k_best = nullptr;
            for (const auto & k : it->ckpts) {
                if (k.n_tokens > 0 && k.n_tokens <= reach && (!k_best || k.n_tokens > k_best->n_tokens) && disk_ckpts.count(k.key)) {
                    k_best = &k;
                }
            }
            if (!k_best) {
                continue;
            }
            // not a long entry dragged in for a short prefix, and it must beat what the slot or the RAM tier already
            // offers; among the rest, the one that skips the most tokens
            const float f_keep_cur = float(lcp) / it->tokens.size();
            const float f_sim_cur  = float(k_best->n_tokens) / tokens_new.size();
            if (f_keep_cur < 0.25f || f_sim_cur <= f_sim_best || k_best->n_tokens <= n) {
                continue;
            }
            n    = k_best->n_tokens;
            at   = *k_best;
            keep = f_keep_cur;
            best = it;
        }
        if (best == disk_index.end()) {
            return false;
        }
        tokens   = best->tokens.clone();
        path     = best->path;
        runs_tgt = best->runs_tgt;
        runs_dft = best->runs_dft;
        for (const auto & r : runs_tgt) {
            if (r.pos0 < n) {
                disk_runs[{ r.hi, r.lo }].refs++;
                run_pins.push_back({ r.hi, r.lo });
            }
        }
        if (ctx_dft) {
            for (const auto & r : runs_dft) {
                if (r.pos0 < n) {
                    disk_runs[{ r.hi, r.lo }].refs++;
                    run_pins.push_back({ r.hi, r.lo });
                }
            }
        }
        for (const auto & k : best->ckpts) {
            if (k.n_tokens <= n && disk_ckpts.count(k.key)) {
                disk_ckpts[k.key].refs++;
                ckpts.push_back(k);
            }
        }
        best->order = ++disk_seq;
        std::error_code ec;
        fs::last_write_time(best->path, fs::file_time_type::clock::now(), ec);   // the LRU order survives a restart
    }
    auto unpin = [&](bool ckpts_too) {
        std::lock_guard<std::mutex> lk(disk_mu);
        disk_unpin_runs.insert(disk_unpin_runs.end(), run_pins.begin(), run_pins.end());
        run_pins.clear();
        if (ckpts_too) {
            for (const auto & k : ckpts) {
                disk_unpin.push_back(k.key);
            }
            ckpts.clear();
        }
        disk_cv.notify_all();
    };

    const int64_t t0 = ggml_time_us();
    g_spc_cost = {};
    if (before_restore) {
        before_restore((size_t) n);
    }

    struct item {
        disk_run_ref    r;
        uint64_t        row;
        llama_context * ctx;
        int32_t         take;                      // positions of it that go in
    };
    std::vector<item> items;
    for (const auto & r : runs_tgt) {
        if (r.pos0 < n) {
            items.push_back({ r, row_tgt, ctx_tgt, (int32_t) std::min<int64_t>(r.n_use, n - r.pos0) });
        }
    }
    if (ctx_dft) {
        for (const auto & r : runs_dft) {
            if (r.pos0 < n) {
                items.push_back({ r, row_dft, ctx_dft, (int32_t) std::min<int64_t>(r.n_use, n - r.pos0) });
            }
        }
    }

    llama_tokens text = tokens.get_text_tokens();
    text.resize((size_t) n);
    std::string why;
    read_fault fault;
    bool ok = llama_strix_kv_alloc(ctx_tgt, id_slot, text.data(), (int32_t) n) &&
              (!ctx_dft || llama_strix_kv_alloc(ctx_dft, id_slot, text.data(), (int32_t) n));
    if (!ok) {
        why = "no cells for it";
    }
    size_t n_read = 0;
    std::vector<uint8_t> spec;
    try {
        spc_stage st[2];
        std::vector<uint8_t> buf[2];
        // a read runs on its own thread: its cost comes back with it
        struct read_res { bool ok; spc_read_cost cost; };
        auto read_one = [&](size_t i) {
            g_spc_cost = {};
            spc_file f(st[i % 2]);
            const bool r = read_run(f, run_path(disk_dir, items[i].r.hi, items[i].r.lo), items[i].r, items[i].row, buf[i % 2]);
            return read_res { r, g_spc_cost };
        };
        std::future<read_res> next;
        if (ok && !items.empty()) {
            next = std::async(std::launch::async, read_one, (size_t) 0);
        }
        for (size_t i = 0; ok && i < items.size(); ++i) {
            const read_res rr = next.get();
            g_spc_cost.alloc_us += rr.cost.alloc_us;
            g_spc_cost.read_us  += rr.cost.read_us;
            g_spc_cost.bytes    += rr.cost.bytes;
            if (!rr.ok) {
                ok = false;
                why = "missing or damaged run " + hex128(items[i].r.hi, items[i].r.lo);
                fault.run = true;
                fault.hi  = items[i].r.hi;
                fault.lo  = items[i].r.lo;
                break;
            }
            if (i + 1 < items.size()) {
                next = std::async(std::launch::async, read_one, i + 1);
            }
            ok = llama_strix_kv_set_rows(items[i].ctx, id_slot, items[i].r.pos0, items[i].take, buf[i % 2].data(), items[i].r.n);
            if (!ok) {
                why = "the cache took no rows";
            }
            // STRIX_V3_VERIFY: read a whole run back out of the cache and check it against the hash it is named by
            if (ok && items[i].take == items[i].r.n && getenv("STRIX_V3_VERIFY")) {
                std::vector<uint8_t> back((size_t) items[i].row * (size_t) items[i].r.n);
                const bool got = llama_strix_kv_get_rows(items[i].ctx, id_slot, items[i].r.pos0, items[i].r.n, back.data(), back.size());
                const XXH128_hash_t h = XXH3_128bits(back.data(), back.size());
                if (!got || h.high64 != items[i].r.hi || h.low64 != items[i].r.lo) {
                    SRV_WRN(" - disk cache: VERIFY run at %d (%d rows, %s) reads back different%s\n", items[i].r.pos0, items[i].r.n,
                            items[i].ctx == ctx_tgt ? "target" : "draft", got ? "" : " (no rows)");
                }
            }
            ++n_read;
        }
        if (next.valid()) {
            next.wait();                           // a read still going when the loop stopped
        }
        if (ok) {
            common_prompt_checkpoint c;
            spc_file f(st[0]);
            if (!read_ckpt(f, ckpt_path(disk_dir, at.key), at, c)) {
                ok = false;
                why = "missing or damaged checkpoint";
                fault.ckpt = true;
                fault.key  = at.key;
            } else {
                ok = llama_state_seq_set_data_ext(ctx_tgt, c.data_tgt.data(), c.data_tgt.size(), id_slot,
                                                  LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == c.data_tgt.size() &&
                     (!ctx_dft || c.data_dft.empty() ||
                      llama_state_seq_set_data_ext(ctx_dft, c.data_dft.data(), c.data_dft.size(), id_slot,
                                                   LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == c.data_dft.size());
                if (!ok) {
                    why = "the recurrent state did not go in";
                }
                spec = std::move(c.data_spec);
            }
        }
    } catch (const std::bad_alloc &) {
        ok = false;
        why = "no memory for a run";
    } catch (const std::exception & ex) {
        ok = false;
        why = ex.what();
    }

    if (!ok) {
        // the cells were taken: give them back, and blame what was at fault
        llama_memory_seq_rm(llama_get_memory(ctx_tgt), id_slot, -1, -1);
        if (ctx_dft) {
            llama_memory_seq_rm(llama_get_memory(ctx_dft), id_slot, -1, -1);
        }
        {
            std::lock_guard<std::mutex> lk(disk_mu);
            if (fault.run) {
                disk_bad_runs.push_back({ fault.hi, fault.lo });
            }
            if (fault.ckpt) {
                disk_bad_ckpts.push_back(fault.key);
            }
        }
        unpin(true);
        SRV_WRN(" - disk cache: could not bring %lld tokens back from %s (%s); processing the prompt instead\n",
                (long long) n, fs::path(path).filename().string().c_str(), why.c_str());
        return true;                               // attempted: the slot's own conversation is gone
    }

    // the conversation up to the restore point, with its checkpoints left on disk and pinned for the slot; the one at
    // the restore point carries its speculative state, for the slot to hand to the drafter
    prompt.tokens = tokens.clone();
    prompt.tokens.keep_first((size_t) n);
    prompt.checkpoints.clear();
    paged_out->clear();
    for (const auto & k : ckpts) {
        common_prompt_checkpoint c;
        c.n_tokens = k.n_tokens;
        c.pos_min  = k.pos_min;
        c.pos_max  = k.pos_max;
        prompt.checkpoints.push_back(std::move(c));
        (*paged_out)[{ k.n_tokens, k.pos_min, k.pos_max }] = k;
    }
    const size_t n_ckpts = ckpts.size();
    ckpts.clear();                                 // their pins went to paged_out
    if (runs_out) {
        auto cut = [n](const std::vector<disk_run_ref> & runs) {
            std::vector<disk_run_ref> res;
            for (auto r : runs) {
                if (r.pos0 >= n) {
                    break;
                }
                r.n_use = (int32_t) std::min<int64_t>(r.n_use, n - r.pos0);
                res.push_back(r);
            }
            return res;
        };
        runs_out->tgt    = cut(runs_tgt);
        runs_out->dft    = ctx_dft ? cut(runs_dft) : std::vector<disk_run_ref> {};
        runs_out->tokens = text;                   // the first n, which these runs were written for
        runs_out->spec   = std::move(spec);
    }
    unpin(false);

    f_keep_best = keep;
    f_sim_best  = float(n) / tokens_new.size();
    restored    = true;
    SRV_INF(" - disk cache: read %lld tokens into the cache, %.3f GiB in %.0f ms (file %.0f ms at %.0f MB/s; %zu runs, "
            "%zu checkpoints left on disk; f_keep = %.3f, f_sim = %.3f)\n",
            (long long) n, g_spc_cost.bytes / GIB, (ggml_time_us() - t0) / 1000.0, g_spc_cost.read_us / 1000.0,
            g_spc_cost.read_us ? g_spc_cost.bytes / 1048576.0 / (g_spc_cost.read_us / 1e6) : 0.0, n_read, n_ckpts, f_keep_best, f_sim_best);
    return true;
}
