# GSQHalo.cpp

GSQHalo.cpp is a fork of [halo-box/strix-llama.cpp](https://github.com/halo-box/strix-llama.cpp) for one model family on
one machine: the [GSQ-RCO quantizations](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) of
Qwen3.8-Flash-Next on AMD Strix Halo (Ryzen AI Max+ 395, Radeon 8060S, `gfx1151`). It adds:

- **GSQ prefill kernels (ROCm/HIP).** The GSQ mixes use low-bit expert types (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, Q2_0) and
  BF16 hyper-connections. On the base, those ran on generic fallback paths.
- **MTP that fits.** The MTP draft head no longer runs its full block over the prompt. It no longer reserves buffers that
  grow with the context, and it no longer keeps three copies of the target's hidden states. Two 256K slots with MTP now
  fit in 96 GB.
- **A fix for multi-slot unified KV caches.** A server with several slots aborted as soon as the slots together held more
  than 262 140 cells.
- **A persistent KV cache on SSD.** This is the disk tier v3 of StrixLlama, ported to Linux without mmap. Conversations
  survive slot switches and server restarts.

Everything else is halo-box/strix-llama.cpp at `307c50d` (PR #123). Its README as of that commit is in
[docs/halo-box/README.md](../halo-box/README.md).

## Lineage

```
GSQHalo.cpp                       this repository
└─ halo-box/strix-llama.cpp       Strix Halo only fork, base 307c50d (PR #123)
   └─ halo-box/llama.cpp          close-to-upstream fork
      └─ ggml-org/llama.cpp       upstream
+  disk tier v3 from StrixLlama   rulith-dev/strixllama, now rulith-dev/rulith-inference (MIT, Victor Shaw)
```

GitHub allows only one fork per account in the llama.cpp fork network, so this repository is not linked as a GitHub fork.
The full upstream history is included, and the GSQHalo commits sit on top of `307c50d`.

## What the commits do

**GSQ prefill kernels** (ROCm/HIP, measured on `gfx1151` only)

| Change | Commit | Flag / switch | What it does |
| --- | --- | --- | --- |
| MoE tiles for low-bit experts | `8fbe99a` | `GGML_LOWBIT_COMPACT_J=80\|96` overrides J | Fixes `ncols_opt` staying 0 in the paired gate/up path. Before, every low-bit expert matmul ran with J=16 tiles: 256 × 512 almost empty blocks. The low-bit types also get the RDNA3.5 compact MoE tiles. |
| MMB slice decoders | `21340f9` | `MMB_ROUTED_LOWBIT=0`, or a type list such as `iq2_s,q2_0` | Each lane decodes exactly its 8 weights of Q2_0, IQ2_XXS, IQ2_XS, IQ2_S or IQ3_XXS and writes them with one 16-byte store, bit-identical to `dequantize_*`. These decoders are used only in the routed MoE path; IQ3_S's fast decoder is re-enabled there too. |
| BF16 hyper-connection paths | `cf23d5e` | `MMB_HCD_TILE=0` (narrow HC-Down tiles off) | The gate-mix fusion, the narrow HC-Down tiles and the inject fusion now also accept the BF16 HC weights of the GSQ mixes, not only Q8_0/F32. |
| Register prefetch and LDS grids | `7ef8134` | | The routed GLU kernel's IQ3_S load path now also serves the IQ2/IQ3_XXS types. IQ2_S grid codes are packed at 2 bits, which keeps 3 blocks per WGP. The BF16 gate mix loads without spills. Bit-identical. |
| Dedicated HC-Down kernel | `ba31c23` | `MMB_HCD_V2=0` restores the previous tiles | `[10240 -> 320]` with 256-wide K pieces, contiguous loads and no scratch: 188 → 129 ms per ubatch. Bit-identical. |
| Shared Q8_1 activations | `6081606` | `GGML_CUDA_MMQ_Y_REUSE=0` | When several dense MMQ readers read the same activation, it is quantized to Q8_1 once instead of once per reader. Bit-identical. |
| Dense MMQ J=128 tiles | `fea432e` | | Two warps split the columns of a J=128 stripe, which leaves registers for the RDNA3.5 Y-tile prefetch (3 blocks per WGP, no spills). Bit-identical. |
| QSA prefill: pack threshold | `a1bd3d0` | `QSA_DIRECT=0\|1` forces a path | Full ubatches use the packed K/V path up to 262K keys instead of switching to the slower direct gather at 128K. |
| QSA prefill: tile skip and class sort | `19c97f4` | `QSA_TILESKIP=0`, `QSA_CSORT=0` | Skips 16-row tiles that have no selected key in a chunk. Sorts the union blocks by tile class, so that more tiles can be skipped. |
| QSA statistics | `3d72923` | `QSA_STATS=1` (with `GGML_CUDA_DISABLE_GRAPHS=1`) | Logs the union sizes, active tile shares and chunk distribution per call. Diagnostics only. |
| Overlapped PLE gather | `e5171c1` | `LLAMA_INPUT_TIMING=1` logs input timings | The per-layer-embedding rows of the later ubatches are read while the first ubatch computes. Before, the reads competed with the first ubatch's cold gather. |

**MTP memory and prefill**

| Change | Commit | Flag / switch | What it does |
| --- | --- | --- | --- |
| No rollback snapshots in prefill | `5bc1d79` | `LLAMA_RS_SNAPSHOT_MAX` (default 255, `0` = old behaviour) | Ubatches larger than a verify batch write no recurrent-state snapshots. This keeps the chunked Gated DeltaNet prefill path with MTP on. |
| K/V-only draft catch-up | `ecd6267` | `LLAMA_MTP_KV_ONLY` (default 64, `0` = full block), `LLAMA_MTP_EH_GEMM=0` | During prompt processing, the draft head computes only its K/V and indexer keys. It no longer runs attention, MoE or the LM head over every prompt token. Its buffers are reserved for that graph. |
| Draft ubatch cap, catch-up in pieces | `742f8f4` | `STRIX_SPEC_DRAFT_UBATCH` (default 2048, `0` = inherit), `LLAMA_MTP_TIMING=1` | Only the last hidden-state row of a prompt batch is kept. The catch-up runs in draft-ubatch pieces, and the draft context's micro-batch is capped. |

**Multi-slot unified KV over 262 140 cells**

| Change | Commit | Flag / switch | What it does |
| --- | --- | --- | --- |
| One limit for gate and kernels | `8143a58` | `GGML_FLASH_ATTN_EXT_TOP_K_MAX_KV` (2^24) in `ggml.h` | The model's gate allowed the maskless selected-key path up to 2^24 keys, but the QSA kernels accepted at most 262 140. With `-kvu`, the limit counts the occupied cells of all slots together, so `fattn.cu` aborted with "maskless selected-key attention was not taken by a QSA kernel". The gate and the kernels now derive from one constant. |
| 16-bit blocks where they fit | `bf0890d` | | Below 262 140 keys the kernels keep 16-bit block ids, instruction-identical to before. Above that limit they switch to 32-bit block ids. |

**Persistent KV cache on SSD** (disk tier v3, ported from StrixLlama)

| Change | Commit | Flag / switch | What it does |
| --- | --- | --- | --- |
| Row API | `e78a6f8` | `llama_strix_kv_*` in `llama.h` | Reads and writes a sequence's attention and indexer rows by position, so conversations move to disk in runs instead of as one state blob. |
| Store on disk | `4d208ba` | `--cache-dir PATH`, `--cache-dir-max MiB` (default -1, no limit), `--cache-run N` (4096), `--cache-ckpt-step N` (32768) | A content-addressed store of runs and checkpoints, with a single writer thread, pins and LRU eviction. On Linux, files are written with O_DIRECT where the file system supports it (`STRIX_DISK_DIRECT=0` turns this off), and nothing is mapped. The file formats match StrixLlama byte for byte. |
| Slot integration | `435a590` | `POST /strix/persist`; with `--cache-dir`, `--cache-ram 0` keeps the prompt cache for the store only | Full runs are written during prefill. A conversation leaving its slot is completed on disk and restored straight into the KV cells. Conversations under 1024 tokens are not stored. |
| Restore before write | `713ff4f` | | The writer pauses while a restore is reading, at most 10 s per object. |
| State hash | `761a95d` | `STRIX_STATE_HASH=1` | Logs a hash of the state at task start, after every prompt batch and when a conversation leaves its slot. Debugging only. |
| Coalesced device reads | `a51ae19` | | A state save reads neighbouring pieces of a tensor in one call instead of once per cell range. |

## Results

All numbers come from one machine, a GMKtec EVO-X2:

- **Hardware:** AMD Ryzen AI Max+ 395, Radeon 8060S (`gfx1151`, 40 CUs), 96 GB LPDDR5X unified memory, NVMe system disk.
- **Toolchain:** Fedora 44 container with ROCm 7.14 (`.devops/gsqhalo.Dockerfile`) and `ROCBLAS_USE_HIPBLASLT=1`.
- **Model:** [Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF),
  a 70.6 GiB file. The 26.8 GiB per-layer-embedding table stays on the NVMe and is read lazily (`-lzm on-direct`).
- **Flags:** `-ngl 99 -fa on -ctk f16 -ctv f16 -lm dio -t 4` unless noted.

On this machine, absolute numbers drift by ±2–3 % between sessions, and single samples occasionally drop by 9–12 %.
Every before/after pair below was therefore measured in the same session, alternating A and B. `llama-bench` uses
random tokens, which is the worst case for the embedding table on disk.

### Prefill, step by step

`llama-bench`, tokens/s. Each row was measured against its predecessor.

| Step | Commit | Setup | Before | After |
| --- | --- | --- | ---: | ---: |
| MoE tiles for low-bit experts | `8fbe99a` | pp4096, `-b/-ub 4096` | 1002.6 | 1084.3 (+8.1 %) |
| MMB slice decoders | `21340f9` | pp4096, `-b/-ub 4096` | 1084.3 | 1248.3 (+15.1 %) |
| BF16 hyper-connection paths | `cf23d5e` | pp4096, `-b/-ub 4096` | 1248.3 | 1326.5 (+6.3 %) |
| Register prefetch, LDS grids | `7ef8134` | pp4096, `-b/-ub 4096` | 1299.7 / 1309.2 | 1349.4 (+3.4 %) |
| Overlapped PLE gather | `e5171c1` | pp32768, `-lzm on-direct -ub 8192`, `-b 8192` vs `-b 32768` | 1373 / 1386 / 1385 | 1433 / 1426 / 1446 (+3.9 %) |
| Shared Q8_1, J=128 tiles | `6081606`, `fea432e` | pp32768, `-lzm on-direct -b 32768 -ub 8192` | 1442.9 | 1460.5 (+1.2 %) |
| QSA packed path at 128K | `a1bd3d0` | pp4096 @ 131072, direct vs packed | 1185.8 | 1226.6 (+3.4 %) |
| QSA tile skip and class sort | `19c97f4` | pp4096 @ 32768, `-lzm on-direct -b 32768 -ub 8192` | 1273.9 / 1288.0 | 1276.7 / 1307.9 (+0.9 %) |
| Dedicated HC-Down kernel | `ba31c23` | pp4096 @ 0, median of 9 | 1374.1 | 1408.5 (+2.5 %) |
| | | pp4096 @ 32768, median of 9 | 1312.3 | 1330.9 (+1.4 %) |

The base is `307c50d`. `llama-bench` does not touch the disk tier, so the disk tier commits do not change these numbers.

### Throughput over context depth

The GSQHalo numbers were measured on `19c97f4` with `-lzm on-direct -b 32768 -ub 8192`, 2 repetitions (± is the spread).
The commits after it either don't change speed without MTP (bit-identical) or are the HC-Down kernel above. The base
numbers were measured on `307c50d` plus the disk tier, with its default flags `-lzm on -b 4096 -ub 4096` and 1 repetition.

| Depth | pp512 | pp4096 | tg128 | pp4096, base | tg128, base |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 873 ± 68 | 1401 ± 11 | 26.7 | 1019 | 26.8 |
| 8192 | 886 ± 7 | 1339 ± 10 | 24.9 | | |
| 16384 | 881 ± 6 | 1324 ± 8 | 24.8 | | |
| 32768 | 869 ± 2 | 1310 ± 4 | 24.7 | 977 | 23.4 |
| 65536 | 846 ± 11 | 1288 ± 0 | 24.1 | | |
| 131072 | 646 ± 219 | 1188 ± 36 | 23.6 | 892 | 24.2 |

A second decode check at 131072, in two alternating rounds, gave the same speed for the base and GSQHalo
(24.15 / 24.33 vs 24.08 / 24.29 t/s).

### llama-server with MTP

One slot, real text (C++ source and prose), 256 output tokens, temperature 0. The server ran with
`-lzm on-direct -b 32768 -ub 8192 -c 139264`, and MTP used `--spec-draft-n-max 3 --spec-draft-p-min 0.3`.
Measured on `bf0890d`, before the HC-Down kernel.

| Prompt | Prefill, MTP off | Prefill, MTP on | Prefill, MTP on, before `5bc1d79`–`742f8f4` | Decode, MTP off | Decode, MTP on |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32K code | 1465 | 1297 | 1185 | 25.2 | 43.6 |
| 32K prose | 1447 | 1386 | 1204 | 25.1 | 39.2 |
| 128K code | 1446 | 1386 | 1193 | 24.5 | 44.2 |
| 128K prose | 1457 | 1383 | 1198 | 24.3 | 37.8 |

The output text is identical with every draft-ubatch cap.

### MTP memory

| | Before `5bc1d79`–`742f8f4` | After |
| --- | ---: | ---: |
| Draft compute buffers, GPU + host, `-c 524288 -np 2 -kvu` | 3403 + 406 MiB | 420 + 100 MiB |
| Draft compute buffers, GPU + host, `-c 139264` | 2739 + 402 MiB | 420 + 100 MiB |
| MTP overhead in GTT after load, 2×256K | 8.4 GiB | 5.5 GiB |
| MTP overhead in GTT after load, 139K | 6.6 GiB | 4.3 GiB |
| Server RSS / anonymous, max, 1 slot, MTP on | 10.3–10.5 / 7.1–7.3 GiB | 4.9 / 2.0 GiB |

Without MTP, the same server uses 2.5 / 1.6 GiB. The hidden-state buffer of the target is `n_batch` × 40 KB of pinned
host memory. With MTP on, `-b 8192 -ub 8192` gives the same prefill as `-b 32768` and uses about 1 GiB less RSS.

### Two slots of 256K

Server flags: `-c 524288 -np 2 -kvu --kv-unified-per-slot 262144 -b 32768 -ub 8192 --cache-ram 0`. A 250K and a 32K
prompt were sent at the same time.

| | MTP off | MTP on |
| --- | ---: | ---: |
| GTT after load / peak | 63.6 / 67.7 GiB | 69.1 / 73.4 GiB |
| Lowest MemAvailable | 16 GiB | 8 GiB |
| Prefill, 250K slot | 1113 t/s | 1067 t/s |
| Decode, both slots at the same time | 12.6 + 13.4 t/s | 14.1 + 16.6 t/s |
| Decode with more than 262 144 occupied cells (80K slot) | 14.8 t/s | 21.5 t/s |

Before `5bc1d79`–`bf0890d`, the MTP-off run aborted at `n_kv` 262 400 (QSA limit), and the MTP-on run dropped below
4 GiB available even at `-ub 4096`. Now the generated text is identical with and without MTP.

### Quality

Perplexity over 32 × 4096 tokens of wikitext, base vs `19c97f4`, paired per chunk:

| Model | PPL, base | PPL, GSQHalo | Ratio, paired | Chunks better / worse |
| --- | ---: | ---: | ---: | ---: |
| GSQ-RCO IQ3_XXS | 3.9044 ± 0.0324 | 3.9037 ± 0.0324 | 0.99982 ± 0.00059 | 17 / 15 |
| GSQ-RCO IQ3_S | 3.8145 ± 0.0316 | 3.8131 ± 0.0316 | 0.99963 ± 0.00085 | 13 / 19 |

Most kernel changes are bit-identical (mean KLD exactly 0 against their predecessor). Two are not:

- The BF16 inject fusion (`cf23d5e`) changes rounding.
- The QSA class sort (`19c97f4`, `QSA_CSORT=1`) changes the block order in the online softmax, giving a KLD of 0.0086.

On this model, any rounding change lands near KLD 0.010, because top-k choices in QSA and in the MoE router flip.

### SSD prompt cache

Server with 2 slots and MTP, ext4 on a USB SSD (~600 MB/s):

- **Restore after a slot switch:** a 29.7K-token conversation came back in 2.5 s, against 33.7 s for a fresh prefill
  (measured before the GSQ kernels).
- **Restore after a server restart:** 31.8K tokens in 2.1 s.
- **Soak test:** two 89K-token conversations alternated in one slot for 10 rounds. Each restore took 5.4–5.5 s and the
  median switch 6.2 s. After a persist and a restart, 22/22 answers were correct.
- **Oracle:** a restore after eviction or restart is bit-identical to the reference, with and without MTP.

## Quick start

**Container** (Podman or Docker; build with the repository root as the context):

```sh
podman build -f .devops/gsqhalo.Dockerfile --build-arg COMMIT=$(git rev-parse --short HEAD) -t gsqhalo .
```

The image builds for `gfx1151` with Fedora 44 and AMD's ROCm 7.14 packages, and installs the binaries to `/usr/local/bin`.
It also sets `ROCBLAS_USE_HIPBLASLT=1`.

**From source** (ROCm installed, `gfx1151` targeted explicitly):

```sh
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
    cmake -B build -DGGML_HIP=ON -DGPU_TARGETS=gfx1151 -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

**Models:**

- The GSQ-RCO GGUFs from [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF).
- For MTP, a GGUF of the model's MTP head (`mtp-head-Q8_0.gguf` below). All measurements used a Q8_0 head.

The halo-box notes on GTT size and kernel parameters in [docs/halo-box/README.md](../halo-box/README.md) apply too.

## Recommended server configs

**One slot, 256K, MTP:**

```sh
llama-server \
  -m Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
  -md mtp-head-Q8_0.gguf \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.3 \
  -c 262144 -ngl 999 -fa on -lm dio -lzm on-direct -t 4 \
  -b 8192 -ub 8192 --cache-ram 2048 -ctxcp 8 \
  --jinja --reasoning off
```

**Two slots of 256K, unified KV, idle conversations on SSD:**

```sh
llama-server \
  -m Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
  -md mtp-head-Q8_0.gguf \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.3 \
  -c 524288 -np 2 -kvu --kv-unified-per-slot 262144 \
  -ngl 999 -fa on -lm dio -lzm on-direct -t 4 -b 4096 -ub 4096 \
  --cache-ram 0 -ctxcp 8 --no-cache-idle-slots \
  --cache-dir /mnt/ssd/llama-cache --cache-dir-max 81920 \
  --jinja --reasoning off
```

To save all idle conversations to disk before stopping the server, send `curl -X POST http://127.0.0.1:8080/strix/persist`.

Tips for these configs:

- **`-lzm on-direct`** reads the embedding table with direct I/O. Never use `-lzm off`: it loads the 26.8 GiB table into
  RAM.
- **Keep `-b 4096` on multi-slot servers.** With `-b 32768`, a long prompt in one slot stalls the decode of the other
  slots until it finishes.
- **One slot with MTP:** `-b 8192 -ub 8192` gives the same prefill as `-b 32768` and uses less pinned host memory.

## Known limits

- **One machine.** Everything was tuned and measured on `gfx1151` with 96 GB. The ROCm/HIP changes sit behind the existing
  halo-box architecture and type checks (MMB runs on RDNA3.5 only), but no other GPU has been tested. The Vulkan backend is unchanged.
- **One model family.** The kernels target the GSQ-RCO mixes of Qwen3.8-Flash-Next. Other models take the halo-box paths.
- **`test-backend-ops -o MMB_QUANT_HC`:** 21 quantized types fail with `ERR=inf`. The failures are already present on the
  base `307c50d` and are not caused by this fork. All other HC, QSA, MoE and MUL_MAT cases pass.
- **Not bit-identical to the base,** because of the BF16 inject fusion and the QSA class sort (see Quality). `QSA_CSORT=0`
  restores the old block order.
- **The disk tier writes through on every full run** (4096 positions by default). Restore speed is bounded by the SSD;
  from a USB SSD at ~600 MB/s, an 89K-token conversation takes about 5.5 s.
- **The multi-slot QSA fix has not been sent upstream yet.**

## Credits

- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) and its contributors.
- [Halo Box](https://github.com/halo-box) for [halo-box/llama.cpp](https://github.com/halo-box/llama.cpp) and
  [halo-box/strix-llama.cpp](https://github.com/halo-box/strix-llama.cpp). All Strix Halo kernels this fork builds on
  come from there.
- Victor Shaw for StrixLlama ([rulith-dev/rulith-inference](https://github.com/rulith-dev/rulith-inference)), whose
  disk tier v3 is ported here under the MIT license (Portions Copyright (c) 2026 Victor Shaw).
- [ISTA DASLab](https://huggingface.co/ISTA-DASLab) for the GSQ-RCO quantizations and the Qwen team for Qwen3.8-Flash-Next.

GSQHalo.cpp is an independent project. It is not affiliated with ISTA DASLab, Halo Box, Rulith or the Qwen team.
