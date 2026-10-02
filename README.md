<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/gsqhalo/banner-dark.svg">
    <img src="docs/gsqhalo/banner-light.svg" alt="GSQHalo.cpp" width="760">
  </picture>
</p>

<p align="center">
  <b>llama.cpp for the GSQ-RCO quantizations of Qwen3.8-Flash-Next on AMD Strix Halo</b>
</p>

<p align="center">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/GPU-gfx1151%20(Radeon%208060S)-E8590C" alt="GPU: gfx1151">
  <img src="https://img.shields.io/badge/backend-ROCm%207.14%20%2F%20HIP-F08C00" alt="Backend: ROCm/HIP">
  <a href="https://github.com/halo-box/strix-llama.cpp"><img src="https://img.shields.io/badge/fork%20of-halo--box%2Fstrix--llama.cpp-59636E" alt="Fork of halo-box/strix-llama.cpp"></a>
</p>

---

The [GSQ-RCO quantizations](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) of Qwen3.8-Flash-Next
fit the model into a 96 GB Strix Halo box. They use expert types such as IQ2_XXS, IQ2_S, IQ3_XXS and Q2_0, and their
hyper-connections are BF16. On the fork this builds on, those tensors ran on generic fallback kernels. MTP used up the
memory needed for a second long-context slot, and a server with two 256K slots crashed.

GSQHalo.cpp fixes those three problems for this model on this hardware, and adds a KV cache on SSD that survives slot
switches and restarts.

## Results

One machine: Ryzen AI Max+ 395, Radeon 8060S (`gfx1151`), 96 GB LPDDR5X. Model: GSQ-RCO IQ3_XXS (70.6 GiB).

**Kernels** (`llama-bench`, tokens/s):

| | halo-box/strix-llama.cpp | GSQHalo.cpp |
| --- | ---: | ---: |
| Prefill pp4096, empty context | 1019 | **1401** |
| Prefill pp4096, at 32K | 977 | **1310** |
| Prefill pp4096, at 128K | 892 | **1188** |
| Decode tg128, at 128K | 24.2 | 24.2 |
| Perplexity, 32 × 4096 tokens of wikitext | 3.9044 | 3.9037 |

Each side ran with its recommended flags: the base with `-lzm on -b 4096 -ub 4096`, GSQHalo with
`-lzm on-direct -b 32768 -ub 8192`. With the same flags (`-b/-ub 4096`), the first four kernel steps alone took pp4096
from 1003 to about 1350 t/s. [The docs](docs/gsqhalo/README.md#results) give every step with its own before/after.

**MTP and memory** (`llama-server`, the same build with and without the MTP commits):

| | Before | GSQHalo.cpp |
| --- | ---: | ---: |
| Draft compute buffers, 2 × 256K | 3809 MiB | **520 MiB** |
| Server RSS, one slot with MTP | 10.5 GiB | **4.9 GiB** |
| Prefill with MTP, 128K prompt | 1193 t/s | **1386 t/s** |
| 2 × 256K slots with MTP in 96 GB | ran out of memory | **runs, ≥ 8 GiB free** |

**SSD KV cache:** a 29.7K-token conversation that left its slot comes back from disk in 2.5 s. On the base,
re-prefilling it took 33.7 s.

## What's inside

- **GSQ prefill kernels** (ROCm/HIP):
  - fixed MoE tile selection for the low-bit experts;
  - slice decoders that write 8 weights per lane;
  - BF16 hyper-connection fusions and a dedicated HC-Down kernel;
  - a Q8_1 activation shared by several dense readers;
  - QSA attention that skips empty tiles;
  - per-layer-embedding reads overlapped with GPU work.
- **MTP that fits:** over the prompt, the draft head computes only its K/V. It reserves buffers for that graph, not for
  the whole context, and keeps one hidden-state row instead of three copies of all of them. Large prefill ubatches skip
  the recurrent rollback snapshots.
- **Multi-slot fix:** with a unified KV cache, the QSA kernels accepted at most 262 140 cells across *all* slots. Above
  that the server aborted. The kernels now share one limit (2^24) with the model's gate.
- **Persistent KV cache on SSD** (`--cache-dir`): the disk tier v3 of StrixLlama, ported to Linux with direct I/O and
  without mmap. Conversations are written while they grow, restored straight into the KV cells, and kept across
  restarts.

[docs/gsqhalo/README.md](docs/gsqhalo/README.md) lists every change with its commit, its off switch where there is one,
and its measurements.

## Quick start

```sh
git clone https://github.com/Aristo94/GSQHalo.cpp && cd GSQHalo.cpp
podman build -f .devops/gsqhalo.Dockerfile --build-arg COMMIT=$(git rev-parse --short HEAD) -t gsqhalo .
```

The container builds for `gfx1151` with Fedora 44 and AMD's ROCm 7.14 packages, the toolchain all numbers above come
from. A native build works as in upstream, with `-DGGML_HIP=ON -DGPU_TARGETS=gfx1151`.

One slot, 256K context, MTP (`mtp-head-Q8_0.gguf` is a GGUF of the model's MTP head):

```sh
llama-server -m Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
  -md mtp-head-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.3 \
  -c 262144 -ngl 999 -fa on -lm dio -lzm on-direct -t 4 -b 8192 -ub 8192 \
  --cache-ram 2048 -ctxcp 8 --jinja --reasoning off
```

The docs have a two-slot config with the SSD cache, along with the flags to avoid.

## Lineage and related forks

```
ggml-org/llama.cpp                 upstream
└─ halo-box/llama.cpp              close to upstream, adds features and speedups
   └─ halo-box/strix-llama.cpp     Strix Halo only; our base is 307c50d (PR #123)
      └─ GSQHalo.cpp               this repository: GSQ-RCO Qwen3.8-Flash-Next
```

- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp): upstream.
- [halo-box/llama.cpp](https://github.com/halo-box/llama.cpp) and
  [halo-box/strix-llama.cpp](https://github.com/halo-box/strix-llama.cpp): everything they add is still here, including
  the Vulkan and ROCm tuning for RDNA 3.5, the reasoning budget, ROCmFPx quants and speculative prefill. Their README as
  of our base is kept in [docs/halo-box/README.md](docs/halo-box/README.md).
- [StrixLlama / Rulith Inference](https://github.com/rulith-dev/rulith-inference): the origin of the disk tier, a
  Windows-first stack for large MoE models on one Strix Halo machine.
- [EngramHalo.cpp](https://github.com/Aristo94/EngramHalo.cpp): a sibling fork of upstream llama.cpp for
  Qwen3.8-Flash-Next on Strix Halo, with the engram table on SSD.

GitHub links only one fork per account in a fork network, so this repository is not shown as a GitHub fork. The full
upstream history is included.

## Documentation

- [docs/gsqhalo/README.md](docs/gsqhalo/README.md): every change with its commit and switch, all measurements, server
  configs, known limits.
- [docs/halo-box/README.md](docs/halo-box/README.md): what halo-box/strix-llama.cpp changes compared to upstream, and how
  to set up a Strix Halo machine (GTT size, ROCm notes).
- [docs/build.md](docs/build.md) and [tools/server/README.md](tools/server/README.md): the upstream build and server
  documentation.

## Credits

Built on the work of the [llama.cpp](https://github.com/ggml-org/llama.cpp) contributors, [Halo Box](https://github.com/halo-box)
and Victor Shaw (StrixLlama, MIT; the disk tier code keeps its copyright notice). The quantizations are by
[ISTA DASLab](https://huggingface.co/ISTA-DASLab), and the model is by the Qwen team. GSQHalo.cpp is an independent
project, not affiliated with any of them.

Licensed under the [MIT License](LICENSE), like llama.cpp.
