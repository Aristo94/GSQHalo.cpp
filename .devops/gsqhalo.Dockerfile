# GSQHalo.cpp for AMD Strix Halo (gfx1151), ROCm/HIP build.
#
# The toolchain is the one every measurement in docs/gsqhalo/README.md was made with: Fedora 44 and
# AMD's multi-arch ROCm 7.14 packages for gfx1151 (amdrocm-core-devel7.14-gfx1151).
#
#   podman build -f .devops/gsqhalo.Dockerfile --build-arg COMMIT=$(git rev-parse --short HEAD) -t gsqhalo .
#
# Run it with the GPU devices passed through, for example:
#
#   podman run --rm -it --device /dev/kfd --device /dev/dri --group-add keep-groups \
#     --security-opt seccomp=unconfined -v /path/to/models:/models:ro -p 8080:8080 \
#     gsqhalo llama-server -m /models/<model>.gguf --host 0.0.0.0 --port 8080 ...
ARG FEDORA_VERSION=44
FROM registry.fedoraproject.org/fedora:${FEDORA_VERSION}

ARG ROCM_VERSION=7.14
ARG ROCM_REPO=https://repo.amd.com/rocm/packages-multi-arch/rhel10/x86_64
ARG COMMIT=unknown
ARG JOBS=16

RUN printf '%s\n' \
      '[rocm]' "name=ROCm ${ROCM_VERSION}" "baseurl=${ROCM_REPO}" 'enabled=1' 'priority=50' 'gpgcheck=1' \
      'gpgkey=https://repo.amd.com/rocm/packages-multi-arch/gpg/rocm.gpg' > /etc/yum.repos.d/rocm.repo \
 && dnf -y --nodocs --setopt=install_weak_deps=False install \
      make gcc gcc-c++ cmake ninja-build libcurl-devel rdma-core-devel git-core python3 procps-ng \
      amdrocm-core-devel${ROCM_VERSION}-gfx1151 \
 && dnf clean all && rm -rf /var/cache/dnf/*

ENV ROCM_PATH=/opt/rocm \
    HIP_PATH=/opt/rocm \
    PATH=/opt/rocm/bin:/opt/rocm/core/bin:/opt/rocm/core/lib/llvm/bin:/usr/local/bin:/usr/bin \
    LD_LIBRARY_PATH=/opt/rocm/core/lib/rocm_sysdeps/lib:/opt/rocm/core/lib \
    ROCBLAS_USE_HIPBLASLT=1

LABEL org.opencontainers.image.source="https://github.com/Aristo94/GSQHalo.cpp" \
      org.opencontainers.image.revision="${COMMIT}" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.description="GSQHalo.cpp: llama.cpp for GSQ-quantized Qwen3.8-Flash-Next on AMD Strix Halo, with a persistent KV cache on SSD"

WORKDIR /opt/llama.cpp
COPY . /opt/llama.cpp
RUN cmake -S . -B build \
      -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151 -DCMAKE_BUILD_TYPE=Release \
      -DROCM_PATH=/opt/rocm -DHIP_PLATFORM=amd -DLLAMA_BUILD_TESTS=ON \
 && cmake --build build --config Release -- -j${JOBS} \
 && cmake --install build --config Release \
 && echo /usr/local/lib64 > /etc/ld.so.conf.d/usr-local-lib64.conf && ldconfig \
 && echo ${COMMIT} > /opt/llama.cpp/BUILD_COMMIT

CMD ["/bin/bash"]
