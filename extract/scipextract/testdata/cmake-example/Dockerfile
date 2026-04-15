# Builds a minimal CMake project and runs scip-clang against it to produce
# a SCIP index. Pinned to linux/amd64 because scip-clang only ships x86_64
# and arm64 Linux binaries; this Dockerfile uses the x86_64 release.
FROM --platform=linux/amd64 ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        clang \
        cmake \
        curl \
        make \
    && rm -rf /var/lib/apt/lists/*

ARG SCIP_CLANG_VERSION=v0.4.0
RUN curl -fsSL -o /usr/local/bin/scip-clang \
        "https://github.com/sourcegraph/scip-clang/releases/download/${SCIP_CLANG_VERSION}/scip-clang-x86_64-linux" \
    && chmod +x /usr/local/bin/scip-clang

WORKDIR /project

# The entrypoint expects the source tree bind-mounted at /project.
# It writes the generated index to /project/index.scip.
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
