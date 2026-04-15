# Indexes a small Cargo workspace with rust-analyzer's SCIP emitter.
# rust-analyzer is installed via rustup as a component of the pinned
# stable toolchain so the fixture is reproducible without chasing
# release binary URLs. No platform pin — the rust image is multi-arch
# and we want to avoid QEMU emulation when running on arm64 hosts.
FROM rust:1-bookworm

RUN rustup component add rust-analyzer

WORKDIR /project

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
