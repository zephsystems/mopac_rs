# Multi-stage Dockerfile for MOPAC_RS (CLI & Python bindings)
# Produces an optimized OCI container image for GitHub Container Registry (ghcr.io)

FROM rust:1.85-slim-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libvulkan-dev \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy workspace source files
COPY Cargo.toml Cargo.lock pyproject.toml README.md LICENSE ./
COPY crates ./crates

# Build release CLI binary
RUN cargo build --release --bin mopac && \
    strip target/release/mopac

# Build Python wheel using maturin
RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip maturin patchelf && \
    /opt/venv/bin/maturin build --release --out /build/dist

# Runtime stage
FROM debian:bookworm-slim AS runtime

LABEL org.opencontainers.image.title="MOPAC_RS" \
      org.opencontainers.image.description="Modern High-Performance Semi-Empirical Quantum Chemistry Engine in Rust" \
      org.opencontainers.image.url="https://github.com/zephsystems/mopac_rs" \
      org.opencontainers.image.source="https://github.com/zephsystems/mopac_rs" \
      org.opencontainers.image.licenses="Apache-2.0"

# Install minimal runtime dependencies: Vulkan loader, software driver fallback, and Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libvulkan1 \
    mesa-vulkan-drivers \
    python3 \
    python3-pip \
    python3-numpy \
    && rm -rf /var/lib/apt/lists/*

# Copy mopac CLI binary
COPY --from=builder /build/target/release/mopac /usr/local/bin/mopac

# Install python mopac_py wheel
COPY --from=builder /build/dist/*.whl /tmp/
RUN pip3 install --no-cache-dir --break-system-packages /tmp/*.whl && \
    rm -rf /tmp/*.whl

WORKDIR /data

# Default entrypoint runs mopac CLI, but allows overriding with bash or python3
ENTRYPOINT ["/usr/local/bin/mopac"]
CMD ["--help"]
