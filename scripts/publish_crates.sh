#!/usr/bin/env bash
set -euo pipefail

# Publish MOPAC_RS crates to crates.io in topological dependency order.
# Requires: cargo login or CARGO_REGISTRY_TOKEN environment variable.

echo "=== Packaging and verifying mopac_core ==="
cargo publish -p mopac_core "$@"

echo "Waiting 30 seconds for crates.io index to update with mopac_core..."
sleep 30

echo "=== Packaging and verifying mopac_gpu ==="
cargo publish -p mopac_gpu "$@"

echo "Waiting 30 seconds for crates.io index to update with mopac_gpu..."
sleep 30

echo "=== Packaging and verifying mopac (CLI) ==="
cargo publish -p mopac "$@"

echo "=== All MOPAC_RS crates successfully published to crates.io ==="
