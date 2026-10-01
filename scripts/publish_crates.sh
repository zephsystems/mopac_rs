#!/usr/bin/env bash
set -euo pipefail

# Publish MOPAC_RS crates to crates.io in topological dependency order.
# Requires: cargo login or CARGO_REGISTRY_TOKEN environment variable.

publish_crate() {
    local crate_name="$1"
    shift
    echo "=== Packaging and verifying $crate_name ==="
    local output
    if output=$(cargo publish -p "$crate_name" "$@" 2>&1); then
        echo "$output"
        echo "Successfully published $crate_name to crates.io"
    else
        echo "$output"
        if echo "$output" | grep -iq "already exists"; then
            echo "Notice: $crate_name version already exists on crates.io index. Skipping upload."
        else
            echo "Error publishing $crate_name" >&2
            return 1
        fi
    fi
    echo "Waiting 20 seconds for crates.io index to settle..."
    sleep 20
}

publish_crate "mopac_core" "$@"
publish_crate "mopac_gpu" "$@"
publish_crate "mopac" "$@"

echo "=== All MOPAC_RS crates verified / published to crates.io ==="

