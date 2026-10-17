#!/usr/bin/env bash
set -euo pipefail

PKG_VER="${1:-0.1.4}"
PKG_VER="${PKG_VER#v}"
DEB_DIR="target/debian_pkg"

rm -rf "$DEB_DIR"
mkdir -p "$DEB_DIR/usr/bin"
mkdir -p "$DEB_DIR/usr/share/doc/mopac"
mkdir -p "$DEB_DIR/DEBIAN"

cp target/release/mopac "$DEB_DIR/usr/bin/mopac"
chmod 755 "$DEB_DIR/usr/bin/mopac"
cp LICENSE "$DEB_DIR/usr/share/doc/mopac/copyright"
cp README.md "$DEB_DIR/usr/share/doc/mopac/README.md"
chmod 644 "$DEB_DIR/usr/share/doc/mopac/copyright" "$DEB_DIR/usr/share/doc/mopac/README.md"

cat << EOF > "$DEB_DIR/DEBIAN/control"
Package: mopac
Version: $PKG_VER
Section: science
Priority: optional
Architecture: amd64
Depends: libc6, libvulkan1
Maintainer: zeph.sys <zeph.systems@gmail.com>
Homepage: https://github.com/zephsystems/mopac_rs
Description: Modern High-Performance Semi-Empirical Quantum Chemistry Engine in Rust
 MOPAC_RS is a pure Rust semi-empirical quantum chemistry engine providing
 canonical OpenMOPAC Fortran 90 parity for MNDO, AM1, PM3, RM1, PM6, PM7,
 MOZYME O(N), COSMO solvation, and Vulkan GPU acceleration.
EOF

dpkg-deb --build "$DEB_DIR" "mopac_${PKG_VER}_amd64.deb"
sha256sum "mopac_${PKG_VER}_amd64.deb" > "mopac_${PKG_VER}_amd64.deb.sha256"
echo "Successfully built mopac_${PKG_VER}_amd64.deb"
