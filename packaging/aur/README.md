# Arch User Repository (AUR) Publishing Guide for `mopac-rs-bin`

This directory contains the [`PKGBUILD`](./PKGBUILD) and [`.SRCINFO`](./.SRCINFO) definitions for distributing the precompiled Linux `x86_64` binary of **MOPAC_RS** on Arch Linux / Manjaro / EndeavourOS via AUR.

---

## 1. Prerequisites

1. A registered account on [aur.archlinux.org](https://aur.archlinux.org/register).
2. Your SSH public key configured in your AUR account profile.
3. Wait until the GitHub Release workflow has built and published `mopac-linux-x86_64.tar.gz` for the target release tag, then verify its SHA-256 digest matches the first entry in `sha256sums` inside `PKGBUILD` and `.SRCINFO` before pushing to AUR.

---

## 2. Publishing or Updating the AUR Package

```bash
# 1. Clone the official AUR package repository
git clone ssh://aur@aur.archlinux.org/mopac-rs-bin.git /tmp/mopac-rs-bin

# 2. Copy the prepared packaging definitions
cp packaging/aur/PKGBUILD packaging/aur/.SRCINFO /tmp/mopac-rs-bin/

# 3. Verify checksums, commit, and push
cd /tmp/mopac-rs-bin
updpkgsums && makepkg --printsrcinfo > .SRCINFO
git add PKGBUILD .SRCINFO
git commit -m "release: v0.2.0 — mopac-rs-bin"
git push origin master
```

---

## 3. User Installation on Arch Linux

Once published, users can install `mopac-rs-bin` using any standard AUR helper or `makepkg`:

```bash
# With yay
yay -S mopac-rs-bin

# With paru
paru -S mopac-rs-bin

# Or manually with makepkg
git clone https://aur.archlinux.org/mopac-rs-bin.git
cd mopac-rs-bin
makepkg -si
```
