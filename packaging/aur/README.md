# Guía de Publicación en Arch User Repository (AUR) para `mopac-rs-bin`

Este directorio contiene el [`PKGBUILD`](./PKGBUILD) y [`.SRCINFO`](./.SRCINFO) oficiales y validados empíricamente con `makepkg` para distribuir el binario precompilado de **MOPAC_RS** en Arch Linux / Manjaro / EndeavourOS.

---

## 1. Requisitos previos
1. Contar con una cuenta registrada en [aur.archlinux.org](https://aur.archlinux.org/register).
2. Tener configurada tu clave pública SSH en tu perfil de AUR (`https://aur.archlinux.org/account/<tu_usuario>/`).

---

## 2. Publicación inicial o actualización en AUR

Ejecuta desde tu terminal los siguientes pasos:

```bash
# 1. Clona el repositorio oficial de tu paquete en AUR (si es la primera vez, se creará vacío automáticamente):
git clone ssh://aur@aur.archlinux.org/mopac-rs-bin.git /tmp/mopac-rs-bin

# 2. Copia los archivos preparados a ese repositorio:
cp packaging/aur/PKGBUILD packaging/aur/.SRCINFO /tmp/mopac-rs-bin/

# 3. Entra al repositorio, haz commit y push:
cd /tmp/mopac-rs-bin
git add PKGBUILD .SRCINFO
git commit -m "release: v0.1.4 — initial release of mopac-rs-bin"
git push origin master
```

---

## 3. Instalación para los usuarios de Arch Linux

Una vez subido, cualquier usuario de Arch Linux puede instalarlo con su asistente AUR preferido:

```bash
# Con yay
yay -S mopac-rs-bin

# Con paru
paru -S mopac-rs-bin

# O manualmente con makepkg
git clone https://aur.archlinux.org/mopac-rs-bin.git
cd mopac-rs-bin
makepkg -si
```
