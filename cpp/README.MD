Sistema de construcción meson:

Nota: dentro de sobel_photonicsens hay configuración.

```shell
sudo apt install build-essentials meson
meson build-rel --buildtype release
cd build-rel
ninja
./sobel-photonicsens img_in img_out
```