# dowp-runtimes

Compilaciones de dependencias opcionales de [DowP](https://github.com/MarckDP/DowP) que no
tienen un paquete oficial para todo el hardware. Hoy: **whisper.cpp** (`whisper-cli`), el
motor de transcripción local de DowP.

Todo se compila en GitHub Actions a partir del **código oficial sin modificar**, y cada
paquete se publica con su **SHA-256**. DowP solo descarga paquetes cuyo hash coincide.

## whisper.cpp

Los paquetes oficiales de whisper.cpp ya cubren la CPU (Windows y Linux) y NVIDIA con CUDA
en Windows; DowP los usa directamente. Este repositorio compila el resto:

| Paquete | Para |
|---|---|
| `windows-x64-vulkan` | GPU **AMD** e **Intel** (Arc e integradas) en Windows; también funciona con NVIDIA |
| `macos-arm64-metal` | Mac con **Apple Silicon** (M1 en adelante) |
| `macos-x64-cpu` | Mac con **Intel** (CPU con Accelerate) |
| `linux-x64-vulkan` | GPU AMD, Intel y NVIDIA en **Linux** (necesita `libvulkan1`, que trae el driver) |
| `windows-x64-cuda` | NVIDIA **RTX serie 50** si el paquete oficial (CUDA 12.4) no funciona; opcional |

### Cómo se compila

1. **Actions → whisper.cpp → Run workflow.**
2. Elegir la versión (`latest` = la última estable de whisper.cpp) y los destinos.
3. El workflow compila cada destino en su máquina, lo **prueba transcribiendo** el audio de
   ejemplo de whisper.cpp y, si se marcó *publish*, crea una release
   `whisper-<versión>-r<número>` con:
   - los paquetes (`.zip` en Windows, `.tar.gz` en macOS y Linux);
   - `SHA256SUMS.txt`;
   - `manifest.json`, con tamaño, hash, sistema, arquitectura y backend de cada paquete.

Las máquinas de GitHub **no tienen GPU utilizable**: la prueba confirma que el programa
arranca, carga sus bibliotecas y transcribe (por CPU si no encuentra GPU; en Linux usa
Vulkan por software). La aceleración real se confirma en equipos con ese hardware.

Cada paquete incluye `BUILD.txt` con la versión, el SHA-256 del código fuente, las opciones
de compilación y el enlace a la corrida que lo produjo.

### Detalles de compilación

- `whisper-cli` estático (sin bibliotecas de ggml aparte) y sin optimizar para la CPU de la
  máquina de GitHub. En x86-64, base AVX2 (2013 en adelante); los equipos más viejos usan el
  paquete oficial de CPU, que detecta la CPU al ejecutarse.
- Windows: sin depender de los redistribuibles de Visual C++ (`/MT`).
- macOS: versión mínima 13.3. El paquete Intel se compila en una Mac ARM y se prueba con
  Rosetta.
- Vulkan: Vulkan SDK de LunarG (versión fijada en el workflow).
- CUDA: CUDA Toolkit 12.8 (primera versión con las RTX serie 50); el paquete incluye las
  bibliotecas de NVIDIA que necesita, redistribuibles según su licencia.

## Licencias

- whisper.cpp y ggml: MIT. Cada paquete incluye `LICENSE-whisper.cpp.txt` y `THIRD_PARTY.txt`.
- Bibliotecas de NVIDIA CUDA (solo en el paquete CUDA): anexo de redistribuibles de la
  [licencia de CUDA Toolkit](https://docs.nvidia.com/cuda/eula/).
