"""Compila whisper-cli de whisper.cpp (código oficial, sin modificar) para DowP.

Lo ejecuta el workflow .github/workflows/whisper.yml, una vez por destino:

    python scripts/build_whisper.py --version latest --backend vulkan --arch x64 --output dist

Pasos:
1. Resuelve la versión ("latest" = la última publicación estable de ggml-org/whisper.cpp)
   y descarga el código fuente de esa etiqueta. Su SHA-256 queda anotado en BUILD.txt.
2. Compila solo whisper-cli, estático, con el backend pedido (cpu, vulkan, metal, cuda).
3. Lo prueba de verdad: transcribe samples/jfk.wav (incluido en whisper.cpp) con el modelo
   tiny.en y comprueba el texto. Las máquinas de GitHub no tienen GPU utilizable, así que
   el backend de GPU no se ejecuta aquí: se comprueba que el programa arranca, carga sus
   bibliotecas y transcribe (por CPU si no encuentra GPU). La aceleración real la prueban
   personas con ese hardware.
4. Empaqueta (zip en Windows, tar.gz en macOS y Linux, que conserva los permisos) con la
   licencia de whisper.cpp, BUILD.txt y, en CUDA, las bibliotecas redistribuibles de
   NVIDIA. Escribe el SHA-256 y un fragmento JSON para el manifiesto de la publicación.

Solo usa la biblioteca estándar de Python.
"""
import argparse
import re
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO = "ggml-org/whisper.cpp"
# Modelo de la prueba: tiny.en de la misma revisión fijada que usa DowP (whisper_catalog.py).
TEST_MODEL_URL = ("https://huggingface.co/ggerganov/whisper.cpp/resolve/"
                  "5359861c739e955e79d9a303bcbc70fb988958b1/ggml-tiny.en.bin")
TEST_MODEL_SIZE = 77704715
TEST_MODEL_SHA256 = "921e4cf8686fdd993dcd081a5da5b6c365bfde1162e72b08d75ac75289920b1f"
TEST_PHRASE = "ask not what your country can do for you"
# Bibliotecas de CUDA que necesita whisper-cli en tiempo de ejecución. La licencia de
# NVIDIA (CUDA EULA, anexo de redistribuibles) permite distribuirlas junto al programa.
CUDA_RUNTIME_DLLS = ("cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll")


def log(message):
    print(f"[build] {message}", flush=True)


def run(cmd, **kwargs):
    log("$ " + " ".join(str(c) for c in cmd))
    return subprocess.run([str(c) for c in cmd], check=True, **kwargs)


def http_get(url, target=None):
    headers = {"User-Agent": "dowp-runtimes"}
    token = os.environ.get("GITHUB_TOKEN")
    if token and url.startswith("https://api.github.com/"):
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        if target is None:
            return response.read()
        with open(target, "wb") as handle:
            shutil.copyfileobj(response, handle)
    return None


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_version(version):
    if version and version != "latest":
        return version
    data = json.loads(http_get(f"https://api.github.com/repos/{REPO}/releases/latest"))
    return data["tag_name"]


def fetch_source(version, work):
    archive = work / f"whisper.cpp-{version}.tar.gz"
    if not archive.exists():
        http_get(f"https://github.com/{REPO}/archive/refs/tags/{version}.tar.gz", archive)
    source_sha = sha256_of(archive)
    with tarfile.open(archive) as package:
        root = work.resolve()
        for member in package.getmembers():
            if not (root / member.name).resolve().is_relative_to(root):
                raise ValueError(f"Ruta insegura en el código fuente: {member.name}")
            if member.issym() or member.islnk():
                target = (root / member.name).parent / member.linkname
                if not target.resolve().is_relative_to(root):
                    raise ValueError(f"Enlace inseguro en el código fuente: {member.name}")
        # filter="data" (Python 3.12+) además rechaza permisos y enlaces raros.
        package.extractall(work, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
    source = next(p for p in work.iterdir() if p.is_dir() and p.name.startswith("whisper.cpp-"))
    return source, source_sha


def cmake_flags(backend, arch, deployment_target):
    flags = [
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_SHARED_LIBS=OFF",          # un solo ejecutable, sin bibliotecas de ggml aparte
        "-DGGML_NATIVE=OFF",                # no optimizar para la CPU de la máquina de GitHub
        "-DWHISPER_BUILD_TESTS=OFF",
        "-DWHISPER_BUILD_EXAMPLES=ON",
        "-DWHISPER_SDL2=OFF",
        "-DWHISPER_CURL=OFF",
    ]
    system = platform.system()
    if system == "Windows":
        flags.append("-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded")  # sin depender de los redistribuibles de VC++
    if arch in ("x64", "x86_64"):
        # Base x86-64 con AVX2 (Intel Haswell / AMD Excavator, 2013-2015 en adelante). Los
        # equipos más viejos usan el paquete oficial de CPU, que detecta la CPU en ejecución.
        flags += ["-DGGML_AVX=ON", "-DGGML_AVX2=ON", "-DGGML_FMA=ON", "-DGGML_F16C=ON",
                  "-DGGML_AVX512=OFF"]
    if system == "Darwin":
        flags += [f"-DCMAKE_OSX_ARCHITECTURES={'x86_64' if arch in ('x64', 'x86_64') else 'arm64'}",
                  f"-DCMAKE_OSX_DEPLOYMENT_TARGET={deployment_target}"]
    flags.append(f"-DGGML_METAL={'ON' if backend == 'metal' else 'OFF'}")
    if backend == "metal":
        flags.append("-DGGML_METAL_EMBED_LIBRARY=ON")   # los shaders van dentro del ejecutable
    flags.append(f"-DGGML_VULKAN={'ON' if backend == 'vulkan' else 'OFF'}")
    flags.append(f"-DGGML_CUDA={'ON' if backend == 'cuda' else 'OFF'}")
    if backend == "cuda":
        # Turing (RTX 20/GTX 16) hasta Blackwell (RTX 50, sm_120, por eso CUDA 12.8+).
        archs = os.environ.get("DOWP_CUDA_ARCHS", "75-real;80-real;86-real;89-real;120-real;120-virtual")
        flags.append(f"-DCMAKE_CUDA_ARCHITECTURES={archs}")
    return flags


def find_binary(build):
    name = "whisper-cli.exe" if platform.system() == "Windows" else "whisper-cli"
    found = sorted(p for p in build.rglob(name) if p.is_file())
    if not found:
        raise RuntimeError("La compilación no produjo whisper-cli.")
    return found[0]


def launcher(arch):
    """En una Mac ARM, el ejecutable x86_64 se prueba con Rosetta."""
    if platform.system() == "Darwin" and arch in ("x64", "x86_64") and platform.machine() == "arm64":
        return ["arch", "-x86_64"]
    return []


def smoke_test(binary, source, arch, work):
    prefix = launcher(arch)
    result = subprocess.run(prefix + [str(binary), "--help"], capture_output=True, timeout=120)
    output = (result.stdout + result.stderr).decode("utf-8", "replace")
    if "--output-srt" not in output:
        raise RuntimeError("whisper-cli --help no respondió como se esperaba:\n" + output[-2000:])
    model = work / "ggml-tiny.en.bin"
    if not model.exists():
        http_get(TEST_MODEL_URL, model)
    if model.stat().st_size != TEST_MODEL_SIZE or sha256_of(model) != TEST_MODEL_SHA256:
        raise RuntimeError("El modelo de prueba no coincide con el hash fijado.")
    sample = source / "samples" / "jfk.wav"
    for forced_cpu in (False, True):
        extra = ["-ng"] if forced_cpu else []
        # Sin -np: whisper informa qué backend usó ("using Vulkan0 backend" o "no GPU found").
        result = subprocess.run(prefix + [str(binary), "-m", str(model), "-f", str(sample), "-nt"] + extra,
                                capture_output=True, timeout=600, cwd=str(binary.parent))
        text = (result.stdout + result.stderr).decode("utf-8", "replace")
        if forced_cpu:
            label = "CPU forzada (-ng)"
        else:
            used = re.search(r"using (\S+) backend", text)
            name = used.group(1) if used else ""
            if name.upper().startswith("BLAS"):
                label = "CPU (BLAS/Accelerate)"   # BLAS acelera en la CPU, no es una GPU
            elif name and not name.upper().startswith("CPU"):
                label = f"GPU ({name})"
            else:
                label = "CPU (la máquina no tiene GPU)"
        if result.returncode == 0 and TEST_PHRASE in text.lower().replace(",", ""):
            log(f"Prueba de transcripción correcta: {label}.")
            return label
        log(f"Prueba sin éxito ({label}, código {result.returncode}):\n{text[-1500:]}")
    raise RuntimeError("whisper-cli no pudo transcribir el audio de prueba.")


def cuda_runtime_files():
    cuda = os.environ.get("CUDA_PATH")
    if not cuda:
        raise RuntimeError("CUDA_PATH no está definido.")
    files = []
    for name in CUDA_RUNTIME_DLLS:
        match = next(Path(cuda).rglob(name), None)
        if match is None:
            raise RuntimeError(f"No se encontró {name} en {cuda}.")
        files.append(match)
    return files


def package(binary, source, backend, arch, version, source_sha, flags, test_result, output):
    system = {"Windows": "windows", "Darwin": "macos", "Linux": "linux"}[platform.system()]
    arch_name = "x64" if arch in ("x64", "x86_64", "amd64") else "arm64"
    label = backend
    if backend == "cuda":
        label = "cuda" + os.environ.get("DOWP_CUDA_VERSION", "").replace(" ", "")
    name = f"whisper-{version}-{system}-{arch_name}-{label}"
    staging = output / name
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    shutil.copy2(binary, staging / binary.name)
    # DLL que el compilador haya dejado junto al ejecutable pese a compilar estático.
    for extra in binary.parent.glob("*.dll"):
        shutil.copy2(extra, staging / extra.name)
    shutil.copy2(source / "LICENSE", staging / "LICENSE-whisper.cpp.txt")
    notices = ["whisper.cpp y ggml: licencia MIT (LICENSE-whisper.cpp.txt). Código sin modificar."]
    if backend == "cuda":
        for dll in cuda_runtime_files():
            shutil.copy2(dll, staging / dll.name)
        notices.append("Bibliotecas de NVIDIA CUDA (" + ", ".join(CUDA_RUNTIME_DLLS) + "): distribuidas según "
                       "el anexo de redistribuibles de la licencia de NVIDIA CUDA Toolkit "
                       "(https://docs.nvidia.com/cuda/eula/).")
    (staging / "THIRD_PARTY.txt").write_text("\n".join(notices) + "\n", encoding="utf-8")
    build_info = {
        "whisper_version": version,
        "source": f"https://github.com/{REPO}/archive/refs/tags/{version}.tar.gz",
        "source_sha256": source_sha,
        "backend": backend,
        "platform": system,
        "arch": arch_name,
        "cmake_flags": flags,
        "smoke_test": test_result,
        "runner": os.environ.get("RUNNER_NAME", platform.node()),
        "workflow_run": os.environ.get("GITHUB_SERVER_URL", "") + "/" + os.environ.get("GITHUB_REPOSITORY", "")
                        + "/actions/runs/" + os.environ.get("GITHUB_RUN_ID", ""),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (staging / "BUILD.txt").write_text(json.dumps(build_info, indent=2) + "\n", encoding="utf-8")

    if system == "windows":
        archive = output / f"{name}.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            for path in sorted(staging.rglob("*")):
                zf.write(path, path.relative_to(output))
    else:
        archive = output / f"{name}.tar.gz"
        with tarfile.open(archive, "w:gz") as tf:
            tf.add(staging, arcname=name)
    digest = sha256_of(archive)
    (output / f"{archive.name}.sha256").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    entry = dict(build_info, file=archive.name, size=archive.stat().st_size, sha256=digest)
    (output / f"{archive.name}.json").write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")
    shutil.rmtree(staging)
    log(f"Paquete: {archive.name} ({archive.stat().st_size} bytes)  SHA-256 {digest}")
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", default="latest", help="Etiqueta de whisper.cpp (ej. v1.9.5) o 'latest'.")
    parser.add_argument("--backend", choices=("cpu", "vulkan", "metal", "cuda"), required=True)
    parser.add_argument("--arch", default="x64" if platform.machine().lower() in ("amd64", "x86_64") else "arm64")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work", type=Path, default=Path("work"))
    parser.add_argument("--deployment-target", default="13.3", help="Versión mínima de macOS.")
    args = parser.parse_args()

    version = resolve_version(args.version)
    log(f"whisper.cpp {version} · backend {args.backend} · arquitectura {args.arch} · {platform.system()}")
    work = args.work.resolve()
    output = args.output.resolve()
    work.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    source, source_sha = fetch_source(version, work)
    log(f"Código fuente: SHA-256 {source_sha}")

    build = work / f"build-{args.backend}-{args.arch}"
    flags = cmake_flags(args.backend, args.arch, args.deployment_target)
    run(["cmake", "-S", source, "-B", build] + flags)
    jobs = str(os.cpu_count() or 4)
    run(["cmake", "--build", build, "--config", "Release", "--target", "whisper-cli", "-j", jobs])
    binary = find_binary(build)
    if args.backend == "cuda":
        # Para la prueba, las DLL de CUDA tienen que estar junto al ejecutable.
        for dll in cuda_runtime_files():
            shutil.copy2(dll, binary.parent / dll.name)
    test_result = smoke_test(binary, source, args.arch, work)
    package(binary, source, args.backend, args.arch, version, source_sha, flags, test_result, output)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"ERROR: {exc}")
        sys.exit(1)
