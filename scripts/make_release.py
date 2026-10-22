"""Junta los paquetes de una corrida del workflow en una release: manifest.json (lo que lee
DowP), SHA256SUMS.txt y RELEASE_NOTES.md. Avisa qué destinos pedidos no llegaron.

    python3 scripts/make_release.py dist v1.9.5 7 https://github.com/.../actions/runs/123
"""
import json
import os
import sys
from pathlib import Path


def main():
    folder, version, run_number, run_url = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
    entries = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("*.json"))
               if p.name != "manifest.json"]
    if not entries:
        raise SystemExit("Ningún destino produjo un paquete: no hay nada que publicar.")
    requested = [t["id"] for t in json.loads(os.environ.get("MATRIX", '{"include": []}'))["include"]]
    # El id de cada destino del workflow es "sistema-arquitectura-backend".
    built = {f"{e['platform']}-{e['arch']}-{e['backend']}" for e in entries}
    missing = [t for t in requested if t not in built]

    manifest = {
        "whisper_version": version,
        "build": int(run_number),
        "workflow_run": run_url,
        "packages": [{key: e[key] for key in ("file", "size", "sha256", "platform", "arch", "backend",
                                              "source_sha256", "smoke_test")} for e in entries],
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (folder / "SHA256SUMS.txt").write_text("".join(f"{e['sha256']}  {e['file']}\n" for e in entries), encoding="utf-8")

    rows = "\n".join(f"| `{e['file']}` | {e['platform']} {e['arch']} | {e['backend']} | {e['size'] / 1_000_000:.1f} MB "
                     f"| `{e['sha256'][:16]}…` | {e['smoke_test']} |" for e in entries)
    notes = [
        f"Compilación {run_number} de **whisper.cpp {version}** (código oficial sin modificar, "
        f"[ggml-org/whisper.cpp](https://github.com/ggml-org/whisper.cpp/releases/tag/{version})) para DowP.",
        "",
        "| Paquete | Sistema | Backend | Tamaño | SHA-256 | Prueba |",
        "|---|---|---|---|---|---|",
        rows,
        "",
        "Cada paquete transcribió el audio de ejemplo de whisper.cpp en la máquina de GitHub, que no "
        "tiene GPU utilizable: la columna *Prueba* dice si lo hizo con su backend o por CPU. La "
        "aceleración real se confirma en equipos con ese hardware.",
        "",
        f"Sumas completas en `SHA256SUMS.txt`; datos para DowP en `manifest.json`. [Registro de la compilación]({run_url}).",
    ]
    if missing:
        notes += ["", "**Destinos que fallaron en esta compilación:** " + ", ".join(missing) + " (ver el registro)."]
    (folder / "RELEASE_NOTES.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    print(f"{len(entries)} paquete(s); faltan: {missing or 'ninguno'}")


if __name__ == "__main__":
    main()
