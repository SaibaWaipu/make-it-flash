"""Build a self-contained Kaggle notebook from the current source tree."""

from __future__ import annotations

import base64
import json
import zipfile
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = [Path("pyproject.toml"), Path("README.md"), *Path("src/make_it_flash").rglob("*.py")]


def _source_bundle() -> str:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in INCLUDE:
            path = ROOT / relative
            if path.is_file():
                archive.write(path, relative.as_posix())
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def build_notebook() -> dict:
    bundle = _source_bundle()
    cells = [
        {
            "cell_type": "markdown",
            "id": "overview",
            "metadata": {},
            "source": [
                "# make_it_flash: one-layer GDN pilot\n",
                "\n",
                "Self-contained copy of the tracked pipeline. The tokenized calibration mix runs first.\n",
                "The notebook measures GPU memory first and only loads the 32B teacher if at least 66 GiB is free.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "setup",
            "metadata": {},
            "outputs": [],
            "source": [
                "import base64, os, subprocess, sys, zipfile\n",
                "import importlib.metadata as package_metadata\n",
                "from io import BytesIO\n",
                "from pathlib import Path\n",
                "\n",
                f"BUNDLE = {bundle!r}\n",
                "PROJECT = Path('/kaggle/working/make_it_flash')\n",
                "PROJECT.mkdir(parents=True, exist_ok=True)\n",
                "with zipfile.ZipFile(BytesIO(base64.b64decode(BUNDLE))) as archive:\n",
                "    archive.extractall(PROJECT)\n",
                "source_path = str(PROJECT / 'src')\n",
                "sys.path.insert(0, source_path)\n",
                "os.environ['PYTHONPATH'] = source_path + os.pathsep + os.environ.get('PYTHONPATH', '')\n",
                "versions = {name: package_metadata.version(name) for name in ('torch', 'transformers', 'datasets', 'huggingface-hub', 'accelerate', 'safetensors')}\n",
                "print('Dependency versions:', versions)\n",
                "ARTIFACTS = Path('/kaggle/working/artifacts')\n",
                "DATA = ARTIFACTS / 'data'\n",
                "CACHE = ARTIFACTS / 'cache'\n",
                "GDN = ARTIFACTS / 'gdn'\n",
                "import torch\n",
                "GPU_FREE_GIB = sum(torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())) / (1024**3) if torch.cuda.is_available() else 0.0\n",
                "GPU_READY = GPU_FREE_GIB >= 66.0\n",
                "print(f'Free CUDA VRAM: {GPU_FREE_GIB:.1f} GiB; teacher pass available: {GPU_READY}')\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "prepare",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Stage 1: small, streaming public-data sample; no 32B weights are loaded here.\n",
                "subprocess.run([sys.executable, '-m', 'make_it_flash.cli', 'prepare', '--output-dir', str(DATA), '--max-tokens', '100000', '--max-seq-len', '2048'], cwd=PROJECT, check=True)\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "teacher-cache",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Stage 2: run automatically only if the Kaggle GPU clears the 66 GiB guard.\n",
                "if GPU_READY:\n",
                "    subprocess.run([sys.executable, '-m', 'make_it_flash.cli', 'cache', '--data-file', str(DATA / 'calibration.jsonl'), '--output-dir', str(CACHE), '--layers', '0'], cwd=PROJECT, check=True)\n",
                "else:\n",
                "    print('Teacher cache skipped before weight download; use the recommended Hugging Face Jobs workflow.')\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "fit",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Stage 3: fit only when the teacher cache was created.\n",
                "if (CACHE / 'cache_manifest.json').is_file():\n",
                "    subprocess.run([sys.executable, '-m', 'make_it_flash.cli', 'fit', '--cache-dir', str(CACHE), '--output-dir', str(GDN), '--layer', '0'], cwd=PROJECT, check=True)\n",
                "else:\n",
                "    print('GDN fitting skipped; no teacher cache exists.')\n",
            ],
        },
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    output = ROOT / "kaggle" / "make_it_flash.ipynb"
    output.write_text(json.dumps(build_notebook(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
