from __future__ import annotations

import os
import time
import zipfile

import requests


def download_file(
    url: str,
    out_path: str,
    timeout: int = 60,
    chunk_mb: int = 16,
    retries: int = 3,
) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=timeout) as resp:
                resp.raise_for_status()
                with open(out_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=chunk_mb * 1024 * 1024):
                        if chunk:
                            f.write(chunk)
            return out_path
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(3)


def extract_first_tif(zip_path: str, extract_dir: str) -> str:
    if os.path.exists(extract_dir):
        import shutil

        shutil.rmtree(extract_dir)
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    tif_paths = []
    for root, _, files in os.walk(extract_dir):
        for name in files:
            if name.lower().endswith(".tif"):
                tif_paths.append(os.path.join(root, name))

    if not tif_paths:
        raise FileNotFoundError(f"No .tif found inside {zip_path}")

    return tif_paths[0]
