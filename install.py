from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from importlib.metadata import version  # python >= 3.8
from pathlib import Path

from packaging.version import parse

import_name = {"py-cpuinfo": "cpuinfo", "protobuf": "google.protobuf"}


def is_installed(
    package: str,
    min_version: str | None = None,
    max_version: str | None = None,
):
    name = import_name.get(package, package)
    try:
        spec = importlib.util.find_spec(name)
    except ModuleNotFoundError:
        return False

    if spec is None:
        return False

    if not min_version and not max_version:
        return True

    if not min_version:
        min_version = "0.0.0"
    if not max_version:
        max_version = "99999999.99999999.99999999"

    try:
        pkg_version = version(package)
        return parse(min_version) <= parse(pkg_version) <= parse(max_version)
    except Exception:
        return False


def run_pip(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", *args], check=True)


def download_models():
    """Download required YOLO models for ADetailer"""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[-] ADetailer: Installing huggingface_hub for model download...")
        run_pip("huggingface_hub")
        from huggingface_hub import hf_hub_download
    
    # Get the extension directory
    ext_dir = Path(__file__).parent
    models_dir = ext_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    # List of models to download
    models_to_download = [
        ("Bingsu/adetailer", "face_yolov8n.pt"),
        ("Bingsu/adetailer", "face_yolov8s.pt"),
        ("Bingsu/adetailer", "hand_yolov8n.pt"),
        ("Bingsu/adetailer", "person_yolov8n-seg.pt"),
        ("Bingsu/adetailer", "person_yolov8s-seg.pt"),
        ("Bingsu/yolo-world-mirror", "yolov8x-worldv2.pt"),
    ]
    
    print("[-] ADetailer: Checking for required models...")
    for repo_id, filename in models_to_download:
        model_path = models_dir / filename
        if model_path.exists():
            continue
        
        try:
            print(f"[-] ADetailer: Downloading {filename}...")
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(models_dir),
                local_dir_use_symlinks=False
            )
            print(f"[-] ADetailer: Downloaded {filename}")
        except Exception as e:
            print(f"[-] ADetailer: Failed to download {filename}: {e}")
    
    print("[-] ADetailer: Model check complete")


def install():
    deps = [
        # requirements
        ("ultralytics", "8.3.75", None),
        ("rich", "13.0.0", None),
        ("huggingface_hub", None, None),
    ]

    pkgs = []
    for pkg, low, high in deps:
        if not is_installed(pkg, low, high):
            if low and high:
                cmd = f"{pkg}>={low},<={high}"
            elif low:
                cmd = f"{pkg}>={low}"
            elif high:
                cmd = f"{pkg}<={high}"
            else:
                cmd = pkg
            pkgs.append(cmd)

    if pkgs:
        run_pip(*pkgs)
    
    # Download models after installing dependencies
    download_models()


try:
    import launch

    skip_install = launch.args.skip_install
except Exception:
    skip_install = False

if not skip_install:
    install()
