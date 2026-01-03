#!/usr/bin/env python3
"""
00-install-unsloth.py - Install Unsloth and dependencies

WHAT THIS DOES:
- Installs the Unsloth library for fast fine-tuning
- Upgrades required dependencies (transformers, peft, accelerate)

WHEN TO RUN:
- Run this ONCE when setting up a new environment
- Run again if you get import errors

ENVIRONMENT:
- Works on: Colab, RunPod, local GPU machine
- Requires: Python 3.10-3.12, pip
"""

import subprocess
import sys

def install_packages():
    """Install Unsloth and dependencies."""
    print("=" * 50)
    print("Installing Unsloth - 2x faster fine-tuning")
    print("=" * 50)

    # Core dependencies
    packages = [
        "transformers",
        "peft",
        "accelerate",
        "trl",
        "datasets",
    ]

    print("\n[1/2] Installing core dependencies...")
    for pkg in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", pkg])

    # Install Unsloth
    print("\n[2/2] Installing Unsloth...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "--upgrade", "--no-cache-dir",
        "unsloth"
    ])

    print("\n" + "=" * 50)
    print("COMPLETE - Unsloth installed")
    print("=" * 50)
    print("\nNext step: Run 01-check-gpu.py")

if __name__ == "__main__":
    install_packages()
