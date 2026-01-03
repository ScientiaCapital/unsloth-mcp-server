#!/usr/bin/env python3
"""
01-check-gpu.py - Verify GPU is available

WHAT THIS DOES:
- Checks if CUDA GPU is available
- Shows GPU name and memory

IF GPU NOT FOUND:
- Colab: Runtime -> Change runtime type -> T4 GPU
- RunPod: Select a GPU pod template
- Local: Ensure NVIDIA drivers are installed
"""

import torch

def check_gpu():
    """Check for CUDA GPU availability."""
    print("=" * 50)
    print("Checking GPU availability")
    print("=" * 50)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9

        print(f"\nGPU found: {gpu_name}")
        print(f"GPU memory: {gpu_memory:.1f} GB")
        print(f"CUDA version: {torch.version.cuda}")

        print("\n" + "=" * 50)
        print("COMPLETE - GPU ready")
        print("=" * 50)
        print("\nNext step: Run 02-load-model.py")
        return True
    else:
        print("\nNO GPU FOUND!")
        print("\nTo fix:")
        print("- Colab: Runtime -> Change runtime type -> T4 GPU")
        print("- RunPod: Select a GPU pod template")
        print("- Local: Install NVIDIA CUDA drivers")
        return False

if __name__ == "__main__":
    check_gpu()
