#!/usr/bin/env python3
"""GPU diagnostic script for RunPod/CUDA environments.

Run with: uv run python scripts/check_gpu.py
"""

import sys


def check_torch_cuda():
    """Check PyTorch CUDA availability and configuration."""
    print("=" * 60)
    print("PyTorch CUDA Diagnostics")
    print("=" * 60)

    try:
        import torch

        print(f"\nPyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"cuDNN version: {torch.backends.cudnn.version()}")
            print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")

            device_count = torch.cuda.device_count()
            print(f"\nGPU count: {device_count}")

            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                print(f"\n  GPU {i}: {props.name}")
                print(f"    Total memory: {props.total_memory / 1024**3:.2f} GB")
                print(f"    Compute capability: {props.major}.{props.minor}")
                print(f"    Multi-processors: {props.multi_processor_count}")

                # Check current memory usage
                if torch.cuda.is_available():
                    torch.cuda.set_device(i)
                    allocated = torch.cuda.memory_allocated(i) / 1024**3
                    reserved = torch.cuda.memory_reserved(i) / 1024**3
                    print(f"    Memory allocated: {allocated:.2f} GB")
                    print(f"    Memory reserved: {reserved:.2f} GB")

            # Test computation
            print("\n" + "-" * 40)
            print("Testing CUDA computation...")
            device = torch.device("cuda:0")
            x = torch.randn(1000, 1000, device=device)
            y = torch.randn(1000, 1000, device=device)
            z = torch.mm(x, y)
            torch.cuda.synchronize()
            print(f"  Matrix multiplication test: PASSED")
            print(f"  Result shape: {z.shape}, dtype: {z.dtype}")

            # Memory after test
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            print(f"  Memory after test: {allocated:.2f} GB")

            # Cleanup
            del x, y, z
            torch.cuda.empty_cache()

            return True
        else:
            print("\nWARNING: CUDA is not available!")
            print("Possible reasons:")
            print("  - No NVIDIA GPU present")
            print("  - CUDA drivers not installed")
            print("  - PyTorch installed without CUDA support")
            print("\nTo install PyTorch with CUDA:")
            print("  pip install torch --index-url https://download.pytorch.org/whl/cu121")
            return False

    except ImportError as e:
        print(f"ERROR: Could not import torch: {e}")
        return False


def check_a6000_optimizations():
    """Check optimizations specific to RTX A6000."""
    print("\n" + "=" * 60)
    print("RTX A6000 Specific Checks")
    print("=" * 60)

    try:
        import torch

        if not torch.cuda.is_available():
            print("CUDA not available, skipping A6000 checks")
            return

        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name.lower()

        if "a6000" in gpu_name or "rtx" in gpu_name:
            print(f"\nDetected: {props.name}")

            # A6000 has compute capability 8.6 (Ampere)
            if props.major >= 8:
                print(f"  Ampere architecture detected (CC {props.major}.{props.minor})")
                print("  TF32 tensor cores available")

                # Enable TF32 for faster training
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                print("  TF32 enabled for matmul and cuDNN")

                # Check if we can use bfloat16
                if torch.cuda.is_bf16_supported():
                    print("  BFloat16 supported")
                else:
                    print("  BFloat16 not supported")

            # Memory recommendations for A6000 (48GB)
            total_mem_gb = props.total_memory / 1024**3
            print(f"\n  Total memory: {total_mem_gb:.1f} GB")

            if total_mem_gb >= 40:  # A6000 has 48GB
                print("  Recommended batch sizes:")
                print("    - Training: 64-128")
                print("    - Integration experiment: 128-256")
                print("  Large memory allows full dataset in memory")

        else:
            print(f"GPU is not RTX A6000: {props.name}")
            print("Settings may need adjustment")

    except Exception as e:
        print(f"Error during A6000 checks: {e}")


def check_dependencies():
    """Check all required dependencies."""
    print("\n" + "=" * 60)
    print("Dependency Check")
    print("=" * 60)

    required = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("sklearn", "scikit-learn"),
        ("mne", "MNE-Python"),
        ("xgboost", "XGBoost"),
        ("umap", "UMAP"),
        ("hdbscan", "HDBSCAN"),
        ("numba", "Numba"),
        ("hydra", "Hydra"),
    ]

    all_ok = True
    for module, name in required:
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "unknown")
            print(f"  {name}: {version}")
        except ImportError:
            print(f"  {name}: NOT FOUND")
            all_ok = False

    return all_ok


def main():
    """Run all diagnostics."""
    print("\n" + "#" * 60)
    print("#  EEG State Biomarkers - GPU Diagnostic")
    print("#" * 60)

    cuda_ok = check_torch_cuda()
    check_a6000_optimizations()
    deps_ok = check_dependencies()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    if cuda_ok and deps_ok:
        print("  All checks PASSED - ready for training!")
        sys.exit(0)
    else:
        if not cuda_ok:
            print("  WARNING: CUDA not available")
        if not deps_ok:
            print("  WARNING: Some dependencies missing")
        sys.exit(1)


if __name__ == "__main__":
    main()
