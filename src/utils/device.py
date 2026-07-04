"""
Device Management Utilities

This module handles device (CPU/GPU) detection and management for PyTorch.

Educational Notes:
- CUDA is NVIDIA's parallel computing platform for GPUs
- For deep learning, GPUs provide 10-100x speedup over CPUs
- This module automatically detects and configures the best available device
- Always check device availability before assuming GPU is present
"""

import torch
from typing import Optional, Tuple


# Backend installation hints -------------------------------------------------
# Educational note:
# - Each accelerator backend ships in a *different* PyTorch build or add-on
#   package. If a backend is requested but unavailable, we tell the user exactly
#   what to install instead of only reporting that it is missing.
# - Pick the install command that matches the CUDA/ROCm/XPU version of your
#   hardware; see https://pytorch.org/get-started/locally/ for the picker.
BACKEND_INSTALL_HINTS = {
    "cuda": (
        "Install a CUDA-enabled PyTorch build and an NVIDIA driver, e.g.:\n"
        "    pip install torch --index-url https://download.pytorch.org/whl/cu124\n"
        "  AMD GPUs instead use a ROCm PyTorch build (which also reports as 'cuda'):\n"
        "    pip install torch --index-url https://download.pytorch.org/whl/rocm6.1"
    ),
    "xpu": (
        "Install an Intel GPU (XPU) enabled PyTorch build plus the Intel oneAPI runtime:\n"
        "    pip install torch --index-url https://download.pytorch.org/whl/xpu\n"
        "  Older PyTorch versions also require the extension:\n"
        "    pip install intel-extension-for-pytorch"
    ),
    "mps": (
        "MPS is only available on Apple Silicon (M-series) Macs with a recent PyTorch\n"
        "  build (the default macOS wheel already includes it). It cannot be installed\n"
        "  on Windows or Linux."
    ),
    "directml": (
        "Install the DirectML backend package on Windows (works with any DirectX 12 GPU\n"
        "  from AMD/Intel/NVIDIA):\n"
        "    pip install torch-directml"
    ),
    "cpu": (
        "CPU support ships with every PyTorch build, so no extra install is needed:\n"
        "    pip install torch"
    ),
}


def backend_install_hint(backend: str) -> str:
    """
    Return a human-readable hint describing what to install for a given backend.

    Educational note:
    - Centralising these strings keeps error messages, docs and the device report
      consistent, so there is a single source of truth for install instructions.

    Args:
        backend: One of "auto", "cuda", "xpu", "mps", "directml", "cpu".

    Returns:
        A multi-line install hint, or a generic message for unknown backends.
    """
    backend = (backend or "").lower()
    if backend == "auto":
        return (
            "'auto' selects the best available backend and never fails to install; "
            "install the wheel that matches your GPU vendor (see the per-backend hints)."
        )
    return BACKEND_INSTALL_HINTS.get(
        backend,
        "Unknown backend. Choose from: auto, cuda, xpu, mps, directml, cpu.",
    )


def _xpu_available() -> bool:
    """Return True if an Intel XPU (oneAPI) accelerator is available."""
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def _mps_available() -> bool:
    """Return True if an Apple Silicon (Metal / MPS) accelerator is available."""
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def _directml_device() -> Optional["torch.device"]:
    """
    Return a DirectML device if the optional ``torch-directml`` package is
    installed and an adapter is available, otherwise None.

    Educational note:
    - DirectML is a vendor-agnostic GPU backend on Windows built on DirectX 12.
    - It lets the same code run on AMD, Intel and NVIDIA GPUs without CUDA.
    - See https://learn.microsoft.com/windows/ai/directml/dml
    """
    try:
        import torch_directml  # type: ignore
    except ImportError:
        return None
    try:
        if torch_directml.device_count() > 0:
            return torch_directml.device()
    except Exception:
        return None
    return None


def get_device(prefer_cuda: bool = True, backend: str = "auto") -> torch.device:
    """
    Get the best available device for training/inference (vendor-agnostic).

    Educational note:
    - Deep learning runs on many kinds of accelerator, not just NVIDIA CUDA:
        * CUDA    - NVIDIA GPUs (and AMD GPUs via ROCm, which also reports as "cuda")
        * XPU     - Intel GPUs (oneAPI)
        * MPS     - Apple Silicon (Metal Performance Shaders)
        * DirectML- any DirectX 12 GPU on Windows (AMD/Intel/NVIDIA), optional package
        * CPU     - always available fallback
    - "auto" tries the fastest available backend in a sensible order.

    Args:
        prefer_cuda: If True, prefer a CUDA device when available (auto mode only)
        backend: One of "auto", "cuda", "xpu", "mps", "directml", "cpu".
                 Use this to force a specific vendor/backend.

    Returns:
        torch.device (or a DirectML device object) for use with ``.to(device)``

    Example:
        >>> device = get_device()                 # auto-detect any GPU
        >>> device = get_device(backend="directml")  # force DirectML
    """
    backend = (backend or "auto").lower()

    # Explicit backend selection ------------------------------------------------
    if backend == "cpu":
        print("Using CPU (forced)")
        return torch.device("cpu")
    if backend == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "backend='cuda' requested but no CUDA device is available.\n"
                f"  {backend_install_hint('cuda')}"
            )
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    if backend == "xpu":
        if not _xpu_available():
            raise RuntimeError(
                "backend='xpu' requested but no Intel XPU is available.\n"
                f"  {backend_install_hint('xpu')}"
            )
        print("Using Intel XPU device")
        return torch.device("xpu")
    if backend == "mps":
        if not _mps_available():
            raise RuntimeError(
                "backend='mps' requested but MPS is not available.\n"
                f"  {backend_install_hint('mps')}"
            )
        print("Using MPS (Apple Silicon) device")
        return torch.device("mps")
    if backend == "directml":
        dml = _directml_device()
        if dml is None:
            raise RuntimeError(
                "backend='directml' requested but torch-directml is not installed/available.\n"
                f"  {backend_install_hint('directml')}"
            )
        print("Using DirectML device (vendor-agnostic GPU)")
        return dml
    if backend != "auto":
        raise ValueError(f"Unknown backend '{backend}'. Choose from: auto, cuda, xpu, mps, directml, cpu")

    # Auto-detection ------------------------------------------------------------
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        # Educational note: AMD ROCm builds of PyTorch also report through this API.
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        if torch.version.cuda:
            print(f"  CUDA Version: {torch.version.cuda}")
        if getattr(torch.version, "hip", None):
            print(f"  ROCm/HIP Version: {torch.version.hip}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return device

    if _xpu_available():
        print("Using Intel XPU device")
        return torch.device("xpu")

    if _mps_available():
        print("Using MPS (Apple Silicon) device")
        return torch.device("mps")

    dml = _directml_device()
    if dml is not None:
        print("Using DirectML device (vendor-agnostic GPU)")
        return dml

    print("Using CPU (no GPU detected)")
    return torch.device("cpu")


def get_gpu_memory_info() -> Tuple[float, float]:
    """
    Get GPU memory usage information.

    Educational note:
    - Monitoring GPU memory is crucial for training large models
    - Out-of-memory errors are common when batch size is too large
    - This function helps diagnose memory issues

    Returns:
        Tuple of (allocated_gb, cached_gb) in GB
        Returns (0, 0) if CUDA is not available

    Example:
        >>> allocated, cached = get_gpu_memory_info()
        >>> print(f"GPU Memory: {allocated:.2f} GB allocated, {cached:.2f} GB cached")
        GPU Memory: 2.34 GB allocated, 3.12 GB cached
    """
    if not torch.cuda.is_available():
        return 0.0, 0.0

    allocated = torch.cuda.memory_allocated(0) / 1e9  # Convert to GB
    cached = torch.cuda.memory_reserved(0) / 1e9

    return allocated, cached


def clear_gpu_cache() -> None:
    """
    Clear GPU memory cache.

    Educational note:
    - PyTorch caches memory allocations for efficiency
    - Sometimes you need to clear cache (e.g., before training a new model)
    - This function forces Python to release unused GPU memory back to CUDA
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU cache cleared")


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Educational note:
    - Deep learning has many sources of randomness
    - Setting seeds ensures experiments are reproducible
    - Important for debugging and comparing different approaches
    - Seeds affect: weight initialization, data shuffling, dropout, etc.

    Args:
        seed: Random seed value

    Example:
        >>> set_seed(42)
        >>> # Now training will be reproducible
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Also set for other libraries if they're imported
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def available_backends() -> dict:
    """
    Report which compute backends are usable in the current environment.

    Educational note:
    - Lets tooling and users see at a glance which accelerators are ready and
      which need an extra install, without triggering an error.

    Returns:
        Ordered dict mapping backend name -> bool (True if available now).
        "cpu" is always True; "auto" is omitted (it is a selection strategy).
    """
    return {
        "cuda": torch.cuda.is_available(),
        "xpu": _xpu_available(),
        "mps": _mps_available(),
        "directml": _directml_device() is not None,
        "cpu": True,
    }


def print_backend_install_guide() -> None:
    """
    Print the availability and install instructions for every backend.

    Educational note:
    - A single, friendly summary of "what do I install to use my GPU?" that
      reuses the same hints emitted by :func:`get_device` on failure.
    """
    print("\n" + "=" * 70)
    print("BACKENDS & INSTALL REQUIREMENTS")
    print("=" * 70)
    for backend, ready in available_backends().items():
        status = "available" if ready else "NOT available"
        print(f"\n{backend} ({status}):")
        for line in backend_install_hint(backend).splitlines():
            print(f"  {line}")
    print("=" * 70)


def print_device_info() -> None:
    """
    Print detailed information about available devices.

    Educational note:
    - Comprehensive overview of the computing environment
    - Useful for debugging and understanding performance
    - Shows both CPU and GPU capabilities
    """
    print("\n" + "=" * 70)
    print("DEVICE INFORMATION")
    print("=" * 70)

    # PyTorch version
    print(f"\nPyTorch Version: {torch.__version__}")

    # CUDA information
    print("\nCUDA Information:")
    if torch.cuda.is_available():
        print(f"  CUDA Available: Yes")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"  Number of GPUs: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            print(f"\n  GPU {i}:")
            print(f"    Name: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"    Memory: {props.total_memory / 1e9:.1f} GB")
            print(f"    Compute Capability: {props.major}.{props.minor}")
            print(f"    Multi-processors: {props.multi_processor_count}")
    else:
        print("  CUDA Available: No")
        print(f"    To enable: {BACKEND_INSTALL_HINTS['cuda'].splitlines()[0]}")

    # MPS information (Apple Silicon)
    print("\nMPS Information (Apple Silicon):")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("  MPS Available: Yes")
        print("  MPS Built: Yes" if torch.backends.mps.is_built() else "  MPS Built: No")
    else:
        print("  MPS Available: No")

    # Intel XPU information
    print("\nIntel XPU Information:")
    if _xpu_available():
        print("  XPU Available: Yes")
    else:
        print("  XPU Available: No")
        print(f"    To enable: {BACKEND_INSTALL_HINTS['xpu'].splitlines()[0]}")

    # DirectML information (vendor-agnostic GPU on Windows)
    print("\nDirectML Information (Windows, DirectX 12):")
    if _directml_device() is not None:
        print("  DirectML Available: Yes")
    else:
        print("  DirectML Available: No")
        print(f"    To enable: {BACKEND_INSTALL_HINTS['directml'].splitlines()[0].strip()}")

    # CPU information
    import platform

    print("\nCPU Information:")
    print(f"  Processor: {platform.processor()}")
    print(f"  Machine: {platform.machine()}")

    # Current memory usage
    if torch.cuda.is_available():
        allocated, cached = get_gpu_memory_info()
        print(f"\nGPU Memory Usage:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Cached: {cached:.2f} GB")

    print("=" * 70)


def get_dtype(use_fp16: bool = True) -> torch.dtype:
    """
    Get the appropriate data type for training.

    Educational note:
    - FP32 (float32): Standard precision, slower, more memory
    - FP16 (float16): Half precision, faster, less memory
    - BF16 (bfloat16): Better range than FP16, less precision than FP32
    - Mixed precision training uses FP16 for computation but FP32 for master weights

    Mixed precision benefits:
    - 2x faster training on modern GPUs (Tensor Cores)
    - 50% less memory usage
    - Similar or better model quality

    Args:
        use_fp16: If True, use FP16; otherwise use FP32

    Returns:
        torch.dtype object

    Example:
        >>> dtype = get_dtype(use_fp16=True)
        >>> model = model.to(dtype)
        >>> print(f"Using dtype: {dtype}")
        Using dtype: torch.float16
    """
    if use_fp16:
        # Check if device supports FP16
        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 7:
            # Volta, Turing, Ampere, Hopper architectures support FP16 well
            return torch.float16
        else:
            print("Warning: FP16 requested but GPU may not fully support it")
            return torch.float16
    else:
        return torch.float32


if __name__ == "__main__":
    # Educational: Demonstrate device utilities
    print_device_info()

    # Show which backends are available and how to install the missing ones
    print_backend_install_guide()

    # Get device
    device = get_device()
    print(f"\nSelected device: {device}")

    # Get data type
    dtype = get_dtype(use_fp16=True)
    print(f"Selected dtype: {dtype}")

    # Test GPU memory
    if torch.cuda.is_available():
        allocated, cached = get_gpu_memory_info()
        print(f"\nGPU Memory: {allocated:.2f} GB allocated, {cached:.2f} GB cached")

    # Set seed
    set_seed(42)
    print("\nRandom seed set to 42 for reproducibility")
