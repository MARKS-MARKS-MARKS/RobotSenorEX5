from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DeviceInfo:
    device: str
    cuda_available: bool
    message: str


def choose_device(prefer_gpu: bool = True) -> DeviceInfo:
    try:
        import torch
    except Exception as exc:
        return DeviceInfo("cpu", False, f"torch import failed; using CPU. {exc}")

    if prefer_gpu and torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return DeviceInfo("cuda", True, f"Using device: cuda ({name})")

    if prefer_gpu:
        return DeviceInfo(
            "cpu",
            False,
            "Using device: cpu (CUDA requested but torch.cuda.is_available() is False). "
            "Check NVIDIA driver with nvidia-smi.",
        )

    return DeviceInfo("cpu", False, "Using device: cpu")
