"""Device utilities."""

import torch


def get_device(preference: str = "auto") -> torch.device:
    """
    Get the appropriate device for computation.

    Args:
        preference: "auto", "cuda", "mps", or "cpu"

    Returns:
        torch.device
    """
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(preference)
