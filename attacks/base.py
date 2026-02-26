"""Base class for all audio adversarial attacks."""

from abc import ABC, abstractmethod
from typing import Dict, Any

import torch
import numpy as np


class BaseAttack(ABC):
    """Abstract base for adversarial audio attacks.

    All attacks accept a float32 waveform tensor of shape (1, T) or (T,),
    normalized to [-1, 1], and return a perturbed waveform of the same shape.
    """

    def __init__(self, cfg: Dict[str, Any], device: torch.device):
        self.cfg = cfg
        self.device = device

    @abstractmethod
    def apply(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Return perturbed waveform. No in-place modification of input."""
        ...

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_2d(waveform: torch.Tensor) -> torch.Tensor:
        """Ensure shape is (1, T)."""
        if waveform.dim() == 1:
            return waveform.unsqueeze(0)
        return waveform

    @staticmethod
    def _clip(waveform: torch.Tensor, eps: float, original: torch.Tensor) -> torch.Tensor:
        """Project perturbation to L-inf ball of radius eps around original."""
        delta = waveform - original
        delta = delta.clamp(-eps, eps)
        return (original + delta).clamp(-1.0, 1.0)

    @staticmethod
    def _snr_db(original: torch.Tensor, perturbed: torch.Tensor) -> float:
        signal_power = (original ** 2).mean().item()
        noise_power = ((perturbed - original) ** 2).mean().item()
        if noise_power < 1e-12:
            return float("inf")
        return 10.0 * np.log10(signal_power / noise_power)
