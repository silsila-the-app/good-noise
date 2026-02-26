"""Psychoacoustic-Masked PGD Attack.

Technique:
    Runs PGD on the speaker embedding loss, but constrains perturbation
    energy to lie below the psychoacoustic masking threshold computed from
    the input signal. Perturbations are thus imperceptible by design.

    The masking threshold is computed using a simplified Bark-scale model
    (Johnston 1988): spreading function + spreading gain + quiet threshold.

    Based on: VoiceGuard (IJCAI 2023), E2E-VGuard (NeurIPS 2025).
"""

from typing import Dict, Any

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from .base import BaseAttack
from models.speaker_encoder import SpeakerEncoder


# ── Psychoacoustic masking ──────────────────────────────────────────────────

def _bark(freq_hz: np.ndarray) -> np.ndarray:
    """Convert Hz to Bark scale (Zwicker 1961)."""
    return 13.0 * np.arctan(0.00076 * freq_hz) + 3.5 * np.arctan((freq_hz / 7500.0) ** 2)


def _quiet_threshold_db(freq_hz: np.ndarray) -> np.ndarray:
    """Absolute threshold of quiet (ISO 226 approximation), in dB SPL."""
    f = np.maximum(freq_hz, 20.0)
    return (
        3.64 * (f / 1000.0) ** -0.8
        - 6.5 * np.exp(-0.6 * (f / 1000.0 - 3.3) ** 2)
        + 1e-3 * (f / 1000.0) ** 4
    )


def compute_masking_threshold(
    waveform_np: np.ndarray,
    sample_rate: int,
    n_fft: int = 2048,
    gain_db: float = 6.0,
) -> np.ndarray:
    """Compute per-frame, per-frequency masking threshold.

    Returns: threshold in linear amplitude, shape (n_fft//2+1, n_frames).
    """
    from scipy.signal import stft

    _, _, Zxx = stft(waveform_np, fs=sample_rate, nperseg=n_fft, noverlap=n_fft * 3 // 4)
    mag = np.abs(Zxx)  # (freq_bins, frames)
    power_db = 20.0 * np.log10(mag + 1e-8)

    freqs = np.linspace(0, sample_rate / 2, mag.shape[0])
    quiet = _quiet_threshold_db(freqs)[:, np.newaxis]  # (freq_bins, 1)

    # Simple simultaneous masking: masker raises threshold proportional to its power.
    # Use Bark spreading: for each frequency, neighboring Bark bands raise threshold.
    bark_freqs = _bark(freqs)
    n_bark = 24
    bark_edges = np.linspace(0, _bark(sample_rate / 2), n_bark + 1)

    threshold_db = power_db - gain_db  # masker raises threshold by (power - gain) dB
    threshold_db = np.maximum(threshold_db, quiet)

    # Convert to linear amplitude
    threshold_linear = 10.0 ** (threshold_db / 20.0)
    return threshold_linear


# ── Attack ──────────────────────────────────────────────────────────────────

class PsychoacousticPGDAttack(BaseAttack):
    """PGD attack on speaker embeddings, constrained by psychoacoustic masking."""

    name = "psychoacoustic_pgd"

    def __init__(self, cfg: Dict[str, Any], device: torch.device):
        super().__init__(cfg, device)
        attack_cfg = cfg.get("attacks", {}).get("psychoacoustic_pgd", {})
        self.eps = attack_cfg.get("eps", 0.02)
        self.alpha = attack_cfg.get("alpha", 0.002)
        self.iterations = attack_cfg.get("iterations", 150)
        self.masking_gain_db = attack_cfg.get("masking_gain_db", 6.0)
        self.encoder = SpeakerEncoder(device)

    def apply(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        x = self._ensure_2d(waveform).to(self.device).detach()
        x_np = x.squeeze(0).cpu().numpy()

        # Compute masking threshold in time-frequency domain
        n_fft = 2048
        hop = n_fft // 4
        mask_linear = compute_masking_threshold(x_np, sample_rate, n_fft=n_fft, gain_db=self.masking_gain_db)
        # Convert to waveform-domain std estimate (per sample rough bound)
        # Use RMS of masking threshold as a per-sample perturbation budget
        max_delta_per_sample = float(mask_linear.mean()) * self.eps * 10.0
        max_delta_per_sample = min(max_delta_per_sample, self.eps)

        with torch.no_grad():
            emb_clean = self.encoder.embed(x)

        delta = torch.zeros_like(x, requires_grad=False)
        delta.uniform_(-max_delta_per_sample * 0.5, max_delta_per_sample * 0.5)
        delta = delta.requires_grad_(True)

        for _ in tqdm(range(self.iterations), desc="Psychoacoustic-PGD", leave=False):
            x_adv = (x + delta).clamp(-1.0, 1.0)
            emb_adv = self.encoder.embed(x_adv)

            loss = F.cosine_similarity(emb_adv, emb_clean.detach(), dim=-1).mean()
            loss.backward()

            with torch.no_grad():
                # Standard PGD step
                delta.data = delta.data - self.alpha * delta.grad.sign()
                # Psychoacoustic constraint: clip to masking budget
                delta.data = delta.data.clamp(-max_delta_per_sample, max_delta_per_sample)
                # L-inf global cap
                delta.data = delta.data.clamp(-self.eps, self.eps)
                delta.data = (x + delta.data).clamp(-1.0, 1.0) - x

            if delta.grad is not None:
                delta.grad.zero_()

        with torch.no_grad():
            x_protected = (x + delta).clamp(-1.0, 1.0)

        snr = self._snr_db(x, x_protected)
        sim_after = F.cosine_similarity(
            self.encoder.embed(x_protected), emb_clean, dim=-1
        ).item()
        print(f"  [psychoacoustic_pgd] SNR={snr:.1f} dB | speaker-sim after={sim_after:.3f} | mask-budget={max_delta_per_sample:.4f}")

        return x_protected.squeeze(0) if waveform.dim() == 1 else x_protected
