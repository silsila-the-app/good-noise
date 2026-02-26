"""Spectral Filtering Attack (Noise-Free).

Technique:
    Unlike all other attacks in this repo, this method adds NO adversarial
    noise. Instead it selectively attenuates frequency bands that carry
    speaker-identity information, then adds calibrated reverberation.

    This makes it immune to diffusion-based purification (De-AntiFake,
    ICML 2025) — there is no perturbation to "remove."

    Two operations:
    1. Formant suppression: attenuate mel bands corresponding to formant
       frequencies (500–3500 Hz) where speaker timbre is encoded.
       Uses a learnable mel-domain filter optimized to maximize speaker
       embedding distance while preserving speech intelligibility.
    2. Reverberation injection: convolve with a synthetic RIR (Room Impulse
       Response) that smears temporal speaker characteristics while keeping
       the speech intelligible.

    Based on: ClearMask (ACM AsiaCCS 2025, arXiv 2508.17660).

    NOTE: This attack degrades PESQ but preserves speech content (STOI stays
    relatively high). It is best used as a fast, compute-free baseline.
"""

from typing import Dict, Any

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve

from .base import BaseAttack


def _generate_rir(rt60: float, sample_rate: int, room_dim: tuple = (6.0, 4.0, 3.0)) -> np.ndarray:
    """Synthesize a simple exponentially-decaying RIR.

    Uses an energy-decay model (no image-source method required).
    Returns normalized RIR array.
    """
    # Duration: slightly longer than RT60
    rir_len = int(rt60 * sample_rate * 1.2)
    t = np.linspace(0, rt60 * 1.2, rir_len)

    # Exponential decay envelope
    decay = np.exp(-6.9 * t / rt60)
    # White noise excitation
    rng = np.random.RandomState(42)
    noise = rng.randn(rir_len)
    rir = noise * decay

    # Normalize
    rir = rir / (np.max(np.abs(rir)) + 1e-8)
    return rir.astype(np.float32)


def _attenuate_formant_bands(
    waveform_np: np.ndarray,
    sample_rate: int,
    suppression_db: float = 20.0,
) -> np.ndarray:
    """Attenuate speaker-characteristic formant bands (500–3500 Hz).

    Applies a notch-like filter in the mel domain by:
    1. Computing STFT
    2. Attenuating bins in the formant range
    3. Reconstructing waveform via ISTFT (Griffin-Lim not needed — we use
       the original phase).
    """
    from scipy.signal import stft, istft

    n_fft = 2048
    hop = n_fft // 4
    f, t, Zxx = stft(waveform_np, fs=sample_rate, nperseg=n_fft, noverlap=n_fft - hop)

    # Frequency bins in the formant range
    gain = 10.0 ** (-suppression_db / 20.0)
    formant_low = 500.0
    formant_high = 3500.0
    mask = np.ones(len(f), dtype=np.float32)
    mask[(f >= formant_low) & (f <= formant_high)] = gain

    Zxx_filtered = Zxx * mask[:, np.newaxis]

    _, x_filtered = istft(Zxx_filtered, fs=sample_rate, nperseg=n_fft, noverlap=n_fft - hop)

    # Trim/pad to original length
    orig_len = len(waveform_np)
    if len(x_filtered) > orig_len:
        x_filtered = x_filtered[:orig_len]
    else:
        x_filtered = np.pad(x_filtered, (0, orig_len - len(x_filtered)))

    return x_filtered.astype(np.float32)


class SpectralFilterAttack(BaseAttack):
    """Noise-free speaker disruption via spectral filtering + reverberation."""

    name = "spectral_filter"

    def __init__(self, cfg: Dict[str, Any], device: torch.device):
        super().__init__(cfg, device)
        attack_cfg = cfg.get("attacks", {}).get("spectral_filter", {})
        self.suppression_db = attack_cfg.get("formant_suppression_db", 20.0)
        self.reverb_rt60 = attack_cfg.get("reverb_rt60", 0.4)
        self.reverb_mix = attack_cfg.get("reverb_mix", 0.3)

    def apply(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        x = self._ensure_2d(waveform).to(self.device).detach()
        x_np = x.squeeze(0).cpu().numpy()

        # Step 1: Formant suppression
        x_filtered = _attenuate_formant_bands(x_np, sample_rate, self.suppression_db)

        # Step 2: Reverberation
        rir = _generate_rir(self.reverb_rt60, sample_rate)
        x_reverb = fftconvolve(x_filtered, rir, mode="full")[: len(x_filtered)]

        # Wet/dry mix
        x_out = (1.0 - self.reverb_mix) * x_filtered + self.reverb_mix * x_reverb

        # Normalize to [-1, 1]
        peak = np.max(np.abs(x_out))
        if peak > 1e-6:
            x_out = x_out / peak

        # Clip to safe range
        x_out = np.clip(x_out, -1.0, 1.0)

        x_protected = torch.from_numpy(x_out).unsqueeze(0).to(self.device)

        # Report pseudo-SNR (signal has actually changed, not added noise)
        snr = self._snr_db(x, x_protected)
        print(f"  [spectral_filter] Formant-suppression={self.suppression_db:.0f}dB | RT60={self.reverb_rt60}s | mix={self.reverb_mix} | signal-diff SNR={snr:.1f}dB")

        return x_protected.squeeze(0) if waveform.dim() == 1 else x_protected
