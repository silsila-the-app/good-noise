"""Multi-Scale Mel Spectrogram Disruption Attack.

Technique:
    Optimizes perturbation to maximize disruption of mel-spectrogram features
    across multiple FFT scales simultaneously. Also includes a KL-toward-noise
    component that pushes the perturbed mel toward Gaussian noise (degrading
    any model that uses mel as input for conditioning or reconstruction).

    Based on: SafeSpeech SPEC loss (USENIX 2025), CloneShield multi-scale mel
    (arXiv 2505.19119), POP pivotal objective perturbation (ACM LAMPS 2024).

    Loss:
        L_mel   = sum_s ||mel_s(x') - mel_s(x)||_1     (mel distance, maximized)
        L_kl    = -KL(mel(x') || N(0,1))               (drive toward noise, maximized)
        L_total = L_mel + kl_weight * L_kl
"""

from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .base import BaseAttack


def _make_mel_fb(n_fft: int, n_mels: int, sr: int) -> np.ndarray:
    """Returns mel filterbank (n_mels, n_fft//2+1)."""
    f_min, f_max = 0.0, sr / 2.0
    m_min = 2595 * np.log10(1 + f_min / 700)
    m_max = 2595 * np.log10(1 + f_max / 700)
    mel_pts = np.linspace(m_min, m_max, n_mels + 2)
    hz_pts = 700 * (10 ** (mel_pts / 2595) - 1)
    bins = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    fbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        s, c, e = bins[m-1], bins[m], bins[m+1]
        for k in range(s, c):
            if c > s: fbank[m-1, k] = (k - s) / (c - s)
        for k in range(c, e):
            if e > c: fbank[m-1, k] = (e - k) / (e - c)
    return fbank


class _DifferentiableMelSpec(torch.nn.Module):
    """Pure-PyTorch differentiable mel spectrogram (no torchaudio needed)."""

    def __init__(self, n_fft: int, n_mels: int, sample_rate: int, device: torch.device):
        super().__init__()
        self.n_fft = n_fft
        self.hop = n_fft // 4
        self.register_buffer("window", torch.hann_window(n_fft))
        fb = torch.from_numpy(_make_mel_fb(n_fft, n_mels, sample_rate))
        self.register_buffer("mel_fb", fb)  # (n_mels, n_fft//2+1)
        self.to(device)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 2:
            waveform = waveform.squeeze(0)
        stft = torch.stft(waveform, n_fft=self.n_fft, hop_length=self.hop,
                          window=self.window, return_complex=True, center=True)
        mag = stft.abs()  # (freq_bins, frames)
        mel = self.mel_fb @ mag  # (n_mels, frames)
        return torch.log1p(mel)  # log-mel


class MelDisruptionAttack(BaseAttack):
    """Multi-scale mel disruption with optional KL-toward-noise component."""

    name = "mel_disruption"

    def __init__(self, cfg: Dict[str, Any], device: torch.device):
        super().__init__(cfg, device)
        attack_cfg = cfg.get("attacks", {}).get("mel_disruption", {})
        self.eps = attack_cfg.get("eps", 0.01)
        self.alpha = attack_cfg.get("alpha", 0.001)
        self.iterations = attack_cfg.get("iterations", 100)
        self.kl_weight = attack_cfg.get("kl_weight", 0.5)
        self.n_mels = attack_cfg.get("n_mels", 80)
        scales = attack_cfg.get("n_fft_scales", [2048, 1024, 512])
        self.audio_sr = cfg.get("audio", {}).get("sample_rate", 16000)

        self.mel_transforms: List[_DifferentiableMelSpec] = [
            _DifferentiableMelSpec(n_fft, self.n_mels, self.audio_sr, device)
            for n_fft in scales
        ]

    def apply(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        x = self._ensure_2d(waveform).to(self.device).detach()

        # Compute clean mel specs at each scale
        with torch.no_grad():
            clean_mels = [mt(x).detach() for mt in self.mel_transforms]

        delta = torch.zeros_like(x).uniform_(-self.eps * 0.5, self.eps * 0.5)
        delta.requires_grad_(True)

        for _ in tqdm(range(self.iterations), desc="Mel-Disruption", leave=False):
            x_adv = (x + delta).clamp(-1.0, 1.0)

            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            for mt, clean_mel in zip(self.mel_transforms, clean_mels):
                adv_mel = mt(x_adv)

                # Maximize L1 distance in mel space — disrupts any mel-conditioned model
                l_mel = -F.l1_loss(adv_mel, clean_mel)

                # KL divergence: push adv mel distribution toward N(0,1)
                # Treat mel values as un-normalized log-probs, compute soft KL
                mean = adv_mel.mean()
                var = adv_mel.var()
                l_kl = -0.5 * (1 + torch.log(var + 1e-8) - mean ** 2 - var)

                loss = loss + l_mel + self.kl_weight * l_kl

            loss = loss / len(self.mel_transforms)
            loss.backward()

            with torch.no_grad():
                delta.data = delta.data - self.alpha * delta.grad.sign()
                delta.data = delta.data.clamp(-self.eps, self.eps)
                delta.data = (x + delta.data).clamp(-1.0, 1.0) - x

            if delta.grad is not None:
                delta.grad.zero_()

        with torch.no_grad():
            x_protected = (x + delta).clamp(-1.0, 1.0)

        snr = self._snr_db(x, x_protected)
        with torch.no_grad():
            mel_dist = sum(
                F.l1_loss(mt(x_protected), cm).item()
                for mt, cm in zip(self.mel_transforms, clean_mels)
            ) / len(self.mel_transforms)
        print(f"  [mel_disruption] SNR={snr:.1f} dB | avg mel-L1-dist={mel_dist:.4f}")

        return x_protected.squeeze(0) if waveform.dim() == 1 else x_protected
