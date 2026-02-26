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

import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

from .base import BaseAttack


class _DifferentiableMelSpec(torch.nn.Module):
    """Differentiable mel spectrogram (fully compatible with autograd)."""

    def __init__(self, n_fft: int, n_mels: int, sample_rate: int, device: torch.device):
        super().__init__()
        hop = n_fft // 4
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop,
            n_mels=n_mels,
            power=1.0,  # amplitude spectrogram
        ).to(device)
        self.to(device)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: (1, T) or (T,)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        mel = self.mel(waveform)  # (1, n_mels, frames)
        return torch.log1p(mel)   # log-mel


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
