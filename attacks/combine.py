"""Combined Multi-Surface Attack.

The six individual attacks each target one weak point in the voice cloning
pipeline. This module runs a **single unified PGD loop** that optimizes all
attack surfaces simultaneously with a weighted composite loss:

    L = w_emb  · L_embedding          (speaker identity disruption)
      + w_mel  · L_mel_multiscale     (mel-feature disruption across 3 scales)
      + w_kl   · L_kl_noise           (drive mel toward Gaussian noise)
      + w_psy  · L_psychoacoustic     (psychoacoustic masking penalty)

Because all gradients flow through a single optimization loop, the solver
finds a perturbation that satisfies ALL objectives simultaneously — this is
strictly stronger than applying attacks independently and stacking deltas.
The loss landscape explored is the intersection of all individual attack
feasible regions, not their union.

─────────────────────────────────────────────────────────────────────────────
WEIGHT SELECTION

The weights (w_emb, w_mel, w_kl, w_psy) are the lever that determines the
character of the resulting protection. Choosing them well requires knowing:

  • The speaker's fundamental frequency and vocal tract length
  • The spectral density distribution (sparse vs. harmonically rich voices)
  • The target cloning architecture's dominant loss surface
  • The desired trade-off between SNR, PESQ, and disruption strength

**This weight selection logic is proprietary and not included in this
repository.** The open-source weights here are fixed defaults that work
well across the LibriSpeech benchmark. The GoodNoise API provides
per-file adaptive weight selection based on audio fingerprinting.

─────────────────────────────────────────────────────────────────────────────
"""

from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .base import BaseAttack
from .mel_disruption import _DifferentiableMelSpec, _make_mel_fb
from .psychoacoustic_pgd import compute_masking_threshold
from models.speaker_encoder import SpeakerEncoder


class CombineAttack(BaseAttack):
    """Joint multi-surface adversarial attack with weighted composite loss.

    All four attack surfaces are optimized in a single PGD pass:
      1. Speaker embedding distance (ECAPA-TDNN)
      2. Multi-scale mel-spectrogram disruption (3 FFT scales)
      3. KL divergence: drive mel toward Gaussian noise
      4. Psychoacoustic masking penalty (keep perturbation imperceptible)
    """

    name = "combine"

    def __init__(self, cfg: Dict[str, Any], device: torch.device):
        super().__init__(cfg, device)
        ccfg = cfg.get("attacks", {}).get("combine", {})

        self.eps        = ccfg.get("eps",        0.012)
        self.alpha      = ccfg.get("alpha",       0.0008)
        self.iterations = ccfg.get("iterations",  200)

        # Composite loss weights (open-source defaults)
        self.w_emb = ccfg.get("w_emb",  3.0)   # speaker embedding pull
        self.w_mel = ccfg.get("w_mel",  1.0)   # mel-space disruption
        self.w_kl  = ccfg.get("w_kl",   0.5)   # push mel toward noise
        self.w_psy = ccfg.get("w_psy",  0.8)   # psychoacoustic penalty

        self.masking_gain_db = ccfg.get("masking_gain_db", 6.0)
        self.n_mels  = ccfg.get("n_mels", 80)
        self.scales  = ccfg.get("n_fft_scales", [2048, 1024, 512])
        self.audio_sr = cfg.get("audio", {}).get("sample_rate", 16000)

        self.encoder = SpeakerEncoder(device)

        self.mel_transforms = [
            _DifferentiableMelSpec(n_fft, self.n_mels, self.audio_sr, device)
            for n_fft in self.scales
        ]

    # ──────────────────────────────────────────────────────────────────────

    def apply(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        weights: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        """Apply the combined attack.

        Args:
            waveform:    (1, T) or (T,) float32 at 16 kHz.
            sample_rate: must be 16000.
            weights:     Optional override dict with keys w_emb, w_mel, w_kl, w_psy.
                         If None, uses defaults from config (or proprietary API values
                         if supplied by the GoodNoise weight-selection service).

        Returns:
            Protected waveform, same shape as input.
        """
        if weights:
            self.w_emb = weights.get("w_emb", self.w_emb)
            self.w_mel = weights.get("w_mel", self.w_mel)
            self.w_kl  = weights.get("w_kl",  self.w_kl)
            self.w_psy = weights.get("w_psy", self.w_psy)

        x = self._ensure_2d(waveform).to(self.device).detach()
        x_np = x.squeeze(0).cpu().numpy()

        # ── Anchors ──────────────────────────────────────────────────────
        with torch.no_grad():
            emb_clean = self.encoder.embed(x)
            clean_mels = [mt(x).detach() for mt in self.mel_transforms]

        # ── Psychoacoustic budget ─────────────────────────────────────────
        mask_linear = compute_masking_threshold(
            x_np, sample_rate, n_fft=2048, gain_db=self.masking_gain_db
        )
        psy_budget = float(mask_linear.mean()) * self.eps * 10.0
        psy_budget = min(psy_budget, self.eps)

        # ── Initialize delta ──────────────────────────────────────────────
        delta = torch.zeros_like(x).uniform_(-self.eps * 0.3, self.eps * 0.3)
        delta.requires_grad_(True)

        # ── Unified PGD loop ──────────────────────────────────────────────
        for step in tqdm(range(self.iterations), desc="CombineAttack", leave=False):
            x_adv = (x + delta).clamp(-1.0, 1.0)

            # 1 · Speaker embedding loss
            emb_adv = self.encoder.embed(x_adv)
            l_emb = F.cosine_similarity(emb_adv, emb_clean.detach(), dim=-1).mean()

            # 2 · Multi-scale mel disruption + KL-toward-noise
            l_mel = torch.tensor(0.0, device=self.device, requires_grad=True)
            l_kl  = torch.tensor(0.0, device=self.device, requires_grad=True)
            for mt, clean_mel in zip(self.mel_transforms, clean_mels):
                adv_mel = mt(x_adv)
                # Maximize L1 distance in mel space
                l_mel = l_mel + (-F.l1_loss(adv_mel, clean_mel))
                # KL: drive mel distribution toward N(0,1)
                mean = adv_mel.mean()
                var  = adv_mel.var()
                l_kl = l_kl + (-0.5 * (1 + torch.log(var + 1e-8) - mean**2 - var))
            l_mel = l_mel / len(self.mel_transforms)
            l_kl  = l_kl  / len(self.mel_transforms)

            # 3 · Psychoacoustic penalty: penalize exceeding masking threshold
            delta_rms = (delta ** 2).mean().sqrt()
            l_psy = F.relu(delta_rms - psy_budget)

            # ── Composite loss ────────────────────────────────────────────
            loss = (
                self.w_emb * l_emb
                + self.w_mel * l_mel
                + self.w_kl  * l_kl
                + self.w_psy * l_psy
            )

            loss.backward()

            with torch.no_grad():
                delta.data = delta.data - self.alpha * delta.grad.sign()
                delta.data = delta.data.clamp(-self.eps, self.eps)
                delta.data = (x + delta.data).clamp(-1.0, 1.0) - x

            if delta.grad is not None:
                delta.grad.zero_()

        # ── Output ────────────────────────────────────────────────────────
        with torch.no_grad():
            x_protected = (x + delta).clamp(-1.0, 1.0)

        snr = self._snr_db(x, x_protected)
        sim_after = F.cosine_similarity(
            self.encoder.embed(x_protected), emb_clean, dim=-1
        ).item()
        print(
            f"  [combine] SNR={snr:.1f} dB | speaker-sim={sim_after:.3f} "
            f"| weights=(emb={self.w_emb}, mel={self.w_mel}, kl={self.w_kl}, psy={self.w_psy})"
        )

        return x_protected.squeeze(0) if waveform.dim() == 1 else x_protected
