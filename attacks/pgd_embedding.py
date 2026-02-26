"""PGD Speaker Embedding Attack.

Technique:
    Maximizes the cosine distance between the perturbed audio's speaker
    embedding and the original embedding, optionally steering toward a
    target centroid (opposite-gender or random speaker).

    Based on: AntiFake (CCS 2023), AttackVC (SLT 2021).

    Optimizer: PGD (Projected Gradient Descent) with L-inf constraint.
    Loss: -cosine_similarity(embed(x + delta), embed(x))    [untargeted]
          +cosine_similarity(embed(x + delta), target_emb)  [targeted]
"""

from typing import Dict, Any, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .base import BaseAttack
from models.speaker_encoder import SpeakerEncoder


class PGDEmbeddingAttack(BaseAttack):
    """Iterative PGD attack on the speaker embedding space."""

    name = "pgd_embedding"

    def __init__(self, cfg: Dict[str, Any], device: torch.device):
        super().__init__(cfg, device)
        attack_cfg = cfg.get("attacks", {}).get("pgd_embedding", cfg.get("attacks", {}).get("pgd", {}))
        self.eps = attack_cfg.get("eps", 0.01)
        self.alpha = attack_cfg.get("alpha", 0.001)
        self.iterations = attack_cfg.get("iterations", 100)
        self.random_start = attack_cfg.get("random_start", True)
        self.target_mode = attack_cfg.get("target_mode", "centroid")
        self.encoder = SpeakerEncoder(device)

    def apply(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        target_waveforms: Optional[list] = None,
    ) -> torch.Tensor:
        """
        Args:
            waveform: (1, T) or (T,) float32 audio at 16 kHz.
            sample_rate: must be 16000.
            target_waveforms: optional list of waveforms for centroid target.

        Returns:
            Protected waveform of the same shape as input.
        """
        x = self._ensure_2d(waveform).to(self.device).detach()

        # Get clean embedding (no grad needed)
        with torch.no_grad():
            emb_clean = self.encoder.embed(x)

        # Optionally compute target embedding (for targeted attack)
        target_emb = None
        if self.target_mode == "centroid" and target_waveforms:
            with torch.no_grad():
                target_emb = self.encoder.embed_batch_mean(target_waveforms)
        elif self.target_mode == "random":
            target_emb = F.normalize(torch.randn_like(emb_clean), dim=-1)

        # Initialize delta
        if self.random_start:
            delta = torch.empty_like(x).uniform_(-self.eps, self.eps)
        else:
            delta = torch.zeros_like(x)
        delta.requires_grad_(True)

        for i in tqdm(range(self.iterations), desc="PGD-Embed", leave=False):
            x_adv = (x + delta).clamp(-1.0, 1.0)
            emb_adv = self.encoder.embed(x_adv)

            # Untargeted: maximize distance from clean embedding
            loss = F.cosine_similarity(emb_adv, emb_clean.detach(), dim=-1).mean()

            # Targeted: also pull toward target
            if target_emb is not None:
                loss = loss - F.cosine_similarity(emb_adv, target_emb.detach(), dim=-1).mean()

            loss.backward()

            with torch.no_grad():
                # Gradient sign ascent (maximizing cosine sim to clean = wrong direction
                # for protection; we want to MINIMIZE sim, so we ASCEND -sim = DESCEND sim)
                # loss = +sim_clean - sim_target, we're minimizing loss via gradient descent
                delta.data = delta.data - self.alpha * delta.grad.sign()
                delta.data = delta.data.clamp(-self.eps, self.eps)
                delta.data = (x + delta.data).clamp(-1.0, 1.0) - x

            if delta.grad is not None:
                delta.grad.zero_()

        with torch.no_grad():
            x_protected = (x + delta).clamp(-1.0, 1.0)

        snr = self._snr_db(x, x_protected)
        sim_after = F.cosine_similarity(
            self.encoder.embed(x_protected),
            emb_clean,
            dim=-1,
        ).item()
        print(f"  [pgd_embedding] SNR={snr:.1f} dB | speaker-sim after={sim_after:.3f} (was 1.000)")

        return x_protected.squeeze(0) if waveform.dim() == 1 else x_protected
