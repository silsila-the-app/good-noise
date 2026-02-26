"""Universal Perturbation Attack.

Technique:
    Computes a SINGLE perturbation delta that, when added to ANY utterance
    from the corpus, disrupts speaker embedding extraction. The perturbation
    is utterance-agnostic and dataset-independent — one delta protects all.

    Inspired by CloneShield (arXiv 2505.19119) which uses MGDA for multi-
    objective optimization. We implement a simpler but effective gradient
    aggregation across N training utterances.

    Algorithm:
        delta = 0
        for epoch in range(iterations):
            for each utterance x_i in corpus:
                compute grad_i of speaker-embedding loss w.r.t. delta
            delta -= alpha * sign(mean(grad_i))   [aggregated gradient step]
            project delta to L-inf ball

    This perturbation then works universally: apply the learned delta to
    new, unseen utterances at inference time (no per-utterance optimization).
"""

from typing import Dict, Any, List, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .base import BaseAttack
from models.speaker_encoder import SpeakerEncoder


class UniversalPGDAttack(BaseAttack):
    """Universal (utterance-agnostic) perturbation against speaker encoders."""

    name = "universal_pgd"

    def __init__(self, cfg: Dict[str, Any], device: torch.device):
        super().__init__(cfg, device)
        attack_cfg = cfg.get("attacks", {}).get("universal_pgd", {})
        self.eps = attack_cfg.get("eps", 0.015)
        self.alpha = attack_cfg.get("alpha", 0.001)
        self.iterations = attack_cfg.get("iterations", 200)
        self.n_samples = attack_cfg.get("n_samples", 10)
        self.encoder = SpeakerEncoder(device)
        self._universal_delta: Optional[torch.Tensor] = None

    def train(self, corpus: List[torch.Tensor]) -> torch.Tensor:
        """Learn a universal delta from a list of training waveforms.

        Args:
            corpus: List of (1, T) or (T,) waveforms at 16 kHz.
                    If len(corpus) > self.n_samples, samples randomly.

        Returns:
            Universal delta tensor of shape (1, max_T) — padded for longest utterance.
        """
        import random

        if len(corpus) > self.n_samples:
            corpus = random.sample(corpus, self.n_samples)

        # Pad all to same length
        max_len = max(w.shape[-1] for w in corpus)
        padded = []
        for w in corpus:
            w = self._ensure_2d(w).to(self.device)
            pad = max_len - w.shape[-1]
            padded.append(F.pad(w, (0, pad)))

        # Initialize universal delta
        delta = torch.zeros(1, max_len, device=self.device)
        delta.uniform_(-self.eps * 0.5, self.eps * 0.5)
        delta.requires_grad_(True)

        # Get clean embeddings
        with torch.no_grad():
            clean_embs = [self.encoder.embed(w[:, :max_len]) for w in padded]

        for step in tqdm(range(self.iterations), desc="Universal-PGD training", leave=False):
            total_grad = torch.zeros_like(delta.data)

            for w, emb_clean in zip(padded, clean_embs):
                x_adv = (w + delta[:, : w.shape[-1]]).clamp(-1.0, 1.0)
                emb_adv = self.encoder.embed(x_adv)

                loss = F.cosine_similarity(emb_adv, emb_clean.detach(), dim=-1).mean()
                loss.backward()

                if delta.grad is not None:
                    total_grad += delta.grad.data
                    delta.grad.zero_()

            with torch.no_grad():
                # Gradient sign aggregation (mean across samples)
                delta.data = delta.data - self.alpha * total_grad.sign()
                delta.data = delta.data.clamp(-self.eps, self.eps)

        self._universal_delta = delta.detach()
        return self._universal_delta

    def apply(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Apply pre-trained universal delta (or train on this single sample)."""
        x = self._ensure_2d(waveform).to(self.device).detach()

        if self._universal_delta is None:
            # No pre-trained delta: train on this utterance alone (still useful)
            print("  [universal_pgd] No pre-trained delta. Running single-utterance training.")
            self.train([waveform])

        # Trim or pad delta to match input length
        T = x.shape[-1]
        delta = self._universal_delta
        if delta.shape[-1] < T:
            delta = F.pad(delta, (0, T - delta.shape[-1]))
        else:
            delta = delta[:, :T]

        with torch.no_grad():
            x_protected = (x + delta).clamp(-1.0, 1.0)

        snr = self._snr_db(x, x_protected)
        with torch.no_grad():
            emb_clean = self.encoder.embed(x)
            sim_after = F.cosine_similarity(
                self.encoder.embed(x_protected), emb_clean, dim=-1
            ).item()
        print(f"  [universal_pgd] SNR={snr:.1f} dB | speaker-sim after={sim_after:.3f}")

        return x_protected.squeeze(0) if waveform.dim() == 1 else x_protected
