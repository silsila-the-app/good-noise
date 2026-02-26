"""Pretrained ECAPA-TDNN speaker encoder wrapper (SpeechBrain).

Downloads model weights on first use (~80 MB). Subsequent calls use cache.
"""

import torch
import torch.nn.functional as F
from pathlib import Path

_SAVEDIR = str(Path.home() / ".cache" / "good-noise" / "ecapa-tdnn")


class SpeakerEncoder:
    """Wraps SpeechBrain's ECAPA-TDNN for differentiable speaker embedding extraction.

    Usage:
        enc = SpeakerEncoder(device)
        emb = enc.embed(waveform_16khz)   # (1, 192) float32 tensor, differentiable
        sim = enc.cosine_similarity(a, b)
    """

    def __init__(self, device: torch.device):
        self.device = device
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:
            from speechbrain.pretrained import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy

        self._model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=_SAVEDIR,
            run_opts={"device": str(self.device)},
            local_strategy=LocalStrategy.COPY,
        )
        self._model.eval()

    def embed(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract speaker embedding.

        Args:
            waveform: (1, T) or (T,) float32 at 16 kHz, values in [-1, 1].

        Returns:
            Embedding tensor of shape (1, 192), differentiable w.r.t. waveform.
        """
        self._load()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # SpeechBrain expects (batch, time) on the model device
        wav = waveform.to(self.device)
        lens = torch.tensor([1.0], device=self.device)

        # Use encode_batch which supports autograd
        emb = self._model.encode_batch(wav, wav_lens=lens)  # (1, 1, 192)
        return emb.squeeze(1)  # (1, 192)

    @staticmethod
    def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Cosine similarity between two embedding tensors."""
        return F.cosine_similarity(a, b, dim=-1)

    def embed_batch_mean(self, waveforms: list) -> torch.Tensor:
        """Compute mean embedding across a list of waveforms (for centroid target)."""
        embs = [self.embed(w) for w in waveforms]
        return torch.stack(embs, dim=0).mean(dim=0)
