"""ASR Disruption Attack (Whisper).

Technique:
    Constructs a universal adversarial audio prefix that, when prepended to
    any speech input, causes OpenAI Whisper to output only end-of-sequence
    tokens (complete transcription failure). This dual-disrupts:

      1. Voice cloning pipelines that use ASR → TTS (E2E pipelines), and
      2. Any downstream processing that depends on accurate transcripts.

    Based on: "Muting Whisper" (EMNLP 2024, arXiv 2405.06134).

    Loss:
        L_asr = CTC/cross-entropy of Whisper decoder toward [EOS] token,
                given the adversarial prefix prepended to input audio.

    The prefix is a learnable tensor of length `prefix_seconds * sample_rate`.
    Optimization: PGD on the prefix (input-independent — universal attack).

    NOTE on Whisper architecture:
        Whisper encodes log-mel spectrograms. We optimize delta in the
        waveform space with autograd flowing through torchaudio's mel pipeline
        into Whisper's encoder. The EOS token ID for Whisper large/medium/small
        is 50257 (encoder_eos_token_id).
"""

from typing import Dict, Any

import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

from .base import BaseAttack


class ASRDisruptionAttack(BaseAttack):
    """Trains a universal adversarial prefix that silences Whisper's output."""

    name = "asr_disruption"

    def __init__(self, cfg: Dict[str, Any], device: torch.device):
        super().__init__(cfg, device)
        attack_cfg = cfg.get("attacks", {}).get("asr_disruption", {})
        self.eps = attack_cfg.get("eps", 0.02)
        self.alpha = attack_cfg.get("alpha", 0.002)
        self.iterations = attack_cfg.get("iterations", 200)
        self.prefix_seconds = attack_cfg.get("prefix_seconds", 0.64)
        self.target_token = attack_cfg.get("target_token", "eos")
        self._whisper = None
        self._processor = None
        self._prefix_delta: torch.Tensor | None = None

    def _load_whisper(self):
        if self._whisper is not None:
            return
        try:
            import whisper as openai_whisper
            self._whisper_type = "openai"
            self._whisper = openai_whisper.load_model("base", device=self.device)
            self._whisper.eval()
            # EOS token id for openai whisper
            self._eos_id = self._whisper.dims.n_vocab - 1  # 50256 for base
        except ImportError:
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            self._whisper_type = "hf"
            self._processor = WhisperProcessor.from_pretrained("openai/whisper-base")
            self._whisper = WhisperForConditionalGeneration.from_pretrained(
                "openai/whisper-base"
            ).to(self.device)
            self._whisper.eval()
            self._eos_id = self._processor.tokenizer.eos_token_id

    def _log_mel(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Compute Whisper-compatible log-mel spectrogram (differentiable)."""
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            hop_length=160,
            n_mels=80,
            window_fn=torch.hann_window,
        ).to(self.device)
        mel = mel_transform(waveform.squeeze(0))  # (80, frames)
        log_mel = torch.log(mel.clamp(min=1e-10))
        # Whisper expects shape (batch, 80, 3000); pad/crop
        target_len = 3000
        if log_mel.shape[-1] < target_len:
            log_mel = F.pad(log_mel, (0, target_len - log_mel.shape[-1]))
        else:
            log_mel = log_mel[..., :target_len]
        return log_mel.unsqueeze(0)  # (1, 80, 3000)

    def train_prefix(self, sample_rate: int = 16000) -> torch.Tensor:
        """Optimize the adversarial prefix to maximize Whisper EOS probability."""
        self._load_whisper()
        prefix_len = int(self.prefix_seconds * sample_rate)
        prefix = torch.zeros(1, prefix_len, device=self.device)
        prefix.uniform_(-self.eps * 0.5, self.eps * 0.5)
        prefix.requires_grad_(True)

        # EOS target token sequence for cross-entropy
        target = torch.tensor([self._eos_id], device=self.device)

        for _ in tqdm(range(self.iterations), desc="ASR-Disruption prefix training", leave=False):
            log_mel = self._log_mel(prefix, sample_rate)

            if self._whisper_type == "openai":
                # openai whisper: model.encoder + decoder with forced_tokens
                audio_features = self._whisper.encoder(log_mel)
                # Decode toward EOS: compute logits for first output token
                tokens = torch.tensor([[self._whisper.dims.sot]], device=self.device)
                logits = self._whisper.decoder(tokens, audio_features)  # (1, 1, vocab)
                logits = logits[:, -1, :]  # last token
                loss = F.cross_entropy(logits, target)
            else:
                # HuggingFace Whisper
                outputs = self._whisper(
                    input_features=log_mel,
                    decoder_input_ids=torch.tensor(
                        [[self._whisper.config.decoder_start_token_id]], device=self.device
                    ),
                )
                logits = outputs.logits[:, -1, :]  # (1, vocab)
                loss = F.cross_entropy(logits, target)

            loss.backward()

            with torch.no_grad():
                prefix.data = prefix.data - self.alpha * prefix.grad.sign()
                prefix.data = prefix.data.clamp(-self.eps, self.eps).clamp(-1.0, 1.0)

            if prefix.grad is not None:
                prefix.grad.zero_()

        self._prefix_delta = prefix.detach()
        return self._prefix_delta

    def apply(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Prepend the adversarial prefix to the input waveform."""
        self._load_whisper()
        x = self._ensure_2d(waveform).to(self.device).detach()

        if self._prefix_delta is None:
            print("  [asr_disruption] Training adversarial prefix...")
            self.train_prefix(sample_rate)

        prefix = self._prefix_delta
        # Clamp prefix and concatenate
        prefix_clamped = prefix.clamp(-1.0, 1.0)
        x_protected = torch.cat([prefix_clamped, x], dim=-1)

        # Normalize to [-1, 1] after concatenation
        peak = x_protected.abs().max()
        if peak > 1.0:
            x_protected = x_protected / peak

        print(f"  [asr_disruption] Prepended {self.prefix_seconds:.2f}s adversarial prefix | total length={x_protected.shape[-1]/sample_rate:.2f}s")

        return x_protected.squeeze(0) if waveform.dim() == 1 else x_protected
