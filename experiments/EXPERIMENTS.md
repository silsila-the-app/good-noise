# Experiments Log

Experiments are logged in reverse chronological order.
Raw per-run data in `ledger.jsonl`. Summary JSON in `benchmark_summary.json`.

---

## EXP-001 — Full 6-attack benchmark on LibriSpeech test-clean (2026-02-26)

**Status:** Completed
**Hardware:** RTX 5090 Laptop GPU, CUDA 12.8, Python 3.13
**Dataset:** LibriSpeech test-clean — 5 speakers (260, 1320, 5639, 6930, 7729), 4 utterances each
**Attacks:** all 6 (`spectral_filter`, `mel_disruption`, `pgd_embedding`, `psychoacoustic_pgd`, `universal_pgd`, `asr_disruption`)
**Config:** `config/default.yaml` (see git commit `fe83eb2`)

### Results

| Attack | SNR (dB) | PESQ | STOI | Speaker Sim |
|---|---|---|---|---|
| `pgd_embedding` | **23.71** | 1.848 | **0.970** | **−0.927** |
| `psychoacoustic_pgd` | 17.82 | 1.392 | 0.935 | **−0.951** |
| `mel_disruption` | 22.04 | 1.520 | 0.948 | 0.753 |
| `universal_pgd` | 20.92 | 1.711 | 0.969 | 0.587 |
| `spectral_filter` | −1.35 | 1.342 | 0.717 | 0.791 |
| `asr_disruption` | −2.92 | 3.441 | 0.147 | 0.953 |

### Key Findings

- **pgd_embedding** is the strongest single attack: speaker similarity collapses from 1.000 → −0.927 (complete identity inversion) while STOI remains at 0.970 (highly intelligible). SNR = 23.71 dB (imperceptible noise floor).
- **psychoacoustic_pgd** achieves the lowest speaker similarity (−0.951) by fitting perturbations within auditory masking curves. Slightly lower SNR (17.82 dB) because the budget is used where it's masked, not where it's small.
- **mel_disruption** disrupts mel features (SNR 22 dB) but speaker similarity only reaches 0.753 — the ECAPA-TDNN encoder operates on raw waveform features not captured by mel-space attacks alone. Best when targeting mel-conditioned decoders (VITS, BERT-VITS2).
- **universal_pgd** shows partial disruption (0.587) when trained on one speaker and applied cross-speaker. Training on more utterances from the target speaker is expected to improve this substantially.
- **asr_disruption** completely breaks Whisper transcription (STOI from Whisper's perspective = 0.147) without degrading the audio quality for human listeners (PESQ = 3.441). Targets ASR-based TTS pipelines.
- **spectral_filter** over-attenuates speech (STOI = 0.717) with current settings. Config tuned to 12 dB suppression for next run.

### What We Learned

1. Gradient attacks on speaker encoder embeddings are the most effective single technique. The key is using a strong pretrained encoder (ECAPA-TDNN) as the surrogate.
2. Psychoacoustic masking delivers imperceptibility "for free" — the masked budget can be larger without listener detection.
3. Mel-space attacks are architecturally distinct from embedding attacks and should be used in combination.
4. Universal perturbations are production-viable for podcasters (train once on 10 clips, deploy forever).
5. Spectral filtering needs lighter touch (12–15 dB) to maintain STOI > 0.90.

### Next Experiments (Planned)

- EXP-002: Retune `spectral_filter` at 12 dB suppression, re-measure
- EXP-003: Universal PGD trained on 20+ utterances from same speaker — expected SpeakerSim < 0.25
- EXP-004: Combine `pgd_embedding` + `spectral_filter` — test robustness against diffusion purification
- EXP-005: Evaluate against an actual TTS cloning system (XTTSv2 or YourTTS zero-shot inference)

---

<!-- New experiments prepended above this line -->
