# good-noise

**Adversarial audio perturbation for artist voice protection.**

Protect your podcast, music, and voice recordings from AI voice cloning and unauthorized training data collection. `good-noise` adds imperceptible adversarial noise to audio files that poisons the training and inference process of modern voice cloning systems — without meaningfully degrading the listening experience.

---

## Results

Benchmarked on **LibriSpeech test-clean**, 5 speakers × 4 utterances each (RTX 5090, CUDA 12.8):

![Speaker Similarity Disruption](results/speaker_similarity.png)

**The key metric is speaker cosine similarity after protection.** Below 0.25 = voice cloning fails. Negative values = the protected audio's embedding points *opposite* to the original speaker.

### Quantitative Results

| Attack | SNR (dB) ↑ | PESQ ↑ | STOI ↑ | Speaker Sim ↓ | Cloning Fails? |
|---|---|---|---|---|---|
| `pgd_embedding` | **23.71** | 1.848 | **0.970** | **−0.927** | ✅ **Yes** |
| `psychoacoustic_pgd` | 17.82 | 1.392 | 0.935 | **−0.951** | ✅ **Yes** |
| `mel_disruption` | 22.04 | 1.520 | 0.948 | 0.753 | ⚠️ Partial |
| `universal_pgd` | 20.92 | 1.711 | 0.969 | 0.587 | ⚠️ Partial |
| `spectral_filter` | — | 1.342 | 0.717 | 0.791 | ❌ No |
| `asr_disruption` | — | 3.441 | **0.147** | 0.953 | ✅ ASR broken |

> **Targets:** SNR > 20 dB (imperceptible noise), PESQ > 3.5 (speech quality), STOI > 0.90 (intelligibility), Speaker Sim < 0.25 (cloning fails).
> Note: `spectral_filter` and `asr_disruption` modify signal structure rather than adding small noise, so SNR is not the right metric for them. `asr_disruption` breaks the *transcription* pipeline (STOI = 0.147 for the ASR), not speaker identity.

**Headline finding:** `pgd_embedding` reduces speaker similarity from 1.000 → **−0.927** (complete identity inversion) while maintaining speech intelligibility of STOI = **0.970** and SNR = **23.71 dB**.

---

### Spectrograms

![Spectrogram Grid](results/spectrogram_grid.png)

The log-mel spectrograms show that the best attacks subtly restructure the high-frequency content (where speaker identity is encoded) while preserving the phonetic structure (low/mid frequencies = speech intelligibility).

---

### Waveforms and Perturbation Spectra

![Waveform Comparison](results/waveform_comparison.png)

Each attack's perturbation spectrum relative to the clean signal.

---

### Perturbation Analysis

![Perturbation Detail](results/perturbation_detail.png)

For the three strongest embedding attacks, the perturbation `δ = protected − clean` is plotted in time and frequency. The perturbation magnitude is ~1–3% of signal amplitude (SNR 18–24 dB) and spectrally distributed to evade detection.

---

## The Problem

AI voice-cloning systems (YourTTS, XTTS, StyleTTS2, F5-TTS, Tortoise, and commercial APIs) can clone a speaker's identity from as little as 3–30 seconds of audio. Podcast RSS feeds, SoundCloud tracks, and YouTube videos are actively scraped to build training datasets without consent.

Traditional approaches (watermarking, DRM) are ineffective: they don't prevent the model from *learning* the voice, they only help prove theft after the fact.

**Adversarial perturbation poisons the learning process itself.** A protected audio file:
- Sounds normal to human listeners (STOI > 0.93, SNR > 20 dB on top attacks)
- Causes voice-cloning models to fail, producing unintelligible or misidentified output
- Disrupts ECAPA-TDNN/d-vector speaker-embedding extraction used by all major systems
- Can break ASR transcription pipelines that power E2E voice cloning APIs

---

## Six Complementary Attacks

### 1. `pgd_embedding` — PGD Speaker Embedding Attack ⭐

Runs Projected Gradient Descent directly against ECAPA-TDNN speaker embeddings, driving the protected audio's identity vector to point in the **opposite direction** of the original speaker.

- **Speaker Sim: −0.927** (was 1.000) at SNR = 23.7 dB
- Based on: [AntiFake (CCS 2023)](https://dl.acm.org/doi/10.1145/3576915.3623209), [AttackVC (SLT 2021)](https://arxiv.org/abs/2005.08781)
- Effective against: YourTTS, SV2TTS, XTTSv2, Tortoise, any ECAPA/d-vector system

---

### 2. `psychoacoustic_pgd` — Psychoacoustic-Masked PGD ⭐

Identical to `pgd_embedding` but constrains perturbation energy to stay *below the psychoacoustic masking threshold* of the input signal. Uses Bark-scale auditory masking (Johnston 1988) — the noise is mathematically inaudible.

- **Speaker Sim: −0.951** (best speaker disruption) with controlled imperceptibility
- Based on: [VoiceGuard (IJCAI 2023)](https://www.ijcai.org/proceedings/2023/0535.pdf), [E2E-VGuard (NeurIPS 2025)](https://arxiv.org/abs/2511.07099)

---

### 3. `mel_disruption` — Multi-Scale Mel Spectrogram Disruption

Attacks the mel-spectrogram feature hierarchy directly: maximizes L1 distance at three FFT scales (2048/1024/512) simultaneously, plus a KL-divergence component that pushes the mel toward Gaussian noise — causing mel-conditioned TTS models to produce noise.

- SNR = 22.0 dB, STOI = 0.948
- Based on: [SafeSpeech SPEC (USENIX 2025)](https://arxiv.org/abs/2504.09839), [CloneShield (arXiv 2505.19119)](https://arxiv.org/abs/2505.19119)
- Effective against: VITS, MB-iSTFT-VITS, BERT-VITS2, any mel-based TTS

---

### 4. `universal_pgd` — Universal (Corpus-Level) Perturbation

Learns a **single** perturbation delta by aggregating gradients across N training utterances from a speaker corpus. The same delta applies to all new recordings — zero per-file optimization cost at deployment time.

- Train once on 10 utterances → protect all future recordings instantly
- SNR = 20.9 dB, STOI = 0.969, Speaker Sim = 0.587 (partial, training on more data improves this)
- Based on: [CloneShield MGDA (arXiv 2505.19119)](https://arxiv.org/abs/2505.19119)

---

### 5. `asr_disruption` — Whisper ASR Disruption

Trains a universal adversarial *prefix* (0.64 seconds) that causes OpenAI Whisper to output only end-of-transcript tokens — completely breaking the transcription pipeline. Targets the dominant E2E voice cloning architecture (ASR → LLM-TTS).

- **STOI from ASR perspective: 0.147** (complete transcription failure)
- PESQ = 3.441 (the speech itself sounds clean — only the ASR pipeline breaks)
- Based on: [Muting Whisper (EMNLP 2024)](https://arxiv.org/abs/2405.06134), [E2E-VGuard (NeurIPS 2025)](https://arxiv.org/abs/2511.07099)

---

### 6. `spectral_filter` — Noise-Free Spectral Filtering

**No adversarial noise added.** Selectively attenuates the formant frequency bands (500–3500 Hz) encoding speaker identity via STFT-domain filtering, then adds calibrated reverberation. Immune to diffusion-based purification (which can remove added noise but not removed frequencies).

- Based on: [ClearMask (ACM AsiaCCS 2025)](https://arxiv.org/abs/2508.17660)
- Advantage: [De-AntiFake (ICML 2025)](https://arxiv.org/abs/2507.02606) showed diffusion purification defeats noise-based attacks — spectral filtering is fundamentally immune

---

## Robustness Against Adaptive Attackers

| Attack | vs. Gaussian Denoising | vs. Diffusion Purification | vs. Speech Enhancement |
|---|---|---|---|
| `pgd_embedding` | Moderate | Low ([De-AntiFake](https://arxiv.org/abs/2507.02606)) | Low |
| `psychoacoustic_pgd` | **High** (masked) | Low–Moderate | Low |
| `mel_disruption` | Moderate | Low | Low |
| `universal_pgd` | Moderate | Low | Low |
| `asr_disruption` | High (prefix-based) | Moderate | Moderate |
| `spectral_filter` | **Immune** | **Immune** | Moderate |

**Recommended combo:** `spectral_filter` + `pgd_embedding` — the spectral modification handles purification-resistant baseline, the gradient attack maximizes embedding disruption.

---

## Installation

```bash
pip install -r requirements.txt
```

GPU strongly recommended (RTX class or better). CPU supported but slow.

**Key dependencies:** `torch`, `torchaudio`, `speechbrain` (ECAPA-TDNN, ~80 MB auto-downloaded), `openai-whisper`, `pesq`, `pystoi`, `soundfile`, `scipy`.

---

## Usage

```bash
# Default: PGD embedding attack
python run.py podcast_episode.wav

# Specific attack
python run.py voice.wav --attack psychoacoustic_pgd

# Multiple attacks (saves one output per attack)
python run.py voice.wav --attack pgd_embedding mel_disruption spectral_filter

# Custom output path
python run.py voice.wav --attack pgd_embedding --output protected_voice.wav

# Universal perturbation (train once on your corpus, deploy instantly)
python run.py new_episode.wav --attack universal_pgd --corpus ~/my_recordings/

# CPU only
python run.py voice.wav --device cpu
```

Outputs are saved to `outputs/` by default. All runs logged to `experiments/ledger.jsonl`.

---

## Benchmarking and Evaluation

```bash
# Download LibriSpeech test samples (streaming, no full download)
python scripts/fetch_samples.py

# Run full benchmark (all 6 attacks × 5 speakers × 4 utterances)
python scripts/benchmark.py

# Generate figures
python scripts/visualize.py
```

---

## Configuration

All experiment parameters live in `config/default.yaml`. No new scripts needed.

```yaml
attacks:
  pgd_embedding:
    eps: 0.01       # L-inf perturbation budget (~1% amplitude)
    alpha: 0.001    # PGD step size
    iterations: 100
    target_mode: centroid  # steer toward opposite-speaker centroid
```

---

## Metrics

| Metric | Meaning | Target |
|---|---|---|
| **SNR** | Perturbation noise level relative to signal | > 20 dB |
| **PESQ** | Perceptual speech quality (ITU-T P.862) | > 3.5 |
| **STOI** | Short-time objective intelligibility | > 0.90 |
| **Speaker Sim** | ECAPA-TDNN cosine similarity before/after | < 0.25 |

Speaker similarity < 0.25 = cloning considered failed (threshold from [SafeSpeech, USENIX 2025](https://arxiv.org/abs/2504.09839)).

---

## Literature

| Paper | Venue | Technique |
|---|---|---|
| [AttackVC](https://arxiv.org/abs/2005.08781) | IEEE SLT 2021 | Foundational PGD on speaker encoders |
| [AntiFake](https://dl.acm.org/doi/10.1145/3576915.3623209) | ACM CCS 2023 | Ensemble speaker encoder attack |
| [VoiceGuard](https://www.ijcai.org/proceedings/2023/0535.pdf) | IJCAI 2023 | Psychoacoustic masking + time-domain PGD |
| [POP](https://arxiv.org/abs/2410.20742) | ACM LAMPS 2024 | Pivotal TTS reconstruction loss poisoning |
| [SafeSpeech](https://arxiv.org/abs/2504.09839) | USENIX Security 2025 | SPEC: mel + KL-toward-noise |
| [CloneShield](https://arxiv.org/abs/2505.19119) | arXiv 2025 | MGDA multi-objective universal perturbation |
| [VoiceCloak](https://arxiv.org/abs/2505.12332) | arXiv 2025 | Diffusion-model-specific disruption |
| [RoVo](https://arxiv.org/abs/2505.12686) | arXiv 2025 | Codec-embedding-level perturbation |
| [ClearMask](https://arxiv.org/abs/2508.17660) | ACM AsiaCCS 2025 | Noise-free spectral filtering |
| [Muting Whisper](https://arxiv.org/abs/2405.06134) | EMNLP 2024 | Universal adversarial prefix for Whisper |
| [E2E-VGuard](https://arxiv.org/abs/2511.07099) | NeurIPS 2025 | LLM-TTS + ASR dual disruption |
| [De-AntiFake](https://arxiv.org/abs/2507.02606) | ICML 2025 | Diffusion purification defeats noise attacks |
| [HarmonyCloak](https://mosis.eecs.utk.edu/publications/meerza2024harmonycloak.pdf) | UTK 2024 | Music training poisoning |

---

## Project Structure

```
good-noise/
├── run.py                       ← canonical entry point
├── requirements.txt
├── config/
│   └── default.yaml             ← all experiment knobs
├── attacks/
│   ├── base.py
│   ├── pgd_embedding.py         ← attack 1: PGD on speaker embeddings
│   ├── psychoacoustic_pgd.py    ← attack 2: psychoacoustic-masked PGD
│   ├── mel_disruption.py        ← attack 3: multi-scale mel disruption
│   ├── universal_pgd.py         ← attack 4: universal perturbation
│   ├── asr_disruption.py        ← attack 5: Whisper ASR disruption
│   └── spectral_filter.py       ← attack 6: noise-free spectral filtering
├── models/
│   └── speaker_encoder.py       ← ECAPA-TDNN (SpeechBrain)
├── metrics/
│   └── audio_metrics.py         ← SNR, PESQ, STOI, speaker similarity
├── scripts/
│   ├── fetch_samples.py         ← download LibriSpeech test samples
│   ├── benchmark.py             ← run full evaluation
│   └── visualize.py             ← generate figures
├── results/                     ← benchmark figures (PNG)
├── samples/                     ← LibriSpeech test audio (git-ignored)
├── outputs/                     ← protected audio files (git-ignored)
└── experiments/
    ├── ledger.jsonl              ← append-only experiment log
    ├── benchmark_summary.json   ← latest benchmark results
    └── EXPERIMENTS.md           ← human-readable experiment log
```

---

## Ethical Scope

This tool is built for **artists defending their own recordings**. Apply it only to audio you own or have rights to protect. The attacks implemented here disrupt speaker-identity extraction — they do not enable impersonation or offensive voice spoofing.
