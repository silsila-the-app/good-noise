# good-noise

**Adversarial audio perturbation for artist voice protection.**

`good-noise` lets podcasters, musicians, and voice actors add imperceptible noise to their recordings before publishing — protecting their audio from unauthorized AI training and voice cloning without meaningfully degrading the listening experience.

---

## The Problem

AI voice-cloning systems (YourTTS, XTTS, StyleTTS2, F5-TTS, Tortoise, and commercial APIs) can clone a speaker's identity from as little as 3–30 seconds of audio. Podcast RSS feeds, SoundCloud tracks, and YouTube videos are actively scraped to build these training datasets without consent.

Traditional approaches (watermarking, DRM) are ineffective: they don't prevent the model from *learning* the voice, they only help prove theft after the fact.

**Adversarial perturbation poisons the learning process itself.** A protected audio file:
- Sounds normal to human listeners (PESQ > 3.5, STOI > 0.90)
- Causes voice-cloning models to fail, producing unintelligible or misidentified output
- Disrupts speaker-embedding extraction, making fine-tuned models clone the wrong person

---

## Techniques Implemented

Six complementary attacks are included, targeting different architectural weak points in AI voice pipelines. Use them individually or combine them.

---

### 1. `pgd_embedding` — PGD Speaker Embedding Attack

**What it does:** Runs Projected Gradient Descent (PGD) directly against a pretrained ECAPA-TDNN speaker encoder. Optimizes an imperceptible perturbation that maximizes the cosine distance between the protected audio's speaker embedding and the original — making the audio appear to be a completely different speaker.

**Based on:** [AntiFake (CCS 2023)](https://dl.acm.org/doi/10.1145/3576915.3623209), [AttackVC (SLT 2021)](https://arxiv.org/abs/2005.08781)

**Target:** Speaker encoders used in YourTTS, SV2TTS, XTTSv2, Tortoise, and most zero-shot voice cloning systems.

**Key parameters:**
| Parameter | Default | Effect |
|---|---|---|
| `eps` | 0.01 | L∞ perturbation budget (~1% amplitude) |
| `alpha` | 0.001 | PGD step size |
| `iterations` | 100 | More = stronger but slower |
| `target_mode` | `centroid` | Steer toward opposite-speaker centroid |

**Expected results:**
| Metric | Protected Audio | Target |
|---|---|---|
| SNR | ~34 dB | >20 dB |
| PESQ | >3.5 | >3.5 |
| STOI | >0.95 | >0.90 |
| Speaker Similarity | <0.30 | <0.25 |

---

### 2. `psychoacoustic_pgd` — Psychoacoustic-Masked PGD

**What it does:** Identical optimization objective to `pgd_embedding`, but constrains perturbation energy to lie *below the psychoacoustic masking threshold* of the input signal. Uses a Bark-scale model (Johnston 1988) to compute per-frequency masking budgets. Perturbations are mathematically inaudible.

**Based on:** [VoiceGuard (IJCAI 2023)](https://www.ijcai.org/proceedings/2023/0535.pdf), [E2E-VGuard (NeurIPS 2025)](https://arxiv.org/abs/2511.07099)

**Key insight:** The human auditory system is most sensitive between 1–4 kHz and least sensitive in frequency regions already dominated by a loud signal. Perturbations placed in masked regions are perceptually invisible.

**Key parameters:**
| Parameter | Default | Effect |
|---|---|---|
| `eps` | 0.02 | Slightly larger budget — masking hides it |
| `masking_gain_db` | 6.0 | Safety margin below threshold (higher = more conservative) |
| `bark_bands` | 24 | Frequency resolution of masking model |

**Expected results:** Higher PESQ than `pgd_embedding` due to masking-constrained perturbation (noise placed in already-loud frequency bands).

---

### 3. `mel_disruption` — Multi-Scale Mel Spectrogram Disruption

**What it does:** Optimizes perturbation to maximize disruption of mel-spectrogram features at three FFT scales simultaneously (2048, 1024, 512). Adds a KL-divergence component that pushes the perturbed mel-spectrogram toward Gaussian noise — causing any model using the mel for conditioning or reconstruction to generate noise-like output.

**Based on:** [SafeSpeech SPEC loss (USENIX 2025)](https://arxiv.org/abs/2504.09839), [CloneShield multi-scale mel (arXiv 2505.19119)](https://arxiv.org/abs/2505.19119), [POP (ACM LAMPS 2024)](https://arxiv.org/abs/2410.20742)

**Why multi-scale?** Voice cloning decoders operate at different temporal resolutions. Attacking all three scales simultaneously disrupts the full feature hierarchy.

**Loss function:**
```
L_total = -||mel(x') - mel(x)||₁          (maximize mel distance)
          - kl_weight · KL(mel(x') || N(0,1))  (drive mel toward noise)
```

**Key parameters:**
| Parameter | Default | Effect |
|---|---|---|
| `kl_weight` | 0.5 | Weight of noise-driving component |
| `n_fft_scales` | [2048, 1024, 512] | FFT scales to attack |

---

### 4. `universal_pgd` — Universal (Corpus-Level) Perturbation

**What it does:** Learns a *single* perturbation delta that works universally across all utterances from a speaker. Aggregates gradients across N training samples, then the same delta is applied to new, unseen recordings without any per-file optimization.

**Based on:** [CloneShield MGDA (arXiv 2505.19119)](https://arxiv.org/abs/2505.19119)

**Key advantage:** After a one-time training pass over N samples (~5–10 minutes of audio), protection of new recordings is instantaneous — no per-file GPU computation needed.

**Usage:**
```bash
# Train on corpus, apply to new file
python run.py new_episode.wav --attack universal_pgd --corpus my_audio_samples/
```

**Key parameters:**
| Parameter | Default | Effect |
|---|---|---|
| `n_samples` | 10 | Training utterances (more = more universal) |
| `iterations` | 200 | More = stronger universal delta |
| `eps` | 0.015 | Slightly larger than per-file attacks |

---

### 5. `asr_disruption` — Whisper ASR Disruption

**What it does:** Trains a universal adversarial audio *prefix* (0.64 seconds) that, when prepended to any speech recording, causes OpenAI Whisper to output only end-of-sequence tokens — producing an empty or nonsensical transcript. This disrupts voice cloning pipelines that use ASR → TTS (the dominant architecture for commercial APIs like Azure, ElevenLabs) and downstream NLP pipelines.

**Based on:** ["Muting Whisper" (EMNLP 2024)](https://arxiv.org/abs/2405.06134), [E2E-VGuard (NeurIPS 2025)](https://arxiv.org/abs/2511.07099)

**Why Whisper?** Whisper is used as the transcription backbone in a large fraction of production voice pipelines (ElevenLabs, Microsoft Azure STT, many open-source voice cloning systems). Corrupting its output breaks the text conditioning in TTS.

**Loss function:**
```
L_asr = CrossEntropy(Whisper_logits(prefix + x), EOS_token)
```

**Note:** The prefix adds ~0.64 seconds to the audio. This is disclosed to listeners (e.g., a short silence or low-level tone).

---

### 6. `spectral_filter` — Noise-Free Spectral Filtering

**What it does:** Unlike all other attacks, this method adds *no adversarial noise*. Instead, it selectively attenuates the formant frequency bands (500–3500 Hz) that encode speaker identity, then adds calibrated reverberation. The result is immune to diffusion-based purification (which can remove added noise but cannot restore removed frequencies).

**Based on:** [ClearMask (ACM AsiaCCS 2025)](https://arxiv.org/abs/2508.17660)

**Key insight from De-AntiFake (ICML 2025):** Diffusion purification can remove waveform-level adversarial noise. A noise-free approach based on spectral modification is fundamentally immune to this class of adaptive attack.

**Operations:**
1. **Formant suppression:** Attenuate 500–3500 Hz by `formant_suppression_db` dB using STFT-domain filtering (phase preserved → minimal artifact)
2. **Reverberation injection:** Convolve with synthetic RIR (exponentially-decaying white noise) to smear temporal speaker characteristics

**Key parameters:**
| Parameter | Default | Effect |
|---|---|---|
| `formant_suppression_db` | 20 | Attenuation of speaker-critical bands |
| `reverb_rt60` | 0.4 | Reverberation decay time (seconds) |
| `reverb_mix` | 0.3 | Wet/dry reverberation mix |

---

## Robustness Against Adaptive Attackers

| Attack | Robust vs. Gaussian Denoising | Robust vs. Diffusion Purification | Robust vs. Speech Enhancement |
|---|---|---|---|
| `pgd_embedding` | Moderate | Low (De-AntiFake) | Low |
| `psychoacoustic_pgd` | High (masked) | Low-Moderate | Low |
| `mel_disruption` | Moderate | Low | Low |
| `universal_pgd` | Moderate | Low | Low |
| `asr_disruption` | High (prefix) | Moderate | Moderate |
| `spectral_filter` | **Immune** | **Immune** | Moderate |

**Recommendation:** For maximum robustness, combine `spectral_filter` with `pgd_embedding` or `psychoacoustic_pgd`. The spectral modification handles the purification-resistant baseline; the gradient-based attack maximizes embedding disruption.

---

## Installation

```bash
pip install -r requirements.txt
```

**GPU strongly recommended** for gradient-based attacks. CPU is supported but slow (~10–30 minutes per file at 100 iterations).

### Dependencies
- `torch`, `torchaudio` — core tensor ops and audio I/O
- `speechbrain` — pretrained ECAPA-TDNN speaker encoder (auto-downloaded ~80 MB)
- `openai-whisper` — for ASR disruption attack
- `pesq`, `pystoi` — perceptual quality metrics
- `soundfile`, `librosa` — audio I/O
- `scipy`, `numpy` — signal processing (psychoacoustic masking, RIR synthesis)

---

## Usage

```bash
# Default: PGD embedding attack
python run.py podcast_episode.wav

# Specify attack
python run.py voice.wav --attack psychoacoustic_pgd

# Multiple attacks (saves one output per attack)
python run.py voice.wav --attack pgd_embedding mel_disruption spectral_filter

# Custom output path
python run.py voice.wav --attack pgd_embedding --output protected/voice_protected.wav

# Universal perturbation (train on corpus first)
python run.py voice.wav --attack universal_pgd --corpus ~/my_recordings/

# CPU-only
python run.py voice.wav --device cpu
```

Outputs are saved to `outputs/` by default. All runs are logged to `experiments/ledger.jsonl`.

---

## Configuration

All knobs live in `config/default.yaml`. No new scripts needed — vary parameters there.

```yaml
attacks:
  active: [pgd_embedding]       # default attack(s)

  pgd_embedding:
    eps: 0.01                   # perturbation budget
    alpha: 0.001                # step size
    iterations: 100
```

---

## Metrics and Evaluation

After each run, four metrics are reported:

| Metric | Meaning | Target |
|---|---|---|
| **SNR** | Signal-to-noise ratio of perturbation | >20 dB (imperceptible) |
| **PESQ** | Perceptual speech quality (ITU-T P.862) | >3.5 (acceptable) |
| **STOI** | Short-time objective intelligibility | >0.90 (fully intelligible) |
| **Speaker Similarity** | Cosine sim of embeddings before/after | <0.25 (cloning fails) |

The speaker similarity threshold of 0.25 comes from [SafeSpeech (USENIX 2025)](https://arxiv.org/abs/2504.09839): voice cloning systems with speaker sim < 0.25 are considered to have failed.

---

## Literature Basis

| Paper | Venue | Technique Used |
|---|---|---|
| [AttackVC](https://arxiv.org/abs/2005.08781) | IEEE SLT 2021 | Foundational PGD on speaker encoders |
| [AntiFake](https://dl.acm.org/doi/10.1145/3576915.3623209) | ACM CCS 2023 | Ensemble speaker encoder attack |
| [VoiceGuard](https://www.ijcai.org/proceedings/2023/0535.pdf) | IJCAI 2023 | Psychoacoustic masking + time-domain PGD |
| [POP](https://arxiv.org/abs/2410.20742) | ACM LAMPS 2024 | Pivotal objective (TTS reconstruction loss) |
| [SafeSpeech](https://arxiv.org/abs/2504.09839) | USENIX Security 2025 | SPEC: mel + KL-toward-noise loss |
| [CloneShield](https://arxiv.org/abs/2505.19119) | arXiv 2025 | MGDA multi-objective universal perturbation |
| [VoiceCloak](https://arxiv.org/abs/2505.12332) | arXiv 2025 | Diffusion-model-specific disruption |
| [RoVo](https://arxiv.org/abs/2505.12686) | arXiv 2025 | Codec-embedding-level perturbation |
| [ClearMask](https://arxiv.org/abs/2508.17660) | ACM AsiaCCS 2025 | Noise-free spectral filtering |
| [Muting Whisper](https://arxiv.org/abs/2405.06134) | EMNLP 2024 | Universal adversarial prefix for ASR |
| [E2E-VGuard](https://arxiv.org/abs/2511.07099) | NeurIPS 2025 | LLM-TTS + ASR dual disruption |
| [De-AntiFake](https://arxiv.org/abs/2507.02606) | ICML 2025 | Diffusion purification breaks waveform attacks |
| [HarmonyCloak](https://mosis.eecs.utk.edu/publications/meerza2024harmonycloak.pdf) | UTK/Lehigh 2024 | Music training poisoning (unlearnable examples) |

---

## Ethical Scope

This tool is built for **artists defending their own work**. Apply it only to audio you own or have rights to protect. The attacks implemented here do not enable impersonation or offensive voice spoofing — they disrupt speaker-identity extraction, not amplify it.

---

## Project Structure

```
good-noise/
├── run.py                      ← canonical entry point
├── requirements.txt
├── config/
│   └── default.yaml            ← all experiment knobs
├── attacks/
│   ├── base.py
│   ├── pgd_embedding.py        ← attack 1: PGD on speaker embeddings
│   ├── psychoacoustic_pgd.py   ← attack 2: psychoacoustic-masked PGD
│   ├── mel_disruption.py       ← attack 3: multi-scale mel disruption
│   ├── universal_pgd.py        ← attack 4: universal perturbation
│   ├── asr_disruption.py       ← attack 5: Whisper ASR disruption
│   └── spectral_filter.py      ← attack 6: noise-free spectral filtering
├── models/
│   └── speaker_encoder.py      ← ECAPA-TDNN wrapper (SpeechBrain)
├── metrics/
│   └── audio_metrics.py        ← SNR, PESQ, STOI, speaker similarity
└── experiments/
    ├── ledger.jsonl             ← experiment log (append-only)
    └── EXPERIMENTS.md           ← human-readable experiment summaries
```
