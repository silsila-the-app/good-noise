"""Audio quality and protection metrics.

Metrics computed:
  - SNR         : Signal-to-Noise Ratio of perturbation (higher = more imperceptible)
  - PESQ        : Perceptual Evaluation of Speech Quality (wideband; higher = better quality)
  - STOI        : Short-Time Objective Intelligibility (0–1; higher = more intelligible)
  - SpeakerSim  : Cosine similarity of speaker embeddings before/after (lower = better disruption)

Target thresholds (from literature):
  - Protected audio PESQ > 3.5 ("acceptable")
  - Protected audio STOI > 0.9
  - Protected audio SNR > 20 dB
  - Speaker sim after protection < 0.25 (cloning considered failed)
"""

from typing import Dict, Any, Optional
import numpy as np
import torch


def _to_numpy_16k(waveform: torch.Tensor, sample_rate: int) -> np.ndarray:
    """Convert tensor to float32 numpy array at 16 kHz."""
    import torchaudio
    if waveform.dim() > 1:
        waveform = waveform.squeeze(0)
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    return waveform.detach().cpu().float().numpy()


def compute_snr(original: np.ndarray, perturbed: np.ndarray) -> float:
    """SNR in dB of the added perturbation."""
    noise = perturbed - original
    sig_power = np.mean(original ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power < 1e-12:
        return float("inf")
    return 10.0 * np.log10(sig_power / noise_power)


def compute_pesq(original: np.ndarray, perturbed: np.ndarray, sample_rate: int = 16000, mode: str = "wb") -> Optional[float]:
    """PESQ score. Returns None if pesq package unavailable."""
    try:
        from pesq import pesq
        score = pesq(sample_rate, original, perturbed, mode)
        return float(score)
    except ImportError:
        return None
    except Exception as e:
        return None


def compute_stoi(original: np.ndarray, perturbed: np.ndarray, sample_rate: int = 16000) -> Optional[float]:
    """STOI intelligibility score (0–1). Returns None if pystoi unavailable."""
    try:
        from pystoi import stoi
        score = stoi(original, perturbed, sample_rate, extended=False)
        return float(score)
    except ImportError:
        return None
    except Exception as e:
        return None


def compute_speaker_similarity(
    original: torch.Tensor,
    perturbed: torch.Tensor,
    sample_rate: int,
    device: torch.device,
) -> Optional[float]:
    """Cosine similarity of speaker embeddings. Requires SpeechBrain ECAPA-TDNN."""
    try:
        from models.speaker_encoder import SpeakerEncoder
        import torch.nn.functional as F

        enc = SpeakerEncoder(device)
        orig_wav = original.to(device)
        pert_wav = perturbed.to(device)
        if orig_wav.dim() == 1:
            orig_wav = orig_wav.unsqueeze(0)
        if pert_wav.dim() == 1:
            pert_wav = pert_wav.unsqueeze(0)

        # Trim to same length
        min_len = min(orig_wav.shape[-1], pert_wav.shape[-1])
        orig_wav = orig_wav[..., :min_len]
        pert_wav = pert_wav[..., :min_len]

        with torch.no_grad():
            emb_orig = enc.embed(orig_wav)
            emb_pert = enc.embed(pert_wav)
            sim = F.cosine_similarity(emb_orig, emb_pert, dim=-1).item()
        return float(sim)
    except Exception:
        return None


def compute_all_metrics(
    original: torch.Tensor,
    perturbed: torch.Tensor,
    sample_rate: int,
    device: torch.device,
    cfg: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Compute all metrics and return as dict.

    Args:
        original:  clean waveform tensor (1, T) or (T,)
        perturbed: protected waveform tensor (same shape; may be longer for ASR prefix)
        sample_rate: Hz
        device: torch device
        cfg: optional config dict

    Returns:
        Dict with keys: snr, pesq, stoi, speaker_similarity
        Missing metrics have value None.
    """
    cfg = cfg or {}
    pesq_mode = cfg.get("metrics", {}).get("pesq_mode", "wb")

    # Convert to numpy at 16 kHz
    orig_np = _to_numpy_16k(original, sample_rate)
    pert_np = _to_numpy_16k(perturbed, sample_rate)

    # Match lengths for comparison (trim to shorter)
    min_len = min(len(orig_np), len(pert_np))
    orig_np_cmp = orig_np[:min_len]
    pert_np_cmp = pert_np[:min_len]

    results: Dict[str, Any] = {}

    results["snr"] = compute_snr(orig_np_cmp, pert_np_cmp)
    results["pesq"] = compute_pesq(orig_np_cmp, pert_np_cmp, 16000, pesq_mode)
    results["stoi"] = compute_stoi(orig_np_cmp, pert_np_cmp, 16000)
    results["speaker_similarity"] = compute_speaker_similarity(
        original, perturbed, sample_rate, device
    )

    return results


def print_metrics(metrics: Dict[str, Any], attack_name: str = ""):
    """Pretty-print a metrics dict."""
    label = f"[{attack_name}] " if attack_name else ""
    print(f"\n{'─'*55}")
    print(f"  {label}Metrics")
    print(f"{'─'*55}")
    snr = metrics.get("snr")
    pesq = metrics.get("pesq")
    stoi = metrics.get("stoi")
    sim = metrics.get("speaker_similarity")

    print(f"  SNR (perturbation imperceptibility) : {f'{snr:.2f} dB' if snr is not None else 'N/A':>12}  [target: >20 dB]")
    print(f"  PESQ (speech quality)               : {f'{pesq:.3f}' if pesq is not None else 'N/A':>12}  [target: >3.5]")
    print(f"  STOI (intelligibility)              : {f'{stoi:.3f}' if stoi is not None else 'N/A':>12}  [target: >0.90]")
    print(f"  Speaker Similarity (lower=better)   : {f'{sim:.3f}' if sim is not None else 'N/A':>12}  [target: <0.25]")
    print(f"{'─'*55}\n")
